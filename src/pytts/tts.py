from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pytts.domain import SpeechChunk
from pytts.errors import ModelError, SynthesisError
from pytts.model_store import DownloadProgress, ModelSpec, ModelStore


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    name: str
    warning: str | None


def _validate_package(path: Path) -> None:
    """Check a newly downloaded package before it is placed in the cache."""
    try:
        torch.package.PackageImporter(str(path))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ModelError(f"Silero file is not a readable Torch package: {error}") from error


def _load_packaged_model(path: Path) -> Any:
    try:
        importer = torch.package.PackageImporter(str(path))
        return importer.load_pickle("tts_models", "model")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ModelError(f"Could not load Silero tts_models/model: {error}") from error


class SileroRuntime:
    """One CPU-resident Silero model and its runtime-discovered voices."""

    def __init__(self, model: Any, spec: ModelSpec) -> None:
        try:
            model.to(torch.device("cpu"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise ModelError(f"Could not move Silero model to CPU: {error}") from error

        try:
            model.eval()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise ModelError(f"Could not put Silero model in evaluation mode: {error}") from error

        try:
            raw_speakers = getattr(model, "speakers", None)
            if not isinstance(raw_speakers, (list, tuple)):
                raise ModelError("Silero model.speakers is missing or invalid")
            speakers = tuple(str(item) for item in raw_speakers if str(item))
        except (KeyboardInterrupt, SystemExit):
            raise
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"Could not read Silero model.speakers: {error}") from error

        if not speakers:
            raise ModelError("Silero model.speakers is empty")

        self._model = model
        self._spec = spec
        self._speakers = speakers

    @classmethod
    def load(
        cls,
        store: ModelStore,
        spec: ModelSpec,
        download_progress: DownloadProgress | None = None,
    ) -> SileroRuntime:
        # ModelStore verifies the SHA-256 before this one runtime deserialization
        # of a cached artifact. A fresh download is separately package-validated
        # before its atomic placement.
        path = store.ensure(spec, _validate_package, download_progress)
        return cls(_load_packaged_model(path), spec)

    @property
    def speakers(self) -> tuple[str, ...]:
        return self._speakers

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        if requested is not None:
            if requested not in self._speakers:
                available = ", ".join(self._speakers)
                raise ModelError(f"Unknown voice {requested!r}; runtime voices: {available}")
            return VoiceSelection(requested, None)

        if self._spec.preferred_voice in self._speakers:
            return VoiceSelection(self._spec.preferred_voice, None)

        fallback = self._speakers[0]
        return VoiceSelection(
            fallback,
            f"Preferred voice {self._spec.preferred_voice!r} is unavailable; using {fallback!r}",
        )

    def synthesize(self, chunk: SpeechChunk, voice: str) -> torch.Tensor:
        if voice not in self._speakers:
            raise ModelError(f"Voice {voice!r} disappeared from runtime speakers")

        try:
            audio = self._model.apply_tts(
                ssml_text=chunk.ssml_text,
                speaker=voice,
                sample_rate=self._spec.sample_rate,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise SynthesisError(f"Silero synthesis failed: {error}") from error

        if not isinstance(audio, torch.Tensor):
            raise SynthesisError("Silero returned non-empty finite mono PCM required")

        try:
            pcm = audio.detach().cpu().float()
            if not isinstance(pcm, torch.Tensor) or pcm.ndim != 1 or pcm.numel() == 0:
                raise SynthesisError("Silero returned non-empty finite mono PCM required")
            if pcm.dtype != torch.float32:
                raise SynthesisError("Silero returned non-empty finite mono PCM required")
            is_finite = bool(torch.isfinite(pcm).all().item())
        except (KeyboardInterrupt, SystemExit):
            raise
        except SynthesisError:
            raise
        except Exception as error:
            raise SynthesisError(f"Silero returned non-empty finite mono PCM required: {error}") from error

        if not is_finite:
            raise SynthesisError("Silero returned non-empty finite mono PCM required")
        return pcm
