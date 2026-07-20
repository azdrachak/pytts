from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from pytts.domain import SpeechChunk
from pytts.errors import ModelError, SynthesisError
from pytts.model_store import ModelSpec
from pytts.tts import SileroRuntime


_DEFAULT_AUDIO = object()


def _spec(preferred: str = "xenia") -> ModelSpec:
    return ModelSpec(
        "v5_5_ru",
        "https://example.test/model.pt",
        "a" * 64,
        preferred,
        48000,
        800,
    )


class FakeModel:
    def __init__(self, speakers: list[str], audio: object = _DEFAULT_AUDIO) -> None:
        self.speakers = speakers
        self.device: torch.device | None = None
        self.calls: list[dict[str, object]] = []
        self.audio = (
            torch.tensor([0.0, 0.25, -0.25]) if audio is _DEFAULT_AUDIO else audio
        )

    def to(self, device: torch.device) -> FakeModel:
        self.device = device
        return self

    def apply_tts(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.audio


class FailingModel(FakeModel):
    def apply_tts(self, **kwargs: object) -> object:
        raise RuntimeError("boom")


def _chunk() -> SpeechChunk:
    return SpeechChunk('<speak><prosody rate="fast">Текст &amp; ещё</prosody></speak>', 350)


def test_prefers_manifest_voice_when_runtime_has_it() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "xenia"]), _spec())

    assert runtime.resolve_voice(None).name == "xenia"
    assert runtime.resolve_voice(None).warning is None


def test_falls_back_to_first_runtime_voice_with_warning() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())

    selection = runtime.resolve_voice(None)

    assert selection.name == "aidar"
    assert selection.warning is not None
    assert "xenia" in selection.warning


def test_resolves_requested_voice_only_against_runtime_speakers() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())

    assert runtime.resolve_voice("baya").name == "baya"
    with pytest.raises(ModelError, match="aidar, baya"):
        runtime.resolve_voice("xenia")


def test_synthesizes_chunk_ssml_on_cpu_at_manifest_sample_rate() -> None:
    model = FakeModel(["xenia"])
    runtime = SileroRuntime(model, _spec())
    chunk = _chunk()

    audio = runtime.synthesize(chunk, "xenia")

    assert model.device == torch.device("cpu")
    assert model.calls == [
        {
            "ssml_text": chunk.ssml_text,
            "speaker": "xenia",
            "sample_rate": 48000,
        }
    ]
    assert audio.shape == (3,)
    assert audio.device.type == "cpu"


def test_wraps_model_synthesis_failure_as_synthesis_error() -> None:
    runtime = SileroRuntime(FailingModel(["xenia"]), _spec())

    with pytest.raises(SynthesisError, match="boom"):
        runtime.synthesize(_chunk(), "xenia")


@pytest.mark.parametrize(
    "audio",
    [
        None,
        torch.empty(0),
        torch.ones((2, 3)),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_rejects_non_mono_empty_or_non_finite_pcm(audio: object) -> None:
    runtime = SileroRuntime(FakeModel(["xenia"], audio), _spec())

    with pytest.raises(SynthesisError, match="non-empty finite mono PCM"):
        runtime.synthesize(_chunk(), "xenia")


def test_load_verifies_cache_before_single_runtime_package_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    events: list[tuple[object, ...]] = []
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            events.append(("open", source))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            events.append(("load_pickle", package, resource))
            return model

    class FakeStore:
        def ensure(self, spec: object, validator: object, progress: object) -> Path:
            events.append(("ensure", spec, validator, progress))
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)

    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]

    assert runtime.speakers == ("xenia",)
    assert model.device == torch.device("cpu")
    assert [event[0] for event in events] == ["ensure", "open", "load_pickle"]
    assert events[1:] == [
        ("open", str(path)),
        ("load_pickle", "tts_models", "model"),
    ]


def test_load_reuses_its_single_deserialized_model_for_every_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    imports: list[Path] = []
    path = tmp_path / "v5_5_ru.pt"

    class FakeImporter:
        def __init__(self, source: str) -> None:
            imports.append(Path(source))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            assert (package, resource) == ("tts_models", "model")
            return model

    class FakeStore:
        def ensure(self, *_: object) -> Path:
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)
    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]

    runtime.synthesize(_chunk(), "xenia")
    runtime.synthesize(_chunk(), "xenia")

    assert imports == [path]
    assert len(model.calls) == 2


def test_maps_package_load_failure_to_model_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenImporter:
        def __init__(self, _: str) -> None:
            raise RuntimeError("broken package")

    class FakeStore:
        def ensure(self, *_: object) -> Path:
            return tmp_path / "v5_5_ru.pt"

    monkeypatch.setattr(torch.package, "PackageImporter", BrokenImporter)

    with pytest.raises(ModelError, match="broken package"):
        SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]


def test_maps_cpu_setup_failure_to_model_error() -> None:
    class BrokenCpuModel(FakeModel):
        def to(self, _: torch.device) -> FakeModel:
            raise RuntimeError("cpu unavailable")

    with pytest.raises(ModelError, match="cpu unavailable"):
        SileroRuntime(BrokenCpuModel(["xenia"]), _spec())


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_does_not_wrap_control_flow_from_synthesis(error: BaseException) -> None:
    class InterruptedModel(FakeModel):
        def apply_tts(self, **kwargs: object) -> object:
            raise error

    runtime = SileroRuntime(InterruptedModel(["xenia"]), _spec())

    with pytest.raises(type(error)):
        runtime.synthesize(_chunk(), "xenia")


def test_does_not_wrap_control_flow_from_package_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedImporter:
        def __init__(self, _: str) -> None:
            raise KeyboardInterrupt

    class FakeStore:
        def ensure(self, *_: Any) -> Path:
            return tmp_path / "v5_5_ru.pt"

    monkeypatch.setattr(torch.package, "PackageImporter", InterruptedImporter)

    with pytest.raises(KeyboardInterrupt):
        SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]
