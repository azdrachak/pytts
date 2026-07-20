from __future__ import annotations

import os
import sys
from array import array
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import lameenc
import torch

from pytts.errors import AudioError, InputError

_MAX_SILENCE_BUFFER_MS = 700


def pcm16_bytes(audio: torch.Tensor) -> bytes:
    """Return a mono float waveform as signed little-endian 16-bit PCM bytes."""
    try:
        if audio.ndim != 1 or audio.numel() == 0:
            raise AudioError("Audio chunk must be a non-empty mono tensor")
        # Scaling by 32767 deliberately maps -1 and 1 to equal magnitudes and cannot overflow.
        samples = audio.detach().cpu().clamp(-1.0, 1.0).mul(32767).round().to(torch.int16)
        pcm = array("h", samples.tolist())
    except (KeyboardInterrupt, SystemExit):
        raise
    except AudioError:
        raise
    except Exception as error:
        raise AudioError(f"Could not convert audio chunk to PCM: {error}") from error
    if pcm.itemsize != 2:
        raise AudioError("Platform signed-short is not 16 bit")
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


class AtomicMp3Writer:
    def __init__(
        self,
        output_path: Path,
        force: bool,
        sample_rate: int = 48000,
        encoder_factory: Callable[[], Any] = lameenc.Encoder,
    ) -> None:
        if output_path.exists() and not force:
            raise InputError(f"Output already exists: {output_path}; pass --force to replace it")
        if not output_path.parent.is_dir():
            raise InputError(f"Output directory does not exist: {output_path.parent}")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")

        self.output_path = output_path
        self.partial_path = output_path.with_name(output_path.name + ".part")
        self._sample_rate = sample_rate
        self._committed = False
        self._closed = False
        try:
            self.partial_path.unlink(missing_ok=True)
            self._encoder = encoder_factory()
            self._encoder.set_bit_rate(96)
            self._encoder.set_in_sample_rate(sample_rate)
            self._encoder.set_channels(1)
            self._encoder.set_quality(2)
            self._target = self.partial_path.open("wb")
        except (KeyboardInterrupt, SystemExit):
            self._remove_partial()
            raise
        except Exception as error:
            self._remove_partial()
            raise AudioError(f"Could not initialize MP3 output {self.partial_path}: {error}") from error

    def __enter__(self) -> AtomicMp3Writer:
        return self

    def _remove_partial(self) -> None:
        try:
            self.partial_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _encode(self, pcm: bytes) -> None:
        if self._closed:
            raise AudioError("MP3 writer is already closed")
        try:
            encoded = self._encoder.encode(pcm)
            if encoded:
                self._target.write(encoded)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise AudioError(f"LAME encoding failed: {error}") from error

    def write_audio(self, audio: torch.Tensor) -> None:
        self._encode(pcm16_bytes(audio))

    def write_silence(self, milliseconds: int) -> None:
        if type(milliseconds) is not int or milliseconds < 0:
            raise ValueError("Silence duration must not be negative")
        remaining_frames = self._sample_rate * milliseconds // 1000
        max_frames = self._sample_rate * _MAX_SILENCE_BUFFER_MS // 1000
        while remaining_frames:
            frames = min(remaining_frames, max_frames)
            self._encode(b"\x00\x00" * frames)
            remaining_frames -= frames

    def commit(self) -> None:
        if self._closed:
            raise AudioError("MP3 writer is already closed")
        try:
            flushed = self._encoder.flush()
            if flushed:
                self._target.write(flushed)
            self._target.flush()
            os.fsync(self._target.fileno())
            self._target.close()
            self._closed = True
            os.replace(self.partial_path, self.output_path)
            self._committed = True
        except (KeyboardInterrupt, SystemExit):
            self.abort()
            raise
        except Exception as error:
            self.abort()
            raise AudioError(f"Could not finalize MP3 {self.output_path}: {error}") from error

    def abort(self) -> None:
        if not self._closed:
            try:
                self._target.close()
            except OSError:
                pass
            self._closed = True
        self._remove_partial()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._committed:
            self.abort()
