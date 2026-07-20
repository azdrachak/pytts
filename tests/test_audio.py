from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mutagen.mp3 import BitrateMode, MP3

from pytts.audio import AtomicMp3Writer, pcm16_bytes
from pytts.errors import AudioError, InputError


class FakeEncoder:
    instances: list["FakeEncoder"] = []

    def __init__(self) -> None:
        self.settings: dict[str, int] = {}
        self.inputs: list[bytes] = []
        self.__class__.instances.append(self)

    def set_bit_rate(self, value: int) -> None:
        self.settings["bit_rate"] = value

    def set_in_sample_rate(self, value: int) -> None:
        self.settings["sample_rate"] = value

    def set_channels(self, value: int) -> None:
        self.settings["channels"] = value

    def set_quality(self, value: int) -> None:
        self.settings["quality"] = value

    def encode(self, pcm: bytes) -> bytes:
        self.inputs.append(pcm)
        return b"encoded:" + pcm

    def flush(self) -> bytes:
        return b":flushed"


def test_pcm16_clamps_and_uses_little_endian_signed_samples() -> None:
    assert pcm16_bytes(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])) == (
        b"\x01\x80\x01\x80\x00\x00\xff\x7f\xff\x7f"
    )


@pytest.mark.parametrize("audio", [torch.empty(0), torch.ones((1, 2))])
def test_pcm16_rejects_empty_or_non_mono_audio(audio: torch.Tensor) -> None:
    with pytest.raises(AudioError, match="non-empty mono"):
        pcm16_bytes(audio)


def test_streams_audio_and_silence_then_atomically_commits(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
        writer.write_audio(torch.tensor([0.0, 0.5]))
        writer.write_silence(120)
        assert not output.exists()
        assert output.with_name("article.mp3.part").exists()
        writer.commit()

    encoder = FakeEncoder.instances[-1]
    assert encoder.settings == {
        "bit_rate": 96,
        "sample_rate": 48000,
        "channels": 1,
        "quality": 2,
    }
    assert len(encoder.inputs[1]) == 48000 * 120 // 1000 * 2
    assert output.read_bytes().endswith(b":flushed")
    assert not output.with_name("article.mp3.part").exists()


def test_silence_uses_writer_sample_rate_and_rejects_negative_duration(tmp_path: Path) -> None:
    output = tmp_path / "custom-rate.mp3"
    with AtomicMp3Writer(
        output, force=False, sample_rate=8000, encoder_factory=FakeEncoder
    ) as writer:
        writer.write_silence(125)
        with pytest.raises(ValueError, match="must not be negative"):
            writer.write_silence(-1)
        writer.commit()

    assert len(FakeEncoder.instances[-1].inputs[0]) == 8000 * 125 // 1000 * 2


def test_silence_is_encoded_in_bounded_700_ms_buffers(tmp_path: Path) -> None:
    output = tmp_path / "long-pause.mp3"
    with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
        writer.write_silence(1501)
        writer.commit()

    inputs = FakeEncoder.instances[-1].inputs
    assert [len(pcm) for pcm in inputs] == [48000 * 700 // 1000 * 2] * 2 + [48000 * 101 // 1000 * 2]


def test_exception_removes_only_exact_partial(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    neighbor = tmp_path / "article.mp3.part.keep"
    neighbor.write_bytes(b"user data")
    with pytest.raises(RuntimeError, match="stop"):
        with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
            writer.write_audio(torch.tensor([0.0]))
            raise RuntimeError("stop")
    assert not output.with_name("article.mp3.part").exists()
    assert neighbor.read_bytes() == b"user data"


def test_keyboard_interrupt_removes_partial(tmp_path: Path) -> None:
    output = tmp_path / "interrupted.mp3"
    with pytest.raises(KeyboardInterrupt):
        with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
            writer.write_audio(torch.tensor([0.0]))
            raise KeyboardInterrupt
    assert not output.exists()
    assert not output.with_name("interrupted.mp3.part").exists()


def test_refuses_existing_output_without_force(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    output.write_bytes(b"existing")
    with pytest.raises(InputError, match="--force"):
        AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder)
    assert output.read_bytes() == b"existing"


def test_real_lame_output_has_required_metadata(tmp_path: Path) -> None:
    output = tmp_path / "real.mp3"
    audio = torch.sin(torch.arange(48000, dtype=torch.float32) * (2 * torch.pi * 440 / 48000))
    with AtomicMp3Writer(output, force=False) as writer:
        writer.write_audio(audio)
        writer.commit()

    info = MP3(output).info
    assert info.sample_rate == 48000
    assert info.channels == 1
    assert info.bitrate_mode in {BitrateMode.CBR, BitrateMode.UNKNOWN}
    assert 90000 <= info.bitrate <= 100000
    assert info.length == pytest.approx(1.0, abs=0.15)
