from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import TracebackType

import torch

from pytts.domain import ConversionRequest
from pytts.model_store import ModelSpec
from pytts.pipeline import ConversionPipeline
from pytts.readers.input import InputReader
from pytts.tts import VoiceSelection


class Runtime:
    speakers = ("xenia",)

    def __init__(self) -> None:
        self.ssml: list[str] = []

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        return VoiceSelection("xenia", None)

    def synthesize(self, chunk: object, voice: str) -> torch.Tensor:
        self.ssml.append(chunk.ssml_text)  # type: ignore[attr-defined]
        return torch.tensor([0.0, 0.1])


class Writer:
    def __init__(self, output: Path, force: bool) -> None:
        self.output = output
        self.pauses: list[int] = []
        self.committed = False

    def __enter__(self) -> Writer:
        return self

    def write_audio(self, audio: torch.Tensor) -> None:
        assert audio.ndim == 1

    def write_silence(self, milliseconds: int) -> None:
        self.pauses.append(milliseconds)

    def commit(self) -> None:
        self.committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


def test_e2e_fixture_resolves_outside_project_cwd(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).resolve()),
            "-k",
            "markdown_to_audio_contract",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_markdown_to_audio_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pytts.yaml").write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n', encoding="utf-8"
    )
    source = Path(__file__).parent / "fixtures" / "smoke.md"
    runtime = Runtime()
    writers: list[Writer] = []

    def writer_factory(output: Path, force: bool) -> Writer:
        writer = Writer(output, force)
        writers.append(writer)
        return writer

    pipeline = ConversionPipeline(
        input_reader=InputReader(),
        model_spec=ModelSpec(
            "v5_5_ru", "https://example.test", "a" * 64, "xenia", 48000, 800
        ),
        runtime_factory=lambda progress: runtime,
        writer_factory=writer_factory,
        project_root=project_root,
    )

    result = pipeline.convert(
        ConversionRequest(source, output_path=tmp_path / "smoke.mp3")
    )

    combined = " ".join(runtime.ssml)
    assert "Проверка синтеза" in combined
    assert "Уважаемый читатель" in combined
    assert "2026" in combined and "25 %" in combined and "1 500 ₽" in combined
    assert "https://" not in combined
    assert result.chunk_count == len(runtime.ssml)
    assert writers[0].committed
    assert 700 in writers[0].pauses
    assert 350 in writers[0].pauses
