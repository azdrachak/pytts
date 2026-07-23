from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest
import torch

from pytts.domain import Article, BlockKind, ConversionRequest, TextBlock
from pytts.errors import InputError, SynthesisError
from pytts.model_store import ModelSpec
from pytts.pipeline import (
    ArticleReader,
    ConversionPipeline,
    ProgressEvent,
    ProgressStage,
)
from pytts.readers.input import InputReader
from pytts.text.cleaner import clean_article
from pytts.tts import VoiceSelection


class FakeInputReader:
    def __init__(self, text: str = "Ув. автор.") -> None:
        self.text = text
        self.paths: list[Path] = []

    def read(self, path: Path) -> Article:
        self.paths.append(path)
        return Article(path, (TextBlock(BlockKind.PARAGRAPH, self.text),))


class FakeRuntime:
    speakers = ("aidar", "xenia")

    def __init__(self, fail: bool = False, voice_warning: str | None = None) -> None:
        self.fail = fail
        self.voice_warning = voice_warning
        self.chunks: list[str] = []

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        return VoiceSelection(requested or "xenia", self.voice_warning)

    def synthesize(self, chunk: object, voice: str) -> torch.Tensor:
        if self.fail:
            raise SynthesisError("failed chunk")
        self.chunks.append(chunk.ssml_text)  # type: ignore[attr-defined]
        return torch.tensor([0.0, 0.1])


class FakeWriter:
    def __init__(self, output: Path, force: bool) -> None:
        self.output = output
        self.force = force
        self.audio_count = 0
        self.pauses: list[int] = []
        self.committed = False
        self.aborted = False

    def __enter__(self) -> FakeWriter:
        return self

    def write_audio(self, audio: torch.Tensor) -> None:
        self.audio_count += 1

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
        self.aborted = not self.committed


def _pipeline(
    tmp_path: Path,
    runtime: FakeRuntime,
    events: list[ProgressEvent],
    writers: list[FakeWriter],
    *,
    reader: ArticleReader | None = None,
    runtime_calls: list[object] | None = None,
    max_text_chars: int = 800,
) -> ConversionPipeline:
    (tmp_path / "pytts.yaml").write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n'
        'transliterations:\n  "Brent": "Брент"\n',
        encoding="utf-8",
    )
    spec = ModelSpec(
        "v5_5_ru",
        "https://example.test",
        "a" * 64,
        "xenia",
        48000,
        max_text_chars,
    )

    def runtime_factory(progress: object) -> FakeRuntime:
        if runtime_calls is not None:
            runtime_calls.append(progress)
        return runtime

    def writer_factory(output: Path, force: bool) -> FakeWriter:
        writer = FakeWriter(output, force)
        writers.append(writer)
        return writer

    return ConversionPipeline(
        input_reader=reader or FakeInputReader(),
        model_spec=spec,
        runtime_factory=runtime_factory,
        writer_factory=writer_factory,
        project_root=tmp_path,
        progress=events.append,
    )


def test_conversion_expands_chunks_synthesizes_and_commits(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    writers: list[FakeWriter] = []
    runtime = FakeRuntime()
    pipeline = _pipeline(tmp_path, runtime, events, writers)
    source = tmp_path / "article.md"
    source.write_text("ignored by fake", encoding="utf-8")

    result = pipeline.convert(ConversionRequest(input_path=source))

    assert result.output_path == tmp_path / "article.mp3"
    assert result.voice == "xenia"
    assert "Уважаемый автор." in runtime.chunks[0]
    assert writers[0].audio_count == result.chunk_count
    assert writers[0].pauses == [350]
    assert writers[0].committed
    first_occurrences = list(dict.fromkeys(event.stage for event in events))
    assert first_occurrences == [
        ProgressStage.CONFIG,
        ProgressStage.READ,
        ProgressStage.CLEAN,
        ProgressStage.CHUNK,
        ProgressStage.MODEL,
        ProgressStage.VOICE,
        ProgressStage.SYNTHESIS,
        ProgressStage.FINALIZE,
        ProgressStage.COMPLETE,
    ]


def test_normalizes_after_abbreviations_and_before_chunking(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    pipeline = _pipeline(
        tmp_path,
        runtime,
        [],
        [],
        reader=FakeInputReader("Ув. автор: в 2026 году Brent вырос на 15%."),
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    combined = " ".join(runtime.chunks)
    assert "Уважаемый автор" in combined
    assert "две тысячи двадцать шестом году" in combined
    assert "Брент вырос на пятнадцать процентов" in combined
    assert not any(character.isdigit() for character in combined)


def test_article_artifact_fixture_leaves_only_censored_star_warning(
    tmp_path: Path,
) -> None:
    events: list[ProgressEvent] = []
    runtime = FakeRuntime()
    source = Path(__file__).parent / "fixtures" / "article_artifacts.md"
    cleaned = clean_article(InputReader().read(source))
    cleaned_text = " ".join(block.text for block in cleaned.article.blocks)
    assert "sponsr.ru/crimsonanalytics/126423" not in cleaned_text.casefold()

    pipeline = _pipeline(
        tmp_path,
        runtime,
        events,
        [],
        reader=InputReader(),
    )

    pipeline.convert(
        ConversionRequest(
            source,
            output_path=tmp_path / "article_artifacts.mp3",
        )
    )

    combined = " ".join(runtime.chunks)
    for fragment in (
        "Данная информация",
        "Данный документ не является рекомендацией",
        "примерно двадцать восемь миллиардов",
        "примерно двадцать пять процентов",
        "Петрокси Традинг",
        "виндфалл такс",
        "икс пять Гроуп",
        "Интерфакс",
        "маин.пи",
        "конфиг.ямл",
    ):
        assert fragment in combined
    assert "sponsr" not in combined.casefold()
    assert not any(character.isdigit() for character in combined)
    assert all(emoji not in combined for emoji in ("🎮", "*️⃣", "👨‍👩‍👧‍👦"))

    warnings = [
        event.message
        for event in events
        if event.warning
        and event.message is not None
        and event.message.startswith("Removed ")
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith("Removed 3 unsupported character(s)")
    assert warnings[0].count("'*' near") == 3


def test_normalization_failure_precedes_model_and_writer(tmp_path: Path) -> None:
    runtime_calls: list[object] = []
    writers: list[FakeWriter] = []
    pipeline = _pipeline(
        tmp_path,
        FakeRuntime(),
        [],
        writers,
        reader=FakeInputReader("Дата 10.20.2021 неверна."),
        runtime_calls=runtime_calls,
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    with pytest.raises(InputError, match="Invalid calendar date"):
        pipeline.convert(ConversionRequest(source))

    assert not runtime_calls
    assert not writers


def test_unspeakable_character_warns_but_still_converts(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    writers: list[FakeWriter] = []
    runtime = FakeRuntime()
    pipeline = _pipeline(
        tmp_path,
        runtime,
        events,
        writers,
        reader=FakeInputReader("Коэффициент α важен."),
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    assert writers[0].committed
    assert "α" not in " ".join(runtime.chunks)
    assert any(
        event.warning and event.message and "'α'" in event.message for event in events
    )


def test_chunks_post_normalization_text_with_requested_ssml_speed(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    source_text = " ".join(["1 500"] * 12)
    pipeline = _pipeline(
        tmp_path,
        runtime,
        [],
        [],
        reader=FakeInputReader(source_text),
        max_text_chars=64,
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    plain_payloads = [
        chunk.split(">", 2)[2].split("<", 1)[0] for chunk in runtime.chunks
    ]
    assert all(len(payload) <= 64 for payload in plain_payloads)
    assert all('<prosody rate="medium">' in chunk for chunk in runtime.chunks)
    assert len(runtime.chunks) > 1


def test_synthesis_failure_aborts_writer(tmp_path: Path) -> None:
    writers: list[FakeWriter] = []
    pipeline = _pipeline(tmp_path, FakeRuntime(fail=True), [], writers)
    source = tmp_path / "article.md"
    source.write_text("ignored", encoding="utf-8")

    with pytest.raises(SynthesisError, match="failed chunk"):
        pipeline.convert(ConversionRequest(input_path=source))

    assert writers[0].aborted
    assert not writers[0].committed


def test_list_voices_loads_runtime_without_input(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, FakeRuntime(), [], [])

    assert pipeline.list_voices() == ("aidar", "xenia")


def test_rejects_output_conflicts_before_loading_runtime(tmp_path: Path) -> None:
    calls: list[object] = []
    pipeline = _pipeline(tmp_path, FakeRuntime(), [], [], runtime_calls=calls)
    source = tmp_path / "article.md"
    source.write_text("original", encoding="utf-8")
    existing = tmp_path / "existing.mp3"
    existing.write_bytes(b"audio")

    with pytest.raises(InputError, match="must end in .mp3"):
        pipeline.convert(ConversionRequest(source, output_path=tmp_path / "audio.wav"))
    with pytest.raises(InputError, match="must differ from input"):
        pipeline.convert(ConversionRequest(source, output_path=source))
    with pytest.raises(InputError, match="already exists"):
        pipeline.convert(ConversionRequest(source, output_path=existing))
    with pytest.raises(InputError, match="directory does not exist"):
        pipeline.convert(
            ConversionRequest(source, output_path=tmp_path / "missing" / "audio.mp3")
        )

    assert not calls
    assert source.read_text(encoding="utf-8") == "original"
    assert existing.read_bytes() == b"audio"


def test_force_allows_existing_output_after_preflight(tmp_path: Path) -> None:
    writers: list[FakeWriter] = []
    pipeline = _pipeline(tmp_path, FakeRuntime(), [], writers)
    source = tmp_path / "article.md"
    output = tmp_path / "existing.mp3"
    source.write_text("source", encoding="utf-8")
    output.write_bytes(b"audio")

    pipeline.convert(ConversionRequest(source, output_path=output, force=True))

    assert writers[0].output == output
    assert writers[0].force


def test_rejects_existing_directory_output_before_downstream_work(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    writers: list[FakeWriter] = []
    runtime_calls: list[object] = []
    reader = FakeInputReader()
    pipeline = _pipeline(
        tmp_path,
        FakeRuntime(),
        events,
        writers,
        reader=reader,
        runtime_calls=runtime_calls,
    )
    source = tmp_path / "article.md"
    output = tmp_path / "output.mp3"
    source.write_text("source", encoding="utf-8")
    output.mkdir()

    with pytest.raises(InputError, match="must not be a directory"):
        pipeline.convert(ConversionRequest(source, output_path=output, force=True))

    assert not events
    assert not reader.paths
    assert not runtime_calls
    assert not writers


@pytest.mark.parametrize(
    ("target", "label", "error"),
    [
        ("input", "input", RuntimeError("symlink loop")),
        ("output", "output", OSError("filesystem failure")),
    ],
)
def test_wraps_path_resolution_failures_before_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    label: str,
    error: Exception,
) -> None:
    events: list[ProgressEvent] = []
    writers: list[FakeWriter] = []
    runtime_calls: list[object] = []
    reader = FakeInputReader()
    pipeline = _pipeline(
        tmp_path,
        FakeRuntime(),
        events,
        writers,
        reader=reader,
        runtime_calls=runtime_calls,
    )
    source = tmp_path / "article.md"
    output = tmp_path / "output.mp3"
    source.write_text("source", encoding="utf-8")
    failing_path = source if target == "input" else output
    original_resolve = Path.resolve

    def fail_for_target(path: Path, strict: bool = False) -> Path:
        if path == failing_path:
            raise error
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_for_target)

    with pytest.raises(InputError, match=rf"Could not resolve {label} path"):
        pipeline.convert(ConversionRequest(source, output_path=output))

    assert not events
    assert not reader.paths
    assert not runtime_calls
    assert not writers


def test_explicit_config_replaces_root_abbreviations(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    runtime = FakeRuntime()
    explicit = tmp_path / "custom.yaml"
    explicit.write_text(
        'version: 1\nabbreviations:\n  "авт.": "автор"\n', encoding="utf-8"
    )
    pipeline = _pipeline(
        tmp_path,
        runtime,
        events,
        [],
        reader=FakeInputReader("Ув. авт."),
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source, config_path=explicit))

    assert "Ув. автор" in runtime.chunks[0]
    assert "Уважаемый" not in runtime.chunks[0]


def test_surfaces_cleaning_and_voice_warnings_as_progress_events(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    pipeline = _pipeline(
        tmp_path,
        FakeRuntime(voice_warning="Using fallback voice"),
        events,
        [],
        reader=FakeInputReader("Mostly English и"),
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    warnings = [event for event in events if event.warning]
    assert [(event.stage, event.message) for event in warnings] == [
        (ProgressStage.CLEAN, warnings[0].message),
        (ProgressStage.VOICE, "Using fallback voice"),
    ]
    assert warnings[0].message is not None
    assert "below 70%" in warnings[0].message
