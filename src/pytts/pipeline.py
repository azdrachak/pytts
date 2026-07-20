from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol

import torch

from pytts.config import load_config
from pytts.domain import Article, ConversionRequest, ConversionResult, SpeechChunk
from pytts.errors import InputError
from pytts.model_store import DownloadProgress, ModelSpec
from pytts.text.abbreviations import AbbreviationExpander
from pytts.text.chunker import chunk_article
from pytts.text.cleaner import clean_article
from pytts.text.pronunciation import PronunciationNormalizer
from pytts.tts import VoiceSelection


class ProgressStage(StrEnum):
    CONFIG = "config"
    READ = "read"
    CLEAN = "clean"
    CHUNK = "chunk"
    MODEL = "model"
    VOICE = "voice"
    SYNTHESIS = "synthesis"
    FINALIZE = "finalize"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: ProgressStage
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    warning: bool = False


class ArticleReader(Protocol):
    def read(self, path: Path) -> Article: ...


class Runtime(Protocol):
    @property
    def speakers(self) -> tuple[str, ...]: ...

    def resolve_voice(self, requested: str | None) -> VoiceSelection: ...

    def synthesize(self, chunk: SpeechChunk, voice: str) -> torch.Tensor: ...


class AudioWriter(Protocol):
    def __enter__(self) -> AudioWriter: ...

    def write_audio(self, audio: torch.Tensor) -> None: ...

    def write_silence(self, milliseconds: int) -> None: ...

    def commit(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


RuntimeFactory = Callable[[DownloadProgress | None], Runtime]
WriterFactory = Callable[[Path, bool], AudioWriter]
ProgressSink = Callable[[ProgressEvent], None]


class ConversionPipeline:
    def __init__(
        self,
        input_reader: ArticleReader,
        model_spec: ModelSpec,
        runtime_factory: RuntimeFactory,
        writer_factory: WriterFactory,
        project_root: Path,
        progress: ProgressSink | None = None,
    ) -> None:
        self._input_reader = input_reader
        self._spec = model_spec
        self._runtime_factory = runtime_factory
        self._writer_factory = writer_factory
        self._project_root = project_root
        self._progress = progress or (lambda event: None)

    def _emit(
        self,
        stage: ProgressStage,
        *,
        completed: int | None = None,
        total: int | None = None,
        message: str | None = None,
        warning: bool = False,
    ) -> None:
        self._progress(
            ProgressEvent(
                stage=stage,
                completed=completed,
                total=total,
                message=message,
                warning=warning,
            )
        )

    def _download_progress(self, completed: int, total: int | None) -> None:
        self._emit(ProgressStage.MODEL, completed=completed, total=total)

    def _runtime(self) -> Runtime:
        self._emit(ProgressStage.MODEL, message=f"Loading Silero {self._spec.model_id}")
        return self._runtime_factory(self._download_progress)

    @staticmethod
    def _resolved_path(path: Path, label: str) -> Path:
        try:
            return path.resolve()
        except (OSError, RuntimeError) as error:
            raise InputError(f"Could not resolve {label} path {path}: {error}") from error

    @classmethod
    def _output_path(cls, request: ConversionRequest) -> Path:
        output = request.output_path or request.input_path.with_suffix(".mp3")
        resolved_output = cls._resolved_path(output, "output")
        if resolved_output.is_dir():
            raise InputError(f"Output path must not be a directory: {output}")
        resolved_input = cls._resolved_path(request.input_path, "input")
        if resolved_output == resolved_input:
            raise InputError("Output path must differ from input path")
        if output.suffix.casefold() != ".mp3":
            raise InputError(f"Output path must end in .mp3: {output}")
        if output.exists() and not request.force:
            raise InputError(f"Output already exists: {output}; pass --force to replace it")
        if not output.parent.is_dir():
            raise InputError(f"Output directory does not exist: {output.parent}")
        return output

    def list_voices(self) -> tuple[str, ...]:
        return self._runtime().speakers

    def convert(self, request: ConversionRequest) -> ConversionResult:
        output = self._output_path(request)

        self._emit(ProgressStage.CONFIG, message="Loading text normalization config")
        config = load_config(request.config_path, self._project_root)
        self._emit(ProgressStage.READ, message=f"Reading {request.input_path}")
        article = self._input_reader.read(request.input_path)
        cleaned = clean_article(article)
        self._emit(
            ProgressStage.CLEAN,
            message=cleaned.warning or "Article text cleaned",
            warning=cleaned.warning is not None,
        )
        expanded = AbbreviationExpander(config.abbreviations).expand_article(cleaned.article)
        normalized = PronunciationNormalizer(config.transliterations).normalize_article(
            expanded
        )
        chunks = chunk_article(normalized, request.rate, self._spec.max_text_chars)
        self._emit(ProgressStage.CHUNK, completed=len(chunks), total=len(chunks))

        runtime = self._runtime()
        self._emit(ProgressStage.VOICE, message="Validating voice against model.speakers")
        selection = runtime.resolve_voice(request.voice)
        if selection.warning:
            self._emit(ProgressStage.VOICE, message=selection.warning, warning=True)

        with self._writer_factory(output, request.force) as writer:
            for index, chunk in enumerate(chunks, start=1):
                writer.write_audio(runtime.synthesize(chunk, selection.name))
                writer.write_silence(chunk.pause_after_ms)
                self._emit(ProgressStage.SYNTHESIS, completed=index, total=len(chunks))
            self._emit(ProgressStage.FINALIZE, message="Finalizing MP3")
            writer.commit()

        self._emit(ProgressStage.COMPLETE, message=f"Wrote {output}")
        return ConversionResult(output, selection.name, len(chunks))
