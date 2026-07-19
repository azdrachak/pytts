from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    QUOTE = "quote"


class SpeechRate(StrEnum):
    X_SLOW = "x-slow"
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    X_FAST = "x-fast"

    @property
    def ssml_value(self) -> str:
        return "medium" if self is SpeechRate.NORMAL else self.value


@dataclass(frozen=True, slots=True)
class TextBlock:
    kind: BlockKind
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TextBlock text must not be blank")


@dataclass(frozen=True, slots=True)
class Article:
    source: Path
    blocks: tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("Article requires at least one block")


@dataclass(frozen=True, slots=True)
class CleaningResult:
    article: Article
    warning: str | None


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    ssml_text: str
    pause_after_ms: int


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    input_path: Path
    output_path: Path | None = None
    config_path: Path | None = None
    voice: str | None = None
    rate: SpeechRate = SpeechRate.NORMAL
    force: bool = False


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_path: Path
    voice: str
    chunk_count: int
