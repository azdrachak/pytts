from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from pytts.domain import Article, TextBlock
from pytts.errors import InputError
from pytts.text.latin import TransliterationOverrides, normalize_latin
from pytts.text.numeric_normalizer import NumericNormalizer

_CONTEXT = 20

_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
_SYMBOL_WORDS = {
    "×": "умножить на",
    "÷": "разделить на",
    "±": "плюс-минус",
    "+": "плюс",
    "=": "равно",
    "‰": "промилле",
    "§": "параграф",
    "№": "номер",
    "&": "и",
    "@": "собака",
    "/": "или",
}
_EXPLICIT_FORBIDDEN = frozenset("%‰№°§&@#")
_ALLOWED_PUNCTUATION = frozenset(
    ".,;:!?…"
    "-‐‑‒–—―"
    "'\"«»„“”‘’"
    "()[]"
)


def _normalize_symbols(text: str) -> str:
    text = re.sub(r"#(?=\s*\d)", " номер ", text)
    text = re.sub(r"#(?=\s*[A-Za-zА-Яа-яЁё])", " хештег ", text)
    for symbol, words in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f" {words} ")
    collapsed = _SPACE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed)


def _is_cyrillic_letter(character: str) -> bool:
    return character.isalpha() and "CYRILLIC" in unicodedata.name(character, "")


def _is_speakable(character: str) -> bool:
    category = unicodedata.category(character)
    return not (
        character.isdigit()
        or (character.isalpha() and not _is_cyrillic_letter(character))
        or character in _EXPLICIT_FORBIDDEN
        or category in {"Sc", "Sm", "Sk", "So"}
        or (category.startswith("P") and character not in _ALLOWED_PUNCTUATION)
    )


def _strip_unspeakable(text: str) -> tuple[str, list[str]]:
    """Drop characters Silero cannot voice, reporting each with its context."""
    kept: list[str] = []
    notices: list[str] = []
    for index, character in enumerate(text):
        if _is_speakable(character):
            kept.append(character)
            continue
        start = max(0, index - _CONTEXT)
        end = min(len(text), index + _CONTEXT + 1)
        notices.append(f"{character!r} near {text[start:end]!r}")
    return "".join(kept), notices


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    article: Article
    warning: str | None


class PronunciationNormalizer:
    def __init__(self, transliterations: Mapping[str, str]) -> None:
        self._overrides = TransliterationOverrides(transliterations)
        self._numbers = NumericNormalizer()

    def normalize_article(self, article: Article) -> NormalizationResult:
        blocks: list[TextBlock] = []
        notices: list[str] = []
        for block in article.blocks:
            text = self._overrides.apply(block.text)
            text = self._numbers.normalize(text)
            text = _normalize_symbols(text)
            text = normalize_latin(text)
            text, block_notices = _strip_unspeakable(text)
            notices.extend(block_notices)
            text = _SPACE.sub(" ", text).strip()
            if text:
                blocks.append(TextBlock(kind=block.kind, text=text))
        if not blocks:
            raise InputError(
                f"No speakable text remains after normalization: {article.source}"
            )
        warning = None
        if notices:
            warning = (
                f"Removed {len(notices)} unsupported character(s) before synthesis: "
                + "; ".join(notices)
            )
        return NormalizationResult(
            Article(source=article.source, blocks=tuple(blocks)), warning
        )
