from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from pytts.domain import Article, TextBlock
from pytts.errors import InputError
from pytts.text.latin import TransliterationOverrides, normalize_latin
from pytts.text.numeric_normalizer import NumericNormalizer

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
}
_EXPLICIT_FORBIDDEN = frozenset("%‰№°§&@#")


def _normalize_symbols(text: str) -> str:
    text = re.sub(r"#(?=\s*\d)", " номер ", text)
    text = re.sub(r"#(?=\s*[A-Za-zА-Яа-яЁё])", " хештег ", text)
    for symbol, words in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f" {words} ")
    collapsed = _SPACE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed)


def _is_cyrillic_letter(character: str) -> bool:
    return character.isalpha() and "CYRILLIC" in unicodedata.name(character, "")


def _validate_speakable(text: str) -> None:
    for index, character in enumerate(text):
        category = unicodedata.category(character)
        invalid = (
            character.isdigit()
            or character.isalpha()
            and not _is_cyrillic_letter(character)
            or character in _EXPLICIT_FORBIDDEN
            or category in {"Sc", "Sm", "Sk", "So"}
        )
        if invalid:
            start = max(0, index - 20)
            end = min(len(text), index + 21)
            raise InputError(
                f"Unsupported character {character!r} remains after pronunciation "
                f"normalization near {text[start:end]!r}"
            )


class PronunciationNormalizer:
    def __init__(self, transliterations: Mapping[str, str]) -> None:
        self._overrides = TransliterationOverrides(transliterations)
        self._numbers = NumericNormalizer()

    def normalize_article(self, article: Article) -> Article:
        blocks: list[TextBlock] = []
        for block in article.blocks:
            text = self._overrides.apply(block.text)
            text = self._numbers.normalize(text)
            text = _normalize_symbols(text)
            text = normalize_latin(text)
            _validate_speakable(text)
            blocks.append(TextBlock(kind=block.kind, text=text))
        return Article(source=article.source, blocks=tuple(blocks))
