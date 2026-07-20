from __future__ import annotations

import re
import unicodedata

from pytts.domain import Article, CleaningResult, TextBlock
from pytts.errors import InputError

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_SPACE = re.compile(r"\s+")


def _warning(text: str) -> str | None:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "Input contains no alphabetic characters; Russian pronunciation may be unusable"
    cyrillic = sum(
        "а" <= character.casefold() <= "я" or character.casefold() == "ё"
        for character in letters
    )
    ratio = cyrillic / len(letters)
    if ratio < 0.70:
        return (
            f"Cyrillic letters are below 70% ({ratio:.0%}); "
            "the Russian model may mispronounce text"
        )
    return None


def clean_article(article: Article) -> CleaningResult:
    blocks: list[TextBlock] = []
    for block in article.blocks:
        normalized = unicodedata.normalize("NFC", block.text)
        text = _SPACE.sub(" ", _URL.sub("", normalized)).strip()
        if text:
            blocks.append(TextBlock(kind=block.kind, text=text))
    if not blocks:
        raise InputError(f"Input contains no readable text after cleanup: {article.source}")
    cleaned = Article(source=article.source, blocks=tuple(blocks))
    return CleaningResult(article=cleaned, warning=_warning(" ".join(block.text for block in blocks)))
