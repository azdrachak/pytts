from __future__ import annotations

import re
import unicodedata

from pytts.domain import Article, CleaningResult, TextBlock
from pytts.errors import InputError
from pytts.text.confusables import repair_mixed_scripts

_URL = re.compile(
    r"(?<![\w@])(?:"
    r"(?ai:https?://|www\.)[^\s<>]+"
    r"|(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,}(?:[/?#][^\s<>]*)"
    r")"
)
_URL_TRAILING_PUNCTUATION = frozenset(".,;:!?…)]}»”’\"'")
_SPACE = re.compile(r"\s+")
_TECHNICAL_UNICODE = frozenset("\u00ad\u200b\u2060\ufeff")


def _remove_url(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if match.start() > 0:
        previous = match.string[match.start() - 1]
        if unicodedata.category(previous).startswith("M"):
            return candidate
    boundary = len(candidate)
    while (
        boundary > 0
        and unicodedata.normalize("NFC", candidate[boundary - 1])
        in _URL_TRAILING_PUNCTUATION
    ):
        boundary -= 1
    return candidate[boundary:]


def _remove_technical_artifacts(text: str) -> str:
    return "".join(
        character
        for character in text
        if character not in _TECHNICAL_UNICODE
        and not (unicodedata.category(character) == "Cc" and not character.isspace())
    )


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
        without_artifacts = _remove_technical_artifacts(block.text)
        without_urls = _URL.sub(_remove_url, without_artifacts)
        normalized = unicodedata.normalize("NFC", without_urls)
        repaired = repair_mixed_scripts(normalized)
        text = _SPACE.sub(" ", repaired).strip()
        if text:
            blocks.append(TextBlock(kind=block.kind, text=text))
    if not blocks:
        raise InputError(f"Input contains no readable text after cleanup: {article.source}")
    cleaned = Article(source=article.source, blocks=tuple(blocks))
    return CleaningResult(article=cleaned, warning=_warning(" ".join(block.text for block in blocks)))
