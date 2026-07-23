from __future__ import annotations

import re
import unicodedata

from pytts.domain import Article, CleaningResult, TextBlock
from pytts.errors import InputError
from pytts.text.confusables import repair_mixed_scripts

_URL = re.compile(
    r"(?i)(?<![\w@])(?:"
    r"(?:https?://|www\.)[^\s<>]+"
    r"|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,}(?:[/?#][^\s<>]*)"
    r")"
)
_URL_TRAILING_PUNCTUATION = frozenset(".,;:!?…)]}»”’\"'")
_SPACE = re.compile(r"\s+")
_TECHNICAL_UNICODE = frozenset("\u00ad\u200b\u2060\ufeff")


def _remove_url(match: re.Match[str]) -> str:
    candidate = match.group(0)
    boundary = len(candidate)
    while boundary > 0 and candidate[boundary - 1] in _URL_TRAILING_PUNCTUATION:
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
        normalized = unicodedata.normalize("NFC", block.text)
        repaired = repair_mixed_scripts(normalized)
        without_artifacts = _remove_technical_artifacts(repaired)
        text = _SPACE.sub(" ", _URL.sub(_remove_url, without_artifacts)).strip()
        if text:
            blocks.append(TextBlock(kind=block.kind, text=text))
    if not blocks:
        raise InputError(f"Input contains no readable text after cleanup: {article.source}")
    cleaned = Article(source=article.source, blocks=tuple(blocks))
    return CleaningResult(article=cleaned, warning=_warning(" ".join(block.text for block in blocks)))
