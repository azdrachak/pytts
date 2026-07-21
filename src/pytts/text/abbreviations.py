from __future__ import annotations

import re
from collections.abc import Mapping

from pytts.domain import Article, TextBlock

_URL = r"\b(?:https?://|www\.)\S+"
_WHITESPACE = re.compile(r"\s+")


class AbbreviationExpander:
    """Expand configured abbreviations literally, once, within token boundaries."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._replacements = {key.casefold(): value for key, value in mapping.items()}
        if mapping:
            alternatives = "|".join(
                re.escape(key) for key in sorted(mapping, key=len, reverse=True)
            )
            self._pattern = re.compile(
                rf"(?P<url>{_URL})|(?<![^\W_])(?:{alternatives})(?![^\W_])",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            self._pattern = None

    def _replace(self, match: re.Match[str]) -> str:
        if match.group("url") is not None:
            return match.group(0)
        source = match.group(0)
        replacement = self._replacements.get(source.casefold())
        if replacement is None:
            return source
        first_letter = next((character for character in source if character.isalpha()), "")
        if first_letter.isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    def expand_article(self, article: Article) -> Article:
        if self._pattern is None:
            return article
        blocks: list[TextBlock] = []
        for block in article.blocks:
            # Blank replacements delete a token, so tidy the surrounding spaces
            # and drop a block that a deletion emptied entirely.
            text = _WHITESPACE.sub(" ", self._pattern.sub(self._replace, block.text)).strip()
            if text:
                blocks.append(TextBlock(kind=block.kind, text=text))
        if not blocks:
            return article
        return Article(source=article.source, blocks=tuple(blocks))
