from __future__ import annotations

import re
from collections.abc import Mapping

from pytts.domain import Article, TextBlock


class AbbreviationExpander:
    """Expand configured abbreviations literally, once, within token boundaries."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._replacements = {key.casefold(): value for key, value in mapping.items()}
        if mapping:
            alternatives = "|".join(
                re.escape(key) for key in sorted(mapping, key=len, reverse=True)
            )
            self._pattern = re.compile(
                rf"(?<![^\W_])(?:{alternatives})(?![^\W_])",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            self._pattern = None

    def _replace(self, match: re.Match[str]) -> str:
        source = match.group(0)
        replacement = self._replacements[source.casefold()]
        if source[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    def expand_article(self, article: Article) -> Article:
        if self._pattern is None:
            return article
        blocks = tuple(
            TextBlock(kind=block.kind, text=self._pattern.sub(self._replace, block.text))
            for block in article.blocks
        )
        return Article(source=article.source, blocks=blocks)
