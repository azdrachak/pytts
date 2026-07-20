from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
from pytts.text.cleaner import clean_article


def _article(*texts: str) -> Article:
    return Article(
        Path("article.md"),
        tuple(TextBlock(BlockKind.PARAGRAPH, text) for text in texts),
    )


def test_normalizes_whitespace_removes_urls_and_drops_empty_blocks() -> None:
    result = clean_article(
        _article("  Первый\u00a0абзац https://example.com/x?q=1  ", "https://example.org")
    )

    assert [block.text for block in result.article.blocks] == ["Первый абзац"]
    assert result.warning is None


@pytest.mark.parametrize(
    ("text", "warning_fragment"),
    [("12345 — %", "no alphabetic"), ("Mostly English и", "below 70%")],
)
def test_warns_but_returns_text_for_non_russian_content(
    text: str, warning_fragment: str
) -> None:
    result = clean_article(_article(text))

    assert result.article.blocks[0].text
    assert result.warning is not None
    assert warning_fragment in result.warning


def test_rejects_article_empty_after_cleanup() -> None:
    with pytest.raises(InputError, match="no readable text"):
        clean_article(_article("https://example.com"))
