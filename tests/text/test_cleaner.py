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


def test_removes_embedded_technical_unicode_and_control_artifacts() -> None:
    result = clean_article(
        _article("Пр\u00adив\u200bет\u2060, \ufeffмир!\x00\x01 2026 — 15%.")
    )

    assert result.article.blocks[0].text == "Привет, мир! 2026 — 15%."


def test_repairs_safari_mixed_script_artifacts_during_cleanup() -> None:
    result = clean_article(_article("ĸиевсĸий текст и FР-5."))

    assert result.article.blocks[0].text == "киевский текст и FP-5."


def test_rejects_article_containing_only_technical_artifacts() -> None:
    with pytest.raises(InputError, match="no readable text"):
        clean_article(_article("\u00ad\u200b\u2060\ufeff\x00\x01"))


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
