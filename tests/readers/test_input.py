from pathlib import Path
from unittest.mock import Mock

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
from pytts.readers.input import InputReader


def _article(path: Path) -> Article:
    return Article(path, (TextBlock(BlockKind.PARAGRAPH, "Текст"),))


@pytest.mark.parametrize("suffix", [".md", ".MD"])
def test_dispatches_markdown_case_insensitively(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"article{suffix}"
    path.write_text("Текст", encoding="utf-8")
    markdown = Mock()
    markdown.read.return_value = _article(path)
    pdf = Mock()

    result = InputReader(markdown=markdown, pdf=pdf).read(path)

    assert result.source == path
    markdown.read.assert_called_once_with(path)
    pdf.read.assert_not_called()


def test_rejects_missing_and_unsupported_input(tmp_path: Path) -> None:
    reader = InputReader(markdown=Mock(), pdf=Mock())
    with pytest.raises(InputError, match="does not exist"):
        reader.read(tmp_path / "missing.md")
    unsupported = tmp_path / "article.txt"
    unsupported.write_text("Текст", encoding="utf-8")
    with pytest.raises(InputError, match="Unsupported input extension"):
        reader.read(unsupported)
