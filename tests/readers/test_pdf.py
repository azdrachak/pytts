from pathlib import Path

import fitz
import pytest

from pytts.domain import BlockKind
from pytts.errors import InputError
from pytts.readers.pdf import PDFReader

_TEST_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def _insert_text(
    page: fitz.Page, point: tuple[float, float], text: str, size: float
) -> None:
    if not _TEST_FONT.is_file():
        raise RuntimeError(f"PDF test font is unavailable: {_TEST_FONT}")
    page.insert_text(
        point,
        text,
        fontsize=size,
        fontname="ArialUnicode",
        fontfile=_TEST_FONT,
    )


def _page(document: fitz.Document, number: int) -> None:
    page = document.new_page(width=595, height=842)
    _insert_text(page, (50, 25), "Сайт · сохранённая статья", 8)
    _insert_text(page, (50, 130), f"Раздел {number}", 18)
    _insert_text(page, (50, 180), "Это основной абзац статьи.", 11)
    _insert_text(page, (50, 210), "- Пункт списка", 11)
    _insert_text(page, (50, 810), str(number), 8)


def test_removes_repeated_header_and_page_numbers_and_detects_structure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "article.pdf"
    document = fitz.open()
    _page(document, 1)
    _page(document, 2)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert "сохранённая статья" not in " ".join(block.text for block in article.blocks)
    assert [(block.kind, block.text) for block in article.blocks[:3]] == [
        (BlockKind.HEADING, "Раздел 1"),
        (BlockKind.PARAGRAPH, "Это основной абзац статьи."),
        (BlockKind.LIST_ITEM, "Пункт списка"),
    ]
    assert all(block.text not in {"1", "2"} for block in article.blocks)


def test_joins_line_end_hyphen_only_for_lowercase_word(tmp_path: Path) -> None:
    source = tmp_path / "hyphen.pdf"
    document = fitz.open()
    page = document.new_page()
    _insert_text(page, (50, 80), "Долго-\nжданное событие. Санкт-\nПетербург.", 11)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert article.blocks[0].text == "Долгожданное событие. Санкт- Петербург."


def test_rejects_pdf_without_text_layer(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    with pytest.raises(InputError, match="text layer"):
        PDFReader().read(source)


def test_rejects_password_protected_pdf(tmp_path: Path) -> None:
    source = tmp_path / "protected.pdf"
    document = fitz.open()
    page = document.new_page()
    _insert_text(page, (50, 100), "Секретный текст", 11)
    document.save(
        source,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    document.close()

    with pytest.raises(InputError, match="password"):
        PDFReader().read(source)


def test_uniform_font_heading_degrades_to_paragraph(tmp_path: Path) -> None:
    source = tmp_path / "uniform.pdf"
    document = fitz.open()
    page = document.new_page()
    _insert_text(page, (50, 100), "Возможный заголовок", 11)
    _insert_text(page, (50, 150), "Основной текст статьи.", 11)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert [block.kind for block in article.blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
    ]
