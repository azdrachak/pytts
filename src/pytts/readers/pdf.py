from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError

_PAGE_NUMBER = re.compile(r"^(?:стр\.?\s*)?\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:[-–—•●▪◦]|\d+[.)])\s+")
_LOWERCASE_HYPHEN = re.compile(r"(?<=[а-яё])-\s*\n\s*(?=[а-яё])")


@dataclass(frozen=True, slots=True)
class _RawBlock:
    page: int
    page_height: float
    y0: float
    y1: float
    font_size: float
    text: str

    @property
    def band(self) -> str:
        if self.y1 <= self.page_height * 0.12:
            return "header"
        if self.y0 >= self.page_height * 0.88:
            return "footer"
        return "body"


def _normalize(text: str) -> str:
    text = _LOWERCASE_HYPHEN.sub("", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _signature(block: _RawBlock) -> tuple[str, str]:
    normalized = re.sub(r"\d+", "#", block.text.casefold())
    return block.band, normalized


def _extract(document: fitz.Document) -> list[_RawBlock]:
    result: list[_RawBlock] = []
    for page_index, page in enumerate(document):
        layout = page.get_text("dict", sort=True)
        for block in layout.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            spans = [span for line in lines for span in line.get("spans", [])]
            text = "\n".join(
                "".join(span.get("text", "") for span in line.get("spans", []))
                for line in lines
            )
            text = _normalize(text)
            if not text or not spans:
                continue
            font_size = min(float(span.get("size", 0.0)) for span in spans)
            _, y0, _, y1 = block["bbox"]
            result.append(
                _RawBlock(page_index, page.rect.height, y0, y1, font_size, text)
            )
    return result


def _body_font_size(blocks: list[_RawBlock]) -> float:
    candidates = [block for block in blocks if block.band == "body"] or blocks
    weighted = sorted((block.font_size, max(1, len(block.text))) for block in candidates)
    midpoint = sum(weight for _, weight in weighted) / 2
    cumulative = 0
    for font_size, weight in weighted:
        cumulative += weight
        if cumulative >= midpoint:
            return font_size
    raise RuntimeError("Could not determine PDF body font size")


class PDFReader:
    def read(self, path: Path) -> Article:
        try:
            with fitz.open(path) as document:
                if document.needs_pass:
                    raise InputError(
                        f"Encrypted PDF cannot be read without a password: {path}"
                    )
                page_count = document.page_count
                blocks = _extract(document)
        except (OSError, RuntimeError, ValueError) as error:
            raise InputError(f"Could not read PDF {path}: {error}") from error

        if not blocks:
            raise InputError(f"PDF has no text layer: {path}")

        signature_pages: dict[tuple[str, str], set[int]] = {}
        for block in blocks:
            if block.band != "body":
                signature_pages.setdefault(_signature(block), set()).add(block.page)
        repeated = {
            signature
            for signature, pages in signature_pages.items()
            if len(pages) >= 2 and len(pages) / page_count >= 0.50
        }
        body_size = _body_font_size(blocks)
        article_blocks: list[TextBlock] = []
        for block in blocks:
            if _signature(block) in repeated or (
                block.band != "body" and _PAGE_NUMBER.fullmatch(block.text)
            ):
                continue
            marker = _LIST_MARKER.match(block.text)
            text = _LIST_MARKER.sub("", block.text, count=1) if marker else block.text
            if marker:
                kind = BlockKind.LIST_ITEM
            elif block.font_size >= body_size * 1.25 and len(text) <= 200:
                kind = BlockKind.HEADING
            else:
                kind = BlockKind.PARAGRAPH
            article_blocks.append(TextBlock(kind=kind, text=text))

        if not article_blocks:
            raise InputError(f"PDF has no readable article text after cleanup: {path}")
        return Article(source=path, blocks=tuple(article_blocks))
