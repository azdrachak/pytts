from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pytts.domain import Article
from pytts.errors import InputError
from pytts.readers.markdown import MarkdownReader
from pytts.readers.pdf import PDFReader


class Reader(Protocol):
    def read(self, path: Path) -> Article: ...


class InputReader:
    def __init__(
        self,
        markdown: Reader | None = None,
        pdf: Reader | None = None,
    ) -> None:
        self._readers = {
            ".md": markdown or MarkdownReader(),
            ".pdf": pdf or PDFReader(),
        }

    def read(self, path: Path) -> Article:
        if not path.is_file():
            raise InputError(f"Input file does not exist: {path}")
        reader = self._readers.get(path.suffix.casefold())
        if reader is None:
            raise InputError(
                f"Unsupported input extension {path.suffix!r}; expected .md or .pdf"
            )
        return reader.read(path)
