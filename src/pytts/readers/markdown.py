from __future__ import annotations

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_FOOTNOTE_DEFINITION = re.compile(r"(?m)^\[\^[^]]+\]:.*(?:\n(?: {2,}|\t).*)*")
_FOOTNOTE_REFERENCE = re.compile(r"\[\^[^]]+\]")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")


def _drop_pipe_tables(text: str) -> str:
    lines = text.splitlines()
    dropped: set[int] = set()
    for index, line in enumerate(lines):
        if _TABLE_DELIMITER.match(line):
            if index > 0:
                dropped.add(index - 1)
            dropped.add(index)
            cursor = index + 1
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                dropped.add(cursor)
                cursor += 1
    return "\n".join(line for index, line in enumerate(lines) if index not in dropped)


def _prepare(text: str) -> str:
    text = _FRONT_MATTER.sub("", text)
    text = _FOOTNOTE_DEFINITION.sub("", text)
    return _drop_pipe_tables(text)


def _inline_text(token: Token) -> str:
    pieces: list[str] = []
    for child in token.children or ():
        if child.type in {"text", "text_special"}:
            pieces.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
        elif child.type in {"code_inline", "html_inline", "image"}:
            continue
    text = html.unescape("".join(pieces))
    text = _FOOTNOTE_REFERENCE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class MarkdownReader:
    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": True})

    def read(self, path: Path) -> Article:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise InputError(f"Could not read Markdown {path}: {error}") from error

        tokens = self._parser.parse(_prepare(text))
        list_depth = 0
        quote_depth = 0
        blocks: list[TextBlock] = []
        for index, token in enumerate(tokens):
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                list_depth += 1
            elif token.type in {"bullet_list_close", "ordered_list_close"}:
                list_depth -= 1
            elif token.type == "blockquote_open":
                quote_depth += 1
            elif token.type == "blockquote_close":
                quote_depth -= 1
            elif token.type in {"heading_open", "paragraph_open"}:
                inline = tokens[index + 1]
                if inline.type != "inline":
                    continue
                content = _inline_text(inline)
                if not content:
                    continue
                if token.type == "heading_open":
                    kind = BlockKind.HEADING
                elif list_depth:
                    kind = BlockKind.LIST_ITEM
                elif quote_depth:
                    kind = BlockKind.QUOTE
                else:
                    kind = BlockKind.PARAGRAPH
                blocks.append(TextBlock(kind=kind, text=content))

        if not blocks:
            raise InputError(f"Markdown contains no readable article text: {path}")
        return Article(source=path, blocks=tuple(blocks))
