from __future__ import annotations

import html
import re

from pytts.domain import Article, BlockKind, SpeechChunk, SpeechRate
from pytts.errors import InputError

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_PREFERRED_PUNCTUATION = frozenset(",;:—–")
_FINAL_PAUSE = {
    BlockKind.HEADING: 700,
    BlockKind.PARAGRAPH: 350,
    BlockKind.QUOTE: 350,
    BlockKind.LIST_ITEM: 350,
}


def _split_long(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        preferred: list[int] = []
        whitespace: list[int] = []
        for index, character in enumerate(remaining[:limit]):
            is_boundary = index + 1 == len(remaining) or remaining[index + 1].isspace()
            if character in _PREFERRED_PUNCTUATION and is_boundary:
                preferred.append(index + 1)
            if character.isspace():
                whitespace.append(index)

        break_at = (preferred or whitespace)[-1] if preferred or whitespace else None
        if break_at is None:
            token = remaining.split(maxsplit=1)[0]
            if len(token) > limit:
                raise InputError(
                    f"A single token has {len(token)} characters, above model limit {limit}"
                )
            raise InputError(f"Could not split text at model limit {limit}")

        piece = remaining[:break_at].strip()
        if not piece:
            raise InputError(f"Could not split text at model limit {limit}")
        pieces.append(piece)
        remaining = remaining[break_at:].lstrip()

    if remaining:
        pieces.append(remaining)
    return pieces


def _units(text: str, limit: int) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]
    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_long(sentence, limit) if len(sentence) > limit else [sentence])
    return units


def _pack(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in _units(text, limit):
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _ssml(text: str, rate: SpeechRate) -> str:
    """Wrap a bounded clean-text payload; escaped XML is deliberately not re-limited."""
    escaped = html.escape(text, quote=False)
    return f'<speak><prosody rate="{rate.ssml_value}">{escaped}</prosody></speak>'


def chunk_article(
    article: Article,
    rate: SpeechRate,
    max_text_chars: int,
) -> tuple[SpeechChunk, ...]:
    if max_text_chars < 64:
        raise ValueError("max_text_chars must be at least 64")

    chunks: list[SpeechChunk] = []
    for block_index, block in enumerate(article.blocks, start=1):
        try:
            pieces = _pack(block.text, max_text_chars)
        except InputError as error:
            raise InputError(
                f"Block {block_index} ({block.kind.value}) cannot be chunked: {error}"
            ) from error
        for piece_index, piece in enumerate(pieces):
            pause = _FINAL_PAUSE[block.kind] if piece_index == len(pieces) - 1 else 120
            chunks.append(SpeechChunk(ssml_text=_ssml(piece, rate), pause_after_ms=pause))
    return tuple(chunks)
