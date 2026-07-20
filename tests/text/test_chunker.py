from html import unescape
from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, SpeechRate, TextBlock
from pytts.errors import InputError
from pytts.text.chunker import chunk_article


def _article(*blocks: tuple[BlockKind, str]) -> Article:
    return Article(
        Path("article.md"),
        tuple(TextBlock(kind, text) for kind, text in blocks),
    )


def _plain(ssml: str) -> str:
    return unescape(ssml.split(">", 2)[2].rsplit("<", 2)[0])


@pytest.mark.parametrize(
    ("rate", "ssml_rate"),
    [
        (SpeechRate.X_SLOW, "x-slow"),
        (SpeechRate.SLOW, "slow"),
        (SpeechRate.NORMAL, "medium"),
        (SpeechRate.FAST, "fast"),
        (SpeechRate.X_FAST, "x-fast"),
    ],
)
def test_maps_every_rate_and_escapes_xml(rate: SpeechRate, ssml_rate: str) -> None:
    chunks = chunk_article(
        _article((BlockKind.PARAGRAPH, "Пять < шести & семь.")), rate, 100
    )

    assert chunks[0].ssml_text == (
        f'<speak><prosody rate="{ssml_rate}">Пять &lt; шести &amp; семь.</prosody></speak>'
    )
    assert chunks[0].pause_after_ms == 350


def test_splits_on_sentences_and_sets_intermediate_pause() -> None:
    text = (
        "Первое предложение содержит несколько обычных русских слов. "
        "Второе предложение содержит ещё несколько обычных русских слов. "
        "Третье предложение завершает проверку."
    )
    chunks = chunk_article(_article((BlockKind.HEADING, text)), SpeechRate.NORMAL, 64)

    assert len(chunks) >= 3
    assert [chunk.pause_after_ms for chunk in chunks[:-1]] == [120] * (len(chunks) - 1)
    assert chunks[-1].pause_after_ms == 700
    assert all(len(_plain(chunk.ssml_text)) <= 64 for chunk in chunks)
    assert " ".join(_plain(chunk.ssml_text) for chunk in chunks) == text


def test_long_sentence_prefers_listed_punctuation_before_whitespace() -> None:
    article = _article(
        (
            BlockKind.PARAGRAPH,
            "Раз два, три четыре пять шесть семь восемь девять десять одиннадцать двенадцать.",
        )
    )

    chunks = chunk_article(article, SpeechRate.NORMAL, 64)

    assert _plain(chunks[0].ssml_text) == "Раз два,"


def test_ascii_hyphen_is_not_a_preferred_break() -> None:
    text = "Один два, три четыре пять шесть семь восемь девять десять- одиннадцать двенадцать."

    chunks = chunk_article(_article((BlockKind.PARAGRAPH, text)), SpeechRate.NORMAL, 64)

    assert _plain(chunks[0].ssml_text) == "Один два,"


def test_preserves_numbers_dates_percent_currency_and_range_literally() -> None:
    source = "В 2026 году: 25 %, 1 500 ₽ и диапазон 3–5."
    chunks = chunk_article(_article((BlockKind.PARAGRAPH, source)), SpeechRate.NORMAL, 100)

    assert _plain(chunks[0].ssml_text) == source


def test_assigns_350_ms_to_paragraph_quote_and_list() -> None:
    article = _article(
        (BlockKind.PARAGRAPH, "Абзац."),
        (BlockKind.QUOTE, "Цитата."),
        (BlockKind.LIST_ITEM, "Пункт."),
    )

    assert [chunk.pause_after_ms for chunk in chunk_article(article, SpeechRate.NORMAL, 100)] == [
        350,
        350,
        350,
    ]


def test_rejects_token_longer_than_model_limit_with_block_context() -> None:
    article = _article((BlockKind.PARAGRAPH, "а" * 65))

    with pytest.raises(InputError, match=r"Block 1 \(paragraph\).*single token"):
        chunk_article(article, SpeechRate.NORMAL, 64)


@pytest.mark.parametrize("length", [447, 448])
def test_accepts_clean_tokens_at_the_manifest_boundary(length: int) -> None:
    text = "а" * length

    chunks = chunk_article(_article((BlockKind.PARAGRAPH, text)), SpeechRate.NORMAL, 448)

    assert [_plain(chunk.ssml_text) for chunk in chunks] == [text]


def test_rejects_clean_token_above_the_manifest_boundary() -> None:
    with pytest.raises(InputError, match=r"449 characters, above model limit 448"):
        chunk_article(
            _article((BlockKind.PARAGRAPH, "а" * 449)), SpeechRate.NORMAL, 448
        )


def test_long_block_reconstructs_exact_clean_text_without_empty_chunks() -> None:
    text = (
        "Первый фрагмент содержит ровно те слова, которые должны остаться. "
        "Второй фрагмент: с запятой, двоеточием и диапазоном 3–5 сохраняется буквально. "
        "Третий фрагмент завершает проверку."
    )

    chunks = chunk_article(_article((BlockKind.PARAGRAPH, text)), SpeechRate.NORMAL, 64)
    payloads = [_plain(chunk.ssml_text) for chunk in chunks]

    assert payloads
    assert all(payload for payload in payloads)
    assert all(len(payload) <= 64 for payload in payloads)
    assert " ".join(payloads) == text


@pytest.mark.parametrize("limit", [0, 63])
def test_rejects_limits_below_the_runtime_minimum(limit: int) -> None:
    with pytest.raises(ValueError, match="at least 64"):
        chunk_article(_article((BlockKind.PARAGRAPH, "Текст.")), SpeechRate.NORMAL, limit)
