from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.text.pronunciation import PronunciationNormalizer


def _article(text: str) -> Article:
    return Article(Path("article.md"), (TextBlock(BlockKind.PARAGRAPH, text),))


def _normalize(text: str, mapping: dict[str, str] | None = None) -> str:
    normalizer = PronunciationNormalizer(mapping or {})
    return normalizer.normalize_article(_article(text)).article.blocks[0].text


def test_applies_override_then_numbers_then_remaining_latin() -> None:
    result = _normalize(
        "Brent и Bloomberg: F-16 стоил $5.",
        {"Brent": "Брент", "Bloomberg": "Блумберг"},
    )
    assert result == "Брент и Блумберг: эф шестнадцать стоил пять долларов."


def test_expands_semantic_symbols() -> None:
    assert _normalize("№ 5, 2 × 3 = 6, ±5, 7‰, § 2, А + Б & В") == (
        "номер пять, два умножить на три равно шесть, плюс-минус пять, "
        "семь промилле, параграф два, А плюс Б и В"
    )


def test_symbol_expansion_does_not_add_space_before_punctuation() -> None:
    assert _normalize("7‰, § 2.") == "семь промилле, параграф два."


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("Коэффициент α равен единице.", "α"),
        ("Температура 🌡 высокая.", "🌡"),
        ("Цена в неизвестной валюте ₿.", "₿"),
        ("Два * три.", "*"),
        ("Один ⁂ два.", "⁂"),
    ],
)
def test_strips_unspeakable_characters_and_warns(source: str, fragment: str) -> None:
    result = PronunciationNormalizer({}).normalize_article(_article(source))

    assert fragment not in result.article.blocks[0].text
    assert result.warning is not None
    assert repr(fragment) in result.warning


def test_strip_warning_includes_surrounding_context() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("Коэффициент α равен единице.")
    )

    assert result.warning is not None
    assert "'α'" in result.warning
    assert "Коэффициент" in result.warning


def test_override_can_make_other_alphabet_speakable() -> None:
    assert _normalize("Коэффициент α.", {"α": "альфа"}) == "Коэффициент альфа."


def test_expands_slash_as_spoken_alternative() -> None:
    assert _normalize("освобождении/взятии и А/Б") == (
        "освобождении или взятии и А или Б"
    )


def test_preserves_documented_russian_punctuation() -> None:
    source = '«Текст», “пример” — да; [верно]… (точно)!'

    assert _normalize(source) == source


def test_preserves_blocks_and_is_idempotent() -> None:
    article = Article(
        Path("article.md"),
        (
            TextBlock(BlockKind.HEADING, "NASA и 2026"),
            TextBlock(BlockKind.PARAGRAPH, "Brent: 15%."),
        ),
    )
    normalizer = PronunciationNormalizer({"Brent": "Брент"})
    once = normalizer.normalize_article(article)
    twice = normalizer.normalize_article(once.article)
    assert [block.kind for block in once.article.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
    ]
    assert twice == once
