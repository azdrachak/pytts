from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("~28 миллиардов", "примерно двадцать восемь миллиардов"),
        ("≈25%", "примерно двадцать пять процентов"),
        ("к прочтению -> название", "к прочтению: название"),
        ("к прочтению → название", "к прочтению: название"),
    ],
)
def test_normalizes_approximation_and_arrows(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**это не рекомендация**", "это не рекомендация"),
        ("** Данная информация", "Данная информация"),
        ("решения **", "решения"),
        ("*Данный документ*", "Данный документ"),
    ],
)
def test_strips_one_or_two_stars_only_at_block_boundaries(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("***бать", "бать"),
        ("слово***", "слово"),
    ],
)
def test_triple_boundary_stars_remain_visible_to_warning_guard(
    source: str, expected: str
) -> None:
    result = PronunciationNormalizer({}).normalize_article(_article(source))

    assert result.article.blocks[0].text == expected
    assert result.warning is not None
    assert result.warning.startswith("Removed 3 unsupported character(s)")
    assert result.warning.count("'*' near") == 3


def test_interior_mathematical_star_remains_visible_to_warning_guard() -> None:
    result = PronunciationNormalizer({}).normalize_article(_article("2 * 3"))

    assert result.article.blocks[0].text == "два три"
    assert result.warning is not None
    assert result.warning.count("'*' near") == 1


def test_double_tilde_is_not_treated_as_approximation() -> None:
    result = PronunciationNormalizer({}).normalize_article(_article("~~текст~~"))

    assert result.article.blocks[0].text == "текст"
    assert result.warning is not None
    assert result.warning.count("'~' near") == 4


def test_removes_remaining_emoji_sequences_without_warning() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("До 🎮 и 👩🏽‍💻 после 🇷🇺 знака.")
    )

    assert result.article.blocks[0].text == "До и после знака."
    assert result.warning is None


def test_emoji_removal_precedes_boundary_star_cleanup() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("*️⃣ **текст** 👨‍👩‍👧‍👦")
    )

    assert result.article.blocks[0].text == "текст"
    assert result.warning is None


def test_transliteration_override_can_pronounce_emoji_before_removal() -> None:
    assert _normalize("🎮", {"🎮": "игра"}) == "игра"


def test_rejects_article_containing_only_unmapped_emoji() -> None:
    with pytest.raises(InputError, match="No speakable text remains"):
        PronunciationNormalizer({}).normalize_article(_article("🎮 👨‍👩‍👧‍👦"))


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
