from pathlib import Path

from pytts.domain import Article, BlockKind, TextBlock
from pytts.text.abbreviations import AbbreviationExpander


def _expand(text: str, mapping: dict[str, str]) -> str:
    article = Article(Path("article.md"), (TextBlock(BlockKind.PARAGRAPH, text),))
    return AbbreviationExpander(mapping).expand_article(article).blocks[0].text


def test_expands_literal_punctuation_without_a_pause_after_dot() -> None:
    assert _expand(
        "Ув. г-н Иванов, т.е. автор.",
        {
            "г-н": "господин",
            "ув.": "уважаемый",
            "т.е.": "то есть",
        },
    ) == "Уважаемый господин Иванов, то есть автор."


def test_uses_longest_case_insensitive_match_and_transfers_initial_case() -> None:
    assert _expand(
        "Т.Е. пример, т. пример.",
        {"т.": "товарищ", "т.е.": "то есть"},
    ) == "То есть пример, товарищ пример."


def test_respects_token_boundaries() -> None:
    assert _expand("авт.е.слово, _т.е._ и т.е. отдельно", {"т.е.": "то есть"}) == (
        "авт.е.слово, _то есть_ и то есть отдельно"
    )


def test_escapes_regex_metacharacters_in_keys() -> None:
    assert _expand("a+b. aab.", {"a+b.": "literal"}) == "literal aab."


def test_does_not_expand_abbreviations_inside_urls() -> None:
    assert _expand(
        "Ув. https://example.test/ув./docs, г-н.",
        {"ув.": "уважаемый", "г-н": "господин"},
    ) == "Уважаемый https://example.test/ув./docs, господин."


def test_leaves_unicode_ignorecase_match_without_casefolded_key_unchanged() -> None:
    assert _expand("İ. i.", {"i.": "item"}) == "İ. item"


def test_transfers_case_from_first_alphabetic_character() -> None:
    assert _expand("(«Ув.)", {"«ув.": "уважаемый"}) == "(Уважаемый)"


def test_replacements_are_not_scanned_again() -> None:
    assert _expand("а.", {"а.": "б.", "б.": "в"}) == "б."


def test_empty_mapping_returns_equal_article() -> None:
    article = Article(Path("a.md"), (TextBlock(BlockKind.HEADING, "Заголовок"),))
    assert AbbreviationExpander({}).expand_article(article) == article
