import pytest

from pytts.errors import InputError
from pytts.text.latin import (
    TransliterationOverrides,
    normalize_latin,
    spell_code_letters,
    spell_latin_letters,
)


def test_overrides_are_longest_first_case_insensitive_and_one_pass() -> None:
    overrides = TransliterationOverrides(
        {"New": "Нью", "New York Times": "Нью-Йорк таймс", "Brent": "Брент"}
    )
    assert overrides.apply("NEW YORK TIMES и Brent") == "Нью-Йорк таймс и Брент"


def test_overrides_respect_outer_token_boundaries() -> None:
    overrides = TransliterationOverrides({"Brent": "Брент"})
    assert overrides.apply("Brent Brentwood _Brent_") == "Брент Brentwood _Брент_"


@pytest.mark.parametrize(
    ("token", "expected"),
    [("F", "эф"), ("FPV", "эф пи ви"), ("NASA", "эн эй эс эй")],
)
def test_spells_short_caps_by_english_letter_names(token: str, expected: str) -> None:
    assert spell_latin_letters(token) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [("FP", "эф пи"), ("С", "эс"), ("X", "икс")],
)
def test_spells_code_letter_groups(token: str, expected: str) -> None:
    assert spell_code_letters(token) == expected


def test_transliterates_words_but_spells_short_caps() -> None:
    assert normalize_latin("Brent NASA UNESCO") == "Брент эн эй эс эй УНЕСКО"


def test_normalization_is_idempotent() -> None:
    once = normalize_latin("Brent FPV")
    assert normalize_latin(once) == once


def test_unsupported_latin_character_is_actionable() -> None:
    with pytest.raises(InputError, match="Latin token.*Straße"):
        normalize_latin("Straße")
