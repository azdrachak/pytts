import pytest

from pytts.errors import InputError
from pytts.text.russian_numbers import (
    cardinal,
    decimal_words,
    noun_form,
    ordinal,
    parse_integer,
)


@pytest.mark.parametrize(
    ("raw", "value"),
    [("0", 0), ("1500", 1500), ("1 500", 1500), ("12 345 678", 12345678)],
)
def test_parses_plain_and_three_digit_grouped_integers(raw: str, value: int) -> None:
    assert parse_integer(raw) == value


@pytest.mark.parametrize("raw", ["", "12 34", "1 50 000", "5 до 7"])
def test_rejects_invalid_integer_grouping(raw: str) -> None:
    with pytest.raises(InputError, match="integer"):
        parse_integer(raw)


def test_generates_verified_cardinal_cases() -> None:
    assert cardinal(56, case="g") == "пятидесяти шести"
    assert cardinal(142, case="g") == "ста сорока двух"
    assert cardinal("1 500") == "одна тысяча пятьсот"


def test_generates_verified_ordinal_cases_and_genders() -> None:
    assert ordinal(142) == "сто сорок второй"
    assert ordinal(19, case="g") == "девятнадцатого"
    assert ordinal(2026, case="p") == "две тысячи двадцать шестом"
    assert ordinal(2026, case="d") == "две тысячи двадцать шестому"
    assert ordinal(1, gender="f") == "первая"


def test_generates_plural_ordinals_for_decades() -> None:
    assert ordinal(90, case="g", plural=True) == "девяностых"
    assert ordinal(1990, case="g", plural=True) == "тысяча девятьсот девяностых"
    assert ordinal(1990, case="i", plural=True) == "тысяча девятьсот девяностыми"
    # default remains singular
    assert ordinal(90, case="g") == "девяностого"


def test_preserves_decimal_precision_from_text() -> None:
    assert decimal_words("3", "14") == "три целых четырнадцать сотых"
    assert decimal_words("5", "10") == "пять целых десять сотых"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "рубль"), (2, "рубля"), (5, "рублей"), (11, "рублей"), (21, "рубль")],
)
def test_selects_russian_noun_form(value: int, expected: str) -> None:
    assert noun_form(value, ("рубль", "рубля", "рублей")) == expected
