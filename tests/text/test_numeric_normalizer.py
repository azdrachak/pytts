import pytest

from pytts.errors import InputError
from pytts.text.numeric_normalizer import NumericNormalizer


def _normalize(text: str) -> str:
    return NumericNormalizer().normalize(text)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "19 июля 2026 года",
            "девятнадцатого июля две тысячи двадцать шестого года",
        ),
        (
            "19.07.2026",
            "девятнадцатого июля две тысячи двадцать шестого года",
        ),
        ("в 2026 году", "в две тысячи двадцать шестом году"),
        ("с 2026 года", "с две тысячи двадцать шестого года"),
        ("к 2026 году", "к две тысячи двадцать шестому году"),
        ("142-й день", "сто сорок второй день"),
        (
            "56-летний и 56-летнего",
            "пятидесятишестилетний и пятидесятишестилетнего",
        ),
        (
            "90-долларовый и 90-долларового",
            "девяностодолларовый и девяностодолларового",
        ),
        # The first letter in С-300 is Cyrillic U+0421, not Latin C.
        ("FP-5, F-16, С-300", "эф пи пять, эф шестнадцать, эс триста"),
    ],
)
def test_normalizes_dates_years_ordinals_compounds_and_codes(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


def test_rejects_invalid_numeric_calendar_date() -> None:
    with pytest.raises(InputError, match="Invalid calendar date"):
        _normalize("10.20.2021")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "1 ₽, 2 ₽, 5 ₽, 11 ₽, 21 ₽",
            "один рубль, два рубля, пять рублей, одиннадцать рублей, "
            "двадцать один рубль",
        ),
        ("1 €, 2 EUR, 5 евро", "один евро, два евро, пять евро"),
        ("1 £, 2 GBP, 5 USD", "один фунт, два фунта, пять долларов"),
        ("12,50 ₽", "двенадцать рублей пятьдесят копеек"),
        ("1,01 ₽", "один рубль одна копейка"),
        ("1,02 ₽", "один рубль две копейки"),
        ("0,21 ₽", "ноль рублей двадцать одна копейка"),
        ("$5.", "пять долларов."),
        ("€10,", "десять евро,"),
        ("15% и 21 %", "пятнадцать процентов и двадцать один процент"),
        ("от 5 до 7", "от пяти до семи"),
        ("3–5", "от трёх до пяти"),
        ("15–20%", "от пятнадцати до двадцати процентов"),
        ("$5–7", "от пяти до семи долларов"),
    ],
)
def test_normalizes_currency_percentages_and_ranges(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "20 °C, 1° и 2°",
            "двадцать градусов Цельсия, один градус и два градуса",
        ),
        (
            "3,14; 3, 14; 3.14",
            "три целых четырнадцать сотых; три, четырнадцать; "
            "три целых четырнадцать сотых",
        ),
        (
            "1 500 и 12 345 678",
            "одна тысяча пятьсот и двенадцать миллионов триста сорок пять "
            "тысяч шестьсот семьдесят восемь",
        ),
        ("Версия 12 34", "Версия двенадцать тридцать четыре"),
    ],
)
def test_normalizes_degrees_decimals_grouping_and_remaining_integers(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


def test_numeric_normalization_is_idempotent() -> None:
    once = _normalize("В 2026 году было 15–20% и 1 500 ₽.")
    assert _normalize(once) == once


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("X5 Group", "икс пять Group"),
        ("FPV5", "эф пи ви пять"),
        ("F16 и F-16", "эф шестнадцать и эф шестнадцать"),
    ],
)
def test_normalizes_joined_uppercase_letter_number_codes(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    "source",
    ["preX5", "X5post", "_X5", "X5_", "x5"],
)
def test_joined_codes_require_whole_word_boundaries_and_uppercase(
    source: str,
) -> None:
    assert _normalize(source) == source


def test_joined_code_normalization_is_idempotent() -> None:
    once = _normalize("X5 и FPV5")

    assert _normalize(once) == once


@pytest.mark.parametrize("sign", ["-", "−"])
@pytest.mark.parametrize(
    ("source_template", "expected"),
    [
        ("{}5 °C", "минус пять градусов Цельсия"),
        ("{}15%", "минус пятнадцать процентов"),
        ("{}5 ₽", "минус пять рублей"),
        ("{}3,14", "минус три целых четырнадцать сотых"),
        ("{}42", "минус сорок два"),
    ],
)
def test_normalizes_ascii_and_unicode_minus_for_numeric_families(
    sign: str, source_template: str, expected: str
) -> None:
    assert _normalize(source_template.format(sign)) == expected


@pytest.mark.parametrize("source", ["-$5", "$-5", "−$5", "$−5"])
def test_normalizes_minus_on_either_side_of_prefix_currency(source: str) -> None:
    assert _normalize(source) == "минус пять долларов"


@pytest.mark.parametrize("sign", ["-", "−"])
@pytest.mark.parametrize("separator", [",", "."])
@pytest.mark.parametrize("terminator", [".", ",", "»", ")", "]"])
def test_normalizes_signed_decimal_before_sentence_punctuation(
    sign: str, separator: str, terminator: str
) -> None:
    source = f"{sign}3{separator}14{terminator}"
    expected = f"минус три целых четырнадцать сотых{terminator}"

    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Рост - 15%.", "Рост - пятнадцать процентов."),
        ("Цена - 5 ₽.", "Цена - пять рублей."),
        ("Итог - 5 пунктов.", "Итог - пять пунктов."),
        ("Рост − 15%.", "Рост минус пятнадцать процентов."),
    ],
)
def test_distinguishes_prose_dash_from_numeric_minus(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("XX века", "двадцатого века"),
        ("XXI век", "двадцать первый век"),
        ("в XIX веке", "в девятнадцатом веке"),
        ("до середины XX века", "до середины двадцатого века"),
        ("XVIII веком", "восемнадцатым веком"),
    ],
)
def test_normalizes_roman_centuries(source: str, expected: str) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "XX съезд",  # roman without a century noun: numeric stage leaves it
        "IIII века",  # non-canonical roman: left untouched
    ],
)
def test_leaves_non_century_roman_untouched(source: str) -> None:
    assert _normalize(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("39+%", "более тридцати девяти процентов"),
        (
            "рост 39+% в год",
            "рост более тридцати девяти процентов в год",
        ),
    ],
)
def test_normalizes_open_ended_percent(source: str, expected: str) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("19,7%", "девятнадцать целых семь десятых процента"),
        ("3,5 %", "три целых пять десятых процента"),
        ("0,1%", "ноль целых одна десятая процента"),
        (
            "было всего 19,7%.",
            "было всего девятнадцать целых семь десятых процента.",
        ),
        ("-3,5%", "минус три целых пять десятых процента"),
        # integer percent stays on its own rule
        ("15%", "пятнадцать процентов"),
    ],
)
def test_normalizes_decimal_percent(source: str, expected: str) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("в 1990-х годах", "в тысяча девятьсот девяностых годах"),
        ("в конце 1970-х", "в конце тысяча девятьсот семидесятых"),
        ("в начале 1980-х", "в начале тысяча девятьсот восьмидесятых"),
        ("в 80-х", "в восьмидесятых"),
        ("2020-х", "две тысячи двадцатых"),
        ("1990-ми", "тысяча девятьсот девяностыми"),
    ],
)
def test_normalizes_decades_as_plural_ordinals(source: str, expected: str) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # round-number singular ordinals must stay singular, not become decades
        ("20-й век", "двадцатый век"),
        ("к 90-му дню", "к девяностому дню"),
        ("142-й день", "сто сорок второй день"),
    ],
)
def test_singular_ordinals_of_round_numbers_are_not_decades(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected
