from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from pytts.errors import InputError
from pytts.text.latin import spell_code_letters
from pytts.text.russian_numbers import (
    RussianGender,
    cardinal,
    decimal_words,
    noun_form,
    ordinal,
    parse_integer,
)

_INTEGER = r"(?:\d{1,3}(?: \d{3})+|\d+)"
_DASH = r"[-–—]"
_SIGN = r"(?:-|−\s*)"
_MONTHS = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}
_MONTH_NUMBERS = {name: number for number, name in _MONTHS.items()}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_CANONICAL = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
# Roman numerals are only resolved before a century noun, where they are
# unambiguous; a bare "XX" (e.g. "XX съезд") is left for later Latin spelling to
# avoid misreading Latin words made of Roman letters ("MIX", "DID").
_CENTURY_CASES = {"век": "n", "века": "g", "веке": "p", "веком": "i"}
_ROMAN_CENTURY = re.compile(
    r"(?<!\w)(?P<roman>[IVXLCDM]+)\s+(?P<noun>веком|века|веке|век)(?!\w)"
)


def _roman_to_int(token: str) -> int | None:
    if not token or not _ROMAN_CANONICAL.fullmatch(token):
        return None
    total = 0
    highest = 0
    for character in reversed(token):
        value = _ROMAN_VALUES[character]
        if value < highest:
            total -= value
        else:
            total += value
            highest = value
    return total


_NUMERIC_DATE = re.compile(r"(?<!\w)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\w)")
_TEXT_DATE = re.compile(
    rf"(?<!\w)(?P<day>\d{{1,2}})\s+(?P<month>{'|'.join(_MONTH_NUMBERS)})\s+"
    r"(?P<year>\d{4})\s+года(?!\w)",
    re.IGNORECASE,
)
_YEAR_CONTEXT = re.compile(
    r"(?<!\w)(?P<preposition>в|на|с|к)\s+(?P<year>\d{4})\s+"
    r"(?P<noun>год|года|году)(?!\w)",
    re.IGNORECASE,
)
_CODE = re.compile(
    rf"(?<!\w)(?P<letters>[A-Z]{{1,5}}|[А-ЯЁ])-(?P<number>{_INTEGER})(?!\w)"
)
_COMPOUND_YEARS = re.compile(
    rf"(?<!\w)(?P<number>{_INTEGER})-(?P<suffix>летн[а-яё]*)(?!\w)",
    re.IGNORECASE,
)
_COMPOUND_DOLLARS = re.compile(
    rf"(?<!\w)(?P<number>{_INTEGER})-(?P<suffix>долларов[а-яё]*)(?!\w)",
    re.IGNORECASE,
)
_ORDINAL_SUFFIXES = {
    "й": ("n", "m"),
    "я": ("n", "f"),
    "е": ("n", "n"),
    "го": ("g", "m"),
    "му": ("d", "m"),
    "м": ("p", "m"),
    "ую": ("a", "f"),
    "ой": ("g", "f"),
}
_ORDINAL = re.compile(
    rf"(?<!\w)(?P<number>{_INTEGER})-"
    rf"(?P<suffix>{'|'.join(sorted(_ORDINAL_SUFFIXES, key=len, reverse=True))})(?!\w)",
    re.IGNORECASE,
)
# Decade forms like "1990-х"/"80-х": only plural-unambiguous suffixes, and only
# numbers ending in a zero, so singular ordinals such as "20-й"/"90-м" are left
# to _ORDINAL. "-е"/"-м" are intentionally excluded because they collide with
# singular neuter/prepositional ordinals of round numbers ("20-е число").
_DECADE_CASES = {"ыми": "i", "ых": "g", "ми": "i", "х": "g"}
_DECADE = re.compile(
    rf"(?<!\w)(?P<number>\d{{1,3}}0)-"
    rf"(?P<suffix>{'|'.join(sorted(_DECADE_CASES, key=len, reverse=True))})(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CurrencyForms:
    major: tuple[str, str, str]
    minor: tuple[str, str, str]
    major_gender: RussianGender = "m"
    minor_gender: RussianGender = "m"


_CURRENCIES = {
    "RUB": CurrencyForms(
        ("рубль", "рубля", "рублей"),
        ("копейка", "копейки", "копеек"),
        minor_gender="f",
    ),
    "USD": CurrencyForms(
        ("доллар", "доллара", "долларов"),
        ("цент", "цента", "центов"),
    ),
    "EUR": CurrencyForms(
        ("евро", "евро", "евро"),
        ("цент", "цента", "центов"),
    ),
    "GBP": CurrencyForms(
        ("фунт", "фунта", "фунтов"),
        ("пенс", "пенса", "пенсов"),
    ),
}
_CURRENCY_ALIASES = {
    "₽": "RUB",
    "руб.": "RUB",
    "rub": "RUB",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "евро": "EUR",
}
_AMOUNT = rf"{_INTEGER}(?:[.,]\d{{1,2}})?"
_CURRENCY_TOKEN = r"(?:руб\.|RUB|USD|EUR|GBP|евро|₽|\$|€|£)"
_CURRENCY_PREFIX_RANGE = re.compile(
    rf"(?<!\w)(?P<currency>[₽$€£])\s*(?P<left>{_INTEGER})\s*{_DASH}\s*"
    rf"(?P<right>{_INTEGER})(?!\w)"
)
_CURRENCY_SUFFIX_RANGE = re.compile(
    rf"(?<!\w)(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})\s*"
    rf"(?P<currency>{_CURRENCY_TOKEN})(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RANGE = re.compile(
    rf"(?<!\w)(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})\s*%(?!\w)"
)
_CURRENCY_PREFIX = re.compile(
    rf"(?<!\w)(?P<outer_sign>{_SIGN})?(?P<currency>[₽$€£])\s*"
    rf"(?P<inner_sign>{_SIGN})?(?P<amount>{_AMOUNT})(?!\w|[.,]\d)"
)
_CURRENCY_SUFFIX = re.compile(
    rf"(?<![\w.,])(?P<sign>{_SIGN})?(?P<amount>{_AMOUNT})\s*"
    rf"(?P<currency>{_CURRENCY_TOKEN})(?!\w)",
    re.IGNORECASE,
)
_PERCENT_PLUS = re.compile(
    rf"(?<![\w.,])(?P<number>{_INTEGER})\s*\+\s*%(?!\w)"
)
_PERCENT = re.compile(
    rf"(?<![\w.,])(?P<sign>{_SIGN})?"
    rf"(?P<number>{_INTEGER})\s*%(?!\w)"
)
_EXPLICIT_RANGE = re.compile(
    rf"(?<!\w)от\s+(?P<left>{_INTEGER})\s+до\s+(?P<right>{_INTEGER})(?!\w)",
    re.IGNORECASE,
)
_BARE_RANGE = re.compile(
    rf"(?<![\w-])(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})(?!\w)"
)
_DEGREES = re.compile(
    rf"(?<!\w)(?P<sign>{_SIGN})?(?P<number>{_INTEGER})\s*°\s*"
    rf"(?P<celsius>[CС])?(?!\w)"
)
_DECIMAL = re.compile(
    rf"(?<![\w.,])(?P<sign>{_SIGN})?"
    rf"(?P<integer>{_INTEGER})(?P<separator>[.,])"
    r"(?P<fraction>\d{1,2})(?!\w|[.,]\d)"
)
_REMAINING_INTEGER = re.compile(
    rf"(?<!\w)(?P<sign>{_SIGN})?(?P<number>{_INTEGER})(?!\w)"
)


def _spoken_date(day: int, month: int, year: int) -> str:
    try:
        date(year, month, day)
    except ValueError as error:
        raise InputError(
            f"Invalid calendar date: {day:02d}.{month:02d}.{year}"
        ) from error
    return f"{ordinal(day, case='g')} {_MONTHS[month]} {ordinal(year, case='g')} года"


def _currency_amount_words(raw: str, code: str) -> str:
    normalized = raw.replace(" ", "").replace(",", ".")
    whole_text, separator, fraction_text = normalized.partition(".")
    whole = int(whole_text)
    forms = _CURRENCIES[code]
    pieces = [
        cardinal(whole, gender=forms.major_gender),
        noun_form(whole, forms.major),
    ]
    if separator:
        minor = int(fraction_text.ljust(2, "0"))
        pieces.extend(
            (
                cardinal(minor, gender=forms.minor_gender),
                noun_form(minor, forms.minor),
            )
        )
    return " ".join(pieces)


def _range_words(left: str, right: str, noun: str | None = None) -> str:
    result = f"от {cardinal(left, case='g')} до {cardinal(right, case='g')}"
    return f"{result} {noun}" if noun else result


def _with_sign(words: str, sign: str | None) -> str:
    return f"минус {words}" if sign else words


class NumericNormalizer:
    def normalize(self, text: str) -> str:
        text = _ROMAN_CENTURY.sub(self._roman_century, text)
        text = _TEXT_DATE.sub(self._text_date, text)
        text = _NUMERIC_DATE.sub(self._numeric_date, text)
        text = _YEAR_CONTEXT.sub(self._year_context, text)
        text = _CURRENCY_PREFIX_RANGE.sub(self._currency_range, text)
        text = _CURRENCY_SUFFIX_RANGE.sub(self._currency_range, text)
        text = _PERCENT_RANGE.sub(self._percent_range, text)
        text = _CURRENCY_PREFIX.sub(self._currency_amount, text)
        text = _CURRENCY_SUFFIX.sub(self._currency_amount, text)
        text = _PERCENT_PLUS.sub(self._percent_plus, text)
        text = _PERCENT.sub(self._percent, text)
        text = _EXPLICIT_RANGE.sub(self._explicit_range, text)
        text = _BARE_RANGE.sub(self._bare_range, text)
        text = _CODE.sub(self._code, text)
        text = _COMPOUND_YEARS.sub(self._compound_years, text)
        text = _COMPOUND_DOLLARS.sub(self._compound_dollars, text)
        text = _DECADE.sub(self._decade, text)
        text = _ORDINAL.sub(self._ordinal, text)
        text = _DEGREES.sub(self._degrees, text)
        text = _DECIMAL.sub(self._decimal, text)
        text = _REMAINING_INTEGER.sub(self._remaining_integer, text)
        return text

    def _roman_century(self, match: re.Match[str]) -> str:
        value = _roman_to_int(match.group("roman"))
        if value is None:
            return match.group(0)
        noun = match.group("noun")
        return f"{ordinal(value, case=_CENTURY_CASES[noun.casefold()])} {noun}"

    def _text_date(self, match: re.Match[str]) -> str:
        day = int(match.group("day"))
        month = _MONTH_NUMBERS[match.group("month").casefold()]
        year = int(match.group("year"))
        return _spoken_date(day, month, year)

    def _numeric_date(self, match: re.Match[str]) -> str:
        day, month, year = (int(group) for group in match.groups())
        return _spoken_date(day, month, year)

    def _year_context(self, match: re.Match[str]) -> str:
        preposition = match.group("preposition")
        noun = match.group("noun")
        case = {
            ("в", "году"): "p",
            ("на", "году"): "p",
            ("с", "года"): "g",
            ("к", "году"): "d",
        }.get((preposition.casefold(), noun.casefold()), "n")
        return f"{preposition} {ordinal(match.group('year'), case=case)} {noun}"

    def _currency_amount(self, match: re.Match[str]) -> str:
        raw_currency = match.group("currency")
        code = _CURRENCY_ALIASES[raw_currency.casefold()]
        outer_sign = match.groupdict().get("outer_sign")
        inner_sign = match.groupdict().get("inner_sign")
        sign = match.groupdict().get("sign")
        signs = [value for value in (outer_sign, inner_sign, sign) if value]
        if len(signs) > 1:
            raise InputError(f"Currency amount has multiple signs: {match.group(0)!r}")
        words = _currency_amount_words(match.group("amount"), code)
        return _with_sign(words, signs[0] if signs else None)

    def _currency_range(self, match: re.Match[str]) -> str:
        raw_currency = match.group("currency")
        code = _CURRENCY_ALIASES[raw_currency.casefold()]
        noun = _CURRENCIES[code].major[2]
        return _range_words(match.group("left"), match.group("right"), noun)

    def _percent_range(self, match: re.Match[str]) -> str:
        return _range_words(
            match.group("left"), match.group("right"), "процентов"
        )

    def _percent(self, match: re.Match[str]) -> str:
        raw = match.group("number")
        value = parse_integer(raw)
        forms = ("процент", "процента", "процентов")
        words = f"{cardinal(raw)} {noun_form(value, forms)}"
        return _with_sign(words, match.group("sign"))

    def _percent_plus(self, match: re.Match[str]) -> str:
        return f"более {cardinal(match.group('number'), case='g')} процентов"

    def _decade(self, match: re.Match[str]) -> str:
        case = _DECADE_CASES[match.group("suffix").casefold()]
        return ordinal(match.group("number"), case=case, plural=True)

    def _explicit_range(self, match: re.Match[str]) -> str:
        return _range_words(match.group("left"), match.group("right"))

    def _bare_range(self, match: re.Match[str]) -> str:
        return _range_words(match.group("left"), match.group("right"))

    def _code(self, match: re.Match[str]) -> str:
        return (
            f"{spell_code_letters(match.group('letters'))} "
            f"{cardinal(match.group('number'))}"
        )

    def _compound_years(self, match: re.Match[str]) -> str:
        prefix = cardinal(match.group("number"), case="g").replace(" ", "")
        return prefix + match.group("suffix")

    def _compound_dollars(self, match: re.Match[str]) -> str:
        prefix = cardinal(match.group("number")).replace(" ", "")
        return prefix + match.group("suffix")

    def _ordinal(self, match: re.Match[str]) -> str:
        case, gender = _ORDINAL_SUFFIXES[match.group("suffix").casefold()]
        return ordinal(match.group("number"), case=case, gender=gender)

    def _degrees(self, match: re.Match[str]) -> str:
        raw = match.group("number")
        value = parse_integer(raw)
        forms = ("градус", "градуса", "градусов")
        result = f"{cardinal(raw)} {noun_form(value, forms)}"
        if match.group("celsius"):
            result = f"{result} Цельсия"
        return _with_sign(result, match.group("sign"))

    def _decimal(self, match: re.Match[str]) -> str:
        words = decimal_words(match.group("integer"), match.group("fraction"))
        return _with_sign(words, match.group("sign"))

    def _remaining_integer(self, match: re.Match[str]) -> str:
        words = cardinal(match.group("number"))
        return _with_sign(words, match.group("sign"))
