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
    rf"(?<!\w)(?P<currency>[₽$€£])\s*(?P<amount>{_AMOUNT})(?!\w|[.,]\d)"
)
_CURRENCY_SUFFIX = re.compile(
    rf"(?<![\w.,])(?P<amount>{_AMOUNT})\s*"
    rf"(?P<currency>{_CURRENCY_TOKEN})(?!\w)",
    re.IGNORECASE,
)
_PERCENT = re.compile(rf"(?<![\w.,])(?P<number>{_INTEGER})\s*%(?!\w)")
_EXPLICIT_RANGE = re.compile(
    rf"(?<!\w)от\s+(?P<left>{_INTEGER})\s+до\s+(?P<right>{_INTEGER})(?!\w)",
    re.IGNORECASE,
)
_BARE_RANGE = re.compile(
    rf"(?<![\w-])(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})(?!\w)"
)
_DEGREES = re.compile(
    rf"(?<!\w)(?P<number>{_INTEGER})\s*°\s*(?P<celsius>[CС])?(?!\w)"
)
_DECIMAL = re.compile(
    rf"(?<![\w.,])(?P<integer>{_INTEGER})(?P<separator>[.,])"
    r"(?P<fraction>\d{1,2})(?![\w.,])"
)
_REMAINING_INTEGER = re.compile(_INTEGER)


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


class NumericNormalizer:
    def normalize(self, text: str) -> str:
        text = _TEXT_DATE.sub(self._text_date, text)
        text = _NUMERIC_DATE.sub(self._numeric_date, text)
        text = _YEAR_CONTEXT.sub(self._year_context, text)
        text = _CURRENCY_PREFIX_RANGE.sub(self._currency_range, text)
        text = _CURRENCY_SUFFIX_RANGE.sub(self._currency_range, text)
        text = _PERCENT_RANGE.sub(self._percent_range, text)
        text = _CURRENCY_PREFIX.sub(self._currency_amount, text)
        text = _CURRENCY_SUFFIX.sub(self._currency_amount, text)
        text = _PERCENT.sub(self._percent, text)
        text = _EXPLICIT_RANGE.sub(self._explicit_range, text)
        text = _BARE_RANGE.sub(self._bare_range, text)
        text = _CODE.sub(self._code, text)
        text = _COMPOUND_YEARS.sub(self._compound_years, text)
        text = _COMPOUND_DOLLARS.sub(self._compound_dollars, text)
        text = _ORDINAL.sub(self._ordinal, text)
        text = _DEGREES.sub(self._degrees, text)
        text = _DECIMAL.sub(self._decimal, text)
        text = _REMAINING_INTEGER.sub(self._remaining_integer, text)
        return text

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
        return _currency_amount_words(match.group("amount"), code)

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
        return f"{cardinal(raw)} {noun_form(value, forms)}"

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
        return f"{result} Цельсия" if match.group("celsius") else result

    def _decimal(self, match: re.Match[str]) -> str:
        return decimal_words(match.group("integer"), match.group("fraction"))

    def _remaining_integer(self, match: re.Match[str]) -> str:
        return cardinal(match.group(0))
