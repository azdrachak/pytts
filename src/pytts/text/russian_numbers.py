from __future__ import annotations

import re
from typing import Literal

from num2words import num2words

from pytts.errors import InputError

RussianCase = Literal["n", "g", "d", "a", "i", "p"]
RussianGender = Literal["m", "f", "n"]
_GROUPED_INTEGER = re.compile(r"(?:\d{1,3}(?: \d{3})+|\d+)")


def parse_integer(raw: str) -> int:
    if _GROUPED_INTEGER.fullmatch(raw) is None:
        raise InputError(f"Invalid integer grouping: {raw!r}")
    return int(raw.replace(" ", ""))


def _number_value(raw: str | int) -> int:
    if type(raw) is int:
        return raw
    return parse_integer(raw)


def _words(
    raw: str | int,
    *,
    to: Literal["cardinal", "ordinal"],
    case: RussianCase,
    gender: RussianGender,
    plural: bool = False,
) -> str:
    value = _number_value(raw)
    try:
        if to == "ordinal":
            result = num2words(
                value, lang="ru", to=to, case=case, gender=gender, plural=plural
            )
        else:
            result = num2words(value, lang="ru", to=to, case=case, gender=gender)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise InputError(f"Could not normalize number {raw!r}: {error}") from error
    if not isinstance(result, str) or not result:
        raise InputError(f"Could not normalize number {raw!r}: empty result")
    return result


def cardinal(
    raw: str | int,
    *,
    case: RussianCase = "n",
    gender: RussianGender = "m",
) -> str:
    return _words(raw, to="cardinal", case=case, gender=gender)


def ordinal(
    raw: str | int,
    *,
    case: RussianCase = "n",
    gender: RussianGender = "m",
    plural: bool = False,
) -> str:
    return _words(raw, to="ordinal", case=case, gender=gender, plural=plural)


def decimal_words(integer: str, fraction: str) -> str:
    whole = parse_integer(integer)
    if not fraction.isdigit() or not 1 <= len(fraction) <= 2:
        raise InputError(f"Invalid decimal fraction: {fraction!r}")
    value = f"{whole}.{fraction}"
    try:
        result = num2words(value, lang="ru")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise InputError(f"Could not normalize decimal {value!r}: {error}") from error
    if not isinstance(result, str) or not result:
        raise InputError(f"Could not normalize decimal {value!r}: empty result")
    return result


def noun_form(value: int, forms: tuple[str, str, str]) -> str:
    remainder = abs(value) % 100
    if 11 <= remainder <= 14:
        return forms[2]
    final = remainder % 10
    if final == 1:
        return forms[0]
    if 2 <= final <= 4:
        return forms[1]
    return forms[2]
