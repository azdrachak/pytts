from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "a": "а",
        "B": "В",
        "C": "С",
        "c": "с",
        "E": "Е",
        "e": "е",
        "H": "Н",
        "K": "К",
        "k": "к",
        "ĸ": "к",
        "M": "М",
        "O": "О",
        "o": "о",
        "P": "Р",
        "p": "р",
        "T": "Т",
        "X": "Х",
        "x": "х",
    }
)
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "а": "a",
        "В": "B",
        "С": "C",
        "с": "c",
        "Е": "E",
        "е": "e",
        "Н": "H",
        "К": "K",
        "к": "k",
        "М": "M",
        "О": "O",
        "о": "o",
        "Р": "P",
        "р": "p",
        "Т": "T",
        "Х": "X",
        "х": "x",
    }
)


def _script(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    return None


def _repair_token(match: re.Match[str]) -> str:
    token = match.group(0)
    scripts = [_script(character) for character in token]
    latin = sum(script == "latin" for script in scripts)
    cyrillic = sum(script == "cyrillic" for script in scripts)
    if not latin or not cyrillic:
        return token
    if latin == cyrillic:
        code_like = re.fullmatch(r"[A-ZА-ЯЁ0-9-]+", token) is not None
        target = "latin" if code_like else "cyrillic"
    else:
        target = "latin" if latin > cyrillic else "cyrillic"
    table = _CYRILLIC_TO_LATIN if target == "latin" else _LATIN_TO_CYRILLIC
    return token.translate(table)


def repair_mixed_scripts(text: str) -> str:
    return _TOKEN.sub(_repair_token, text)
