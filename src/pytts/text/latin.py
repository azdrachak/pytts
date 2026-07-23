from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import cyrtranslit

from pytts.errors import InputError

_ENGLISH_LETTER_NAMES = {
    "A": "эй",
    "B": "би",
    "C": "си",
    "D": "ди",
    "E": "и",
    "F": "эф",
    "G": "джи",
    "H": "эйч",
    "I": "ай",
    "J": "джей",
    "K": "кей",
    "L": "эл",
    "M": "эм",
    "N": "эн",
    "O": "оу",
    "P": "пи",
    "Q": "кью",
    "R": "ар",
    "S": "эс",
    "T": "ти",
    "U": "ю",
    "V": "ви",
    "W": "дабл-ю",
    "X": "икс",
    "Y": "уай",
    "Z": "зед",
}
_RUSSIAN_LETTER_NAMES = {
    "А": "а",
    "Б": "бэ",
    "В": "вэ",
    "Г": "гэ",
    "Д": "дэ",
    "Е": "е",
    "Ё": "ё",
    "Ж": "жэ",
    "З": "зэ",
    "И": "и",
    "Й": "и краткое",
    "К": "ка",
    "Л": "эл",
    "М": "эм",
    "Н": "эн",
    "О": "о",
    "П": "пэ",
    "Р": "эр",
    "С": "эс",
    "Т": "тэ",
    "У": "у",
    "Ф": "эф",
    "Х": "ха",
    "Ц": "цэ",
    "Ч": "че",
    "Ш": "ша",
    "Щ": "ща",
    "Ъ": "твёрдый знак",
    "Ы": "ы",
    "Ь": "мягкий знак",
    "Э": "э",
    "Ю": "ю",
    "Я": "я",
}
_LETTER_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)


def spell_latin_letters(token: str) -> str:
    if (
        not 1 <= len(token) <= 5
        or not token.isascii()
        or not token.isupper()
        or not token.isalpha()
    ):
        raise ValueError("Latin letter group must contain 1-5 uppercase ASCII letters")
    return " ".join(_ENGLISH_LETTER_NAMES[letter] for letter in token)


def spell_code_letters(token: str) -> str:
    if token.isascii():
        return spell_latin_letters(token)
    if len(token) == 1 and token in _RUSSIAN_LETTER_NAMES:
        return _RUSSIAN_LETTER_NAMES[token]
    raise ValueError("Code letters must be 1-5 uppercase Latin letters or one Cyrillic letter")


class TransliterationOverrides:
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._replacements = {key.casefold(): value for key, value in mapping.items()}
        if mapping:
            alternatives = "|".join(
                re.escape(key) for key in sorted(mapping, key=len, reverse=True)
            )
            self._pattern = re.compile(
                rf"(?<![^\W_])(?:{alternatives})(?![^\W_])",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            self._pattern = None

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text

        def replace(match: re.Match[str]) -> str:
            return self._replacements[match.group(0).casefold()]

        return self._pattern.sub(replace, text)


def _is_latin_letter(character: str) -> bool:
    return character.isalpha() and "LATIN" in unicodedata.name(character, "")


def _ascii_latin(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token.replace("ĸ", "k"))
    result = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    if not result.isascii() or not result.isalpha():
        raise InputError(f"Unsupported Latin token {token!r}; add it to transliterations")
    return result


def _latin_replacement(match: re.Match[str]) -> str:
    token = match.group(0)
    if not all(_is_latin_letter(character) for character in token):
        return token
    ascii_token = _ascii_latin(token)
    if ascii_token.isupper() and len(ascii_token) <= 5:
        return spell_latin_letters(ascii_token)
    try:
        transliterated = cyrtranslit.to_cyrillic(ascii_token, "ru")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise InputError(f"Could not transliterate Latin token {token!r}: {error}") from error
    return transliterated.replace("X", "Кс").replace("x", "кс")


def normalize_latin(text: str) -> str:
    return _LETTER_RUN.sub(_latin_replacement, text)
