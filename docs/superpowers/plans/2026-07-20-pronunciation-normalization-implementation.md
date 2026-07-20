# Pronunciation Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert digits, Russian numeric constructions, currencies, semantic symbols and Latin fragments into deterministic Russian speech text before SSML chunking, including Safari PDF mixed-script repair.

**Architecture:** Keep extraction, pronunciation normalization and synthesis separate. `cleaner.py` repairs mixed-script lookalikes; focused text modules handle user overrides, Latin tokens and Russian number morphology; `PronunciationNormalizer` composes them before the existing chunker. Silero continues to receive only SSML, so all speed modes remain available.

**Tech Stack:** Python 3.12, `num2words` Russian forms, `cyrtranslit`, PyYAML, PyMuPDF, Silero `v5_5_ru`, pytest, Ruff, uv.

## Global Constraints

- Runtime remains Python `>=3.12,<3.13` on local Apple Silicon macOS CPU.
- The application remains Russian-only, single-input CLI; no OCR, cloud API, GPU, `ffmpeg`, translation or LLM.
- Normalization runs before `chunk_article`; every clean-text payload remains at most the manifest limit of 448 characters.
- `SileroRuntime.synthesize` continues to call `apply_tts` with the `ssml_text` keyword; do not add a plain-text fallback.
- Normalization is local, deterministic and idempotent. A cached model must still work with the network disabled.
- Unsupported letters/symbols fail closed as `InputError` before model loading and before opening the MP3 writer.
- Preserve `KeyboardInterrupt` and `SystemExit`; do not wrap them as application errors.
- Preserve user-owned `.DS_Store`, `.idea`, model files and MP3 artifacts; stage only files named by each task.
- The old Silero Phase 0 probe remains a raw-model diagnostic and must not call the new application normalizer.
- Follow TDD in every task: observe the requested failure before implementation, then run the focused and regression tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pytts/config.py` | Strict version-1 parsing of abbreviation and transliteration dictionaries. |
| `src/pytts/text/confusables.py` | Context-sensitive repair of mixed Latin/Cyrillic lookalikes only. |
| `src/pytts/text/latin.py` | Longest-first overrides, letter names and approximate remaining Latin transliteration. |
| `src/pytts/text/russian_numbers.py` | Thin, exception-safe wrapper around verified Russian `num2words` APIs and noun-form selection. |
| `src/pytts/text/numeric_normalizer.py` | Ordered regex rules for dates, years, currency, ranges, percentages, codes, compound forms and remaining numbers. |
| `src/pytts/text/pronunciation.py` | Article-level orchestration, semantic-symbol expansion and final speakability guard. |
| `src/pytts/text/cleaner.py` | Existing cleanup plus one call to mixed-script repair. |
| `src/pytts/pipeline.py` | Invoke pronunciation normalization after abbreviation expansion and before chunking. |
| `pytts.yaml` | Root abbreviation and exact pronunciation overrides. |
| `tests/text/test_*.py` | Focused contracts for each text unit. |
| `tests/test_config.py`, `tests/test_pipeline.py`, `tests/test_e2e.py` | Configuration and whole-pipeline regression contracts. |
| `tests/fixtures/pronunciation.md` | Small manual/real-Silero acceptance article containing every new input family. |
| `tests/test_silero_integration.py` | Opt-in real-model acceptance for the new fixture. |
| `README.md` | User-facing configuration, behavior, errors and known limitations. |

---

### Task 1: Extend the strict root configuration

**Files:**
- Modify: `src/pytts/config.py:37-87`
- Modify: `pytts.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `load_config(explicit_path: Path | None, project_root: Path) -> AppConfig`.
- Produces: `AppConfig.abbreviations: Mapping[str, str]` and `AppConfig.transliterations: Mapping[str, str]`.
- Produces: version-1 YAML accepting optional `transliterations`, with independent case-folded duplicate validation.

- [ ] **Step 1: Write failing configuration tests**

Add these tests to `tests/test_config.py` and update existing successful assertions to inspect both mappings:

```python
def test_loads_optional_transliterations(tmp_path: Path) -> None:
    path = tmp_path / "pytts.yaml"
    path.write_text(
        """version: 1
abbreviations: {}
transliterations:
  Brent: Брент
  New York Times: Нью-Йорк таймс
""",
        encoding="utf-8",
    )

    config = load_config(path, tmp_path)

    assert dict(config.abbreviations) == {}
    assert dict(config.transliterations) == {
        "Brent": "Брент",
        "New York Times": "Нью-Йорк таймс",
    }


def test_config_without_transliterations_is_backward_compatible(tmp_path: Path) -> None:
    path = tmp_path / "pytts.yaml"
    path.write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n',
        encoding="utf-8",
    )

    config = load_config(path, tmp_path)

    assert dict(config.abbreviations) == {"ув.": "уважаемый"}
    assert dict(config.transliterations) == {}


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nabbreviations: {}\ntransliterations: []\n",
        'version: 1\nabbreviations: {}\ntransliterations:\n  Brent: ""\n',
        'version: 1\nabbreviations: {}\ntransliterations:\n  Brent: один\n  BRENT: два\n',
        'version: 1\nabbreviations: {}\ntransliterations:\n  "": пустой\n',
    ],
)
def test_invalid_transliterations_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="transliteration"):
        load_config(path, tmp_path)
```

Also add an exact-duplicate YAML case for `transliterations` to
`test_exact_duplicate_yaml_keys_are_rejected`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: failures report that `AppConfig` has no `transliterations` attribute and that the field is unknown.

- [ ] **Step 3: Generalize mapping validation and add the field**

Refactor `src/pytts/config.py` to this contract:

```python
@dataclass(frozen=True, slots=True)
class AppConfig:
    abbreviations: Mapping[str, str]
    transliterations: Mapping[str, str]


def _validate_mapping(
    value: object,
    path: Path,
    field: str,
) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {field} must be a mapping")
    result: dict[str, str] = {}
    folded: set[str] = set()
    singular = "abbreviation" if field == "abbreviations" else "transliteration"
    for key, replacement in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{path}: {singular} keys must be non-empty strings")
        if not isinstance(replacement, str) or not replacement:
            raise ConfigError(f"{path}: {singular} replacements must be non-empty strings")
        normalized = key.casefold()
        if normalized in folded:
            raise ConfigError(f"{path}: duplicate {singular} ignoring case: {key}")
        folded.add(normalized)
        result[key] = replacement
    return MappingProxyType(result)
```

Change the allowed field set and return paths exactly as follows:

```python
unknown = set(loaded) - {"version", "abbreviations", "transliterations"}

empty: Mapping[str, str] = MappingProxyType({})
if not path.exists() and explicit_path is None:
    return AppConfig(empty, empty)

return AppConfig(
    abbreviations=_validate_mapping(loaded.get("abbreviations"), path, "abbreviations"),
    transliterations=_validate_mapping(
        loaded.get("transliterations", {}), path, "transliterations"
    ),
)
```

Keep `abbreviations` required for compatibility with the current strict schema; a config containing only transliterations must write `abbreviations: {}`.

- [ ] **Step 4: Add root pronunciation overrides**

Append to `pytts.yaml`:

```yaml
transliterations:
  "Brent": "Брент"
  "Bloomberg": "Блумберг"
  "New York Times": "Нью-Йорк таймс"
```

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```bash
uv run pytest tests/test_config.py -v
uv run ruff check src/pytts/config.py tests/test_config.py
```

Expected: both commands pass with no failures or diagnostics.

- [ ] **Step 6: Commit**

```bash
git add src/pytts/config.py pytts.yaml tests/test_config.py
git commit -m "feat: configure pronunciation overrides"
```

---

### Task 2: Repair mixed-script PDF lookalikes

**Files:**
- Create: `src/pytts/text/confusables.py`
- Modify: `src/pytts/text/cleaner.py:40-51`
- Create: `tests/text/test_confusables.py`
- Modify: `tests/text/test_cleaner.py`

**Interfaces:**
- Consumes: arbitrary Unicode text after NFC normalization.
- Produces: `repair_mixed_scripts(text: str) -> str`.
- Guarantee: pure Latin and pure Cyrillic tokens are unchanged; only mixed tokens choose a dominant script.

- [ ] **Step 1: Write failing mixed-script tests**

Create `tests/text/test_confusables.py`:

```python
import pytest

from pytts.text.confusables import repair_mixed_scripts


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("уkраинсĸий", "украинский"),
        ("мoсква и cтатья", "москва и статья"),
        ("FР-5", "FP-5"),
        ("Bloomberg Москва", "Bloomberg Москва"),
        ("Cи", "Си"),
    ],
)
def test_repairs_only_mixed_script_tokens(source: str, expected: str) -> None:
    assert repair_mixed_scripts(source) == expected


@pytest.mark.parametrize(
    ("latin", "cyrillic"),
    [
        ("A", "А"), ("a", "а"), ("B", "В"), ("C", "С"), ("c", "с"),
        ("E", "Е"), ("e", "е"), ("H", "Н"), ("K", "К"), ("k", "к"),
        ("ĸ", "к"), ("M", "М"), ("O", "О"), ("o", "о"), ("P", "Р"),
        ("p", "р"), ("T", "Т"), ("X", "Х"), ("x", "х"),
    ],
)
def test_repairs_every_approved_latin_lookalike_toward_cyrillic(
    latin: str, cyrillic: str
) -> None:
    assert repair_mixed_scripts(f"{cyrillic}{latin}{cyrillic}") == cyrillic * 3
```

Add to `tests/text/test_cleaner.py`:

```python
def test_repairs_safari_mixed_script_artifacts_during_cleanup() -> None:
    result = clean_article(_article("ĸиевсĸий текст и FР-5."))

    assert result.article.blocks[0].text == "киевский текст и FP-5."
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_confusables.py tests/text/test_cleaner.py -v
```

Expected: collection fails because `pytts.text.confusables` does not exist.

- [ ] **Step 3: Implement dominant-script repair**

Create `src/pytts/text/confusables.py` with these tables and algorithm:

```python
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А", "a": "а", "B": "В", "C": "С", "c": "с",
        "E": "Е", "e": "е", "H": "Н", "K": "К", "k": "к",
        "ĸ": "к", "M": "М", "O": "О", "o": "о", "P": "Р",
        "p": "р", "T": "Т", "X": "Х", "x": "х",
    }
)
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A", "а": "a", "В": "B", "С": "C", "с": "c",
        "Е": "E", "е": "e", "Н": "H", "К": "K", "к": "k",
        "М": "M", "О": "O", "о": "o", "Р": "P", "р": "p",
        "Т": "T", "Х": "X", "х": "x",
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
```

In `clean_article`, call it after NFC and before URL/space cleanup:

```python
normalized = unicodedata.normalize("NFC", block.text)
repaired = repair_mixed_scripts(normalized)
without_artifacts = _remove_technical_artifacts(repaired)
```

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
uv run pytest tests/text/test_confusables.py tests/text/test_cleaner.py tests/readers/test_pdf.py -v
uv run ruff check src/pytts/text/confusables.py src/pytts/text/cleaner.py tests/text/test_confusables.py
```

Expected: both commands pass. Existing PDF extraction expectations remain unchanged except for mixed-script repair after reading.

- [ ] **Step 5: Commit**

```bash
git add src/pytts/text/confusables.py src/pytts/text/cleaner.py tests/text/test_confusables.py tests/text/test_cleaner.py
git commit -m "fix: repair mixed-script pdf text"
```

---

### Task 3: Add deterministic Latin pronunciation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/pytts/text/latin.py`
- Create: `tests/text/test_latin.py`

**Interfaces:**
- Consumes: `Mapping[str, str]` from `AppConfig.transliterations` and text already repaired by Task 2.
- Produces: `TransliterationOverrides.apply(text: str) -> str`.
- Produces: `spell_latin_letters(token: str) -> str`, `spell_code_letters(token: str) -> str`, and `normalize_latin(text: str) -> str`.

- [ ] **Step 1: Add the locked runtime dependencies**

Run:

```bash
uv add 'num2words>=0.5.14,<0.6' 'cyrtranslit>=1.2,<2'
uv lock --check
```

Expected: `pyproject.toml` contains both direct dependencies, `uv.lock` resolves them, and the lock check succeeds.

- [ ] **Step 2: Write failing Latin and override tests**

Create `tests/text/test_latin.py`:

```python
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
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_latin.py -v
```

Expected: collection fails because `pytts.text.latin` does not exist.

- [ ] **Step 4: Implement exact overrides and letter-name maps**

Create `src/pytts/text/latin.py`. Define the complete maps, not a partial subset:

```python
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import cyrtranslit

from pytts.errors import InputError

_ENGLISH_LETTER_NAMES = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и",
    "F": "эф", "G": "джи", "H": "эйч", "I": "ай", "J": "джей",
    "K": "кей", "L": "эл", "M": "эм", "N": "эн", "O": "оу",
    "P": "пи", "Q": "кью", "R": "ар", "S": "эс", "T": "ти",
    "U": "ю", "V": "ви", "W": "дабл-ю", "X": "икс",
    "Y": "уай", "Z": "зед",
}
_RUSSIAN_LETTER_NAMES = {
    "А": "а", "Б": "бэ", "В": "вэ", "Г": "гэ", "Д": "дэ",
    "Е": "е", "Ё": "ё", "Ж": "жэ", "З": "зэ", "И": "и",
    "Й": "и краткое", "К": "ка", "Л": "эл", "М": "эм", "Н": "эн",
    "О": "о", "П": "пэ", "Р": "эр", "С": "эс", "Т": "тэ",
    "У": "у", "Ф": "эф", "Х": "ха", "Ц": "цэ", "Ч": "че",
    "Ш": "ша", "Щ": "ща", "Ъ": "твёрдый знак", "Ы": "ы",
    "Ь": "мягкий знак", "Э": "э", "Ю": "ю", "Я": "я",
}
```

Implement the public functions with these contracts:

```python
def spell_latin_letters(token: str) -> str:
    if not 1 <= len(token) <= 5 or not token.isascii() or not token.isupper() or not token.isalpha():
        raise ValueError("Latin letter group must contain 1-5 uppercase ASCII letters")
    return " ".join(_ENGLISH_LETTER_NAMES[letter] for letter in token)


def spell_code_letters(token: str) -> str:
    if token.isascii():
        return spell_latin_letters(token)
    if len(token) == 1 and token in _RUSSIAN_LETTER_NAMES:
        return _RUSSIAN_LETTER_NAMES[token]
    raise ValueError("Code letters must be 1-5 uppercase Latin letters or one Cyrillic letter")
```

`TransliterationOverrides` must compile escaped keys by descending length, use the same outer token-boundary lookarounds as `AbbreviationExpander`, look up with `casefold()`, return configured values exactly, and never rescan replacements.

Use this implementation shape:

```python
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
```

For remaining Latin words, match Unicode letter runs, process only runs whose letters have `LATIN` in `unicodedata.name`, replace `ĸ` with `k`, remove combining marks from NFKD, and require the result to be ASCII `A-Z/a-z`. Use this callback:

```python
def _latin_replacement(match: re.Match[str]) -> str:
    token = match.group(0)
    if not all(_is_latin_letter(character) for character in token):
        return token
    ascii_token = _ascii_latin(token)
    if ascii_token.isupper() and len(ascii_token) <= 5:
        return spell_latin_letters(ascii_token)
    try:
        return cyrtranslit.to_cyrillic(ascii_token, "ru")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise InputError(f"Could not transliterate Latin token {token!r}: {error}") from error
```

Define the surrounding helpers exactly as follows:

```python
_LETTER_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_latin_letter(character: str) -> bool:
    return character.isalpha() and "LATIN" in unicodedata.name(character, "")


def _ascii_latin(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token.replace("ĸ", "k"))
    result = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    if not result.isascii() or not result.isalpha():
        raise InputError(
            f"Unsupported Latin token {token!r}; add it to transliterations"
        )
    return result


def normalize_latin(text: str) -> str:
    return _LETTER_RUN.sub(_latin_replacement, text)
```

- [ ] **Step 5: Run focused tests, dependency audit and Ruff**

Run:

```bash
uv run pytest tests/text/test_latin.py -v
uv tree
uv run ruff check src/pytts/text/latin.py tests/text/test_latin.py
```

Expected: tests and Ruff pass; the dependency tree contains `num2words` and `cyrtranslit`, and contains no newly introduced ML/runtime framework.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/pytts/text/latin.py tests/text/test_latin.py
git commit -m "feat: normalize latin pronunciation"
```

---

### Task 4: Wrap verified Russian number morphology

**Files:**
- Create: `src/pytts/text/russian_numbers.py`
- Create: `tests/text/test_russian_numbers.py`

**Interfaces:**
- Consumes: integer strings with optional valid three-digit space grouping and decimal strings with `,` or `.`.
- Produces: `parse_integer(raw: str) -> int`.
- Produces: `cardinal(raw: str | int, *, case: RussianCase = "n", gender: RussianGender = "m") -> str`.
- Produces: `ordinal(raw: str | int, *, case: RussianCase = "n", gender: RussianGender = "m") -> str`.
- Produces: `decimal_words(integer: str, fraction: str) -> str` and `noun_form(value: int, forms: tuple[str, str, str]) -> str`.

- [ ] **Step 1: Write failing number-wrapper tests**

Create `tests/text/test_russian_numbers.py`:

```python
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


def test_preserves_decimal_precision_from_text() -> None:
    assert decimal_words("3", "14") == "три целых четырнадцать сотых"
    assert decimal_words("5", "10") == "пять целых десять сотых"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "рубль"), (2, "рубля"), (5, "рублей"), (11, "рублей"), (21, "рубль")],
)
def test_selects_russian_noun_form(value: int, expected: str) -> None:
    assert noun_form(value, ("рубль", "рубля", "рублей")) == expected
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
uv run pytest tests/text/test_russian_numbers.py -v
```

Expected: collection fails because `pytts.text.russian_numbers` does not exist.

- [ ] **Step 3: Implement the exception-safe wrapper**

Create `src/pytts/text/russian_numbers.py`:

```python
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
) -> str:
    value = _number_value(raw)
    try:
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
) -> str:
    return _words(raw, to="ordinal", case=case, gender=gender)


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
```

- [ ] **Step 4: Run focused tests and the recorded spike cases**

Run:

```bash
uv run pytest tests/text/test_russian_numbers.py -v
uv run python -c 'from pytts.text.russian_numbers import cardinal, ordinal; assert cardinal(56, case="g") == "пятидесяти шести"; assert ordinal(2026, case="p") == "две тысячи двадцать шестом"'
uv run ruff check src/pytts/text/russian_numbers.py tests/text/test_russian_numbers.py
```

Expected: every command succeeds without output other than passing pytest cases.

- [ ] **Step 5: Commit**

```bash
git add src/pytts/text/russian_numbers.py tests/text/test_russian_numbers.py
git commit -m "feat: format russian number forms"
```

---

### Task 5: Normalize structured numeric constructions

**Files:**
- Create: `src/pytts/text/numeric_normalizer.py`
- Create: `tests/text/test_numeric_normalizer.py`

**Interfaces:**
- Consumes: cleaned text after configured literal replacements and mixed-script repair.
- Consumes: Task 3 `spell_code_letters` and Task 4 number functions.
- Produces: `NumericNormalizer.normalize(text: str) -> str`.
- Guarantee: all matched structured values become Cyrillic words; generic digit replacement runs last.

- [ ] **Step 1: Write failing tests for dates, years, ordinals and compounds**

Create `tests/text/test_numeric_normalizer.py` with a helper and the first contract group:

```python
import pytest

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
        ("56-летний и 56-летнего", "пятидесятишестилетний и пятидесятишестилетнего"),
        ("90-долларовый и 90-долларового", "девяностодолларовый и девяностодолларового"),
        ("FP-5, F-16, С-300", "эф пи пять, эф шестнадцать, эс триста"),
    ],
)
def test_normalizes_dates_years_ordinals_compounds_and_codes(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected
```

- [ ] **Step 2: Run the first focused cases and confirm RED**

Run:

```bash
uv run pytest tests/text/test_numeric_normalizer.py -v
```

Expected: collection fails because `pytts.text.numeric_normalizer` does not exist.

- [ ] **Step 3: Implement the class, shared patterns and first ordered rules**

Create `src/pytts/text/numeric_normalizer.py` with these exact public/data contracts:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from pytts.errors import InputError
from pytts.text.latin import spell_code_letters
from pytts.text.russian_numbers import cardinal, decimal_words, noun_form, ordinal, parse_integer

_INTEGER = r"(?:\d{1,3}(?: \d{3})+|\d+)"
_DASH = r"[-–—]"
_MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}
_MONTH_NUMBERS = {name: number for number, name in _MONTHS.items()}

_NUMERIC_DATE = re.compile(r"(?<!\w)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\w)")
_TEXT_DATE = re.compile(
    rf"(?<!\w)(?P<day>\d{{1,2}})\s+(?P<month>{'|'.join(_MONTH_NUMBERS)})\s+"
    r"(?P<year>\d{4})\s+года(?!\w)",
    re.IGNORECASE,
)
_YEAR_CONTEXT = re.compile(
    r"(?<!\w)(?P<preposition>в|на|с|к)\s+(?P<year>\d{4})\s+(?P<noun>год|года|году)(?!\w)",
    re.IGNORECASE,
)
_CODE = re.compile(rf"(?<!\w)(?P<letters>[A-Z]{{1,5}}|[А-ЯЁ])-(?P<number>{_INTEGER})(?!\w)")
_COMPOUND_YEARS = re.compile(rf"(?<!\w)(?P<number>{_INTEGER})-(?P<suffix>летн[а-яё]*)(?!\w)", re.IGNORECASE)
_COMPOUND_DOLLARS = re.compile(rf"(?<!\w)(?P<number>{_INTEGER})-(?P<suffix>долларов[а-яё]*)(?!\w)", re.IGNORECASE)
_ORDINAL_SUFFIXES = {
    "й": ("n", "m"), "я": ("n", "f"), "е": ("n", "n"),
    "го": ("g", "m"), "му": ("d", "m"), "м": ("p", "m"),
    "ую": ("a", "f"), "ой": ("g", "f"),
}
_ORDINAL = re.compile(
    rf"(?<!\w)(?P<number>{_INTEGER})-(?P<suffix>{'|'.join(sorted(_ORDINAL_SUFFIXES, key=len, reverse=True))})(?!\w)",
    re.IGNORECASE,
)


class NumericNormalizer:
    def normalize(self, text: str) -> str:
        text = _TEXT_DATE.sub(self._text_date, text)
        text = _NUMERIC_DATE.sub(self._numeric_date, text)
        text = _YEAR_CONTEXT.sub(self._year_context, text)
        text = _CODE.sub(self._code, text)
        text = _COMPOUND_YEARS.sub(self._compound_years, text)
        text = _COMPOUND_DOLLARS.sub(self._compound_dollars, text)
        text = _ORDINAL.sub(self._ordinal, text)
        return text
```

Implement callbacks with the following exact formulas:

```python
def _spoken_date(day: int, month: int, year: int) -> str:
    try:
        date(year, month, day)
    except ValueError as error:
        raise InputError(f"Invalid calendar date: {day:02d}.{month:02d}.{year}") from error
    return f"{ordinal(day, case='g')} {_MONTHS[month]} {ordinal(year, case='g')} года"


def _year_context(self, match: re.Match[str]) -> str:
    preposition = match.group("preposition")
    noun = match.group("noun")
    case = {("в", "году"): "p", ("на", "году"): "p", ("с", "года"): "g", ("к", "году"): "d"}.get(
        (preposition.casefold(), noun.casefold()), "n"
    )
    return f"{preposition} {ordinal(match.group('year'), case=case)} {noun}"


def _code(self, match: re.Match[str]) -> str:
    return f"{spell_code_letters(match.group('letters'))} {cardinal(match.group('number'))}"


def _compound_years(self, match: re.Match[str]) -> str:
    prefix = cardinal(match.group("number"), case="g").replace(" ", "")
    return prefix + match.group("suffix")


def _compound_dollars(self, match: re.Match[str]) -> str:
    prefix = cardinal(match.group("number")).replace(" ", "")
    return prefix + match.group("suffix")
```

`_ordinal` must case-fold the suffix, read `(case, gender)` from `_ORDINAL_SUFFIXES`, and call `ordinal(number, case=case, gender=gender)`. `_text_date` uses `_MONTH_NUMBERS`; `_numeric_date` parses the three capture groups and calls `_spoken_date`.

- [ ] **Step 4: Run the first contract group and confirm GREEN**

Run:

```bash
uv run pytest tests/text/test_numeric_normalizer.py -v
```

Expected: every date/year/ordinal/compound/code case passes.

- [ ] **Step 5: Add failing tests for currency, percentages and ranges**

Append:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 ₽, 2 ₽, 5 ₽, 11 ₽, 21 ₽", "один рубль, два рубля, пять рублей, одиннадцать рублей, двадцать один рубль"),
        ("1 €, 2 EUR, 5 евро", "один евро, два евро, пять евро"),
        ("1 £, 2 GBP, 5 USD", "один фунт, два фунта, пять долларов"),
        ("12,50 ₽", "двенадцать рублей пятьдесят копеек"),
        ("15% и 21 %", "пятнадцать процентов и двадцать один процент"),
        ("от 5 до 7", "от пяти до семи"),
        ("3–5", "от трёх до пяти"),
        ("15–20%", "от пятнадцати до двадцати процентов"),
        ("$5–7", "от пяти до семи долларов"),
    ],
)
def test_normalizes_currency_percentages_and_ranges(source: str, expected: str) -> None:
    assert _normalize(source) == expected
```

Run:

```bash
uv run pytest tests/text/test_numeric_normalizer.py -v
```

Expected: the new currency/range cases fail because `normalize` does not yet run those patterns.

- [ ] **Step 6: Implement explicit currency and range tables before generic numbers**

Add:

```python
@dataclass(frozen=True, slots=True)
class CurrencyForms:
    major: tuple[str, str, str]
    minor: tuple[str, str, str]


_CURRENCIES = {
    "RUB": CurrencyForms(("рубль", "рубля", "рублей"), ("копейка", "копейки", "копеек")),
    "USD": CurrencyForms(("доллар", "доллара", "долларов"), ("цент", "цента", "центов")),
    "EUR": CurrencyForms(("евро", "евро", "евро"), ("цент", "цента", "центов")),
    "GBP": CurrencyForms(("фунт", "фунта", "фунтов"), ("пенс", "пенса", "пенсов")),
}
_CURRENCY_ALIASES = {
    "₽": "RUB", "руб.": "RUB", "rub": "RUB",
    "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP", "евро": "EUR",
}
```

Compile currency aliases with `re.escape`, sorted longest-first and `re.IGNORECASE`. Add patterns for prefix/suffix single amounts, prefix/suffix ranges, percentage ranges, single percentages, explicit `от X до Y`, and bare `X-Y`. Apply them after dates but before `_CODE`, compounds and generic numbers.

Use these exact pattern shapes:

```python
_AMOUNT = rf"{_INTEGER}(?:[.,]\d{{1,2}})?"
_CURRENCY_TOKEN = r"(?:руб\.|RUB|USD|EUR|GBP|евро|₽|\$|€|£)"
_CURRENCY_PREFIX_RANGE = re.compile(
    rf"(?<!\w)(?P<currency>[₽$€£])\s*(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})(?!\w)"
)
_CURRENCY_SUFFIX_RANGE = re.compile(
    rf"(?<!\w)(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})\s*(?P<currency>{_CURRENCY_TOKEN})(?!\w)",
    re.IGNORECASE,
)
_PERCENT_RANGE = re.compile(
    rf"(?<!\w)(?P<left>{_INTEGER})\s*{_DASH}\s*(?P<right>{_INTEGER})\s*%(?!\w)"
)
_CURRENCY_PREFIX = re.compile(
    rf"(?<!\w)(?P<currency>[₽$€£])\s*(?P<amount>{_AMOUNT})(?![\w.,])"
)
_CURRENCY_SUFFIX = re.compile(
    rf"(?<![\w.,])(?P<amount>{_AMOUNT})\s*(?P<currency>{_CURRENCY_TOKEN})(?!\w)",
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
```

Insert substitutions in this exact order after the date/year rules and before `_CODE`:

```python
text = _CURRENCY_PREFIX_RANGE.sub(self._currency_range, text)
text = _CURRENCY_SUFFIX_RANGE.sub(self._currency_range, text)
text = _PERCENT_RANGE.sub(self._percent_range, text)
text = _CURRENCY_PREFIX.sub(self._currency_amount, text)
text = _CURRENCY_SUFFIX.sub(self._currency_amount, text)
text = _PERCENT.sub(self._percent, text)
text = _EXPLICIT_RANGE.sub(self._explicit_range, text)
text = _BARE_RANGE.sub(self._bare_range, text)
```

The callbacks must use these exact rules:

```python
def _currency_amount_words(raw: str, code: str) -> str:
    normalized = raw.replace(" ", "").replace(",", ".")
    whole_text, separator, fraction_text = normalized.partition(".")
    whole = int(whole_text)
    forms = _CURRENCIES[code]
    pieces = [cardinal(whole), noun_form(whole, forms.major)]
    if separator:
        minor = int(fraction_text.ljust(2, "0"))
        pieces.extend((cardinal(minor), noun_form(minor, forms.minor)))
    return " ".join(pieces)


def _range_words(left: str, right: str, noun: str | None = None) -> str:
    result = f"от {cardinal(left, case='g')} до {cardinal(right, case='g')}"
    return f"{result} {noun}" if noun else result
```

Resolve `match.group("currency")` through `_CURRENCY_ALIASES[raw.casefold()]`; include symbol keys unchanged in the same map. `_currency_amount` calls `_currency_amount_words`. `_currency_range` calls `_range_words` with the third major form (`долларов`, `рублей`, `евро`, `фунтов`). `_percent_range` appends `процентов`. `_percent` selects `процент/процента/процентов` from the parsed integer value. `_explicit_range` and `_bare_range` call `_range_words` without a noun. Applying explicit `от X до Y` before bare ranges prevents a second rewrite.

- [ ] **Step 7: Add and implement generic degrees, decimals and integers**

Append tests:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("20 °C, 1° и 2°", "двадцать градусов Цельсия, один градус и два градуса"),
        ("3,14; 3, 14; 3.14", "три целых четырнадцать сотых; три, четырнадцать; три целых четырнадцать сотых"),
        ("1 500 и 12 345 678", "одна тысяча пятьсот и двенадцать миллионов триста сорок пять тысяч шестьсот семьдесят восемь"),
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
```

Add final ordered patterns:

```python
_DEGREES = re.compile(rf"(?<!\w)(?P<number>{_INTEGER})\s*°\s*(?P<celsius>[CС])?(?!\w)")
_DECIMAL = re.compile(rf"(?<![\w.,])(?P<integer>{_INTEGER})(?P<separator>[.,])(?P<fraction>\d{{1,2}})(?![\w.,])")
_REMAINING_INTEGER = re.compile(_INTEGER)
```

`_DEGREES` selects `градус/градуса/градусов` with `noun_form` and adds ` Цельсия` when `celsius` is present. `_DECIMAL` calls `decimal_words`. `_REMAINING_INTEGER` calls `cardinal`. These three substitutions are the final operations in `normalize`.

The invalid grouping example `12 34` deliberately becomes two independent numbers because `_INTEGER` cannot consume it as one grouped value.

- [ ] **Step 8: Run all numeric tests and Ruff**

Run:

```bash
uv run pytest tests/text/test_russian_numbers.py tests/text/test_numeric_normalizer.py -v
uv run ruff check src/pytts/text/russian_numbers.py src/pytts/text/numeric_normalizer.py tests/text/test_numeric_normalizer.py
```

Expected: every numeric contract passes and Ruff prints no diagnostics.

- [ ] **Step 9: Commit**

```bash
git add src/pytts/text/numeric_normalizer.py tests/text/test_numeric_normalizer.py
git commit -m "feat: normalize russian numeric text"
```

---

### Task 6: Compose article pronunciation and fail closed

**Files:**
- Create: `src/pytts/text/pronunciation.py`
- Create: `tests/text/test_pronunciation.py`

**Interfaces:**
- Consumes: `Article`, `Mapping[str, str]`, Task 3 `TransliterationOverrides`/`normalize_latin`, and Task 5 `NumericNormalizer`.
- Produces: `PronunciationNormalizer(transliterations: Mapping[str, str])`.
- Produces: `PronunciationNormalizer.normalize_article(article: Article) -> Article`.
- Guarantee: the returned article contains no digits, non-Cyrillic letters or unhandled semantic Unicode symbols.

- [ ] **Step 1: Write failing article-level, symbol and guard tests**

Create `tests/text/test_pronunciation.py`:

```python
from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
from pytts.text.pronunciation import PronunciationNormalizer


def _article(text: str) -> Article:
    return Article(Path("article.md"), (TextBlock(BlockKind.PARAGRAPH, text),))


def _normalize(text: str, mapping: dict[str, str] | None = None) -> str:
    normalizer = PronunciationNormalizer(mapping or {})
    return normalizer.normalize_article(_article(text)).blocks[0].text


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


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("Коэффициент α равен единице.", "α"),
        ("Температура 🌡 высокая.", "🌡"),
        ("Цена в неизвестной валюте ₿.", "₿"),
    ],
)
def test_rejects_unhandled_letters_and_symbols(source: str, fragment: str) -> None:
    with pytest.raises(InputError, match=repr(fragment)):
        _normalize(source)


def test_override_can_make_other_alphabet_speakable() -> None:
    assert _normalize("Коэффициент α.", {"α": "альфа"}) == "Коэффициент альфа."


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
    twice = normalizer.normalize_article(once)
    assert [block.kind for block in once.blocks] == [BlockKind.HEADING, BlockKind.PARAGRAPH]
    assert twice == once
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_pronunciation.py -v
```

Expected: collection fails because `pytts.text.pronunciation` does not exist.

- [ ] **Step 3: Implement symbol expansion and the final guard**

Create `src/pytts/text/pronunciation.py` with these helpers:

```python
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from pytts.domain import Article, TextBlock
from pytts.errors import InputError
from pytts.text.latin import TransliterationOverrides, normalize_latin
from pytts.text.numeric_normalizer import NumericNormalizer

_SPACE = re.compile(r"\s+")
_SYMBOL_WORDS = {
    "×": "умножить на",
    "÷": "разделить на",
    "±": "плюс-минус",
    "+": "плюс",
    "=": "равно",
    "‰": "промилле",
    "§": "параграф",
    "№": "номер",
    "&": "и",
    "@": "собака",
}
_EXPLICIT_FORBIDDEN = frozenset("%‰№°§&@#")


def _normalize_symbols(text: str) -> str:
    text = re.sub(r"#(?=\s*\d)", " номер ", text)
    text = re.sub(r"#(?=\s*[A-Za-zА-Яа-яЁё])", " хештег ", text)
    for symbol, words in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f" {words} ")
    return _SPACE.sub(" ", text).strip()


def _is_cyrillic_letter(character: str) -> bool:
    return character.isalpha() and "CYRILLIC" in unicodedata.name(character, "")


def _validate_speakable(text: str) -> None:
    for index, character in enumerate(text):
        category = unicodedata.category(character)
        invalid = (
            character.isdigit()
            or character.isalpha() and not _is_cyrillic_letter(character)
            or character in _EXPLICIT_FORBIDDEN
            or category in {"Sc", "Sm", "Sk", "So"}
        )
        if invalid:
            start = max(0, index - 20)
            end = min(len(text), index + 21)
            raise InputError(
                f"Unsupported character {character!r} remains after pronunciation "
                f"normalization near {text[start:end]!r}"
            )
```

- [ ] **Step 4: Implement the article orchestrator in the approved order**

Add:

```python
class PronunciationNormalizer:
    def __init__(self, transliterations: Mapping[str, str]) -> None:
        self._overrides = TransliterationOverrides(transliterations)
        self._numbers = NumericNormalizer()

    def normalize_article(self, article: Article) -> Article:
        blocks: list[TextBlock] = []
        for block in article.blocks:
            text = self._overrides.apply(block.text)
            text = self._numbers.normalize(text)
            text = _normalize_symbols(text)
            text = normalize_latin(text)
            _validate_speakable(text)
            blocks.append(TextBlock(kind=block.kind, text=text))
        return Article(source=article.source, blocks=tuple(blocks))
```

Do not catch `InputError`, `KeyboardInterrupt` or `SystemExit` here.

- [ ] **Step 5: Run all text tests and Ruff**

Run:

```bash
uv run pytest tests/text -v
uv run ruff check src/pytts/text tests/text
```

Expected: all text tests pass. Existing chunker tests still prove that chunking itself preserves whatever clean text it is given.

- [ ] **Step 6: Commit**

```bash
git add src/pytts/text/pronunciation.py tests/text/test_pronunciation.py
git commit -m "feat: compose pronunciation normalization"
```

---

### Task 7: Integrate normalization before chunking

**Files:**
- Modify: `src/pytts/pipeline.py:12-19,148-163`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_e2e.py:66-113`

**Interfaces:**
- Consumes: `AppConfig.transliterations` and `PronunciationNormalizer.normalize_article`.
- Preserves: `ConversionPipeline.convert(request: ConversionRequest) -> ConversionResult` and all CLI exit-code mappings.
- Guarantee: normalization failure occurs before `_runtime()` and before `writer_factory`.

- [ ] **Step 1: Write failing pipeline-order and early-failure tests**

Update `_pipeline` in `tests/test_pipeline.py` so its root config includes an exact override:

```python
(tmp_path / "pytts.yaml").write_text(
    'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n'
    'transliterations:\n  "Brent": "Брент"\n',
    encoding="utf-8",
)
```

Add:

```python
def test_normalizes_after_abbreviations_and_before_chunking(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    pipeline = _pipeline(
        tmp_path,
        runtime,
        [],
        [],
        reader=FakeInputReader("Ув. автор: в 2026 году Brent вырос на 15%."),
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    combined = " ".join(runtime.chunks)
    assert "Уважаемый автор" in combined
    assert "две тысячи двадцать шестом году" in combined
    assert "Брент вырос на пятнадцать процентов" in combined
    assert not any(character.isdigit() for character in combined)


def test_normalization_failure_precedes_model_and_writer(tmp_path: Path) -> None:
    runtime_calls: list[object] = []
    writers: list[FakeWriter] = []
    pipeline = _pipeline(
        tmp_path,
        FakeRuntime(),
        [],
        writers,
        reader=FakeInputReader("Значение α."),
        runtime_calls=runtime_calls,
    )
    source = tmp_path / "article.md"
    source.write_text("source", encoding="utf-8")

    with pytest.raises(InputError, match="α"):
        pipeline.convert(ConversionRequest(source))

    assert not runtime_calls
    assert not writers
```

Change the `tests/test_e2e.py::test_markdown_to_audio_contract` assertion to:

```python
assert "две тысячи двадцать шестом году" in combined
assert "двадцать пять процентов" in combined
assert "одна тысяча пятьсот рублей" in combined
assert not any(character.isdigit() for character in combined)
assert all(symbol not in combined for symbol in "%₽$€£")
```

- [ ] **Step 2: Run pipeline/e2e tests and confirm RED**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_e2e.py -v
```

Expected: new assertions fail because raw text is still passed directly to `chunk_article`.

- [ ] **Step 3: Insert the normalizer before chunking**

Modify `src/pytts/pipeline.py` imports:

```python
from pytts.text.pronunciation import PronunciationNormalizer
```

Replace the current abbreviation/chunk lines with:

```python
expanded = AbbreviationExpander(config.abbreviations).expand_article(cleaned.article)
normalized = PronunciationNormalizer(config.transliterations).normalize_article(expanded)
chunks = chunk_article(normalized, request.rate, self._spec.max_text_chars)
```

Change the config progress message to `Loading text normalization config`. Do not add a new progress stage; the established stage ordering remains stable.

- [ ] **Step 4: Verify post-expansion chunk limits and SSML speed**

Add a `max_text_chars: int = 800` keyword to the `_pipeline` test helper and construct its spec with:

```python
spec = ModelSpec(
    "v5_5_ru",
    "https://example.test",
    "a" * 64,
    "xenia",
    48000,
    max_text_chars,
)
```

Then add a pipeline test that passes `max_text_chars=64` and a numeric-heavy source. Assert:

```python
plain_payloads = [
    chunk.split(">", 2)[2].split("<", 1)[0]
    for chunk in runtime.chunks
]
assert all(len(payload) <= 64 for payload in plain_payloads)
assert all('<prosody rate="medium">' in chunk for chunk in runtime.chunks)
assert len(runtime.chunks) > 1
```

Use input containing at least twelve grouped four-digit values so normalization expands beyond one chunk. Do not inspect escaped SSML by regex in production code; this extraction is test-only.

- [ ] **Step 5: Run regression tests and Ruff**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_e2e.py tests/test_cli.py tests/text -v
uv run ruff check src/pytts/pipeline.py tests/test_pipeline.py tests/test_e2e.py
```

Expected: all tests pass, progress stage order remains unchanged, and errors still map to exit code 3 through existing `InputError` handling.

- [ ] **Step 6: Commit**

```bash
git add src/pytts/pipeline.py tests/test_pipeline.py tests/test_e2e.py
git commit -m "feat: normalize text before synthesis"
```

---

### Task 8: Document and verify the real article workflow

**Files:**
- Create: `tests/fixtures/pronunciation.md`
- Modify: `tests/test_silero_integration.py`
- Modify: `README.md:47-85`
- Modify: `docs/superpowers/verification/2026-07-19-pytts-mvp-acceptance.md`

**Interfaces:**
- Consumes: public `pytts` CLI and cached Silero `v5_5_ru`.
- Produces: a reusable representative normalization fixture and recorded acceptance evidence.
- Acceptance output: ignored local MP3 artifacts; do not commit audio or the supplied PDF.

- [ ] **Step 1: Add the raw pronunciation fixture**

Create `tests/fixtures/pronunciation.md`:

```markdown
# Проверка произношения

19 июля 2026 года показатель вырос на 15%, сумма составила 1 500 ₽, а диапазон оказался от 5 до 7 единиц.

Цена Brent преодолела 90-долларовый барьер. На 142-й день 56-летний автор упомянул F-16, FPV и С-300.

Диапазоны: 15–20% и $5–7. Символы: № 5, 20 °C, 2 × 3 = 6, ±5, 7‰, § 2.
```

- [ ] **Step 2: Extend the opt-in real-model test**

Parametrize `test_real_cli_synthesizes_smoke_fixture_to_valid_mp3` over `smoke.md` and `pronunciation.md`, with distinct output names:

```python
@pytest.mark.parametrize("fixture_name", ["smoke.md", "pronunciation.md"])
def test_real_cli_synthesizes_fixture_to_valid_mp3(
    tmp_path: Path, fixture_name: str
) -> None:
    output = tmp_path / f"{Path(fixture_name).stem}.mp3"
    source = Path(__file__).parent / "fixtures" / fixture_name
    result = subprocess.run(
        [sys.executable, "-m", "pytts", str(source), "--output", str(output)],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    info = MP3(output).info
    assert info.sample_rate == 48000
    assert 90000 <= info.bitrate <= 100000
    assert info.length > 0.5
```

- [ ] **Step 3: Update README behavior and limitations**

Rename `## Abbreviations` to `## Pronunciation configuration`, show both YAML mappings, and state:

```markdown
`abbreviations` expands literal Russian abbreviations. `transliterations` supplies exact spoken
forms for foreign names, brands, non-Cyrillic letters, or symbols before the general normalizer.
Both mappings are case-insensitive, longest-first, token-bounded, and one-pass. An explicit
`--config` replaces the root config rather than merging with it.

Before Silero, pytts converts common dates, years, integers, decimals, currencies, percentages,
ranges, letter-number codes, and documented semantic symbols to Russian words. Mixed-script PDF
lookalikes are repaired contextually. Unsupported letters or symbols stop conversion before model
loading so they cannot disappear silently from the audio.
```

Replace the obsolete Latin/numeric limitation bullets with these exact limitations:

```markdown
- General foreign-word transliteration is approximate; use `transliterations` for exact names.
- Uppercase Latin groups of 1–5 letters are spelled by letter names, so `NASA` is read as
  `эн эй эс эй`; override it in `transliterations` when word-like pronunciation is preferred.
- Greek and other non-Cyrillic letters, emoji, and unknown semantic symbols fail closed unless an
  exact `transliterations` replacement makes them speakable.
- Context rules cover the documented dates, years, currency, ranges and two compound-adjective
  families; ambiguous Russian syntax can still produce a non-ideal case.
```

- [ ] **Step 4: Run the complete network-free quality gate**

Run:

```bash
uv run pytest -m 'not silero'
uv run ruff check .
uv lock --check
git diff --check
```

Expected: the complete non-Silero suite passes, Ruff and lock checks succeed, and `git diff --check` prints nothing.

- [ ] **Step 5: Run real Silero tests with the verified cache**

Run with network disabled by invalid proxies:

```bash
PYTTS_RUN_SILERO=1 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 uv run pytest -m silero -v
```

Expected: both fixture conversions and the direct runtime test pass without a download.

- [ ] **Step 6: Convert the supplied Safari PDF**

Run:

```bash
uv run pytts '/Users/azdrachek/Downloads/Киев не знает, что делать, Трамп не знает будущего, Британия не знает своё новое начальство | Средство немассовой информации СНМИ | Sponsr.pdf' --output artifacts/acceptance-browser-print.mp3 --force
uv run mutagen-inspect artifacts/acceptance-browser-print.mp3
```

Expected: CLI exits 0; there is no `.mp3.part`; metadata reports mono, 48 kHz and approximately 96 kbps. The output path is ignored by git.

- [ ] **Step 7: Generate and listen to the compact acceptance fixture**

Run:

```bash
uv run pytts tests/fixtures/pronunciation.md --output artifacts/pronunciation.mp3 --force
afplay artifacts/pronunciation.mp3
```

Expected: the listener can identify the date, year, percentage, rubles, both ranges, `Brent`, `F-16`, `FPV`, `С-300`, `142-й`, `56-летний`, degrees and arithmetic symbols. Record the user's exact observation; do not infer listening quality from metadata.

- [ ] **Step 8: Record acceptance evidence**

Append a `## Pronunciation normalization follow-up (2026-07-20)` subsection to
`docs/superpowers/verification/2026-07-19-pytts-mvp-acceptance.md`. It must contain six factual
bullets copied from Steps 4–7: the exact network-free pytest summary; the exact real-Silero pytest
summary and confirmation that invalid proxies caused no download; Safari MP3 duration, channels,
sample rate, bitrate and byte size; the complete post-normalization control text; the user's exact
listening words in quotation marks; and every residual mispronounced token. If the user reports no
residual token, write `Known residual pronunciation issues: none observed.` Do not estimate or infer
any of these values.

- [ ] **Step 9: Run final verification and inspect scope**

Run:

```bash
uv run pytest -m 'not silero'
uv run ruff check .
uv lock --check
git diff --check
git status --short
```

Expected: all checks pass; only intended source, tests, docs and config changes appear. `.DS_Store` remains untracked and unstaged; MP3/model artifacts do not appear.

- [ ] **Step 10: Commit**

```bash
git add README.md tests/fixtures/pronunciation.md tests/test_silero_integration.py docs/superpowers/verification/2026-07-19-pytts-mvp-acceptance.md
git commit -m "docs: verify pronunciation normalization"
```

---

## Completion Gate

Before claiming the feature complete:

1. Run `superpowers:verification-before-completion` against the current HEAD.
2. Request a full code review from the merge base of `feature/pytts-mvp` through the current HEAD.
3. Fix every Critical or Important finding with focused tests, then re-run the full review.
4. Confirm `git status --short` contains only the preserved untracked `.DS_Store`.
5. Use `superpowers:finishing-a-development-branch` and let the user choose merge, PR, branch retention or cleanup.
