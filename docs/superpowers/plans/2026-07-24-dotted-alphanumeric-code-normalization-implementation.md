# Dotted Alphanumeric Code Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize uppercase dotted codes such as `V.629S` to the neutral spoken form `ви шестьсот двадцать девять эс` before unsupported digits reach the warning guard.

**Architecture:** Add one bounded regular expression to `NumericNormalizer` immediately before the existing `_CODE` substitution. Reuse a single `_code` callback for hyphenated, joined, and dotted code shapes, with an optional suffix group; keep downstream Latin normalization and warn-and-strip behavior unchanged.

**Tech Stack:** Python 3.12, `re`, existing `spell_code_letters` and Russian `cardinal` helpers, pytest, Ruff, uv.

## Global Constraints

- Work only on `codex/dotted-code-normalization`; do not implement on `master`.
- Match only whole tokens shaped as `(?<!\w)[A-Z]{1,5}\.\d+[A-Z]{1,5}(?!\w)`.
- Read `V.629S` as `ви шестьсот двадцать девять эс`; do not interpret `V.` as `рейс` and do not speak the dot.
- Preserve the existing priority of user `transliterations` over built-in normalization.
- Keep `V.629`, `629S`, lowercase or mixed-case forms, embedded identifiers, and groups longer than five letters outside the new rule.
- Groups longer than five letters must continue to leave their unsupported digit visible to the existing warn-and-strip guard.
- Do not change readers, chunking, Silero, configuration schema, error classes, exit codes, dependencies, `pytts.yaml`, or `uv.lock`.
- Do not weaken or suppress unsupported-character warnings.
- Do not commit the user-provided article from `/Users/azdrachek/Downloads`.
- Treat the supplied-article acceptance as a local, non-CI check; absence of the external file is not a product regression.
- Preserve and do not stage the unrelated untracked `.DS_Store`.

## File Map

- Modify `src/pytts/text/numeric_normalizer.py`: recognize dotted codes and consolidate code callbacks.
- Modify `tests/text/test_numeric_normalizer.py`: cover the grammar, boundaries, compatibility, and idempotence.
- Modify `tests/text/test_pronunciation.py`: verify ordering relative to Latin normalization and the warning guard.
- Modify `tests/test_pipeline.py`: verify the synthesized chunk and progress-event contract.
- Modify `README.md`: document the exact dotted-code behavior and scope.
- Do not create or modify runtime dependencies, fixtures, configuration, or reader files.

---

### Task 1: Normalize dotted codes across the pronunciation pipeline

**Files:**
- Modify: `src/pytts/text/numeric_normalizer.py:76-81`
- Modify: `src/pytts/text/numeric_normalizer.py:250-274`
- Modify: `src/pytts/text/numeric_normalizer.py:352-362`
- Test: `tests/text/test_numeric_normalizer.py:109-136`
- Test: `tests/text/test_pronunciation.py:19-24`
- Test: `tests/test_pipeline.py:155-174`

**Interfaces:**
- Consumes: `spell_code_letters(token: str) -> str` from `pytts.text.latin`.
- Consumes: `cardinal(raw: str | int, *, case: RussianCase = "n", gender: RussianGender = "m") -> str` from `pytts.text.russian_numbers`.
- Produces: `NumericNormalizer.normalize("V.629S") == "ви шестьсот двадцать девять эс"`.
- Produces: the existing private callback `NumericNormalizer._code(match: re.Match[str]) -> str`, extended to append an optional named `suffix` group.
- Preserves: `PronunciationNormalizer.normalize_article(article) -> NormalizationResult` and all public pipeline interfaces unchanged.

- [ ] **Step 1: Add failing unit tests for the dotted-code grammar**

Insert the following tests after
`test_joined_code_normalization_is_idempotent` in
`tests/text/test_numeric_normalizer.py`:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("A.1B", "эй один би"),
        ("V.629S", "ви шестьсот двадцать девять эс"),
        ("AB.12CD", "эй би двенадцать си ди"),
        (
            "ABCDE.12ABCDE",
            "эй би си ди и двенадцать эй би си ди и",
        ),
    ],
)
def test_normalizes_dotted_uppercase_letter_number_codes(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("preV.629S", "preV.629S"),
        ("V.629Spost", "V.629Spost"),
        ("_V.629S", "_V.629S"),
        ("V.629S_", "V.629S_"),
        ("ABCDEF.1S", "ABCDEF.1S"),
        ("V.1ABCDEF", "V.1ABCDEF"),
        ("629S", "629S"),
        ("v.629S", "v.629S"),
        ("V.629s", "V.629s"),
        ("V.629", "V.шестьсот двадцать девять"),
    ],
)
def test_dotted_codes_require_supported_groups_and_whole_word_boundaries(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


def test_dotted_code_normalization_is_idempotent() -> None:
    once = _normalize("V.629S и AB.12CD")

    assert _normalize(once) == once
```

- [ ] **Step 2: Add failing pronunciation-layer tests**

Insert the following tests after
`test_applies_override_then_numbers_then_remaining_latin` in
`tests/text/test_pronunciation.py`:

```python
def test_normalizes_dotted_code_before_latin_and_warning_guard() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("Судно VSL MEDKON MIRA V.629S будет перенаправлено.")
    )

    spoken = result.article.blocks[0].text
    assert "ви шестьсот двадцать девять эс" in spoken
    assert not any(character.isdigit() for character in spoken)
    assert result.warning is None


def test_override_can_replace_dotted_code_before_builtin_rule() -> None:
    assert _normalize(
        "Судно V.629S.",
        {"V.629S": "рейс южного направления"},
    ) == "Судно рейс южного направления."


@pytest.mark.parametrize("source", ["ABCDEF.1S", "V.1ABCDEF"])
def test_overlong_dotted_code_groups_remain_visible_to_warning_guard(
    source: str,
) -> None:
    result = PronunciationNormalizer({}).normalize_article(_article(source))

    assert not any(
        character.isdigit()
        for block in result.article.blocks
        for character in block.text
    )
    assert result.warning is not None
    assert result.warning.count("'1' near") == 1
```

- [ ] **Step 3: Add a failing pipeline regression test**

Insert the following test after
`test_normalizes_after_abbreviations_and_before_chunking` in
`tests/test_pipeline.py`:

```python
def test_dotted_code_reaches_synthesis_without_unsupported_warning(
    tmp_path: Path,
) -> None:
    events: list[ProgressEvent] = []
    runtime = FakeRuntime()
    pipeline = _pipeline(
        tmp_path,
        runtime,
        events,
        [],
        reader=FakeInputReader(
            "Судно VSL MEDKON MIRA V.629S будет перенаправлено."
        ),
    )
    source = tmp_path / "article.md"
    source.write_text("ignored by fake", encoding="utf-8")

    pipeline.convert(ConversionRequest(source))

    combined = " ".join(runtime.chunks)
    assert "ви шестьсот двадцать девять эс" in combined
    assert not any(character.isdigit() for character in combined)
    assert not any(
        event.warning
        and event.message is not None
        and event.message.startswith("Removed ")
        for event in events
    )
```

- [ ] **Step 4: Run the positive regression tests and verify RED**

Run:

```bash
uv run pytest \
  tests/text/test_numeric_normalizer.py::test_normalizes_dotted_uppercase_letter_number_codes \
  tests/text/test_pronunciation.py::test_normalizes_dotted_code_before_latin_and_warning_guard \
  tests/test_pipeline.py::test_dotted_code_reaches_synthesis_without_unsupported_warning \
  -v
```

Expected: FAIL. The numeric tests retain raw `V.629S`-shaped tokens, while the
pronunciation and pipeline tests observe `ви.эс` and an unsupported-character
warning because `629` is removed.

- [ ] **Step 5: Implement the bounded pattern and shared callback**

In `src/pytts/text/numeric_normalizer.py`, insert
`_DOTTED_ALNUM_CODE` immediately before `_CODE`. Leave the existing `_CODE`
definition unchanged; add only this new definition above it:

```python
_DOTTED_ALNUM_CODE = re.compile(
    r"(?<!\w)(?P<letters>[A-Z]{1,5})\.(?P<number>\d+)"
    r"(?P<suffix>[A-Z]{1,5})(?!\w)"
)
```

In `NumericNormalizer.normalize`, place the new substitution immediately
before the existing `_CODE` substitution, and route `_ALNUM_CODE` through the
same callback:

```python
        text = _EXPLICIT_RANGE.sub(self._explicit_range, text)
        text = _BARE_RANGE.sub(self._bare_range, text)
        text = _DOTTED_ALNUM_CODE.sub(self._code, text)
        text = _CODE.sub(self._code, text)
        text = _ALNUM_CODE.sub(self._code, text)
        text = _COMPOUND_YEARS.sub(self._compound_years, text)
```

Replace `_code` and remove `_alnum_code`:

```python
    def _code(self, match: re.Match[str]) -> str:
        pieces = [
            spell_code_letters(match.group("letters")),
            cardinal(match.group("number")),
        ]
        suffix = match.groupdict().get("suffix")
        if suffix is not None:
            pieces.append(spell_code_letters(suffix))
        return " ".join(pieces)
```

Do not add fallback matching, case-insensitive flags, additional separators, or
warning suppression.

- [ ] **Step 6: Run every new dotted-code test and verify GREEN**

Run:

```bash
uv run pytest \
  tests/text/test_numeric_normalizer.py \
  tests/text/test_pronunciation.py \
  tests/test_pipeline.py \
  -k "dotted or overlong" \
  -v
```

Expected: PASS. Supported dotted codes have no digits or unsupported warning;
embedded, lowercase, overlong, and otherwise unsupported forms retain the
documented existing behavior.

- [ ] **Step 7: Run the complete affected test files and static check**

Run:

```bash
uv run pytest \
  tests/text/test_numeric_normalizer.py \
  tests/text/test_pronunciation.py \
  tests/test_pipeline.py \
  -v
uv run ruff check \
  src/pytts/text/numeric_normalizer.py \
  tests/text/test_numeric_normalizer.py \
  tests/text/test_pronunciation.py \
  tests/test_pipeline.py
git diff --check
```

Expected: all three test modules pass with zero failures; Ruff and
`git diff --check` report no issues. The existing `F-16`, `F16`, `X5`, and
`3.14` tests remain green.

- [ ] **Step 8: Commit the implementation and regression tests**

Run:

```bash
git add \
  src/pytts/text/numeric_normalizer.py \
  tests/text/test_numeric_normalizer.py \
  tests/text/test_pronunciation.py \
  tests/test_pipeline.py
git commit -m "fix: normalize dotted alphanumeric codes"
```

Expected: one commit containing only the normalizer and its three test files.
Do not stage `.DS_Store`.

---

### Task 2: Document and accept the behavior

**Files:**
- Modify: `README.md:99-108`
- Reference: `docs/superpowers/specs/2026-07-24-dotted-alphanumeric-code-normalization-design.md`
- Reference only, do not commit: `/Users/azdrachek/Downloads/Maersk вышел из чата, Йемен зашёл в чат, Израиль по-прежнему админ | Средство немассовой информации СНМИ.md`

**Interfaces:**
- Consumes: the `NumericNormalizer` behavior delivered by Task 1.
- Consumes: `InputReader`, `clean_article`, `AbbreviationExpander`, `load_config`, and `PronunciationNormalizer` for text-only acceptance.
- Produces: a reader-facing README contract for `[A-Z]{1,5}\.\d+[A-Z]{1,5}`.
- Produces: evidence that the supplied article retains `629`, emits no corresponding warning, and contains no digits after normalization.

- [ ] **Step 1: Document the exact dotted-code contract**

Replace the README paragraph beginning `Before Silero, pytts removes` with the
following text:

```markdown
Before Silero, pytts removes scheme/`www` URLs and bare URLs that contain a path, query, or
fragment. Ambiguous host-only tokens such as `main.py` and `example.com` are preserved rather than
silently discarded. After exact `transliterations`, any remaining emoji are silently removed as
decoration before numeric normalization. It then converts common dates, years, decades (`1990-х`),
Roman-numeral centuries
(`XX века`), integers, decimals, currencies, percentages, ranges, hyphenated, joined, and dotted
uppercase letter-number codes, and documented semantic symbols to Russian words. Dotted codes
shaped like `V.629S` are read by letter names and a cardinal number:
`ви шестьсот двадцать девять эс`; the dot acts only as a structural separator. `~` and `≈` are read
as `примерно`; `->` and `→` become a colon pause. One or two stars at a text-block boundary are
treated as formatting. Mixed-script PDF lookalikes are repaired contextually. A standalone
lowercase `x` is read as `икс`, while residual intraword Latin `x` is mapped mechanically to `кс`.
```

Keep the existing known-limitations list unchanged: the new paragraph already
states the exact supported shape, and exact exceptions remain configurable via
`transliterations`.

- [ ] **Step 2: Run local-only text acceptance on the supplied article**

This is a manual, non-CI check because the source file intentionally lives
outside the repository. Run it from the repository root when the supplied file
is present. If the file is absent in another environment, record this step as
skipped rather than treating the absence as a product regression:

```bash
uv run python - <<'PY'
from pathlib import Path

from pytts.config import load_config
from pytts.readers.input import InputReader
from pytts.text.abbreviations import AbbreviationExpander
from pytts.text.cleaner import clean_article
from pytts.text.pronunciation import PronunciationNormalizer

source = Path(
    "/Users/azdrachek/Downloads/"
    "Maersk вышел из чата, Йемен зашёл в чат, Израиль по-прежнему админ | "
    "Средство немассовой информации СНМИ.md"
)
config = load_config(None, Path.cwd())
article = InputReader().read(source)
cleaned = clean_article(article).article
expanded = AbbreviationExpander(config.abbreviations).expand_article(cleaned)
result = PronunciationNormalizer(config.transliterations).normalize_article(expanded)
spoken = " ".join(block.text for block in result.article.blocks)

assert "ви шестьсот двадцать девять эс" in spoken
assert not any(character.isdigit() for character in spoken)
warning = result.warning or ""
assert all(f"'{digit}' near" not in warning for digit in "629"), warning
print("article acceptance passed")
PY
```

Expected: exit code 0 and `article acceptance passed`. The source article
remains outside the repository and unchanged. An unrelated unsupported glyph
may still produce a warning without failing this acceptance step; warnings for
removed `6`, `2`, or `9` fail it.

- [ ] **Step 3: Run the full non-Silero quality gate**

Run:

```bash
uv run pytest -m "not silero"
uv run ruff check .
uv lock --check
git diff --check
```

Expected: all non-Silero tests pass; Ruff reports no issues; the lock file is
current; `git diff --check` reports no whitespace errors. No real model download
or audio synthesis is required for this fix.

- [ ] **Step 4: Commit the README update**

Run:

```bash
git add README.md
git commit -m "docs: document dotted code pronunciation"
```

Expected: one documentation-only commit. Do not stage the external article or
`.DS_Store`.

- [ ] **Step 5: Verify final branch state**

Run:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

Expected: current branch is `codex/dotted-code-normalization`; the implementation
and README commits appear above the existing design and plan commits; the only
permitted unrelated working-tree entry is the pre-existing untracked
`.DS_Store`.
