# Article Artifact Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove or pronounce the web-article artifacts found in the supplied PDF and Markdown consistently, leaving warnings only for genuinely ambiguous content such as the three censoring stars in `***бать`.

**Architecture:** Keep the existing shared text pipeline and place each rule in its current responsibility: URL recognition in `cleaner.py`, joined uppercase letter-number codes in `numeric_normalizer.py`, residual Latin `x` repair in `latin.py`, and speech symbols, boundary stars, and emoji in `pronunciation.py`. Preserve user `transliterations` as the first pronunciation override and keep the existing warn-and-strip guard as the final fallback.

**Tech Stack:** Python 3.12, `re`, `unicodedata`, `cyrtranslit`, `emoji>=2,<3`, markdown-it-py, PyMuPDF, Silero `v5_5_ru`, pytest, Ruff, uv.

## Global Constraints

- Runtime remains Python `>=3.12,<3.13` on local Apple Silicon macOS CPU.
- PDF and Markdown use the same post-reader normalization; do not add reader-specific branches.
- Remove scheme and `www` URLs, but remove a bare ASCII hostname only when it has a path, query, or fragment.
- Preserve ambiguous host-only tokens such as `main.py`, `README.md`, `config.yaml`, `index.html`, and `example.com`.
- All URL branches preserve trailing `.`, `,`, `;`, `:`, `!`, `?`, `…`, `)`, `]`, `}` and closing quotes consistently.
- User `transliterations` run before numeric, emoji, symbol, Latin, and final guard rules.
- Emoji left after user overrides are deleted silently as decoration; the `emoji` package must not perform network access at runtime.
- Remove only one or two stars at a block boundary. Three or more stars and interior mathematical stars continue through the warning guard.
- Joined codes match exactly `(?<!\w)[A-Z]{1,5}\d+(?!\w)` and run before `_REMAINING_INTEGER`.
- The final guard remains warn-and-strip for unsupported semantic content; do not add new exit codes or error classes.
- Do not change ambiguous decade suffixes `-е`/`-м`, decimals with three or more fractional digits, Silero voices, speed, chunking, or MP3 encoding.
- Do not commit the two supplied article files from `/Users/azdrachek/Downloads`.
- Preserve the user-owned untracked `.DS_Store`; stage only files named by each task.
- Follow RED/GREEN TDD for every production behavior and retain the current non-Silero baseline of 364 passing tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pytts/text/cleaner.py` | Recognize all removable URL forms with shared trailing-punctuation behavior. |
| `src/pytts/text/numeric_normalizer.py` | Expand joined uppercase letter-number codes before remaining integers. |
| `src/pytts/text/latin.py` | Repair residual ASCII `x`/`X` left by Russian `cyrtranslit`. |
| `src/pytts/text/pronunciation.py` | Apply emoji removal, boundary-star cleanup, arrows, approximation markers, symbols, Latin normalization, and the final guard in the specified order. |
| `tests/text/test_cleaner.py` | URL policy, punctuation, filename preservation, and false-positive contracts. |
| `tests/text/test_numeric_normalizer.py` | Joined-code matching, ordering, boundaries, regressions, and idempotence. |
| `tests/text/test_latin.py` | Residual `x` pronunciation, uppercase exceptions, approximate mixed-case behavior, and idempotence. |
| `tests/text/test_pronunciation.py` | Symbol, star, emoji, override-priority, warning, and block-removal contracts. |
| `tests/fixtures/article_artifacts.md` | Small original fixture reproducing the warning classes without copying the supplied article. |
| `tests/test_pipeline.py` | Whole-pipeline ordering and fixture-specific three-warning contract. |
| `tests/test_silero_integration.py` | Opt-in real-model MP3 synthesis for the artifact fixture. |
| `pyproject.toml`, `uv.lock` | Pinned offline-runtime emoji dependency. |
| `README.md` | Exact URL, symbol, emoji, residual-`x`, warning, and known-limitation behavior. |

---

### Task 1: Unify URL cleanup without deleting filenames

**Files:**
- Modify: `src/pytts/text/cleaner.py:10-51`
- Modify: `tests/text/test_cleaner.py:17-61`

**Interfaces:**
- Consumes: block text after NFC normalization, mixed-script repair, and technical-character removal.
- Produces: `_remove_url(match: re.Match[str]) -> str`, returning only punctuation trimmed from the URL candidate.
- Preserves: `clean_article(article: Article) -> CleaningResult`.

- [ ] **Step 1: Write failing URL boundary and ambiguity tests**

Add these tests to `tests/text/test_cleaner.py`:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Текст (https://example.com/path).", "Текст ()."),
        ("Текст [www.example.com/path],", "Текст [],"),
        ("Текст «sponsr.ru/path».", "Текст «»."),
        ("Текст example.com?q=1!", "Текст !"),
        ("Текст example.com#section?", "Текст ?"),
        ("Текст https://example.com/path…", "Текст …"),
    ],
)
def test_removes_all_url_forms_but_preserves_trailing_punctuation(
    source: str, expected: str
) -> None:
    assert clean_article(_article(source)).article.blocks[0].text == expected


def test_preserves_ambiguous_host_only_tokens_and_filenames() -> None:
    source = "Откройте main.py, README.md, config.yaml, index.html и example.com."

    result = clean_article(_article(source))

    assert result.article.blocks[0].text == source


def test_does_not_remove_email_decimal_or_embedded_domain_like_text() -> None:
    source = "Пишите user@example.com/path; версия 3.14; токен token_example.com/path."

    result = clean_article(_article(source))

    assert result.article.blocks[0].text == source
```

Keep `test_rejects_article_empty_after_cleanup`: a scheme URL without a path,
`https://example.com`, is still unambiguous and must still leave no readable
text.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_cleaner.py -v
```

Expected: the existing greedy scheme/`www` expression consumes closing
punctuation, while path-bearing bare URLs remain in the output; the new tests
fail for those exact differences.

- [ ] **Step 3: Replace the URL expression with one shared matcher and callback**

Replace `_URL` in `src/pytts/text/cleaner.py` and add the callback:

```python
_URL = re.compile(
    r"(?i)(?<![\w@])(?:"
    r"(?:https?://|www\.)[^\s<>]+"
    r"|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,}(?:[/?#][^\s<>]*)"
    r")"
)
_URL_TRAILING_PUNCTUATION = frozenset(".,;:!?…)]}»”’\"'")
_SPACE = re.compile(r"\s+")
_TECHNICAL_UNICODE = frozenset("\u00ad\u200b\u2060\ufeff")


def _remove_url(match: re.Match[str]) -> str:
    candidate = match.group(0)
    boundary = len(candidate)
    while (
        boundary > 0
        and candidate[boundary - 1] in _URL_TRAILING_PUNCTUATION
    ):
        boundary -= 1
    return candidate[boundary:]
```

Change the substitution in `clean_article` to use the callback:

```python
text = _SPACE.sub(" ", _URL.sub(_remove_url, without_artifacts)).strip()
```

The scheme/`www` branch deliberately accepts host-only URLs. The bare branch
requires `/`, `?`, or `#` immediately after the ASCII hostname, which is the
fail-open distinction between `example.com/path` and `main.py`.

- [ ] **Step 4: Run focused and reader regression tests**

Run:

```bash
uv run pytest tests/text/test_cleaner.py tests/readers/test_markdown.py tests/readers/test_pdf.py -v
uv run ruff check src/pytts/text/cleaner.py tests/text/test_cleaner.py
```

Expected: all selected tests pass and Ruff emits no diagnostics.

- [ ] **Step 5: Commit the URL behavior**

```bash
git add src/pytts/text/cleaner.py tests/text/test_cleaner.py
git commit -m "fix: normalize web article URLs safely"
```

---

### Task 2: Read joined uppercase letter-number codes

**Files:**
- Modify: `src/pytts/text/numeric_normalizer.py:76-78,247-270,348-352`
- Modify: `tests/text/test_numeric_normalizer.py`

**Interfaces:**
- Consumes: a standalone token with one to five uppercase ASCII letters followed by one or more digits.
- Produces: the existing English letter names from `spell_code_letters()` followed by `cardinal()` Russian number words.
- Preserves: hyphenated `_CODE`, date, range, currency, percentage, ordinal, and remaining-integer behavior.

- [ ] **Step 1: Write failing joined-code tests**

Add to `tests/text/test_numeric_normalizer.py`:

```python
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
```

- [ ] **Step 2: Run the joined-code tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_numeric_normalizer.py -k "joined" -v
```

Expected: `X5`, `FPV5`, and `F16` remain unchanged because
`_REMAINING_INTEGER` cannot consume digits adjacent to a word character.

- [ ] **Step 3: Add the bounded matcher before remaining integers**

Add beside `_CODE`:

```python
_ALNUM_CODE = re.compile(
    r"(?<!\w)(?P<letters>[A-Z]{1,5})(?P<number>\d+)(?!\w)"
)
```

Call it after the existing hyphenated code rule and before all later numeric
fallbacks:

```python
text = _CODE.sub(self._code, text)
text = _ALNUM_CODE.sub(self._alnum_code, text)
text = _COMPOUND_YEARS.sub(self._compound_years, text)
```

Add the callback to `NumericNormalizer`:

```python
def _alnum_code(self, match: re.Match[str]) -> str:
    return (
        f"{spell_code_letters(match.group('letters'))} "
        f"{cardinal(match.group('number'))}"
    )
```

Do not merge `_CODE` and `_ALNUM_CODE`: their separators and supported Cyrillic
case differ, and the separate expressions keep those contracts visible.

- [ ] **Step 4: Run focused and full numeric regressions**

Run:

```bash
uv run pytest tests/text/test_numeric_normalizer.py tests/text/test_russian_numbers.py -v
uv run ruff check src/pytts/text/numeric_normalizer.py tests/text/test_numeric_normalizer.py
```

Expected: all selected tests pass, including `F-16`, decade, signed-number,
decimal-percentage, and idempotence tests.

- [ ] **Step 5: Commit joined-code normalization**

```bash
git add src/pytts/text/numeric_normalizer.py tests/text/test_numeric_normalizer.py
git commit -m "fix: pronounce joined letter number codes"
```

---

### Task 3: Repair residual Latin x after transliteration

**Files:**
- Modify: `src/pytts/text/latin.py:134-150`
- Modify: `tests/text/test_latin.py:40-51`

**Interfaces:**
- Consumes: the string returned by `cyrtranslit.to_cyrillic(ascii_token, "ru")`.
- Produces: every residual ASCII `x` as `кс` and `X` as `Кс`.
- Preserves: the early 1–5 uppercase acronym path, so standalone `X` remains `икс`.

- [ ] **Step 1: Write failing residual-x tests**

Add to `tests/text/test_latin.py`:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Petroxi", "Петрокси"),
        ("tax", "такс"),
        ("Interfax", "Интерфакс"),
        ("Xerox", "Ксерокс"),
    ],
)
def test_repairs_residual_x_after_cyrtranslit(
    source: str, expected: str
) -> None:
    assert normalize_latin(source) == expected


def test_preserves_uppercase_x_spelling_and_documents_mixed_case_fallback() -> None:
    assert normalize_latin("X Xi") == "икс Кси"


def test_residual_x_repair_is_idempotent() -> None:
    once = normalize_latin("Petroxi и Interfax")

    assert normalize_latin(once) == once
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_latin.py -k "residual_x or mixed_case" -v
```

Expected: `cyrtranslit` leaves ASCII `x`/`X` in the returned words, so the exact
Russian expectations fail; standalone uppercase `X` already remains correctly
spelled as `икс`.

- [ ] **Step 3: Repair only the transliteration result**

Change `_latin_replacement` in `src/pytts/text/latin.py`:

```python
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
```

The replacement is intentionally after the acronym branch. It is mechanical,
so names such as `Xi` use the documented approximate `Кси` reading unless the
user provides a `transliterations` entry.

- [ ] **Step 4: Run Latin and pronunciation regressions**

Run:

```bash
uv run pytest tests/text/test_latin.py tests/text/test_pronunciation.py -v
uv run ruff check src/pytts/text/latin.py tests/text/test_latin.py
```

Expected: all selected tests pass with no unsupported ASCII `x` left by the
new examples.

- [ ] **Step 5: Commit residual-x repair**

```bash
git add src/pytts/text/latin.py tests/text/test_latin.py
git commit -m "fix: pronounce residual latin x"
```

---

### Task 4: Normalize arrows, approximation, boundary stars, and emoji

**Files:**
- Modify: `pyproject.toml:11-23`
- Modify: `uv.lock`
- Modify: `src/pytts/text/pronunciation.py:3-112`
- Modify: `tests/text/test_pronunciation.py`
- Create: `tests/fixtures/article_artifacts.md`
- Modify: `tests/test_pipeline.py:9-13,74-115`

**Interfaces:**
- Consumes: text after user overrides and numeric normalization.
- Produces: `_strip_boundary_stars(text: str) -> str`.
- Uses: `emoji.replace_emoji(text, replace="")` before any star rule.
- Preserves: `PronunciationNormalizer.normalize_article(article) -> NormalizationResult`.
- Guarantees: an all-emoji article still raises the existing `InputError`; unknown non-emoji symbols still generate warning notices.

- [ ] **Step 1: Write failing focused pronunciation tests**

Import `InputError` in `tests/text/test_pronunciation.py`:

```python
from pytts.errors import InputError
```

Remove `("Температура 🌡 высокая.", "🌡")` from
`test_strips_unspeakable_characters_and_warns`, because remaining emoji no
longer belong to the warning contract.

Add:

```python
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("~28 миллиардов", "примерно двадцать восемь миллиардов"),
        ("≈25%", "примерно двадцать пять процентов"),
        ("к прочтению -> название", "к прочтению: название"),
        ("к прочтению → название", "к прочтению: название"),
    ],
)
def test_normalizes_approximation_and_arrows(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**это не рекомендация**", "это не рекомендация"),
        ("** Данная информация", "Данная информация"),
        ("решения **", "решения"),
        ("*Данный документ*", "Данный документ"),
    ],
)
def test_strips_one_or_two_stars_only_at_block_boundaries(
    source: str, expected: str
) -> None:
    assert _normalize(source) == expected


def test_triple_boundary_stars_remain_visible_to_warning_guard() -> None:
    result = PronunciationNormalizer({}).normalize_article(_article("***бать"))

    assert result.article.blocks[0].text == "бать"
    assert result.warning is not None
    assert result.warning.startswith("Removed 3 unsupported character(s)")
    assert result.warning.count("'*' near") == 3


def test_interior_mathematical_star_remains_visible_to_warning_guard() -> None:
    result = PronunciationNormalizer({}).normalize_article(_article("2 * 3"))

    assert result.article.blocks[0].text == "два три"
    assert result.warning is not None
    assert result.warning.count("'*' near") == 1


def test_double_tilde_is_not_treated_as_approximation() -> None:
    result = PronunciationNormalizer({}).normalize_article(_article("~~текст~~"))

    assert result.article.blocks[0].text == "текст"
    assert result.warning is not None
    assert result.warning.count("'~' near") == 4


def test_removes_remaining_emoji_sequences_without_warning() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("До 🎮 и 👩🏽‍💻 после 🇷🇺 знака.")
    )

    assert result.article.blocks[0].text == "До и после знака."
    assert result.warning is None


def test_emoji_removal_precedes_boundary_star_cleanup() -> None:
    result = PronunciationNormalizer({}).normalize_article(
        _article("*️⃣ **текст** 👨‍👩‍👧‍👦")
    )

    assert result.article.blocks[0].text == "текст"
    assert result.warning is None


def test_transliteration_override_can_pronounce_emoji_before_removal() -> None:
    assert _normalize("🎮", {"🎮": "игра"}) == "игра"


def test_rejects_article_containing_only_unmapped_emoji() -> None:
    with pytest.raises(InputError, match="No speakable text remains"):
        PronunciationNormalizer({}).normalize_article(_article("🎮 👨‍👩‍👧‍👦"))
```

- [ ] **Step 2: Add the failing cross-module fixture contract**

Create `tests/fixtures/article_artifacts.md`:

```markdown
# Проверка артефактов

\*\*Данная информация\*\*

\*Данный документ не является рекомендацией\*

Цензура: \*\*\*бать.

Оценка: ~28 миллиардов и ≈25%.

Переход -> название, затем → вывод.

Petroxi Trading, windfall tax, X5 Group и Interfax.

Ссылка sponsr.ru/crimsonanalytics/126423 удаляется, а main.py и config.yaml остаются.

Декор: 🎮 \*️⃣ 👨‍👩‍👧‍👦.
```

Extend the pipeline imports:

```python
from pytts.pipeline import (
    ArticleReader,
    ConversionPipeline,
    ProgressEvent,
    ProgressStage,
)
from pytts.readers.input import InputReader
```

Widen the existing `_pipeline` helper argument without changing its behavior:

```python
reader: ArticleReader | None = None,
```

Add to `tests/test_pipeline.py`:

```python
def test_article_artifact_fixture_leaves_only_censored_star_warning(
    tmp_path: Path,
) -> None:
    events: list[ProgressEvent] = []
    runtime = FakeRuntime()
    source = Path(__file__).parent / "fixtures" / "article_artifacts.md"
    pipeline = _pipeline(
        tmp_path,
        runtime,
        events,
        [],
        reader=InputReader(),
    )

    pipeline.convert(
        ConversionRequest(
            source,
            output_path=tmp_path / "article_artifacts.mp3",
        )
    )

    combined = " ".join(runtime.chunks)
    for fragment in (
        "Данная информация",
        "Данный документ не является рекомендацией",
        "примерно двадцать восемь миллиардов",
        "примерно двадцать пять процентов",
        "Петрокси Традинг",
        "виндфалл такс",
        "икс пять Гроуп",
        "Интерфакс",
        "маин.пи",
        "конфиг.ямл",
    ):
        assert fragment in combined
    assert "sponsr" not in combined.casefold()
    assert not any(character.isdigit() for character in combined)
    assert all(emoji not in combined for emoji in ("🎮", "*️⃣", "👨‍👩‍👧‍👦"))

    warnings = [
        event.message
        for event in events
        if event.warning
        and event.message is not None
        and event.message.startswith("Removed ")
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith("Removed 3 unsupported character(s)")
    assert warnings[0].count("'*' near") == 3
```

- [ ] **Step 3: Run focused and integration tests and confirm RED**

Run:

```bash
uv run pytest tests/text/test_pronunciation.py tests/test_pipeline.py -v
```

Expected: approximation signs and arrows are stripped or warned instead of
spoken, boundary stars generate notices, emoji generate notices, and the
fixture has more than the three intended warning entries.

- [ ] **Step 4: Add the pinned emoji dependency**

Run:

```bash
uv add "emoji>=2,<3"
```

Expected: `pyproject.toml` contains `"emoji>=2,<3"` in runtime dependencies and
`uv.lock` contains the resolved `emoji` package. This download is only
dependency installation; normalization and synthesis remain offline after
`uv sync`.

- [ ] **Step 5: Implement the correctness-critical pronunciation order**

Add the import and patterns to `src/pytts/text/pronunciation.py`:

```python
import emoji

_CONTEXT = 20

_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
_LEADING_BOUNDARY_STARS = re.compile(r"^\s*\*{1,2}(?!\*)\s*")
_TRAILING_BOUNDARY_STARS = re.compile(r"\s*(?<!\*)\*{1,2}\s*$")
_APPROXIMATION = re.compile(r"(?<!~)~(?!~)|≈")
```

Add:

```python
def _strip_boundary_stars(text: str) -> str:
    text = _LEADING_BOUNDARY_STARS.sub("", text)
    return _TRAILING_BOUNDARY_STARS.sub("", text)
```

Replace `_normalize_symbols` with:

```python
def _normalize_symbols(text: str) -> str:
    text = text.replace("->", " : ").replace("→", " : ")
    text = _APPROXIMATION.sub(" примерно ", text)
    text = re.sub(r"#(?=\s*\d)", " номер ", text)
    text = re.sub(r"#(?=\s*[A-Za-zА-Яа-яЁё])", " хештег ", text)
    for symbol, words in _SYMBOL_WORDS.items():
        text = text.replace(symbol, f" {words} ")
    collapsed = _SPACE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed)
```

Change the per-block body of `normalize_article` to this exact order:

```python
text = self._overrides.apply(block.text)
text = self._numbers.normalize(text)
text = emoji.replace_emoji(text, replace="")
text = _strip_boundary_stars(text)
text = _normalize_symbols(text)
text = normalize_latin(text)
text, block_notices = _strip_unspeakable(text)
notices.extend(block_notices)
text = _SPACE.sub(" ", text).strip()
```

Emoji removal must precede boundary stars so `*️⃣` is removed as one emoji
sequence. Overrides must remain first so `{"🎮": "игра"}` wins over deletion.
The approximation expression handles a single `~` and `≈` but intentionally
leaves `~~` for the final warning guard.

- [ ] **Step 6: Run focused, fixture, and full non-Silero regressions**

Run:

```bash
uv run pytest tests/text/test_pronunciation.py tests/test_pipeline.py -v
uv run pytest -m "not silero"
uv run ruff check src/pytts/text/pronunciation.py tests/text/test_pronunciation.py tests/test_pipeline.py
uv lock --check
```

Expected: the focused suite passes, the full suite has no failures, Ruff has
no diagnostics, and the lock file is current.

- [ ] **Step 7: Commit pronunciation artifacts and integration coverage**

```bash
git add pyproject.toml uv.lock src/pytts/text/pronunciation.py tests/text/test_pronunciation.py tests/fixtures/article_artifacts.md tests/test_pipeline.py
git commit -m "fix: normalize web article speech artifacts"
```

---

### Task 5: Document behavior and complete real-file acceptance

**Files:**
- Modify: `README.md:93-137`
- Modify: `tests/test_silero_integration.py:48`

**Interfaces:**
- Consumes: the implemented behavior and the committed artifact fixture.
- Produces: user-facing rules, an opt-in real-Silero fixture case, and fresh
  verification evidence for both supplied article formats.

- [ ] **Step 1: Add the artifact fixture to opt-in real synthesis**

Replace the fixture parameter in `tests/test_silero_integration.py`:

```python
@pytest.mark.parametrize(
    "fixture_name",
    ["smoke.md", "pronunciation.md", "article_artifacts.md"],
)
```

Add the channel assertion to both MP3 metadata checks:

```python
assert info.sample_rate == 48000
assert info.channels == 1
assert 90000 <= info.bitrate <= 100000
```

Run the non-Silero collection check:

```bash
uv run pytest tests/test_silero_integration.py --collect-only
```

Expected: four Silero cases are collected: the direct model test and the three
CLI fixture parameters. Use the exact count printed by pytest as evidence; do
not hard-code it in production code.

- [ ] **Step 2: Update pronunciation behavior in README**

Replace the paragraph beginning `Before Silero, pytts converts` with:

```markdown
Before Silero, pytts removes scheme/`www` URLs and bare URLs that contain a path, query, or
fragment. Ambiguous host-only tokens such as `main.py` and `example.com` are preserved rather than
silently discarded. It converts common dates, years, decades (`1990-х`), Roman-numeral centuries
(`XX века`), integers, decimals, currencies, percentages, ranges, hyphenated and joined
letter-number codes, and documented semantic symbols to Russian words. `~` and `≈` are read as
`примерно`; `->` and `→` become a colon pause. One or two stars at a text-block boundary are treated
as formatting. Mixed-script PDF lookalikes are repaired contextually, and residual Latin `x` is
mapped mechanically to `кс`.

Any remaining emoji after exact `transliterations` are removed silently as decoration. Any other
letter or symbol Silero cannot voice is dropped rather than aborting the run, and a single warning
lists each removed character with its surrounding context so you can find it in the source.
```

Replace the current known-limitation bullets about approximate foreign words
and warning behavior with:

```markdown
- General foreign-word transliteration is approximate. Residual Latin `x` is mapped mechanically
  to `кс`, so mixed-case names such as `Xi` may be imperfect; use `transliterations` for exact names.
- Uppercase Latin groups of 1–5 letters are spelled by letter names, so `NASA` is read as
  `эн эй эс эй`; override it in `transliterations` when word-like pronunciation is preferred.
- Emoji left after exact `transliterations` are silently removed as decoration. Greek and other
  non-Cyrillic letters and unknown semantic symbols are dropped with a warning containing the
  character and context.
```

Replace the URL limitation bullet with:

```markdown
- Scheme/`www` URLs and bare URLs with a path, query, or fragment are discarded as non-article
  content. Ambiguous host-only tokens are preserved and may be transliterated into speech. Code
  blocks, images, metadata, tables, footnotes, and raw HTML remain discarded by the readers.
```

- [ ] **Step 3: Run text-only acceptance on both supplied article files**

Run this from the project root:

```bash
uv run python - <<'PY'
from pathlib import Path

from pytts.config import load_config
from pytts.readers.input import InputReader
from pytts.text.abbreviations import AbbreviationExpander
from pytts.text.cleaner import clean_article
from pytts.text.pronunciation import PronunciationNormalizer

project_root = Path.cwd()
config = load_config(None, project_root)
paths = (
    Path("/Users/azdrachek/Downloads/Новостной деск.pdf"),
    Path(
        "/Users/azdrachek/Downloads/"
        "Новостной деск Апдейты по акциям Русского списка.md"
    ),
)
required = (
    "Петрокси Традинг",
    "виндфалл такс",
    "икс пять Гроуп",
    "Интерфакс",
    "примерно двадцать восемь миллиардов",
)

for path in paths:
    article = InputReader().read(path)
    cleaned = clean_article(article)
    expanded = AbbreviationExpander(config.abbreviations).expand_article(
        cleaned.article
    )
    result = PronunciationNormalizer(config.transliterations).normalize_article(
        expanded
    )
    text = "\n".join(block.text for block in result.article.blocks)
    assert result.warning is not None, path
    assert result.warning.startswith("Removed 3 unsupported character(s)"), (
        path,
        result.warning,
    )
    assert result.warning.count("'*' near") == 3, (path, result.warning)
    for fragment in required:
        assert fragment in text, (path, fragment)
    assert "sponsr.ru/crimsonanalytics/126423" not in text.casefold(), path
    assert "🎮" not in text, path
    print(f"{path.name}: {result.warning}")
PY
```

Expected: each filename is printed once with a warning beginning
`Removed 3 unsupported character(s)`; every assertion succeeds. The count of
three is specific to these two saved versions of the article, not a general
normalizer invariant.

- [ ] **Step 4: Run the complete static and test gate**

Run:

```bash
uv run pytest -m "not silero"
uv run ruff check .
uv lock --check
git diff --check
```

Expected: all non-Silero tests pass, Ruff emits no diagnostics, the lock is
current, and Git reports no whitespace errors.

- [ ] **Step 5: Run real cached Silero acceptance**

With the model already present in
`~/Library/Caches/pytts/models/v5_5_ru.pt`, disconnect macOS networking and
run:

```bash
PYTTS_RUN_SILERO=1 uv run pytest -m silero -v
```

Expected: all opt-in Silero tests pass, including
`article_artifacts.md`; generated MP3 metadata remains mono, 48 kHz, and
approximately 96 kbit/s. Reconnect networking after the run. If the model is
not already cached, stop instead of downloading during this offline gate.

- [ ] **Step 6: Commit documentation and real-model coverage**

```bash
git add README.md tests/test_silero_integration.py
git commit -m "docs: describe article artifact normalization"
```

- [ ] **Step 7: Request final code review and verify the reviewed tree**

Use `superpowers:requesting-code-review` against the full branch diff from
`master`. Resolve every Critical or Important finding in a separate focused
commit, then rerun:

```bash
uv run pytest -m "not silero"
PYTTS_RUN_SILERO=1 uv run pytest -m silero
uv run ruff check .
uv lock --check
git diff --check
git status --short
```

Expected: both pytest commands pass, Ruff and lock checks are clean, the diff
has no whitespace errors, and `git status --short` contains only the
user-owned untracked `.DS_Store`.
