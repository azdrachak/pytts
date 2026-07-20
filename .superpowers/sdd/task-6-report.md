# Task 6 report: literal abbreviation expansion

Implemented `AbbreviationExpander` as a single, Unicode case-insensitive
substitution pass per text block. Keys are escaped before compiling the
longest-first alternation; word-character boundaries prevent matching inside
words while preserving punctuation and underscore boundaries. Replacements are
looked up case-insensitively and receive only the matched source's initial
capitalization, so all-caps input intentionally expands to sentence case.

The existing root `pytts.yaml` is the configured abbreviation dictionary and
contains the selected `"ув.": "уважаемый"` entry alongside `"г-н"`.

## TDD evidence

- RED: `uv run pytest tests/text/test_abbreviations.py -v` failed during
  collection with `ModuleNotFoundError: No module named 'pytts.text.abbreviations'`.
- GREEN: the same focused command passed: 6 tests.
- Ruff: `uv run ruff check src/pytts/text/abbreviations.py tests/text/test_abbreviations.py`
  reported `All checks passed!`.
- Regression suite: `uv run pytest -v` passed: 68 tests.

The full suite emits pre-existing third-party PyMuPDF/PyTorch deprecation and
NumPy-initialization warnings but has no test failures.
