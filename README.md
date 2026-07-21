# pytts

`pytts` locally converts Russian text-layer PDF and Markdown articles into one MP3 on Apple Silicon
macOS. It uses Silero `v5_5_ru` on CPU; article text and audio never leave the computer.

## Requirements and installation

- Apple Silicon macOS
- [uv](https://docs.astral.sh/uv/)
- network access for initial uv/dependency setup and the first model download

```bash
uv sync --group dev
uv run pytts --help
```

The project pins Python 3.12 and the resolved dependencies in `uv.lock`. PyTorch comes from the
normal macOS PyPI wheel; CUDA, `ffmpeg`, `torchaudio`, and `torchvision` are not used. Run the command
from this checkout so the editable root `pytts.yaml` remains the default config. `uv tool install .`
is therefore not the primary installation path.

## Usage

```bash
uv run pytts article.md
uv run pytts article.pdf --output article.mp3 --voice xenia --speed fast
uv run pytts article.md --config another.yaml --force
uv run pytts --list-voices
uv run --project /Users/azdrachek/Project/pytts pytts /Users/azdrachek/Downloads/article.pdf
```

The first conversion, first `--list-voices`, and first `--voice` validation may download and fully
load the model before producing a result. Voice names come from `model.speakers`; `xenia` is preferred
when available, otherwise the first runtime voice is used with a warning. The Phase 0 verification
report records the speakers confirmed for its runtime; use `uv run pytts --list-voices` to discover the
actual speakers available in the local model. `INPUT` and `--list-voices` cannot be combined.

Supported `--speed` values are `x-slow`, `slow`, `normal`, `fast`, and `x-fast`; `normal` is the
default.

The model is cached at `~/Library/Caches/pytts/models/v5_5_ru.pt`. Every use verifies the committed
SHA-256 from `src/pytts/model_manifest.yaml`; a verified cache works offline. A hash mismatch reports
the expected and actual values and explains the deliberate manifest-update process.
The repeat SHA-256 pass reads the roughly 100 MB model and can add a short startup delay; this is the
intentional integrity tradeoff for the MVP.

## Pronunciation configuration

The root `pytts.yaml` is versioned and editable:

```yaml
version: 1
abbreviations:
  "г-н": "господин"
  "т.е.": "то есть"
  "ЧМ": "чемпионат мира"
  "ув.": ""            # a blank value removes the token from speech
transliterations:
  "Brent": "Брент"
  "New York Times": "Нью-Йорк таймс"
```

`abbreviations` expands literal Russian abbreviations. `transliterations` supplies exact spoken
forms for foreign names, brands, non-Cyrillic letters, or symbols before the general normalizer.
Both mappings are case-insensitive, longest-first, token-bounded, and one-pass. A blank value (for
example `"ув.": ""`) deletes the matched token instead of expanding it, which suits filler words
with no correct spoken form. An explicit `--config` replaces the root config rather than merging
with it.

Before Silero, pytts converts common dates, years, decades (`1990-х`), Roman-numeral centuries
(`XX века`), integers, decimals, currencies, percentages, ranges, letter-number codes, and
documented semantic symbols to Russian words. Mixed-script PDF lookalikes are repaired contextually.
Any letter or symbol Silero cannot voice is dropped rather than aborting the run, and a single
warning lists each removed character with its surrounding context so you can find it in the source.

## Output and errors

Output is one mono 48 kHz, 16-bit-source, CBR 96 Kbit/s MP3. The encoder writes `OUTPUT.part` and
atomically installs `OUTPUT` only after successful completion. Use `--force` to replace an existing
output. Exit codes are 2 for usage, 3 for input/config/output-path validation, 4 for model failures,
5 for synthesis/encoding/output I/O, and 130 for interruption. `--debug` includes a traceback.

## Known limitations

- Russian only. Low Cyrillic content emits a warning but is not blocked; there is no language
  validation.
- General foreign-word transliteration is approximate; use `transliterations` for exact names.
- Uppercase Latin groups of 1–5 letters are spelled by letter names, so `NASA` is read as
  `эн эй эс эй`; override it in `transliterations` when word-like pronunciation is preferred.
- Greek and other non-Cyrillic letters, emoji, and unknown semantic symbols are dropped with a
  warning (character plus context) unless an exact `transliterations` replacement makes them speakable.
- Numeric shapes `DD.MM.YYYY` and `DD/MM/YYYY` are always treated as calendar dates. Version-like
  triples require an exact replacement or source rewrite; invalid calendar dates fail closed.
- Context rules cover the documented dates, years, currency, ranges and two compound-adjective
  families; ambiguous Russian syntax can still produce a non-ideal case.
- Roman numerals are read only directly before a century noun (`XX века` → `двадцатого века`);
  other uses (`том XIV`, monarch names) are spelled letter by letter.
- Only unit and multiplier abbreviations you configure (`км`, `млн`, …) are expanded, and their case
  agreement after a number is approximate.
- PDF requires an existing text layer; there is no OCR.
- Complex multi-column PDF reading order is not guaranteed; the target is a browser-printed article.
- PDF headings are a font-size heuristic; uniform browser-print typography gets paragraph pauses.
- Input is limited to articles, not books or other long-form document workflows.
- Speech chunks contain at most 448 clean text characters.
- URLs, code, images, metadata, tables, footnotes, and raw HTML are discarded as non-article content.
- Context-sensitive abbreviation gender and morphology are outside this release.

## Model license

The Silero model is distributed under
[CC BY-NC-SA 4.0](https://github.com/snakers4/silero-models/blob/master/LICENSE). This project is for
personal, non-commercial use. The model file is downloaded separately and is never committed or
packaged with `pytts`.
