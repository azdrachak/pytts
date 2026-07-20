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

## Abbreviations

The root `pytts.yaml` is versioned and editable:

```yaml
version: 1
abbreviations:
  "г-н": "господин"
  "г-жа": "госпожа"
  "ув.": "уважаемый"
  "т.е.": "то есть"
```

Replacements are literal, case-insensitive, longest-first, token-bounded, and one-pass. Only the
first letter's case is transferred. An explicit `--config` replaces the root mapping rather than
merging with it.

## Output and errors

Output is one mono 48 kHz, 16-bit-source, CBR 96 Kbit/s MP3. The encoder writes `OUTPUT.part` and
atomically installs `OUTPUT` only after successful completion. Use `--force` to replace an existing
output. Exit codes are 2 for usage, 3 for input/config/output, 4 for model failures, 5 for
synthesis/encoding, and 130 for interruption. `--debug` includes a traceback.

## Known limitations

- Russian only. Low Cyrillic content emits a warning but is not blocked; there is no language
  validation.
- Latin tokens in otherwise Russian input can be rejected by the Silero SSML parser; map them through
  abbreviations or write them in Cyrillic.
- PDF requires an existing text layer; there is no OCR.
- Complex multi-column PDF reading order is not guaranteed; the target is a browser-printed article.
- PDF headings are a font-size heuristic; uniform browser-print typography gets paragraph pauses.
- Input is limited to articles, not books or other long-form document workflows.
- Speech chunks contain at most 448 clean text characters.
- URLs, code, images, metadata, tables, footnotes, and raw HTML are discarded as non-article content.
- Numbers, dates, currencies, percentages, and ranges are preserved literally and are not converted
  to words, so pronunciation and grammatical agreement depend on Silero.
- Context-sensitive abbreviation gender and morphology are outside this release.

## Model license

The Silero model is distributed under
[CC BY-NC-SA 4.0](https://github.com/snakers4/silero-models/blob/master/LICENSE). This project is for
personal, non-commercial use. The model file is downloaded separately and is never committed or
packaged with `pytts`.
