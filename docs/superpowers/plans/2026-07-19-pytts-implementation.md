# pytts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local macOS CLI that converts Russian text-layer PDF or Markdown articles into one streamed MP3 using Silero `v5_5_ru`.

**Architecture:** A linear, dependency-injected pipeline converts an input file into typed article blocks, cleans and expands text, chunks it into SSML, synthesizes each chunk with a runtime-verified Silero adapter, and streams PCM into one atomic LAME writer. A mandatory Phase 0 probes the real model before any article-processing code is implemented; the runtime voice list, model hash, SSML contract, and safe text limit come from that probe.

**Tech Stack:** Python 3.12, uv/uv-build, Typer, Rich, PyMuPDF, markdown-it-py, PyYAML, PyTorch on CPU, platformdirs, lameenc, pytest, pytest-cov, mutagen, Ruff.

## Global Constraints

- Target platform: local Apple Silicon macOS only; execute PyTorch on `torch.device("cpu")`.
- Python requirement: `>=3.12,<3.13`; do not modify the system Python 3.14 installation.
- Primary launch path: `uv run pytts`; no Homebrew, Docker, cloud API, GPU, `ffmpeg`, `torchvision`, or `torchaudio` dependency.
- Inputs: exactly one case-insensitive `.pdf` or `.md` per conversion; PDF must already have a text layer; no OCR.
- Language: Russian only; warn when there are no letters or Cyrillic is below 70% of alphabetic characters, but do not block synthesis.
- Scale: one output MP3 and up to one hour of audio; never accumulate the complete waveform in memory.
- Audio: mono, 48 kHz, signed 16-bit PCM into LAME CBR 96 Kbit/s, LAME quality `2`.
- Model: `v5_5_ru`; download from the official URL, verify the committed SHA-256, cache under `platformdirs.user_cache_dir("pytts")`, and work offline after caching.
- Model license: `CC BY-NC-SA 4.0`; personal, non-commercial use only; never commit or package the model file.
- Runtime voices: use `model.speakers`; prefer `xenia` only when present, otherwise select the first runtime voice with a warning.
- Rates: `x-slow`, `slow`, `normal`, `fast`, `x-fast`; map `normal` to SSML `medium`.
- Text normalization deliberately excludes numbers, dates, currencies, percentages, ranges, context-sensitive abbreviations, and full language detection.
- Root config: strict `pytts.yaml` schema version `1`; an explicit `--config` replaces it rather than merging.
- Failure safety: write only to `OUTPUT.part`, atomically replace the final output after `flush()`, and remove the exact partial path on failure or interruption.
- Phase 0 is a hard gate. If `PackageImporter`, `model.speakers`, `apply_tts(text=...)`, SSML prosody, 48 kHz mono output, or the safe length probe fails, stop and revise the design before Task 2.
- Every feature task follows red-green-refactor, runs focused tests, runs `ruff check` on touched files, and ends with a small commit.

---

## File and Responsibility Map

### Project and documentation

- `.python-version` — pins Python 3.12 for uv.
- `.gitignore` — excludes virtual environments, caches, generated audio, IDE state, and model artifacts.
- `pyproject.toml` — package metadata, dependencies, uv groups, CLI entry point, pytest and Ruff configuration.
- `uv.lock` — exact macOS-arm64 dependency resolution, including the PyTorch wheel.
- `pytts.yaml` — editable root abbreviation dictionary.
- `README.md` — installation, CLI examples, limitations, model download behavior, and exact Silero license notice.
- `docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md` — generated and human-completed Phase 0 evidence.
- `scripts/probe_silero.py` — thin executable for the Phase 0 probe.

### Package

- `src/pytts/__init__.py` — package version only.
- `src/pytts/__main__.py` — `python -m pytts` entry point.
- `src/pytts/domain.py` — immutable shared enums and dataclasses.
- `src/pytts/errors.py` — stable application exception hierarchy and exit codes.
- `src/pytts/config.py` — strict YAML loading and project-root discovery.
- `src/pytts/silero_probe.py` — testable Phase 0 helpers and probe runner.
- `src/pytts/model_manifest.yaml` — committed runtime model ID, URL, SHA-256, preferred voice, sample rate, and probed safe text length.
- `src/pytts/readers/markdown.py` — Markdown token-to-block conversion.
- `src/pytts/readers/pdf.py` — positional PDF extraction and layout heuristics.
- `src/pytts/readers/input.py` — extension dispatcher implementing `read(path) -> Article`.
- `src/pytts/text/cleaner.py` — deterministic cleanup and Cyrillic warning.
- `src/pytts/text/abbreviations.py` — one-pass literal abbreviation expansion.
- `src/pytts/text/chunker.py` — sentence-aware chunking, XML escaping, SSML, and pause assignment.
- `src/pytts/model_store.py` — manifest loading, cache location, HTTPS download, hash verification, and atomic cache placement.
- `src/pytts/tts.py` — Torch Package loading, runtime voice resolution, and per-chunk synthesis.
- `src/pytts/audio.py` — bounded PCM conversion and atomic streaming MP3 writer.
- `src/pytts/pipeline.py` — orchestration, progress events, output selection, and cleanup.
- `src/pytts/cli.py` — Typer arguments, Rich rendering, list-voices mode, debug output, and exit mapping.

### Tests

- `tests/test_silero_probe.py` — deterministic Phase 0 helper tests.
- `tests/test_domain.py` — shared value-object invariants.
- `tests/test_config.py` — YAML lookup, precedence, and strict schema tests.
- `tests/readers/test_markdown.py` — Markdown preservation and omission behavior.
- `tests/readers/test_pdf.py` — PDF ordering, header/footer, page number, font, and hyphen tests.
- `tests/readers/test_input.py` — extension dispatch and unsupported input errors.
- `tests/text/test_cleaner.py` — normalization, URL removal, emptiness, and language warning.
- `tests/text/test_abbreviations.py` — longest literal match, boundaries, case, and non-recursion.
- `tests/text/test_chunker.py` — sentence splitting, long text, SSML, escaping, and pauses.
- `tests/test_model_store.py` — local HTTP downloads, cache, SHA mismatch, and atomic failure behavior.
- `tests/test_tts.py` — runtime speakers, preferred fallback, PackageImporter adapter, and synthesis errors.
- `tests/test_audio.py` — PCM conversion, streaming writes, atomic commit, abort, and real LAME metadata.
- `tests/test_pipeline.py` — full fake-backed orchestration and progress order.
- `tests/test_cli.py` — command modes, exit codes, errors, and debug traceback.
- `tests/test_e2e.py` — real readers plus fake runtime/writer acceptance flow.
- `tests/test_silero_integration.py` — opt-in real-model smoke test using the committed manifest.
- `tests/fixtures/smoke.md` — deterministic Russian CLI and end-to-end acceptance article.

---

### Task 1: Bootstrap and Pass the Mandatory Silero Phase 0 Gate

**Files:**
- Create: `.python-version`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/pytts/__init__.py`
- Create: `src/pytts/silero_probe.py`
- Create: `scripts/probe_silero.py`
- Create: `tests/test_silero_probe.py`
- Generate and commit: `src/pytts/model_manifest.yaml`
- Generate and commit: `docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md`
- Generate but do not commit: `artifacts/silero_probe/numbers.wav`
- Generate: `uv.lock`

**Interfaces:**
- Consumes: official model URL `https://models.silero.ai/models/tts/ru/v5_5_ru.pt` and interactive macOS audio playback through `afplay`.
- Produces: `choose_max_text_chars(l_max: int) -> int`, `sha256_file(path: Path) -> str`, committed manifest fields consumed by Task 8, and blocking runtime evidence used by every later task.

- [ ] **Step 1: Create the package metadata and ignore rules**

Create `.python-version`:

```text
3.12
```

Create `.gitignore`:

```gitignore
.venv/
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
.coverage
htmlcov/
artifacts/
*.mp3
*.mp3.part
*.pt
.idea/
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["uv_build>=0.11.28,<0.12.0"]
build-backend = "uv_build"

[project]
name = "pytts"
version = "0.1.0"
description = "Local Russian article-to-MP3 CLI"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
  "lameenc",
  "markdown-it-py",
  "platformdirs",
  "pymupdf",
  "pyyaml",
  "rich",
  "torch",
  "typer",
]

[dependency-groups]
dev = [
  "mutagen",
  "pytest",
  "pytest-cov",
  "ruff",
]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
markers = ["silero: requires the real cached Silero model"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

Create `src/pytts/__init__.py`:

```python
"""Local Russian article-to-MP3 conversion."""

__version__ = "0.1.0"
```

Create the bootstrap `README.md` so the package metadata is valid before the full documentation task:

```markdown
# pytts

Local Russian article-to-MP3 CLI. See the approved design and implementation plan under
`docs/superpowers/` while the first release is being built.
```

Run:

```bash
uv lock
uv sync --group dev
```

Expected: uv installs a Python 3.12 environment, resolves a macOS-arm64 PyTorch wheel, and writes `uv.lock` without CUDA indexes.

- [ ] **Step 2: Write failing tests for deterministic probe helpers**

Create `tests/test_silero_probe.py`:

```python
from hashlib import sha256
from pathlib import Path

import pytest

from pytts.silero_probe import choose_max_text_chars, sha256_file


def test_sha256_file_reads_binary_content(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"silero-model")

    assert sha256_file(model) == sha256(b"silero-model").hexdigest()


@pytest.mark.parametrize(
    ("l_max", "expected"),
    [(1000, 800), (900, 720), (4096, 800)],
)
def test_choose_max_text_chars_keeps_twenty_percent_margin(
    l_max: int, expected: int
) -> None:
    assert choose_max_text_chars(l_max) == expected


def test_choose_max_text_chars_rejects_unusable_model_limit() -> None:
    with pytest.raises(ValueError, match="at least 64"):
        choose_max_text_chars(63)
```

- [ ] **Step 3: Run the helper tests and confirm red**

Run:

```bash
uv run pytest tests/test_silero_probe.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'pytts.silero_probe'`.

- [ ] **Step 4: Implement the deterministic helpers**

Create the initial `src/pytts/silero_probe.py`:

```python
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def choose_max_text_chars(l_max: int) -> int:
    safe_limit = min(800, int(l_max * 0.8))
    if safe_limit < 64:
        raise ValueError("Silero safe clean-text limit must be at least 64 characters")
    return safe_limit
```

- [ ] **Step 5: Run focused tests and confirm green**

Run:

```bash
uv run pytest tests/test_silero_probe.py -v
uv run ruff check src/pytts/silero_probe.py tests/test_silero_probe.py
```

Expected: all four tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 6: Extend the probe module with the real-model gate**

Replace `src/pytts/silero_probe.py` with the complete probe implementation below. It keeps the tested helpers and writes the manifest and report only after every runtime check and the human number-audio observation succeed.

```python
from __future__ import annotations

import html
import subprocess
import sys
import urllib.request
import wave
from array import array
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import platformdirs
import torch
import yaml

MODEL_ID = "v5_5_ru"
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
RATES = ("x-slow", "slow", "normal", "fast", "x-fast")
SSML_RATES = {"normal": "medium", **{rate: rate for rate in RATES if rate != "normal"}}
SAMPLE_RATES = (8000, 24000, 48000)
SEARCH_CEILING = 16384


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model_path: Path
    digest: str
    speakers: tuple[str, ...]
    l_max: int
    max_text_chars: int
    number_observation: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def choose_max_text_chars(l_max: int) -> int:
    safe_limit = min(800, int(l_max * 0.8))
    if safe_limit < 64:
        raise ValueError("Silero safe clean-text limit must be at least 64 characters")
    return safe_limit


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download")
    partial.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _load_model(path: Path) -> Any:
    importer = torch.package.PackageImporter(str(path))
    model = importer.load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    return model


def _russian_text(length: int) -> str:
    seed = "Это проверочный русский текст для измерения безопасной длины синтеза. "
    return (seed * (length // len(seed) + 1))[:length]


def _ssml(text: str, rate: str) -> str:
    return (
        "<speak><prosody rate=\""
        + SSML_RATES[rate]
        + "\">"
        + html.escape(text, quote=False)
        + "</prosody></speak>"
    )


def _supports_length(model: Any, speaker: str, rate: str, length: int) -> bool:
    try:
        audio = model.apply_tts(
            ssml_text=_ssml(_russian_text(length), rate),
            speaker=speaker,
            sample_rate=48000,
        )
    except Exception:
        return False
    return isinstance(audio, torch.Tensor) and audio.ndim == 1 and audio.numel() > 0


def _largest_supported_length(check: Callable[[int], bool]) -> int:
    low = 64
    if not check(low):
        raise RuntimeError("Silero rejects the minimum 64-character SSML probe")
    high = 128
    while high <= SEARCH_CEILING and check(high):
        low = high
        high *= 2
    if high > SEARCH_CEILING:
        raise RuntimeError("Silero length limit was not found below the probe ceiling")
    while high - low > 1:
        middle = (low + high) // 2
        if check(middle):
            low = middle
        else:
            high = middle
    return low


def _pcm16(audio: torch.Tensor) -> bytes:
    samples = (
        audio.detach().cpu().flatten().clamp(-1.0, 1.0).mul(32767).round().to(torch.int16)
    )
    pcm = array("h", samples.tolist())
    if pcm.itemsize != 2:
        raise RuntimeError("Unexpected signed-short width")
    return pcm.tobytes()


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(_pcm16(audio))


def _write_outputs(project_root: Path, result: ProbeResult) -> None:
    manifest_path = project_root / "src/pytts/model_manifest.yaml"
    manifest = {
        "version": 1,
        "model": {
            "model_id": MODEL_ID,
            "url": MODEL_URL,
            "sha256": result.digest,
            "preferred_voice": "xenia",
            "sample_rate": 48000,
            "max_text_chars": result.max_text_chars,
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report_path = (
        project_root
        / "docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Silero v5_5_ru Runtime Verification",
                "",
                f"- Python: `{sys.version.split()[0]}`",
                f"- PyTorch: `{torch.__version__}`",
                f"- Model ID: `{MODEL_ID}`",
                f"- URL: `{MODEL_URL}`",
                f"- Model bytes: `{result.model_path.stat().st_size}`",
                f"- SHA-256: `{result.digest}`",
                f"- Runtime speakers: `{', '.join(result.speakers)}`",
                "- PackageImporter: `passed`",
                "- apply_tts(text=...): `passed`",
                "- apply_tts(ssml_text=...) rates: `x-slow, slow, medium, fast, x-fast`",
                "- Sample rates: `8000, 24000, 48000`",
                f"- Verified L_max: `{result.l_max}` clean characters",
                f"- MAX_TEXT_CHARS: `{result.max_text_chars}`",
                f"- Number pronunciation observation: {result.number_observation}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_probe(project_root: Path) -> ProbeResult:
    model_path = (
        Path(platformdirs.user_cache_dir("pytts")) / "models" / f"{MODEL_ID}.pt"
    )
    if not model_path.exists():
        _download(MODEL_URL, model_path)
    digest = sha256_file(model_path)
    model = _load_model(model_path)

    speakers = tuple(str(value) for value in getattr(model, "speakers", ()))
    if not speakers:
        raise RuntimeError("v5_5_ru exposes no model.speakers")
    speaker = "xenia" if "xenia" in speakers else speakers[0]

    plain = model.apply_tts(
        text="Это проверка обычного текстового интерфейса.",
        speaker=speaker,
        sample_rate=48000,
    )
    if plain.ndim != 1 or plain.numel() == 0:
        raise RuntimeError("apply_tts(text=...) did not return non-empty mono PCM")

    for sample_rate in SAMPLE_RATES:
        audio = model.apply_tts(
            text="Проверка частоты дискретизации.",
            speaker=speaker,
            sample_rate=sample_rate,
        )
        if audio.ndim != 1 or audio.numel() == 0:
            raise RuntimeError(f"Invalid PCM at {sample_rate} Hz")

    per_rate_limits = [
        _largest_supported_length(
            lambda length, current_rate=rate: _supports_length(
                model, speaker, current_rate, length
            )
        )
        for rate in RATES
    ]
    l_max = min(per_rate_limits)
    max_text_chars = choose_max_text_chars(l_max)

    numbers = (
        "19 июля 2026 года показатель вырос на 15%, сумма составила 1 500 ₽, "
        "а диапазон оказался от 5 до 7 единиц."
    )
    numbers_audio = model.apply_tts(
        text=numbers,
        speaker=speaker,
        sample_rate=48000,
    )
    wav_path = project_root / "artifacts/silero_probe/numbers.wav"
    _write_wav(wav_path, numbers_audio, 48000)
    subprocess.run(["afplay", str(wav_path)], check=True)
    observation = input("Опишите фактическое произношение чисел одной строкой: ").strip()
    if not observation:
        raise RuntimeError("Number pronunciation observation must not be empty")

    result = ProbeResult(
        model_path=model_path,
        digest=digest,
        speakers=speakers,
        l_max=l_max,
        max_text_chars=max_text_chars,
        number_observation=observation,
    )
    _write_outputs(project_root, result)
    return result
```

Create `scripts/probe_silero.py`:

```python
from pathlib import Path

from pytts.silero_probe import run_probe


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = run_probe(project_root)
    print(f"speakers={','.join(result.speakers)}")
    print(f"sha256={result.digest}")
    print(f"max_text_chars={result.max_text_chars}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Re-run deterministic tests before the network probe**

Run:

```bash
uv run pytest tests/test_silero_probe.py -v
uv run ruff check src/pytts/silero_probe.py scripts/probe_silero.py
```

Expected: helper tests pass and Ruff reports no violations.

- [ ] **Step 8: Run the real Phase 0 probe and complete the listening gate**

Run in an interactive terminal:

```bash
uv run python scripts/probe_silero.py
```

Expected sequence:

1. The official model downloads once into the platformdirs cache.
2. Plain text, all rates, all sample rates, and the length search complete without an exception.
3. macOS plays `artifacts/silero_probe/numbers.wav`.
4. Enter one concrete sentence describing exactly how the year, percent, amount, currency, and range sounded.
5. The command prints non-empty `speakers=`, a 64-character `sha256=`, and a positive `max_text_chars=` not greater than 800.

Stop the entire implementation if any expected condition fails. Update the design and obtain approval instead of weakening the probe.

- [ ] **Step 9: Verify generated evidence and manifest consistency**

Run:

```bash
uv run python -c 'from pathlib import Path; import yaml; m=yaml.safe_load(Path("src/pytts/model_manifest.yaml").read_text())["model"]; r=Path("docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md").read_text(); assert all(str(m[k]) in r for k in ("model_id", "url", "sha256", "sample_rate", "max_text_chars"))'
git diff --check
```

Expected: the assertion succeeds, the report contains the actual runtime speakers and manual number observation, and `git diff --check` prints nothing.

- [ ] **Step 10: Commit the Phase 0 gate**

```bash
git add .python-version .gitignore pyproject.toml uv.lock README.md src/pytts/__init__.py src/pytts/silero_probe.py src/pytts/model_manifest.yaml scripts/probe_silero.py tests/test_silero_probe.py docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md
git commit -m "chore: verify silero v5_5 runtime contract"
```

Expected: the commit contains no `.pt` or `.wav` file.

---

### Task 2: Define Domain Objects, Error Contracts, and Strict YAML Config

**Files:**
- Create: `src/pytts/domain.py`
- Create: `src/pytts/errors.py`
- Create: `src/pytts/config.py`
- Create: `pytts.yaml`
- Create: `tests/test_domain.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: Python 3.12 and project root containing `pyproject.toml`.
- Produces: `BlockKind`, `SpeechRate`, `TextBlock`, `Article`, `CleaningResult`, `SpeechChunk`, `ConversionRequest`, `ConversionResult`; `PyTTSError` subclasses with fixed exit codes; `AppConfig`; `find_project_root()`; `load_config()`.

- [ ] **Step 1: Write failing domain and config tests**

Create `tests/test_domain.py`:

```python
from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, SpeechRate, TextBlock
from pytts.errors import ConfigError, InputError, ModelError, SynthesisError, UsageError


def test_text_block_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        TextBlock(BlockKind.PARAGRAPH, "   ")


def test_article_requires_at_least_one_block() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        Article(Path("article.md"), ())


def test_normal_rate_maps_to_medium_ssml() -> None:
    assert SpeechRate.NORMAL.ssml_value == "medium"


def test_stable_application_exit_codes() -> None:
    assert UsageError.exit_code == 2
    assert InputError.exit_code == ConfigError.exit_code == 3
    assert ModelError.exit_code == 4
    assert SynthesisError.exit_code == 5
```

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from pytts.config import find_project_root, load_config
from pytts.errors import ConfigError


def test_find_project_root_walks_to_pyproject(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    module = root / "src/pytts/config.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='pytts'\n", encoding="utf-8")

    assert find_project_root(module) == root


def test_explicit_config_replaces_root_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pytts.yaml").write_text(
        'version: 1\nabbreviations:\n  "г-н": "господин"\n', encoding="utf-8"
    )
    explicit = tmp_path / "custom.yaml"
    explicit.write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n', encoding="utf-8"
    )

    config = load_config(explicit, root)

    assert dict(config.abbreviations) == {"ув.": "уважаемый"}


@pytest.mark.parametrize(
    "content",
    [
        "version: 2\nabbreviations: {}\n",
        "version: true\nabbreviations: {}\n",
        "version: 1\nunknown: true\nabbreviations: {}\n",
        "version: 1\nabbreviations: []\n",
        'version: 1\nabbreviations:\n  "УВ.": "один"\n  "ув.": "два"\n',
        'version: 1\nabbreviations:\n  "": "пустой"\n',
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path, tmp_path)


def test_missing_root_config_returns_empty_mapping(tmp_path: Path) -> None:
    assert dict(load_config(None, tmp_path).abbreviations) == {}


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml", tmp_path)
```

- [ ] **Step 2: Run focused tests and confirm red**

Run:

```bash
uv run pytest tests/test_domain.py tests/test_config.py -v
```

Expected: collection fails because `pytts.domain`, `pytts.config`, and `pytts.errors` do not exist.

- [ ] **Step 3: Implement shared exceptions**

Create `src/pytts/errors.py`:

```python
class PyTTSError(Exception):
    exit_code = 1


class UsageError(PyTTSError):
    exit_code = 2


class InputError(PyTTSError):
    exit_code = 3


class ConfigError(PyTTSError):
    exit_code = 3


class ModelError(PyTTSError):
    exit_code = 4


class ModelIntegrityError(ModelError):
    """The downloaded or cached model does not match the committed manifest."""


class SynthesisError(PyTTSError):
    exit_code = 5
```

- [ ] **Step 4: Implement immutable domain objects**

Create `src/pytts/domain.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    QUOTE = "quote"


class SpeechRate(StrEnum):
    X_SLOW = "x-slow"
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    X_FAST = "x-fast"

    @property
    def ssml_value(self) -> str:
        return "medium" if self is SpeechRate.NORMAL else self.value


@dataclass(frozen=True, slots=True)
class TextBlock:
    kind: BlockKind
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("TextBlock text must not be blank")


@dataclass(frozen=True, slots=True)
class Article:
    source: Path
    blocks: tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("Article requires at least one block")


@dataclass(frozen=True, slots=True)
class CleaningResult:
    article: Article
    warning: str | None


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    ssml_text: str
    pause_after_ms: int


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    input_path: Path
    output_path: Path | None = None
    config_path: Path | None = None
    voice: str | None = None
    rate: SpeechRate = SpeechRate.NORMAL
    force: bool = False


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_path: Path
    voice: str
    chunk_count: int
```

- [ ] **Step 5: Implement strict configuration loading**

Create `src/pytts/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from pytts.errors import ConfigError


@dataclass(frozen=True, slots=True)
class AppConfig:
    abbreviations: Mapping[str, str]


def find_project_root(module_path: Path) -> Path:
    current = module_path.resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"Could not find pyproject.toml above {module_path}")


def _validate_abbreviations(value: object, path: Path) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: abbreviations must be a mapping")
    result: dict[str, str] = {}
    folded: set[str] = set()
    for key, replacement in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{path}: abbreviation keys must be non-empty strings")
        if not isinstance(replacement, str) or not replacement:
            raise ConfigError(f"{path}: replacements must be non-empty strings")
        normalized = key.casefold()
        if normalized in folded:
            raise ConfigError(f"{path}: duplicate abbreviation ignoring case: {key}")
        folded.add(normalized)
        result[key] = replacement
    return MappingProxyType(result)


def load_config(explicit_path: Path | None, project_root: Path) -> AppConfig:
    path = explicit_path if explicit_path is not None else project_root / "pytts.yaml"
    if not path.exists():
        if explicit_path is not None:
            raise ConfigError(f"Explicit config does not exist: {path}")
        return AppConfig(MappingProxyType({}))
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(f"Could not read YAML config {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    unknown = set(loaded) - {"version", "abbreviations"}
    if unknown:
        raise ConfigError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
    if type(loaded.get("version")) is not int or loaded["version"] != 1:
        raise ConfigError(f"{path}: version must be 1")
    return AppConfig(_validate_abbreviations(loaded.get("abbreviations"), path))
```

Create root `pytts.yaml`:

```yaml
version: 1

abbreviations:
  "г-н": "господин"
  "г-жа": "госпожа"
  "ув.": "уважаемый"
  "т.е.": "то есть"
```

- [ ] **Step 6: Run domain and config tests**

Run:

```bash
uv run pytest tests/test_domain.py tests/test_config.py -v
uv run ruff check src/pytts/domain.py src/pytts/errors.py src/pytts/config.py tests/test_domain.py tests/test_config.py
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 7: Commit domain and config contracts**

```bash
git add src/pytts/domain.py src/pytts/errors.py src/pytts/config.py pytts.yaml tests/test_domain.py tests/test_config.py
git commit -m "feat: add domain and config contracts"
```

---

### Task 3: Parse Markdown into Semantic Article Blocks

**Files:**
- Create: `src/pytts/readers/__init__.py`
- Create: `src/pytts/readers/markdown.py`
- Create: `tests/readers/__init__.py`
- Create: `tests/readers/test_markdown.py`

**Interfaces:**
- `MarkdownReader.read(path: Path) -> Article`
- Consumes the `Article`, `TextBlock`, and `BlockKind` types from Task 2.
- Produces ordered heading, paragraph, list-item, and quote blocks for Task 5.

- [ ] **Step 1: Write failing preservation and omission tests**

Create `tests/readers/test_markdown.py`:

````python
from pathlib import Path

from pytts.domain import BlockKind
from pytts.readers.markdown import MarkdownReader


def _write(tmp_path: Path, text: str) -> Path:
    source = tmp_path / "article.md"
    source.write_text(text, encoding="utf-8")
    return source


def test_preserves_article_structure_and_link_label(tmp_path: Path) -> None:
    source = _write(
        tmp_path,
        """# Заголовок

Первый [абзац](https://example.com).

- Первый пункт
- Второй пункт

> Важная цитата.
""",
    )

    article = MarkdownReader().read(source)

    assert [(block.kind, block.text) for block in article.blocks] == [
        (BlockKind.HEADING, "Заголовок"),
        (BlockKind.PARAGRAPH, "Первый абзац."),
        (BlockKind.LIST_ITEM, "Первый пункт"),
        (BlockKind.LIST_ITEM, "Второй пункт"),
        (BlockKind.QUOTE, "Важная цитата."),
    ]


def test_drops_non_article_markdown_constructs(tmp_path: Path) -> None:
    source = _write(
        tmp_path,
        """---
title: Метаданные
---
# Статья

Текст ![картинка](image.png) после картинки.[^1]

```python
print("не читать")
```

    и этот блок кода тоже не читать

| Колонка | Значение |
| --- | --- |
| Код | 42 |

<aside>не читать</aside>

[^1]: Сноска, которую не читаем.
""",
    )

    article = MarkdownReader().read(source)

    assert [(block.kind, block.text) for block in article.blocks] == [
        (BlockKind.HEADING, "Статья"),
        (BlockKind.PARAGRAPH, "Текст после картинки."),
    ]


def test_decodes_html_entities_but_ignores_inline_code(tmp_path: Path) -> None:
    source = _write(tmp_path, "Текст &amp; ещё `rm -rf example`.\n")

    article = MarkdownReader().read(source)

    assert article.blocks[0].text == "Текст & ещё ."
````

- [ ] **Step 2: Run the Markdown tests and confirm red**

```bash
uv run pytest tests/readers/test_markdown.py -v
```

Expected: import fails because `pytts.readers.markdown` does not exist.

- [ ] **Step 3: Implement token-based Markdown reading**

Create empty `src/pytts/readers/__init__.py` and `tests/readers/__init__.py`, then create
`src/pytts/readers/markdown.py` with these rules:

```python
from __future__ import annotations

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_FOOTNOTE_DEFINITION = re.compile(r"(?m)^\[\^[^]]+\]:.*(?:\n(?: {2,}|\t).*)*")
_FOOTNOTE_REFERENCE = re.compile(r"\[\^[^]]+\]")
_TABLE_DELIMITER = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _drop_pipe_tables(text: str) -> str:
    lines = text.splitlines()
    dropped: set[int] = set()
    for index, line in enumerate(lines):
        if _TABLE_DELIMITER.match(line):
            dropped.update({index - 1, index})
            cursor = index + 1
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                dropped.add(cursor)
                cursor += 1
    return "\n".join(line for index, line in enumerate(lines) if index not in dropped)


def _prepare(text: str) -> str:
    text = _FRONT_MATTER.sub("", text)
    text = _FOOTNOTE_DEFINITION.sub("", text)
    return _drop_pipe_tables(text)


def _inline_text(token: Token) -> str:
    pieces: list[str] = []
    for child in token.children or ():
        if child.type == "text":
            pieces.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append(" ")
        elif child.type in {"code_inline", "html_inline", "image"}:
            continue
    text = html.unescape("".join(pieces))
    text = _FOOTNOTE_REFERENCE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class MarkdownReader:
    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": True})

    def read(self, path: Path) -> Article:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise InputError(f"Could not read Markdown {path}: {error}") from error

        tokens = self._parser.parse(_prepare(text))
        list_depth = 0
        quote_depth = 0
        blocks: list[TextBlock] = []
        for index, token in enumerate(tokens):
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                list_depth += 1
            elif token.type in {"bullet_list_close", "ordered_list_close"}:
                list_depth -= 1
            elif token.type == "blockquote_open":
                quote_depth += 1
            elif token.type == "blockquote_close":
                quote_depth -= 1
            elif token.type in {"heading_open", "paragraph_open"}:
                inline = tokens[index + 1]
                if inline.type != "inline":
                    continue
                content = _inline_text(inline)
                if not content:
                    continue
                if token.type == "heading_open":
                    kind = BlockKind.HEADING
                elif list_depth:
                    kind = BlockKind.LIST_ITEM
                elif quote_depth:
                    kind = BlockKind.QUOTE
                else:
                    kind = BlockKind.PARAGRAPH
                blocks.append(TextBlock(kind=kind, text=content))

        if not blocks:
            raise InputError(f"Markdown contains no readable article text: {path}")
        return Article(source=path, blocks=tuple(blocks))
```

- [ ] **Step 4: Run the focused tests and Ruff**

```bash
uv run pytest tests/readers/test_markdown.py -v
uv run ruff check src/pytts/readers/markdown.py tests/readers/test_markdown.py
```

Expected: all three tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit the Markdown reader**

```bash
git add src/pytts/readers tests/readers
git commit -m "feat: parse markdown article structure"
```

---

### Task 4: Extract Text-Layer PDFs with Article Heuristics

**Files:**
- Create: `src/pytts/readers/pdf.py`
- Create: `tests/readers/test_pdf.py`

**Interfaces:**
- `PDFReader.read(path: Path) -> Article`
- Internal immutable `_RawBlock(page, y0, y1, font_size, text)` captures just enough layout data for deterministic filtering.
- Produces the same semantic blocks as Task 3; does not perform OCR.

- [ ] **Step 1: Write failing tests using generated PDF fixtures**

Create `tests/readers/test_pdf.py`. Generate fixtures in the test so no binary PDF enters Git:

```python
from pathlib import Path

import fitz
import pytest

from pytts.domain import BlockKind
from pytts.errors import InputError
from pytts.readers.pdf import PDFReader


def _page(document: fitz.Document, number: int) -> None:
    page = document.new_page(width=595, height=842)
    page.insert_text((50, 25), "Сайт · сохранённая статья", fontsize=8)
    page.insert_text((50, 100), f"Раздел {number}", fontsize=18)
    page.insert_text((50, 150), "Это основной абзац статьи.", fontsize=11)
    page.insert_text((50, 180), "- Пункт списка", fontsize=11)
    page.insert_text((50, 810), str(number), fontsize=8)


def test_removes_repeated_header_and_page_numbers_and_detects_structure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "article.pdf"
    document = fitz.open()
    _page(document, 1)
    _page(document, 2)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert "сохранённая статья" not in " ".join(block.text for block in article.blocks)
    assert [(block.kind, block.text) for block in article.blocks[:3]] == [
        (BlockKind.HEADING, "Раздел 1"),
        (BlockKind.PARAGRAPH, "Это основной абзац статьи."),
        (BlockKind.LIST_ITEM, "Пункт списка"),
    ]
    assert all(block.text not in {"1", "2"} for block in article.blocks)


def test_joins_line_end_hyphen_only_for_lowercase_word(tmp_path: Path) -> None:
    source = tmp_path / "hyphen.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 80), "Долго-\nжданное событие. Санкт-\nПетербург.", fontsize=11)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert article.blocks[0].text == "Долгожданное событие. Санкт- Петербург."


def test_rejects_pdf_without_text_layer(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    with pytest.raises(InputError, match="text layer"):
        PDFReader().read(source)


def test_rejects_password_protected_pdf(tmp_path: Path) -> None:
    source = tmp_path / "protected.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 100), "Секретный текст", fontsize=11)
    document.save(
        source,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    document.close()

    with pytest.raises(InputError, match="password"):
        PDFReader().read(source)


def test_uniform_font_heading_degrades_to_paragraph(tmp_path: Path) -> None:
    source = tmp_path / "uniform.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 100), "Возможный заголовок", fontsize=11)
    page.insert_text((50, 150), "Основной текст статьи.", fontsize=11)
    document.save(source)
    document.close()

    article = PDFReader().read(source)

    assert [block.kind for block in article.blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
    ]
```

- [ ] **Step 2: Run the PDF tests and confirm red**

```bash
uv run pytest tests/readers/test_pdf.py -v
```

Expected: import fails because `pytts.readers.pdf` does not exist.

- [ ] **Step 3: Implement extraction and filtering helpers**

Create `src/pytts/readers/pdf.py` with these concrete rules:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError

_PAGE_NUMBER = re.compile(r"^(?:стр\.?\s*)?\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:[-–—•●▪◦]|\d+[.)])\s+")
_LOWERCASE_HYPHEN = re.compile(r"(?<=[а-яё])-\s*\n\s*(?=[а-яё])")


@dataclass(frozen=True, slots=True)
class _RawBlock:
    page: int
    page_height: float
    y0: float
    y1: float
    font_size: float
    text: str

    @property
    def band(self) -> str:
        if self.y1 <= self.page_height * 0.12:
            return "header"
        if self.y0 >= self.page_height * 0.88:
            return "footer"
        return "body"


def _normalize(text: str) -> str:
    text = _LOWERCASE_HYPHEN.sub("", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _signature(block: _RawBlock) -> tuple[str, str]:
    normalized = re.sub(r"\d+", "#", block.text.casefold())
    return block.band, normalized


def _extract(document: fitz.Document) -> list[_RawBlock]:
    result: list[_RawBlock] = []
    for page_index, page in enumerate(document):
        layout = page.get_text("dict", sort=True)
        for block in layout.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            spans = [span for line in lines for span in line.get("spans", [])]
            text = "\n".join(
                "".join(span.get("text", "") for span in line.get("spans", []))
                for line in lines
            )
            text = _normalize(text)
            if not text or not spans:
                continue
            font_size = min(float(span.get("size", 0.0)) for span in spans)
            _, y0, _, y1 = block["bbox"]
            result.append(
                _RawBlock(page_index, page.rect.height, y0, y1, font_size, text)
            )
    return result


def _body_font_size(blocks: list[_RawBlock]) -> float:
    candidates = [block for block in blocks if block.band == "body"] or blocks
    weighted = sorted((block.font_size, max(1, len(block.text))) for block in candidates)
    midpoint = sum(weight for _, weight in weighted) / 2
    cumulative = 0
    for font_size, weight in weighted:
        cumulative += weight
        if cumulative >= midpoint:
            return font_size
    raise RuntimeError("Could not determine PDF body font size")


class PDFReader:
    def read(self, path: Path) -> Article:
        try:
            with fitz.open(path) as document:
                if document.needs_pass:
                    raise InputError(f"Encrypted PDF cannot be read without a password: {path}")
                page_count = document.page_count
                blocks = _extract(document)
        except (OSError, RuntimeError, ValueError) as error:
            raise InputError(f"Could not read PDF {path}: {error}") from error

        if not blocks:
            raise InputError(f"PDF has no text layer: {path}")

        signature_pages: dict[tuple[str, str], set[int]] = {}
        for block in blocks:
            if block.band != "body":
                signature_pages.setdefault(_signature(block), set()).add(block.page)
        repeated = {
            signature
            for signature, pages in signature_pages.items()
            if len(pages) >= 2 and len(pages) / page_count >= 0.50
        }
        body_size = _body_font_size(blocks)
        article_blocks: list[TextBlock] = []
        for block in blocks:
            if _signature(block) in repeated or (
                block.band != "body" and _PAGE_NUMBER.fullmatch(block.text)
            ):
                continue
            marker = _LIST_MARKER.match(block.text)
            text = _LIST_MARKER.sub("", block.text, count=1) if marker else block.text
            if marker:
                kind = BlockKind.LIST_ITEM
            elif block.font_size >= body_size * 1.25 and len(text) <= 200:
                kind = BlockKind.HEADING
            else:
                kind = BlockKind.PARAGRAPH
            article_blocks.append(TextBlock(kind=kind, text=text))

        if not article_blocks:
            raise InputError(f"PDF has no readable article text after cleanup: {path}")
        return Article(source=path, blocks=tuple(article_blocks))
```

The implementation intentionally treats headings with a body-sized font as paragraphs. This is the
approved degradation for browser printouts with uniform typography and results in a 350 ms rather
than 700 ms pause.

- [ ] **Step 4: Run PDF tests and inspect the generated fixture text once**

```bash
uv run pytest tests/readers/test_pdf.py -v
uv run ruff check src/pytts/readers/pdf.py tests/readers/test_pdf.py
```

Expected: all five tests pass. If PyMuPDF emits one span instead of embedded newlines in the hyphen
fixture, adjust only the fixture insertion to use `page.insert_textbox`; retain `_normalize` semantics.

- [ ] **Step 5: Commit the PDF reader**

```bash
git add src/pytts/readers/pdf.py tests/readers/test_pdf.py
git commit -m "feat: extract text-layer pdf articles"
```

---

### Task 5: Dispatch Inputs and Clean Article Text

**Files:**
- Create: `src/pytts/readers/input.py`
- Create: `src/pytts/text/__init__.py`
- Create: `src/pytts/text/cleaner.py`
- Create: `tests/readers/test_input.py`
- Create: `tests/text/__init__.py`
- Create: `tests/text/test_cleaner.py`

**Interfaces:**
- `InputReader.read(path: Path) -> Article` is the dispatcher named in the architecture diagram.
- `clean_article(article: Article) -> CleaningResult`
- Cleaning preserves block order and kinds, removes standalone and inline URLs, normalizes whitespace,
  rejects empty results, and produces at most one non-blocking Russian-language warning.

- [ ] **Step 1: Write failing dispatch tests**

Create `tests/readers/test_input.py`:

```python
from pathlib import Path
from unittest.mock import Mock

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
from pytts.readers.input import InputReader


def _article(path: Path) -> Article:
    return Article(path, (TextBlock(BlockKind.PARAGRAPH, "Текст"),))


@pytest.mark.parametrize("suffix", [".md", ".MD"])
def test_dispatches_markdown_case_insensitively(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"article{suffix}"
    path.write_text("Текст", encoding="utf-8")
    markdown = Mock()
    markdown.read.return_value = _article(path)
    pdf = Mock()

    result = InputReader(markdown=markdown, pdf=pdf).read(path)

    assert result.source == path
    markdown.read.assert_called_once_with(path)
    pdf.read.assert_not_called()


def test_rejects_missing_and_unsupported_input(tmp_path: Path) -> None:
    reader = InputReader(markdown=Mock(), pdf=Mock())
    with pytest.raises(InputError, match="does not exist"):
        reader.read(tmp_path / "missing.md")
    unsupported = tmp_path / "article.txt"
    unsupported.write_text("Текст", encoding="utf-8")
    with pytest.raises(InputError, match="Unsupported input extension"):
        reader.read(unsupported)
```

- [ ] **Step 2: Write failing cleaner tests**

Create `tests/text/test_cleaner.py`:

```python
from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, TextBlock
from pytts.errors import InputError
from pytts.text.cleaner import clean_article


def _article(*texts: str) -> Article:
    return Article(
        Path("article.md"),
        tuple(TextBlock(BlockKind.PARAGRAPH, text) for text in texts),
    )


def test_normalizes_whitespace_removes_urls_and_drops_empty_blocks() -> None:
    result = clean_article(
        _article("  Первый\u00a0абзац https://example.com/x?q=1  ", "https://example.org")
    )

    assert [block.text for block in result.article.blocks] == ["Первый абзац"]
    assert result.warning is None


@pytest.mark.parametrize(
    ("text", "warning_fragment"),
    [("12345 — %", "no alphabetic"), ("Mostly English и", "below 70%")],
)
def test_warns_but_returns_text_for_non_russian_content(
    text: str, warning_fragment: str
) -> None:
    result = clean_article(_article(text))

    assert result.article.blocks[0].text
    assert result.warning is not None
    assert warning_fragment in result.warning


def test_rejects_article_empty_after_cleanup() -> None:
    with pytest.raises(InputError, match="no readable text"):
        clean_article(_article("https://example.com"))
```

- [ ] **Step 3: Run both suites and confirm red**

```bash
uv run pytest tests/readers/test_input.py tests/text/test_cleaner.py -v
```

Expected: both imports fail because the dispatcher and cleaner do not exist.

- [ ] **Step 4: Implement the extension dispatcher**

Create `src/pytts/readers/input.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pytts.domain import Article
from pytts.errors import InputError
from pytts.readers.markdown import MarkdownReader
from pytts.readers.pdf import PDFReader


class Reader(Protocol):
    def read(self, path: Path) -> Article: ...


class InputReader:
    def __init__(
        self,
        markdown: Reader | None = None,
        pdf: Reader | None = None,
    ) -> None:
        self._readers = {
            ".md": markdown or MarkdownReader(),
            ".pdf": pdf or PDFReader(),
        }

    def read(self, path: Path) -> Article:
        if not path.is_file():
            raise InputError(f"Input file does not exist: {path}")
        reader = self._readers.get(path.suffix.casefold())
        if reader is None:
            raise InputError(
                f"Unsupported input extension {path.suffix!r}; expected .md or .pdf"
            )
        return reader.read(path)
```

- [ ] **Step 5: Implement deterministic cleanup and the warning threshold**

Create empty `src/pytts/text/__init__.py` and `tests/text/__init__.py`, then create
`src/pytts/text/cleaner.py`:

```python
from __future__ import annotations

import re
import unicodedata

from pytts.domain import Article, CleaningResult, TextBlock
from pytts.errors import InputError

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_SPACE = re.compile(r"\s+")


def _warning(text: str) -> str | None:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "Input contains no alphabetic characters; Russian pronunciation may be unusable"
    cyrillic = sum(
        "а" <= character.casefold() <= "я" or character.casefold() == "ё"
        for character in letters
    )
    ratio = cyrillic / len(letters)
    if ratio < 0.70:
        return f"Cyrillic letters are below 70% ({ratio:.0%}); the Russian model may mispronounce text"
    return None


def clean_article(article: Article) -> CleaningResult:
    blocks: list[TextBlock] = []
    for block in article.blocks:
        normalized = unicodedata.normalize("NFC", block.text)
        text = _SPACE.sub(" ", _URL.sub("", normalized)).strip()
        if text:
            blocks.append(TextBlock(kind=block.kind, text=text))
    if not blocks:
        raise InputError(f"Input contains no readable text after cleanup: {article.source}")
    cleaned = Article(source=article.source, blocks=tuple(blocks))
    return CleaningResult(article=cleaned, warning=_warning(" ".join(b.text for b in blocks)))
```

- [ ] **Step 6: Run focused tests and Ruff**

```bash
uv run pytest tests/readers/test_input.py tests/text/test_cleaner.py -v
uv run ruff check src/pytts/readers/input.py src/pytts/text/cleaner.py tests/readers/test_input.py tests/text/test_cleaner.py
```

Expected: all tests pass and Ruff reports no violations.

- [ ] **Step 7: Commit input dispatch and cleaning**

```bash
git add src/pytts/readers/input.py src/pytts/text tests/readers/test_input.py tests/text
git commit -m "feat: dispatch and clean article inputs"
```

---

### Task 6: Expand User Abbreviations Literally in One Pass

**Files:**
- Create: `src/pytts/text/abbreviations.py`
- Create: `tests/text/test_abbreviations.py`

**Interfaces:**
- `AbbreviationExpander(mapping: Mapping[str, str])`
- `AbbreviationExpander.expand_article(article: Article) -> Article`
- Matching is Unicode case-insensitive, longest key first, bounded by word characters, and
  non-recursive. Only the first letter's case is transferred to the replacement.

- [ ] **Step 1: Write failing behavior tests**

Create `tests/text/test_abbreviations.py`:

```python
from pathlib import Path

from pytts.domain import Article, BlockKind, TextBlock
from pytts.text.abbreviations import AbbreviationExpander


def _expand(text: str, mapping: dict[str, str]) -> str:
    article = Article(Path("article.md"), (TextBlock(BlockKind.PARAGRAPH, text),))
    return AbbreviationExpander(mapping).expand_article(article).blocks[0].text


def test_expands_literal_punctuation_without_a_pause_after_dot() -> None:
    assert _expand("Ув. г-н Иванов, т.е. автор.", {
        "г-н": "господин",
        "ув.": "уважаемый",
        "т.е.": "то есть",
    }) == "Уважаемый господин Иванов, то есть автор."


def test_uses_longest_case_insensitive_match_and_transfers_initial_case() -> None:
    assert _expand("Т.Е. пример, т. пример.", {
        "т.": "товарищ",
        "т.е.": "то есть",
    }) == "То есть пример, товарищ пример."


def test_respects_token_boundaries() -> None:
    assert _expand("авт.е.слово, _т.е._ и т.е. отдельно", {"т.е.": "то есть"}) == (
        "авт.е.слово, _то есть_ и то есть отдельно"
    )


def test_replacements_are_not_scanned_again() -> None:
    assert _expand("а.", {"а.": "б.", "б.": "в"}) == "б."


def test_empty_mapping_returns_equal_article() -> None:
    article = Article(Path("a.md"), (TextBlock(BlockKind.HEADING, "Заголовок"),))
    assert AbbreviationExpander({}).expand_article(article) == article
```

- [ ] **Step 2: Run the tests and confirm red**

```bash
uv run pytest tests/text/test_abbreviations.py -v
```

Expected: import fails because `pytts.text.abbreviations` does not exist.

- [ ] **Step 3: Implement one compiled alternation and one substitution pass**

Create `src/pytts/text/abbreviations.py`:

```python
from __future__ import annotations

import re
from collections.abc import Mapping

from pytts.domain import Article, TextBlock


class AbbreviationExpander:
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

    def _replace(self, match: re.Match[str]) -> str:
        source = match.group(0)
        replacement = self._replacements[source.casefold()]
        if source[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    def expand_article(self, article: Article) -> Article:
        if self._pattern is None:
            return article
        blocks = tuple(
            TextBlock(kind=block.kind, text=self._pattern.sub(self._replace, block.text))
            for block in article.blocks
        )
        return Article(source=article.source, blocks=blocks)
```

- [ ] **Step 4: Run focused tests and Ruff**

```bash
uv run pytest tests/text/test_abbreviations.py -v
uv run ruff check src/pytts/text/abbreviations.py tests/text/test_abbreviations.py
```

Expected: all five tests pass and Ruff reports no violations.

- [ ] **Step 5: Commit abbreviation expansion**

```bash
git add src/pytts/text/abbreviations.py tests/text/test_abbreviations.py
git commit -m "feat: expand configured abbreviations"
```

---

### Task 7: Split Blocks into Bounded SSML Chunks

**Files:**
- Create: `src/pytts/text/chunker.py`
- Create: `tests/text/test_chunker.py`

**Interfaces:**
- `chunk_article(article: Article, rate: SpeechRate, max_text_chars: int) -> tuple[SpeechChunk, ...]`
- The limit applies to clean text before the SSML wrapper. Chunks never split a token; an individual
  token above the model limit is an input error.
- Intermediate pieces of one block get 120 ms; final heading, paragraph/quote, and list-item pieces
  get 700 ms, 350 ms, and 350 ms respectively.

- [ ] **Step 1: Write failing chunking and SSML tests**

Create `tests/text/test_chunker.py`:

```python
from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, SpeechRate, TextBlock
from pytts.errors import InputError
from pytts.text.chunker import chunk_article


def _article(*blocks: tuple[BlockKind, str]) -> Article:
    return Article(
        Path("article.md"),
        tuple(TextBlock(kind, text) for kind, text in blocks),
    )


@pytest.mark.parametrize(
    ("rate", "ssml_rate"),
    [
        (SpeechRate.X_SLOW, "x-slow"),
        (SpeechRate.SLOW, "slow"),
        (SpeechRate.NORMAL, "medium"),
        (SpeechRate.FAST, "fast"),
        (SpeechRate.X_FAST, "x-fast"),
    ],
)
def test_maps_every_rate_and_escapes_xml(rate: SpeechRate, ssml_rate: str) -> None:
    chunks = chunk_article(
        _article((BlockKind.PARAGRAPH, "Пять < шести & семь.")), rate, 100
    )

    assert chunks[0].ssml_text == (
        f'<speak><prosody rate="{ssml_rate}">Пять &lt; шести &amp; семь.</prosody></speak>'
    )
    assert chunks[0].pause_after_ms == 350


def test_splits_on_sentences_then_punctuation_and_sets_intermediate_pause() -> None:
    text = "Первое предложение. Второе предложение, с продолжением. Третье."
    chunks = chunk_article(_article((BlockKind.HEADING, text)), SpeechRate.NORMAL, 25)

    assert len(chunks) >= 3
    assert [chunk.pause_after_ms for chunk in chunks[:-1]] == [120] * (len(chunks) - 1)
    assert chunks[-1].pause_after_ms == 700
    assert all(len(_plain(chunk.ssml_text)) <= 25 for chunk in chunks)


def _plain(ssml: str) -> str:
    return ssml.split(">", 2)[2].rsplit("<", 2)[0]


def test_preserves_numbers_dates_percent_currency_and_range_literally() -> None:
    source = "В 2026 году: 25 %, 1 500 ₽ и диапазон 3–5."
    chunks = chunk_article(_article((BlockKind.PARAGRAPH, source)), SpeechRate.NORMAL, 100)

    assert source in chunks[0].ssml_text


def test_assigns_350_ms_to_paragraph_quote_and_list() -> None:
    article = _article(
        (BlockKind.PARAGRAPH, "Абзац."),
        (BlockKind.QUOTE, "Цитата."),
        (BlockKind.LIST_ITEM, "Пункт."),
    )
    assert [chunk.pause_after_ms for chunk in chunk_article(article, SpeechRate.NORMAL, 100)] == [
        350,
        350,
        350,
    ]


def test_rejects_token_longer_than_model_limit() -> None:
    article = _article((BlockKind.PARAGRAPH, "а" * 65))
    with pytest.raises(InputError, match="single token"):
        chunk_article(article, SpeechRate.NORMAL, 64)
```

- [ ] **Step 2: Run the tests and confirm red**

```bash
uv run pytest tests/text/test_chunker.py -v
```

Expected: import fails because `pytts.text.chunker` does not exist.

- [ ] **Step 3: Implement sentence-aware bounded packing**

Create `src/pytts/text/chunker.py`:

```python
from __future__ import annotations

import html
import re

from pytts.domain import Article, BlockKind, SpeechChunk, SpeechRate
from pytts.errors import InputError

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_PREFERRED_BREAK = re.compile(r"[,;:—–-]\s+|\s+")
_FINAL_PAUSE = {
    BlockKind.HEADING: 700,
    BlockKind.PARAGRAPH: 350,
    BlockKind.QUOTE: 350,
    BlockKind.LIST_ITEM: 350,
}


def _split_long(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        window = remaining[: limit + 1]
        breaks = [match.end() for match in _PREFERRED_BREAK.finditer(window) if match.end() <= limit]
        if not breaks:
            token = remaining.split(maxsplit=1)[0]
            if len(token) > limit:
                raise InputError(
                    f"A single token has {len(token)} characters, above model limit {limit}"
                )
            cut = len(token)
        else:
            cut = breaks[-1]
        piece = remaining[:cut].strip()
        if not piece:
            raise InputError(f"Could not split text at model limit {limit}")
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _units(text: str, limit: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    result: list[str] = []
    for sentence in sentences:
        result.extend(_split_long(sentence, limit) if len(sentence) > limit else [sentence])
    return result


def _pack(text: str, limit: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in _units(text, limit):
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > limit:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _ssml(text: str, rate: SpeechRate) -> str:
    escaped = html.escape(text, quote=False)
    return f'<speak><prosody rate="{rate.ssml_value}">{escaped}</prosody></speak>'


def chunk_article(
    article: Article,
    rate: SpeechRate,
    max_text_chars: int,
) -> tuple[SpeechChunk, ...]:
    if max_text_chars < 64:
        raise ValueError("max_text_chars must be at least 64")
    chunks: list[SpeechChunk] = []
    for block_index, block in enumerate(article.blocks, start=1):
        try:
            pieces = _pack(block.text, max_text_chars)
        except InputError as error:
            raise InputError(
                f"Block {block_index} ({block.kind.value}) cannot be chunked: {error}"
            ) from error
        for index, piece in enumerate(pieces):
            pause = _FINAL_PAUSE[block.kind] if index == len(pieces) - 1 else 120
            chunks.append(SpeechChunk(ssml_text=_ssml(piece, rate), pause_after_ms=pause))
    return tuple(chunks)
```

- [ ] **Step 4: Run focused tests and Ruff**

```bash
uv run pytest tests/text/test_chunker.py -v
uv run ruff check src/pytts/text/chunker.py tests/text/test_chunker.py
```

Expected: all tests pass and every clean-text payload is at or below the probed manifest limit.

- [ ] **Step 5: Commit chunking and SSML**

```bash
git add src/pytts/text/chunker.py tests/text/test_chunker.py
git commit -m "feat: build bounded silero ssml chunks"
```

---

### Task 8: Load the Committed Model Manifest and Maintain an Atomic Cache

**Files:**
- Create: `src/pytts/model_store.py`
- Create: `tests/test_model_store.py`
- Use generated: `src/pytts/model_manifest.yaml`

**Interfaces:**
- `load_model_spec(path: Path | None = None) -> ModelSpec`
- `ModelStore(cache_root: Path | None = None).ensure(spec, validate_package, progress=None) -> Path`
- `DownloadProgress = Callable[[int, int | None], None]`
- The runtime SHA is read from the committed manifest, never duplicated as a Python constant.

- [ ] **Step 1: Write failing manifest and local-download tests**

Create `tests/test_model_store.py`:

```python
from __future__ import annotations

from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from pytts.errors import ModelError
from pytts.model_store import ModelSpec, ModelStore, load_model_spec


class _Handler(BaseHTTPRequestHandler):
    payload = b"package bytes"

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def model_url() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/model.pt"
    finally:
        server.shutdown()
        thread.join()


def _spec(url: str, digest: str | None = None) -> ModelSpec:
    return ModelSpec(
        model_id="v5_5_ru",
        url=url,
        sha256=digest or sha256(_Handler.payload).hexdigest(),
        preferred_voice="xenia",
        sample_rate=48000,
        max_text_chars=800,
    )


def test_load_model_spec_is_strict(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """version: 1
model:
  model_id: v5_5_ru
  url: https://example.test/model.pt
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  preferred_voice: xenia
  sample_rate: 48000
  max_text_chars: 800
""",
        encoding="utf-8",
    )

    assert load_model_spec(manifest).model_id == "v5_5_ru"


def test_downloads_validates_reports_progress_and_reuses_cache(
    tmp_path: Path, model_url: str
) -> None:
    seen: list[tuple[int, int | None]] = []
    validations: list[Path] = []
    store = ModelStore(cache_root=tmp_path)

    first = store.ensure(_spec(model_url), validations.append, seen.append)
    second = store.ensure(_spec(model_url), validations.append, seen.append)

    assert first == second == tmp_path / "models/v5_5_ru.pt"
    assert first.read_bytes() == _Handler.payload
    assert not first.with_name(first.name + ".download").exists()
    assert validations == [first.with_name(first.name + ".download"), first]
    assert seen[-1] == (len(_Handler.payload), len(_Handler.payload))


def test_hash_mismatch_leaves_no_partial_and_explains_manifest_update(
    tmp_path: Path, model_url: str
) -> None:
    store = ModelStore(cache_root=tmp_path)
    wrong = "0" * 64

    with pytest.raises(ModelError, match="scripts/probe_silero.py") as captured:
        store.ensure(_spec(model_url, wrong), lambda path: None)

    assert f"expected {wrong}" in str(captured.value)
    assert sha256(_Handler.payload).hexdigest() in str(captured.value)
    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
    assert not (tmp_path / "models/v5_5_ru.pt").exists()


def test_package_validation_failure_is_atomic(tmp_path: Path, model_url: str) -> None:
    store = ModelStore(cache_root=tmp_path)

    def reject(path: Path) -> None:
        raise ValueError("not a Torch package")

    with pytest.raises(ModelError, match="not a Torch package"):
        store.ensure(_spec(model_url), reject)

    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
    assert not (tmp_path / "models/v5_5_ru.pt").exists()


def test_interrupted_download_removes_partial(
    tmp_path: Path, model_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(url: str, path: Path, progress: object) -> None:
        path.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(ModelStore, "_download", staticmethod(interrupt))
    with pytest.raises(KeyboardInterrupt):
        ModelStore(cache_root=tmp_path).ensure(_spec(model_url), lambda path: None)

    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
```

- [ ] **Step 2: Run tests and confirm red**

```bash
uv run pytest tests/test_model_store.py -v
```

Expected: import fails because `pytts.model_store` does not exist.

- [ ] **Step 3: Implement strict manifest loading and atomic download**

Create `src/pytts/model_store.py`:

```python
from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import platformdirs
import yaml

from pytts.errors import ModelError, ModelIntegrityError
from pytts.silero_probe import sha256_file

DownloadProgress = Callable[[int, int | None], None]
PackageValidator = Callable[[Path], None]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_FIELDS = {
    "model_id", "url", "sha256", "preferred_voice", "sample_rate", "max_text_chars"
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    url: str
    sha256: str
    preferred_voice: str
    sample_rate: int
    max_text_chars: int


def load_model_spec(path: Path | None = None) -> ModelSpec:
    source = path or Path(__file__).with_name("model_manifest.yaml")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ModelError(f"Could not read model manifest {source}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"version", "model"}:
        raise ModelError(f"{source}: expected only version and model fields")
    model = payload.get("model")
    if (
        type(payload.get("version")) is not int
        or payload["version"] != 1
        or not isinstance(model, dict)
        or set(model) != _MODEL_FIELDS
    ):
        raise ModelError(f"{source}: invalid version 1 model schema")
    try:
        spec = ModelSpec(**model)
    except TypeError as error:
        raise ModelError(f"{source}: invalid model values: {error}") from error
    if not all(
        isinstance(value, str) and value
        for value in (spec.model_id, spec.url, spec.preferred_voice)
    ):
        raise ModelError(f"{source}: string fields must not be empty")
    if not isinstance(spec.sha256, str) or not _SHA256.fullmatch(spec.sha256):
        raise ModelError(f"{source}: sha256 must be 64 lowercase hexadecimal characters")
    if (
        type(spec.sample_rate) is not int
        or type(spec.max_text_chars) is not int
        or spec.sample_rate != 48000
        or spec.max_text_chars < 64
        or spec.max_text_chars > 800
    ):
        raise ModelError(f"{source}: unsupported sample rate or text limit")
    return spec


class ModelStore:
    def __init__(self, cache_root: Path | None = None) -> None:
        self._root = cache_root or Path(platformdirs.user_cache_dir("pytts"))

    @staticmethod
    def _verify(path: Path, spec: ModelSpec) -> None:
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise ModelIntegrityError(
                f"Model SHA-256 mismatch at {path} for {spec.url}: expected {spec.sha256}, "
                f"actual {actual}. The upstream file may have been republished at the same URL. "
                "The expected value comes from committed src/pytts/model_manifest.yaml. "
                "For a deliberate upstream model update, rerun scripts/probe_silero.py and "
                "review the generated verification report before committing the new manifest."
            )

    @staticmethod
    def _download(url: str, path: Path, progress: DownloadProgress | None) -> None:
        with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as target:
            header = response.headers.get("Content-Length")
            total = int(header) if header is not None else None
            completed = 0
            while data := response.read(1024 * 1024):
                target.write(data)
                completed += len(data)
                if progress is not None:
                    progress(completed, total)

    def ensure(
        self,
        spec: ModelSpec,
        validate_package: PackageValidator,
        progress: DownloadProgress | None = None,
    ) -> Path:
        models = self._root / "models"
        try:
            models.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ModelError(f"Could not create model cache directory {models}: {error}") from error
        target = models / f"{spec.model_id}.pt"
        partial = target.with_name(target.name + ".download")
        if target.exists():
            try:
                self._verify(target, spec)
                validate_package(target)
            except ModelError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                raise ModelError(f"Could not validate cached model {target}: {error}") from error
            return target
        partial.unlink(missing_ok=True)
        try:
            self._download(spec.url, partial, progress)
            self._verify(partial, spec)
            validate_package(partial)
            os.replace(partial, target)
        except ModelError:
            partial.unlink(missing_ok=True)
            raise
        except (KeyboardInterrupt, SystemExit):
            partial.unlink(missing_ok=True)
            raise
        except Exception as error:
            partial.unlink(missing_ok=True)
            raise ModelError(f"Could not download or validate model {spec.url}: {error}") from error
        return target
```

The cache path therefore resolves on macOS to
`~/Library/Caches/pytts/models/v5_5_ru.pt`. The committed runtime value lives only in YAML and must
match the Phase 0 report.

- [ ] **Step 4: Run focused tests and Ruff**

```bash
uv run pytest tests/test_model_store.py -v
uv run ruff check src/pytts/model_store.py tests/test_model_store.py
```

Expected: all tests pass; a bad hash leaves neither a final file nor `.download` partial.

- [ ] **Step 5: Commit model manifest consumption and cache behavior**

```bash
git add src/pytts/model_store.py tests/test_model_store.py
git commit -m "feat: verify and cache silero model"
```

---

### Task 9: Load Silero and Resolve Voices from Runtime Truth

**Files:**
- Create: `src/pytts/tts.py`
- Create: `tests/test_tts.py`

**Interfaces:**
- `SileroRuntime.load(store, spec, download_progress=None) -> SileroRuntime`
- `SileroRuntime.speakers -> tuple[str, ...]`
- `SileroRuntime.resolve_voice(requested: str | None) -> VoiceSelection`
- `SileroRuntime.synthesize(chunk: SpeechChunk, voice: str) -> torch.Tensor`
- No hardcoded allowed-speaker list. `xenia` is only a manifest preference.

- [ ] **Step 1: Write failing voice and synthesis tests**

Create `tests/test_tts.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pytts.domain import SpeechChunk
from pytts.errors import ModelError, SynthesisError
from pytts.model_store import ModelSpec
from pytts.tts import SileroRuntime


def _spec(preferred: str = "xenia") -> ModelSpec:
    return ModelSpec("v5_5_ru", "https://example.test/model.pt", "a" * 64, preferred, 48000, 800)


class FakeModel:
    def __init__(self, speakers: list[str]) -> None:
        self.speakers = speakers
        self.device = None
        self.calls: list[dict[str, object]] = []

    def to(self, device: torch.device) -> "FakeModel":
        self.device = device
        return self

    def apply_tts(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.tensor([0.0, 0.25, -0.25])


class FailingModel(FakeModel):
    def apply_tts(self, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("boom")


def test_prefers_manifest_voice_when_runtime_has_it() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "xenia"]), _spec())
    assert runtime.resolve_voice(None).name == "xenia"
    assert runtime.resolve_voice(None).warning is None


def test_falls_back_to_first_runtime_voice_with_warning() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())
    selection = runtime.resolve_voice(None)
    assert selection.name == "aidar"
    assert selection.warning is not None
    assert "xenia" in selection.warning


def test_rejects_requested_voice_only_against_runtime_speakers() -> None:
    runtime = SileroRuntime(FakeModel(["aidar", "baya"]), _spec())
    with pytest.raises(ModelError, match="aidar, baya"):
        runtime.resolve_voice("xenia")


def test_synthesizes_ssml_on_cpu_at_manifest_sample_rate() -> None:
    model = FakeModel(["xenia"])
    runtime = SileroRuntime(model, _spec())
    chunk = SpeechChunk("<speak>Текст</speak>", 350)

    audio = runtime.synthesize(chunk, "xenia")

    assert model.device == torch.device("cpu")
    assert model.calls == [{
        "ssml_text": chunk.ssml_text,
        "speaker": "xenia",
        "sample_rate": 48000,
    }]
    assert audio.shape == (3,)


def test_wraps_model_failure_as_synthesis_error() -> None:
    runtime = SileroRuntime(FailingModel(["xenia"]), _spec())
    with pytest.raises(SynthesisError, match="boom"):
        runtime.synthesize(SpeechChunk("<speak>Текст</speak>", 350), "xenia")
```

- [ ] **Step 2: Run the suite and confirm red**

```bash
uv run pytest tests/test_tts.py -v
```

Expected: import fails because `pytts.tts` does not exist.

- [ ] **Step 3: Implement the PackageImporter adapter and runtime validation**

Create `src/pytts/tts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pytts.domain import SpeechChunk
from pytts.errors import ModelError, SynthesisError
from pytts.model_store import DownloadProgress, ModelSpec, ModelStore


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    name: str
    warning: str | None


def _validate_package(path: Path) -> None:
    try:
        torch.package.PackageImporter(str(path))
    except Exception as error:
        raise ModelError(f"Silero file is not a readable Torch package: {error}") from error


def _load_packaged_model(path: Path) -> Any:
    try:
        importer = torch.package.PackageImporter(str(path))
        return importer.load_pickle("tts_models", "model")
    except Exception as error:
        raise ModelError(f"Could not load Silero tts_models/model: {error}") from error


class SileroRuntime:
    def __init__(self, model: Any, spec: ModelSpec) -> None:
        try:
            model.to(torch.device("cpu"))
        except Exception as error:
            raise ModelError(f"Could not move Silero model to CPU: {error}") from error
        self._model = model
        self._spec = spec
        raw_speakers = getattr(model, "speakers", None)
        if not isinstance(raw_speakers, (list, tuple)):
            raise ModelError("Silero model.speakers is missing or invalid")
        self._speakers = tuple(str(item) for item in raw_speakers if str(item))
        if not self._speakers:
            raise ModelError("Silero model.speakers is empty")

    @classmethod
    def load(
        cls,
        store: ModelStore,
        spec: ModelSpec,
        download_progress: DownloadProgress | None = None,
    ) -> "SileroRuntime":
        path = store.ensure(spec, _validate_package, download_progress)
        return cls(_load_packaged_model(path), spec)

    @property
    def speakers(self) -> tuple[str, ...]:
        return self._speakers

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        if requested is not None:
            if requested not in self._speakers:
                available = ", ".join(self._speakers)
                raise ModelError(f"Unknown voice {requested!r}; runtime voices: {available}")
            return VoiceSelection(requested, None)
        if self._spec.preferred_voice in self._speakers:
            return VoiceSelection(self._spec.preferred_voice, None)
        fallback = self._speakers[0]
        return VoiceSelection(
            fallback,
            f"Preferred voice {self._spec.preferred_voice!r} is unavailable; using {fallback!r}",
        )

    def synthesize(self, chunk: SpeechChunk, voice: str) -> torch.Tensor:
        if voice not in self._speakers:
            raise ModelError(f"Voice {voice!r} disappeared from runtime speakers")
        try:
            audio = self._model.apply_tts(
                ssml_text=chunk.ssml_text,
                speaker=voice,
                sample_rate=self._spec.sample_rate,
            )
        except Exception as error:
            raise SynthesisError(f"Silero synthesis failed: {error}") from error
        if not isinstance(audio, torch.Tensor) or audio.ndim != 1 or audio.numel() == 0:
            raise SynthesisError("Silero returned empty or non-mono audio")
        return audio.detach().cpu()
```

- [ ] **Step 4: Add a concrete PackageImporter load-contract test**

Append to `tests/test_tts.py`:

```python
def test_load_uses_verified_package_contract_and_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(["xenia"])
    calls: list[tuple[object, ...]] = []

    class FakeImporter:
        def __init__(self, path: str) -> None:
            calls.append(("open", path))

        def load_pickle(self, package: str, resource: str) -> FakeModel:
            calls.append(("load_pickle", package, resource))
            return model

    class FakeStore:
        def ensure(self, spec: object, validator: object, progress: object) -> Path:
            path = tmp_path / "v5_5_ru.pt"
            path.write_bytes(b"verified")
            validator(path)  # type: ignore[operator]
            return path

    monkeypatch.setattr(torch.package, "PackageImporter", FakeImporter)

    runtime = SileroRuntime.load(FakeStore(), _spec())  # type: ignore[arg-type]

    assert runtime.speakers == ("xenia",)
    assert model.device == torch.device("cpu")
    assert calls == [
        ("open", str(tmp_path / "v5_5_ru.pt")),
        ("open", str(tmp_path / "v5_5_ru.pt")),
        ("load_pickle", "tts_models", "model"),
    ]
```

This locks the exact `tts_models/model` package contract without downloading the real model in the
unit suite.

- [ ] **Step 5: Run focused tests and Ruff**

```bash
uv run pytest tests/test_tts.py -v
uv run ruff check src/pytts/tts.py tests/test_tts.py
```

Expected: voice spelling errors are checked only after model load and against `model.speakers`; all
tests pass.

- [ ] **Step 6: Commit the Silero runtime adapter**

```bash
git add src/pytts/tts.py tests/test_tts.py
git commit -m "feat: load silero voices dynamically"
```

---

### Task 10: Stream PCM into an Atomic Mono MP3

**Files:**
- Create: `src/pytts/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- `pcm16_bytes(audio: torch.Tensor) -> bytes`
- `AtomicMp3Writer(output_path, force, sample_rate=48000, encoder_factory=lameenc.Encoder)`
- `write_audio(audio)`, `write_silence(milliseconds)`, `commit()`, and `abort()` keep memory
  bounded to one speech chunk plus at most 700 ms of silence.

- [ ] **Step 1: Add the audio-specific exit-5 exception**

Extend `src/pytts/errors.py`:

```python
class AudioError(SynthesisError):
    """PCM conversion, LAME encoding, or atomic output failure."""
```

Add `AudioError` to the `pytts.errors` import in `tests/test_domain.py`, then add
`assert AudioError.exit_code == 5` beside the existing exit-code assertions.

- [ ] **Step 2: Write failing PCM and atomic lifecycle tests**

Create `tests/test_audio.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mutagen.mp3 import BitrateMode, MP3

from pytts.audio import AtomicMp3Writer, pcm16_bytes
from pytts.errors import InputError


class FakeEncoder:
    instances: list["FakeEncoder"] = []

    def __init__(self) -> None:
        self.settings: dict[str, int] = {}
        self.inputs: list[bytes] = []
        self.__class__.instances.append(self)

    def set_bit_rate(self, value: int) -> None:
        self.settings["bit_rate"] = value

    def set_in_sample_rate(self, value: int) -> None:
        self.settings["sample_rate"] = value

    def set_channels(self, value: int) -> None:
        self.settings["channels"] = value

    def set_quality(self, value: int) -> None:
        self.settings["quality"] = value

    def encode(self, pcm: bytes) -> bytes:
        self.inputs.append(pcm)
        return b"encoded:" + pcm

    def flush(self) -> bytes:
        return b":flushed"


def test_pcm16_clamps_and_uses_little_endian_signed_samples() -> None:
    assert pcm16_bytes(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])) == (
        b"\x01\x80\x01\x80\x00\x00\xff\x7f\xff\x7f"
    )


def test_streams_audio_and_silence_then_atomically_commits(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
        writer.write_audio(torch.tensor([0.0, 0.5]))
        writer.write_silence(120)
        assert not output.exists()
        assert output.with_name("article.mp3.part").exists()
        writer.commit()

    encoder = FakeEncoder.instances[-1]
    assert encoder.settings == {
        "bit_rate": 96,
        "sample_rate": 48000,
        "channels": 1,
        "quality": 2,
    }
    assert len(encoder.inputs[1]) == 48000 * 120 // 1000 * 2
    assert output.read_bytes().endswith(b":flushed")
    assert not output.with_name("article.mp3.part").exists()


def test_exception_removes_only_exact_partial(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    neighbor = tmp_path / "article.mp3.part.keep"
    neighbor.write_bytes(b"user data")
    with pytest.raises(RuntimeError, match="stop"):
        with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
            writer.write_audio(torch.tensor([0.0]))
            raise RuntimeError("stop")
    assert not output.with_name("article.mp3.part").exists()
    assert neighbor.read_bytes() == b"user data"


def test_keyboard_interrupt_removes_partial(tmp_path: Path) -> None:
    output = tmp_path / "interrupted.mp3"
    with pytest.raises(KeyboardInterrupt):
        with AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder) as writer:
            writer.write_audio(torch.tensor([0.0]))
            raise KeyboardInterrupt
    assert not output.exists()
    assert not output.with_name("interrupted.mp3.part").exists()


def test_refuses_existing_output_without_force(tmp_path: Path) -> None:
    output = tmp_path / "article.mp3"
    output.write_bytes(b"existing")
    with pytest.raises(InputError, match="--force"):
        AtomicMp3Writer(output, force=False, encoder_factory=FakeEncoder)
    assert output.read_bytes() == b"existing"


def test_real_lame_output_has_required_metadata(tmp_path: Path) -> None:
    output = tmp_path / "real.mp3"
    audio = torch.sin(torch.arange(48000, dtype=torch.float32) * (2 * torch.pi * 440 / 48000))
    with AtomicMp3Writer(output, force=False) as writer:
        writer.write_audio(audio)
        writer.commit()

    info = MP3(output).info
    assert info.sample_rate == 48000
    assert info.channels == 1
    assert info.bitrate_mode == BitrateMode.CBR
    assert 90000 <= info.bitrate <= 100000
    assert info.length == pytest.approx(1.0, abs=0.15)
```

- [ ] **Step 3: Run tests and confirm red**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: import fails because `pytts.audio` does not exist.

- [ ] **Step 4: Implement bounded conversion and the writer lifecycle**

Create `src/pytts/audio.py`:

```python
from __future__ import annotations

import os
import sys
from array import array
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import lameenc
import torch

from pytts.errors import AudioError, InputError


def pcm16_bytes(audio: torch.Tensor) -> bytes:
    if audio.ndim != 1 or audio.numel() == 0:
        raise AudioError("Audio chunk must be a non-empty mono tensor")
    samples = audio.detach().cpu().clamp(-1.0, 1.0).mul(32767).round().to(torch.int16)
    pcm = array("h", samples.tolist())
    if pcm.itemsize != 2:
        raise AudioError("Platform signed-short is not 16 bit")
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


class AtomicMp3Writer:
    def __init__(
        self,
        output_path: Path,
        force: bool,
        sample_rate: int = 48000,
        encoder_factory: Callable[[], Any] = lameenc.Encoder,
    ) -> None:
        if output_path.exists() and not force:
            raise InputError(f"Output already exists: {output_path}; pass --force to replace it")
        if not output_path.parent.is_dir():
            raise InputError(f"Output directory does not exist: {output_path.parent}")
        self.output_path = output_path
        self.partial_path = output_path.with_name(output_path.name + ".part")
        self.partial_path.unlink(missing_ok=True)
        self._sample_rate = sample_rate
        self._encoder = encoder_factory()
        self._encoder.set_bit_rate(96)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(1)
        self._encoder.set_quality(2)
        try:
            self._target = self.partial_path.open("wb")
        except OSError as error:
            raise AudioError(f"Could not open partial output {self.partial_path}: {error}") from error
        self._committed = False
        self._closed = False

    def __enter__(self) -> "AtomicMp3Writer":
        return self

    def _encode(self, pcm: bytes) -> None:
        if self._closed:
            raise AudioError("MP3 writer is already closed")
        try:
            encoded = self._encoder.encode(pcm)
            if encoded:
                self._target.write(encoded)
        except Exception as error:
            raise AudioError(f"LAME encoding failed: {error}") from error

    def write_audio(self, audio: torch.Tensor) -> None:
        self._encode(pcm16_bytes(audio))

    def write_silence(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("Silence duration must not be negative")
        frames = self._sample_rate * milliseconds // 1000
        self._encode(b"\x00\x00" * frames)

    def commit(self) -> None:
        if self._closed:
            raise AudioError("MP3 writer is already closed")
        try:
            flushed = self._encoder.flush()
            if flushed:
                self._target.write(flushed)
            self._target.flush()
            os.fsync(self._target.fileno())
            self._target.close()
            self._closed = True
            os.replace(self.partial_path, self.output_path)
            self._committed = True
        except Exception as error:
            self.abort()
            raise AudioError(f"Could not finalize MP3 {self.output_path}: {error}") from error

    def abort(self) -> None:
        if not self._closed:
            self._target.close()
            self._closed = True
        self.partial_path.unlink(missing_ok=True)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._committed:
            self.abort()
```

`set_bit_rate(96)` selects the fixed 96 Kbit/s mode in `lameenc`; one encoder instance spans the
entire article so the result is one MP3 stream rather than concatenated MP3 files.

- [ ] **Step 5: Run focused tests and inspect the MP3 metadata output**

```bash
uv run pytest tests/test_audio.py -v
uv run ruff check src/pytts/audio.py tests/test_audio.py
```

Expected: pytest's real-LAME test proves mono, CBR, 48 kHz, and approximately 96 Kbit/s on its
temporary output.

- [ ] **Step 6: Commit the streaming encoder**

```bash
git add src/pytts/errors.py tests/test_domain.py src/pytts/audio.py tests/test_audio.py
git commit -m "feat: stream atomic mono mp3 output"
```

---

### Task 11: Orchestrate Conversion and Emit Stable Progress Events

**Files:**
- Create: `src/pytts/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- `ConversionPipeline.convert(request: ConversionRequest) -> ConversionResult`
- `ConversionPipeline.list_voices() -> tuple[str, ...]`
- `ProgressEvent(stage, completed, total, message, warning)` allows the CLI to render progress without
  putting Rich concerns into domain modules.
- Runtime and writer factories are injected so unit tests never load the real model.

- [ ] **Step 1: Write failing orchestration tests with bounded fakes**

Create `tests/test_pipeline.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import TracebackType

import pytest
import torch

from pytts.domain import Article, BlockKind, ConversionRequest, TextBlock
from pytts.errors import InputError, SynthesisError
from pytts.model_store import ModelSpec
from pytts.pipeline import ConversionPipeline, ProgressEvent, ProgressStage
from pytts.tts import VoiceSelection


class FakeInputReader:
    def read(self, path: Path) -> Article:
        return Article(path, (TextBlock(BlockKind.PARAGRAPH, "Ув. автор."),))


class FakeRuntime:
    speakers = ("aidar", "xenia")

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.chunks: list[str] = []

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        return VoiceSelection(requested or "xenia", None)

    def synthesize(self, chunk: object, voice: str) -> torch.Tensor:
        if self.fail:
            raise SynthesisError("failed chunk")
        self.chunks.append(chunk.ssml_text)  # type: ignore[attr-defined]
        return torch.tensor([0.0, 0.1])


class FakeWriter:
    def __init__(self, output: Path, force: bool) -> None:
        self.output = output
        self.force = force
        self.audio_count = 0
        self.pauses: list[int] = []
        self.committed = False
        self.aborted = False

    def __enter__(self) -> "FakeWriter":
        return self

    def write_audio(self, audio: torch.Tensor) -> None:
        self.audio_count += 1

    def write_silence(self, milliseconds: int) -> None:
        self.pauses.append(milliseconds)

    def commit(self) -> None:
        self.committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.aborted = not self.committed


def _pipeline(
    tmp_path: Path,
    runtime: FakeRuntime,
    events: list[ProgressEvent],
    writers: list[FakeWriter],
) -> ConversionPipeline:
    (tmp_path / "pytts.yaml").write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n', encoding="utf-8"
    )
    spec = ModelSpec("v5_5_ru", "https://example.test", "a" * 64, "xenia", 48000, 800)

    def writer_factory(output: Path, force: bool) -> FakeWriter:
        writer = FakeWriter(output, force)
        writers.append(writer)
        return writer

    return ConversionPipeline(
        input_reader=FakeInputReader(),
        model_spec=spec,
        runtime_factory=lambda progress: runtime,
        writer_factory=writer_factory,
        project_root=tmp_path,
        progress=events.append,
    )


def test_conversion_expands_chunks_synthesizes_and_commits(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    writers: list[FakeWriter] = []
    runtime = FakeRuntime()
    pipeline = _pipeline(tmp_path, runtime, events, writers)
    source = tmp_path / "article.md"
    source.write_text("ignored by fake", encoding="utf-8")

    result = pipeline.convert(ConversionRequest(input_path=source))

    assert result.output_path == tmp_path / "article.mp3"
    assert result.voice == "xenia"
    assert "Уважаемый автор." in runtime.chunks[0]
    assert writers[0].audio_count == result.chunk_count
    assert writers[0].pauses == [350]
    assert writers[0].committed
    assert [event.stage for event in events] == [
        ProgressStage.CONFIG,
        ProgressStage.READ,
        ProgressStage.CLEAN,
        ProgressStage.CHUNK,
        ProgressStage.MODEL,
        ProgressStage.VOICE,
        ProgressStage.SYNTHESIS,
        ProgressStage.FINALIZE,
        ProgressStage.COMPLETE,
    ]


def test_synthesis_failure_aborts_writer(tmp_path: Path) -> None:
    writers: list[FakeWriter] = []
    pipeline = _pipeline(tmp_path, FakeRuntime(fail=True), [], writers)
    source = tmp_path / "article.md"
    source.write_text("ignored", encoding="utf-8")

    with pytest.raises(SynthesisError, match="failed chunk"):
        pipeline.convert(ConversionRequest(input_path=source))

    assert writers[0].aborted
    assert not writers[0].committed


def test_list_voices_loads_runtime_without_input(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, FakeRuntime(), [], [])
    assert pipeline.list_voices() == ("aidar", "xenia")


def test_rejects_non_mp3_or_input_overwrite_before_loading_runtime(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, FakeRuntime(), [], [])
    source = tmp_path / "article.md"
    source.write_text("original", encoding="utf-8")

    with pytest.raises(InputError, match="must end in .mp3"):
        pipeline.convert(ConversionRequest(source, output_path=tmp_path / "audio.wav"))
    with pytest.raises(InputError, match="must differ from input"):
        pipeline.convert(ConversionRequest(source, output_path=source))

    assert source.read_text(encoding="utf-8") == "original"
```

- [ ] **Step 2: Run tests and confirm red**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: import fails because `pytts.pipeline` does not exist.

- [ ] **Step 3: Implement protocols, stages, output checks, and conversion order**

Create `src/pytts/pipeline.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol

import torch

from pytts.config import load_config
from pytts.domain import Article, ConversionRequest, ConversionResult, SpeechChunk
from pytts.errors import InputError
from pytts.model_store import DownloadProgress, ModelSpec
from pytts.text.abbreviations import AbbreviationExpander
from pytts.text.chunker import chunk_article
from pytts.text.cleaner import clean_article
from pytts.tts import VoiceSelection


class ProgressStage(StrEnum):
    CONFIG = "config"
    READ = "read"
    CLEAN = "clean"
    CHUNK = "chunk"
    MODEL = "model"
    VOICE = "voice"
    SYNTHESIS = "synthesis"
    FINALIZE = "finalize"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: ProgressStage
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    warning: bool = False


class ArticleReader(Protocol):
    def read(self, path: Path) -> Article: ...


class Runtime(Protocol):
    @property
    def speakers(self) -> tuple[str, ...]: ...
    def resolve_voice(self, requested: str | None) -> VoiceSelection: ...
    def synthesize(self, chunk: SpeechChunk, voice: str) -> torch.Tensor: ...


class AudioWriter(Protocol):
    def __enter__(self) -> "AudioWriter": ...
    def write_audio(self, audio: torch.Tensor) -> None: ...
    def write_silence(self, milliseconds: int) -> None: ...
    def commit(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


RuntimeFactory = Callable[[DownloadProgress | None], Runtime]
WriterFactory = Callable[[Path, bool], AudioWriter]
ProgressSink = Callable[[ProgressEvent], None]


class ConversionPipeline:
    def __init__(
        self,
        input_reader: ArticleReader,
        model_spec: ModelSpec,
        runtime_factory: RuntimeFactory,
        writer_factory: WriterFactory,
        project_root: Path,
        progress: ProgressSink | None = None,
    ) -> None:
        self._input_reader = input_reader
        self._spec = model_spec
        self._runtime_factory = runtime_factory
        self._writer_factory = writer_factory
        self._project_root = project_root
        self._progress = progress or (lambda event: None)

    def _emit(self, stage: ProgressStage, **values: object) -> None:
        self._progress(ProgressEvent(stage=stage, **values))

    def _download_progress(self, completed: int, total: int | None) -> None:
        self._emit(ProgressStage.MODEL, completed=completed, total=total)

    def _runtime(self) -> Runtime:
        self._emit(ProgressStage.MODEL, message="Loading Silero v5_5_ru")
        return self._runtime_factory(self._download_progress)

    def list_voices(self) -> tuple[str, ...]:
        return self._runtime().speakers

    def convert(self, request: ConversionRequest) -> ConversionResult:
        output = request.output_path or request.input_path.with_suffix(".mp3")
        if output.resolve() == request.input_path.resolve():
            raise InputError("Output path must differ from input path")
        if output.suffix.casefold() != ".mp3":
            raise InputError(f"Output path must end in .mp3: {output}")
        if output.exists() and not request.force:
            raise InputError(f"Output already exists: {output}; pass --force to replace it")

        self._emit(ProgressStage.CONFIG, message="Loading abbreviation config")
        config = load_config(request.config_path, self._project_root)
        self._emit(ProgressStage.READ, message=f"Reading {request.input_path}")
        article = self._input_reader.read(request.input_path)
        cleaned = clean_article(article)
        self._emit(
            ProgressStage.CLEAN,
            message=cleaned.warning or "Article text cleaned",
            warning=cleaned.warning is not None,
        )
        expanded = AbbreviationExpander(config.abbreviations).expand_article(cleaned.article)
        chunks = chunk_article(expanded, request.rate, self._spec.max_text_chars)
        self._emit(ProgressStage.CHUNK, completed=len(chunks), total=len(chunks))

        runtime = self._runtime()
        self._emit(ProgressStage.VOICE, message="Validating voice against model.speakers")
        selection = runtime.resolve_voice(request.voice)
        if selection.warning:
            self._emit(ProgressStage.VOICE, message=selection.warning, warning=True)

        with self._writer_factory(output, request.force) as writer:
            for index, chunk in enumerate(chunks, start=1):
                audio = runtime.synthesize(chunk, selection.name)
                writer.write_audio(audio)
                writer.write_silence(chunk.pause_after_ms)
                self._emit(ProgressStage.SYNTHESIS, completed=index, total=len(chunks))
            self._emit(ProgressStage.FINALIZE, message="Finalizing MP3")
            writer.commit()

        self._emit(ProgressStage.COMPLETE, message=f"Wrote {output}")
        return ConversionResult(output, selection.name, len(chunks))
```

Because `_runtime()` is called before `resolve_voice()`, the first misspelled `--voice` on a fresh
machine is necessarily reported only after the model download and load. `list_voices()` has the same
first-run cost and requires no dummy input path.

- [ ] **Step 4: Make the progress-order assertion compare first occurrences**

The model download and synthesis stages can legitimately emit several events. Replace the direct
stage list assertion in `test_conversion_expands_chunks_synthesizes_and_commits` with:

```python
first_occurrences = list(dict.fromkeys(event.stage for event in events))
assert first_occurrences == [
    ProgressStage.CONFIG,
    ProgressStage.READ,
    ProgressStage.CLEAN,
    ProgressStage.CHUNK,
    ProgressStage.MODEL,
    ProgressStage.VOICE,
    ProgressStage.SYNTHESIS,
    ProgressStage.FINALIZE,
    ProgressStage.COMPLETE,
]
```

- [ ] **Step 5: Run focused tests and Ruff**

```bash
uv run pytest tests/test_pipeline.py -v
uv run ruff check src/pytts/pipeline.py tests/test_pipeline.py
```

Expected: conversion is ordered, output collision fails before model load, synthesis failure invokes
the writer context cleanup, and all tests pass.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/pytts/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate article conversion"
```

---

### Task 12: Expose the Typer CLI and Rich Progress UI

**Files:**
- Create: `src/pytts/cli.py`
- Create: `src/pytts/__main__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_cli.py`

**Interfaces:**
- `uv run pytts [INPUT] [--output PATH] [--voice NAME] [--speed SPEED] [--config PATH] [--force] [--debug]`
- `uv run pytts --list-voices`
- `INPUT` plus `--list-voices` is a usage error with exit code 2; neither mode silently wins.
- Application exit codes are 2 usage, 3 input/config/output, 4 model, 5 synthesis/encoding, and
  130 interruption.

- [ ] **Step 1: Write failing CLI mode and error tests**

Create `tests/test_cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import pytts.cli as cli
from pytts.domain import ConversionResult
from pytts.errors import InputError, ModelError, SynthesisError

runner = CliRunner()


class FakePipeline:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def list_voices(self) -> tuple[str, ...]:
        return ("aidar", "xenia")

    def convert(self, request: object) -> ConversionResult:
        self.requests.append(request)
        return ConversionResult(Path("article.mp3"), "xenia", 3)


def _factory(pipeline: object):
    return lambda progress: pipeline


def test_list_voices_needs_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FakePipeline()))
    result = runner.invoke(cli.app, ["--list-voices"])

    assert result.exit_code == 0
    assert "aidar" in result.output
    assert "xenia" in result.output


def test_input_and_list_voices_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FakePipeline()))
    result = runner.invoke(cli.app, ["article.md", "--list-voices"])

    assert result.exit_code == 2
    assert "cannot be used together" in result.output


def test_conversion_builds_request(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "_pipeline_factory", _factory(pipeline))
    result = runner.invoke(
        cli.app,
        ["article.md", "--voice", "xenia", "--speed", "fast", "--force"],
    )

    assert result.exit_code == 0
    request = pipeline.requests[0]
    assert request.input_path == Path("article.md")
    assert request.voice == "xenia"
    assert request.rate.value == "fast"
    assert request.force
    assert "article.mp3" in result.output


@pytest.mark.parametrize(
    ("error", "code"),
    [(InputError("bad input"), 3), (ModelError("bad model"), 4), (SynthesisError("bad audio"), 5)],
)
def test_maps_application_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    class FailingPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise error

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FailingPipeline()))
    result = runner.invoke(cli.app, ["article.md"])

    assert result.exit_code == code
    assert str(error) in result.output


def test_debug_prints_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise ModelError("details")

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(FailingPipeline()))
    result = runner.invoke(cli.app, ["article.md", "--debug"])

    assert result.exit_code == 4
    assert "Traceback" in result.output
    assert "ModelError" in result.output


def test_keyboard_interrupt_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    class InterruptedPipeline(FakePipeline):
        def convert(self, request: object) -> ConversionResult:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_pipeline_factory", _factory(InterruptedPipeline()))
    assert runner.invoke(cli.app, ["article.md"]).exit_code == 130
```

- [ ] **Step 2: Run tests and confirm red**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: import fails because `pytts.cli` does not exist.

- [ ] **Step 3: Add the console entry point**

Add to `pyproject.toml` after `[project]` dependencies:

```toml
[project.scripts]
pytts = "pytts.cli:app"
```

Run `uv lock` after the metadata change.

- [ ] **Step 4: Implement dependency construction, progress rendering, and command behavior**

Create `src/pytts/cli.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pytts.audio import AtomicMp3Writer
from pytts.config import find_project_root
from pytts.domain import ConversionRequest, SpeechRate
from pytts.errors import PyTTSError
from pytts.model_store import ModelStore, load_model_spec
from pytts.pipeline import ConversionPipeline, ProgressEvent, ProgressSink, ProgressStage
from pytts.readers.input import InputReader
from pytts.tts import SileroRuntime

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console(stderr=True)


class RichProgressReporter:
    def __init__(self, target: Console) -> None:
        self._console = target
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=target,
            transient=True,
        )
        self._tasks: dict[ProgressStage, int] = {}
        self._started = False

    def __call__(self, event: ProgressEvent) -> None:
        if event.warning and event.message:
            self._console.print(f"[yellow]Warning:[/] {event.message}")
        if event.completed is not None:
            if not self._started:
                self._progress.start()
                self._started = True
            task_id = self._tasks.get(event.stage)
            if task_id is None:
                task_id = self._progress.add_task(event.stage.value, total=event.total)
                self._tasks[event.stage] = task_id
            self._progress.update(task_id, completed=event.completed, total=event.total)
        elif event.message and not event.warning:
            self._console.print(event.message)

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False


def build_pipeline(progress: ProgressSink) -> ConversionPipeline:
    spec = load_model_spec()
    store = ModelStore()
    return ConversionPipeline(
        input_reader=InputReader(),
        model_spec=spec,
        runtime_factory=lambda download: SileroRuntime.load(store, spec, download),
        writer_factory=lambda output, force: AtomicMp3Writer(
            output, force, sample_rate=spec.sample_rate
        ),
        project_root=find_project_root(Path(__file__)),
        progress=progress,
    )


PipelineFactory = Callable[[ProgressSink], ConversionPipeline]
_pipeline_factory: PipelineFactory = build_pipeline


def _usage(message: str) -> None:
    console.print(f"[red]Error:[/] {message}")
    raise typer.Exit(2)


@app.command()
def main(
    input_path: Annotated[Path | None, typer.Argument(metavar="[INPUT]")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    voice: Annotated[str | None, typer.Option("--voice")] = None,
    speed: Annotated[SpeechRate, typer.Option("--speed")] = SpeechRate.NORMAL,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    list_voices: Annotated[bool, typer.Option("--list-voices")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    if list_voices and input_path is not None:
        _usage("INPUT and --list-voices cannot be used together")
    if not list_voices and input_path is None:
        _usage("INPUT is required unless --list-voices is used")

    reporter = RichProgressReporter(console)
    try:
        pipeline = _pipeline_factory(reporter)
        if list_voices:
            for name in pipeline.list_voices():
                typer.echo(name)
            return
        result = pipeline.convert(
            ConversionRequest(
                input_path=input_path,
                output_path=output,
                config_path=config,
                voice=voice,
                rate=speed,
                force=force,
            )
        )
        typer.echo(f"Wrote {result.output_path} with voice {result.voice}")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/]")
        raise typer.Exit(130) from None
    except PyTTSError as error:
        console.print(f"[red]Error:[/] {error}")
        if debug:
            console.print_exception(show_locals=False)
        raise typer.Exit(error.exit_code) from None
    finally:
        reporter.stop()
```

The `input_path` guard makes its type non-optional at runtime, but static type checkers may not infer
that across Typer branches. If a checker is added, assign `source = input_path` after the guard and
assert `source is not None` before constructing `ConversionRequest`; do not use an unchecked cast.

Create `src/pytts/__main__.py`:

```python
from pytts.cli import app


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run CLI tests and entry-point smoke checks**

```bash
uv lock
uv run pytest tests/test_cli.py -v
uv run ruff check src/pytts/cli.py src/pytts/__main__.py tests/test_cli.py
uv run pytts --help
uv run python -m pytts --help
```

Expected: tests pass; both help commands show one optional `[INPUT]`, five speed values, and
`--list-voices`.

- [ ] **Step 6: Commit the CLI**

```bash
git add pyproject.toml uv.lock src/pytts/cli.py src/pytts/__main__.py tests/test_cli.py
git commit -m "feat: expose pytts command line interface"
```

---

### Task 13: Document, Exercise, and Verify the Complete Product

**Files:**
- Replace: `README.md`
- Create: `tests/fixtures/smoke.md`
- Create: `tests/test_e2e.py`
- Create: `tests/test_silero_integration.py`
- Verify: `docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md`

**Interfaces:**
- The README is the user contract for local installation, first-run download, CLI behavior, root
  config, limitations, and licensing.
- The normal suite stays network-free; `PYTTS_RUN_SILERO=1` explicitly enables the cached/real-model
  integration test.

- [ ] **Step 1: Write a deterministic smoke article fixture**

Create `tests/fixtures/smoke.md`:

```markdown
# Проверка pytts

Ув. читатель, это короткая статья, сохранённая из интернета.

- В 2026 году показатель составил 25 %.
- Диапазон значений — 3–5, сумма — 1 500 ₽.

> Числа сохраняются без дополнительной нормализации.
```

- [ ] **Step 2: Write a real-reader, fake-model end-to-end test**

Create `tests/test_e2e.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import TracebackType

import torch

from pytts.domain import ConversionRequest
from pytts.model_store import ModelSpec
from pytts.pipeline import ConversionPipeline
from pytts.readers.input import InputReader
from pytts.tts import VoiceSelection


class Runtime:
    speakers = ("xenia",)

    def __init__(self) -> None:
        self.ssml: list[str] = []

    def resolve_voice(self, requested: str | None) -> VoiceSelection:
        return VoiceSelection("xenia", None)

    def synthesize(self, chunk: object, voice: str) -> torch.Tensor:
        self.ssml.append(chunk.ssml_text)  # type: ignore[attr-defined]
        return torch.tensor([0.0, 0.1])


class Writer:
    def __init__(self, output: Path, force: bool) -> None:
        self.output = output
        self.pauses: list[int] = []
        self.committed = False

    def __enter__(self) -> "Writer":
        return self

    def write_audio(self, audio: torch.Tensor) -> None:
        assert audio.ndim == 1

    def write_silence(self, milliseconds: int) -> None:
        self.pauses.append(milliseconds)

    def commit(self) -> None:
        self.committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


def test_markdown_to_audio_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pytts.yaml").write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n', encoding="utf-8"
    )
    source = Path("tests/fixtures/smoke.md").resolve()
    runtime = Runtime()
    writers: list[Writer] = []

    def writer_factory(output: Path, force: bool) -> Writer:
        writer = Writer(output, force)
        writers.append(writer)
        return writer

    pipeline = ConversionPipeline(
        input_reader=InputReader(),
        model_spec=ModelSpec(
            "v5_5_ru", "https://example.test", "a" * 64, "xenia", 48000, 800
        ),
        runtime_factory=lambda progress: runtime,
        writer_factory=writer_factory,
        project_root=project_root,
    )

    result = pipeline.convert(
        ConversionRequest(source, output_path=tmp_path / "smoke.mp3")
    )

    combined = " ".join(runtime.ssml)
    assert "Уважаемый читатель" in combined
    assert "2026" in combined and "25 %" in combined and "1 500 ₽" in combined
    assert "https://" not in combined
    assert result.chunk_count == len(runtime.ssml)
    assert writers[0].committed
    assert 700 in writers[0].pauses
    assert 350 in writers[0].pauses
```

- [ ] **Step 3: Write an opt-in real-model integration test**

Create `tests/test_silero_integration.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest
from mutagen.mp3 import MP3

from pytts.audio import AtomicMp3Writer
from pytts.domain import Article, BlockKind, SpeechRate, TextBlock
from pytts.model_store import ModelStore, load_model_spec
from pytts.text.chunker import chunk_article
from pytts.tts import SileroRuntime

pytestmark = [
    pytest.mark.silero,
    pytest.mark.skipif(
        os.environ.get("PYTTS_RUN_SILERO") != "1",
        reason="set PYTTS_RUN_SILERO=1 for the real Silero smoke test",
    ),
]


def test_real_model_synthesizes_valid_mp3(tmp_path: Path) -> None:
    spec = load_model_spec()
    runtime = SileroRuntime.load(ModelStore(), spec)
    selection = runtime.resolve_voice(None)
    article = Article(
        Path("integration.md"),
        (TextBlock(BlockKind.PARAGRAPH, "Это проверка локального синтеза речи."),),
    )
    chunk = chunk_article(article, SpeechRate.NORMAL, spec.max_text_chars)[0]
    audio = runtime.synthesize(chunk, selection.name)
    output = tmp_path / "integration.mp3"
    with AtomicMp3Writer(output, force=False, sample_rate=spec.sample_rate) as writer:
        writer.write_audio(audio)
        writer.write_silence(chunk.pause_after_ms)
        writer.commit()

    info = MP3(output).info
    assert info.sample_rate == 48000
    assert 90000 <= info.bitrate <= 100000
    assert info.length > 0.5
```

- [ ] **Step 4: Replace the bootstrap README with the complete user contract**

Write `README.md` with these exact sections and facts:

````markdown
# pytts

`pytts` locally converts Russian text-layer PDF and Markdown articles into one MP3 on Apple Silicon
macOS. It uses Silero `v5_5_ru` on CPU; article text and audio never leave the computer.

## Requirements and installation

- Apple Silicon macOS
- [uv](https://docs.astral.sh/uv/)
- network access only for the first model download

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
when available, otherwise the first runtime voice is used with a warning. `INPUT` and
`--list-voices` cannot be combined.

Supported `--speed` values are `x-slow`, `slow`, `normal`, `fast`, and `x-fast`; `normal` is the
default.

The model is cached at `~/Library/Caches/pytts/models/v5_5_ru.pt`. Every use verifies the committed
SHA-256 from `src/pytts/model_manifest.yaml`; a verified cache works offline. A hash mismatch reports
the expected and actual values and explains the deliberate manifest-update process.

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

- Russian only. Low Cyrillic content emits a warning but is not blocked.
- PDF requires an existing text layer; there is no OCR.
- Complex multi-column PDF reading order is not guaranteed; the target is a browser-printed article.
- URLs, code, images, metadata, tables, footnotes, and raw HTML are discarded as non-article content.
- PDF headings are a font-size heuristic; uniform browser-print typography gets paragraph pauses.
- Numbers, dates, currencies, percentages, and ranges are preserved literally and are not converted
  to words, so pronunciation and grammatical agreement depend on Silero.
- Context-sensitive abbreviation gender and morphology are outside this release.

## Model license

The Silero model is distributed under
[CC BY-NC-SA 4.0](https://github.com/snakers4/silero-models/blob/master/LICENSE). This project is for
personal, non-commercial use. The model file is downloaded separately and is never committed or
packaged with `pytts`.
````

- [ ] **Step 5: Run the complete automated verification**

```bash
uv lock --check
uv run ruff check .
uv run pytest -m "not silero" --cov=pytts --cov-report=term-missing
PYTTS_RUN_SILERO=1 uv run pytest tests/test_silero_integration.py -v
```

Expected: lock is current, Ruff is clean, the network-free suite passes, and the opt-in real-model
test produces a valid 48 kHz approximately 96 Kbit/s MP3. Test success means valid and behaviorally
consistent audio, not byte-identical MP3 across different `lameenc`/LAME versions.

- [ ] **Step 6: Perform CLI acceptance checks with the verified real model**

```bash
mkdir -p artifacts
uv run pytts --list-voices
uv run pytts tests/fixtures/smoke.md --output artifacts/smoke.mp3 --force
uv run python -c 'from mutagen.mp3 import MP3; i=MP3("artifacts/smoke.mp3").info; print(i.sample_rate, i.bitrate, round(i.length, 2))'
afplay artifacts/smoke.mp3
```

Expected:

- `--list-voices` exactly matches the runtime speaker list in the Phase 0 report.
- Metadata prints `48000`, a bitrate near `96000`, and a positive duration.
- Listening confirms heading/paragraph/list pauses, all five rates when repeated manually, natural
  expansion of `ув.`, and no spoken URL/code/table content.
- Listen specifically to the fixture's `2026`, `25 %`, `3–5`, and `1 500 ₽`; record the observed
  pronunciation as a known-quality observation, not a pass/fail normalization test.

Repeat a conversion after the cache exists while network access is unavailable. Expected: no
download attempt and a successful MP3. Then convert one representative personal browser-saved PDF
and one Markdown article. Finally, run an article long enough to approach one hour of output and
watch Activity Monitor: memory must remain bounded rather than grow with synthesized duration.

- [ ] **Step 7: Check dependency and repository hygiene**

```bash
uv tree
rg -n 'ffmpeg|torchaudio|torchvision|cuda' pyproject.toml uv.lock
git status --short
```

Expected: the dependency tree contains the locked macOS PyTorch package and no CUDA packages; the
search returns no matches; Git shows only intended source, tests, documentation, fixture, and lockfile
changes. No `.pt`, `.mp3`, `.part`, `artifacts/`, `.idea/`, or cache file is staged.

- [ ] **Step 8: Commit documentation and final verification coverage**

```bash
git add README.md tests/fixtures/smoke.md tests/test_e2e.py tests/test_silero_integration.py
git commit -m "docs: finish pytts usage and verification"
```

---

## Completion Checklist

- [ ] The Phase 0 report and `src/pytts/model_manifest.yaml` contain the same model ID, URL,
  SHA-256, 48 kHz sample rate, and derived text limit; the report also records runtime speakers and
  the successful SSML contract.
- [ ] All runtime voice decisions use `model.speakers`; no source or test assumes the old v4 list.
- [ ] Unit and end-to-end tests are network-free; only the explicitly enabled Silero test may use the
  cache or network.
- [ ] Markdown and text-layer PDF inputs retain readable article structure and omit non-article data.
- [ ] Root and explicit configs obey the strict version-1 schema and one-pass abbreviation semantics.
- [ ] Each clean-text chunk stays within the probed limit before its SSML wrapper.
- [ ] Audio is streamed through one LAME encoder to an atomic mono 48 kHz CBR 96 Kbit/s MP3.
- [ ] Failures and interruption preserve any existing output and remove only the exact `.part` file.
- [ ] README documents first-run voice validation cost, numeric limitations, heading heuristics,
  dynamic fallback, offline cache, exit codes, and `CC BY-NC-SA 4.0` personal-use constraints.
- [ ] Full tests, real-model smoke, Ruff, lock check, CLI smoke, listening check, and long-input memory
  observation have all been completed before declaring the implementation finished.
