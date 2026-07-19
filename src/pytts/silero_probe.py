from __future__ import annotations

import html
import subprocess
import sys
import urllib.request
import wave
import warnings
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
LENGTH_REJECTION_MESSAGE = "Model couldn't generate your text, probably it's too long"


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


def _expected_manifest_digest(project_root: Path) -> str | None:
    manifest_path = project_root / "src/pytts/model_manifest.yaml"
    if not manifest_path.exists():
        return None
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        digest = manifest["model"]["sha256"]
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise RuntimeError("Committed model manifest has no valid SHA-256") from error
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise RuntimeError("Committed model manifest has no valid SHA-256")
    return digest.lower()


def _select_speaker(speakers: tuple[str, ...]) -> str:
    if "xenia" in speakers:
        return "xenia"
    if not speakers:
        raise RuntimeError("v5_5_ru exposes no model.speakers")
    fallback = speakers[0]
    warnings.warn(
        f"Preferred speaker xenia is unavailable; using {fallback}",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


def _validate_mono_pcm(audio: Any, description: str) -> torch.Tensor:
    if not isinstance(audio, torch.Tensor) or audio.ndim != 1 or audio.numel() == 0:
        raise RuntimeError(f"{description} did not return non-empty mono PCM")
    return audio


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
        '<speak><prosody rate="'
        + SSML_RATES[rate]
        + '">'
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
    except Exception as error:
        if type(error) is Exception and " ".join(str(error).split()) == LENGTH_REJECTION_MESSAGE:
            return False
        raise
    _validate_mono_pcm(audio, "SSML length probe")
    return True


def _largest_supported_length(check: Callable[[int], bool]) -> int:
    low = 64
    if not check(low):
        raise RuntimeError("Silero rejects the minimum 64-character SSML probe")
    high = 128
    while check(high):
        low = high
        if high == SEARCH_CEILING:
            return SEARCH_CEILING
        high = min(high * 2, SEARCH_CEILING)
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

    verified_limit = (
        f"at least {result.l_max}" if result.l_max == SEARCH_CEILING else str(result.l_max)
    )
    report_path = project_root / "docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md"
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
                f"- Verified L_max: `{verified_limit}` clean characters",
                f"- MAX_TEXT_CHARS: `{result.max_text_chars}`",
                f"- Number pronunciation observation: {result.number_observation}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_probe(project_root: Path) -> ProbeResult:
    model_path = Path(platformdirs.user_cache_dir("pytts")) / "models" / f"{MODEL_ID}.pt"
    if not model_path.exists():
        _download(MODEL_URL, model_path)
    digest = sha256_file(model_path)
    expected_digest = _expected_manifest_digest(project_root)
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError(
            "Cached Silero model SHA-256 mismatch: "
            f"expected {expected_digest}, got {digest}"
        )
    model = _load_model(model_path)

    speakers = tuple(str(value) for value in getattr(model, "speakers", ()))
    speaker = _select_speaker(speakers)

    _validate_mono_pcm(
        model.apply_tts(
            text="Это проверка обычного текстового интерфейса.",
            speaker=speaker,
            sample_rate=48000,
        ),
        "apply_tts(text=...)",
    )

    for sample_rate in SAMPLE_RATES:
        _validate_mono_pcm(
            model.apply_tts(
                text="Проверка частоты дискретизации.",
                speaker=speaker,
                sample_rate=sample_rate,
            ),
            f"apply_tts(text=...) at {sample_rate} Hz",
        )

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
    numbers_audio = _validate_mono_pcm(
        model.apply_tts(
            text=numbers,
            speaker=speaker,
            sample_rate=48000,
        ),
        "numbers",
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
