from hashlib import sha256
from pathlib import Path
import warnings

import pytest
import torch

import pytts.silero_probe as probe
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


def test_largest_supported_length_returns_exact_boundary() -> None:
    assert probe._largest_supported_length(lambda length: length <= 1000) == 1000


def test_largest_supported_length_treats_probe_ceiling_as_success() -> None:
    assert probe._largest_supported_length(lambda length: True) == probe.SEARCH_CEILING


def test_run_probe_rejects_manifest_digest_mismatch_before_load_or_evidence_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    model_path = cache_dir / "models" / f"{probe.MODEL_ID}.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"unexpected-model")

    manifest_path = tmp_path / "src/pytts/model_manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    original_manifest = "version: 1\nmodel:\n  sha256: '" + "0" * 64 + "'\n"
    manifest_path.write_text(original_manifest, encoding="utf-8")
    report_path = (
        tmp_path / "docs/superpowers/verification/2026-07-19-silero-v5_5-runtime.md"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text("existing evidence", encoding="utf-8")

    loads: list[Path] = []

    def fail_if_loaded(path: Path) -> None:
        loads.append(path)
        raise AssertionError("model must not load after a digest mismatch")

    monkeypatch.setattr(probe.platformdirs, "user_cache_dir", lambda _: str(cache_dir))
    monkeypatch.setattr(probe, "_load_model", fail_if_loaded)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        probe.run_probe(tmp_path)

    assert loads == []
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert report_path.read_text(encoding="utf-8") == "existing evidence"


def test_expected_manifest_digest_allows_first_run_without_manifest(tmp_path: Path) -> None:
    assert probe._expected_manifest_digest(tmp_path) is None


def test_expected_manifest_digest_normalizes_valid_uppercase_hex(tmp_path: Path) -> None:
    manifest_path = tmp_path / "src/pytts/model_manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "model:\n  sha256: '" + "A" * 64 + "'\n", encoding="utf-8"
    )

    assert probe._expected_manifest_digest(tmp_path) == "a" * 64


def test_select_speaker_prefers_xenia_without_warning() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        speaker = probe._select_speaker(("aidar", "xenia"))

    assert speaker == "xenia"
    assert recorded == []


def test_select_speaker_warns_when_xenia_is_unavailable() -> None:
    with pytest.warns(RuntimeWarning, match="xenia.*aidar"):
        speaker = probe._select_speaker(("aidar", "baya"))

    assert speaker == "aidar"


@pytest.mark.parametrize("audio", [torch.empty(0), torch.ones((2, 3))])
def test_validate_mono_pcm_rejects_empty_or_non_mono_audio(audio: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="numbers"):
        probe._validate_mono_pcm(audio, "numbers")
