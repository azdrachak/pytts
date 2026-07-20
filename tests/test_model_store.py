from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.error import URLError

import pytest

from pytts.errors import ModelError, ModelIntegrityError
from pytts.model_store import ModelSpec, ModelStore, load_model_spec


def _spec(url: str = "https://example.test/model.pt", digest: str | None = None) -> ModelSpec:
    payload = b"package bytes"
    return ModelSpec(
        model_id="v5_5_ru",
        url=url,
        sha256=digest or sha256(payload).hexdigest(),
        preferred_voice="xenia",
        sample_rate=48000,
        max_text_chars=448,
    )


def _write_manifest(path: Path, **overrides: object) -> None:
    values: dict[str, object] = {
        "model_id": "v5_5_ru",
        "url": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        "sha256": "a" * 64,
        "preferred_voice": "xenia",
        "sample_rate": 48000,
        "max_text_chars": 448,
    }
    values.update(overrides)
    path.write_text(
        "version: 1\nmodel:\n" + "".join(f"  {key}: {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def test_load_model_spec_reads_the_committed_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "model_manifest.yaml"
    _write_manifest(manifest)

    assert load_model_spec(manifest) == ModelSpec(
        model_id="v5_5_ru",
        url="https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        sha256="a" * 64,
        preferred_voice="xenia",
        sample_rate=48000,
        max_text_chars=448,
    )


@pytest.mark.parametrize(
    "contents",
    [
        "version: 2\nmodel: {}\n",
        "version: 1\nmodel:\n  model_id: v5_5_ru\n",
        "version: 1\nmodel:\n  model_id: v5_5_ru\n  url: not-a-url\n  sha256: BAD\n  preferred_voice: xenia\n  sample_rate: 48000\n  max_text_chars: 448\n",
    ],
)
def test_load_model_spec_rejects_invalid_manifests(tmp_path: Path, contents: str) -> None:
    manifest = tmp_path / "model_manifest.yaml"
    manifest.write_text(contents, encoding="utf-8")

    with pytest.raises(ModelError):
        load_model_spec(manifest)


def test_downloads_validates_fsyncs_reports_progress_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"package bytes"
    completed: list[tuple[int, int | None]] = []
    validations: list[Path] = []
    fsynced: list[int] = []

    class _Response:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            if getattr(self, "sent", False):
                return b""
            self.sent = True
            return payload

    monkeypatch.setattr("pytts.model_store.urllib.request.urlopen", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr("pytts.model_store.os.fsync", fsynced.append)
    store = ModelStore(cache_root=tmp_path)

    def report_progress(current: int, total: int | None) -> None:
        completed.append((current, total))

    first = store.ensure(_spec(digest=sha256(payload).hexdigest()), validations.append, report_progress)
    second = store.ensure(_spec(digest=sha256(payload).hexdigest()), validations.append, report_progress)

    assert first == second == tmp_path / "models/v5_5_ru.pt"
    assert first.read_bytes() == payload
    assert validations == [first.with_name(first.name + ".download")]
    assert completed == [(len(payload), len(payload))]
    assert fsynced
    assert not first.with_name(first.name + ".download").exists()


def test_bad_download_hash_leaves_no_partial_and_explains_manifest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"package bytes"

    def download(_: str, path: Path, __: object) -> None:
        path.write_bytes(payload)

    monkeypatch.setattr(ModelStore, "_download", staticmethod(download))
    wrong = "0" * 64

    with pytest.raises(ModelIntegrityError, match="scripts/probe_silero.py") as captured:
        ModelStore(cache_root=tmp_path).ensure(_spec(digest=wrong), lambda path: None)

    assert f"expected {wrong}" in str(captured.value)
    assert sha256(payload).hexdigest() in str(captured.value)
    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
    assert not (tmp_path / "models/v5_5_ru.pt").exists()


def test_cached_hash_mismatch_does_not_redownload_or_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "models/v5_5_ru.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unexpected cached bytes")
    downloads: list[Path] = []

    def download(_: str, path: Path, __: object) -> None:
        downloads.append(path)

    monkeypatch.setattr(ModelStore, "_download", staticmethod(download))

    with pytest.raises(ModelIntegrityError):
        ModelStore(cache_root=tmp_path).ensure(_spec(), lambda path: None)

    assert target.read_bytes() == b"unexpected cached bytes"
    assert downloads == []


def test_package_validation_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def download(_: str, path: Path, __: object) -> None:
        path.write_bytes(b"package bytes")

    monkeypatch.setattr(ModelStore, "_download", staticmethod(download))

    def reject(_: Path) -> None:
        raise ValueError("not a Torch package")

    with pytest.raises(ModelError, match="not a Torch package"):
        ModelStore(cache_root=tmp_path).ensure(_spec(), reject)

    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
    assert not (tmp_path / "models/v5_5_ru.pt").exists()


def test_network_failure_maps_to_model_error_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: str, path: Path, __: object) -> None:
        path.write_bytes(b"partial")
        raise URLError("offline")

    monkeypatch.setattr(ModelStore, "_download", staticmethod(fail))

    with pytest.raises(ModelError, match="Could not download or validate model"):
        ModelStore(cache_root=tmp_path).ensure(_spec(), lambda path: None)

    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()


def test_interrupted_download_removes_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_: str, path: Path, __: object) -> None:
        path.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(ModelStore, "_download", staticmethod(interrupt))

    with pytest.raises(KeyboardInterrupt):
        ModelStore(cache_root=tmp_path).ensure(_spec(), lambda path: None)

    assert not (tmp_path / "models/v5_5_ru.pt.download").exists()
