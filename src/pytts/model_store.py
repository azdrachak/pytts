from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4

import fcntl

import platformdirs
import yaml

from pytts.errors import ModelError, ModelIntegrityError
from pytts.silero_probe import sha256_file

DownloadProgress = Callable[[int, int | None], None]
PackageValidator = Callable[[Path], None]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MODEL_FIELDS = {
    "model_id",
    "url",
    "sha256",
    "preferred_voice",
    "sample_rate",
    "max_text_chars",
}
_THREAD_LOCKS: dict[Path, Lock] = {}
_THREAD_LOCKS_GUARD = Lock()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    url: str
    sha256: str
    preferred_voice: str
    sample_rate: int
    max_text_chars: int


def _validate_spec(spec: ModelSpec) -> None:
    if not all(
        isinstance(value, str) and value
        for value in (spec.model_id, spec.url, spec.preferred_voice)
    ):
        raise ModelError("model string fields must not be empty")
    if not _MODEL_ID.fullmatch(spec.model_id):
        raise ModelError("model_id must use only filename-safe letters, digits, dots, underscores, or hyphens")
    try:
        parsed = urlsplit(spec.url)
        port = parsed.port
    except ValueError as error:
        raise ModelError(f"model URL is invalid: {error}") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None and not 0 < port < 65536
    ):
        raise ModelError("model URL must be an absolute HTTPS URL with a hostname and no credentials")
    if not isinstance(spec.sha256, str) or not _SHA256.fullmatch(spec.sha256):
        raise ModelError("sha256 must be 64 lowercase hexadecimal characters")
    if (
        type(spec.sample_rate) is not int
        or type(spec.max_text_chars) is not int
        or spec.sample_rate != 48000
        or not 64 <= spec.max_text_chars <= 800
    ):
        raise ModelError("unsupported sample rate or text limit")


def _thread_lock_for(target: Path) -> Lock:
    try:
        key = target.resolve()
    except OSError as error:
        raise ModelError(f"Could not resolve model cache lock {target}: {error}") from error
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, Lock())


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def load_model_spec(path: Path | None = None) -> ModelSpec:
    """Read the strictly versioned, committed model manifest."""
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

    try:
        _validate_spec(spec)
    except ModelError as error:
        raise ModelError(f"{source}: {error}") from error
    return spec


class ModelStore:
    """Maintain the verified local model cache without deserializing cached models."""

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
            target.flush()
            os.fsync(target.fileno())

    def ensure(
        self,
        spec: ModelSpec,
        validate_package: PackageValidator,
        progress: DownloadProgress | None = None,
    ) -> Path:
        """Return a re-hashed cached model or atomically cache a validated download."""
        _validate_spec(spec)
        models = self._root / "models"
        try:
            models.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ModelError(f"Could not create model cache directory {models}: {error}") from error

        target = models / f"{spec.model_id}.pt"
        lock_path = target.with_name(target.name + ".lock")
        with _thread_lock_for(target):
            try:
                lock_file = lock_path.open("a+b")
            except OSError as error:
                raise ModelError(f"Could not open model cache lock {lock_path}: {error}") from error
            try:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                except OSError as error:
                    raise ModelError(f"Could not acquire model cache lock {lock_path}: {error}") from error
                if target.exists():
                    try:
                        self._verify(target, spec)
                    except ModelError:
                        raise
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as error:
                        raise ModelError(f"Could not validate cached model {target}: {error}") from error
                    return target

                partial = target.with_name(f"{target.name}.{uuid4().hex}.download")
                try:
                    self._download(spec.url, partial, progress)
                    self._verify(partial, spec)
                    validate_package(partial)
                    os.replace(partial, target)
                except ModelError:
                    _best_effort_unlink(partial)
                    raise
                except (KeyboardInterrupt, SystemExit):
                    _best_effort_unlink(partial)
                    raise
                except Exception as error:
                    _best_effort_unlink(partial)
                    raise ModelError(f"Could not download or validate model {spec.url}: {error}") from error
                return target
            finally:
                try:
                    lock_file.close()
                except OSError:
                    pass
