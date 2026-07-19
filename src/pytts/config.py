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
