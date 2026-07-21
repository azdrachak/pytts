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
    assert dict(config.transliterations) == {}


def test_relative_explicit_config_is_resolved_from_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    current_directory = tmp_path / "working-directory"
    current_directory.mkdir()
    (current_directory / "custom.yaml").write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n', encoding="utf-8"
    )
    monkeypatch.chdir(current_directory)

    config = load_config(Path("custom.yaml"), root)

    assert dict(config.abbreviations) == {"ув.": "уважаемый"}
    assert dict(config.transliterations) == {}


def test_loads_optional_transliterations(tmp_path: Path) -> None:
    path = tmp_path / "pytts.yaml"
    path.write_text(
        """version: 1
abbreviations: {}
transliterations:
  Brent: Брент
  New York Times: Нью-Йорк таймс
""",
        encoding="utf-8",
    )

    config = load_config(path, tmp_path)

    assert dict(config.abbreviations) == {}
    assert dict(config.transliterations) == {
        "Brent": "Брент",
        "New York Times": "Нью-Йорк таймс",
    }


def test_config_without_transliterations_is_backward_compatible(tmp_path: Path) -> None:
    path = tmp_path / "pytts.yaml"
    path.write_text(
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n',
        encoding="utf-8",
    )

    config = load_config(path, tmp_path)

    assert dict(config.abbreviations) == {"ув.": "уважаемый"}
    assert dict(config.transliterations) == {}


def test_blank_value_marks_token_for_removal(tmp_path: Path) -> None:
    path = tmp_path / "pytts.yaml"
    path.write_text(
        'version: 1\nabbreviations:\n  "ув.": ""\n  "г-н": "господин"\n'
        'transliterations:\n  Foo: "   "\n',
        encoding="utf-8",
    )

    config = load_config(path, tmp_path)

    assert dict(config.abbreviations) == {"ув.": "", "г-н": "господин"}
    assert dict(config.transliterations) == {"Foo": ""}


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nabbreviations: {}\ntransliterations: []\n",
        (
            "version: 1\nabbreviations: {}\ntransliterations:\n"
            "  Brent: один\n  BRENT: два\n"
        ),
        'version: 1\nabbreviations: {}\ntransliterations:\n  "": пустой\n',
        'version: 1\nabbreviations: {}\ntransliterations:\n  "   ": пустой\n',
    ],
)
def test_invalid_transliterations_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="transliteration"):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "version: 2\nabbreviations: {}\n",
        "version: true\nabbreviations: {}\n",
        "version: 1\nunknown: true\nabbreviations: {}\n",
        "version: 1\nabbreviations: []\n",
        'version: 1\nabbreviations:\n  "ув.": null\n',
        'version: 1\nabbreviations:\n  "УВ.": "один"\n  "ув.": "два"\n',
        'version: 1\nabbreviations:\n  "": "пустой"\n',
        'version: 1\nabbreviations:\n  "   ": "пустой"\n',
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nversion: 1\nabbreviations: {}\n",
        'version: 1\nabbreviations:\n  "ув.": "уважаемый"\n  "ув.": "уважаемая"\n',
        (
            "version: 1\nabbreviations: {}\ntransliterations:\n"
            "  Brent: Брент\n  Brent: Бренд\n"
        ),
    ],
)
def test_exact_duplicate_yaml_keys_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\n7: value\nabbreviations: {}\n",
        "version: 1\n7: value\nunknown: true\nabbreviations: {}\n",
    ],
)
def test_non_string_root_keys_are_config_errors(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad-root-key.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path, tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "version: 1\n",
        "version: 1\nabbreviations: {}\n  malformed\n",
        "- version: 1\n",
    ],
)
def test_missing_or_malformed_config_content_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(path, tmp_path)


def test_missing_root_config_returns_empty_mapping(tmp_path: Path) -> None:
    config = load_config(None, tmp_path)

    assert dict(config.abbreviations) == {}
    assert dict(config.transliterations) == {}


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml", tmp_path)
