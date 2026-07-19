from hashlib import sha256
from pathlib import Path

import pytest

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
