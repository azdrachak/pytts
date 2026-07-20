from pathlib import Path

import pytest

from pytts.domain import Article, BlockKind, SpeechRate, TextBlock
from pytts.errors import AudioError, ConfigError, InputError, ModelError, SynthesisError, UsageError


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
    assert AudioError.exit_code == 5
