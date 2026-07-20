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
