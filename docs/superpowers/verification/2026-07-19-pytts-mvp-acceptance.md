# pytts MVP Acceptance Evidence

This record distinguishes completed controller-run evidence from checks that still require a manual
environment or listening review.

## Phase 0 baseline

- Model: `v5_5_ru`; SHA-256:
  `50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437`.
- Runtime speakers: `aidar, baya, kseniya, eugene, xenia`.
- Verified `L_max`: `560` clean characters; committed `MAX_TEXT_CHARS`: `448`.
- Recorded listening observation: Озвученный текст: «июля года показатель вырос на сумма
  составила а диапазон оказался до единиц».

## Controller-run automated and CLI evidence

- The initial opt-in test exposed an actual inference wrapper without `eval`; the TDD fix is
  `1be4a5a fix: support silero inference wrapper`.
- A later real CLI run isolated a second failure to the Latin fixture token `pytts` in the title.
  The Russian-only fixture now uses `Проверка синтеза`; the numeric, percentage, range, and currency
  chunks were accepted by the runtime.
- `PYTTS_RUN_SILERO=1 uv run pytest tests/test_silero_integration.py -v` completed with `2 passed`.
- `uv run pytts --list-voices` printed exactly: `aidar`, `baya`, `kseniya`, `eugene`, `xenia`.
- `uv run pytts tests/fixtures/smoke.md --output artifacts/smoke.mp3 --force` produced a valid
  `artifacts/smoke.mp3`: mono, 48 kHz, 96,000 bit/s, 14.3 s, and `BitrateMode.UNKNOWN`.
- With `PYTTS_RUN_SILERO=1`, `HTTPS_PROXY`, `HTTP_PROXY`, and `ALL_PROXY` set to
  `http://127.0.0.1:9`, and an empty `NO_PROXY`, the real integration test completed with `2 passed`.
  The verified cached model required no download.
- The actual CLI generated ignored artifacts successfully at `x-slow`, `slow`, `normal`, `fast`, and
  `x-fast`. Sequential `afplay` playback completed. The user's observation was: «скорость ок,
  изначальная проблема осталась - пропущены все числа и валюты». The generated speed variants were
  acceptable and distinguishable, but the original quality problem remains: numbers and currencies are
  omitted in speech. This is not a successful numeric-normalization result.
- A synthetic `AtomicMp3Writer` soak encoded 3,600 one-second float32 PCM chunks into a 3,600.02 s
  mono 48 kHz / 96 Kbit/s MP3 of 43,200,288 bytes. `resource.ru_maxrss` was exactly 186.4 MiB at 600,
  1,800, and 3,600 encoded seconds; a repeat exited cleanly. The 41 MiB temporary artifact was removed.
  This establishes bounded writer memory only, not a real one-hour Silero article conversion.
- The completed production Silero soak loaded the model once and converted 302 repeated 204-character
  Russian paragraph blocks at normal speed through `chunk_article`, `SileroRuntime`, and one
  `AtomicMp3Writer`. At block 1 the audio duration was 11.9 s and peak RSS 768.0 MiB; at block 100,
  1,192.5 s and 772.6 MiB; at block 200, 2,385.0 s and 772.6 MiB; at block 302, 3,601.3 s and
  772.6 MiB. Peak RSS stabilized by block 100 and did not grow through one hour.
- The final real-Silero MP3 was 3,601.39 s, mono, 48,000 Hz, 96,000 bit/s, and 43,216,704 bytes. Its
  temporary 41 MiB file and directory were removed afterward and are not recoverable; no repository
  artifact remains.
- The latest network-free suite from `7945a86` reported `191 passed, 2 deselected`.
- Expected warnings were PyTorch reporting missing optional NumPy and the packaged model's
  `SyntaxWarning`.
- Generated artifacts are ignored. `.DS_Store` remains untracked user/system state and is intentionally
  excluded from acceptance commits.

## Pending manual acceptance

The following checks are intentionally not claimed as complete:

- Convert and listen to one representative personal browser-saved PDF and one Markdown article.

The controller is proceeding with these manual checks and will update this record with their results.
