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
- The final network-free suite reported `190 passed, 2 deselected`.
- Expected warnings were PyTorch reporting missing optional NumPy and the packaged model's
  `SyntaxWarning`.
- Generated artifacts are ignored; the acceptance worktree was clean after verification.

## Pending manual acceptance

The following checks are intentionally not claimed as complete:

- Repeat conversion with network access disabled after the verified model is cached.
- Listen to all five rates and inspect heading, paragraph, and list pauses.
- Convert and listen to one representative personal browser-saved PDF and one Markdown article.
- Run an article approaching one hour while monitoring memory in Activity Monitor.

The controller is proceeding with these manual checks and will update this record with their results.
