# Silero v5_5_ru Runtime Verification

- Python: `3.12.13`
- PyTorch: `2.13.0`
- Model ID: `v5_5_ru`
- URL: `https://models.silero.ai/models/tts/ru/v5_5_ru.pt`
- Model bytes: `145420684`
- SHA-256: `50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437`
- Runtime speakers: `aidar, baya, kseniya, eugene, xenia`
- PackageImporter: `passed`
- apply_tts(text=...): `passed`
- apply_tts(ssml_text=...) rates: `x-slow, slow, medium, fast, x-fast`
- Sample rates: `8000, 24000, 48000`
- Verified L_max: `560` clean characters
- MAX_TEXT_CHARS: `448`
- Number pronunciation observation: Озвученный текст: «июля года показатель вырос на сумма составила а диапазон оказался до единиц»

## Task 13 CLI acceptance follow-up

- The real CLI acceptance run found that the `v5_5_ru` SSML parser rejected the fixture title
  `Проверка pytts` with a `NoneType` key error. The same Cyrillic-only title and the fixture's
  `2026`, `25 %`, `3–5`, and `1 500 ₽` chunks synthesized successfully.
- The smoke fixture now uses `Проверка синтеза`; the network-free E2E test preserves this contract.
- Re-run the two real acceptance tests with `PYTTS_RUN_SILERO=1 uv run pytest
  tests/test_silero_integration.py -v`.
