# Pronunciation normalization dependency spike

Date: 2026-07-20

## Purpose

Verify the high-risk Russian morphology and currency assumptions in
`2026-07-20-pronunciation-normalization-design.md` before writing an
implementation plan. The spike ran in disposable `uv run --isolated`
environments and did not change project dependencies or source code.

Tested package versions:

- `num2words==0.5.14`
- `pymorphy3==2.0.6`
- `cyrtranslit==1.2.0`
- `rutextnorm==2.1.0` as an evaluated alternative

## `num2words` cases and ordinals

The Russian implementation uses one-character case codes rather than OpenCorpora
names. Direct calls produced:

```text
num2words(56, lang="ru", case="g")
  -> пятидесяти шести

num2words(90, lang="ru", case="n")
  -> девяносто

num2words(142, lang="ru", to="ordinal", case="n")
  -> сто сорок второй

num2words(2026, lang="ru", to="ordinal", case="g")
  -> две тысячи двадцать шестого

num2words(2026, lang="ru", to="ordinal", case="d")
  -> две тысячи двадцать шестому

num2words(2026, lang="ru", to="ordinal", case="p")
  -> две тысячи двадцать шестом

num2words(1500, lang="ru", case="n")
  -> одна тысяча пятьсот
```

The forms required for textual dates, standalone years and ordinals are
available directly. The earlier `gent`/`loct` assumption was invalid; those
strings raise `KeyError`. The package's Russian documentation specifies `n`,
`g`, `d`, `a`, `i`, `p`.

Source: <https://github.com/savoirfairelinux/num2words/wiki/Russian>

## Bounded compound adjectives

The two contract families can be built without determining the case or gender
of an arbitrary adjective:

```text
remove_spaces(num2words(56, case="g")) + "летний"
  -> пятидесятишестилетний

remove_spaces(num2words(56, case="g")) + "летнего"
  -> пятидесятишестилетнего

remove_spaces(num2words(90, case="n")) + "долларовый"
  -> девяностодолларовый

remove_spaces(num2words(90, case="n")) + "долларового"
  -> девяностодолларового
```

The source suffix is preserved verbatim, so its existing gender, number and case
ending are not regenerated. This supports the two explicit families but is not a
general derivational morphology engine. The design was narrowed accordingly.

## Currency behavior

`num2words(..., to="currency")` supports Russian RUB, USD and EUR but treats an
integer input as minor units:

```text
RUB, 1       -> ноль рублей, одна копейка
RUB, "12.50" -> двенадцать рублей, пятьдесят копеек
USD, "12.50" -> двенадцать долларов, пятьдесят центов
EUR, "12.50" -> двенадцать евро, пятьдесят центов
GBP, "12.50" -> NotImplementedError
```

The application therefore needs an explicit amount parser and small noun-form
tables. `num2words` supplies only the numeric words. GBP and `£` require custom
forms, and `евро` must remain invariant.

## Transliteration behavior

`cyrtranslit.to_cyrillic(..., "ru")` is usable as a deterministic fallback but
is not an English pronunciation model:

```text
Brent          -> Брент
Bloomberg      -> Блоомберг
New York Times -> Нев Ёрк Тимес
NASA           -> НАСА
UNESCO         -> УНЕСКО
```

This confirms the need for user overrides and a separate 1–5 uppercase-letter
rule. Mixed-script repair must run before this library.

## Alternative `rutextnorm`

The new TTS-focused package was evaluated because it already covers many target
categories. It correctly produced the two compound adjective families and many
currency/symbol forms. It did not satisfy this project's full contract:

- `19.07.2026 -> девятнадцатое июля ...`, not the required genitive day;
- `12,50 ₽` left the `₽` symbol after decimal expansion;
- `F-16 -> ф-шестнадцать`, not `эф шестнадцать`;
- mixed `FР-5` and PDF `ĸиев` were not repaired;
- `+` remained in `A + B`.

It also performs a broader inventory of abbreviation, unit and symbol rewrites
than the approved application design. Adopting it would still require wrappers
for the exact acceptance cases while expanding behavior outside the requested
scope. The chosen design keeps explicit rules around `num2words` and
`cyrtranslit` instead.

Source: <https://pypi.org/project/rutextnorm/>

## `pymorphy3` conclusion

`pymorphy3` is not needed for the approved contract. Naively inflecting every
word in a generated ordinal phrase can select the wrong parse, while `num2words`
already returns the verified complete cardinal and ordinal case forms. Removing
`pymorphy3` reduces dependencies and avoids ambiguous whole-phrase morphology.

## Decision

Proceed to planning with:

- `num2words` for verified Russian cardinal/ordinal forms;
- `cyrtranslit` as the approximate last-resort word transliterator;
- explicit application rules for mixed scripts, acronym letter names, dates,
  currency parsing/noun agreement, semantic symbols and the two compound
  adjective families;
- no `pymorphy3` and no whole-pipeline `rutextnorm` dependency.
