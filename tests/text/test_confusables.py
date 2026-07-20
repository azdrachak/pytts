import pytest

from pytts.text.confusables import repair_mixed_scripts


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("уkраинсĸий", "украинский"),
        ("мoсква и cтатья", "москва и статья"),
        ("FР-5", "FP-5"),
        ("FРV-дронов", "FPV-дронов"),
        ("Bloomberg Москва", "Bloomberg Москва"),
        ("Cи", "Си"),
    ],
)
def test_repairs_only_mixed_script_tokens(source: str, expected: str) -> None:
    assert repair_mixed_scripts(source) == expected


@pytest.mark.parametrize(
    ("latin", "cyrillic"),
    [
        ("A", "А"),
        ("a", "а"),
        ("B", "В"),
        ("C", "С"),
        ("c", "с"),
        ("E", "Е"),
        ("e", "е"),
        ("H", "Н"),
        ("K", "К"),
        ("k", "к"),
        ("ĸ", "к"),
        ("M", "М"),
        ("O", "О"),
        ("o", "о"),
        ("P", "Р"),
        ("p", "р"),
        ("T", "Т"),
        ("X", "Х"),
        ("x", "х"),
    ],
)
def test_repairs_every_approved_latin_lookalike_toward_cyrillic(
    latin: str, cyrillic: str
) -> None:
    assert repair_mixed_scripts(f"{cyrillic}{latin}{cyrillic}") == cyrillic * 3
