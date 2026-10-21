"""Numeric/date guard: canonicalisation and the deterministic INV-04 check.

The guard exists because a plausible wrong number is the most damaging answer a support
bot can give. It must be *stable* (both sides canonicalised identically) and it must not
ask a model (FR-GEN-12: "deterministic and independent of the model").

    # AT-GEN-12  # AT-GEN-02
"""

from __future__ import annotations

import pytest

from answerdesk.generate.numeric import (
    canonical_number,
    canonical_numeric_tokens,
    numeric_violations,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1000", "1000"),
        ("1 000", "1000"),
        ("1,000", "1000"),
        ("1.000", "1000"),
        ("12,5", "12.5"),
        ("12.50", "12.5"),
        ("0.500", "0.5"),
        ("05", "5"),
        ("007", "7"),
        ("10:30", "10:30"),
        ("9:05", "9:5"),
        ("12.5", "12.5"),
    ],
)
def test_canonical_number_forms(raw: str, expected: str) -> None:
    """# AT-GEN-12 — the documented canonicalisation table."""
    assert canonical_number(raw) == expected


def test_canonical_number_rejects_text_without_digits() -> None:
    assert canonical_number("") is None
    assert canonical_number("рублей") is None


def test_amounts_and_percentages_compare_after_canonicalisation() -> None:
    """# AT-GEN-12 — "12,5 €" and "12.5" are the same amount."""
    claim = canonical_numeric_tokens("Комиссия 12,5 € за перевод")
    evidence = canonical_numeric_tokens("The fee is 12.50 per transfer.")
    assert "12.5" in claim
    assert claim & evidence == {"12.5"}


def test_thousand_separators_are_not_decimals() -> None:
    assert canonical_numeric_tokens("150 000 рублей") == {"150000"}
    assert canonical_numeric_tokens("150,000 RUB") == {"150000"}
    assert canonical_numeric_tokens("0,150") == {"0.15"}


def test_dates_match_across_formats() -> None:
    """# AT-GEN-12 — a date is the same date whether written ISO, dotted or with a month name."""
    iso = canonical_numeric_tokens("до 2026-03-05")
    dotted = canonical_numeric_tokens("до 05.03.2026")
    named_ru = canonical_numeric_tokens("до 5 марта")
    named_en = canonical_numeric_tokens("until March 5")
    assert "5-3-2026" in iso and "5-3" in iso
    assert "5-3-2026" in dotted
    assert "5-3" in named_ru
    assert "5-3" in named_en
    assert iso & dotted == {"5-3-2026", "5-3"}
    assert named_ru <= iso | dotted


def test_two_digit_years_are_widened() -> None:
    assert "5-3-2026" in canonical_numeric_tokens("05.03.26")
    assert "5-3-1996" in canonical_numeric_tokens("05.03.96")


def test_identifiers_are_compared_by_their_numeric_part() -> None:
    """An error code is not a number, but its digits still have to appear."""
    assert canonical_numeric_tokens("код ошибки TS-999") == {"999"}


def test_violation_reports_missing_tokens() -> None:
    """# AT-GEN-12 — the guard's answer is a list of numbers, not a boolean guess."""
    assert numeric_violations("Комиссия 999 рублей.", ["Комиссия 500 рублей."]) == ["999"]
    assert numeric_violations("Срок 14 дней.", ["within 14 days of approval"]) == []
    assert numeric_violations("Без чисел вообще.", ["anything"]) == []
    assert numeric_violations("Срок 14 дней.", []) == ["14"]


def test_guard_is_symmetric_for_equivalent_writings() -> None:
    claim = "Перевод до 50 000 рублей"
    evidence = "Лимит на один перевод — 50,000 рублей."
    assert numeric_violations(claim, [evidence]) == []


def test_multiple_numbers_must_all_be_present() -> None:
    assert numeric_violations("От 1 000 до 5 000 рублей", ["от 1000 рублей"]) == ["5000"]
