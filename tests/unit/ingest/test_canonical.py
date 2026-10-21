"""Canonicalisation tests: dates, money, identifiers (FR-ING-05).

# AT-ING-05
"""

from __future__ import annotations

import pytest

from answerdesk.ingest.canonical import CanonicalOptions, canonicalize, luhn_valid


def test_russian_genitive_dates_become_iso() -> None:
    """AT-ING-05: ``5 марта 2026`` -> ``2026-03-05`` (the requirement's own example)."""
    assert "2026-03-05" in canonicalize("Возврат придёт 5 марта 2026 года.")
    assert canonicalize("5 марта") == ["2026-03-05"]  # year-less -> fixed reference year


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-05", "2026-03-05"),
        ("05.03.2026", "2026-03-05"),
        ("5/3/2026", "2026-03-05"),
        ("5 mart 2026", "2026-03-05"),
        ("5-mart 2026", "2026-03-05"),
        ("5-март 2026", "2026-03-05"),
        ("5 Ocak 2026", "2026-01-05"),
        ("2026年3月5日", "2026-03-05"),
        ("3月5日", "2026-03-05"),
    ],
)
def test_dates_in_every_locale_format(text: str, expected: str) -> None:
    """AT-ING-05: one ISO token per locale's date style (six locales, five scripts)."""
    assert expected in canonicalize(f"Срок: {text}.")


def test_impossible_dates_are_not_invented() -> None:
    """AT-ING-05: ``31 февраля`` is not a date; no token is produced for it."""
    assert canonicalize("31 февраля 2026") == []


def test_money_is_normalised_to_iso_code_and_two_decimals() -> None:
    """AT-ING-05: ``12,5 €`` -> ``12.50 EUR``; ``1 200 ₽`` -> ``1200.00 RUB``."""
    assert "12.50 EUR" in canonicalize("Комиссия 12,5 € за перевод.")
    assert "12.50 EUR" in canonicalize("Fee €12.50 per transfer.")
    assert "12.50 EUR" in canonicalize("Fee 12.50 EUR per transfer.")
    assert "1200.00 RUB" in canonicalize("Лимит 1 200 ₽ в сутки.")
    assert "35.00 TJS" in canonicalize("Плата 35 сомони.")
    assert "99.90 TRY" in canonicalize("Ücret 99,90 TL değil, 99,90 TRY.")


def test_plain_numbers_are_not_money() -> None:
    """AT-ING-05: ``14 дней`` must not become a money token."""
    assert canonicalize("Возврат за 14 дней") == []


def test_identifiers_are_upper_cased_and_kept() -> None:
    """AT-ING-05: order ids and error codes survive canonicalisation, upper-cased."""
    tokens = canonicalize("Заказ ord-2026-000123, ошибка err-trf-4012, код TS-999.")
    assert "ORD-2026-000123" in tokens
    assert "ERR-TRF-4012" in tokens
    assert "TS-999" in tokens


def test_card_numbers_are_reduced_and_masked() -> None:
    """AT-ING-05: a PAN yields the digits and a masked form (never a truncated prefix)."""
    tokens = canonicalize("Карта 4000 0000 0000 0002 заблокирована.")
    assert "4000000000000002" in tokens
    assert "****0002" in tokens


def test_original_text_is_never_modified() -> None:
    """AT-ING-05: canonicalisation is a sidecar; the source string is byte-identical."""
    text = "5 марта 2026 комиссия 12,5 €, заказ ORD-2026-000123."
    before = text.encode("utf-8")
    canonicalize(text)
    assert text.encode("utf-8") == before


def test_tokens_are_ordered_and_unique() -> None:
    """AT-ING-05: tokens appear in text order and are de-duplicated."""
    tokens = canonicalize("12,5 € ... снова 12,5 € ... и 5 марта 2026.")
    assert tokens == ["12.50 EUR", "2026-03-05"]


def test_token_budget_is_respected() -> None:
    """AT-ING-05: the sidecar is bounded so a chunk's metadata cannot explode."""
    text = " ".join(f"{day}.03.2026" for day in range(1, 29))
    options = CanonicalOptions(max_tokens=5)
    assert len(canonicalize(text, options)) == 5


def test_luhn_helper_matches_known_test_pans() -> None:
    """AT-ING-05 (§13.4): the documented test PANs are the only Luhn-valid ones we emit."""
    assert luhn_valid("4000000000000002")
    assert luhn_valid("5555555555554444")
    assert not luhn_valid("4000000000000001")
    assert not luhn_valid("not-digits")
