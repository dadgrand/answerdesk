"""Regression: business identifiers must not be classified as PII.

The phone classifier used to treat any 7-15 digit run with a hyphen as a phone,
so order codes such as ``ORD-2026-10314`` were redacted and escalated. Identifier
strings are business data, not personal data; real phone numbers must still be
detected (INV-05, FR-ESC-08).
"""

from __future__ import annotations

import pytest

from answerdesk.policy.pii import PIIKind, detect


@pytest.mark.parametrize(
    "text",
    [
        "ORD-2026-10314",
        "Заявка ORD-2026-10314 принята в работу",
        "TRX-2026-000123",
        "ACC_2026_10314",
        "ERR-2026-0042",
    ],
)
def test_business_identifiers_are_not_pii(text: str) -> None:
    assert detect(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "+7 999 123-45-67",
        "8-999-123-45-67",
        "89991234567",
        "+44 20 7946 0958",
    ],
)
def test_real_phone_numbers_are_still_detected(text: str) -> None:
    kinds = [finding.kind for finding in detect(text)]
    assert PIIKind.PHONE in kinds


def test_card_number_is_still_detected() -> None:
    kinds = [finding.kind for finding in detect("4111 1111 1111 1111")]
    assert PIIKind.CARD in kinds


def test_email_is_still_detected() -> None:
    kinds = [finding.kind for finding in detect("support@example.com")]
    assert PIIKind.EMAIL in kinds
