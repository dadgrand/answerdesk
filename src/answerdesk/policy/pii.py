"""PII detection and redaction: before the index, before the log, before the provider.

INV-05 and FR-ESC-08: phones, e-mails, card numbers and account-like numbers are detected
by deterministic, locale-aware rules and redacted at every boundary. The detectors never
store the matched text — a finding keeps a hashed token so a trace can correlate two
occurrences of the same value without carrying it (NFR-SEC-02, NFR-SEC-03).

Classification order matters, because the shapes overlap:

1. **e-mail** — unambiguous.
2. **card** — 13–19 digits passing the Luhn checksum (the checksum is what separates a card
   from an account number or a long date).
3. **phone** — 7–15 digits that either carry a ``+``/separator or are a bare 10–11 digit run
   (the RU/KZ/UZ mobile shape).
4. **account / IBAN** — remaining 8–20 digit runs and IBAN-shaped strings.

Date-like and amount-like strings are deliberately *not* PII: ``05.03.2026`` and ``1 000,50``
are answer content, and flagging them would make the guard fire on real policy text (a guard
that fires on normal content gets switched off). That exclusion list is tested.

    # AT-ESC-08  # AT-SEC-02
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

from answerdesk.contracts import Contract, Message, Role

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_DIGIT_RUN = re.compile(r"(?<![\w])\+?\d[\d\s().\-]{5,}\d(?![\w])")
_IBAN = re.compile(r"(?<![\w])[A-Z]{2}\d{2}[A-Z0-9]{10,30}(?![\w])")
_DATE_LIKE = re.compile(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$")
_THOUSANDS_LIKE = re.compile(r"^\d{1,3}(?:[ .,]\d{3})+$")
_TIME_LIKE = re.compile(r"^\d{1,2}:\d{2}$")
_BIDI = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


class PIIKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"
    ACCOUNT = "account"


PLACEHOLDERS: dict[PIIKind, str] = {
    PIIKind.EMAIL: "[EMAIL]",
    PIIKind.PHONE: "[PHONE]",
    PIIKind.CARD: "[CARD]",
    PIIKind.ACCOUNT: "[ACCOUNT]",
}


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a digit string (the card-number test)."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


class PIIFinding(Contract):
    """One detected span — coordinates, kind and a *hashed* token, never the value."""

    kind: PIIKind
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    length: int = Field(ge=0)
    token_hash: str = Field(min_length=8, description="sha256[:16] of the matched text")

    def __str__(self) -> str:  # pragma: no cover - logging convenience
        return f"{self.kind.value}@{self.start}-{self.end} ({self.token_hash[:8]})"


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _classify_digit_run(text: str, match: re.Match[str]) -> PIIKind | None:
    raw = match.group(0)
    if _DATE_LIKE.match(raw) or _THOUSANDS_LIKE.match(raw) or _TIME_LIKE.match(raw):
        return None
    digits = re.sub(r"\D", "", raw)
    if not 7 <= len(digits) <= 20:
        return None
    if 13 <= len(digits) <= 19 and luhn_valid(digits):
        return PIIKind.CARD
    has_prefix = raw.startswith("+")
    has_separator = bool(re.search(r"[\s().\-]", raw))
    if 7 <= len(digits) <= 15 and (has_prefix or has_separator or len(digits) in (10, 11)):
        return PIIKind.PHONE
    if 8 <= len(digits) <= 20:
        return PIIKind.ACCOUNT
    return None


def detect(text: str) -> list[PIIFinding]:
    """All PII findings in ``text``, ordered by position and non-overlapping."""
    if not text:
        return []
    findings: list[PIIFinding] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in claimed)

    for match in _EMAIL.finditer(text):
        start, end = match.span()
        findings.append(
            PIIFinding(
                kind=PIIKind.EMAIL,
                start=start,
                end=end,
                length=end - start,
                token_hash=_hash_token(match.group(0)),
            )
        )
        claimed.append((start, end))

    for match in _IBAN.finditer(text):
        start, end = match.span()
        if overlaps(start, end):
            continue
        findings.append(
            PIIFinding(
                kind=PIIKind.ACCOUNT,
                start=start,
                end=end,
                length=end - start,
                token_hash=_hash_token(match.group(0)),
            )
        )
        claimed.append((start, end))

    for match in _DIGIT_RUN.finditer(text):
        start, end = match.span()
        if overlaps(start, end):
            continue
        kind = _classify_digit_run(text, match)
        if kind is None:
            continue
        findings.append(
            PIIFinding(
                kind=kind,
                start=start,
                end=end,
                length=end - start,
                token_hash=_hash_token(match.group(0)),
            )
        )
        claimed.append((start, end))

    return sorted(findings, key=lambda finding: finding.start)


def contains_pii(text: str) -> bool:
    return bool(detect(text))


class RedactionResult(Contract):
    """Redacted text plus what was removed (as findings, never as values)."""

    text: str
    findings: list[PIIFinding] = Field(default_factory=list)

    @property
    def redacted_count(self) -> int:
        return len(self.findings)

    @property
    def kinds(self) -> list[str]:
        return sorted({finding.kind.value for finding in self.findings})


def redact(text: str) -> RedactionResult:
    """Replace every detected span with its kind placeholder (``[EMAIL]``, ``[CARD]`` …)."""
    findings = detect(text)
    if not findings:
        return RedactionResult(text=text, findings=[])
    pieces: list[str] = []
    cursor = 0
    for finding in findings:
        pieces.append(text[cursor : finding.start])
        pieces.append(PLACEHOLDERS[finding.kind])
        cursor = finding.end
    pieces.append(text[cursor:])
    return RedactionResult(text="".join(pieces), findings=findings)


def redact_messages(messages: Sequence[Message]) -> tuple[list[Message], list[PIIFinding]]:
    """Redact every message before it reaches an external provider (NFR-SEC-02 boundary 3)."""
    redacted: list[Message] = []
    findings: list[PIIFinding] = []
    for message in messages:
        result = redact(message.content)
        findings.extend(result.findings)
        redacted.append(Message(role=message.role, content=result.text))
    return redacted, findings


def scan_messages(messages: Sequence[Message]) -> list[PIIFinding]:
    """Detect without redacting (used by the pre-provider assertion test)."""
    findings: list[PIIFinding] = []
    for message in messages:
        findings.extend(detect(message.content))
    return findings


def redact_system_messages_only(
    messages: Sequence[Message], *, redact_roles: Sequence[Role] = (Role.SYSTEM, Role.USER)
) -> tuple[list[Message], list[PIIFinding]]:
    """Redact a subset of roles (kept for callers that must preserve assistant text)."""
    redacted: list[Message] = []
    findings: list[PIIFinding] = []
    allowed = set(redact_roles)
    for message in messages:
        if message.role in allowed:
            result = redact(message.content)
            findings.extend(result.findings)
            redacted.append(Message(role=message.role, content=result.text))
        else:
            redacted.append(message)
    return redacted, findings


__all__ = [
    "PLACEHOLDERS",
    "PIIFinding",
    "PIIKind",
    "RedactionResult",
    "contains_pii",
    "detect",
    "luhn_valid",
    "redact",
    "redact_messages",
    "redact_system_messages_only",
    "scan_messages",
]
