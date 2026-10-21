"""Deterministic numeric/date guard (INV-04, citation rule CR-9, FR-GEN-12).

A plausible wrong number — a fee, a limit, a deadline — is the most damaging thing a
support answer can contain, and support-checking models are weakest exactly there. So
the guard does not ask a model: it extracts every numeric and date token from a
displayed claim, canonicalises it, and requires it to appear in a chunk the claim
cites.

Canonicalisation is deliberately documented and small (a superset belongs to ingestion,
FR-ING-05; this module only needs equality to be *stable*):

===========================  ==========================  ==================
raw                          canonical token             note
===========================  ==========================  ==================
``1000``, ``1 000``          ``1000``                    spaces are thousand separators
``1,000`` / ``1.000``        ``1000``                    one separator + 3 digits
``12,5`` / ``12.50``         ``12.5``                    decimal, trailing zeros dropped
``05``                       ``5``                       leading zeros dropped
``05.03.2026``               ``5-3-2026``, ``5-3``       day-month-year, and day-month
``2026-03-05``               ``5-3-2026``, ``5-3``       ISO order
``5 марта`` / ``March 5``    ``5-3``                     month names: ru + en
``10:30``                    ``10:30``                   time of day
===========================  ==========================  ==================

A date emits both its full form and its day-month form, because a claim may legitimately
omit the year that the chunk states ("until 5 March" vs "5 марта 2026"). Both sides are
canonicalised by the same function, so equality is symmetric.

**Limitation (R-30, stated not hidden).** The guard sees *digit-bearing* tokens only. A
number written as words — "сто двадцать дней", "twenty days" — contains no token this
module extracts, so a claim that spells the number out is not checked against its evidence
by INV-04. Month *names* are understood (they are part of a date); numerals-as-words are not.
Closing this needs a locale-aware numeral-word extractor (ru/uz/ky/tg/tr/zh) feeding the
same canonicalisation seam; until then prompts ask for digits, and a reviewer should treat a
spelled-out number as unchecked rather than as verified. The defect is recorded as R-30 in
``docs/review-report.md``; the reason-code consequence is in ``docs/escalation.md`` §4.1.

    # AT-GEN-02  # AT-GEN-12
"""

from __future__ import annotations

import re

_MONTHS: dict[str, int] = {
    # Russian (genitive, as used in dates)
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
    # English
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

_TOKEN = re.compile(
    rf"(?P<iso>\b\d{{4}}-\d{{1,2}}-\d{{1,2}}\b)"
    rf"|(?P<dmy>\b\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}\b)"
    rf"|(?P<day_month>\b(?P<day1>\d{{1,2}})\s+(?P<month1>{_MONTH_ALT})\b)"
    rf"|(?P<month_day>\b(?P<month2>{_MONTH_ALT})\s+(?P<day2>\d{{1,2}})\b)"
    rf"|(?P<num>\d[\d\u00a0 .,:/]*\d|\d)",
    re.IGNORECASE,
)

_THOUSANDS = re.compile(r"^(?!0)\d{1,3}(?:[.]\d{3})+$")
_SPACES = re.compile(r"[\u00a0\u2009\u202f ]")


def _year4(raw: str) -> str:
    """Two-digit years are widened the way a reader would: 26 -> 2026, 96 -> 1996."""
    value = int(raw)
    if len(raw) <= 2:
        return str(2000 + value) if value < 70 else str(1900 + value)
    return str(value)


def canonical_number(raw: str) -> str | None:
    """Canonical form of a plain numeric token, or ``None`` when there is no digit."""
    token = _SPACES.sub("", raw.strip()).strip(".,:/")
    if not token or not any(char.isdigit() for char in token):
        return None
    if ":" in token:
        parts = [part for part in token.split(":") if part]
        if all(part.isdigit() for part in parts) and len(parts) >= 2:
            return ":".join(str(int(part)) for part in parts)
        return token
    if _THOUSANDS.match(token):
        return str(int(token.replace(".", "")))
    if token.isdigit():
        return str(int(token))
    match = re.fullmatch(r"(\d+)[.,](\d+)", token)
    if match is not None:
        whole, fraction = match.groups()
        if len(fraction) == 3 and 1 <= len(whole) <= 3 and whole != "0":
            # "1,000" / "1.000" / "150.000": a thousand separator, not a decimal.
            # A leading zero can never head a thousands group ("0.500" is a decimal).
            return str(int(whole + fraction))
        trimmed = fraction.rstrip("0")
        whole_int = str(int(whole))
        return whole_int if not trimmed else f"{whole_int}.{trimmed}"
    digits = re.sub(r"[^\d]", "", token)
    return str(int(digits)) if digits else None


def date_tokens(day: str, month: int, year: str | None = None) -> set[str]:
    """A date contributes its full form (when known) and its day-month form."""
    tokens = {f"{int(day)}-{month}"}
    if year is not None:
        tokens.add(f"{int(day)}-{month}-{_year4(year)}")
    return tokens


def canonical_numeric_tokens(text: str) -> set[str]:
    """Every canonical number/date token in ``text``."""
    tokens: set[str] = set()
    for match in _TOKEN.finditer(text.lower()):
        if match.group("iso"):
            year, month, day = match.group(0).split("-")
            tokens |= date_tokens(day, int(month), year)
            continue
        if match.group("dmy"):
            day, month, year = re.split(r"[./-]", match.group(0))
            tokens |= date_tokens(day, int(month), year)
            continue
        if match.group("day_month"):
            month_name = match.group("month1").lower()
            tokens |= date_tokens(match.group("day1"), _MONTHS[month_name])
            continue
        if match.group("month_day"):
            month_name = match.group("month2").lower()
            tokens |= date_tokens(match.group("day2"), _MONTHS[month_name])
            continue
        number = canonical_number(match.group("num"))
        if number is not None:
            tokens.add(number)
    return tokens


def numeric_tokens(text: str) -> set[str]:
    """Alias kept for readability at call sites (same function, public name)."""
    return canonical_numeric_tokens(text)


def numeric_violations(claim_text: str, chunk_texts: list[str]) -> list[str]:
    """Canonical claim numbers that appear in **none** of the cited chunks (CR-9)."""
    claim_tokens = canonical_numeric_tokens(claim_text)
    if not claim_tokens:
        return []
    evidence: set[str] = set()
    for chunk_text in chunk_texts:
        evidence |= canonical_numeric_tokens(chunk_text)
    return sorted(claim_tokens - evidence)


__all__ = [
    "canonical_number",
    "canonical_numeric_tokens",
    "date_tokens",
    "numeric_tokens",
    "numeric_violations",
]
