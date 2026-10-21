"""Canonicalisation of dates, money and identifiers (FR-ING-05).

The rule that makes retrieval work across six locales and five writing systems is:
**never rewrite the chunk text, always add a normalised sidecar.** :func:`canonicalize`
returns tokens and leaves its input untouched — a test asserts the chunk text is
byte-identical before and after (``tests/unit/ingest/test_canonical.py``).

=====================================  ===========================================
source form                            canonical token
=====================================  ===========================================
``5 марта 2026``, ``5 mart 2026``,     ``2026-03-05``
``2026-03-05``, ``05.03.2026``,
``3月5日``
``12,5 €``, ``€12.50``, ``12.50 EUR``  ``12.50 EUR``
``1 200 ₽``, ``1200 руб.``             ``1200.00 RUB``
``ORD-2026-000123``, ``err-trf-4012``  ``ORD-2026-000123`` (upper-cased)
``4000 0000 0000 0002``                ``4000000000000002`` + ``****0002``
=====================================  ===========================================

A date without a year (``5 марта``) is completed with
:data:`DEFAULT_REFERENCE_YEAR` — documented, fixed and part of the corpus version,
because a wall-clock year would make the corpus non-reproducible.
"""

from __future__ import annotations

import calendar
import re

from answerdesk.ingest.structure import IngestModel

#: Year used to complete a year-less date. Fixed on purpose (reproducibility).
DEFAULT_REFERENCE_YEAR = 2026

#: Month names per locale. Lower-case keys; a name maps to its 1-based month number.
MONTHS: dict[str, int] = {
    # Russian (genitive + nominative + common abbreviations)
    "января": 1,
    "январь": 1,
    "янв": 1,
    "февраля": 2,
    "февраль": 2,
    "фев": 2,
    "марта": 3,
    "март": 3,
    "мар": 3,
    "апреля": 4,
    "апрель": 4,
    "апр": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июн": 6,
    "июля": 7,
    "июль": 7,
    "июл": 7,
    "августа": 8,
    "август": 8,
    "авг": 8,
    "сентября": 9,
    "сентябрь": 9,
    "сен": 9,
    "сент": 9,
    "октября": 10,
    "октябрь": 10,
    "окт": 10,
    "ноября": 11,
    "ноябрь": 11,
    "ноя": 11,
    "декабря": 12,
    "декабрь": 12,
    "дек": 12,
    # Tajik (Cyrillic; the month stems drop the Russian soft sign)
    "январ": 1,
    "феврал": 2,
    "апрел": 4,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
    # Uzbek (Latin script); Kyrgyz uses the Russian forms plus "5-март"
    "yanvar": 1,
    "fevral": 2,
    "mart": 3,
    "aprel": 4,
    "may": 5,
    "iyun": 6,
    "iyul": 7,
    "avgust": 8,
    "sentabr": 9,
    "oktabr": 10,
    "noyabr": 11,
    "dekabr": 12,
    # Turkish
    "ocak": 1,
    "şubat": 2,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "kasım": 11,
    "aralık": 12,
}

#: Currency spelling → ISO-4217 code. Covers the six demo locales' natural forms.
CURRENCIES: dict[str, str] = {
    "€": "EUR",
    "eur": "EUR",
    "евро": "EUR",
    "$": "USD",
    "usd": "USD",
    "долл": "USD",
    "₽": "RUB",
    "rub": "RUB",
    "руб": "RUB",
    "рубль": "RUB",
    "рублей": "RUB",
    "tjs": "TJS",
    "сомони": "TJS",
    "uzs": "UZS",
    "сум": "UZS",
    "сўм": "UZS",
    "so'm": "UZS",
    "soʻm": "UZS",
    "kgs": "KGS",
    "сом": "KGS",
    "try": "TRY",
    "₺": "TRY",
    "лира": "TRY",
    "лир": "TRY",
    "cny": "CNY",
    "¥": "CNY",
    "юань": "CNY",
    "юаня": "CNY",
    "元": "CNY",
}

_MONTH_ALTERNATION = "|".join(sorted((re.escape(name) for name in MONTHS), key=len, reverse=True))
_MONTH_DATE_RE = re.compile(
    rf"(?<![\d.])(?P<day>\d{{1,2}})(?:-|\s)\s*"
    rf"(?P<month>{_MONTH_ALTERNATION})(?![\w])"
    rf"(?:\s*,?\s*(?P<year>\d{{4}}))?",
    re.IGNORECASE | re.UNICODE,
)
_ISO_DATE_RE = re.compile(r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?!\d)")
_NUMERIC_DATE_RE = re.compile(
    r"(?<![\d.])(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{4}|\d{2})(?!\d)"
)
_ZH_DATE_RE = re.compile(r"(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]")

_CURRENCY_ALTERNATION = "|".join(
    sorted((re.escape(name) for name in CURRENCIES), key=len, reverse=True)
)
_MONEY_SUFFIX_RE = re.compile(
    r"(?P<number>\d[\d\s\u00a0\u202f.,]*\d|\d)\s*(?P<currency>"
    + _CURRENCY_ALTERNATION
    + r")(?![\w])",
    re.IGNORECASE | re.UNICODE,
)
_MONEY_PREFIX_RE = re.compile(
    r"(?P<currency>" + _CURRENCY_ALTERNATION + r")\s?(?P<number>\d[\d\s\u00a0\u202f.,]*\d|\d)",
    re.IGNORECASE | re.UNICODE,
)

#: ``ORD-2026-000123``, ``ERR-TRF-4012``, ``TS-999`` — two or more code-like groups.
#: Case-insensitive: the canonical form is upper-case, but users type ``err-trf-4012``.
_CODE_RE = re.compile(r"(?<![\w-])(?P<code>[A-Za-z]{2,6}(?:-[A-Za-z0-9]{1,10}){1,4})(?![\w-])")
#: A card-like digit run (13–19 digits, optionally grouped by spaces or dashes).
_PAN_RE = re.compile(r"(?<![\d-])(?P<pan>(?:\d[ -]?){12,18}\d)(?![\d-])")

_MAX_TOKENS = 96


class CanonicalOptions(IngestModel):
    """Knobs of :func:`canonicalize` — fixed defaults, no environment reads."""

    reference_year: int = DEFAULT_REFERENCE_YEAR
    max_tokens: int = _MAX_TOKENS
    include_pan: bool = True


def canonicalize(text: str, options: CanonicalOptions | None = None) -> list[str]:
    """Ordered, de-duplicated canonical tokens of ``text`` (input is never modified)."""
    opts = options or CanonicalOptions()
    hits: list[tuple[int, str]] = []
    _collect_dates(text, opts, hits)
    _collect_money(text, hits)
    _collect_codes(text, hits)
    if opts.include_pan:
        _collect_pans(text, hits)
    hits.sort(key=lambda item: item[0])
    tokens: list[str] = []
    seen: set[str] = set()
    for _, token in hits:
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= opts.max_tokens:
            break
    return tokens


# --------------------------------------------------------------------------------------
# collectors
# --------------------------------------------------------------------------------------


def _collect_dates(text: str, opts: CanonicalOptions, hits: list[tuple[int, str]]) -> None:
    for match in _ISO_DATE_RE.finditer(text):
        value = _iso(int(match["year"]), int(match["month"]), int(match["day"]))
        if value:
            hits.append((match.start(), value))
    for match in _NUMERIC_DATE_RE.finditer(text):
        year = int(match["year"])
        if year < 100:
            year += 2000
        value = _iso(year, int(match["month"]), int(match["day"]))
        if value:
            hits.append((match.start(), value))
    for match in _ZH_DATE_RE.finditer(text):
        year = int(match["year"]) if match["year"] else opts.reference_year
        value = _iso(year, int(match["month"]), int(match["day"]))
        if value:
            hits.append((match.start(), value))
    for match in _MONTH_DATE_RE.finditer(text):
        month = MONTHS.get(match["month"].lower())
        if month is None:
            continue
        year = int(match["year"]) if match["year"] else opts.reference_year
        value = _iso(year, month, int(match["day"]))
        if value:
            hits.append((match.start(), value))


def _collect_money(text: str, hits: list[tuple[int, str]]) -> None:
    for regex in (_MONEY_SUFFIX_RE, _MONEY_PREFIX_RE):
        for match in regex.finditer(text):
            code = CURRENCIES.get(match["currency"].lower())
            amount = _parse_amount(match["number"])
            if code and amount is not None:
                hits.append((match.start(), f"{amount} {code}"))


def _collect_codes(text: str, hits: list[tuple[int, str]]) -> None:
    for match in _CODE_RE.finditer(text):
        hits.append((match.start(), match["code"].upper()))


def _collect_pans(text: str, hits: list[tuple[int, str]]) -> None:
    for match in _PAN_RE.finditer(text):
        digits = re.sub(r"[ -]", "", match["pan"])
        if not 13 <= len(digits) <= 19:
            continue
        hits.append((match.start(), digits))
        hits.append((match.start(), f"****{digits[-4:]}"))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _iso(year: int, month: int, day: int) -> str | None:
    """``2026-03-05`` when the triple is a real calendar date, else ``None``."""
    if not 1900 <= year <= 2100 or not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    if day > calendar.monthrange(year, month)[1]:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_amount(raw: str) -> str | None:
    """Normalise a locale money amount to ``"1234.50"``; ``None`` when unparseable."""
    text = raw.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if not text:
        return None
    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        head, _, tail = text.rpartition(",")
        text = f"{head}.{tail}" if 1 <= len(tail) <= 2 else text.replace(",", "")
    elif has_dot:
        head, _, tail = text.rpartition(".")
        if not 1 <= len(tail) <= 2:
            text = head.replace(".", "") + tail
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0 or value > 1e12:
        return None
    return f"{value:.2f}"


def luhn_valid(digits: str) -> bool:
    """Luhn checksum — used by the corpus generator and its identifier test."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


__all__ = [
    "CURRENCIES",
    "DEFAULT_REFERENCE_YEAR",
    "MONTHS",
    "CanonicalOptions",
    "canonicalize",
    "luhn_valid",
]
