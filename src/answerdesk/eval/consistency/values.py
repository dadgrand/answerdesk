"""Value extraction and language identification for the consistency audit.

Fidelity is judged *deterministically*: the amounts, dates, identifiers, percentages and day
counts an answer asserts must appear in the gold document it cites. That check needs no judge,
which is what makes the cross-language numbers reviewable — the judge is used only for the
softer "does this answer state the same fact" question.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CURRENCY_TOKENS = (
    "RUB",
    "KGS",
    "TJS",
    "UZS",
    "KZT",
    "USD",
    "EUR",
    "CNY",
    "TRY",
    "₽",
    "₸",
    "元",
    "руб",
    "сом",
    "сум",
    "сомонӣ",
    "сум",
    "теңге",
    "лир",
    "юань",
)
DAY_TOKENS = (
    "день",
    "дня",
    "дней",
    "рабоч",
    "күн",
    "кун",
    "рӯз",
    "руз",
    "day",
    "days",
    "天",
    "iş",
    "gün",
    "kun",
)
MONTHS = (
    "январ",
    "феврал",
    "март",
    "апрел",
    "ма",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
    "январ",
    "феврал",
    "март",
    "апрел",
    "май",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
    "ocak",
    "şubat",
    "mart",
    "nisan",
    "mayıs",
    "haziran",
    "temmuz",
    "ağustos",
    "eylül",
    "ekim",
    "kasım",
    "aralık",
    "一月",
    "二月",
    "三月",
    "四月",
    "五月",
    "六月",
    "七月",
    "八月",
    "九月",
    "十月",
    "十一月",
    "十二月",
)

_ID = re.compile(r"\b[A-Z]{2,6}-\d{2,4}-\d{2,6}\b")
_LONG_NUMBER = re.compile(r"\b\d{6,}\b")
_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"\d[\d\s\u00a0.,]*")
_WORD_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class Values:
    amounts: tuple[str, ...]
    percentages: tuple[str, ...]
    day_counts: tuple[str, ...]
    dates: tuple[str, ...]
    identifiers: tuple[str, ...]

    def all(self) -> set[str]:
        return {
            *(f"amount:{value}" for value in self.amounts),
            *(f"pct:{value}" for value in self.percentages),
            *(f"days:{value}" for value in self.day_counts),
            *(f"date:{value}" for value in self.dates),
            *(f"id:{value}" for value in self.identifiers),
        }

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "amounts": list(self.amounts),
            "percentages": list(self.percentages),
            "day_counts": list(self.day_counts),
            "dates": list(self.dates),
            "identifiers": list(self.identifiers),
        }


def normalise_digits(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def _contextual_numbers(text: str, tokens: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    lowered = text.lower()
    for match in _NUMBER.finditer(lowered):
        window = lowered[max(0, match.start() - 24) : match.end() + 24]
        if any(token.lower() in window for token in tokens):
            cleaned = match.group(0).strip().rstrip(".,")
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned and any(character.isdigit() for character in cleaned):
                found.append(cleaned)
    return found


def extract(text: str) -> Values:
    """Canonical value set of one answer or document."""
    normalised = normalise_digits(text)
    lowered = normalised.lower()
    identifiers = [match.group(0) for match in _ID.finditer(normalised)]
    identifiers.extend(match.group(0) for match in _LONG_NUMBER.finditer(normalised))
    dates = [match.group(0) for match in _DATE.finditer(normalised)]
    for match in re.finditer(
        r"\b\d{1,2}\s+(?:" + "|".join(re.escape(month) for month in MONTHS) + r")\w*",
        lowered,
    ):
        dates.append(match.group(0).strip())
    percentages = [
        _WORD_NUMBER.search(match.group(0)).group(0)  # type: ignore[union-attr]
        for match in _NUMBER.finditer(normalised)
        if "%" in normalised[match.end() : match.end() + 2]
    ]
    day_counts = _contextual_numbers(normalised, DAY_TOKENS)
    amounts = _contextual_numbers(normalised, CURRENCY_TOKENS)
    return Values(
        amounts=tuple(sorted(set(amounts))),
        percentages=tuple(sorted(set(percentages))),
        day_counts=tuple(sorted(set(day_counts))),
        dates=tuple(sorted(set(dates))),
        identifiers=tuple(sorted(set(identifiers))),
    )


def fidelity(answer: str, gold_document: str) -> dict[str, object]:
    """Every value asserted in the answer must appear in the gold document."""
    asserted = extract(answer)
    supported = extract(gold_document)
    asserted_set = asserted.all()
    unsupported = asserted_set - supported.all()
    return {
        "asserted": sorted(asserted_set),
        "unsupported": sorted(unsupported),
        "asserted_count": len(asserted_set),
        "supported_count": len(asserted_set) - len(unsupported),
        "fidelity": (
            1.0 if not asserted_set else (len(asserted_set) - len(unsupported)) / len(asserted_set)
        ),
        "exact_match": bool(asserted_set) and not unsupported,
    }


# --------------------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------------------


def identify_language(text: str) -> tuple[str, float]:
    """Query-analysis language ID for arbitrary text, with a script fallback for short answers."""
    from answerdesk.retrieval.query_analysis import detect_language

    locale, confidence = detect_language(text)
    if locale != "und":
        return locale, float(confidence)
    counts = {"han": 0, "arabic": 0, "cyrillic": 0, "latin": 0}
    for character in text:
        if "\u4e00" <= character <= "\u9fff":
            counts["han"] += 1
        elif "\u0600" <= character <= "\u06ff":
            counts["arabic"] += 1
        elif "\u0400" <= character <= "\u04ff":
            counts["cyrillic"] += 1
        elif character.isalpha():
            counts["latin"] += 1
    total = sum(counts.values()) or 1
    if counts["han"] / total >= 0.3:
        return "zh", round(counts["han"] / total, 3)
    if counts["cyrillic"] / total >= 0.5:
        return "cyr", round(counts["cyrillic"] / total, 3)
    if counts["latin"] / total >= 0.5:
        return "lat", round(counts["latin"] / total, 3)
    return "und", 0.0


CYRILLIC_LOCALES = ("ru", "tg", "ky")
LATIN_LOCALES = ("uz", "tr")


def language_discipline(
    *, query_locale: str, answer: str, answer_language_setting: str | None = None
) -> dict[str, object]:
    """Does the answer use the query's language (or an explicitly documented fallback)?"""
    detected, confidence = identify_language(answer)
    expected = query_locale
    compatible = False
    if (
        detected == expected
        or (query_locale in CYRILLIC_LOCALES and detected == "cyr")
        or (query_locale in LATIN_LOCALES and detected == "lat")
    ):
        compatible = True
    elif detected == "und":
        compatible = False
    fallback_used = answer_language_setting not in (None, "", query_locale)
    return {
        "expected": expected,
        "detected": detected,
        "detected_confidence": confidence,
        "compatible": compatible,
        "fallback_declared": fallback_used,
        "ok": compatible or fallback_used,
    }


__all__ = [
    "CYRILLIC_LOCALES",
    "LATIN_LOCALES",
    "Values",
    "extract",
    "fidelity",
    "identify_language",
    "language_discipline",
    "normalise_digits",
]
