"""Cache keys: normalisation, the version tuple and the entity guard.

Requirements §9.1 says a cache entry is keyed by ``{normalized_query, locale, entity_set,
corpus_version, index_version, prompt_version, model_id, policy_artifact_hash}``. Two of
those deserve care:

* **normalisation** must be aggressive enough that "Сколько стоит карта?" and "сколько
  стоит карта" are the same query, but must not change identifiers, numbers or locale —
  NFKC + case-fold + whitespace collapse, never transliteration (NFR-ML-02, FR-RET-09).
* **the entity set** is the guard that stops a semantically similar question about a
  *different* plan/amount/date/order from hitting (G-2, §9.2). It is built from the same
  canonicalisation the numeric guard uses, so "12,5 €" and "12.50" are the same entity.

The exact key is a content hash over the whole tuple, which makes exact lookup O(1) and
makes "the same query with a different corpus version" a guaranteed miss.

    # AT-CAC-01  # AT-CAC-02
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from pydantic import Field

from answerdesk.contracts import Contract, IntentBucket
from answerdesk.generate.numeric import canonical_numeric_tokens

#: Identifier shapes the entity guard treats as answer-relevant literals.
_IDENTIFIER = re.compile(r"\b[A-Z]{2,6}-\d{2,6}\b|\b[A-Z]{1,3}\d{4,10}\b")
_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCTUATION = " \t\n\r?!.,;:»«\"'()[]"


def normalise_query(text: str) -> str:
    """NFKC, case-fold, collapse whitespace, trim punctuation at the edges."""
    normalised = unicodedata.normalize("NFKC", text).casefold()
    normalised = _WHITESPACE.sub(" ", normalised).strip()
    return normalised.strip(_EDGE_PUNCTUATION)


class EntitySet(Contract):
    """The answer-relevant literals of a query: identifiers and canonical numbers."""

    identifiers: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.identifiers and not self.numbers

    def fingerprint(self) -> str:
        """Stable string form used inside the exact key."""
        return "|".join(
            [
                ",".join(sorted(set(self.identifiers))),
                ",".join(sorted(set(self.numbers))),
            ]
        )

    def matches(self, other: EntitySet) -> bool:
        """Exact equality of both sets — the G-2 guard (no subset semantics)."""
        return set(self.identifiers) == set(other.identifiers) and set(self.numbers) == set(
            other.numbers
        )

    def describes_only(self, other: EntitySet) -> bool:
        """True when ``other`` mentions a subset of these entities (diagnostic helper)."""
        return set(other.identifiers) <= set(self.identifiers) and set(other.numbers) <= set(
            self.numbers
        )


def extract_entities(text: str) -> EntitySet:
    """Identify order/error-code literals and canonical numeric tokens."""
    normalised = unicodedata.normalize("NFKC", text)
    identifiers = sorted({match.group(0).upper() for match in _IDENTIFIER.finditer(normalised)})
    numbers = sorted(canonical_numeric_tokens(normalised))
    return EntitySet(identifiers=identifiers, numbers=numbers)


class VersionTuple(Contract):
    """Everything that must be identical for a cached answer to still be valid (G-1)."""

    corpus_version: str = ""
    index_version: str = ""
    prompt_version: str = ""
    model_id: str = ""
    policy_artifact_hash: str = ""

    def matches(self, other: VersionTuple) -> bool:
        return self == other

    def fingerprint(self) -> str:
        return "|".join(
            [
                self.corpus_version,
                self.index_version,
                self.prompt_version,
                self.model_id,
                self.policy_artifact_hash,
            ]
        )


class CacheKey(Contract):
    """The full key of §9.1."""

    normalised_query: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    intent_bucket: IntentBucket = IntentBucket.INFORMATIONAL
    entities: EntitySet = Field(default_factory=EntitySet)
    versions: VersionTuple = Field(default_factory=VersionTuple)

    def canonical(self) -> dict[str, object]:
        return {
            "query": self.normalised_query,
            "locale": self.locale,
            "intent": self.intent_bucket.value,
            "entities": self.entities.fingerprint(),
            "versions": self.versions.fingerprint(),
        }

    def exact_key(self) -> str:
        material = json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @classmethod
    def build(
        cls,
        *,
        query: str,
        locale: str,
        intent_bucket: IntentBucket,
        versions: VersionTuple,
    ) -> CacheKey:
        return cls(
            normalised_query=normalise_query(query),
            locale=locale,
            intent_bucket=intent_bucket,
            entities=extract_entities(query),
            versions=versions,
        )


def cohort_of(
    *,
    intent_bucket: IntentBucket,
    locale: str,
    query: str,
    length_bands: tuple[int, ...] = (25, 60),
) -> str:
    """Cohort id: intent × locale family × question-length band (requirements §9.3)."""
    length = len(normalise_query(query))
    band = 0
    for index, limit in enumerate(length_bands):
        if length > limit:
            band = index + 1
    family = locale_family(locale)
    return f"{intent_bucket.value}:{family}:len{band}"


#: Locale families for cohort grouping (per requirement §9.3 "locale family").
_LOCALE_FAMILIES: dict[str, str] = {
    "ru": "cyrillic",
    "tg": "cyrillic",
    "ky": "cyrillic",
    "uz": "latin-uz",
    "tr": "latin-tr",
    "zh": "cjk",
    "en": "latin-en",
}


def locale_family(locale: str) -> str:
    return _LOCALE_FAMILIES.get(locale, locale)


__all__ = [
    "CacheKey",
    "EntitySet",
    "VersionTuple",
    "cohort_of",
    "extract_entities",
    "locale_family",
    "normalise_query",
]
