"""Deterministic, multilingual tokenisation for the sparse index (FR-RET-01, FR-RET-09).

Three properties matter here, and every one of them is tested:

1. **Symmetric** — documents and queries go through the *same* function. A BM25 score is
   only meaningful when both sides use one vocabulary definition.
2. **Script-aware** — scripts without inter-word spaces (Han, Kana, Hangul, Thai) are
   emitted as characters plus character bigrams, because there is no segmenter in the
   base install (NFR-PORT-03) and a whole Han run as one token would never match.
3. **Identifier-preserving** — numerals, dates, money amounts, order ids and error codes
   are emitted twice: as ordinary lower-cased word tokens *and* as canonical tokens from
   :func:`answerdesk.ingest.canonical.canonicalize` (``ORD-2026-10009``, ``2026-03-05``,
   ``12.50 EUR``). Canonicalisation is **additive**: the raw token is never removed, so a
   query for ``TS-999`` matches both ``ts-999`` and ``TS-999`` in the corpus.

The module also owns :class:`TermVocabulary`, the corpus vocabulary used by
typo-tolerant query expansion and keyboard-layout repair. Keeping it next to the
tokeniser guarantees that "what counts as a known term" and "how text becomes terms"
cannot drift apart.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from answerdesk.ingest.canonical import canonicalize
from answerdesk.ingest.tokenizer import is_cjk_char

#: Bumped whenever tokenisation changes — part of the index version (FR-OPS-02).
TOKENIZER_VERSION = "retrieval-tok-v1"

#: Letters and digits. Underscore is deliberately excluded: ``snake_case`` and
#: ``HTTP_ERROR_500`` are two terms, not one, which matches how support text is written.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

#: Cyrillic letters sorted by frequency in Russian support text — used only for
#: ordering candidates deterministically, never for scoring.
_CYRILLIC_RANGE = (0x0400, 0x04FF)
_HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F))

#: Russian/English keyboard rows, same length by construction (see the layout test).
_RU_KEYS = "йцукенгшщзхъфывапролджэячсмитьбю"
_EN_KEYS = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
assert len(_RU_KEYS) == len(_EN_KEYS) == 32
_RU_TO_EN = str.maketrans(_RU_KEYS, _EN_KEYS)
_EN_TO_RU = str.maketrans(_EN_KEYS, _RU_KEYS)

#: Minimum term length before fuzzy expansion is attempted. Short terms carry too little
#: signal and fuzzy-matching them is how ``карта`` becomes ``карте`` by accident.
MIN_FUZZY_LEN = 5


def normalize_text(text: str) -> str:
    """NFKC + single-spaced whitespace. Case is **preserved** on purpose (FR-RET-09).

    Case folding is a ranking decision and lives in :func:`raw_terms`; the *normalised
    query* keeps its original case so that identifier fingerprints
    (``ORD-2026-10009``) stay byte-identical pre/post analysis.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def is_han_char(char: str) -> bool:
    """True for CJK ideographs (written without spaces)."""
    code = ord(char)
    return any(low <= code <= high for low, high in _HAN_RANGES)


def is_cyrillic_char(char: str) -> bool:
    return _CYRILLIC_RANGE[0] <= ord(char) <= _CYRILLIC_RANGE[1]


def script_counts(text: str) -> dict[str, int]:
    """Character counts per script class: ``cyrillic``, ``han``, ``latin``, ``other``."""
    counts = {"cyrillic": 0, "han": 0, "latin": 0, "other": 0}
    for char in text:
        if not char.isalpha():
            continue
        if is_cyrillic_char(char):
            counts["cyrillic"] += 1
        elif is_han_char(char):
            counts["han"] += 1
        elif char.isascii():
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def raw_terms(text: str) -> list[str]:
    """Word tokens in order, with repetition (term frequency is preserved).

    * Latin/Cyrillic/other alphabetic scripts: lower-cased word runs.
    * CJK runs: one token per character **plus** character bigrams, on both the document
      and the query side, so precision does not collapse to single characters.
    """
    terms: list[str] = []
    for word in _WORD_RE.findall(normalize_text(text)):
        if all(is_cjk_char(char) for char in word):
            terms.extend(word)
            terms.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            terms.append(word.lower())
    return terms


def canonical_terms(text: str) -> list[str]:
    """Canonical identifier/date/money tokens of ``text`` (upper-cased codes included).

    Reuses the ingestion canonicaliser so that a query and the corpus chunk it targets
    agree on the canonical form (FR-ING-05, FR-RET-01).
    """
    return canonicalize(text)


def identifier_tokens(text: str) -> list[str]:
    """Canonical tokens that carry an exact-match obligation (identifiers, numbers, dates).

    A token qualifies when it contains a digit: every form ``canonicalize`` produces does,
    and a digit is precisely what "must not be stemmed or fuzzy-matched away" protects
    (FR-RET-06, FR-RET-09).
    """
    return [token for token in canonical_terms(text) if any(char.isdigit() for char in token)]


def tokenize(text: str, *, with_canonical: bool = True) -> list[str]:
    """The one tokenisation function used for both documents and queries (FR-RET-01)."""
    terms = raw_terms(text)
    if with_canonical:
        seen = set(terms)
        for token in canonical_terms(text):
            lowered = token.lower()
            if lowered not in seen:
                seen.add(lowered)
                terms.append(lowered)
    return terms


def trigrams(term: str) -> frozenset[str]:
    """Character trigrams of a term, padded so short terms still have three of them."""
    padded = f"^{term}$"
    if len(padded) <= 3:
        return frozenset({padded})
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def trigram_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two trigram sets — the fuzzy-matching distance measure."""
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def layout_variant(term: str) -> str | None:
    """The same keystrokes typed in the other keyboard layout, or ``None``.

    ``ghbdtn`` -> ``привет`` and ``привет`` -> ``ghbdtn``. This is the permitted
    locale normalisation of FR-RET-09: it is offered as an *additional* candidate for a
    corpus lookup and is never applied destructively to the query text.
    """
    if not term:
        return None
    if all(char in _RU_KEYS for char in term):
        return term.translate(_RU_TO_EN)
    if all(char in _EN_KEYS for char in term):
        return term.translate(_EN_TO_RU)
    return None


class TermVocabulary:
    """The corpus vocabulary, with fuzzy lookup and layout repair (FR-RET-06).

    Built once per index build and passed to
    :class:`answerdesk.retrieval.query_analysis.QueryAnalyzer`; it answers exactly one
    question — *is this query term something the corpus knows, and if not, what is the
    nearest thing it does know?*
    """

    __slots__ = ("_terms", "_trigram_index")

    def __init__(self, token_lists: Iterable[Sequence[str]] | Iterable[str]) -> None:
        terms: set[str] = set()
        for item in token_lists:
            if isinstance(item, str):
                terms.add(item)
            else:
                terms.update(item)
        self._terms: frozenset[str] = frozenset(terms)
        index: dict[str, list[str]] = {}
        for term in sorted(self._terms):
            if len(term) < MIN_FUZZY_LEN:
                continue
            for gram in trigrams(term):
                index.setdefault(gram, []).append(term)
        self._trigram_index: dict[str, list[str]] = index

    @classmethod
    def from_index(cls, index: object) -> TermVocabulary:
        """Build from any object exposing ``tokens`` (the sparse index does)."""
        tokens = getattr(index, "tokens", None)
        if tokens is None:
            raise TypeError("expected an object with a 'tokens' attribute")
        return cls(tokens)

    def __contains__(self, term: object) -> bool:
        return term in self._terms

    def __len__(self) -> int:
        return len(self._terms)

    @property
    def terms(self) -> frozenset[str]:
        return self._terms

    def fuzzy(self, term: str, *, threshold: float = 0.6, limit: int = 3) -> list[str]:
        """Corpus terms close to ``term`` by trigram Jaccard, best first, deterministic.

        Deterministic ordering is ``(-similarity, term)``; without the ``term`` tie-break
        two equally-similar candidates would be ordered by dict insertion, which Python
        promises for a single process but not across a rebuild.
        """
        if len(term) < MIN_FUZZY_LEN or term in self._terms:
            return []
        query_trigrams = trigrams(term)
        candidates: dict[str, int] = {}
        for gram in query_trigrams:
            for candidate in self._trigram_index.get(gram, ()):
                candidates[candidate] = candidates.get(candidate, 0) + 1
        scored = [
            (-trigram_jaccard(query_trigrams, trigrams(candidate)), candidate)
            for candidate in candidates
        ]
        scored = [item for item in scored if -item[0] >= threshold]
        scored.sort()
        return [candidate for _, candidate in scored[:limit]]

    def repair_layout(self, term: str) -> str | None:
        """The other-layout spelling of ``term`` **if the corpus contains it**."""
        variant = layout_variant(term)
        if variant is not None and variant in self._terms:
            return variant
        return None


__all__ = [
    "MIN_FUZZY_LEN",
    "TOKENIZER_VERSION",
    "TermVocabulary",
    "canonical_terms",
    "identifier_tokens",
    "is_cjk_char",
    "is_cyrillic_char",
    "is_han_char",
    "layout_variant",
    "normalize_text",
    "raw_terms",
    "script_counts",
    "tokenize",
    "trigram_jaccard",
    "trigrams",
]
