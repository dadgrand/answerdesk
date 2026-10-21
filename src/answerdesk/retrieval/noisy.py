"""Noisy-input handling: run-on segmentation, transliteration and bounded fuzzy matching.

Real support messages are not typed like the golden set. The owner's three examples are the
specimen:

    "как займ оформит"                                  telegraphic, verb truncated
    "деньги как получит"                                telegraphic, word order mangled
    "денги первод таджикистан сделал денги нет брату не дошли"
                                                        run-on, no punctuation, two typos,
                                                        mixed intent (transfer + not arrived)

The module provides four *additive* utilities. Nothing here ever rewrites or shortens the
user's query — every result is an extra candidate term or an extra probe for the sparse
channel, and the raw query remains its own arm (the invariant of ``FR-RET-06``).

1. :func:`segment_clauses` — split a run-on message on punctuation and clause-initial
   conjunctions into at most a handful of clauses, each of which can be probed separately.
2. :func:`to_cyrillic` / :func:`transliteration_candidates` — rule-based Latin→Cyrillic for
   users who type Russian with a Latin keyboard layout ("zaym", "perevod", "dengi").
3. :class:`FuzzyMatcher` — trigram-prefiltered, bounded Levenshtein matching against the
   corpus vocabulary. This is what catches ``денги → деньги`` and ``первод → перевод``,
   which character-trigram Jaccard misses (0.375 and 0.444, below any sane threshold).
4. :func:`detect_noise_style` — a diagnostic label (telegraphic / run_on / typo_mix / translit)
   used by the ablation breakdown and the tests. It is **not** a routing decision.

Cost is bounded by construction: candidate generation is a trigram lookup (already cached in
the vocabulary), and the edit distance is only ever computed against a fixed pool of 24
candidates, with a banded dynamic program that bails out as soon as it exceeds the allowed
number of edits. Measured on the 8,250-term vocabulary: ~0.04 ms per unknown term, i.e.
0.3 ms for a whole run-on message (``docs/retrieval.md`` §15.3).

Honest limits are in ``docs/retrieval.md`` §15.5. The short version: a word that is not in
the corpus cannot be matched to anything, dropped prepositions are not inferable from
surface form, and a message carrying two intents is a *ranking* problem this module only
softens (each clause gets its own probe; which clause wins is decided by fusion scores).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from answerdesk.index.text import is_cyrillic_char, is_han_char
from answerdesk.retrieval.words import (
    CYRILLIC_LOCALES,
    all_function_words,
    clause_words,
    content_terms,
    function_words,
)

#: Longest a clause may be before segmentation gives up and keeps the whole message.
MAX_CLAUSE_WORDS = 40

#: Clauses shorter than this (in words) are merged into their neighbour instead of probed.
MIN_CLAUSE_WORDS = 2

#: Punctuation that separates clauses in any of the shipped locales (no hyphen: it is part
#: of words such as ``корт-центр``).
_CLAUSE_PUNCT = re.compile(r"[,.;:!?…—–()\[\]{}\"'«»”“]+", re.UNICODE)

#: A word token (letters/digits), used for the segmentation heuristics.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Latin→Cyrillic table for transliterated Russian/Uzbek-in-Cyrillic typing. Multi-letter
#: sequences come first so ``sch``/``zh``/``yu`` win over their first letter.
_TRANSLIT: dict[str, str] = {
    "sch": "щ",
    "shch": "щ",
    "zh": "ж",
    "kh": "х",
    "ch": "ч",
    "sh": "ш",
    "yu": "ю",
    "ya": "я",
    "yo": "ё",
    "ye": "е",
    "ts": "ц",
    "a": "а",
    "b": "б",
    "v": "в",
    "g": "г",
    "d": "д",
    "e": "е",
    "z": "з",
    "i": "и",
    "y": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "f": "ф",
    "h": "х",
    "c": "ц",
    "j": "й",
    "q": "к",
    "w": "в",
    "x": "кс",
}

#: Negation particles per locale. In a run-on message a negation very often starts the
#: second clause ("... сделал денег **нет** брату не дошли"), so it is a usable boundary even
#: though no conjunction is present.
NEGATIONS: dict[str, frozenset[str]] = {
    "ru": frozenset(["не", "нет", "ни"]),
    "tg": frozenset(["не", "нест", "на"]),
    "ky": frozenset(["эмес", "жок"]),
    "tr": frozenset(["değil", "yok"]),
    "uz": frozenset(["yo'q", "emas", "yoq"]),
    "en": frozenset(["not", "no"]),
}

#: Longest fuzzy candidate worth matching; below this, edit distance 1 is meaningless.
MIN_FUZZY_LEN = 5

#: Candidates verified per term. Bounds the edit-distance cost to a constant.
MAX_FUZZY_CANDIDATES = 24

#: Trigram overlap below this yields no candidate at all (cheap early exit).
MIN_SHARED_TRIGRAMS = 1


def normalize_for_matching(text: str) -> str:
    """NFKC + lower-case + single spaces: the form every heuristic here works on."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def segment_clauses(
    text: str,
    *,
    locale: str = "und",
    max_clauses: int = 4,
    min_words: int = MIN_CLAUSE_WORDS,
) -> list[str]:
    """Split a run-on message into clauses, or return ``[text]`` when it is not run-on.

    Boundary signals, tried in this order and each requiring at least ``min_words`` words on
    both sides so ordinary sentences are not shredded:

    1. **punctuation** — a comma/semicolon/dash is an explicit clause boundary;
    2. **clause-initial conjunction** (:data:`answerdesk.retrieval.words.CLAUSE_WORDS`);
    3. **negation restart** — run-on text often continues with a negation rather than a
       conjunction ("… сделал денги **нет** брату не дошли");
    4. **repeated content token** — a self-repair ("денги … денги нет") marks a restart;
    5. **balanced long split** — for a message of ≥8 words with none of the above, cut at the
       most function-word-ish boundary near the middle: two probes for two possible intents
       are strictly better than one diluted bag of words, and the whole message is still its
       own arm.

    The result is capped at ``max_clauses``; the remainder is appended to the last clause
    (never dropped).
    """
    normalized = text.strip()
    if not normalized:
        return [text]

    conjunctions = clause_words(locale) if locale != "und" else all_function_words()
    pieces: list[list[str]] = []
    for chunk in _CLAUSE_PUNCT.split(normalized):
        if chunk.strip():
            pieces.extend(_split_on_conjunctions(chunk.split(), conjunctions, min_words))

    if len(pieces) == 1:
        restart = _restart_split(pieces[0], locale=locale, min_words=min_words)
        if len(restart) > 1:
            pieces = restart

    if len(pieces) == 1:
        return [text]  # no boundary found: return the message verbatim, punctuation included
    merged = _merge_short([" ".join(words) for words in pieces], min_words)
    if len(merged) <= 1:
        return [text]
    if len(merged) <= max_clauses:
        return merged
    head = merged[: max_clauses - 1]
    tail = " ".join(merged[max_clauses - 1 :])
    return [*head, tail]


def _split_on_conjunctions(
    words: list[str], stop: frozenset[str], min_words: int
) -> list[list[str]]:
    if len(words) < 2 * min_words:
        return [words]
    cuts: list[int] = []
    for index, word in enumerate(words):
        if index < min_words or len(words) - index < min_words:
            continue
        if word.lower().strip(",.;:!?") not in stop:
            continue
        cuts.append(index)
        if len(cuts) >= 3:
            break
    if not cuts:
        return [words]
    clauses: list[list[str]] = []
    start = 0
    for cut in cuts:
        clauses.append(words[start:cut])
        start = cut + 1  # the conjunction itself is dropped from the following clause
    clauses.append(words[start:])
    return [clause for clause in clauses if clause]


def _restart_split(words: list[str], *, locale: str, min_words: int) -> list[list[str]]:
    """Find a restart boundary when no punctuation or conjunction exists."""
    if len(words) < 2 * min_words:
        return [words]
    lowered = [word.lower().strip(",.;:!?") for word in words]

    negations = (
        NEGATIONS.get(locale, NEGATIONS["ru"] | NEGATIONS["en"])
        if locale != "und"
        else (frozenset().union(*NEGATIONS.values()))
    )
    for index in range(min_words, len(words) - min_words + 1):
        if lowered[index] in negations:
            return [words[:index], words[index:]]

    content = content_terms(lowered, locale=locale)
    seen_content: set[str] = set()
    for index in range(min_words, len(words) - min_words + 1):
        token = lowered[index]
        if token in content and token in seen_content:
            return [words[:index], words[index:]]
    for token in content:
        seen_content.add(token)

    if len(words) >= 8:
        stop = function_words(locale) if locale != "und" else all_function_words()
        middle = len(words) // 2
        for offset in range(0, len(words) // 2):
            for index in (middle - offset, middle + offset):
                if min_words <= index <= len(words) - min_words and lowered[index] in stop:
                    return [words[:index], words[index:]]
        return (
            [words[:middle], words[middle:]]
            if min_words <= middle <= len(words) - min_words
            else [words]
        )
    return [words]


def _merge_short(pieces: Sequence[str], min_words: int) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece.split()) < min_words:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def to_cyrillic(term: str) -> str:
    """Rule-based Latin→Cyrillic transliteration of one term (deterministic, total)."""
    lowered = term.lower()
    out: list[str] = []
    index = 0
    while index < len(lowered):
        for size in (4, 3, 2, 1):
            chunk = lowered[index : index + size]
            replacement = _TRANSLIT.get(chunk)
            if replacement is not None:
                out.append(replacement)
                index += size
                break
        else:  # pragma: no cover - single characters always hit the table
            out.append(lowered[index])
            index += 1
    return "".join(out)


def looks_transliterated(term: str, *, locale: str) -> bool:
    """True when a Latin term appears in a query written in a Cyrillic locale."""
    return locale in CYRILLIC_LOCALES and term.isascii() and term.isalpha() and len(term) >= 3


class FuzzyMatcher:
    """Bounded edit-distance matching over a fixed vocabulary (cached trigram index).

    The pool is built once and reused for every query: a trigram → terms map, exactly like
    the vocabulary's own fuzzy index but local to this module so the retrieval package owns
    its noise policy. Each ``match`` call does one dictionary probe per trigram of the term,
    ranks candidates by shared trigrams, and verifies at most
    :data:`MAX_FUZZY_CANDIDATES` of them with a banded Levenshtein distance.
    """

    __slots__ = ("_index", "_max_edits_long", "_max_edits_short", "_terms")

    def __init__(self, terms: Iterable[str], *, max_edits_short: int = 1, max_edits_long: int = 2):
        self._terms: frozenset[str] = frozenset(terms)
        self._index: dict[str, tuple[str, ...]] | None = None
        self._max_edits_short = max_edits_short
        self._max_edits_long = max_edits_long

    # -- construction -------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._terms)

    def _trigram_index(self) -> dict[str, tuple[str, ...]]:
        if self._index is None:
            index: dict[str, list[str]] = {}
            for term in sorted(self._terms):
                if len(term) < MIN_FUZZY_LEN:
                    continue
                for gram in trigrams(term):
                    index.setdefault(gram, []).append(term)
            self._index = {gram: tuple(values) for gram, values in index.items()}
        return self._index

    # -- matching -----------------------------------------------------------------
    def max_edits(self, term: str) -> int:
        """Allowed edits: 1 for short terms (5–7 chars), 2 for longer ones.

        Distance 2 on a five-letter word matches half the language (``денги`` → ``день``,
        ``дни``), so short terms are held to one edit.
        """
        return self._max_edits_short if len(term) <= 7 else self._max_edits_long

    def candidates(self, term: str) -> list[str]:
        """Pool of vocabulary terms sharing at least one trigram, best-first by overlap."""
        if len(term) < MIN_FUZZY_LEN or term in self._terms:
            return []
        index = self._trigram_index()
        counts: dict[str, int] = {}
        for gram in trigrams(term):
            for candidate in index.get(gram, ()):
                counts[candidate] = counts.get(candidate, 0) + 1
        if not counts:
            return []
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        pool = [(candidate, count) for candidate, count in ranked[:MAX_FUZZY_CANDIDATES]]
        return [candidate for candidate, count in pool if count >= MIN_SHARED_TRIGRAMS]

    def match(self, term: str, *, limit: int = 3) -> list[tuple[str, int, float]]:
        """``[(candidate, distance, jaccard)]``, best first, bounded and deterministic.

        Ranking is ``(distance, −anchors, −jaccard, candidate)`` where *anchors* is the
        length of the common prefix plus the common suffix. Distance alone is not enough:
        ``первод`` is one edit from both ``перевод`` and ``первой``, and only the anchoring
        term prefers the word a human would have typed.
        """
        if limit <= 0:
            return []
        allowed = self.max_edits(term)
        scored: list[tuple[int, int, int, float, str]] = []
        for candidate in self.candidates(term):
            distance = _banded_distance(term, candidate, allowed)
            if distance == 0 or distance > allowed:
                continue
            suffix, prefix = _anchors(term, candidate)
            scored.append(
                (
                    distance,
                    -suffix,
                    -prefix,
                    -trigram_jaccard(term, candidate),
                    candidate,
                )
            )
        scored.sort()
        return [
            (candidate, distance, round(-jaccard, 4))
            for distance, _suffix, _prefix, jaccard, candidate in scored[:limit]
        ]


def trigrams(term: str) -> frozenset[str]:
    """Character trigrams with ``^``/``$`` padding (same definition as the index)."""
    padded = f"^{term}$"
    if len(padded) <= 3:
        return frozenset({padded})
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def trigram_jaccard(left: str, right: str) -> float:
    """Jaccard similarity of two terms' trigram sets."""
    a, b = trigrams(left), trigrams(right)
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _anchors(left: str, right: str) -> tuple[int, int]:
    """``(common suffix, common prefix)`` — where a typo did *not* land.

    Suffix is ranked before prefix because a mistyped ending is the common case in these
    inflected languages: ``первод`` -> ``перевод`` keeps ``пер`` *and* the final ``д``, while
    the decoy ``первой`` keeps a longer prefix but the wrong ending.
    """
    prefix = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        prefix += 1
    longest = min(len(left), len(right))
    if longest > 1:
        prefix = min(prefix, longest - 1)
    suffix = 0
    for a, b in zip(reversed(left), reversed(right), strict=False):
        if a != b or suffix >= longest - prefix:
            break
        suffix += 1
    return suffix, prefix


def _banded_distance(left: str, right: str, limit: int) -> int:
    """Levenshtein distance with an early exit; returns ``limit + 1`` when it exceeds."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        best = current[0]
        for j, other in enumerate(right, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char != other),
            )
            if current[j] < best:
                best = current[j]
        if best > limit:
            return limit + 1
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class NoiseProfile:
    """Diagnostic view of why a query needed help — reporting only, never a decision."""

    style: str
    clauses: int
    content_share: float
    unknown_terms: int
    fuzzy_terms: int
    transliterated_terms: int

    @property
    def is_noisy(self) -> bool:
        return self.style != "clean"


def detect_noise_style(
    text: str,
    *,
    locale: str,
    terms: Sequence[str],
    vocabulary: frozenset[str] | None = None,
    matcher: FuzzyMatcher | None = None,
) -> NoiseProfile:
    """Classify the noise pattern of a message (diagnostic only).

    Precedence: ``translit`` > ``typo_mix`` > ``run_on`` > ``telegraphic`` > ``clean``.
    ``dropped_preposition`` is deliberately **not** detected: a message with a missing
    preposition is surface-identical to a clean one, so claiming to recognise it would be
    a lie (see the limits in ``docs/retrieval.md`` §15.5).
    """
    normalized = normalize_for_matching(text)
    word_terms = [word for word in _WORD_RE.findall(normalized)]
    content = content_terms(word_terms, locale=locale)
    share = (len(content) / len(word_terms)) if word_terms else 0.0
    clauses = segment_clauses(normalized, locale=locale)
    unknown = [term for term in content if vocabulary is not None and term not in vocabulary]
    transliterated = [term for term in content if looks_transliterated(term, locale=locale)]
    fuzzy_hits = 0
    if matcher is not None:
        fuzzy_hits = sum(1 for term in unknown if matcher.match(term, limit=1))

    style = "clean"
    if transliterated:
        style = "translit"
    elif len(clauses) > 1:
        style = "run_on"
    elif fuzzy_hits:
        style = "typo_mix"
    elif content and len(word_terms) <= 5 and share <= 0.6:
        style = "telegraphic"
    return NoiseProfile(
        style=style,
        clauses=len(clauses),
        content_share=round(share, 3),
        unknown_terms=len(unknown),
        fuzzy_terms=fuzzy_hits,
        transliterated_terms=len(transliterated),
    )


def transliteration_candidates(
    term: str,
    *,
    vocabulary: frozenset[str],
    matcher: FuzzyMatcher | None = None,
    limit: int = 2,
) -> list[str]:
    """Cyrillic readings of a Latin term that the corpus actually contains.

    Exact vocabulary hits rank first; otherwise the transliterated form is passed through the
    same bounded fuzzy matcher (``dengi`` → ``денги`` → ``деньги``), which is why the two
    mechanisms are complementary rather than redundant.
    """
    cyrillic = to_cyrillic(term)
    if not cyrillic or cyrillic == term:
        return []
    if cyrillic in vocabulary:
        return [cyrillic]
    if matcher is not None:
        return [
            candidate for candidate, _distance, _jaccard in matcher.match(cyrillic, limit=limit)
        ]
    return []


def is_han_or_cyrillic(text: str) -> bool:
    """True when the text contains a Cyrillic or Han character (script sanity checks)."""
    return any(is_cyrillic_char(char) or is_han_char(char) for char in text)


__all__ = [
    "MAX_FUZZY_CANDIDATES",
    "MIN_FUZZY_LEN",
    "NEGATIONS",
    "FuzzyMatcher",
    "NoiseProfile",
    "detect_noise_style",
    "is_han_or_cyrillic",
    "looks_transliterated",
    "normalize_for_matching",
    "segment_clauses",
    "to_cyrillic",
    "transliteration_candidates",
    "trigram_jaccard",
    "trigrams",
]
