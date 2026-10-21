"""Query analysis: normalisation, language ID, identifier protection, typo tolerance.

This is the module that decides what the retrievers are actually asked. It is deliberately
conservative, because every aggressive transformation here is a silent recall loss:

* **NFKC + whitespace folding only.** Case is preserved on the query's ``normalized`` form
  so identifier fingerprints stay byte-identical (FR-RET-09), and no stemming, no
  transliteration, no translation happens anywhere (D-07, NFR-ML-02).
* **Identifiers are extracted, never rewritten.** ``ORD-2026-10009``, ``TS-999``,
  ``2026-03-05`` and ``12.50 EUR`` are canonicalised by the ingestion canonicaliser and
  handed to the sparse retriever as a **must-match** probe, so exact-token evidence cannot
  be normalised out of existence (FR-RET-06, RT-8).
* **Expansion is additive.** A typo or a wrong keyboard layout produces *extra* candidate
  terms that the corpus actually contains; the original terms always remain in the query.
  The dense arm always sees the untouched normalised query, so an expansion can never
  make the query mean something else.

Language identification is a cue-based heuristic over the six shipped locales (ru, tg, uz,
ky, tr, zh) plus ``en`` and ``und``. No model, no download, deterministic — and its
accuracy on the checked-in query set is measured (``docs/retrieval.md`` §4): the point is
not linguistic purity but choosing a locale for reporting and for locale-specific
normalisation. It is allowed to be wrong, so it reports a confidence, and downstream code
that needs certainty must not treat ``language`` as truth.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from answerdesk.index.text import (
    MIN_FUZZY_LEN,
    TermVocabulary,
    identifier_tokens,
    normalize_text,
    raw_terms,
    script_counts,
    tokenize,
)
from answerdesk.retrieval.noisy import FuzzyMatcher, segment_clauses, transliteration_candidates
from answerdesk.retrieval.rewrite import QueryRewriter, validate_rewrite
from answerdesk.retrieval.words import FUNCTION_WORDS, content_terms

#: Bumped whenever analysis changes — it is part of the fusion artifact provenance.
QUERY_ANALYSIS_VERSION = "qa-v2"

#: Locales the stack ships (requirements §10.1) plus English and "undetermined".
SUPPORTED_LOCALES: tuple[str, ...] = ("ru", "tg", "uz", "ky", "tr", "zh")

#: Function words and script cues per locale. Cues are cheap, stable and auditable;
#: a statistical detector would need a download the base install does not have. The sets
#: below are tuned on the checked-in query set (720 items, 6 locales): 99.3% accuracy
#: overall after tuning, with the residual being three-character typo queries that carry
#: no function word at all — reported as ``und`` rather than guessed.
#: One source of truth with the noise filter and the clause splitter (see words.py).
_STOPWORDS: dict[str, frozenset[str]] = FUNCTION_WORDS

_LETTER_CUES: dict[str, frozenset[str]] = {
    # ``ы`` is deliberately absent: Kyrgyz writes it too, and including it made Kyrgyz
    # queries look Russian (measured: 92% -> 100% Kyrgyz accuracy after removal).
    "ru": frozenset("эёъ"),
    "tg": frozenset("ҳқҷӯӣ"),
    "ky": frozenset("ңөү"),
    "tr": frozenset("ığşö"),
    "uz": frozenset("ʻʼ"),
}

#: Two-letter locative suffixes that are almost exclusive to Uzbek Latin text.
_UZ_LETTER_PAIRS: tuple[str, ...] = ("oʻ", "gʻ", "o'", "g'")

#: Weight of one function-word hit relative to one distinctive-letter hit.
_STOPWORD_WEIGHT = 1.0
_LETTER_WEIGHT = 1.5
_MIN_CONFIDENCE = 0.0
_HAN_SHARE_THRESHOLD = 0.20


class QueryAnalysisConfig(BaseModel):
    """Knobs of :class:`QueryAnalyzer`; defaults are the shipped behaviour."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    typo_tolerance: bool = True
    max_expansions: int = Field(default=4, ge=0, le=10)
    fuzzy_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    layout_repair: bool = True
    multi_query: bool = False
    detect_language: bool = True
    # --- noisy-input handling (t23); every stage is additive and independently switchable --
    clause_segmentation: bool = True
    max_clauses: int = Field(default=4, ge=1, le=8)
    focus_probe: bool = True
    #: ``None`` follows ``typo_tolerance``: switching typo handling off switches the whole
    #: typo stack off, while an explicit value overrides just this mechanism.
    fuzzy_edit_distance: bool | None = None
    transliteration: bool = True
    rewrite: bool = False


class QueryAnalysis(BaseModel):
    """Everything the retrievers need to know about one incoming query (FR-RET-06/09)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: str
    normalized: str
    language: str = "und"
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    identifier_tokens: list[str] = Field(default_factory=list)
    protected_spans: list[str] = Field(default_factory=list)
    base_terms: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    #: High-precision tokens: the user's own terms, canonical identifiers and layout repairs.
    primary_terms: list[str] = Field(default_factory=list)
    #: Recall-oriented additions (transliteration, trigram and edit-distance matches). Kept
    #: out of the primary probe on purpose: a wrong expansion in the same bag of words dilutes
    #: BM25 for *every* document, while as its own probe it only votes for its own matches.
    expansion_terms: list[str] = Field(default_factory=list)
    #: primary + expansions: the full additive list (never drops a user token).
    lexical_terms: list[str] = Field(default_factory=list)
    dense_text: str = ""
    multi_query: list[str] = Field(default_factory=list)
    #: Clauses of a run-on message (``[normalized]`` when it is not run-on).
    clauses: list[str] = Field(default_factory=list)
    #: Content-word-only view of the query, used as an extra probe for telegraphic input.
    focus_terms: list[str] = Field(default_factory=list)
    #: Terms that only exist because a Latin term was read as Cyrillic (``zaym`` -> ``займ``).
    transliterated_terms: list[str] = Field(default_factory=list)
    #: Optional LLM rewrite, validated: no invented identifiers, never a replacement.
    rewrite: str | None = None
    notes: list[str] = Field(default_factory=list)
    version: str = QUERY_ANALYSIS_VERSION

    @property
    def is_empty(self) -> bool:
        """True when the query carries no retrievable term at all."""
        return not self.lexical_terms and not self.identifier_tokens

    @property
    def is_run_on(self) -> bool:
        """True when the message had to be split into two or more clauses."""
        return len(self.clauses) > 1

    @property
    def is_telegraphic(self) -> bool:
        """True for short, function-word-heavy messages ("деньги как получит").

        Two conditions, both surface-level and deterministic: at most five word tokens, and
        at most 60% of them carrying content. A clean short question ("Refund policy?") has a
        high content share and is therefore *not* routed to the focus probe.
        """
        words = list(self.base_terms)
        if not words or len(words) > 5:
            return False
        content = content_terms(words, locale=self.language)
        if not content or len(content) == len(words):
            return False  # no function word at all: a terse but well-formed query
        return len(content) < len(words)


def detect_language(text: str) -> tuple[str, float]:
    """Return ``(locale, confidence)`` for the six shipped locales, ``en`` or ``und``.

    Scoring: one point per function-word hit, 1.5 points per distinctive letter. The
    confidence is the winner's share of the total evidence, so a text with no cues at all
    returns ``("und", 0.0)`` rather than an arbitrary guess.
    """
    counts = script_counts(text)
    letters = sum(counts.values())
    if letters == 0:
        return "und", 0.0
    if counts["han"] / letters >= _HAN_SHARE_THRESHOLD:
        share = counts["han"] / letters
        return "zh", round(min(1.0, 0.5 + share), 3)

    lowered = text.lower()
    words = set(raw_terms(text))
    scores: dict[str, float] = {}
    for locale, stopwords in _STOPWORDS.items():
        hits = len(words & stopwords)
        score = _STOPWORD_WEIGHT * hits
        cues = _LETTER_CUES.get(locale)
        if cues:
            score += _LETTER_WEIGHT * sum(1 for char in lowered if char in cues)
        if locale == "uz":
            score += _LETTER_WEIGHT * sum(lowered.count(pair) for pair in _UZ_LETTER_PAIRS)
        if score > 0.0:
            scores[locale] = score
    if not scores:
        return "und", 0.0
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_locale, best_score = ranked[0]
    confidence = round(best_score / total, 3) if total else 0.0
    if confidence < _MIN_CONFIDENCE:
        return "und", confidence
    return best_locale, confidence


class QueryAnalyzer:
    """Stateless-per-call analyzer; holds only the corpus vocabulary and config."""

    name = "query-analysis"

    def __init__(
        self,
        vocabulary: TermVocabulary | None = None,
        *,
        config: QueryAnalysisConfig | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.config = config or QueryAnalysisConfig()
        self.rewriter = rewriter if self.config.rewrite else None
        self._matcher: FuzzyMatcher | None = None

    # -- construction helpers ----------------------------------------------------
    def with_vocabulary(self, vocabulary: TermVocabulary) -> QueryAnalyzer:
        """A copy bound to a corpus vocabulary (the pipeline does this once at build)."""
        return QueryAnalyzer(vocabulary, config=self.config, rewriter=self.rewriter)

    @property
    def matcher(self) -> FuzzyMatcher | None:
        """Cached bounded edit-distance matcher over the corpus vocabulary (built once)."""
        if self._matcher is None and self.vocabulary is not None:
            self._matcher = FuzzyMatcher(self.vocabulary.terms)
        return self._matcher

    # -- the analysis ------------------------------------------------------------
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyse one query. Empty input yields an empty analysis, never an exception."""
        normalized = normalize_text(query)
        if not normalized:
            return QueryAnalysis(raw=query, normalized="", dense_text="", notes=["empty query"])

        notes: list[str] = []
        language, confidence = (
            detect_language(normalized) if self.config.detect_language else ("und", 0.0)
        )
        if language == "und":
            notes.append("language undetermined; no locale-specific normalisation applied")

        protected = identifier_tokens(normalized)
        base_terms = tokenize(normalized, with_canonical=False)
        primary_additions, expansion_additions, expansion_notes = self._expand(
            base_terms, language=language
        )
        notes.extend(expansion_notes)
        primary = _dedupe(
            [*base_terms, *primary_additions, *[token.lower() for token in protected]]
        )
        expansions = [term for term in expansion_additions if term not in primary]

        expanded = _dedupe([*primary_additions, *expansions])
        clauses = self._clauses(normalized, language=language)
        if len(clauses) > 1:
            notes.append(f"clause segmentation: {len(clauses)} clauses")
        focus = self._focus(primary, language=language)
        transliterated = (
            [term for term in expansions if term and not term.isascii()]
            if self.config.transliteration
            else []
        )

        rewrite = self._rewrite(normalized, protected, notes)
        lexical = _dedupe([*primary, *expansions])
        multi: list[str] = []
        if self.config.multi_query:
            candidates = [normalized, " ".join(base_terms), " ".join(protected)]
            multi = [value for value in _dedupe(candidates) if value and value != normalized]

        return QueryAnalysis(
            raw=query,
            normalized=normalized,
            language=language,
            language_confidence=confidence,
            identifier_tokens=protected,
            protected_spans=protected,
            base_terms=base_terms,
            expanded_terms=expanded,
            primary_terms=primary,
            expansion_terms=expansions,
            lexical_terms=lexical,
            dense_text=normalized,
            multi_query=multi,
            clauses=clauses,
            focus_terms=focus,
            transliterated_terms=transliterated,
            rewrite=rewrite,
            notes=notes,
        )

    # -- noisy-input stages ------------------------------------------------------
    def _clauses(self, normalized: str, *, language: str) -> list[str]:
        """Clauses of a run-on message; ``[normalized]`` signals "not run-on"."""
        if not self.config.clause_segmentation or not normalized:
            return [normalized] if normalized else []
        return segment_clauses(normalized, locale=language, max_clauses=self.config.max_clauses)

    def _focus(self, base_terms: Sequence[str], *, language: str) -> list[str]:
        """Content words only: the probe that rescues telegraphic, function-word-heavy input."""
        if not self.config.focus_probe:
            return []
        focus = content_terms(base_terms, locale=language)
        return focus if len(focus) >= 1 and focus != list(base_terms) else []

    def _rewrite(self, normalized: str, protected: Sequence[str], notes: list[str]) -> str | None:
        """Optional LLM reformulation, validated: identifiers may not appear or vanish."""
        if self.rewriter is None or not normalized:
            return None
        try:
            candidate = self.rewriter.rewrite(normalized)
        except Exception as exc:  # pragma: no cover - a provider failure must never break search
            notes.append(f"rewrite unavailable: {type(exc).__name__}")
            return None
        confirmed = validate_rewrite(normalized, candidate)
        if confirmed is None:
            notes.append("rewrite rejected (identifier guard, length or emptiness)")
            return None
        missing = [token for token in protected if token.lower() not in confirmed.lower()]
        if missing:
            notes.append(f"rewrite rejected: dropped identifiers {missing}")
            return None
        notes.append("rewrite probe added")
        return confirmed

    # -- expansion ---------------------------------------------------------------
    def _expand(
        self, base_terms: Sequence[str], *, language: str = "und"
    ) -> tuple[list[str], list[str], list[str]]:
        """Additive expansion, split by precision.

        * **layout repairs** go into the *primary* probe: the other-layout spelling only
          exists when the corpus contains it, so the precision is high;
        * **transliteration and edit-distance matches** go into a *separate* recall probe: a
          wrong candidate in the same bag of words dilutes every BM25 score, whereas as its
          own probe it can only vote for its own matches (see ``docs/retrieval.md`` §16).

        The original terms are never removed (``FR-RET-06``).
        """
        config = self.config
        vocabulary = self.vocabulary
        layout: list[str] = []
        recall: list[str] = []
        notes: list[str] = []
        edit_enabled = (
            config.typo_tolerance
            if config.fuzzy_edit_distance is None
            else config.fuzzy_edit_distance
        )
        if vocabulary is None or not (
            config.typo_tolerance or config.layout_repair or edit_enabled
        ):
            if vocabulary is None and (
                config.typo_tolerance or config.layout_repair or edit_enabled
            ):
                notes.append("no corpus vocabulary bound; typo tolerance inactive")
            return layout, recall, notes

        seen = set(base_terms)
        budget = config.max_expansions
        for term in base_terms:
            if len(layout) + len(recall) >= budget:
                break
            if term in vocabulary or len(term) < MIN_FUZZY_LEN:
                continue
            if config.layout_repair:
                repaired = vocabulary.repair_layout(term)
                if repaired is not None and repaired not in seen:
                    seen.add(repaired)
                    layout.append(repaired)
                    notes.append(f"layout repair: {term!r} -> {repaired!r}")
                    continue
            # Precision order: a high-threshold trigram match first (the t5 path), then
            # transliteration, then bounded edit distance as the last resort. Trying the
            # loosest mechanism first measured worse on typo queries (docs/retrieval.md §15.3).
            high_precision: list[str] = []
            if config.typo_tolerance:
                high_precision = vocabulary.fuzzy(
                    term, threshold=config.fuzzy_threshold, limit=budget
                )
            if high_precision:
                for candidate in high_precision:
                    if len(layout) + len(recall) >= budget:
                        break
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    recall.append(candidate)
                    notes.append(f"typo expansion: {term!r} -> {candidate!r}")
                continue
            matcher = self.matcher
            sources: list[tuple[str, list[str]]] = []
            if config.transliteration and matcher is not None and term.isascii():
                sources.append(
                    (
                        "transliteration",
                        transliteration_candidates(
                            term, vocabulary=vocabulary.terms, matcher=matcher
                        ),
                    )
                )
            if edit_enabled and matcher is not None:
                sources.append(
                    (
                        "edit-distance",
                        [candidate for candidate, _d, _j in matcher.match(term, limit=2)],
                    )
                )
            for label, candidates in sources:
                for candidate in candidates:
                    if len(layout) + len(recall) >= budget:
                        break
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    recall.append(candidate)
                    notes.append(f"{label}: {term!r} -> {candidate!r}")
        return layout, recall, notes


def _dedupe(values: Sequence[str]) -> list[str]:
    """Order-preserving de-duplication (the first occurrence wins)."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "QUERY_ANALYSIS_VERSION",
    "SUPPORTED_LOCALES",
    "QueryAnalysis",
    "QueryAnalysisConfig",
    "QueryAnalyzer",
    "detect_language",
]
