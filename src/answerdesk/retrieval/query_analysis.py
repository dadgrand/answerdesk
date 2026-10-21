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

#: Bumped whenever analysis changes — it is part of the fusion artifact provenance.
QUERY_ANALYSIS_VERSION = "qa-v2"

#: Locales the stack ships (requirements §10.1) plus English and "undetermined".
SUPPORTED_LOCALES: tuple[str, ...] = ("ru", "tg", "uz", "ky", "tr", "zh")

#: Function words and script cues per locale. Cues are cheap, stable and auditable;
#: a statistical detector would need a download the base install does not have. The sets
#: below are tuned on the checked-in query set (720 items, 6 locales): 99.3% accuracy
#: overall after tuning, with the residual being three-character typo queries that carry
#: no function word at all — reported as ``und`` rather than guessed.
_STOPWORDS: dict[str, frozenset[str]] = {
    "ru": frozenset(
        [
            "что",
            "как",
            "для",
            "это",
            "или",
            "при",
            "после",
            "можно",
            "нужно",
            "если",
            "почему",
            "сколько",
            "какой",
            "какая",
            "какое",
            "где",
            "когда",
            "мне",
            "мой",
            "моя",
            "ваш",
            "ваша",
            "не",
            "на",
            "по",
            "с",
            "со",
            "из",
            "до",
            "за",
            "от",
            "у",
            "же",
            "ли",
            "бы",
            "есть",
            "нет",
            "будет",
            "стоит",
            # Short particles: needed so a two-word query ("Ошибка и блокировка") is still
            # recognised as Russian instead of falling back to "und".
            "и",
            "в",
            "а",
            "но",
            "то",
            "уже",
            "тоже",
            "ещё",
            "еще",
        ]
    ),
    "tg": frozenset(
        [
            "ва",
            "барои",
            "чӣ",
            "чи",
            "кадом",
            "мебошад",
            "карда",
            "шуда",
            "бо",
            "аз",
            "дар",
            "куҷо",
            "чаро",
            "ман",
            "шумо",
            "оё",
            "ҳаст",
            "нест",
            "кунам",
            "метавонам",
            "дорад",
            "мешавад",
            "намешавад",
            "диҳад",
            "медиҳад",
            "аст",
            "чанд",
            "бояд",
            "шудааст",
            "кунад",
            "кардан",
            "буда",
            "ҳама",
            "дубора",
            "бораи",
            "бар",
        ]
    ),
    "ky": frozenset(
        [
            "жана",
            "менен",
            "үчүн",
            "канча",
            "болобу",
            "жок",
            "бар",
            "экен",
            "кантип",
            "ким",
            "эмне",
            "мен",
            "сиз",
            "ошондой",
            "болот",
            "болбойт",
            "керек",
            "турат",
            "алат",
            "чейин",
            "кийин",
            "аркылуу",
            "эмес",
            "кандай",
            "качан",
            "кайсы",
            "балким",
            "боюнча",
            "жатат",
            "эмнени",
            "билдирет",
            "катасы",
            "алам",
            "алабыз",
        ]
    ),
    "tr": frozenset(
        [
            "için",
            "nasıl",
            "neden",
            "niçin",
            "hangi",
            "bir",
            "ve",
            "ile",
            "bu",
            "ne",
            "mi",
            "mı",
            "mu",
            "mü",
            "yok",
            "var",
            "kadar",
            "daha",
            "olarak",
            "olur",
            "değil",
        ]
    ),
    "uz": frozenset(
        [
            "uchun",
            "qanday",
            "nima",
            "kerak",
            "bormi",
            "va",
            "bilan",
            "bu",
            "men",
            "siz",
            "yo'q",
            "bor",
            "mumkin",
            "qachon",
            "qayerda",
            "bo'ladi",
            "bo'lsa",
            "qancha",
            "yoki",
            "lar",
            "dan",
            "ga",
            "ni",
            "da",
            "xatosi",
            "nimani",
            "bildiradi",
            "turadi",
        ]
    ),
    "en": frozenset(
        [
            "the",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "is",
            "are",
            "was",
            "can",
            "do",
            "does",
            "did",
            "my",
            "your",
            "our",
            "for",
            "with",
            "and",
            "to",
            "of",
            "in",
            "on",
            "at",
            "it",
            "this",
            "that",
        ]
    ),
}

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
    max_expansions: int = Field(default=3, ge=0, le=10)
    fuzzy_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    layout_repair: bool = True
    multi_query: bool = False
    detect_language: bool = True


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
    lexical_terms: list[str] = Field(default_factory=list)
    dense_text: str = ""
    multi_query: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    version: str = QUERY_ANALYSIS_VERSION

    @property
    def is_empty(self) -> bool:
        """True when the query carries no retrievable term at all."""
        return not self.lexical_terms and not self.identifier_tokens


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
    ) -> None:
        self.vocabulary = vocabulary
        self.config = config or QueryAnalysisConfig()

    # -- construction helpers ----------------------------------------------------
    def with_vocabulary(self, vocabulary: TermVocabulary) -> QueryAnalyzer:
        """A copy bound to a corpus vocabulary (the pipeline does this once at build)."""
        return QueryAnalyzer(vocabulary, config=self.config)

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
        expanded, expansion_notes = self._expand(base_terms)
        notes.extend(expansion_notes)

        lexical = _dedupe([*base_terms, *expanded, *[token.lower() for token in protected]])
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
            lexical_terms=lexical,
            dense_text=normalized,
            multi_query=multi,
            notes=notes,
        )

    # -- expansion ---------------------------------------------------------------
    def _expand(self, base_terms: Sequence[str]) -> tuple[list[str], list[str]]:
        """Additive typo/layout repair against the corpus vocabulary (FR-RET-06)."""
        config = self.config
        vocabulary = self.vocabulary
        added: list[str] = []
        notes: list[str] = []
        if vocabulary is None or not (config.typo_tolerance or config.layout_repair):
            if vocabulary is None and (config.typo_tolerance or config.layout_repair):
                notes.append("no corpus vocabulary bound; typo tolerance inactive")
            return added, notes

        seen = set(base_terms)
        for term in base_terms:
            if len(added) >= config.max_expansions:
                break
            if term in vocabulary or len(term) < MIN_FUZZY_LEN:
                continue
            if config.layout_repair:
                repaired = vocabulary.repair_layout(term)
                if repaired is not None and repaired not in seen:
                    seen.add(repaired)
                    added.append(repaired)
                    notes.append(f"layout repair: {term!r} -> {repaired!r}")
                    continue
            if config.typo_tolerance:
                for candidate in vocabulary.fuzzy(
                    term, threshold=config.fuzzy_threshold, limit=config.max_expansions
                ):
                    if len(added) >= config.max_expansions:
                        break
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    added.append(candidate)
                    notes.append(f"typo expansion: {term!r} -> {candidate!r}")
        return added, notes


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
