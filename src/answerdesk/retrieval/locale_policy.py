"""Locale-aware ordering of retrieved evidence, with an explicit cross-locale fallback.

The finding this closes is ``CNS-6`` (``docs/consistency.md``): retrieval had no locale policy,
so a Russian question regularly put Chinese or Uzbek chunks in the context — the pooled
"every cited document carries the query's locale" rate was **0.685**, and the language error
was the mechanical cause of the fact-agreement loss (the model reads a sibling locale's
numbers).

The policy is deliberately *soft*, because the failure modes of a hard filter are worse than
the leak it fixes:

1. **Language must be known and confident.** ``und`` or a detected-language confidence below
   ``min_confidence`` (default 0.5) disables the ordering — guessing a locale and then
   filtering on the guess would silently delete correct evidence from code-switched text.
2. **Same-locale first, everything else kept.** Chunks in the query's locale are moved ahead
   of the rest; nothing is dropped at this stage. Order within each group is preserved, and
   ties fall back to ``chunk_id``, so the ordering is deterministic.
3. **Explicit fallback.** If the preferred group cannot fill ``window`` — i.e. it holds fewer
   than ``ceil(window × min_same_ratio)`` chunks — cross-locale chunks are appended to fill the
   window and the fact is reported (``fallback_used``). Starving the context to protect a
   metric would be a worse answer, not a better one.
4. **Code-switch is a soft case.** A query mixing scripts gets the same order (same-locale
   first) but never loses cross-locale evidence, because the fallback always fills the window;
   the pipeline additionally lowers the ratio to zero for mixed-script queries so the ordering
   is a preference, not a gate.

Everything is a pure function of (passages, language, confidence, window, policy): no state, no
randomness, no I/O — which is what makes the purity measurement reproducible and lets the unit
tests assert exact orders.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from answerdesk.contracts import RetrievedPassage

#: Default share of the window that same-locale evidence must cover before the fallback fires.
DEFAULT_MIN_SAME_RATIO = 0.5

#: Below this detected-language confidence, ordering is skipped entirely.
DEFAULT_MIN_CONFIDENCE = 0.5

#: Languages that carry no locale signal at all.
UNKNOWN_LANGUAGES: frozenset[str] = frozenset({"", "und"})


@dataclass(frozen=True)
class LocalePolicy:
    """The knob set: on by default, with a documented off switch."""

    enabled: bool = True
    min_same_ratio: float = DEFAULT_MIN_SAME_RATIO
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_same_ratio <= 1.0:
            raise ValueError("min_same_ratio must be in [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass
class LocaleOrdering:
    """Result of one ordering pass: the passages plus what the policy actually did."""

    passages: list[RetrievedPassage] = field(default_factory=list)
    language: str = "und"
    same_locale: int = 0
    cross_locale: int = 0
    window: int = 0
    applied: bool = False
    fallback_used: bool = False
    reason: str = "disabled"

    @property
    def purity(self) -> float:
        """Share of the window that is same-locale evidence (0.0 for an empty window)."""
        taken = self.passages[: self.window]
        if not taken:
            return 0.0
        same = sum(1 for passage in taken if _is_same_locale(passage, self.language))
        return same / len(taken)


def _is_same_locale(passage: RetrievedPassage, language: str) -> bool:
    chunk_language = passage.chunk.language or ""
    return bool(language) and chunk_language == language


def order_by_locale(
    passages: Sequence[RetrievedPassage],
    *,
    language: str,
    confidence: float,
    window: int,
    policy: LocalePolicy | None = None,
) -> LocaleOrdering:
    """Reorder ``passages`` so same-locale evidence leads, without dropping anything.

    ``window`` is the number of slots the caller will actually use (the packed context size,
    or the rerank input size). The fallback test is evaluated against that window, not against
    the whole candidate list, because that is what the reader will see.
    """
    resolved = policy or LocalePolicy()
    ordered = list(passages)
    result = LocaleOrdering(passages=ordered, language=language, window=max(0, window))
    if not ordered:
        result.reason = "empty"
        return result
    if not resolved.enabled:
        result.reason = "disabled"
        return result
    if language in UNKNOWN_LANGUAGES:
        result.reason = "undetermined"
        return result
    if confidence < resolved.min_confidence:
        result.reason = "low_confidence"
        return result

    same = [passage for passage in ordered if _is_same_locale(passage, language)]
    cross = [passage for passage in ordered if not _is_same_locale(passage, language)]
    result.same_locale = len(same)
    result.cross_locale = len(cross)
    if not cross:
        result.reason = "already_pure"
        return result
    if not same:
        result.reason = "no_same_locale_evidence"
        result.fallback_used = True
        return result

    slots = max(0, min(result.window, len(ordered)))
    required = max(1, math.ceil(slots * resolved.min_same_ratio)) if slots else 1
    ordered = [*same, *cross]
    result.passages = ordered
    result.applied = True
    result.fallback_used = len(same) < required
    result.reason = "fallback" if result.fallback_used else "preferred"
    return result


def mark_locale_ordering(ordering: LocaleOrdering) -> list[RetrievedPassage]:
    """Attach the policy decision to each passage for the trace (no ranking effect)."""
    language = ordering.language
    marked: list[RetrievedPassage] = []
    for passage in ordering.passages:
        contributions = dict(passage.contributions)
        if _is_same_locale(passage, language):
            contributions["locale_pref"] = 1.0
        elif ordering.applied:
            contributions["cross_locale_fallback"] = 1.0 if ordering.fallback_used else 0.0
        marked.append(passage.model_copy(update={"contributions": contributions}))
    return marked


def chunk_purity(passages: Sequence[RetrievedPassage], language: str) -> float:
    """Share of ``passages`` written in ``language`` (0.0 when empty)."""
    if not passages:
        return 0.0
    return sum(1 for passage in passages if _is_same_locale(passage, language)) / len(passages)


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_MIN_SAME_RATIO",
    "UNKNOWN_LANGUAGES",
    "LocaleOrdering",
    "LocalePolicy",
    "chunk_purity",
    "mark_locale_ordering",
    "order_by_locale",
]
