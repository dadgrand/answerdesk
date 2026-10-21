"""Service-side signals: score normalisation, conversation replay and guard pre-flight.

Three small pieces sit between the frozen modules and the HTTP layer, and each exists for a
reason the tests pin down:

:func:`normalise_passages`
    The policy layer's :class:`~answerdesk.policy.scorers.OverlapScorer` expects a
    relevance prior in ``[0, 1]``; fusion (RRF) produces tiny reciprocal-rank numbers
    (~0.016). Dividing by the best score keeps the *ranking* untouched while giving the
    scorer a comparable prior. This is a documented seam, not a hidden rescale: the raw
    score is still in the trace, and the normalised value is recorded under
    ``scorer.inputs``.

:func:`replay_history`
    FR-SVC-02 requires the conversation to be replayed into the pipeline. A follow-up
    ("а для второй карты?") is not a self-contained query, so the service resolves it
    against the previous turn before retrieval and generation. The resolution rule is
    deterministic and conservative: a short message, or one that opens with an
    anaphoric/conjunction cue in any supported locale, is prefixed with the previous
    customer message.

:func:`preflight_decision`
    Guards must run **before** the provider call (INV-05: PII never leaves the process,
    and a blocked turn should not cost a model call). The engine would run them after
    generation, so the turn runner asks the same guard functions first and, if a verdict
    fires, builds the decision from the engine's own thresholds. A test asserts the
    pre-flight decision equals the decision the engine would have produced.

    # AT-SVC-02  # AT-SVC-08  # AT-ESC-04  # AT-SEC-02
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from answerdesk.contracts import (
    Action,
    Decision,
    IntentBucket,
    RetrievedPassage,
)
from answerdesk.policy.decision import DecisionEngine
from answerdesk.policy.guards import evaluate_guards, scan

#: Follow-up cues across the supported locales (ru, tg, uz, ky, tr, zh, en).
_ANAPHORA = re.compile(
    r"^\s*(?:"
    r"и|а|ещ[её]|также|а\s+если|и\s+что|а\s+как|"
    r"what\s+about|what\s+if|and|also|how\s+about|"
    r"那|还有|并且|那么|"
    r"peki|va|yana|дагы"
    r")\b",
    re.IGNORECASE,
)

#: A message this short needs context to be meaningful once a session exists.
SHORT_MESSAGE_WORDS = 4

#: Bound on the resolved query so a long conversation cannot inflate the prompt.
MAX_RESOLVED_CHARS = 2000


class TurnLike(Protocol):
    """What :func:`replay_history` needs from a stored turn (``sessions.TurnRow``)."""

    seq: int
    message: str
    answer_text: str
    action: str


@dataclass(frozen=True)
class HistoryReplay:
    """The resolved query plus what the trace needs to explain it."""

    query: str
    follow_up: bool
    turns_used: int
    last_user_message: str | None = None
    last_answer: str | None = None
    cues: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "follow_up": self.follow_up,
            "turns_used": self.turns_used,
            "cues": list(self.cues),
            "resolved_query_chars": len(self.query),
        }

    def transcript(self) -> list[dict[str, str]]:
        """The compact transcript the console shows beside a turn."""
        rows: list[dict[str, str]] = []
        if self.last_user_message:
            rows.append({"role": "customer", "content": self.last_user_message})
        if self.last_answer:
            rows.append({"role": "assistant", "content": self.last_answer})
        return rows


def normalise_passages(passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
    """Scale scores into ``[0, 1]`` by the best score, preserving order and rank."""
    if not passages:
        return []
    best = max(passage.score for passage in passages)
    if best <= 0:
        return [passage.model_copy(update={"score": 0.0}) for passage in passages]
    return [
        passage.model_copy(update={"score": round(passage.score / best, 10)})
        for passage in passages
    ]


def replay_history(
    turns: Sequence[TurnLike],
    message: str,
    *,
    max_turns: int = 3,
) -> HistoryReplay:
    """Resolve ``message`` against the last turns of the conversation (FR-SVC-02)."""
    recent = [turn for turn in turns if turn.message][-max_turns:]
    if not recent:
        return HistoryReplay(query=message, follow_up=False, turns_used=0)
    last = recent[-1]
    cues: list[str] = []
    if _ANAPHORA.search(message):
        cues.append("anaphora")
    if len(message.split()) <= SHORT_MESSAGE_WORDS:
        cues.append("short")
    if not cues:
        return HistoryReplay(
            query=message,
            follow_up=False,
            turns_used=len(recent),
            last_user_message=last.message,
            last_answer=last.answer_text or None,
        )
    resolved = f"{last.message.strip()} {message.strip()}".strip()[:MAX_RESOLVED_CHARS]
    return HistoryReplay(
        query=resolved,
        follow_up=True,
        turns_used=len(recent),
        last_user_message=last.message,
        last_answer=last.answer_text or None,
        cues=tuple(cues),
    )


def preflight_decision(
    engine: DecisionEngine,
    *,
    query: str,
    bucket: IntentBucket,
    p_hat: float = 0.0,
    intent_label: str | None = None,
) -> Decision | None:
    """Run the guards before retrieval/generation; return the forced decision, if any.

    Returns ``None`` when no guard fires — the turn continues to the engine, which runs the
    identical guard functions again. Duplicating the *evaluation* is intentional; the
    decision construction uses the engine's own threshold entry so there is one source of
    ``alpha`` and ``threshold_ref``.
    """
    signals = scan(query, (), bucket=bucket, intent_label=intent_label, config=engine.guards)
    verdict = evaluate_guards(signals, engine.guards)
    if verdict is None:
        return None
    entry = engine.thresholds.for_bucket(bucket)
    return Decision(
        action=verdict.action,
        reason_code=verdict.reason_code,
        score=min(1.0, max(0.0, float(p_hat))),
        alpha=entry.alpha_final,
        threshold_ref=engine.threshold_ref(),
    )


def is_answer(decision: Decision) -> bool:
    return decision.action is Action.ANSWER


__all__ = [
    "MAX_RESOLVED_CHARS",
    "SHORT_MESSAGE_WORDS",
    "HistoryReplay",
    "TurnLike",
    "is_answer",
    "normalise_passages",
    "preflight_decision",
    "replay_history",
]
