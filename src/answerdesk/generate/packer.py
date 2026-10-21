"""Context packing: a hard token budget, dedupe, and lost-in-the-middle ordering.

Three requirements meet here (FR-GEN-05):

1. **Hard token budget.** The packer never exceeds the cap; it drops the lowest-ranked
   passage instead. Truncating a passage mid-sentence is not allowed — a truncated quote
   breaks CR-3, so recall loss is measured and truncation is not.
2. **Dedupe.** Two passages whose token-count cosine is at or above the configured
   threshold (default 0.95) are the same evidence twice; keeping both wastes budget and
   invites the model to double-count.
3. **Lost-in-the-middle ordering.** Evidence placed in the middle of a long context is
   used less reliably (research findings: lost-in-the-middle / context rot). So the
   strongest passage goes first, the second strongest goes last, and the remainder fill
   the middle in descending order.

Labels (``p1 … pn``) are assigned in *prompt* order, so the model reads ``p1`` first and
``pN`` last; the answer's citation numbers are derived later, by first appearance
(:mod:`answerdesk.generate.citations`).

    # AT-GEN-05
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import Field

from answerdesk.config import Settings
from answerdesk.contracts import Chunk, Contract, RetrievedPassage
from answerdesk.llm.cost import estimate_tokens

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

#: Passage-delimiter lookalikes inside document text are neutralised (INV-01).
_EVIDENCE_DELIMITER = re.compile(r"</?passage\b", re.IGNORECASE)


def neutralise_delimiters(text: str) -> str:
    """Escape ``<passage``/``</passage`` so evidence cannot close its own block."""
    return _EVIDENCE_DELIMITER.sub(lambda match: match.group(0).replace("<", "&lt;"), text)


#: Default near-duplicate threshold: at or above this cosine, keep only the better passage.
DEFAULT_DEDUPE_THRESHOLD = 0.95

#: Default evidence ordering/selection window (requirements §7.4: 8–12 chunks).
DEFAULT_CONTEXT_K_MIN = 8
DEFAULT_CONTEXT_K_MAX = 12


def _token_counts(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN.findall(text))


def token_cosine(left: str, right: str) -> float:
    """Cosine similarity of two token-count vectors — deterministic, no numpy, no model."""
    a, b = _token_counts(left), _token_counts(right)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(count * b[token] for token, count in a.items() if token in b)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    return dot / (norm_a * norm_b)


def lost_in_the_middle_order(passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
    """Reorder strongest-first / second-strongest-last (FR-GEN-05, requirements §7.4)."""
    ordered = list(passages)
    if len(ordered) <= 2:
        return ordered
    return [ordered[0], *ordered[2:], ordered[1]]


class PackedPassage(Contract):
    """One evidence block as the model sees it: a label, the passage, and its token cost."""

    label: str = Field(pattern=r"^p\d+$")
    passage: RetrievedPassage
    tokens: int = Field(ge=0)

    @property
    def chunk(self) -> Chunk:
        return self.passage.chunk

    @property
    def chunk_id(self) -> str:
        return self.passage.chunk.chunk_id

    @property
    def doc_title(self) -> str:
        return doc_title(self.chunk)

    @property
    def heading(self) -> str:
        return heading_text(self.chunk)


def doc_title(chunk: Chunk) -> str:
    """Human title for a chunk: explicit metadata first, breadcrumb second."""
    for key in ("doc_title", "title"):
        value = chunk.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if chunk.heading_path:
        return chunk.heading_path[0]
    return chunk.doc_id


def heading_text(chunk: Chunk) -> str:
    return chunk.breadcrumb or " > ".join(chunk.heading_path)


class PackedContext(Contract):
    """The packed evidence plus everything the trace and the citation builder need."""

    passages: list[PackedPassage] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    budget: int = Field(default=0, ge=0)
    reserved_tokens: int = Field(default=0, ge=0)
    dropped_duplicates: int = Field(default=0, ge=0)
    dropped_for_budget: int = Field(default=0, ge=0)
    evidence_score: float = 0.0
    dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD
    context_k_min: int = DEFAULT_CONTEXT_K_MIN
    context_k_max: int = DEFAULT_CONTEXT_K_MAX

    # -- lookups ------------------------------------------------------------------
    def by_label(self, label: str) -> PackedPassage | None:
        return next((packed for packed in self.passages if packed.label == label), None)

    def label_for(self, chunk_id: str) -> str | None:
        return next((packed.label for packed in self.passages if packed.chunk_id == chunk_id), None)

    def chunk_map(self) -> dict[str, Chunk]:
        """``chunk_id -> Chunk`` for every packed passage (verifier input)."""
        return {packed.chunk_id: packed.chunk for packed in self.passages}

    @property
    def is_empty(self) -> bool:
        return not self.passages

    @property
    def labels(self) -> list[str]:
        return [packed.label for packed in self.passages]

    # -- prompt rendering ---------------------------------------------------------
    def render_evidence(self) -> str:
        """The delimited evidence block the prompt embeds (INV-01 spotlighting)."""
        blocks: list[str] = []
        for packed in self.passages:
            title = packed.doc_title.replace('"', "'")
            heading = packed.heading.replace('"', "'")
            body = neutralise_delimiters(packed.chunk.text.strip())
            blocks.append(
                f'<passage id="{packed.label}" doc="{title}" heading="{heading}">\n'
                f"{body}\n"
                "</passage>"
            )
        return "\n\n".join(blocks)

    def as_trace(self) -> dict[str, object]:
        """Compact view for a trace record (t9) — no chunk text duplication."""
        return {
            "labels": self.labels,
            "chunk_ids": [packed.chunk_id for packed in self.passages],
            "scores": [round(packed.passage.score, 6) for packed in self.passages],
            "token_count": self.token_count,
            "budget": self.budget,
            "reserved_tokens": self.reserved_tokens,
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_for_budget": self.dropped_for_budget,
        }


class ContextPacker:
    """Deterministic packer: select, dedupe, budget, reorder, label."""

    def __init__(
        self,
        *,
        context_k_min: int = DEFAULT_CONTEXT_K_MIN,
        context_k_max: int = DEFAULT_CONTEXT_K_MAX,
        token_budget: int = 8000,
        dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD,
        reserved_tokens: int = 0,
    ) -> None:
        if context_k_min < 0 or context_k_max < 1:
            raise ValueError("context_k_min must be >= 0 and context_k_max >= 1")
        if context_k_max < context_k_min:
            raise ValueError("context_k_max must be >= context_k_min")
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if not 0.0 < dedupe_threshold <= 1.0:
            raise ValueError("dedupe_threshold must be in (0, 1]")
        self.context_k_min = context_k_min
        self.context_k_max = context_k_max
        self.token_budget = token_budget
        self.dedupe_threshold = dedupe_threshold
        self.reserved_tokens = max(0, reserved_tokens)

    @property
    def evidence_budget(self) -> int:
        """Tokens available for evidence after the non-evidence part of the request."""
        return max(0, self.token_budget - self.reserved_tokens)

    def _passage_tokens(self, passage: RetrievedPassage) -> int:
        chunk = passage.chunk
        # The wrapper (`<passage id="pN" …>`) is part of the request; count it honestly.
        overhead = estimate_tokens('<passage id="p1" doc="" heading=""></passage>')
        return estimate_tokens(chunk.text) + overhead

    def dedupe(self, passages: Sequence[RetrievedPassage]) -> tuple[list[RetrievedPassage], int]:
        """Order-preserving near-duplicate removal; returns (kept, dropped_count)."""
        kept: list[RetrievedPassage] = []
        seen_text: dict[str, str] = {}
        dropped = 0
        for passage in passages:
            text = " ".join(passage.chunk.text.split())
            if not text:
                dropped += 1
                continue
            if text in seen_text:
                dropped += 1
                continue
            duplicate = any(
                token_cosine(text, kept_text) >= self.dedupe_threshold
                for kept_text in seen_text.values()
            )
            if duplicate:
                dropped += 1
                continue
            kept.append(passage)
            seen_text[passage.chunk.chunk_id] = text
        return kept, dropped

    def pack(
        self,
        passages: Sequence[RetrievedPassage],
        *,
        budget: int | None = None,
        reserved_tokens: int | None = None,
    ) -> PackedContext:
        """Pack passages into the budget. Never raises on an empty input."""
        cap = self.token_budget if budget is None else budget
        reserved = self.reserved_tokens if reserved_tokens is None else max(0, reserved_tokens)
        evidence_cap = max(0, cap - reserved)
        kept, dropped_duplicates = self.dedupe(passages)
        selected = kept[: self.context_k_max]
        dropped_for_budget = 0

        packed: list[tuple[RetrievedPassage, int]] = []
        used = 0
        for passage in selected:
            tokens = self._passage_tokens(passage)
            if used + tokens > evidence_cap:
                # Rank order wins: a passage that does not fit is dropped whole. Lower
                # ranked passages may still fill the remaining budget, which is how the
                # 8–12 chunk window survives a small cap without truncating a quote.
                dropped_for_budget += 1
                continue
            packed.append((passage, tokens))
            used += tokens

        ordered = lost_in_the_middle_order([passage for passage, _ in packed])
        tokens_by_chunk = {passage.chunk.chunk_id: tokens for passage, tokens in packed}
        labelled = [
            PackedPassage(
                label=f"p{index}",
                passage=passage,
                tokens=tokens_by_chunk[passage.chunk.chunk_id],
            )
            for index, passage in enumerate(ordered, start=1)
        ]
        evidence_score = max((passage.score for passage, _ in packed), default=0.0)
        return PackedContext(
            passages=labelled,
            token_count=sum(tokens_by_chunk.values()),
            budget=cap,
            reserved_tokens=reserved,
            dropped_duplicates=dropped_duplicates,
            dropped_for_budget=dropped_for_budget,
            evidence_score=evidence_score,
            dedupe_threshold=self.dedupe_threshold,
            context_k_min=self.context_k_min,
            context_k_max=self.context_k_max,
        )

    @classmethod
    def from_settings(cls, settings: Settings, *, reserved_tokens: int = 0) -> ContextPacker:
        """Build the packer from the frozen configuration sections (Appendix C)."""
        return cls(
            context_k_min=settings.generate.context_k_min,
            context_k_max=settings.generate.context_k_max,
            token_budget=settings.generate.token_budget,
            dedupe_threshold=settings.retrieval.dedupe_cosine,
            reserved_tokens=reserved_tokens,
        )


__all__ = [
    "DEFAULT_CONTEXT_K_MAX",
    "DEFAULT_CONTEXT_K_MIN",
    "DEFAULT_DEDUPE_THRESHOLD",
    "ContextPacker",
    "PackedContext",
    "PackedPassage",
    "doc_title",
    "heading_text",
    "lost_in_the_middle_order",
    "neutralise_delimiters",
    "token_cosine",
]
