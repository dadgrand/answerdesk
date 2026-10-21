"""Cross-encoder reranking, lazily loaded, plus the deterministic CI stand-in (FR-RET-07).

Two rerankers implement :class:`answerdesk.contracts.Reranker`:

``LexicalOverlapReranker`` (``lexical-overlap-v1``)
    Always available, no model, no network, fully deterministic. It scores a candidate by
    how much of the query it covers (term coverage in the chunk *and* its breadcrumb) plus
    a bonus when every identifier token of the query appears verbatim. This is what the
    offline gate and the ``+rerank`` ablation arm run: the *machinery* (budget, batching,
    stage bookkeeping, determinism) is exercised without pretending a lexical heuristic is
    a cross-encoder. Its numbers may never be reported as RT-10 evidence.

``LazyCrossEncoderReranker`` (``cross-encoder:...``)
    The real thing: ``sentence_transformers.CrossEncoder`` over ``top_in`` candidates down
    to ``top_out``. The import happens on first use, so the base install never sees torch
    (NFR-PORT-03) and ``answerdesk index``/``ask`` do not pay a multi-second import tax.
    Batching and ``max_length`` are configurable; scores are sorted ``(-score, chunk_id)``
    so two runs produce the same order.

Both rerankers preserve the contract: the returned passages carry ``stage="rerank"``, ranks
1..n, and the previous fused score in ``contributions["fused_score"]``, which is how the
trace shows a rerank delta (FR-RET-08).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from answerdesk.config import Settings
from answerdesk.contracts import Reranker, RetrievedPassage
from answerdesk.errors import ConfigError
from answerdesk.index.text import identifier_tokens, tokenize

#: Default local cross-encoder. Multilingual, 100+ languages, small enough for CPU bursts.
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


def _renumber(
    scored: Sequence[tuple[float, RetrievedPassage]],
    top_n: int,
    *,
    score_key: str,
) -> list[RetrievedPassage]:
    """Sort ``(-score, chunk_id)``, truncate, rebuild rank/contributions."""
    ordered = sorted(scored, key=lambda item: (-item[0], item[1].chunk.chunk_id))[: max(0, top_n)]
    out: list[RetrievedPassage] = []
    for rank, (score, passage) in enumerate(ordered, start=1):
        contributions = {
            **passage.contributions,
            "fused_score": float(passage.score),
            score_key: round(float(score), 6),
        }
        out.append(
            passage.model_copy(
                update={
                    "score": float(score),
                    "rank": rank,
                    "stage": "rerank",
                    "contributions": contributions,
                }
            )
        )
    return out


class LexicalOverlapReranker:
    """Deterministic, offline reranker used by CI and the ablation runner (FR-RET-07)."""

    name = "lexical-overlap-v1"

    def __init__(self, *, identifier_bonus: float = 0.5) -> None:
        self.identifier_bonus = identifier_bonus

    def score(
        self,
        query_terms: Sequence[str],
        identifiers: Sequence[str],
        passage: RetrievedPassage,
    ) -> float:
        """Query-term coverage of the candidate, with an identifier must-match bonus."""
        wanted = set(query_terms)
        if not wanted:
            return 0.0
        haystack = set(
            tokenize(f"{passage.chunk.breadcrumb} {passage.chunk.text}", with_canonical=True)
        )
        coverage = len(wanted & haystack) / len(wanted)
        identifier_hit = (
            sum(1 for token in identifiers if token.lower() in haystack) / len(identifiers)
            if identifiers
            else 0.0
        )
        return coverage + self.identifier_bonus * identifier_hit

    def rerank(
        self, query: str, passages: Sequence[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        """``Reranker`` protocol implementation."""
        terms = tokenize(query, with_canonical=False)
        identifiers = identifier_tokens(query)
        scored = [(self.score(terms, identifiers, passage), passage) for passage in passages]
        return _renumber(scored, top_n, score_key="rerank")


class LazyCrossEncoderReranker:
    """Cross-encoder reranker; imports sentence-transformers on first use (FR-RET-07)."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        *,
        batch_size: int = 16,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.name = f"cross-encoder:{model_name}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:  # pragma: no cover - requires the local extra and model weights
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "cross-encoder reranking needs the 'local' extra: run `uv sync --extra local`, "
                "or set ANSWERDESK_RETRIEVAL__RERANK__ENABLED=false to use the offline path."
            ) from exc
        device = None if self.device == "auto" else self.device
        self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=device)
        return self._model

    def rerank(  # pragma: no cover - needs weights
        self, query: str, passages: Sequence[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        """``Reranker`` protocol implementation with batching and stable ordering."""
        if not passages:
            return []
        model = self._load()
        pairs = [(query, passage.chunk.text) for passage in passages]
        scores: list[float] = []
        for start in range(0, len(pairs), max(1, self.batch_size)):
            batch = pairs[start : start + max(1, self.batch_size)]
            raw = model.predict(batch, show_progress_bar=False)
            scores.extend(float(value) for value in raw)
        scored = list(zip(scores, passages, strict=True))
        return _renumber(scored, top_n, score_key="rerank")


def build_reranker(
    settings: Settings,
    *,
    offline_stand_in: bool = True,
) -> Reranker | None:
    """Resolver: ``rerank.enabled`` + ``rerank.backend`` -> a concrete reranker.

    * disabled (the default, and the offline profile): the deterministic lexical stand-in,
      or ``None`` when the caller wants no rerank stage at all;
    * ``backend="cross_encoder"``: the local sentence-transformers model (``local`` extra);
    * ``backend="lexical"``: the stand-in, requested explicitly (an ablation arm);
    * ``backend="api"``: the hosted Cohere-style ``/rerank`` adapter.

    ``answerdesk.retrieval.ablation`` asks for the stand-in explicitly; a real run leaves
    ``retrieval.rerank.enabled=true`` (the ``local`` profile does this) and gets the model.
    The API import is local so that ``answerdesk.rerank`` stays importable without httpx
    wiring and no cycle forms between this module and :mod:`answerdesk.rerank.api`.
    """
    section = settings.retrieval.rerank
    if not section.enabled:
        return LexicalOverlapReranker() if offline_stand_in else None
    if section.backend == "api":
        from answerdesk.rerank.api import api_reranker_from_settings

        return api_reranker_from_settings(settings)
    if section.backend == "lexical":
        return LexicalOverlapReranker()
    return LazyCrossEncoderReranker(
        section.model,
        batch_size=section.batch_size,
    )


__all__ = [
    "DEFAULT_CROSS_ENCODER",
    "LazyCrossEncoderReranker",
    "LexicalOverlapReranker",
    "build_reranker",
]
