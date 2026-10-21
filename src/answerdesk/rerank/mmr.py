"""Maximal Marginal Relevance diversification (FR-RET-07, the ``+mmr`` ablation arm).

MMR (Carbonell & Goldstein, 1998) picks the next passage by maximising

::

    MMR(i) = λ · rel(i) − (1 − λ) · max_{j ∈ S} sim(i, j)

where ``rel`` is the passage's relevance (its fused score, min–max normalised over the
candidate window) and ``S`` is the set already selected. ``λ = 1`` is pure relevance
(MMR is a no-op), ``λ = 0`` is pure diversity (a bad idea for support: the answer usually
lives in one article).

Why it is here at all: a 12-chunk context built from one article's three overlapping
chunks wastes budget and hides evidence from a second article. MMR is the cheap fix, and
it is **off by default** (``retrieval.mmr_lambda: null``) because it costs a rerank-quality
trade-off that must be measured, not assumed.

Determinism: ties break on ``chunk_id``, and the relevance term is normalised with the same
min–max function fusion uses, so the same candidate window always yields the same order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from answerdesk.contracts import RetrievedPassage
from answerdesk.index.text import tokenize

#: Default λ. 0.7 keeps relevance dominant while still discouraging near-duplicates.
DEFAULT_MMR_LAMBDA = 0.7

#: Similarity backend: (chunk_id_a, chunk_id_b, text_a, text_b) -> similarity in [0, 1].
SimilarityFn = Callable[[str, str, str, str], float]

#: Below this, a score vector is treated as flat (float32 rounding noise).
_FLAT_EPS = 1e-12


def minmax_relevance(scores: Sequence[float]) -> np.ndarray:
    """``(s − min) / (max − min)`` in [0, 1]; a flat vector maps to zeros.

    Same formula as :func:`answerdesk.retrieval.fusion.minmax_normalize`, and duplicated
    here **deliberately**: ``answerdesk.rerank`` must not import ``answerdesk.retrieval``
    at module level, because the retrieval package imports the pipeline and the pipeline
    imports this module — that cycle made ``import answerdesk.rerank`` fail. A test asserts
    the two implementations agree on a fixture.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low <= _FLAT_EPS:
        return np.zeros(values.size, dtype=np.float64)
    return (values - low) / (high - low)


def _jaccard_similarity(left_id: str, right_id: str, left_text: str, right_text: str) -> float:
    """Token Jaccard over chunk text — the no-embedder fallback (uncached, for tests)."""
    left = frozenset(tokenize(left_text, with_canonical=False))
    right = frozenset(tokenize(right_text, with_canonical=False))
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def vector_similarity(vector_for: Callable[[str], np.ndarray | None]) -> SimilarityFn:
    """Cosine similarity from the dense index vectors."""

    def _similarity(left_id: str, right_id: str, left_text: str, right_text: str) -> float:
        left = vector_for(left_id)
        right = vector_for(right_id)
        if left is None or right is None:  # pragma: no cover - coverage asserted upstream
            return _jaccard_similarity(left_id, right_id, left_text, right_text)
        return float(np.dot(left, right))

    return _similarity


def mmr_select(
    passages: Sequence[RetrievedPassage],
    *,
    k: int,
    lambda_: float = DEFAULT_MMR_LAMBDA,
    similarity: SimilarityFn | None = None,
) -> list[RetrievedPassage]:
    """Return ``k`` passages selected by MMR, in selection order (rank 1..k).

    Similarity is computed lazily but **cached per pair**: the Jaccard fallback tokenises
    each candidate once, and the vector backend is one dot product. Without that cache the
    selection is O(k · n²) text tokenisations — the difference between a 0.2 s and a
    150 s arm on a 3,000-chunk corpus (measured; see ``docs/retrieval.md`` §7).
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("mmr lambda must be in [0, 1]")
    if k <= 0 or not passages:
        return []
    relevance = minmax_relevance([passage.score for passage in passages])

    if similarity is None:
        token_sets = [
            frozenset(tokenize(passage.chunk.text, with_canonical=False)) for passage in passages
        ]
        pair_cache: dict[tuple[int, int], float] = {}

        def pair_similarity(left: int, right: int) -> float:
            key = (left, right) if left < right else (right, left)
            cached = pair_cache.get(key)
            if cached is None:
                intersection = len(token_sets[key[0]] & token_sets[key[1]])
                union = len(token_sets[key[0]] | token_sets[key[1]])
                cached = intersection / union if union else 0.0
                pair_cache[key] = cached
            return cached

    else:
        id_cache: dict[tuple[str, str], float] = {}

        def pair_similarity(left: int, right: int) -> float:
            left_id = passages[left].chunk.chunk_id
            right_id = passages[right].chunk.chunk_id
            key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            cached = id_cache.get(key)
            if cached is None:
                cached = float(
                    similarity(
                        left_id,
                        right_id,
                        passages[left].chunk.text,
                        passages[right].chunk.text,
                    )
                )
                id_cache[key] = cached
            return cached

    remaining = list(range(len(passages)))
    selected: list[int] = []
    while remaining and len(selected) < k:
        best_index: int | None = None
        best_key: tuple[float, str] | None = None
        for index in remaining:
            redundancy = 0.0
            for chosen in selected:
                redundancy = max(redundancy, pair_similarity(index, chosen))
            score = lambda_ * float(relevance[index]) - (1.0 - lambda_) * redundancy
            key = (-score, passages[index].chunk.chunk_id)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        if best_index is None:  # pragma: no cover - unreachable with a non-empty set
            break
        selected.append(best_index)
        remaining.remove(best_index)

    return [
        passages[index].model_copy(
            update={
                "rank": rank,
                "stage": "pack",
                "contributions": {
                    **passages[index].contributions,
                    "mmr_relevance": round(float(relevance[index]), 6),
                    "mmr_rank": float(rank),
                },
            }
        )
        for rank, index in enumerate(selected, start=1)
    ]


class MMRDiversifier:
    """Thin object wrapper so the pipeline can hold an optional stage (FR-RET-07)."""

    name = "mmr"

    def __init__(
        self,
        *,
        lambda_: float = DEFAULT_MMR_LAMBDA,
        similarity: SimilarityFn | None = None,
    ) -> None:
        self.lambda_ = lambda_
        self.similarity = similarity

    def diversify(self, passages: Sequence[RetrievedPassage], k: int) -> list[RetrievedPassage]:
        return mmr_select(passages, k=k, lambda_=self.lambda_, similarity=self.similarity)


__all__ = [
    "DEFAULT_MMR_LAMBDA",
    "MMRDiversifier",
    "SimilarityFn",
    "mmr_select",
    "vector_similarity",
]
