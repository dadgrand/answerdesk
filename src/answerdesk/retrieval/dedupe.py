"""Near-duplicate removal before reranking and packing (FR-RET-08 ``dropped_duplicates``).

Chunk overlap means the same paragraph often arrives twice (a 48-token overlap window, an
article repeated across locales, a table header repeating). Feeding both copies to the
generator burns context budget on one fact and makes citation precision look worse than it
is.

Removal is **greedy in rank order**: walk the fused ranking, keep a passage unless a
passage already kept is at least ``threshold`` similar. Greedy is the right order because
the first copy is the better-ranked one; anything cleverer (clustering, then choosing a
representative) would be a ranking decision hidden inside a de-duplication step.

Two similarity backends, one behaviour:

* **cosine over the dense index vectors** when a dense index is available — the same
  geometry the retriever used, so "duplicate" means what the embedding says it means;
* **token Jaccard over chunk text** otherwise (a BM25-only configuration, or a corpus with
  no embedder). Different numbers, same contract, and the choice is recorded in the trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from answerdesk.contracts import RetrievedPassage
from answerdesk.index.text import tokenize

#: Default duplicate threshold. Weaviate ships 0.95 for its duplicate filter; below ~0.9
#: distinct FAQ answers about the same product start collapsing into one.
DEFAULT_DEDUPE_COSINE = 0.95


def _token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize(text, with_canonical=False))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def dedupe_passages(
    passages: Sequence[RetrievedPassage],
    *,
    threshold: float = DEFAULT_DEDUPE_COSINE,
    vector_for: Callable[[str], np.ndarray | None] | None = None,
) -> tuple[list[RetrievedPassage], int]:
    """Keep the first of each near-duplicate group; return ``(kept, dropped_count)``.

    Ranks are renumbered 1..n over the survivors, so the downstream reranker and packer see
    a dense ranking rather than one with holes.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("dedupe threshold must be in (0, 1]")
    kept: list[RetrievedPassage] = []
    kept_vectors: list[np.ndarray] = []
    kept_tokens: list[frozenset[str]] = []
    # Pick the backend once: mixing cosine and Jaccard comparisons inside one pass would
    # make the result depend on which passage happened to be missing a vector.
    use_vectors = vector_for is not None and all(
        vector_for(passage.chunk.chunk_id) is not None for passage in passages
    )
    dropped = 0
    for passage in passages:
        if use_vectors and vector_for is not None:
            vector = vector_for(passage.chunk.chunk_id)
            assert vector is not None  # guaranteed by the backend probe above
            if any(float(np.dot(vector, other)) >= threshold for other in kept_vectors):
                dropped += 1
                continue
            kept_vectors.append(vector)
        else:
            tokens = _token_set(passage.chunk.text)
            if any(_jaccard(tokens, other) >= threshold for other in kept_tokens):
                dropped += 1
                continue
            kept_tokens.append(tokens)
        kept.append(passage)

    rebuilt = [
        passage.model_copy(update={"rank": rank}) for rank, passage in enumerate(kept, start=1)
    ]
    return rebuilt, dropped


__all__ = ["DEFAULT_DEDUPE_COSINE", "dedupe_passages"]
