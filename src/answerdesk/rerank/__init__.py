"""Optional cross-encoder reranking and MMR diversification.

Owning task: **t5**. Both stages are optional and both are *lazy*: the base install must
never import torch (NFR-PORT-03), so the cross-encoder loads on first ``rerank`` call and
the offline gate uses the deterministic :class:`~answerdesk.rerank.cross_encoder.
LexicalOverlapReranker` stand-in instead of pretending a lexical heuristic is a
cross-encoder.

Modules
-------
``cross_encoder``  ``LexicalOverlapReranker`` (offline, deterministic) and
                   ``LazyCrossEncoderReranker`` (sentence-transformers, ``local`` extra)
``mmr``            maximal marginal relevance with a configurable ``lambda``

Requirements: FR-RET-07, NFR-LAT-02, RT-10 (SHOULD arm).

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.rerank.api import ApiReranker, api_reranker_from_settings
from answerdesk.rerank.cross_encoder import (
    DEFAULT_CROSS_ENCODER,
    LazyCrossEncoderReranker,
    LexicalOverlapReranker,
    build_reranker,
)
from answerdesk.rerank.mmr import (
    DEFAULT_MMR_LAMBDA,
    MMRDiversifier,
    mmr_select,
    vector_similarity,
)

__all__ = [
    "DEFAULT_CROSS_ENCODER",
    "DEFAULT_MMR_LAMBDA",
    "ApiReranker",
    "LazyCrossEncoderReranker",
    "LexicalOverlapReranker",
    "MMRDiversifier",
    "api_reranker_from_settings",
    "build_reranker",
    "mmr_select",
    "vector_similarity",
]
