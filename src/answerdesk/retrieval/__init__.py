"""Query analysis, hybrid fusion, the retrieval pipeline, and the ablation runner.

Owning task: **t5**. This package turns two indexes into one ranked answer path, and makes
the choices measurable rather than asserted.

Modules
-------
``query_analysis``  NFKC normalisation, cue-based language ID, identifier extraction,
                    additive typo/keyboard-layout repair against the corpus vocabulary
``fusion``         RRF (``k`` tunable) and convex combination (min–max / z-score) with
                    documented degenerate-distribution behaviour
``fit``             fitting ``rrf_k`` / ``alpha`` / learnt weights on the dev split, the
                    persisted artifact with provenance, and the leakage guard (INV-09)
``pipeline``        retrieve → fuse → dedupe → rerank → diversify → pack, with a full
                    ``RetrievalTrace`` and per-stage timings
``dedupe``          greedy near-duplicate removal by cosine (or token Jaccard)
``metrics``         recall_any / recall_all / nDCG / MRR exactly as §6.1 defines them,
                    plus paired bootstrap, Wilson and McNemar for honest error bars
``splits``          document-disjoint hashed dev/test assignment
``datasets``        retrieval cases derived from the corpus query set (dev harness)
``ablation``        ``python -m answerdesk.retrieval.ablation`` — the six-arm table

Requirements: FR-RET-04 … FR-RET-10, INV-09, RT-1 … RT-10. Maths and measurements:
``docs/retrieval.md``.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.retrieval.dedupe import dedupe_passages
from answerdesk.retrieval.fit import (
    FusionArtifact,
    LeakageError,
    SplitProvenance,
    assert_disjoint,
    coordinate_search,
    fit_fusion_artifact,
)
from answerdesk.retrieval.fusion import (
    ConvexFusion,
    FusionParams,
    Normalization,
    RRFFusion,
    SingleChannelFusion,
    minmax_normalize,
    zscore_normalize,
)
from answerdesk.retrieval.metrics import (
    evaluate_cases,
    mrr_at_k,
    ndcg_at_k,
    recall_all_at_k,
    recall_any_at_k,
)
from answerdesk.retrieval.pipeline import (
    PipelineConfig,
    RetrievalPipeline,
    build_pipeline,
    chunks_of,
)
from answerdesk.retrieval.query_analysis import (
    QueryAnalysis,
    QueryAnalysisConfig,
    QueryAnalyzer,
    detect_language,
)

__all__ = [
    "ConvexFusion",
    "FusionArtifact",
    "FusionParams",
    "LeakageError",
    "Normalization",
    "PipelineConfig",
    "QueryAnalysis",
    "QueryAnalysisConfig",
    "QueryAnalyzer",
    "RRFFusion",
    "RetrievalPipeline",
    "SingleChannelFusion",
    "SplitProvenance",
    "assert_disjoint",
    "build_pipeline",
    "chunks_of",
    "coordinate_search",
    "dedupe_passages",
    "detect_language",
    "evaluate_cases",
    "fit_fusion_artifact",
    "minmax_normalize",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_all_at_k",
    "recall_any_at_k",
    "zscore_normalize",
]
