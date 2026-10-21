"""Retrieval arms over an external passage pool: lexical, dense and hybrid.

The external arm compares the product's two retrieval philosophies on a public qrels set:
BM25 (the lexical channel) against a hybrid of BM25 and hosted bge-m3 (the same embedder the
benchmark's hosted configuration uses). Fusion is a convex combination of min-max normalised
scores, i.e. the same family as the product's ``cc`` fuser; the weight is reported with every
number so a reader can see it was not tuned on the test split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from answerdesk.eval.external.bench import Passage, RetrievalQuery
from answerdesk.eval.metrics import mrr_at_k, ndcg_at_k, recall_any_at_k
from answerdesk.eval.stats import paired_bootstrap


class EmbedderLike(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class RankedQuery:
    item_id: str
    language: str
    ranked: list[str]
    scores: list[float]


def _tokenize(texts: Sequence[str]) -> Any:
    import bm25s

    # no stemming, no stopword list: the languages are agglutinative/morphologically rich and
    # an English stopword list would silently delete content words
    return bm25s.tokenize(list(texts), stopwords=None, show_progress=False)


class LexicalRanker:
    """bm25s over the passage pool (the same library the product's lexical channel uses)."""

    name = "bm25"

    def __init__(self, passages: Sequence[Passage]) -> None:
        import bm25s

        self.passages = list(passages)
        self._ids = [passage.passage_id for passage in self.passages]
        self._index = bm25s.BM25()
        self._index.index(
            _tokenize([passage.text for passage in self.passages]), show_progress=False
        )

    def rank(self, query: str, k: int = 20) -> RankedQuery:
        results, scores = self._index.retrieve(
            _tokenize([query]), k=min(k, len(self._ids)), show_progress=False
        )
        ids = [self._ids[int(index)] for index in results[0]]
        return RankedQuery(
            item_id="", language="", ranked=ids, scores=[float(s) for s in scores[0]]
        )

    def score_matrix(self, queries: Sequence[str]) -> np.ndarray:
        """Full (query, passage) score matrix; every pool document gets a score."""
        matrix = np.zeros((len(queries), len(self.passages)), dtype=np.float64)
        position = {passage_id: index for index, passage_id in enumerate(self._ids)}
        width = len(self._ids)
        for row, query in enumerate(queries):
            ranked = self.rank(query, k=width)
            for passage_id, score in zip(ranked.ranked, ranked.scores, strict=True):
                matrix[row, position[passage_id]] = score
        return matrix


class DenseRanker:
    """Cosine similarity over hosted (or any protocol-compatible) embeddings."""

    name = "dense"

    def __init__(
        self, passages: Sequence[Passage], embedder: EmbedderLike, *, batch: int = 32
    ) -> None:
        self.passages = list(passages)
        self.ids = [passage.passage_id for passage in self.passages]
        vectors: list[list[float]] = []
        for start in range(0, len(self.passages), batch):
            vectors.extend(
                embedder.embed_documents([p.text for p in self.passages[start : start + batch]])
            )
        self.matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.where(norms == 0, 1.0, norms)
        self.embedder = embedder

    def scores(self, query: str) -> np.ndarray:
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return self.matrix @ vector

    def rank(self, query: str, k: int = 20) -> RankedQuery:
        scores = self.scores(query)
        order = np.argsort(-scores)[:k]
        return RankedQuery(
            item_id="",
            language="",
            ranked=[self.ids[int(i)] for i in order],
            scores=[float(scores[int(i)]) for i in order],
        )


def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class HybridRanker:
    """0.5 * normalised BM25 + 0.5 * normalised dense, over the union of candidate pools."""

    name = "hybrid"

    def __init__(
        self,
        passages: Sequence[Passage],
        lexical: LexicalRanker,
        dense: DenseRanker,
        *,
        alpha: float = 0.5,
    ) -> None:
        self.passages = list(passages)
        self.ids = [passage.passage_id for passage in self.passages]
        self.lexical = lexical
        self.dense = dense
        self.alpha = alpha

    def rank(self, query: str, k: int = 20) -> RankedQuery:
        lexical_scores = np.asarray(self._all_lexical(query), dtype=np.float64)
        dense_scores = np.asarray(self.dense.scores(query), dtype=np.float64)
        fused = self.alpha * _minmax(lexical_scores) + (1.0 - self.alpha) * _minmax(dense_scores)
        order = np.argsort(-fused)[:k]
        return RankedQuery(
            item_id="",
            language="",
            ranked=[self.ids[int(i)] for i in order],
            scores=[float(fused[int(i)]) for i in order],
        )

    def _all_lexical(self, query: str) -> list[float]:
        # bm25s ranks internally; scoring every document is cheap for these pool sizes and
        # keeps the fusion honest (only the union of both channels is considered)
        ranked = self.lexical.rank(query, k=len(self.ids))
        lookup = dict(zip(ranked.ranked, ranked.scores, strict=True))
        return [lookup.get(passage_id, 0.0) for passage_id in self.ids]


@dataclass
class RetrievalReport:
    arm: str
    n: int
    ndcg10: float
    ndcg10_ci: tuple[float, float]
    recall10: float
    recall10_ci: tuple[float, float]
    mrr10: float
    per_query: list[dict[str, Any]]


def evaluate(
    ranker: Any,
    queries: Sequence[RetrievalQuery],
    *,
    k: int = 10,
    samples: int = 1000,
    seed: int = 0,
) -> RetrievalReport:
    """nDCG@k, recall@k and MRR@k with paired-bootstrap-ready per-query rows."""
    from answerdesk.eval.stats import bootstrap_interval

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    rows: list[dict[str, Any]] = []
    for query in queries:
        gold = [passage_id for passage_id, grade in query.qrels.items() if grade > 0]
        ranked = ranker.rank(query.query, k=max(k, 20))
        ids = ranked.ranked
        ndcg = ndcg_at_k(ids, gold, k)
        recall = recall_any_at_k(ids, gold, k)
        mrr = mrr_at_k(ids, gold, k)
        ndcgs.append(ndcg)
        recalls.append(recall)
        mrrs.append(mrr)
        rows.append(
            {
                "item_id": query.item_id,
                "language": query.language,
                "ndcg@10": ndcg,
                "recall@10": recall,
                "mrr@10": mrr,
                "ranked": ids[:k],
            }
        )
    ndcg_interval = bootstrap_interval(ndcgs, samples=samples, seed=seed)
    recall_interval = bootstrap_interval(recalls, samples=samples, seed=seed)
    return RetrievalReport(
        arm=getattr(ranker, "name", "unknown"),
        n=len(queries),
        ndcg10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        ndcg10_ci=(ndcg_interval.low, ndcg_interval.high),
        recall10=float(np.mean(recalls)) if recalls else 0.0,
        recall10_ci=(recall_interval.low, recall_interval.high),
        mrr10=float(np.mean(mrrs)) if mrrs else 0.0,
        per_query=rows,
    )


def score_queries(ranker: Any, queries: Sequence[RetrievalQuery]) -> np.ndarray:
    """Score every query against the whole pool, batching dense embedding calls.

    Dense embedding is the expensive step; batching turns thousands of per-query requests into
    tens of ``embed_documents`` calls, which is what keeps the public arm inside the budget.
    """
    texts = [query.query for query in queries]
    if isinstance(ranker, LexicalRanker):
        return ranker.score_matrix(texts)
    if isinstance(ranker, DenseRanker):
        vectors: list[list[float]] = []
        batch = 32
        for start in range(0, len(texts), batch):
            vectors.extend(ranker.embedder.embed_documents(texts[start : start + batch]))
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0, 1.0, norms)
        return matrix @ ranker.matrix.T
    if isinstance(ranker, HybridRanker):
        lexical_rows = []
        for query in queries:
            lexical_rows.append(np.asarray(ranker._all_lexical(query.query), dtype=np.float64))
        lexical_matrix = np.vstack(lexical_rows)
        dense_vectors: list[list[float]] = []
        batch = 32
        for start in range(0, len(texts), batch):
            dense_vectors.extend(
                ranker.dense.embedder.embed_documents(texts[start : start + batch])
            )
        dense_matrix = np.asarray(dense_vectors, dtype=np.float64)
        norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
        dense_matrix = dense_matrix / np.where(norms == 0, 1.0, norms)
        dense_scores = dense_matrix @ ranker.dense.matrix.T
        fused = np.zeros_like(lexical_matrix)
        for row in range(lexical_matrix.shape[0]):
            fused[row] = ranker.alpha * _minmax(lexical_matrix[row]) + (
                1.0 - ranker.alpha
            ) * _minmax(dense_scores[row])
        return fused
    rows = [np.asarray([0.0] * len(ranker.passages), dtype=np.float64) for _ in queries]
    return np.vstack(rows)


def evaluate_scores(
    *,
    arm: str,
    scores: np.ndarray,
    queries: Sequence[RetrievalQuery],
    passage_ids: Sequence[str],
    k: int = 10,
    samples: int = 1000,
    seed: int = 0,
) -> RetrievalReport:
    """Metrics from a precomputed (query, passage) score matrix."""
    from answerdesk.eval.stats import bootstrap_interval

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    rows: list[dict[str, Any]] = []
    ids = np.asarray(passage_ids)
    for position, query in enumerate(queries):
        order = np.argsort(-scores[position])[: max(k, 20)]
        ranked = [str(ids[int(index)]) for index in order]
        gold = [passage_id for passage_id, grade in query.qrels.items() if grade > 0]
        ndcg = ndcg_at_k(ranked, gold, k)
        recall = recall_any_at_k(ranked, gold, k)
        mrr = mrr_at_k(ranked, gold, k)
        ndcgs.append(ndcg)
        recalls.append(recall)
        mrrs.append(mrr)
        rows.append(
            {
                "item_id": query.item_id,
                "language": query.language,
                "ndcg@10": ndcg,
                "recall@10": recall,
                "mrr@10": mrr,
                "ranked": ranked[:k],
            }
        )
    ndcg_interval = bootstrap_interval(ndcgs, samples=samples, seed=seed)
    recall_interval = bootstrap_interval(recalls, samples=samples, seed=seed)
    return RetrievalReport(
        arm=arm,
        n=len(queries),
        ndcg10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        ndcg10_ci=(ndcg_interval.low, ndcg_interval.high),
        recall10=float(np.mean(recalls)) if recalls else 0.0,
        recall10_ci=(recall_interval.low, recall_interval.high),
        mrr10=float(np.mean(mrrs)) if mrrs else 0.0,
        per_query=rows,
    )


def compare(
    baseline: RetrievalReport, candidate: RetrievalReport, *, samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap on the per-query differences (same queries, same qrels)."""
    base = np.array([row["ndcg@10"] for row in baseline.per_query], dtype=np.float64)
    cand = np.array([row["ndcg@10"] for row in candidate.per_query], dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError("paired comparison needs the same query set in both arms")
    ndcg = paired_bootstrap(cand, base, samples=samples, seed=seed)
    base_r = np.array([row["recall@10"] for row in baseline.per_query], dtype=np.float64)
    cand_r = np.array([row["recall@10"] for row in candidate.per_query], dtype=np.float64)
    recall = paired_bootstrap(cand_r, base_r, samples=samples, seed=seed)
    return {
        "baseline": baseline.arm,
        "candidate": candidate.arm,
        "n": int(ndcg.n),
        "ndcg@10": {
            "baseline": round(float(ndcg.mean_b), 6),
            "candidate": round(float(ndcg.mean_a), 6),
            "delta": round(float(ndcg.diff), 6),
            "ci_low": round(float(ndcg.low), 6),
            "ci_high": round(float(ndcg.high), 6),
            "ci_crosses_zero": bool(ndcg.low <= 0.0 <= ndcg.high),
        },
        "recall@10": {
            "baseline": round(float(recall.mean_b), 6),
            "candidate": round(float(recall.mean_a), 6),
            "delta": round(float(recall.diff), 6),
            "ci_low": round(float(recall.low), 6),
            "ci_high": round(float(recall.high), 6),
            "ci_crosses_zero": bool(recall.low <= 0.0 <= recall.high),
        },
    }


__all__ = [
    "DenseRanker",
    "EmbedderLike",
    "HybridRanker",
    "LexicalRanker",
    "RankedQuery",
    "RetrievalReport",
    "compare",
    "evaluate",
]
