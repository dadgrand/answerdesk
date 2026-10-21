"""Offline retrieval arms: BM25, hashed-dense and their RRF fusion.

Why the harness owns retrieval arms at all: every retrieval metric needs a ranking, and
the evidence run (t11) needs the *same* metric code for the shipped retriever. These arms
are the deterministic reference implementations the CI gate can afford — they are
deliberately simple, dependency-free and reproducible, and they exist so that
``answerdesk eval --synthetic --provider fake`` measures a real ranking over the real
corpus instead of a fixture.

They are **not** the shipped retrieval stack: ``answerdesk.retrieval`` (t5) owns BM25 +
dense + fusion with a real embedding model, reranking and MMR. When t5 lands, the runner
accepts any :class:`answerdesk.contracts.Retriever`, so the arms stay useful as the
`bm25`/`dense`/`hybrid-rrf` ablation baseline (§6.6) and as the offline fallback.

Determinism notes: scores are rounded to 12 decimals and ties break on ``chunk_id``, so a
ranking is a pure function of corpus + query, in any process, on any OS.

    # AT-EVL-04  # AT-RET-04  # AT-DET-02
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from answerdesk.contracts import Channel, Chunk, RetrievedPassage
from answerdesk.eval.corpus import Corpus, words
from answerdesk.testing.doubles import HashingEmbedder

#: Okapi BM25 parameters — the literature defaults, pinned so a ranking is reproducible.
BM25_K1 = 1.2
BM25_B = 0.75
#: Reciprocal-rank-fusion constant (Elastic/Weaviate default, §6.2 / research D-05).
RRF_K = 60


def _tiebreak(score: float, chunk_id: str) -> tuple[float, str]:
    return (-round(score, 12), chunk_id)


@dataclass(frozen=True)
class ArmSpec:
    """Identity and description of one retrieval arm, printed in the report."""

    name: str
    description: str


BM25_SPEC = ArmSpec("bm25", "Okapi BM25 over chunk content tokens (k1=1.2, b=0.75)")
DENSE_SPEC = ArmSpec("dense-hash", "hashed bag-of-features cosine over chunk text")
RRF_SPEC = ArmSpec("hybrid-rrf", "reciprocal rank fusion of bm25 and dense-hash (k=60)")
DEFAULT_ARMS: tuple[str, ...] = (BM25_SPEC.name, DENSE_SPEC.name, RRF_SPEC.name)


class _RetrieverBase:
    """Shared scoring loop: turn per-chunk scores into ranked passages."""

    name = "arm"
    channel = Channel.BM25
    _corpus: Corpus

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    def score_all(self, query: str) -> Mapping[str, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        scores = self.score_all(query)
        ordered = [item for item in scores.items() if item[1] > 0.0]
        ordered.sort(key=lambda item: _tiebreak(item[1], item[0]))
        top = ordered[: max(0, k)]
        return [
            RetrievedPassage(
                chunk=self._chunk(chunk_id),
                channel=self.channel,
                score=float(round(score, 12)),
                rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(top, start=1)
        ]

    def _chunk(self, chunk_id: str) -> Chunk:  # pragma: no cover - overridden
        raise NotImplementedError


class BM25Arm(_RetrieverBase):
    """Okapi BM25 over the chunk token bags (no stemming, no stopword list)."""

    name = BM25_SPEC.name
    channel = Channel.BM25

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._tokens = {chunk_id: sorted(bag) for chunk_id, bag in corpus.chunk_tokens().items()}
        self._length = {chunk_id: len(bag) for chunk_id, bag in self._tokens.items()}
        self._average_length = (
            sum(self._length.values()) / len(self._length) if self._length else 0.0
        )
        postings: dict[str, list[tuple[str, int]]] = {}
        for chunk_id, tokens in self._tokens.items():
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                postings.setdefault(token, []).append((chunk_id, count))
        self._postings = postings
        total = max(1, len(self._tokens))
        self._idf = {
            token: math.log(1.0 + (total - len(ids) + 0.5) / (len(ids) + 0.5))
            for token, ids in postings.items()
        }

    @property
    def average_length(self) -> float:
        return self._average_length

    def score_all(self, query: str) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        query_tokens = [token for token in words(query) if token in self._postings]
        if not query_tokens or self._average_length <= 0:
            return scores
        for token in query_tokens:
            idf = self._idf[token]
            for chunk_id, frequency in self._postings[token]:
                length = self._length[chunk_id]
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / self._average_length
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    frequency * (BM25_K1 + 1.0) / denominator
                )
        return scores

    def _chunk(self, chunk_id: str) -> Chunk:
        return self._corpus.by_id[chunk_id]


class DenseHashArm(_RetrieverBase):
    """Cosine similarity between hashed query and chunk vectors (CI-only embedder)."""

    name = DENSE_SPEC.name
    channel = Channel.DENSE

    def __init__(self, corpus: Corpus, embedder: HashingEmbedder | None = None) -> None:
        self._corpus = corpus
        self._embedder = embedder or HashingEmbedder()
        self._chunk_ids = [chunk.chunk_id for chunk in corpus.chunks]
        self._matrix = np.asarray(
            self._embedder.embed_documents([chunk.text for chunk in corpus.chunks]),
            dtype=np.float64,
        )

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def score_all(self, query: str) -> Mapping[str, float]:
        vector = np.asarray(self._embedder.embed_query(query), dtype=np.float64)
        if self._matrix.size == 0:  # pragma: no cover - empty corpus is rejected earlier
            return {}
        similarities = self._matrix @ vector
        return {
            chunk_id: float(round(float(similarity), 12))
            for chunk_id, similarity in zip(self._chunk_ids, similarities, strict=True)
        }

    def _chunk(self, chunk_id: str) -> Chunk:
        return self._corpus.by_id[chunk_id]


class RrfArm(_RetrieverBase):
    """Reciprocal rank fusion of two arms: ``score = Σ 1 / (k + rank)`` (§6.1, D-05)."""

    name = RRF_SPEC.name

    def __init__(self, arms: Sequence[_RetrieverBase], *, k: int = RRF_K, depth: int = 100) -> None:
        if len(arms) < 2:
            raise ValueError("RRF needs at least two component arms")
        self._arms = list(arms)
        self._k = k
        self._depth = depth
        self._corpus = arms[0].corpus

    @property
    def rrf_k(self) -> int:
        return self._k

    def score_all(self, query: str) -> Mapping[str, float]:
        scores: dict[str, float] = {}
        for arm in self._arms:
            for passage in arm.search(query, self._depth):
                scores[passage.chunk.chunk_id] = scores.get(passage.chunk.chunk_id, 0.0) + 1.0 / (
                    self._k + passage.rank
                )
        return scores

    def _chunk(self, chunk_id: str) -> Chunk:
        return self._corpus.by_id[chunk_id]


def build_arm(corpus: Corpus, name: str) -> _RetrieverBase:
    """Construct one arm by name; unknown names are a configuration error."""
    if name == BM25_SPEC.name:
        return BM25Arm(corpus)
    if name == DENSE_SPEC.name:
        return DenseHashArm(corpus)
    if name == RRF_SPEC.name:
        return RrfArm([BM25Arm(corpus), DenseHashArm(corpus)])
    known = ", ".join(DEFAULT_ARMS)
    raise ValueError(f"unknown retrieval arm {name!r}; known arms: {known}")


def build_arms(corpus: Corpus, names: Iterable[str] = DEFAULT_ARMS) -> dict[str, _RetrieverBase]:
    """Build the requested arms, sharing the corpus (and therefore the chunk objects)."""
    arms: dict[str, _RetrieverBase] = {}
    shared: dict[str, _RetrieverBase] = {}
    for name in names:
        if name == RRF_SPEC.name:
            components = [
                shared.setdefault(BM25_SPEC.name, BM25Arm(corpus)),
                shared.setdefault(DENSE_SPEC.name, DenseHashArm(corpus)),
            ]
            arms[name] = RrfArm(components)
        else:
            arms[name] = shared.setdefault(name, build_arm(corpus, name))
    return arms


def arm_specs(names: Iterable[str]) -> list[ArmSpec]:
    table = {spec.name: spec for spec in (BM25_SPEC, DENSE_SPEC, RRF_SPEC)}
    return [table[name] for name in names if name in table]


__all__ = [
    "BM25_SPEC",
    "DEFAULT_ARMS",
    "DENSE_SPEC",
    "RRF_K",
    "RRF_SPEC",
    "ArmSpec",
    "BM25Arm",
    "DenseHashArm",
    "RrfArm",
    "arm_specs",
    "build_arm",
    "build_arms",
]
