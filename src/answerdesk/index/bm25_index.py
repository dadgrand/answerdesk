"""Sparse retrieval: BM25 over chunks with a deterministic top-k (FR-RET-01, FR-RET-03).

Scoring is done by ``bm25s`` (Lucene's BM25 variant: ``k1=1.5``, ``b=0.75``, idf
``ln(1 + (N - df + 0.5) / (df + 0.5))``). Two deliberate design choices sit *around*
that library:

**Deterministic selection.** ``bm25s`` returns its own ``k`` best documents, and ties at
the cut-off are sorted by whatever order the internal heap produced. This index instead
asks for the **full score vector** (``BM25.get_scores``), then sorts with
``numpy.argsort(-scores, kind="stable")`` — ascending corpus order resolves every tie.
Same corpus, same query, same output, in any process (AT-RET-01).

**A must-match probe.** Exact identifiers are a documented dense failure mode (D-07), so
:meth:`BM25Index.search_terms` accepts ``required`` tokens: a chunk must contain *all* of
them to score at all. The retrieval pipeline runs the same sparse channel twice — once
over the expanded query, once restricted to the identifier tokens — and merges the two
rankings.

Persistence stores the *tokens*, not the library's internal state: reloading rebuilds the
postings from the same token lists, which keeps the on-disk format stable across
``bm25s`` upgrades and makes an index rebuild byte-reproducible (FR-OPS-02).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from answerdesk.contracts import Channel, Chunk, RetrievedPassage
from answerdesk.errors import IndexMissingError
from answerdesk.index.text import TermVocabulary, tokenize

#: On-disk file names of the sparse index inside an index directory.
TOKENS_FILE = "tokens.jsonl"
BM25_META_FILE = "bm25.json"

#: ``bm25s`` BM25 parameters. Frozen: changing them changes the index version.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_METHOD = "lucene"


@dataclass(frozen=True)
class BM25Params:
    """Frozen BM25 hyper-parameters; part of the index fingerprint."""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    method: str = DEFAULT_METHOD

    def as_dict(self) -> dict[str, float | str]:
        return {"k1": self.k1, "b": self.b, "method": self.method}


@dataclass
class BM25Index:
    """BM25 index over a chunk list (``Retriever`` protocol: :meth:`search`)."""

    name: str = "bm25"
    chunk_ids: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    tokens: list[list[str]] = field(default_factory=list)
    params: BM25Params = field(default_factory=BM25Params)
    _engine: Any = field(default=None, repr=False)
    _token_sets: list[frozenset[str]] = field(default_factory=list, repr=False)
    _vocabulary: TermVocabulary | None = field(default=None, repr=False)

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        *,
        params: BM25Params | None = None,
        text_for: Callable[[Chunk], str] | None = None,
    ) -> BM25Index:
        """Build the index. ``text_for(chunk) -> str`` defaults to the chunk text.

        The caller supplies the *indexed representation* so the dense and sparse indexes
        see identical text (breadcrumb-prefixed by the store). Chunk text itself is never
        modified (INV-01, FR-ING-04).
        """
        resolved = params or BM25Params()
        render = text_for if text_for is not None else (lambda chunk: chunk.text)
        tokens = [tokenize(render(chunk)) for chunk in chunks]
        index = cls(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            chunks=list(chunks),
            tokens=tokens,
            params=resolved,
        )
        index._build()
        return index

    def _build(self) -> None:
        import bm25s

        engine = bm25s.BM25(k1=self.params.k1, b=self.params.b, method=self.params.method)
        engine.index(self.tokens, show_progress=False)
        self._engine = engine
        self._token_sets = [frozenset(tokens) for tokens in self.tokens]
        self._vocabulary = None

    # -- properties ---------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunk_ids)

    @property
    def vocabulary(self) -> TermVocabulary:
        """Corpus vocabulary, built lazily (one per index, reused by query analysis)."""
        if self._vocabulary is None:
            self._vocabulary = TermVocabulary(self.tokens)
        return self._vocabulary

    @property
    def engine(self) -> Any:
        if self._engine is None:
            raise IndexMissingError("BM25 index is not built; call from_chunks() or load()")
        return self._engine

    # -- retrieval ----------------------------------------------------------------
    def scores(self, terms: Sequence[str]) -> np.ndarray:
        """Full BM25 score vector over the corpus (zeros for non-matching chunks)."""
        if self.size == 0:
            return np.zeros(0, dtype=np.float64)
        if not terms:
            return np.zeros(self.size, dtype=np.float64)
        raw = np.asarray(self.engine.get_scores(list(terms)), dtype=np.float64)
        if raw.shape != (self.size,):  # pragma: no cover - bm25s shape guard
            raise IndexMissingError(
                f"bm25s returned {raw.shape} scores for {self.size} chunks; index is corrupt"
            )
        return raw

    def top_k_from_scores(
        self,
        scores: np.ndarray,
        k: int,
        *,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Deterministic top-k over a score vector: ``(-score, corpus position)``.

        Zero-score chunks are never returned: a chunk that shares no term with the query
        is not a candidate, and padding the list with it would corrupt fusion ranks.
        """
        if k <= 0 or scores.size == 0:
            return []
        working = scores if allowed is None else np.where(allowed, scores, 0.0)
        order = np.argsort(-working, kind="stable")
        picked: list[tuple[int, float]] = []
        for position in order:
            value = float(working[position])
            if value <= 0.0:
                break
            picked.append((int(position), value))
            if len(picked) >= k:
                break
        return picked

    def _allowed_mask(self, required: Sequence[str]) -> np.ndarray:
        """Chunks containing **every** required token (the must-match probe)."""
        mask = np.array(
            [all(token in token_set for token in required) for token_set in self._token_sets],
            dtype=bool,
        )
        return mask

    def search_terms(
        self,
        terms: Sequence[str],
        k: int,
        *,
        required: Sequence[str] = (),
        stage: Literal["retrieval", "fusion", "rerank", "pack"] = "retrieval",
    ) -> list[RetrievedPassage]:
        """Rank the corpus for pre-tokenised ``terms``, optionally must-match filtered."""
        if k <= 0 or self.size == 0 or not terms:
            return []
        scores = self.scores(terms)
        allowed = self._allowed_mask(required) if required else None
        if allowed is not None and not allowed.any():
            return []
        picked = self.top_k_from_scores(scores, k, allowed=allowed)
        contributions_extra = {"identifier_match": 1.0} if required else {}
        return [
            RetrievedPassage(
                chunk=self.chunks[position],
                channel=Channel.BM25,
                score=value,
                rank=rank,
                stage=stage,
                contributions={"bm25": value, **contributions_extra},
            )
            for rank, (position, value) in enumerate(picked, start=1)
        ]

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        """``Retriever`` protocol: tokenise the raw query and rank (FR-RET-03)."""
        return self.search_terms(tokenize(query), k)

    # -- persistence --------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Write tokens + parameters; the postings are rebuilt on load."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / TOKENS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
            for tokens in self.tokens:
                handle.write(json.dumps(tokens, ensure_ascii=False) + "\n")
        meta = {
            "params": self.params.as_dict(),
            "chunk_ids": self.chunk_ids,
            "size": self.size,
        }
        (target / BM25_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path, chunks: Sequence[Chunk]) -> BM25Index:
        """Rebuild the index from persisted tokens, asserting chunk alignment."""
        source = Path(directory)
        meta_path = source / BM25_META_FILE
        tokens_path = source / TOKENS_FILE
        if not meta_path.is_file() or not tokens_path.is_file():
            raise IndexMissingError(
                f"no BM25 index at {source}; run `answerdesk index --demo` first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        params = BM25Params(
            k1=float(meta["params"]["k1"]),
            b=float(meta["params"]["b"]),
            method=str(meta["params"]["method"]),
        )
        chunk_ids: list[str] = [str(value) for value in meta["chunk_ids"]]
        if len(chunk_ids) != len(chunks) or chunk_ids != [chunk.chunk_id for chunk in chunks]:
            raise IndexMissingError(
                "the stored BM25 index does not match the corpus chunk set; rebuild it"
            )
        tokens = [
            [str(token) for token in json.loads(line)]
            for line in tokens_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index = cls(
            chunk_ids=chunk_ids,
            chunks=list(chunks),
            tokens=tokens,
            params=params,
        )
        index._build()
        return index

    def stats(self) -> dict[str, float | int]:
        """Index statistics for the build report and the memory note in the docs."""
        lengths = [len(tokens) for tokens in self.tokens] or [0]
        return {
            "chunks": self.size,
            "vocabulary": len(self.vocabulary),
            "tokens_total": int(sum(lengths)),
            "tokens_mean": round(float(np.mean(lengths)), 3),
            "tokens_max": int(max(lengths)),
        }


def merge_probes(
    primary: Iterable[RetrievedPassage],
    secondary: Iterable[RetrievedPassage],
) -> list[RetrievedPassage]:
    """Merge two rankings of the same channel by best rank, then re-rank 1..n.

    Used to fold the identifier must-match probe into the sparse channel: a chunk that
    both probes found keeps its better rank, and one that only the must-match probe found
    is still a candidate. Ties break on ``chunk_id`` so the merge is deterministic.
    """
    best: dict[str, tuple[int, float, RetrievedPassage]] = {}
    for passage in [*primary, *secondary]:
        current = best.get(passage.chunk.chunk_id)
        candidate = (passage.rank, -passage.score, passage)
        if current is None or candidate[:2] < current[:2]:
            best[passage.chunk.chunk_id] = candidate
    ordered = sorted(best.values(), key=lambda item: (item[0], item[1], item[2].chunk.chunk_id))
    merged: list[RetrievedPassage] = []
    for rank, (_, _, passage) in enumerate(ordered, start=1):
        data = passage.model_dump()
        data["rank"] = rank
        merged.append(RetrievedPassage.model_validate(data))
    return merged


__all__ = [
    "BM25_META_FILE",
    "DEFAULT_B",
    "DEFAULT_K1",
    "DEFAULT_METHOD",
    "TOKENS_FILE",
    "BM25Index",
    "BM25Params",
    "merge_probes",
]
