"""Dense retrieval: exact cosine search over injected multilingual embeddings (FR-RET-02).

The default dense index is *exact* on purpose. At the v1.0 operating point (≥ 20,000
chunks) a full dot product against a float32 matrix is a few milliseconds, so an ANN
index would buy latency we do not need and cost recall we cannot afford to explain. The
optional Qdrant adapter lives in :mod:`answerdesk.index.qdrant_index` behind the
``qdrant`` extra and is never imported by the base install (NFR-PORT-03).

Cosine similarity is computed on **pre-normalised** rows:

    s(q, d) = <q / ||q||, d / ||d||>  =  q̂ · d̂

so a query costs one matrix-vector product and no per-document norm at search time.
Zero vectors (empty text) score 0 instead of raising ``nan``.

:func:`brute_force_top_k` is the reference implementation the exact index is tested
against: a plain Python loop over the same vectors (AT-RET-02). If the two ever disagree
on ids *or* order, the fast path is wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from answerdesk.contracts import Channel, Chunk, Embedder, RetrievedPassage
from answerdesk.errors import IndexMissingError

#: On-disk artefact names of the dense index inside an index directory.
VECTORS_FILE = "vectors.npy"
DENSE_META_FILE = "dense.json"

#: Guard against silently mixing embeddings from two embedders (a corpus-version bug).
_EPS = 1e-12


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows stay zero (no ``nan``, no ``inf``)."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array if norm <= _EPS else (array / norm).astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms <= _EPS, 1.0, norms)
    return (array / safe).astype(np.float32)


def brute_force_top_k(
    vectors: np.ndarray,
    query_vector: Sequence[float],
    k: int,
    *,
    positive_only: bool = False,
) -> list[tuple[int, float]]:
    """Reference exact search: ``(-score, position)``, computed in plain Python.

    Deliberately slow and obvious — it is the oracle for AT-RET-02, not a fast path. With
    ``positive_only=False`` (the default) it is a pure ranking oracle and returns negative
    cosines too, which is what the equivalence test needs.
    """
    if k <= 0 or len(vectors) == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    if query_norm > _EPS:
        query = query / query_norm
    scored: list[tuple[int, float]] = []
    for position in range(vectors.shape[0]):
        row = np.asarray(vectors[position], dtype=np.float64)
        row_norm = float(np.linalg.norm(row))
        if row_norm > _EPS:
            row = row / row_norm
        scored.append((position, float(np.dot(query, row))))
    scored.sort(key=lambda item: (-item[1], item[0]))
    if positive_only:
        scored = [item for item in scored if item[1] > 0.0]
    return scored[:k]


@dataclass
class ExactVectorIndex:
    """Exact cosine index implementing the ``Retriever`` protocol (FR-RET-02/03)."""

    name: str = "dense"
    chunks: list[Chunk] = field(default_factory=list)
    embedder: Embedder | None = None
    _matrix: np.ndarray | None = field(default=None, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, repr=False)

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        batch_size: int = 32,
        text_for: Callable[[Chunk], str] | None = None,
    ) -> ExactVectorIndex:
        """Embed every chunk (batched, deterministic) and normalise the matrix."""
        render = text_for if text_for is not None else (lambda chunk: chunk.text)
        texts = [render(chunk) for chunk in chunks]
        rows: list[list[float]] = []
        for start in range(0, len(texts), max(1, batch_size)):
            rows.extend(embedder.embed_documents(texts[start : start + batch_size]))
        matrix = l2_normalize(np.asarray(rows, dtype=np.float32).reshape(len(chunks), -1))
        if matrix.shape[1] != embedder.dimension:
            raise IndexMissingError(
                f"embedder {embedder.name!r} declares dimension {embedder.dimension} but "
                f"produced {matrix.shape[1]}-dimensional vectors"
            )
        index = cls(chunks=list(chunks), embedder=embedder)
        index.set_matrix(matrix)
        return index

    @classmethod
    def from_matrix(
        cls,
        chunks: Sequence[Chunk],
        matrix: np.ndarray,
        embedder: Embedder,
    ) -> ExactVectorIndex:
        """Wrap an existing matrix (used by tests and by index reload)."""
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != embedder.dimension:
            raise IndexMissingError(
                f"embedder {embedder.name!r} declares dimension {embedder.dimension} but the "
                f"matrix has shape {array.shape}"
            )
        index = cls(chunks=list(chunks), embedder=embedder)
        index.set_matrix(array)
        return index

    def set_matrix(self, matrix: np.ndarray) -> None:
        normalised = l2_normalize(np.asarray(matrix, dtype=np.float32))
        if normalised.shape[0] != len(self.chunks):
            raise IndexMissingError(
                f"matrix has {normalised.shape[0]} rows for {len(self.chunks)} chunks"
            )
        self._matrix = np.ascontiguousarray(normalised, dtype=np.float32)
        self._positions = {chunk.chunk_id: position for position, chunk in enumerate(self.chunks)}

    # -- properties ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise IndexMissingError("dense index is not built; call from_chunks() or load()")
        return self._matrix

    def vector_for(self, chunk_id: str) -> np.ndarray | None:
        """Normalised embedding of one chunk, or ``None`` when unknown (dedupe/MMR use)."""
        if self._matrix is None:
            return None
        position = self._positions.get(chunk_id)
        return None if position is None else self._matrix[position]

    # -- retrieval ----------------------------------------------------------------
    def search_vector(
        self, vector: Sequence[float], k: int, *, positive_only: bool = True
    ) -> list[tuple[int, float]]:
        """Exact top-k for a pre-computed query vector: ``(-score, position)``.

        ``positive_only`` drops non-positive cosines (an orthogonal or opposed chunk is
        not evidence); pass ``False`` to get the raw ranking, which is how the exactness
        test compares against :func:`brute_force_top_k`.
        """
        if k <= 0 or self.size == 0:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32))
        scores = np.asarray(self.matrix @ query, dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        picked: list[tuple[int, float]] = []
        for position in order[: max(1, min(k, self.size))]:
            value = float(scores[position])
            if positive_only and value <= 0.0:
                break
            picked.append((int(position), value))
        return picked

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        """``Retriever`` protocol: embed the query text and rank (FR-RET-03)."""
        if self.embedder is None:
            raise IndexMissingError("dense index has no embedder; cannot embed a query")
        return self.search_embedded(self.embedder.embed_query(query), k)

    def search_embedded(
        self,
        vector: Sequence[float],
        k: int,
        *,
        stage: Literal["retrieval", "fusion", "rerank", "pack"] = "retrieval",
    ) -> list[RetrievedPassage]:
        picked = self.search_vector(vector, k)
        return [
            RetrievedPassage(
                chunk=self.chunks[position],
                channel=Channel.DENSE,
                score=value,
                rank=rank,
                stage=stage,
                contributions={"cosine": value},
            )
            for rank, (position, value) in enumerate(picked, start=1)
        ]

    # -- persistence --------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        import json

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / VECTORS_FILE, self.matrix)
        meta = {
            "chunks": [chunk.chunk_id for chunk in self.chunks],
            "dimension": self.dimension,
            "metric": "cosine",
            "embedder": {
                "name": self.embedder.name if self.embedder else "unknown",
                "dimension": self.embedder.dimension if self.embedder else self.dimension,
            },
        }
        (target / DENSE_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        chunks: Sequence[Chunk],
        embedder: Embedder,
    ) -> ExactVectorIndex:
        import json

        source = Path(directory)
        meta_path, vectors_path = source / DENSE_META_FILE, source / VECTORS_FILE
        if not meta_path.is_file() or not vectors_path.is_file():
            raise IndexMissingError(
                f"no dense index at {source}; run `answerdesk index --demo` first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_ids = [str(value) for value in meta["chunks"]]
        if stored_ids != [chunk.chunk_id for chunk in chunks]:
            raise IndexMissingError(
                "the stored dense index does not match the corpus chunk set; rebuild it"
            )
        if int(meta["embedder"]["dimension"]) != embedder.dimension:
            raise IndexMissingError(
                f"the stored index was built with a {meta['embedder']['dimension']}-dimensional "
                f"embedder ({meta['embedder']['name']!r}); the configured embedder "
                f"{embedder.name!r} is {embedder.dimension}-dimensional. Rebuild the index."
            )
        matrix = np.load(vectors_path)
        return cls.from_matrix(chunks, matrix, embedder)

    def stats(self) -> dict[str, float | int]:
        """Memory note that ends up in ``docs/retrieval.md``."""
        rows, dimension = self.matrix.shape if self._matrix is not None else (0, 0)
        return {
            "chunks": self.size,
            "dimension": dimension,
            "bytes": int(self.matrix.nbytes) if self._matrix is not None else 0,
            "rows": int(rows),
        }


__all__ = [
    "DENSE_META_FILE",
    "VECTORS_FILE",
    "ExactVectorIndex",
    "brute_force_top_k",
    "l2_normalize",
]
