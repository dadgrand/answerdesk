"""The vector index behind the semantic cache: exact cosine, deterministic, in-process.

Requirements §9.3 talks about "embedding ANN"; at the scale this stack targets (thousands of
cached answers, not millions of vectors) exact cosine over a contiguous ``numpy`` matrix is
faster than an index build and — more importantly — gives *exactly* reproducible lookups, so
a replay harness can reason about thresholds without approximating them. The
:class:`SimilarityIndex` protocol is the seam where an ANN backend (faiss/hnswlib) can be
dropped in without touching the cache guards.

    # AT-CAC-02
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SimilarityIndex(Protocol):
    """Minimal nearest-neighbour seam used by the semantic cache."""

    def add(self, entry_id: str, vector: list[float]) -> None: ...

    def remove(self, entry_id: str) -> bool: ...

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class ExactCosineIndex:
    """Exact cosine search over unit-normalised rows, with a stable id ↔ row mapping."""

    def __init__(self, dimension: int | None = None, *, min_similarity: float = -1.0) -> None:
        self.dimension = dimension
        self.min_similarity = min_similarity
        self._ids: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------------
    def _normalise(self, vector: list[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("embedding must be a non-empty 1-D vector")
        if self.dimension is None:
            self.dimension = int(array.size)
        elif array.size != self.dimension:
            raise ValueError(f"embedding has dimension {array.size}, expected {self.dimension}")
        norm = float(np.linalg.norm(array))
        return array if norm == 0.0 else array / norm

    def _rebuild(self) -> None:
        if not self._ids:
            self._matrix = None
            return
        self._matrix = np.vstack([self._vectors[entry_id] for entry_id in self._ids])

    # -- API ----------------------------------------------------------------------
    def add(self, entry_id: str, vector: list[float]) -> None:
        normalised = self._normalise(vector)
        if entry_id not in self._vectors:
            self._ids.append(entry_id)
        self._vectors[entry_id] = normalised
        self._rebuild()

    def remove(self, entry_id: str) -> bool:
        if entry_id not in self._vectors:
            return False
        del self._vectors[entry_id]
        self._ids = [existing for existing in self._ids if existing != entry_id]
        self._rebuild()
        return True

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if not self._ids or self._matrix is None or k <= 0:
            return []
        query = self._normalise(vector)
        scores = self._matrix @ query
        order = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        for row in order:
            similarity = float(scores[int(row)])
            if similarity < self.min_similarity:
                continue
            results.append((self._ids[int(row)], similarity))
            if len(results) >= k:
                break
        return results

    def __len__(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._vectors.clear()
        self._matrix = None


__all__ = ["ExactCosineIndex", "SimilarityIndex"]
