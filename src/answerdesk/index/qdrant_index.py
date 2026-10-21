"""Optional Qdrant adapter — the same ``Retriever`` shape, behind the ``qdrant`` extra.

Why it exists at all: some deployments already run a vector store and expect retrieval to
land there. Why it is *not* the default: at the v1.0 operating point the exact numpy index
in :mod:`answerdesk.index.vector_index` answers in milliseconds with no service to
operate, no drift between "the index" and "the corpus", and no ANN recall to explain.

This module imports ``qdrant_client`` **inside** the methods, never at import time, so the
base install stays free of it (NFR-PORT-03). Constructing the adapter without the extra
raises :class:`answerdesk.errors.ConfigError` with the exact command to fix it, instead of
an ``ImportError`` from three frames deep.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from answerdesk.contracts import Channel, Chunk, Embedder, RetrievedPassage
from answerdesk.errors import ConfigError

#: How many payload fields are copied next to each vector (enough to rebuild a passage).
_PAYLOAD_FIELDS = ("chunk_id", "doc_id", "text", "breadcrumb", "language", "token_count")


class QdrantVectorIndex:
    """Dense retrieval against a Qdrant collection (FR-RET-02 optional adapter)."""

    name = "qdrant"

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        collection: str = "answerdesk_chunks",
        embedder: Embedder,
    ) -> None:
        self.url = url
        self.collection = collection
        self.embedder = embedder
        self._client: Any = None
        self._chunks: dict[str, Chunk] = {}

    # -- lifecycle ---------------------------------------------------------------
    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ConfigError(
                "the Qdrant adapter needs the 'qdrant' extra: run `uv sync --extra qdrant`, "
                "or use the default exact numpy index (retrieval.dense_backend=numpy)."
            ) from exc
        self._client = QdrantClient(url=self.url)
        return self._client

    def create(self, chunks: Sequence[Chunk], *, recreate: bool = False) -> None:
        """Create the collection and upsert the corpus vectors (idempotent upsert)."""
        client = self._connect()
        from qdrant_client import models

        existing = {collection.name for collection in client.get_collections().collections}
        if self.collection in existing:
            if not recreate:
                raise ConfigError(
                    f"Qdrant collection {self.collection!r} already exists; pass recreate=True "
                    "to replace it (the exact numpy index is the default for a reason)."
                )
            client.delete_collection(self.collection)
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        texts = [chunk.text for chunk in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            vectors.extend(self.embedder.embed_documents(texts[start : start + 32]))
        # Embed *before* creating the collection: a hosted embedder may only learn its
        # width from the first response, and Qdrant needs the size up front.
        if self.embedder.dimension <= 0:
            raise ConfigError(
                "the embedder reports dimension 0; set embedding.api_dimension before "
                "creating a Qdrant collection"
            )
        client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        points = [
            models.PointStruct(
                id=index,
                vector=vector,
                payload={field: getattr(chunk, field, None) for field in _PAYLOAD_FIELDS},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), 256):
            client.upsert(collection_name=self.collection, points=points[start : start + 256])

    # -- retrieval ---------------------------------------------------------------
    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        """``Retriever`` protocol over Qdrant's cosine search."""
        if k <= 0:
            return []
        client = self._connect()
        hits = client.search(
            collection_name=self.collection,
            query_vector=list(np.asarray(self.embedder.embed_query(query), dtype=float)),
            limit=k,
            with_payload=True,
        )
        passages: list[RetrievedPassage] = []
        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            chunk = self._chunks.get(str(payload.get("chunk_id", "")))
            if chunk is None:
                continue
            passages.append(
                RetrievedPassage(
                    chunk=chunk,
                    channel=Channel.DENSE,
                    score=float(hit.score),
                    rank=rank,
                    contributions={"cosine": float(hit.score)},
                )
            )
        return passages


def build_qdrant_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    url: str = "http://127.0.0.1:6333",
    collection: str = "answerdesk_chunks",
    recreate: bool = False,
) -> QdrantVectorIndex:
    """Convenience factory used by the ``local`` profile docs."""
    index = QdrantVectorIndex(url=url, collection=collection, embedder=embedder)
    index.create(chunks, recreate=recreate)
    return index


TextRenderer = Callable[[Chunk], str]

__all__ = ["QdrantVectorIndex", "build_qdrant_index"]
