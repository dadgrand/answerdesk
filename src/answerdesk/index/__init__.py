"""Sparse and dense indexes over the corpus.

Owning task: **t5**. Two indexes with one shape — both implement
:class:`answerdesk.contracts.Retriever` and both return
:class:`answerdesk.contracts.RetrievedPassage` records, so nothing downstream can tell
which channel produced a candidate and no adapter detail leaks into callers (FR-RET-03).

Modules
-------
``text``          deterministic multilingual tokenisation, canonical identifier tokens,
                  the corpus :class:`~answerdesk.index.text.TermVocabulary`
``bm25_index``    BM25 over chunks (``bm25s`` scoring, our own deterministic top-k,
                  identifier must-match probe, token-level persistence)
``vector_index``  exact cosine over injected multilingual embeddings, with a brute-force
                  reference implementation that the tests compare against
``qdrant_index``  optional adapter behind the ``qdrant`` extra (lazy import)
``embedders``     configuration → embedder (hashing in CI, sentence-transformers locally)
``store``         the index directory: chunk set, fingerprint, build/reuse, load
``cli``           ``answerdesk index`` (FR-OPS-02), delegated from the root CLI

Requirements: FR-RET-01, FR-RET-02, FR-RET-03, FR-RET-09, FR-OPS-02, NFR-PORT-03.
Maths and measurements: ``docs/retrieval.md``.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.index.bm25_index import BM25Index, BM25Params, merge_probes
from answerdesk.index.embedders import default_embedder
from answerdesk.index.store import (
    ChunkStore,
    EmbedderInfo,
    IndexBuildResult,
    IndexBundle,
    IndexManifest,
    build_indexes,
    compute_index_version,
    index_dir_for,
    load_indexes,
)
from answerdesk.index.text import (
    TOKENIZER_VERSION,
    TermVocabulary,
    identifier_tokens,
    layout_variant,
    normalize_text,
    tokenize,
)
from answerdesk.index.vector_index import ExactVectorIndex, brute_force_top_k, l2_normalize

__all__ = [
    "TOKENIZER_VERSION",
    "BM25Index",
    "BM25Params",
    "ChunkStore",
    "EmbedderInfo",
    "ExactVectorIndex",
    "IndexBuildResult",
    "IndexBundle",
    "IndexManifest",
    "TermVocabulary",
    "brute_force_top_k",
    "build_indexes",
    "compute_index_version",
    "default_embedder",
    "identifier_tokens",
    "index_dir_for",
    "l2_normalize",
    "layout_variant",
    "load_indexes",
    "merge_probes",
    "normalize_text",
    "tokenize",
]
