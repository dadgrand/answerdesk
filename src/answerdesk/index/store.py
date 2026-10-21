"""The index directory: chunk store, manifest, build/load and idempotency (FR-OPS-02).

An *index build* is a function of four things, and this module makes that explicit:

    index_version = H(corpus_version, chunk-set hash, embedder name+dimension,
                      tokenizer version, BM25 params, dense metric)

Anything in that tuple changing forces a rebuild; nothing else does. That is what makes
``answerdesk index --demo`` idempotent (FR-OPS-02) — the second run recognises the
fingerprint and reports ``unchanged`` instead of rewriting 3,000 rows — and what stops a
silent mismatch between a corpus and the vectors that claim to describe it.

Layout on disk (``data/index/<corpus_version>/``)::

    index.json        manifest: fingerprint, counts, embedder, timings
    chunks.jsonl      the chunk set the indexes were built from
    bm25/             tokens.jsonl + bm25.json
    dense/            vectors.npy + dense.json

The chunk set is copied into the index directory on purpose: retrieval must be
reproducible from the index alone, even if the corpus directory is rebuilt underneath it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from answerdesk.config import Settings
from answerdesk.contracts import Chunk, Document, Embedder
from answerdesk.errors import CorpusMissingError, IndexMissingError
from answerdesk.index.bm25_index import BM25Index, BM25Params
from answerdesk.index.embedders import default_embedder
from answerdesk.index.text import TOKENIZER_VERSION
from answerdesk.index.vector_index import ExactVectorIndex

INDEX_SCHEMA_VERSION = "1.0.0"
INDEX_MANIFEST_FILE = "index.json"
CHUNKS_FILE = "chunks.jsonl"
BM25_DIR = "bm25"
DENSE_DIR = "dense"
#: Bumped when the indexed *representation* changes (breadcrumb prefix, field weights).
REPRESENTATION_VERSION = "chunk-text-with-breadcrumb-v1"


class IndexModel(BaseModel):
    """Strict base for index-internal models (mirrors ``contracts.Contract``)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EmbedderInfo(IndexModel):
    """Identity of the embedder an index was built with — part of the fingerprint."""

    name: str = Field(min_length=1)
    dimension: int = Field(gt=0)


class IndexManifest(IndexModel):
    """Everything needed to decide whether an index may be reused."""

    schema_version: str = INDEX_SCHEMA_VERSION
    corpus_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    chunks_sha256: str = Field(min_length=64, max_length=64)
    embedder: EmbedderInfo
    tokenizer: str = TOKENIZER_VERSION
    representation: str = REPRESENTATION_VERSION
    bm25: dict[str, Any] = Field(default_factory=dict)
    dense: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    built_at: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)


class ChunkStore:
    """In-memory chunk set plus the corpus manifest it came from (FR-ING-08 hand-off)."""

    def __init__(self, chunks: Sequence[Chunk], *, corpus_version: str, source: str = "") -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.corpus_version = corpus_version
        self.source = source
        self._by_id = {chunk.chunk_id for chunk in self.chunks}
        if len(self._by_id) != len(self.chunks):
            raise CorpusMissingError("the corpus contains duplicate chunk ids")

    def __len__(self) -> int:
        return len(self.chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    @classmethod
    def from_chunks(
        cls, chunks: Sequence[Chunk], *, corpus_version: str = "corpus-test"
    ) -> ChunkStore:
        return cls(chunks, corpus_version=corpus_version)

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path) -> ChunkStore:
        """Load ``chunks.jsonl`` + ``manifest.json`` produced by ``answerdesk ingest``."""
        from answerdesk.ingest.builder import load_manifest

        root = Path(corpus_dir)
        chunks_path = root / CHUNKS_FILE
        if not chunks_path.is_file():
            raise CorpusMissingError(
                f"no chunks at {chunks_path}; run `answerdesk ingest --demo` first"
            )
        manifest = load_manifest(root)
        chunks = [
            Chunk.model_validate(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if manifest.stats.chunks != len(chunks):
            raise CorpusMissingError(
                f"manifest says {manifest.stats.chunks} chunks but chunks.jsonl has "
                f"{len(chunks)}; re-run `answerdesk ingest --demo`"
            )
        return cls(chunks, corpus_version=manifest.corpus_version, source=str(root))

    @classmethod
    def from_documents(
        cls, documents: Sequence[Document], *, corpus_version: str = "corpus-test"
    ) -> ChunkStore:
        """Test helper: chunk documents with the production chunker (no corpus on disk)."""
        from answerdesk.ingest.builder import DEMO_CHUNKER
        from answerdesk.ingest.chunker import StructureChunker

        chunks = StructureChunker(DEMO_CHUNKER).chunk_all(list(documents))
        return cls(chunks, corpus_version=corpus_version)

    # -- representation ----------------------------------------------------------
    @staticmethod
    def text_for(chunk: Chunk) -> str:
        """The string that is actually indexed: breadcrumb + optional header + text.

        The breadcrumb (``Product > Billing > Refunds``) is prepended because support
        questions very often name the *section*, not the sentence. The chunk's own text is
        never edited (FR-ING-04, INV-01); this is the retrieval-side view only, and it is
        recorded as :data:`REPRESENTATION_VERSION` in the index fingerprint.
        """
        header = f"{chunk.breadcrumb}\n" if chunk.breadcrumb else ""
        context = f"{chunk.context_header}\n" if chunk.context_header else ""
        return f"{header}{context}{chunk.text}"

    # -- persistence -------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / CHUNKS_FILE
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(
                    json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        return path

    @classmethod
    def load(cls, directory: str | Path, *, corpus_version: str) -> ChunkStore:
        path = Path(directory) / CHUNKS_FILE
        if not path.is_file():
            raise IndexMissingError(f"no chunk set at {path}; rebuild the index")
        chunks = [
            Chunk.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks, corpus_version=corpus_version, source=str(directory))

    def fingerprint(self) -> str:
        """sha256 over chunk ids **and** text — a text edit must invalidate the index."""
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def compute_index_version(
    *,
    corpus_version: str,
    chunks_sha256: str,
    embedder: EmbedderInfo,
    bm25_params: BM25Params,
) -> str:
    """The index fingerprint (FR-OPS-02): change any input, get a new version."""
    payload = {
        "corpus_version": corpus_version,
        "chunks_sha256": chunks_sha256,
        "embedder": embedder.model_dump(mode="json"),
        "tokenizer": TOKENIZER_VERSION,
        "representation": REPRESENTATION_VERSION,
        "bm25": bm25_params.as_dict(),
        "dense": {"metric": "cosine", "exact": True},
        "schema": INDEX_SCHEMA_VERSION,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"index-{hashlib.sha256(blob).hexdigest()[:16]}"


class IndexBundle:
    """The built indexes plus their manifest; the pipeline's single input (FR-RET-08)."""

    def __init__(
        self,
        store: ChunkStore,
        bm25: BM25Index,
        dense: ExactVectorIndex,
        manifest: IndexManifest,
        *,
        index_dir: Path | None = None,
        reused: bool = False,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.dense = dense
        self.manifest = manifest
        self.index_dir = index_dir
        self.reused = reused

    @classmethod
    def build(
        cls,
        store: ChunkStore,
        embedder: Embedder,
        *,
        bm25_params: BM25Params | None = None,
        embedder_info: EmbedderInfo | None = None,
    ) -> IndexBundle:
        """Build both indexes over one chunk set, measuring each stage (NFR-COST-03).

        The embedder identity is read **after** the dense build on purpose: a hosted
        embedder may discover its width from the first response (``api_dimension: null``),
        so asking for ``embedder.dimension`` before embedding would fingerprint the index
        with ``0`` and let two different vector spaces share a version.
        """
        params = bm25_params or BM25Params()
        chunks_sha = store.fingerprint()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        bm25 = BM25Index.from_chunks(store.chunks, params=params, text_for=ChunkStore.text_for)
        timings["bm25_build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

        started = time.perf_counter()
        dense = ExactVectorIndex.from_chunks(store.chunks, embedder, text_for=ChunkStore.text_for)
        timings["dense_build_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

        if embedder_info is not None:
            info = embedder_info
        else:
            if embedder.dimension <= 0:
                raise IndexMissingError(
                    f"embedder {embedder.name!r} reports dimension 0 after embedding; "
                    "set embedding.api_dimension explicitly."
                )
            info = EmbedderInfo(name=embedder.name, dimension=embedder.dimension)

        manifest = IndexManifest(
            corpus_version=store.corpus_version,
            index_version=compute_index_version(
                corpus_version=store.corpus_version,
                chunks_sha256=chunks_sha,
                embedder=info,
                bm25_params=params,
            ),
            chunk_count=len(store),
            chunks_sha256=chunks_sha,
            embedder=info,
            bm25={**params.as_dict(), **{k: v for k, v in bm25.stats().items()}},
            dense={"metric": "cosine", "exact": True, "dimension": dense.dimension},
            timings_ms=timings,
            built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            stats={"chunks": len(store), "bm25": bm25.stats(), "dense": dense.stats()},
        )
        return cls(store, bm25, dense, manifest)

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.store.save(target)
        self.bm25.save(target / BM25_DIR)
        self.dense.save(target / DENSE_DIR)
        (target / INDEX_MANIFEST_FILE).write_text(
            json.dumps(
                self.manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.index_dir = target
        return target

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> IndexBundle:
        target = Path(directory)
        manifest_path = target / INDEX_MANIFEST_FILE
        if not manifest_path.is_file():
            raise IndexMissingError(
                f"no index manifest at {manifest_path}; run `answerdesk index --demo` first"
            )
        manifest = IndexManifest.model_validate(json.loads(manifest_path.read_text("utf-8")))
        if manifest.embedder.dimension != embedder.dimension:
            raise IndexMissingError(
                f"index built with {manifest.embedder.name!r} ({manifest.embedder.dimension}-d) "
                f"but {embedder.name!r} ({embedder.dimension}-d) is configured; rebuild the index"
            )
        store = ChunkStore.load(target, corpus_version=manifest.corpus_version)
        bm25 = BM25Index.load(target / BM25_DIR, store.chunks)
        dense = ExactVectorIndex.load(target / DENSE_DIR, store.chunks, embedder)
        return cls(store, bm25, dense, manifest, index_dir=target)

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.store),
            "corpus_version": self.store.corpus_version,
            "index_version": self.manifest.index_version,
            "embedder": self.manifest.embedder.model_dump(mode="json"),
            "bm25": self.bm25.stats(),
            "dense": self.dense.stats(),
        }


class IndexBuildResult(IndexModel):
    """Outcome reported by ``answerdesk index`` (FR-OPS-02)."""

    index_dir: str
    corpus_version: str
    index_version: str
    chunk_count: int
    reused: bool
    embedder: EmbedderInfo
    timings_ms: dict[str, float] = Field(default_factory=dict)


def index_root(settings: Settings) -> Path:
    return settings.resolve_path(settings.app.index_dir)


def index_dir_for(settings: Settings, corpus_version: str) -> Path:
    """``data/index/<corpus_version>`` — one directory per corpus identity."""
    return index_root(settings) / corpus_version


def build_indexes(
    settings: Settings,
    *,
    corpus_dir: Path | str | None = None,
    demo: bool = False,
    force: bool = False,
    embedder: Embedder | None = None,
    echo: Callable[[str], None] | None = None,
) -> IndexBuildResult:
    """Build (or reuse) both indexes for a corpus — the body of ``answerdesk index``."""
    printer = echo or print
    resolved_corpus = (
        Path(corpus_dir)
        if corpus_dir is not None
        else settings.resolve_path(settings.app.corpus_dir)
    )
    if demo and not (resolved_corpus / "manifest.json").is_file():
        from answerdesk.ingest.builder import build_demo
        from answerdesk.ingest.cli import synthetic_config_from_env

        printer(f"no corpus at {resolved_corpus}; building the demo corpus first")
        build_demo(resolved_corpus, config=synthetic_config_from_env())

    store = ChunkStore.from_corpus(resolved_corpus)
    resolved_embedder = embedder or default_embedder(settings)
    target = index_dir_for(settings, store.corpus_version)

    if not force and (target / INDEX_MANIFEST_FILE).is_file():
        existing = IndexManifest.model_validate(
            json.loads((target / INDEX_MANIFEST_FILE).read_text(encoding="utf-8"))
        )
        expected = compute_index_version(
            corpus_version=store.corpus_version,
            chunks_sha256=store.fingerprint(),
            embedder=EmbedderInfo(
                name=resolved_embedder.name, dimension=resolved_embedder.dimension
            ),
            bm25_params=BM25Params(),
        )
        if existing.index_version == expected:
            return IndexBuildResult(
                index_dir=str(target),
                corpus_version=store.corpus_version,
                index_version=existing.index_version,
                chunk_count=existing.chunk_count,
                reused=True,
                embedder=existing.embedder,
                timings_ms=existing.timings_ms,
            )
        printer(f"index fingerprint changed ({existing.index_version} -> {expected}); rebuilding")

    started = time.perf_counter()
    bundle = IndexBundle.build(store, resolved_embedder)
    bundle.save(target)
    total_ms = round((time.perf_counter() - started) * 1000.0, 3)
    timings = {**bundle.manifest.timings_ms, "total_ms": total_ms}
    return IndexBuildResult(
        index_dir=str(target),
        corpus_version=store.corpus_version,
        index_version=bundle.manifest.index_version,
        chunk_count=len(store),
        reused=False,
        embedder=bundle.manifest.embedder,
        timings_ms=timings,
    )


def load_indexes(
    settings: Settings,
    *,
    corpus_dir: Path | str | None = None,
    embedder: Embedder | None = None,
    index_dir: Path | str | None = None,
) -> IndexBundle:
    """Load the index for the corpus identity currently on disk (service path)."""
    resolved_embedder = embedder or default_embedder(settings)
    if index_dir is not None:
        return IndexBundle.load(index_dir, resolved_embedder)
    resolved_corpus = (
        Path(corpus_dir)
        if corpus_dir is not None
        else settings.resolve_path(settings.app.corpus_dir)
    )
    store = ChunkStore.from_corpus(resolved_corpus)
    return IndexBundle.load(index_dir_for(settings, store.corpus_version), resolved_embedder)


__all__ = [
    "BM25_DIR",
    "CHUNKS_FILE",
    "DENSE_DIR",
    "INDEX_MANIFEST_FILE",
    "INDEX_SCHEMA_VERSION",
    "REPRESENTATION_VERSION",
    "ChunkStore",
    "EmbedderInfo",
    "IndexBuildResult",
    "IndexBundle",
    "IndexManifest",
    "IndexModel",
    "build_indexes",
    "compute_index_version",
    "index_dir_for",
    "index_root",
    "load_indexes",
]
