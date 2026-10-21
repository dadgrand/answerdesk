"""Idempotent corpus builder and provenance manifest (FR-ING-07/08, NFR-REP-02).

One function builds a corpus; the same call twice produces byte-identical output. That
is what makes the checked-in demo corpus an *artifact* rather than a hand-edited folder:
the manifest is a pure function of (sources, chunker config, dedup config, generator
seed), with no timestamps, no absolute paths and no dictionary-order dependence.

Provenance guarantees
---------------------
* every source records ``path`` (repo-relative, POSIX), ``sha256`` (of the bytes on
  disk), ``byte_size``, ``license`` and ``language`` — a manifest is auditable without
  the files;
* ``corpus_version`` is a hash of the sources and of the chunking configuration, so any
  change to either produces a new version and invalidates the cache (FR-CAC-05);
* the identifier scan (§13.4) runs on every generated corpus and **refuses to write** a
  corpus that contains a non-reserved phone, PAN, e-mail domain or a denylisted brand;
* ``chunks.jsonl`` and ``queries/queries.jsonl`` are recorded by hash, so a consumer can
  detect a stale derived artifact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from answerdesk.contracts import Chunk, Document
from answerdesk.ingest.chunker import ChunkerConfig, StructureChunker
from answerdesk.ingest.dedup import DedupConfig, DedupResult, NearDuplicateDetector, text_hash
from answerdesk.ingest.errors import CorpusBuildError, SourceNotFoundError
from answerdesk.ingest.loaders import SUPPORTED_FORMATS, load_document
from answerdesk.ingest.structure import IngestModel
from answerdesk.ingest.synthetic import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    GeneratedDocument,
    GeneratedQuery,
    SyntheticConfig,
    generate,
    grounding_report,
    identifier_findings,
)
from answerdesk.ingest.tokenizer import TOKENIZER_VERSION

#: Manifest schema version — bump on any incompatible manifest change.
MANIFEST_SCHEMA_VERSION = "2.1.0"

#: Relative locations inside a corpus directory.
DOCS_DIR = "docs"
CHUNKS_FILE = "chunks.jsonl"
QUERIES_FILE = "queries/queries.jsonl"
MANIFEST_FILE = "manifest.json"

#: Demo chunking profile. 320 tokens keeps a KB section plus its FAQ pair intact and
#: puts the demo corpus at ~4 chunks per article; retrieval ablations (t5/t7) re-run the
#: same build with 256/512 to show the sensitivity. Recorded in the manifest.
DEMO_CHUNKER = ChunkerConfig(token_budget=320, overlap_tokens=48)


class SourceRecord(IngestModel):
    """One document as it exists on disk, with the hashes that prove it."""

    doc_id: str
    path: str
    sha256: str
    byte_size: int
    license: str
    language: str | None = None
    source_format: str
    title: str
    intent: str | None = None
    archetype: str | None = None
    heading_count: int = 0
    block_count: int = 0
    chunks: int = 0


class ChunkDedupRecord(IngestModel):
    """Exact-duplicate removal at chunk level (near-duplicates are removed per document)."""

    removed: int = 0
    by_hash: bool = True


class DocumentDedupRecord(IngestModel):
    """MinHash near-duplicate removal at document level (FR-ING-06)."""

    threshold: float
    num_perm: int
    shingle_size: int
    seed: int
    bands: int
    candidates: int
    kept: int
    removed: int
    pairs: list[dict[str, Any]] = []


class GeneratorRecord(IngestModel):
    """Which generator produced the documents, and how (FR-ING-10)."""

    name: str
    version: str
    seed: int
    locales: list[str]
    docs_per_locale: int
    duplicate_every: int
    documents: int
    duplicates: int
    queries: int
    perturbation_counts: dict[str, int] = {}
    perturbation_shares: dict[str, float] = {}
    categories: dict[str, int] = {}
    archetypes: list[str] = []


class ManifestStats(IngestModel):
    """Aggregate counts. ``chunks`` and ``documents`` are the gated numbers (§16.1)."""

    documents: int
    chunks: int
    tokens: int
    tokens_mean: float
    tokens_max: int
    chunks_with_heading_path: int
    chunks_with_canonical_tokens: int
    chunks_with_breadcrumb: int
    docs_per_locale: dict[str, int] = {}
    chunks_per_locale: dict[str, int] = {}
    blocks_by_kind: dict[str, int] = {}
    tables: int = 0
    faq_pairs: int = 0
    duplicate_documents_removed: int = 0
    duplicate_chunks_removed: int = 0


class CorpusManifest(IngestModel):
    """The single source of truth about a built corpus (FR-ING-08)."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    corpus_version: str
    license: str
    generator: GeneratorRecord | None = None
    sources: list[SourceRecord]
    stats: ManifestStats
    dedup: DocumentDedupRecord
    chunk_dedup: ChunkDedupRecord
    chunking: dict[str, Any]
    tokenizer: str = TOKENIZER_VERSION
    queries: dict[str, Any] = {}
    identifier_scan: dict[str, Any] = {}
    reference_grounding: dict[str, Any] = {}
    archetypes: list[str] = []
    locales: dict[str, Any] = {}
    content_provenance: dict[str, Any] = {}
    artifacts: dict[str, str] = {}


class BuildResult(IngestModel):
    """What a build produced: paths on disk, the manifest and the loaded chunks."""

    corpus_dir: str
    manifest_path: str
    chunks_path: str
    queries_path: str | None = None
    manifest: CorpusManifest


def _json_bytes(payload: object) -> bytes:
    """Canonical JSON: sorted keys, UTF-8, trailing newline, no NaN, no timestamps."""
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# --------------------------------------------------------------------------------------
# public build entry points
# --------------------------------------------------------------------------------------


def build_demo(
    corpus_dir: str | Path,
    *,
    config: SyntheticConfig | None = None,
    chunker_config: ChunkerConfig | None = None,
    dedup_config: DedupConfig | None = None,
    clean: bool = True,
) -> BuildResult:
    """Generate the synthetic corpus and build it into ``corpus_dir`` (FR-ING-07)."""
    cfg = config or SyntheticConfig()
    documents, queries = generate(cfg)
    root = Path(corpus_dir)
    if clean:
        _reset_corpus(root)

    generated: list[GeneratedDocument] = documents
    for document in generated:
        target = root / document.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document.text, encoding="utf-8", newline="\n")

    result = _build_corpus(
        root,
        chunker_config=chunker_config or DEMO_CHUNKER,
        dedup_config=dedup_config or DedupConfig(num_perm=32, bands=8, max_shingles=256),
        queries=queries,
        generator=SyntheticConfig.model_validate(cfg.model_dump()),
        duplicate_docs=sum(1 for document in generated if document.is_duplicate_of),
        intents={document.doc_id: document.intent for document in generated},
    )
    return result


def build_from_directory(
    corpus_dir: str | Path,
    source_dir: str | Path,
    *,
    license: str | None = None,
    chunker_config: ChunkerConfig | None = None,
    dedup_config: DedupConfig | None = None,
    clean: bool = True,
) -> BuildResult:
    """Copy supported files from ``source_dir`` into ``corpus_dir`` and build (FR-ING-08)."""
    source_root = Path(source_dir)
    if not source_root.is_dir():
        raise SourceNotFoundError(f"no such directory: {source_root}")
    root = Path(corpus_dir)
    if clean:
        _reset_corpus(root)
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        target = root / DOCS_DIR / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied += 1
    if copied == 0:
        raise CorpusBuildError(
            f"no supported files under {source_root} (supported: {sorted(SUPPORTED_FORMATS)})"
        )
    return _build_corpus(
        root,
        chunker_config=chunker_config or DEMO_CHUNKER,
        dedup_config=dedup_config or DedupConfig(),
        queries=[],
        generator=None,
        duplicate_docs=0,
        intents={},
        license_override=license,
    )


def load_manifest(corpus_dir: str | Path) -> CorpusManifest:
    """Read and validate ``manifest.json`` (a corrupt manifest is an error, not a warning)."""
    path = Path(corpus_dir) / MANIFEST_FILE
    if not path.is_file():
        raise CorpusBuildError(f"no manifest at {path}; run `answerdesk ingest --demo` first")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(f"manifest {path} is not valid JSON: {exc}") from exc
    return CorpusManifest.model_validate(payload)


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


def _reset_corpus(root: Path) -> None:
    """Remove previous generated artifacts so no stale document survives a rebuild."""
    root.mkdir(parents=True, exist_ok=True)
    for name in (DOCS_DIR, "queries"):
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
    for name in (CHUNKS_FILE, MANIFEST_FILE):
        target = root / name
        if target.is_file():
            target.unlink()


def _build_corpus(
    root: Path,
    *,
    chunker_config: ChunkerConfig,
    dedup_config: DedupConfig,
    queries: Sequence[GeneratedQuery],
    generator: SyntheticConfig | None,
    duplicate_docs: int,
    intents: dict[str, str],
    license_override: str | None = None,
) -> BuildResult:
    docs_root = root / DOCS_DIR
    if not docs_root.is_dir():
        raise CorpusBuildError(f"no documents directory at {docs_root}")
    doc_paths = sorted(path for path in docs_root.rglob("*") if path.is_file())
    if not doc_paths:
        raise CorpusBuildError(f"no documents under {docs_root}")

    documents = [load_document(path, license=license_override) for path in doc_paths]
    for document, path in zip(documents, doc_paths, strict=True):
        # The manifest is a portable artifact: provenance paths are corpus-relative.
        document.source_path = _relative(path, root)
    documents.sort(key=lambda document: document.doc_id)

    scan = identifier_findings(
        [document.text for document in documents] + [query.text for query in queries]
    )
    if scan:
        raise CorpusBuildError(
            "the generated corpus violates the reserved-identifier policy (§13.4): "
            + "; ".join(scan[:5]),
            detail=f"{len(scan)} findings",
        )

    # t15 gate: a reference reply that its own document does not entail is a corpus
    # defect, not a warning — it would leak into answer-correctness labels (t7/t11).
    grounding = grounding_report(documents, queries)
    grounding_findings: list[Any] = grounding["findings"]
    if grounding_findings:
        first = grounding_findings[0]
        raise CorpusBuildError(
            "the synthetic corpus has reference answers its own documents do not entail: "
            f"{len(grounding_findings)} finding(s), first: {first['claim_kind']} "
            f"{first['claim']!r} of {first['query_id']}",
            detail="run verify_reference_grounding() for the full list",
        )

    detector = NearDuplicateDetector(dedup_config)
    dedup: DedupResult = detector.dedupe(
        [(document.doc_id, document.text) for document in documents]
    )
    kept_ids = set(dedup.kept)
    dropped_ids = {pair.dropped_id: pair.kept_id for pair in dedup.dropped}
    for document in documents:
        if document.doc_id in dropped_ids:
            _remove_document_file(root, document)

    kept_documents = [document for document in documents if document.doc_id in kept_ids]
    chunker = StructureChunker(chunker_config)
    chunks: list[Chunk] = []
    for document in kept_documents:
        chunks.extend(chunker.chunk(document))

    chunks, chunk_dedup_removed = _dedupe_chunks(chunks)
    chunks.sort(key=lambda chunk: chunk.chunk_id)

    chunks_path = root / CHUNKS_FILE
    chunks_path.write_bytes(
        "\n".join(
            json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for chunk in chunks
        ).encode("utf-8")
        + b"\n"
    )
    queries_path = _write_queries(root, queries)

    manifest = _compose_manifest(
        root=root,
        documents=kept_documents,
        chunks=chunks,
        dedup=dedup,
        chunk_dedup_removed=chunk_dedup_removed,
        chunker_config=chunker_config,
        queries=queries,
        generator=generator,
        duplicate_docs=duplicate_docs,
        intents=intents,
        chunks_path=chunks_path,
        queries_path=queries_path,
        scan_findings=scan,
        grounding=grounding,
    )
    manifest_path = root / MANIFEST_FILE
    manifest_path.write_bytes(_json_bytes(manifest.model_dump(mode="json")))
    return BuildResult(
        corpus_dir=root.as_posix(),
        manifest_path=manifest_path.as_posix(),
        chunks_path=chunks_path.as_posix(),
        queries_path=None if queries_path is None else queries_path.as_posix(),
        manifest=manifest,
    )


def _remove_document_file(root: Path, document: Document) -> None:
    path = root / document.source_path
    if path.is_file():
        path.unlink()
    parent = path.parent
    docs_root = root / DOCS_DIR
    while parent != docs_root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def _dedupe_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Exact-text duplicate removal across documents; overlap chunks inside a doc stay."""
    seen: dict[str, str] = {}
    kept: list[Chunk] = []
    removed = 0
    for chunk in chunks:
        digest = text_hash(chunk.text)
        owner = seen.get(digest)
        if owner is not None and owner != chunk.doc_id:
            removed += 1
            continue
        seen.setdefault(digest, chunk.doc_id)
        kept.append(chunk)
    return kept, removed


def _write_queries(root: Path, queries: Sequence[GeneratedQuery]) -> Path | None:
    if not queries:
        return None
    path = root / QUERIES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "query_id": query.query_id,
            "locale": query.locale,
            "intent": query.intent,
            "doc_id": query.doc_id,
            "category": query.category,
            "archetype": query.archetype,
            "style": query.style,
            "answerable": query.answerable,
            "supporting_doc_ids": list(query.supporting_doc_ids),
            "perturbations": list(query.perturbations),
            "text": query.text,
            "base_text": query.base_text,
            "operator_reply": query.operator_reply,
        }
        for query in queries
    ]
    path.write_bytes(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in payloads).encode(
            "utf-8"
        )
        + b"\n"
    )
    return path


# --------------------------------------------------------------------------------------
# manifest composition
# --------------------------------------------------------------------------------------


#: Provenance of the locale content banks (t18). Runtime generation never calls a model.
CONTENT_PROVENANCE: dict[str, Any] = {
    "authored_by": "llm-assisted: hand-authored in-repo content banks (no runtime calls)",
    "model": "DeepSeek Harness agent 'data-engineer' (drafting assistant)",
    "authored_on": "2026-10-21",
    "method": "answerdesk.ingest.synthetic_data banks_<locale>.py + extra_<locale>.py",
    "task_contract": "t18 (docs/ingestion.md §12)",
    "native_speaker_validated": False,
    "license": "CC0-1.0",
}


def _count_styles(queries: Sequence[GeneratedQuery]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        if query.style:
            counts[query.style] = counts.get(query.style, 0) + 1
    return dict(sorted(counts.items()))


def _count_categories(queries: Sequence[GeneratedQuery]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        counts[query.category] = counts.get(query.category, 0) + 1
    return dict(sorted(counts.items()))


def _share(queries: Sequence[GeneratedQuery], predicate: Any) -> float:
    if not queries:
        return 0.0
    return round(sum(1 for query in queries if predicate(query)) / len(queries), 4)


def _front_matter_value(document: Document, key: str) -> str | None:
    front_matter = document.metadata.get("front_matter", {})
    if not isinstance(front_matter, dict):
        return None
    value = front_matter.get(key)
    return str(value) if isinstance(value, str | int | float) else None


def _locale_stats(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    queries: Sequence[GeneratedQuery],
) -> dict[str, Any]:
    """Per-locale depth: documents, archetypes, chunks, tokens and query mix (t18)."""
    stats: dict[str, Any] = {}
    locales = sorted({document.language or "und" for document in documents})
    for locale in locales:
        locale_docs = [document for document in documents if (document.language or "und") == locale]
        locale_chunks = [chunk for chunk in chunks if (chunk.language or "und") == locale]
        locale_doc_ids = {document.doc_id for document in locale_docs}
        locale_queries = [query for query in queries if query.locale == locale]
        archetypes: dict[str, int] = {}
        for document in locale_docs:
            archetype = _front_matter_value(document, "archetype") or "unknown"
            archetypes[archetype] = archetypes.get(archetype, 0) + 1
        stats[locale] = {
            "documents": len(locale_docs),
            "chunks": len(locale_chunks),
            "tokens": sum(chunk.token_count for chunk in locale_chunks),
            "archetypes": len(archetypes),
            "documents_by_archetype": dict(sorted(archetypes.items())),
            "queries": len(locale_queries),
            "queries_by_category": _count_categories(locale_queries),
            "unanswerable_share": _share(locale_queries, lambda q: not q.answerable),
            "multi_hop_queries": sum(1 for q in locale_queries if len(q.supporting_doc_ids) > 1),
            "query_documents_known": sum(1 for q in locale_queries if q.doc_id in locale_doc_ids),
        }
    return stats


def _compose_manifest(
    *,
    root: Path,
    documents: list[Document],
    chunks: list[Chunk],
    dedup: DedupResult,
    chunk_dedup_removed: int,
    chunker_config: ChunkerConfig,
    queries: Sequence[GeneratedQuery],
    generator: SyntheticConfig | None,
    duplicate_docs: int,
    intents: dict[str, str],
    chunks_path: Path,
    queries_path: Path | None,
    scan_findings: list[str],
    grounding: dict[str, object],
) -> CorpusManifest:
    chunks_per_doc: dict[str, int] = {}
    for chunk in chunks:
        chunks_per_doc[chunk.doc_id] = chunks_per_doc.get(chunk.doc_id, 0) + 1

    sources: list[SourceRecord] = []
    for document in documents:
        path = root / document.source_path
        sources.append(
            SourceRecord(
                doc_id=document.doc_id,
                path=document.source_path,
                sha256=_sha256_file(path),
                byte_size=path.stat().st_size,
                license=document.license,
                language=document.language,
                source_format=document.source_format,
                title=document.title,
                intent=intents.get(document.doc_id),
                archetype=_front_matter_value(document, "archetype"),
                heading_count=len(document.headings),
                block_count=len(document.metadata.get("blocks", [])),
                chunks=chunks_per_doc.get(document.doc_id, 0),
            )
        )
    sources.sort(key=lambda record: record.doc_id)

    blocks_by_kind: dict[str, int] = {}
    tables = 0
    faq_pairs = 0
    for document in documents:
        blocks = document.metadata.get("blocks", [])
        for block in blocks:
            kind = str(block.get("kind", "unknown"))
            blocks_by_kind[kind] = blocks_by_kind.get(kind, 0) + 1
            if kind == "table":
                tables += 1
            if kind == "faq_question":
                faq_pairs += 1

    stats = ManifestStats(
        documents=len(documents),
        chunks=len(chunks),
        tokens=sum(chunk.token_count for chunk in chunks),
        tokens_mean=(
            round(sum(chunk.token_count for chunk in chunks) / len(chunks), 3) if chunks else 0.0
        ),
        tokens_max=max((chunk.token_count for chunk in chunks), default=0),
        chunks_with_heading_path=sum(1 for chunk in chunks if chunk.heading_path),
        chunks_with_canonical_tokens=sum(1 for chunk in chunks if chunk.canonical_tokens),
        chunks_with_breadcrumb=sum(1 for chunk in chunks if chunk.breadcrumb),
        docs_per_locale=_count_by(document.language or "und" for document in documents),
        chunks_per_locale=_count_by(chunk.language or "und" for chunk in chunks),
        blocks_by_kind=dict(sorted(blocks_by_kind.items())),
        tables=tables,
        faq_pairs=faq_pairs,
        duplicate_documents_removed=dedup.removed_count,
        duplicate_chunks_removed=chunk_dedup_removed,
    )

    generator_record = (
        _generator_record(generator, documents, queries, duplicate_docs) if generator else None
    )
    corpus_version = _corpus_version(sources, chunker_config)
    artifacts = {CHUNKS_FILE: _sha256_file(chunks_path)}
    if queries_path is not None:
        artifacts[QUERIES_FILE] = _sha256_file(queries_path)
    return CorpusManifest(
        corpus_version=corpus_version,
        license=_dominant_license(sources),
        generator=generator_record,
        sources=sources,
        stats=stats,
        dedup=DocumentDedupRecord(
            threshold=dedup.threshold,
            num_perm=dedup.num_perm,
            shingle_size=dedup.shingle_size,
            seed=dedup.seed,
            bands=dedup.bands,
            candidates=len(sources) + len(dedup.dropped),
            kept=len(dedup.kept),
            removed=dedup.removed_count,
            pairs=[pair.model_dump(mode="json") for pair in dedup.dropped],
        ),
        chunk_dedup=ChunkDedupRecord(removed=chunk_dedup_removed),
        chunking={
            "chunker": StructureChunker.name,
            "token_budget": chunker_config.token_budget,
            "overlap_tokens": chunker_config.overlap_tokens,
            "context_header": chunker_config.context_header,
            "repeat_table_header": chunker_config.repeat_table_header,
        },
        queries={
            "file": QUERIES_FILE if queries_path is not None else None,
            "count": len(queries),
            "locales": sorted({query.locale for query in queries}),
            "categories": _count_categories(queries),
            "unanswerable_share": _share(queries, lambda q: not q.answerable),
            "noisy_styles": _count_styles(queries),
            "reference_grounding": "verified" if not grounding["findings"] else "failed",
        },
        archetypes=sorted({query.archetype for query in queries if query.archetype}),
        locales=_locale_stats(documents, chunks, queries),
        content_provenance=CONTENT_PROVENANCE,
        identifier_scan={"checked": len(documents) + len(queries), "findings": scan_findings},
        reference_grounding=grounding,
        artifacts=artifacts,
    )


def _generator_record(
    generator: SyntheticConfig,
    documents: list[Document],
    queries: Sequence[GeneratedQuery],
    duplicate_docs: int,
) -> GeneratorRecord:
    counts: dict[str, int] = {}
    for query in queries:
        for klass in query.perturbations or ("direct",):
            counts[klass] = counts.get(klass, 0) + 1
    shares = (
        {klass: round(count / len(queries), 4) for klass, count in sorted(counts.items())}
        if queries
        else {}
    )
    return GeneratorRecord(
        name=GENERATOR_NAME,
        version=GENERATOR_VERSION,
        seed=generator.seed,
        locales=list(generator.locales),
        docs_per_locale=generator.docs_per_locale,
        duplicate_every=generator.duplicate_every,
        documents=len(documents),
        duplicates=duplicate_docs,
        queries=len(queries),
        perturbation_counts=dict(sorted(counts.items())),
        perturbation_shares=shares,
    )


def _corpus_version(sources: Sequence[SourceRecord], chunker_config: ChunkerConfig) -> str:
    """Hash of the source set and the chunking config — a monotonic corpus identity."""
    payload = {
        "sources": [[record.path, record.sha256] for record in sources],
        "chunking": {
            "token_budget": chunker_config.token_budget,
            "overlap_tokens": chunker_config.overlap_tokens,
            "context_header": chunker_config.context_header,
        },
        "tokenizer": TOKENIZER_VERSION,
    }
    digest = _sha256_bytes(_json_bytes(payload))
    return f"corpus-{digest[:16]}"


def _dominant_license(sources: Sequence[SourceRecord]) -> str:
    licenses = sorted({record.license for record in sources})
    return licenses[0] if len(licenses) == 1 else "mixed"


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "CHUNKS_FILE",
    "DEMO_CHUNKER",
    "DOCS_DIR",
    "MANIFEST_FILE",
    "MANIFEST_SCHEMA_VERSION",
    "QUERIES_FILE",
    "BuildResult",
    "CorpusManifest",
    "GeneratorRecord",
    "ManifestStats",
    "SourceRecord",
    "build_demo",
    "build_from_directory",
    "load_manifest",
]
