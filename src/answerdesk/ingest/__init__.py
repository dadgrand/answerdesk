"""Ingestion: loaders, structure-aware chunking, dedup, synthetic corpus, adapters.

Owning task: **t4**. Public surface:

* :func:`answerdesk.ingest.loaders.load_document` — PDF/DOCX/HTML/MD/TXT -> ``Document``;
* :class:`answerdesk.ingest.chunker.StructureChunker` — structure-first chunking;
* :class:`answerdesk.ingest.dedup.NearDuplicateDetector` — deterministic MinHash dedup;
* :func:`answerdesk.ingest.synthetic.generate` — seeded multilingual corpus generator;
* :func:`answerdesk.ingest.builder.build_demo` — idempotent corpus build + manifest;
* :class:`answerdesk.ingest.datasets.PublicDatasetAdapter` — license-gated adapters.

Requirements: FR-ING-01 … FR-ING-10 (docs/requirements.md §4.1), §13.2/§13.4 for
licensing and reserved identifiers. Every type that crosses a module boundary comes from
:mod:`answerdesk.contracts`; the block model in :mod:`answerdesk.ingest.structure` is
internal to this package and is serialised into ``Document.metadata``.
"""

from __future__ import annotations

from answerdesk.ingest.builder import (
    BuildResult,
    CorpusManifest,
    ManifestStats,
    build_demo,
    build_from_directory,
    load_manifest,
)
from answerdesk.ingest.canonical import canonicalize
from answerdesk.ingest.chunker import ChunkerConfig, StructureChunker
from answerdesk.ingest.dedup import DedupConfig, DedupResult, NearDuplicateDetector
from answerdesk.ingest.errors import (
    CorpusBuildError,
    EmptyDocumentError,
    IngestError,
    LicenseError,
    SourceNotFoundError,
    UnsupportedFormatError,
)
from answerdesk.ingest.loaders import (
    SUPPORTED_FORMATS,
    load_directory,
    load_document,
    load_markdown_text,
)
from answerdesk.ingest.synthetic import (
    DEMO_SEED,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    INTENTS,
    LOCALES,
    SyntheticConfig,
    generate,
    identifier_findings,
)
from answerdesk.ingest.tokenizer import TOKENIZER_VERSION, estimate_tokens

__all__ = [
    "DEMO_SEED",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "INTENTS",
    "LOCALES",
    "SUPPORTED_FORMATS",
    "TOKENIZER_VERSION",
    "BuildResult",
    "ChunkerConfig",
    "CorpusBuildError",
    "CorpusManifest",
    "DedupConfig",
    "DedupResult",
    "EmptyDocumentError",
    "IngestError",
    "LicenseError",
    "ManifestStats",
    "NearDuplicateDetector",
    "SourceNotFoundError",
    "StructureChunker",
    "SyntheticConfig",
    "UnsupportedFormatError",
    "build_demo",
    "build_from_directory",
    "canonicalize",
    "estimate_tokens",
    "generate",
    "identifier_findings",
    "load_directory",
    "load_document",
    "load_manifest",
    "load_markdown_text",
]
