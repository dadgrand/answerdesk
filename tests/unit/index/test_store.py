"""The index directory: fingerprints, idempotency, chunk-set provenance.

# AT-OPS-02  # AT-RET-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.config import load_settings
from answerdesk.contracts import Chunk
from answerdesk.errors import CorpusMissingError, IndexMissingError
from answerdesk.index.bm25_index import BM25Params
from answerdesk.index.store import (
    ChunkStore,
    EmbedderInfo,
    IndexBundle,
    build_indexes,
    compute_index_version,
    index_dir_for,
    load_indexes,
)
from answerdesk.testing.doubles import HashingEmbedder


def _settings(tmp_path: Path, **app: str):
    overrides = {
        "app": {
            "corpus_dir": str(tmp_path / "corpus"),
            "index_dir": str(tmp_path / "index"),
            **app,
        }
    }
    return load_settings(profile="offline", overrides=overrides)


def test_chunk_store_indexes_the_breadcrumb_but_never_edits_the_text(
    fixture_store: ChunkStore,
) -> None:
    """FR-ING-04 / INV-01: the indexed representation adds context, the chunk stays verbatim."""
    chunk = fixture_store.chunks[0]
    rendered = ChunkStore.text_for(chunk)
    assert chunk.breadcrumb in rendered
    assert chunk.text in rendered
    assert chunk.text == fixture_chunks_text(fixture_store, chunk.chunk_id)


def fixture_chunks_text(store: ChunkStore, chunk_id: str) -> str:
    return next(chunk.text for chunk in store.chunks if chunk.chunk_id == chunk_id)


def test_duplicate_chunk_ids_are_a_corpus_error(make_chunk) -> None:
    chunk = make_chunk()
    with pytest.raises(CorpusMissingError, match="duplicate"):
        ChunkStore.from_chunks([chunk, chunk])


def test_missing_corpus_names_the_command_that_builds_it(tmp_path: Path) -> None:
    with pytest.raises(CorpusMissingError, match="answerdesk ingest"):
        ChunkStore.from_corpus(tmp_path / "nothing")


def test_fingerprint_changes_when_any_chunk_text_changes(make_chunk) -> None:
    first = ChunkStore.from_chunks([make_chunk(chunk_id="a::0", text="policy v1")])
    second = ChunkStore.from_chunks([make_chunk(chunk_id="a::0", text="policy v2")])
    assert first.fingerprint() != second.fingerprint()


def test_index_version_is_a_function_of_the_fingerprint_inputs(fixture_store: ChunkStore) -> None:
    """FR-OPS-02: corpus, chunk hash, embedder, tokeniser and BM25 params decide the version."""
    base = dict(
        corpus_version=fixture_store.corpus_version,
        chunks_sha256=fixture_store.fingerprint(),
        embedder=EmbedderInfo(name="hashing-32", dimension=32),
        bm25_params=BM25Params(),
    )
    reference = compute_index_version(**base)
    assert reference == compute_index_version(**base)
    assert reference != compute_index_version(
        **{**base, "embedder": EmbedderInfo(name="hashing-64", dimension=64)}
    )
    assert reference != compute_index_version(**{**base, "corpus_version": "corpus-other"})
    assert reference != compute_index_version(**{**base, "bm25_params": BM25Params(k1=1.2)})
    assert reference != compute_index_version(**{**base, "chunks_sha256": "0" * 64})
    assert reference.startswith("index-") and len(reference) == len("index-") + 16


def test_bundle_roundtrip_preserves_rankings(tmp_path: Path, fixture_store: ChunkStore) -> None:
    embedder = HashingEmbedder(dimension=32)
    bundle = IndexBundle.build(fixture_store, embedder)
    bundle.save(tmp_path / "index")
    loaded = IndexBundle.load(tmp_path / "index", embedder)
    query = "кайра чыгаруу"
    assert [p.chunk.chunk_id for p in loaded.bm25.search(query, 3)] == [
        p.chunk.chunk_id for p in bundle.bm25.search(query, 3)
    ]
    assert [p.chunk.chunk_id for p in loaded.dense.search(query, 3)] == [
        p.chunk.chunk_id for p in bundle.dense.search(query, 3)
    ]
    assert loaded.manifest.index_version == bundle.manifest.index_version


def test_loading_with_another_embedder_is_refused(
    tmp_path: Path, fixture_store: ChunkStore
) -> None:
    bundle = IndexBundle.build(fixture_store, HashingEmbedder(dimension=32))
    bundle.save(tmp_path / "index")
    with pytest.raises(IndexMissingError, match="rebuild"):
        IndexBundle.load(tmp_path / "index", HashingEmbedder(dimension=64))


def test_loading_a_missing_manifest_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(IndexMissingError, match="answerdesk index"):
        IndexBundle.load(tmp_path / "nope", HashingEmbedder(dimension=8))


def test_index_command_builds_then_reuses_for_an_unchanged_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AT-OPS-02: a second run reports ``reused`` and the same index version."""
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")
    settings = _settings(tmp_path)
    first = build_indexes(settings, demo=True, echo=lambda _message: None)
    assert first.reused is False
    assert first.chunk_count > 0
    stored = json.loads((Path(first.index_dir) / "index.json").read_text(encoding="utf-8"))
    assert stored["index_version"] == first.index_version
    assert stored["chunk_count"] == first.chunk_count
    assert Path(first.index_dir) == index_dir_for(settings, first.corpus_version)

    second = build_indexes(settings, demo=True, echo=lambda _message: None)
    assert second.reused is True
    assert second.index_version == first.index_version
    assert second.chunk_count == first.chunk_count


def test_force_rebuilds_even_when_the_fingerprint_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")
    settings = _settings(tmp_path)
    build_indexes(settings, demo=True, echo=lambda _message: None)
    forced = build_indexes(settings, demo=True, force=True, echo=lambda _message: None)
    assert forced.reused is False


def test_load_indexes_finds_the_current_corpus_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWERDESK_INGEST_DOCS_PER_LOCALE", "1")
    settings = _settings(tmp_path)
    built = build_indexes(settings, demo=True, echo=lambda _message: None)
    loaded = load_indexes(settings)
    assert loaded.manifest.index_version == built.index_version
    assert len(loaded.store) == built.chunk_count


def test_index_without_a_corpus_and_without_demo_fails_loudly(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(CorpusMissingError, match="answerdesk ingest"):
        build_indexes(settings, demo=False, echo=lambda _message: None)


def test_bundle_stats_expose_the_measured_shape(fixture_store: ChunkStore) -> None:
    bundle = IndexBundle.build(fixture_store, HashingEmbedder(dimension=32))
    stats = bundle.stats()
    assert stats["chunks"] == len(fixture_store)
    assert stats["dense"]["dimension"] == 32
    assert stats["bm25"]["chunks"] == len(fixture_store)
    assert bundle.manifest.timings_ms["bm25_build_ms"] >= 0.0


def test_from_documents_uses_the_production_chunker(make_document) -> None:
    """The store can be fed documents directly, which is how tests avoid the corpus."""
    store = ChunkStore.from_documents([make_document(text="# Refunds\n\nRefunds take 14 days.")])
    assert store.chunks
    assert all(isinstance(chunk, Chunk) for chunk in store.chunks)
    assert store.corpus_version == "corpus-test"
