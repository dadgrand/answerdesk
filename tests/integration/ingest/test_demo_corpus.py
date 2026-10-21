"""End-to-end corpus build: floors, invariants, determinism, idempotency.

# AT-ING-07  # AT-ING-08  # AT-ING-10
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.ingest.builder import (
    CHUNKS_FILE,
    DEMO_CHUNKER,
    MANIFEST_FILE,
    QUERIES_FILE,
    build_demo,
    build_from_directory,
    load_manifest,
    normalise_line_endings,
)
from answerdesk.ingest.synthetic import ARCHETYPES, LOCALES, QUERY_CATEGORIES

REPO_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus"


@pytest.fixture(scope="module")
def demo_build(tmp_path_factory: pytest.TempPathFactory):
    """One full demo build shared by the assertions (the corpus is ~5 s of work)."""
    corpus_dir = tmp_path_factory.mktemp("corpus")
    return build_demo(corpus_dir)


def test_demo_corpus_meets_the_floor(demo_build) -> None:
    """AT-ING-07: the demo corpus is >= 60 documents and >= 1,500 chunks."""
    stats = demo_build.manifest.stats
    assert stats.documents >= 810, stats
    assert stats.chunks >= 7000, stats
    assert set(stats.docs_per_locale) == set(LOCALES)
    assert set(demo_build.manifest.generator.locales) == set(LOCALES)


def test_manifest_records_provenance_for_every_source(demo_build) -> None:
    """AT-ING-08: each source carries path, sha256, byte size and a non-empty license."""
    manifest = demo_build.manifest
    assert manifest.sources
    for source in manifest.sources:
        assert source.path.startswith("docs/")
        assert source.path.endswith(".md")
        assert len(source.sha256) == 64
        assert source.byte_size > 0
        assert source.license == "CC0-1.0"
        assert source.language in LOCALES
        assert source.chunks >= 1
    assert manifest.license == "CC0-1.0"
    assert manifest.identifier_scan["findings"] == []


def test_manifest_artifacts_are_hashed(demo_build) -> None:
    """AT-ING-08: derived artifacts are recorded by hash, so staleness is detectable."""
    manifest = demo_build.manifest
    assert CHUNKS_FILE in manifest.artifacts
    assert QUERIES_FILE in manifest.artifacts
    for relative, digest in manifest.artifacts.items():
        assert len(digest) == 64, relative


def test_every_chunk_respects_the_chunking_contract(demo_build) -> None:
    """AT-ING-03/04: budget, heading path, canonical tokens and spans hold corpus-wide."""
    chunks_path = Path(demo_build.chunks_path)
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) >= 7000
    ids = set()
    for chunk in chunks:
        assert chunk["token_count"] <= DEMO_CHUNKER.token_budget
        assert chunk["text"].strip()
        assert chunk["chunk_id"] not in ids
        ids.add(chunk["chunk_id"])
        assert chunk["doc_id"]
        assert chunk["heading_path"], chunk["chunk_id"]
        assert chunk["breadcrumb"] == " > ".join(chunk["heading_path"])
        assert chunk["end_char"] > chunk["start_char"]
        assert chunk["metadata"]["license"] == "CC0-1.0"
        assert chunk["metadata"]["chunker"] == "structure-first-v1"
    assert any(chunk["canonical_tokens"] for chunk in chunks)


def test_near_duplicates_were_removed(demo_build) -> None:
    """AT-ING-06/07: the injected near-copies are gone and recorded with their evidence."""
    manifest = demo_build.manifest
    assert manifest.dedup.removed >= 1
    assert manifest.stats.duplicate_documents_removed == manifest.dedup.removed
    for pair in manifest.dedup.pairs:
        assert pair["similarity"] >= manifest.dedup.threshold
        assert pair["dropped_id"].endswith("-dup")
    kept_ids = {source.doc_id for source in manifest.sources}
    assert not any(doc_id.endswith("-dup") for doc_id in kept_ids)


def test_queries_cover_intents_and_perturbations(demo_build) -> None:
    """AT-ING-10: 2,400+ queries across all categories, locales and archetypes."""
    manifest = demo_build.manifest
    queries_path = Path(demo_build.queries_path or "")
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
    assert len(queries) == manifest.queries["count"]
    assert {query["locale"] for query in queries} == set(LOCALES)
    assert {query["archetype"] for query in queries} == set(ARCHETYPES)
    assert {query["category"] for query in queries} == set(QUERY_CATEGORIES)
    assert len(queries) >= 3000, len(queries)
    categories = manifest.queries["categories"]
    assert all(categories.get(category, 0) > 0 for category in QUERY_CATEGORIES)
    unanswerable = manifest.queries["unanswerable_share"]
    assert 0.15 <= unanswerable <= 0.21, unanswerable
    known_docs = {source.doc_id for source in manifest.sources}
    assert all(query["doc_id"] in known_docs for query in queries)


def test_rebuild_is_byte_identical(demo_build, tmp_path: Path) -> None:
    """AT-ING-08: same seed -> byte-identical manifest and chunk file (no timestamps)."""
    second = build_demo(tmp_path / "second")
    first_manifest = Path(demo_build.manifest_path).read_bytes()
    second_manifest = Path(second.manifest_path).read_bytes()
    assert first_manifest == second_manifest
    assert demo_build.manifest.corpus_version == second.manifest.corpus_version
    first_chunks = Path(demo_build.chunks_path).read_bytes()
    second_chunks = Path(second.chunks_path).read_bytes()
    assert first_chunks == second_chunks


def test_rebuild_in_place_is_idempotent(demo_build, tmp_path: Path) -> None:
    """AT-ING-08: re-ingesting unchanged sources leaves the corpus byte-identical."""
    target = tmp_path / "inplace"
    build_demo(target)
    first = (target / MANIFEST_FILE).read_bytes()
    build_demo(target)
    second = (target / MANIFEST_FILE).read_bytes()
    assert first == second
    assert load_manifest(target).corpus_version == demo_build.manifest.corpus_version


def test_checked_in_corpus_matches_a_fresh_build(demo_build) -> None:
    """AT-ING-08: the committed manifest is the artifact of the generator, not hand-edited.

    The comparison is line-ending normalised (t38): a Windows checkout with
    ``core.autocrlf=true`` must not fail the build for CRLF alone, while a genuine content
    difference still fails because every other byte must match.
    """
    committed = REPO_CORPUS / MANIFEST_FILE
    if not committed.is_file():  # pragma: no cover - the release tree always has it
        pytest.skip("checked-in corpus not built in this tree")
    fresh = Path(demo_build.manifest_path).read_bytes()
    assert normalise_line_endings(committed.read_bytes()) == normalise_line_endings(fresh)
    assert b"\r\n" not in fresh, "the generator must write LF-only bytes"
    committed_chunks = REPO_CORPUS / CHUNKS_FILE
    if committed_chunks.is_file():
        fresh_chunks = Path(demo_build.chunks_path).read_bytes()
        assert normalise_line_endings(committed_chunks.read_bytes()) == normalise_line_endings(
            fresh_chunks
        )
    committed_queries = REPO_CORPUS / QUERIES_FILE
    if committed_queries.is_file():
        fresh_queries = Path(demo_build.queries_path).read_bytes()
        assert normalise_line_endings(committed_queries.read_bytes()) == normalise_line_endings(
            fresh_queries
        )


def test_build_from_directory_records_provenance(tmp_path: Path) -> None:
    """AT-ING-08: the file path (not only the generator) produces a complete manifest."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.md").write_text(
        '---\ntitle: "Policy"\nlanguage: "ru"\nlicense: "CC0-1.0"\n---\n\n'
        "# Политика\n\nВозврат за 14 дней.\n\n## Комиссии\n\nКомиссия 2,5 EUR.\n",
        encoding="utf-8",
    )
    (source / "notes.html").write_text(
        "<html><body><h1>Notes</h1><p>Daily limit 25 000 RUB.</p>"
        "<table><tr><th>Fee</th><th>Value</th></tr>"
        "<tr><td>transfer</td><td>2,5 EUR</td></tr></table></body></html>",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    result = build_from_directory(corpus, source, license="CC0-1.0")
    manifest = result.manifest
    assert manifest.stats.documents == 2
    assert manifest.stats.chunks >= 2
    assert {source.source_format for source in manifest.sources} == {"md", "html"}
    assert all(source.license == "CC0-1.0" for source in manifest.sources)
    assert manifest.generator is None
    assert manifest.queries["count"] == 0
    again = build_from_directory(corpus, source, license="CC0-1.0")
    assert Path(again.manifest_path).read_bytes() == Path(result.manifest_path).read_bytes()


def test_corpus_version_changes_with_the_chunking_profile(tmp_path: Path) -> None:
    """AT-ING-08 / FR-CAC-05: a different chunking profile is a different corpus."""
    from answerdesk.ingest.chunker import ChunkerConfig
    from answerdesk.ingest.synthetic import SyntheticConfig

    small = SyntheticConfig(docs_per_locale=12)
    default = build_demo(tmp_path / "a", config=small)
    other = build_demo(
        tmp_path / "b",
        config=small,
        chunker_config=ChunkerConfig(token_budget=256, overlap_tokens=32),
    )
    assert default.manifest.corpus_version != other.manifest.corpus_version
    assert other.manifest.chunking["token_budget"] == 256
