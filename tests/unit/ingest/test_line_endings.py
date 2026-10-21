"""Line-ending determinism of the corpus writers (t38).

A Windows checkout (`core.autocrlf=true`), a Linux checkout and a fresh in-process build
must all produce the same bytes. The corpus writers therefore normalise CRLF/CR to LF, and
the artifact comparison in the integration suite compares the same canonical form.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.ingest.builder import (
    CHUNKS_FILE,
    MANIFEST_FILE,
    QUERIES_FILE,
    build_demo,
    normalise_line_endings,
    write_canonical_bytes,
    write_canonical_text,
)
from answerdesk.ingest.synthetic import SyntheticConfig

REPO_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus"


def test_normalise_line_endings_folds_crlf_and_cr() -> None:
    """AT-ING-08: CRLF and lone CR both become LF; LF input is returned unchanged."""
    assert normalise_line_endings(b"a\r\nb\rc\n") == b"a\nb\nc\n"
    assert normalise_line_endings(b"no newlines") == b"no newlines"
    assert normalise_line_endings(b"already\nlf\n") == b"already\nlf\n"


def test_canonical_writers_emit_no_crlf(tmp_path: Path) -> None:
    """AT-ING-08: the canonical writers never write a CRLF, even for CRLF input."""
    text_path = tmp_path / "nested" / "doc.md"
    write_canonical_text(text_path, "# Title\r\n\r\nBody line\r\n")
    assert text_path.read_bytes() == b"# Title\n\nBody line\n"
    assert b"\r" not in text_path.read_bytes()

    bytes_path = tmp_path / "nested" / "chunks.jsonl"
    write_canonical_bytes(bytes_path, b'{"a": 1}\r\n{"b": 2}\r\n')
    assert bytes_path.read_bytes() == b'{"a": 1}\n{"b": 2}\n'
    assert b"\r" not in bytes_path.read_bytes()


def test_a_fresh_build_writes_lf_only(tmp_path: Path) -> None:
    """AT-ING-08: every artifact of a build is LF-only (docs, chunks, queries, manifest)."""
    result = build_demo(tmp_path / "corpus", config=SyntheticConfig(docs_per_locale=2))
    artifacts = [
        Path(result.manifest_path),
        Path(result.chunks_path),
        Path(result.queries_path),
        *sorted((tmp_path / "corpus" / "docs").rglob("*.md")),
    ]
    assert artifacts
    for path in artifacts:
        payload = path.read_bytes()
        assert b"\r\n" not in payload, path
        assert b"\r" not in payload, path
        assert payload.endswith(b"\n"), path


def test_committed_corpus_is_lf_only() -> None:
    """AT-ING-08: the checked-in artifacts of the release tree are LF-only as well."""
    if not (REPO_CORPUS / MANIFEST_FILE).is_file():  # pragma: no cover - release tree has it
        pytest.skip("checked-in corpus not built in this tree")
    # chunks.jsonl is derived and git-ignored: a fresh checkout only has the manifest,
    # the docs and (optionally) the queries. Test what is actually committed.
    targets = [
        path for path in (REPO_CORPUS / MANIFEST_FILE, REPO_CORPUS / CHUNKS_FILE) if path.is_file()
    ]
    queries = REPO_CORPUS / QUERIES_FILE
    if queries.is_file():
        targets.append(queries)
    docs = sorted((REPO_CORPUS / "docs").rglob("*.md"))
    targets.extend(docs)
    for path in targets:
        payload = path.read_bytes()
        assert b"\r\n" not in payload, f"{path}: CRLF in a committed corpus artifact"
        assert payload.endswith(b"\n"), f"{path}: artifact must end with a single LF"


def test_manifest_hash_is_line_ending_independent() -> None:
    """AT-ING-08: the manifest hash does not depend on the checkout's line endings."""
    import hashlib

    payload = (REPO_CORPUS / MANIFEST_FILE).read_bytes()
    canonical = normalise_line_endings(payload.replace(b"\n", b"\r\n"))
    assert hashlib.sha256(canonical).hexdigest() == hashlib.sha256(payload).hexdigest()
    # the corpus_version is a content hash, so a CRLF copy still agrees with itself
    manifest = json.loads(payload.decode("utf-8"))
    assert manifest["corpus_version"].startswith("corpus-")
