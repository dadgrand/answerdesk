"""Provenance hashing must be line-ending independent (R-03).

The published digests in ``docs/results.md`` are the reproducibility claim: they must mean the
same file content on a Windows working tree (CRLF) and on a clean LF checkout. These tests pin
that property at three levels — the helper itself, the script that publishes the block, and the
committed document.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "docs" / "results.md"
FROZEN_CORPUS_MANIFEST_SHA256 = "da33968a0d0087ecc0c4e8d966c1cdc47d40699dec139a00694818c0e583af69"


def _raw_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script():
    path = REPO_ROOT / "scripts" / "build-results.py"
    spec = importlib.util.spec_from_file_location("build_results_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crlf_and_lf_twins_share_one_digest(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    script = _load_script()
    lf.write_bytes(b"alpha\nbeta\ngamma\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    assert script._canonical_sha256(lf) == script._canonical_sha256(crlf)
    # the raw digests must differ, otherwise the test proves nothing about the fix
    assert _raw_sha256(lf) != _raw_sha256(crlf)
    # lone CR (old-Mac) normalises too
    assert script._canonical_bytes(b"alpha\rbeta\rgamma\r") == script._canonical_bytes(
        lf.read_bytes()
    )


def test_canonicalisation_keeps_content_sensitivity(tmp_path: Path) -> None:
    script = _load_script()
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_bytes(b"value = 400\n")
    two.write_bytes(b"value = 900\n")
    assert script._canonical_sha256(one) != script._canonical_sha256(two)


def test_canonical_bytes_is_idempotent() -> None:
    script = _load_script()
    once = script._canonical_bytes(b"a\r\nb\rc\n")
    assert once == b"a\nb\nc\n"
    assert script._canonical_bytes(once) == once


def test_missing_file_reports_missing(tmp_path: Path) -> None:
    assert _load_script()._canonical_sha256(tmp_path / "absent") == "missing"


def test_script_uses_the_shared_canonical_helper(tmp_path: Path) -> None:
    script = _load_script()
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert script._canonical_sha256(lf) == script._canonical_sha256(crlf)
    assert script._canonical_sha256(lf) == _raw_sha256(lf)  # LF bytes are already canonical


def test_published_block_matches_canonical_digests() -> None:
    """docs/results.md publishes exactly the canonical digests of the committed inputs."""
    if not RESULTS.is_file():
        pytest.skip("docs/results.md is not present in this checkout")
    text = RESULTS.read_text(encoding="utf-8")
    block = text[text.index("## Provenance") :]
    published = dict(re.findall(r"^(\w+)\s+(\S+)$", block, flags=re.MULTILINE))
    inputs = {
        "corpus_manifest": REPO_ROOT / "data" / "corpus" / "manifest.json",
        "golden_manifest": REPO_ROOT / "evals" / "golden" / "v1" / "manifest.json",
        "split_manifest": REPO_ROOT / "evals" / "splits" / "v1.json",
        "pricing_table": REPO_ROOT / "configs" / "pricing.yaml",
        "summary_json": REPO_ROOT / "reports" / "bench" / "summary.json",
    }
    for name, path in inputs.items():
        if not path.is_file():
            pytest.skip(f"{path} is not present in this checkout")
        assert published[name] == _load_script()._canonical_sha256(path), name


def test_frozen_corpus_manifest_digest_is_line_ending_independent() -> None:
    """The canonical digest equals the frozen corpus v2.1 manifest hash published by t22."""
    path = REPO_ROOT / "data" / "corpus" / "manifest.json"
    if not path.is_file():
        pytest.skip("corpus manifest not present")
    assert _load_script()._canonical_sha256(path) == FROZEN_CORPUS_MANIFEST_SHA256
