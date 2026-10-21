"""Determinism: canonical bytes, limit-driven sampling, and a rebuild that matches the commit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench.build import assemble
from bench.config import CACHE_DIR, REPO_ROOT
from bench.sources import MRTYDI_POOL_REL


def test_committed_files_are_canonical_jsonl(data_files: dict[str, bytes]) -> None:
    for path, payload in data_files.items():
        if not path.endswith(".jsonl"):
            continue
        assert payload.endswith(b"\n"), path
        assert b"\r\n" not in payload, path
        for line in payload.decode("utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            canonical = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            assert line == canonical, path


def test_committed_release_is_lf_only(bench_dir: Path) -> None:
    """Every committed text file in the release is LF-only and BOM-free.

    A fresh git checkout must be byte-identical on Windows and Linux: the manifest hashes
    the release files, so CRLF drift would invalidate every hash (``docs/benchmark.md`` §7).
    """
    offenders: list[str] = []
    checked = 0
    for path in sorted(bench_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".jsonl", ".json", ".md", ".txt"):
            continue
        checked += 1
        payload = path.read_bytes()
        if b"\r\n" in payload or payload.startswith(b"\xef\xbb\xbf"):
            offenders.append(path.relative_to(bench_dir).as_posix())
    assert checked >= 30
    assert offenders == []


def test_mini_build_is_byte_identical(mini_outputs: dict[str, Any]) -> None:
    first, second = mini_outputs["first"], mini_outputs["second"]
    assert first["files"] == second["files"]
    assert first["manifest"]["self_sha256"] == second["manifest"]["self_sha256"]
    first_files = {
        path.relative_to(first["out_dir"]).as_posix(): path.read_bytes()
        for path in first["out_dir"].rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second["out_dir"]).as_posix(): path.read_bytes()
        for path in second["out_dir"].rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_mini_build_counts_follow_the_limits(
    mini_outputs: dict[str, Any], mini_build: dict[str, Any]
) -> None:
    counts = mini_outputs["first"]["manifest"]["counts"]
    limits = mini_build["limits"]
    assert counts["by_suite"]["belebele_retrieval"] == len(("ru", "uz", "tg", "ky", "zh")) * (
        limits.belebele_test + limits.belebele_dev
    )
    assert counts["by_suite"]["belebele_qa"] == counts["by_suite"]["belebele_retrieval"]
    assert counts["by_suite"]["domain"] == sum(limits.domain_quotas.values()) * 6
    assert (
        mini_outputs["first"]["manifest"]["sampling"]["belebele"]["test_per_language"]
        == limits.belebele_test
    )


def test_mini_build_prunes_stale_generated_files(
    mini_build: dict[str, Any], mini_outputs: dict[str, Any]
) -> None:
    out_dir = mini_outputs["first"]["out_dir"]
    stale = out_dir / "belebele" / "qa" / "stale.jsonl"
    stale.write_text("{}\n", encoding="utf-8")
    assemble(
        cache_dir=mini_build["cache"],
        corpus_dir=mini_build["corpus"],
        golden_dir=mini_build["golden"],
        splits_file=mini_build["splits"],
        lock_path=mini_build["lock"],
        out_dir=out_dir,
        lax_cache=True,
        write=True,
        enforce_checks=False,
        limits=mini_build["limits"],
    )
    assert not stale.exists()


def test_build_without_a_cache_fails_with_a_fetch_hint(
    tmp_path: Path, mini_build: dict[str, Any]
) -> None:
    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    with pytest.raises(RuntimeError, match="--fetch"):
        assemble(
            cache_dir=empty_cache,
            corpus_dir=mini_build["corpus"],
            golden_dir=mini_build["golden"],
            splits_file=mini_build["splits"],
            lock_path=mini_build["lock"],
            out_dir=tmp_path / "out",
            lax_cache=False,
            write=False,
        )


def _stale_reason() -> str | None:
    """Why a real-cache rebuild may legitimately differ from the committed release.

    The build is a function of the pinned cache, the synthetic corpus and t11's golden
    items. If either repository artifact changed after the release was built, the rebuild
    is *supposed* to differ (and ``--verify`` says so too); the test only proves
    reproducibility on a checkout that matches the release.
    """
    from bench.config import SOURCE_LOCK
    from bench.sources import check_local_sources, load_lock

    if not (CACHE_DIR / MRTYDI_POOL_REL).is_file():
        return "the 1.6 GB download cache is not present; CI builds the fixture instead"
    problems = check_local_sources(load_lock(SOURCE_LOCK), REPO_ROOT)
    if problems:
        return "local sources changed since the release was built: " + problems[0]
    exclusions = json.loads(
        (
            REPO_ROOT
            / "data"
            / "bench"
            / "answerdesk-bench"
            / "v0.1"
            / "domain"
            / "exclusions.json"
        ).read_text(encoding="utf-8")
    )
    from bench.build import golden_exclusions
    from bench.config import GOLDEN_DIR, SPLITS_FILE

    _, _, golden_hashes = golden_exclusions(GOLDEN_DIR, SPLITS_FILE)
    if exclusions["source_files"] != golden_hashes:
        return "the golden snapshot changed since the release was built"
    return None


def test_real_cache_rebuild_matches_the_committed_release(data_files: dict[str, bytes]) -> None:
    reason = _stale_reason()
    if reason is not None:
        pytest.skip(reason)
    from bench.config import CORPUS_DIR, GOLDEN_DIR, SOURCE_LOCK, SPLITS_FILE

    files, _ = assemble(
        cache_dir=CACHE_DIR,
        corpus_dir=CORPUS_DIR,
        golden_dir=GOLDEN_DIR,
        splits_file=SPLITS_FILE,
        lock_path=SOURCE_LOCK,
        out_dir=CACHE_DIR / "unused",
        write=False,
    )
    manifest_bytes = files.pop("manifest.json")
    assert files == data_files
    assert (
        manifest_bytes
        == (
            REPO_ROOT / "data" / "bench" / "answerdesk-bench" / "v0.1" / "manifest.json"
        ).read_bytes()
    )
