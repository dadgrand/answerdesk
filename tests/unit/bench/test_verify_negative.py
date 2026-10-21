"""``--verify`` must accept the committed release and reject every class of corruption.

The negative cases are the point: a checker that never fails is not a checker. Each test
copies the release, introduces exactly one defect and asserts the verdict that names it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from bench.config import SOURCE_LOCK
from bench.normalize import canonical_json
from bench.verify import verify


def _verdicts(bench_dir: Path) -> dict[str, Any]:
    return {verdict.name: verdict for verdict in verify(bench_dir, lock_path=SOURCE_LOCK)}


@pytest.fixture(scope="module")
def clean_verdicts(bench_dir: Path) -> dict[str, Any]:
    return _verdicts(bench_dir)


@pytest.fixture()
def copied(bench_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / "bench"
    shutil.copytree(bench_dir, target)
    return target


def _rewrite_manifest(bench: Path, mutate: Any) -> None:
    """Apply ``mutate(manifest)``, then recompute the data and self hashes honestly."""
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["data_sha256"] = hashlib.sha256(
        canonical_json([[row["path"], row["sha256"]] for row in manifest["files"]]).encode("utf-8")
    ).hexdigest()
    manifest["self_sha256"] = ""
    manifest["self_sha256"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    (bench / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rehash_files(bench: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        for row in manifest["files"]:
            payload = (bench / row["path"]).read_bytes()
            row["bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()

    _rewrite_manifest(bench, mutate)


def test_the_committed_release_passes(clean_verdicts: dict[str, Any]) -> None:
    failures = [name for name, verdict in clean_verdicts.items() if verdict.status == "fail"]
    assert failures == []
    assert len(clean_verdicts) >= 25


def test_corrupted_file_hash_is_rejected(copied: Path) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["files"][0]["sha256"] = "0" * 64

    _rewrite_manifest(copied, mutate)
    verdict = _verdicts(copied)["file_hashes"]
    assert verdict.status == "fail"
    assert "sha256" in verdict.detail


def test_corrupted_self_hash_is_rejected(copied: Path) -> None:
    manifest = json.loads((copied / "manifest.json").read_text(encoding="utf-8"))
    manifest["self_sha256"] = "f" * 64
    (copied / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert _verdicts(copied)["manifest_self_hash"].status == "fail"


def test_count_mismatch_is_rejected(copied: Path) -> None:
    _rewrite_manifest(copied, lambda manifest: manifest["counts"].__setitem__("items", 1))
    assert _verdicts(copied)["counts"].status == "fail"


def test_tampered_item_is_rejected(copied: Path) -> None:
    path = copied / "belebele" / "qa" / "ru.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["answer_index"] = 9
    rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _rehash_files(copied)
    verdicts = _verdicts(copied)
    assert verdicts["item_schema"].status == "fail"
    assert verdicts["manifest_self_hash"].status == "pass"


def test_missing_licence_text_is_rejected(copied: Path) -> None:
    (copied / "mrtydi" / "LICENSE-APACHE-2.0.txt").unlink()
    verdicts = _verdicts(copied)
    assert verdicts["license_files"].status == "fail"
    assert verdicts["file_hashes"].status == "fail"


def test_committed_archive_is_rejected(copied: Path) -> None:
    (copied / "belebele" / "raw.jsonl.gz").write_bytes(b"\x1f\x8b")
    assert _verdicts(copied)["no_raw_archives"].status == "fail"


def test_dataset_card_licence_misstatement_is_rejected(copied: Path) -> None:
    card = copied / "DATASET_CARD.md"
    card.write_text(
        card.read_text(encoding="utf-8").replace(
            "The repository is MIT; this dataset is not.", "The dataset is MIT-licensed."
        ),
        encoding="utf-8",
    )
    assert _verdicts(copied)["license_honesty"].status == "fail"
