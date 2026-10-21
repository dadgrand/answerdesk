"""Offline invariant checker behind ``scripts/build_bench.py --verify``.

``--verify`` must be runnable on a fresh clone with no cache and no network: it reads the
committed release, recomputes every manifest hash and every item-level invariant with the
same functions the builder used, and re-checks the licence and provenance statements that
make the dataset externally auditable. It never imports ``httpx`` (``docs/benchmark.md`` §5).

Each check returns a :class:`Verdict` with ``pass``, ``fail`` or ``warn``. ``warn`` exists
only for checks that depend on a sibling artifact (the t11 golden split, the synthetic
corpus documents) which a downstream consumer may not have checked out; a ``fail`` always
means the release itself is wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import schema
from .build import golden_exclusions
from .checks import all_checks, read_bench
from .config import (
    BELEBELE_LANGS,
    BENCH_DIR,
    CORPUS_DIR,
    DOCS_DIR,
    GOLDEN_DIR,
    LICENSE_FILES,
    REPO_ROOT,
    SOURCE_LOCK,
    SPLITS_FILE,
)
from .normalize import canonical_json, sha256_bytes, sha256_file
from .sources import local_source_records, load_lock, source_records

ALLOWED_DATASETS = {
    "facebook/belebele",
    "castorini/mr-tydi",
    "castorini/mr-tydi-corpus",
    "answerdesk-synthetic",
}

REQUIRED_MANIFEST_KEYS = (
    "benchmark",
    "version",
    "schema_version",
    "seed",
    "languages",
    "suites",
    "counts",
    "sources",
    "files",
    "checks",
    "adapter",
    "data_sha256",
    "self_sha256",
)


@dataclass(frozen=True)
class Verdict:
    """One line of the ``--verify`` report."""

    name: str
    status: str
    detail: str


def _pass(name: str, detail: str) -> Verdict:
    return Verdict(name, "pass", detail)


def _fail(name: str, detail: str) -> Verdict:
    return Verdict(name, "fail", detail)


def _warn(name: str, detail: str) -> Verdict:
    return Verdict(name, "warn", detail)


def _read_release(bench_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    manifest_path = bench_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, bytes] = {}
    for row in manifest.get("files", []):
        path = bench_dir / str(row["path"])
        if path.is_file():
            files[str(row["path"])] = path.read_bytes()
    return files, manifest


def _manifest_self_hash(manifest: Mapping[str, Any]) -> str:
    copy = dict(manifest)
    copy["self_sha256"] = ""
    return sha256_bytes(canonical_json(copy).encode("utf-8"))


def _data_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json([[row["path"], row["sha256"]] for row in rows]).encode("utf-8"))


def _golden_doc_ids(golden_dir: Path, splits_file: Path) -> set[str] | None:
    if not golden_dir.is_dir():
        return None
    documents, _, _ = golden_exclusions(golden_dir, splits_file)
    return documents


def verify(
    bench_dir: Path = BENCH_DIR,
    *,
    corpus_dir: Path = CORPUS_DIR,
    golden_dir: Path = GOLDEN_DIR,
    splits_file: Path = SPLITS_FILE,
    lock_path: Path = SOURCE_LOCK,
    lock: Mapping[str, Any] | None = None,
) -> list[Verdict]:
    """Run every check and return the verdicts in a stable order."""
    verdicts: list[Verdict] = []
    manifest_path = bench_dir / "manifest.json"
    if not manifest_path.is_file():
        return [_fail("manifest_present", f"{manifest_path} does not exist — run --build")]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [_fail("manifest_present", f"{manifest_path}: {error}")]

    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    verdicts.append(
        _pass("manifest_schema", f"{len(manifest)} keys, version {manifest.get('version')}")
        if not missing
        else _fail("manifest_schema", f"missing keys: {missing}")
    )

    expected_self = _manifest_self_hash(manifest)
    verdicts.append(
        _pass("manifest_self_hash", expected_self)
        if expected_self == manifest.get("self_sha256")
        else _fail("manifest_self_hash", f"expected {expected_self}, manifest says {manifest.get('self_sha256')}")
    )

    rows = list(manifest.get("files", []))
    expected_data = _data_hash(rows)
    verdicts.append(
        _pass("manifest_data_hash", expected_data)
        if expected_data == manifest.get("data_sha256")
        else _fail("manifest_data_hash", f"expected {expected_data}, manifest says {manifest.get('data_sha256')}")
    )

    problems: list[str] = []
    files: dict[str, bytes] = {}
    for row in rows:
        path = bench_dir / str(row["path"])
        if not path.is_file():
            problems.append(f"missing {row['path']}")
            continue
        payload = path.read_bytes()
        files[str(row["path"])] = payload
        if len(payload) != int(row["bytes"]):
            problems.append(f"{row['path']}: {len(payload)} bytes != {row['bytes']}")
        if sha256_bytes(payload) != row["sha256"]:
            problems.append(f"{row['path']}: sha256 mismatch")
    verdicts.append(
        _pass("file_hashes", f"{len(rows)} files verified")
        if not problems
        else _fail("file_hashes", "; ".join(problems[:6]))
    )

    generated_roots = ("belebele", "mrtydi", "domain", "LICENSES")
    listed = {str(row["path"]) for row in rows}
    extra: list[str] = []
    for root in generated_roots:
        root_dir = bench_dir / root
        if not root_dir.exists():
            continue
        for path in sorted(root_dir.rglob("*")):
            if path.is_file() and path.suffix in (".jsonl", ".json", ".txt"):
                relative = path.relative_to(bench_dir).as_posix()
                if relative not in listed:
                    extra.append(relative)
    if (bench_dir / "schema.json").is_file() and "schema.json" not in listed:
        extra.append("schema.json")
    verdicts.append(
        _pass("file_inventory", f"{len(listed)} generated files listed, none extra")
        if not extra
        else _fail("file_inventory", f"unlisted generated files: {extra[:6]}")
    )

    items, passages, passage_files = read_bench(files)
    report = all_checks(items, passages, passage_files)

    schema_errors = report["items"]["schema_errors"]
    verdicts.append(
        _pass("item_schema", f"{report['items']['count']} items conform to schema v{manifest.get('schema_version')}")
        if report["items"]["ok"]
        else _fail("item_schema", f"{report['items']['schema_error_count']} violations: {schema_errors[:5]}")
    )
    verdicts.append(
        _pass("item_ids_unique", f"{len(items)} unique item ids")
        if not report["items"]["duplicate_item_ids"]
        else _fail("item_ids_unique", f"duplicates: {report['items']['duplicate_item_ids'][:5]}")
    )

    recomputed = report["counts"]
    manifest_counts = manifest.get("counts", {})
    count_mismatch = [
        key
        for key in ("items", "retrieval_queries", "qa_items", "domain_items", "by_suite", "by_language", "by_split")
        if key in manifest_counts and manifest_counts.get(key) != recomputed.get(key)
    ]
    verdicts.append(
        _pass("counts", f"{recomputed['items']} items: {recomputed['by_suite']}")
        if not count_mismatch
        else _fail("counts", f"manifest disagrees with files on {count_mismatch}")
    )
    verdicts.append(
        _pass("acceptance_floors", "all floors satisfied")
        if report["floors"]["ok"]
        else _fail("acceptance_floors", "; ".join(report["floors"]["problems"][:6]))
    )
    verdicts.append(
        _pass("passages", f"{report['passages']['passages']} passages, {report['passages']['references_checked']} references resolve")
        if report["passages"]["ok"]
        else _fail("passages", "; ".join(report["passages"]["problems"][:6]))
    )
    verdicts.append(
        _pass("answers_verifiable", f"{report['answers']['checked']} items checked")
        if report["answers"]["ok"]
        else _fail("answers_verifiable", "; ".join(report["answers"]["problems"][:6]))
    )
    verdicts.append(
        _pass("minhash_dedup", f"{report['dedup']['groups']} groups, threshold {report['dedup']['threshold']}")
        if report["dedup"]["ok"]
        else _fail("minhash_dedup", f"{report['dedup']['duplicate_count']} near-duplicate pairs: {report['dedup']['duplicates'][:3]}")
    )
    verdicts.append(
        _pass("script_detection", f"minimum share {report['scripts']['minimum_share_by_language']}")
        if report["scripts"]["ok"]
        else _fail("script_detection", "; ".join(report["scripts"]["problems"][:6]))
    )
    verdicts.append(
        _pass("split_disjointness", f"{report['splits']['ids_by_split']} ids; documents {report['splits']['documents_by_split']}")
        if report["splits"]["ok"]
        else _fail("split_disjointness", "; ".join(report["splits"]["problems"][:6]))
    )

    verdicts.extend(_golden_checks(items, bench_dir, golden_dir, splits_file))
    verdicts.extend(_domain_evidence_checks(items, corpus_dir))
    verdicts.extend(_license_checks(bench_dir))
    verdicts.append(_archive_check(bench_dir))
    verdicts.append(_sources_check(manifest, lock_path, lock))
    verdicts.append(_provenance_check(items))
    verdicts.append(_adapter_check(manifest, bench_dir))
    verdicts.extend(_docs_checks(bench_dir))
    return verdicts


def _golden_checks(
    items: Sequence[Mapping[str, Any]], bench_dir: Path, golden_dir: Path, splits_file: Path
) -> list[Verdict]:
    golden_documents = _golden_doc_ids(golden_dir, splits_file)
    domain_documents = {
        str(doc_id)
        for item in items
        if item["suite"] == "domain"
        for doc_id in item.get("gold_evidence_ids", [])
    }
    if golden_documents is None:
        return [
            _warn(
                "golden_disjointness",
                f"{golden_dir} is not present; the t11 exclusion cannot be re-checked on this checkout",
            )
        ]
    overlap = sorted(golden_documents & domain_documents)
    verdicts = [
        _pass("golden_disjointness", f"{len(domain_documents)} domain documents, none in t11's golden split")
        if not overlap
        else _fail("golden_disjointness", f"domain documents also used by t11: {overlap[:6]}")
    ]

    exclusions_path = bench_dir / "domain" / "exclusions.json"
    if not exclusions_path.is_file():
        verdicts.append(_fail("exclusions_recorded", "domain/exclusions.json is missing"))
        return verdicts
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    recorded = {str(value) for value in exclusions.get("excluded_document_ids", [])}
    unrecorded = sorted(golden_documents - recorded)
    stale = sorted(recorded - golden_documents)
    if not unrecorded and not stale:
        verdicts.append(_pass("exclusions_recorded", f"{len(recorded)} documents excluded from the domain slice"))
    elif unrecorded:
        verdicts.append(
            _fail(
                "exclusions_recorded",
                f"{len(unrecorded)} t11 documents are not recorded as excluded, e.g. {unrecorded[:4]} — rebuild the benchmark",
            )
        )
    else:
        verdicts.append(
            _warn(
                "exclusions_recorded",
                f"{len(stale)} recorded exclusions no longer exist in the golden split (t11 changed); rebuild to refresh",
            )
        )
    return verdicts


def _domain_evidence_checks(items: Sequence[Mapping[str, Any]], corpus_dir: Path) -> list[Verdict]:
    documents: dict[str, str] = {}
    for language in (*BELEBELE_LANGS, "tr"):
        directory = corpus_dir / "docs" / language
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            documents[path.stem] = path.read_text(encoding="utf-8")
    if not documents:
        return [_warn("domain_evidence", f"{corpus_dir / 'docs'} is not present; quotes cannot be re-checked")]

    problems: list[str] = []
    checked = 0
    for item in items:
        if item["suite"] != "domain":
            continue
        for entry in item.get("evidence", []):
            checked += 1
            doc_id = str(entry["doc_id"])
            quote = str(entry["quote"])
            document = documents.get(doc_id)
            if document is None:
                problems.append(f"{item['item_id']}: unknown corpus document {doc_id}")
            elif quote not in document:
                problems.append(f"{item['item_id']}: quote is not verbatim in {doc_id}")
    return [
        _pass("domain_evidence", f"{checked} evidence quotes are verbatim corpus slices")
        if not problems
        else _fail("domain_evidence", "; ".join(problems[:6]))
    ]


def _license_checks(bench_dir: Path) -> list[Verdict]:
    expectations = [
        ("belebele/LICENSE-CC-BY-SA-4.0.txt", "Attribution-ShareAlike 4.0 International"),
        ("mrtydi/LICENSE-APACHE-2.0.txt", "Apache License"),
        ("domain/LICENSE-CC0-1.0.txt", "CC0 1.0"),
        ("LICENSES/MIT.txt", "MIT License"),
    ]
    problems: list[str] = []
    for relative, marker in expectations:
        path = bench_dir / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif marker not in path.read_text(encoding="utf-8"):
            problems.append(f"{relative} does not contain {marker!r}")
    repo_license = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8") if (REPO_ROOT / "LICENSE").is_file() else ""
    if "MIT License" not in repo_license:
        problems.append("the repository LICENSE is not an MIT text")
    verdicts = [
        _pass("license_files", f"{len(expectations)} licence texts present")
        if not problems
        else _fail("license_files", "; ".join(problems[:6]))
    ]

    attribution_expectations = [
        ("belebele/ATTRIBUTION.md", ("Belebele", "CC BY-SA 4.0", "share-alike")),
        ("mrtydi/ATTRIBUTION.md", ("Mr. TyDi", "Apache-2.0")),
        ("DATASET_CARD.md", ("Belebele", "Mr. TyDi", "CC0-1.0", "Limitations")),
    ]
    problems = []
    for relative, markers in attribution_expectations:
        path = bench_dir / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
            continue
        text = path.read_text(encoding="utf-8").casefold()
        for marker in markers:
            if marker.casefold() not in text:
                problems.append(f"{relative} does not mention {marker!r}")
    verdicts.append(
        _pass("attribution", "dataset card and per-source attribution present")
        if not problems
        else _fail("attribution", "; ".join(problems[:6]))
    )

    card = (bench_dir / "DATASET_CARD.md").read_text(encoding="utf-8") if (bench_dir / "DATASET_CARD.md").is_file() else ""
    misrepresentation = bool(re.search(r"the dataset is MIT|data (?:is|are) MIT-licensed", card, re.IGNORECASE))
    verdicts.append(
        _fail("license_honesty", "the dataset card claims the data is MIT")
        if misrepresentation
        else _pass("license_honesty", "code MIT, Belebele CC BY-SA 4.0, Mr. TyDi Apache-2.0, domain CC0-1.0")
    )
    return verdicts


def _archive_check(bench_dir: Path) -> Verdict:
    problems: list[str] = []
    for path in sorted(bench_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in (".gz", ".zip", ".tar", ".tgz", ".bz2", ".xz"):
            problems.append(f"raw archive committed: {path.relative_to(bench_dir).as_posix()}")
        if path.stat().st_size > 8 * 1024 * 1024:
            problems.append(f"oversized file: {path.relative_to(bench_dir).as_posix()}")
        if path.name.startswith("belebele_") and path.name.endswith(".jsonl"):
            problems.append(f"raw Belebele file committed: {path.name}")
    return (
        _pass("no_raw_archives", "no archives, no oversized files, no raw Belebele release")
        if not problems
        else _fail("no_raw_archives", "; ".join(problems[:6]))
    )


def _sources_check(manifest: Mapping[str, Any], lock_path: Path, lock: Mapping[str, Any] | None) -> Verdict:
    if not lock_path.is_file() and lock is None:
        return _fail("sources_pinned", f"{lock_path} is missing")
    try:
        loaded = lock if lock is not None else load_lock(lock_path)
    except (OSError, ValueError) as error:  # pragma: no cover - corrupt lock
        return _fail("sources_pinned", str(error))
    expected = [*source_records(loaded), *local_source_records(loaded)]
    actual = manifest.get("sources", [])
    if canonical_json(expected) != canonical_json(actual):
        return _fail("sources_pinned", "manifest sources differ from sources.lock.json")
    return _pass("sources_pinned", f"{len(actual)} sources pinned by SHA-256")


def _provenance_check(items: Sequence[Mapping[str, Any]]) -> Verdict:
    problems: list[str] = []
    for item in items:
        dataset = str(item.get("provenance", {}).get("dataset", ""))
        if dataset not in ALLOWED_DATASETS:
            problems.append(f"{item['item_id']}: unexpected dataset {dataset!r}")
        if dataset.lower() in ("rgb", "crud-rag", "crud_rag"):
            problems.append(f"{item['item_id']}: licence-blocked dataset {dataset!r}")
    return (
        _pass("provenance_allowlist", f"datasets limited to {sorted(ALLOWED_DATASETS)}")
        if not problems
        else _fail("provenance_allowlist", "; ".join(problems[:6]))
    )


def _adapter_check(manifest: Mapping[str, Any], bench_dir: Path) -> Verdict:
    adapter = manifest.get("adapter")
    if not isinstance(adapter, Mapping):
        return _fail("t19_adapter", "manifest.adapter is missing")
    item_files = adapter.get("item_files", {})
    missing = [path for paths in item_files.values() for path in paths if not (bench_dir / path).is_file()]
    if missing:
        return _fail("t19_adapter", f"adapter paths missing: {missing[:4]}")
    return _pass("t19_adapter", f"{sum(len(paths) for paths in item_files.values())} item files mapped")


def _docs_checks(bench_dir: Path) -> list[Verdict]:
    card_path = bench_dir / "DATASET_CARD.md"
    docs_path = DOCS_DIR / "benchmark.md"
    verdicts: list[Verdict] = []
    if not card_path.is_file():
        verdicts.append(_fail("dataset_card", f"{card_path} is missing"))
    else:
        card = card_path.read_text(encoding="utf-8")
        required = ("Sources", "Sampling", "Intended use", "Limitations", "Citation", "License")
        absent = [section for section in required if section.lower() not in card.lower()]
        verdicts.append(
            _pass("dataset_card", f"{len(card.splitlines())} lines, all required sections present")
            if not absent
            else _fail("dataset_card", f"missing sections: {absent}")
        )
    if not docs_path.is_file():
        verdicts.append(_fail("benchmark_docs", f"{docs_path} is missing"))
    else:
        docs = docs_path.read_text(encoding="utf-8")
        absent = [marker for marker in ("t19", "derived", "Limitations") if marker.lower() not in docs.lower()]
        verdicts.append(
            _pass("benchmark_docs", f"{len(docs.splitlines())} lines")
            if not absent
            else _fail("benchmark_docs", f"missing markers: {absent}")
        )
    return verdicts


def summarise(verdicts: Iterable[Verdict]) -> str:
    """Render verdicts as one line each plus a summary line."""
    materialised = list(verdicts)
    lines = [f"{verdict.status.upper():4}  {verdict.name}: {verdict.detail}" for verdict in materialised]
    failures = sum(1 for verdict in materialised if verdict.status == "fail")
    warnings = sum(1 for verdict in materialised if verdict.status == "warn")
    lines.append(f"\n{len(materialised) - failures} passed, {failures} failed, {warnings} warnings")
    return "\n".join(lines)
