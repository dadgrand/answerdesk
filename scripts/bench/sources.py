"""Pinned downloads, the git-ignored cache, and the derived Mr. TyDi passage pool.

Three rules shape this module:

1. **Pins live in the repository, bytes live in the cache.** ``sources.lock.json`` is
   committed and carries the URL, size and SHA-256 of every download; the 1.6 GB
   Mr. TyDi archive and the Belebele files are never committed (``docs/benchmark.md``
   §7). ``--build`` and ``--verify`` never touch the network.
2. **Fetch is explicit.** Only ``--fetch`` opens a connection, and it fails loudly when a
   download does not match its pin (a silently changed upstream file must not enter the
   benchmark).
3. **Derivation is cached, not repeated.** Mr. TyDi's archive holds 9.6 M passages; the
   benchmark needs the 2,428 that are positive evidence in dev+test. ``--build`` derives
   that pool once (no network) and reuses it.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SOURCE_LOCK
from .normalize import sha256_file

MRTYDI_POOL_REL: str = "derived/mrtydi-ru/passages.jsonl"


@dataclass(frozen=True)
class PinnedFile:
    """One download: where it goes in the cache, where it comes from, what it must hash to."""

    source_id: str
    cache_path: str
    url: str
    sha256: str
    bytes: int
    language: str = ""
    published_path: str = ""


def load_lock(path: Path | None = None) -> dict[str, Any]:
    """Read ``sources.lock.json`` (committed) and return the parsed document."""
    lock_path = Path(path) if path is not None else SOURCE_LOCK
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    if not isinstance(lock, dict) or not isinstance(lock.get("sources"), list):
        raise ValueError(f"{lock_path}: not an answerdesk-bench source lock")
    return lock


def iter_pinned(lock: Mapping[str, Any]) -> list[PinnedFile]:
    """Flatten the lock into the list of files the cache must contain."""
    pinned: list[PinnedFile] = []
    for source in lock["sources"]:
        source_id = str(source["id"])
        entries: list[Mapping[str, Any]] = list(source.get("files", []))
        corpus = source.get("corpus")
        if isinstance(corpus, Mapping):
            entries.append(corpus)
        for entry in entries:
            pinned.append(
                PinnedFile(
                    source_id=source_id,
                    cache_path=str(entry["cache_path"]),
                    url=str(entry["url"]),
                    sha256=str(entry["sha256"]),
                    bytes=int(entry["bytes"]),
                    language=str(entry.get("language", "")),
                    published_path=str(entry.get("published_path", "")),
                )
            )
    return pinned


def check_cache(cache_dir: Path, lock: Mapping[str, Any]) -> list[str]:
    """Return one message per missing or hash-mismatched cache file (empty is good)."""
    problems: list[str] = []
    for entry in iter_pinned(lock):
        path = cache_dir / entry.cache_path
        if not path.is_file():
            problems.append(f"missing cached file {entry.cache_path} — run --fetch")
            continue
        size = path.stat().st_size
        if size != entry.bytes:
            problems.append(f"{entry.cache_path}: {size} bytes, pinned {entry.bytes}")
        digest = sha256_file(path)
        if digest != entry.sha256:
            problems.append(f"{entry.cache_path}: sha256 {digest} != pinned {entry.sha256}")
    return problems


def _download(client: Any, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as sink:
                for chunk in response.iter_bytes(chunk_size=1024 * 256):
                    sink.write(chunk)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch(
    cache_dir: Path, lock: Mapping[str, Any], *, force: bool = False, log: Any = print
) -> dict[str, Any]:
    """Download every pinned file into ``cache_dir`` and verify its hash.

    Network access happens here and only here. Returns a small manifest written to
    ``<cache>/fetch.json`` (cache-local, so it may carry timestamps).
    """
    import datetime as _dt

    import httpx

    cache_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"files": [], "fetched_at": _dt.datetime.now(_dt.UTC).isoformat()}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, read=600.0)) as client:
        for entry in iter_pinned(lock):
            destination = cache_dir / entry.cache_path
            if (
                not force
                and destination.is_file()
                and destination.stat().st_size == entry.bytes
                and sha256_file(destination) == entry.sha256
            ):
                log(f"cached  {entry.cache_path}")
                report["files"].append({"cache_path": entry.cache_path, "status": "cached"})
                continue
            log(f"fetch   {entry.cache_path} <- {entry.url}")
            _download(client, entry.url, destination)
            digest = sha256_file(destination)
            if destination.stat().st_size != entry.bytes or digest != entry.sha256:
                raise RuntimeError(
                    f"{entry.cache_path}: downloaded "
                    f"{destination.stat().st_size} bytes / {digest}, "
                    f"pinned {entry.bytes} / {entry.sha256}. The upstream file changed; "
                    "review it and re-pin deliberately."
                )
            report["files"].append(
                {"cache_path": entry.cache_path, "status": "downloaded", "sha256": digest}
            )
    (cache_dir / "fetch.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _needed_mrtydi_docids(cache_dir: Path) -> set[str]:
    """All positive-passage ids of the Mr. TyDi ru dev and test splits."""
    needed: set[str] = set()
    for split in ("dev", "test"):
        path = cache_dir / "raw" / "mrtydi" / f"{split}.ru.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for passage in record.get("positive_passages", []):
                    needed.add(str(passage["docid"]))
    return needed


def derive_mrtydi_pool(
    cache_dir: Path, lock: Mapping[str, Any], *, force: bool = False, log: Any = print
) -> Path:
    """Filter the 1.6 GB Mr. TyDi corpus down to the passages this benchmark needs.

    The result is a deterministic JSONL file sorted by ``docid``: one line per passage with
    ``docid``, ``title`` and ``text``. It lives in the cache and is never committed.
    """
    destination = cache_dir / MRTYDI_POOL_REL
    needed = _needed_mrtydi_docids(cache_dir)
    if destination.is_file() and not force:
        with destination.open(encoding="utf-8") as handle:
            have = {json.loads(line)["docid"] for line in handle if line.strip()}
        if have >= needed:
            log(f"cached  {MRTYDI_POOL_REL} ({len(have)} passages)")
            return destination

    corpus_path = cache_dir / "raw" / "mrtydi" / "corpus.ru.jsonl.gz"
    log(f"derive  {MRTYDI_POOL_REL} from {corpus_path.name} ({len(needed)} passages needed)")
    found: dict[str, dict[str, str]] = {}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            docid = str(record["docid"])
            if docid in needed:
                found[docid] = {
                    "docid": docid,
                    "title": str(record.get("title", "")),
                    "text": str(record["text"]),
                }
    missing = needed - set(found)
    if missing:
        raise RuntimeError(
            f"Mr. TyDi corpus lacks {len(missing)} positive passages, e.g. {sorted(missing)[:3]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for docid in sorted(found):
            handle.write(json.dumps(found[docid], ensure_ascii=False, sort_keys=True) + "\n")
    log(f"wrote   {MRTYDI_POOL_REL} ({len(found)} passages)")
    return destination


def read_mrtydi_pool(path: Path) -> dict[str, dict[str, str]]:
    """Load the derived pool into a ``docid -> {docid,title,text}`` mapping."""
    pool: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pool[str(record["docid"])] = {
                "docid": str(record["docid"]),
                "title": str(record.get("title", "")),
                "text": str(record["text"]),
            }
    return pool


def source_records(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The lock's sources reduced to the fields a provenance record may cite."""
    records: list[dict[str, Any]] = []
    for source in lock["sources"]:
        entries: list[dict[str, Any]] = []
        for entry in list(source.get("files", [])) + (
            [source["corpus"]] if source.get("corpus") else []
        ):
            entries.append(
                {
                    "cache_path": entry["cache_path"],
                    "published_path": entry.get("published_path", ""),
                    "sha256": entry["sha256"],
                    "bytes": entry["bytes"],
                    "language": entry.get("language", ""),
                }
            )
        records.append(
            {
                "id": source["id"],
                "dataset": source["dataset"],
                "version": source["version"],
                "license": source["license"],
                "homepage": source.get("homepage", ""),
                "citation": source.get("citation", ""),
                "files": entries,
            }
        )
    return records


def published_hashes(lock: Mapping[str, Any]) -> dict[str, str]:
    """Map ``published_path`` (e.g. ``data/rus_Cyrl.jsonl``) to its pinned SHA-256."""
    hashes: dict[str, str] = {}
    for source in lock["sources"]:
        for entry in source.get("files", []):
            if entry.get("published_path"):
                hashes[str(entry["published_path"])] = str(entry["sha256"])
    return hashes


def local_source_records(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sources that live in the repository (the synthetic corpus), pinned by hash."""
    records: list[dict[str, Any]] = []
    for source in lock.get("local_sources", []):
        records.append(
            {
                "id": source["id"],
                "dataset": source["dataset"],
                "version": source["version"],
                "license": source["license"],
                "homepage": source.get("homepage", ""),
                "citation": source.get("citation", ""),
                "files": [
                    {
                        "repo_path": entry["repo_path"],
                        "sha256": entry["sha256"],
                        "bytes": entry.get("bytes", 0),
                    }
                    for entry in source.get("files", [])
                ],
            }
        )
    return records


def check_local_sources(lock: Mapping[str, Any], repo_root: Path) -> list[str]:
    """Return one message per repository file that no longer matches its pin."""
    problems: list[str] = []
    for source in lock.get("local_sources", []):
        for entry in source.get("files", []):
            path = repo_root / str(entry["repo_path"])
            if not path.is_file():
                problems.append(f"missing local source file {entry['repo_path']}")
                continue
            digest = sha256_file(path)
            if digest != str(entry["sha256"]):
                problems.append(
                    f"{entry['repo_path']}: sha256 {digest} != pinned {entry['sha256']} "
                    "(regenerate the corpus or re-pin deliberately)"
                )
    return problems


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, ignoring blank lines."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:  # pragma: no cover - corrupt cache
                raise ValueError(f"{path}:{number}: {error}") from error


def take(sequence: Sequence[Any], limit: int) -> list[Any]:
    """Small helper so builders read as data pipelines rather than slicing expressions."""
    return list(sequence[:limit])
