"""External-validity datasets: licence registry, cache and integrity checks.

Nothing here downloads automatically. ``ensure()`` returns the cached copy or raises an
actionable :class:`ExternalDataError`; the CLI turns that into a printed instruction. The
cache lives under ``.cache/answerdesk-eval/`` (gitignored) and every dataset is pinned by a
licence and, where the publisher provides one, a checksum.

Two classes of data are deliberately separated:

* **curated** — ``answerdesk-bench`` v0.1, committed to the repository, the primary suite;
* **upstream** — the original publishers (Belebele, Mr. TyDi, MIRACL, ALCE). CC BY-SA data is
  read in place from the cache and is **never** copied into reports or the repository; only
  aggregate numbers and item ids leave the adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from answerdesk.errors import AnswerdeskError

CURATED_BENCH = Path("data") / "bench" / "answerdesk-bench" / "v0.1"
CACHE_ENV = "ANSWERDESK_EVAL_CACHE"


class ExternalDataError(AnswerdeskError):
    """A dataset is missing, unlicensed, or fails its integrity check."""


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset the external arm may use."""

    name: str
    version: str
    licence: str
    homepage: str
    task: str
    languages: tuple[str, ...]
    #: local cache sub-directory (None for the committed curated bench)
    cache_dir: str | None
    #: True when the licence forbids redistributing the data in this repository
    redistribute: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "homepage": self.homepage,
            "task": self.task,
            "languages": list(self.languages),
            "cache_dir": self.cache_dir,
            "redistribute": self.redistribute,
        }


@dataclass(frozen=True)
class DatasetFile:
    path: str
    sha256: str
    bytes: int
    items: int


@dataclass
class DatasetRecord:
    """A dataset that is present locally and verified."""

    spec: DatasetSpec
    root: Path
    files: list[DatasetFile] = field(default_factory=list)
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            **self.spec.to_dict(),
            "root": str(self.root).replace("\\", "/"),
            "manifest_sha256": self.manifest_sha256,
            "files": [
                {"path": f.path, "sha256": f.sha256, "bytes": f.bytes, "items": f.items}
                for f in self.files
            ],
        }


REGISTRY: dict[str, DatasetSpec] = {
    "answerdesk-bench": DatasetSpec(
        name="answerdesk-bench",
        version="v0.1",
        licence="mixed: CC-BY-SA-4.0 (belebele), CC0-1.0 (domain), Apache-2.0 (mrtydi), MIT (repo)",
        homepage="https://github.com/answerdesk/answerdesk/tree/main/data/bench",
        task="curated suite: Belebele-derived retrieval+QA, Mr. TyDi ru, domain slice",
        languages=("ru", "uz", "tg", "ky", "zh", "tr"),
        cache_dir=None,
        redistribute=True,
    ),
    "belebele": DatasetSpec(
        name="belebele",
        version="1.0",
        licence="CC-BY-SA-4.0",
        homepage="https://huggingface.co/datasets/facebook/belebele",
        task="reading comprehension (4-way multiple choice)",
        languages=("ru", "uz", "tg", "ky", "zh"),
        cache_dir="belebele",
        redistribute=False,
    ),
    "mrtydi": DatasetSpec(
        name="mrtydi",
        version="1.1",
        licence="Apache-2.0",
        homepage="https://huggingface.co/datasets/castorini/mr-tydi",
        task="retrieval with qrels",
        languages=("ru",),
        cache_dir="mrtydi",
        redistribute=True,
    ),
    "miracl": DatasetSpec(
        name="miracl",
        version="1.0",
        licence="Apache-2.0",
        homepage="https://huggingface.co/datasets/miracl/miracl",
        task="retrieval with qrels",
        languages=("ru", "zh"),
        cache_dir="miracl",
        redistribute=True,
    ),
    "alce": DatasetSpec(
        name="alce",
        version="1.0",
        licence="MIT (repo); constituent corpora unverified — not vendored",
        homepage="https://github.com/princeton-nlp/ALCE",
        task="citation (provided top-100 runs only)",
        languages=("en",),
        cache_dir="alce",
        redistribute=False,
    ),
}


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override) if override else Path(".cache") / "answerdesk-eval"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def curated_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    candidate = root / CURATED_BENCH
    if not candidate.is_dir():
        raise ExternalDataError(
            "the curated benchmark is missing; run from the repository root or pass --bench",
            detail=f"looked for {candidate}",
        )
    return candidate


def load_curated(repo_root: Path | None = None, *, verify: bool = False) -> DatasetRecord:
    """Load ``answerdesk-bench`` v0.1 and optionally re-verify every file hash."""
    root = curated_root(repo_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ExternalDataError("manifest.json is missing from the curated benchmark")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [
        DatasetFile(
            path=entry["path"],
            sha256=entry["sha256"],
            bytes=entry.get("bytes", 0),
            items=entry.get("items", 0),
        )
        for entry in manifest.get("files", [])
    ]
    if verify:
        for entry in files:
            path = root / entry.path
            if not path.is_file():
                raise ExternalDataError(f"curated file is missing: {entry.path}")
            actual = _sha256(path)
            if actual != entry.sha256:
                raise ExternalDataError(
                    f"curated file failed its checksum: {entry.path}",
                    detail=f"expected {entry.sha256}, got {actual}",
                )
    return DatasetRecord(
        spec=REGISTRY["answerdesk-bench"],
        root=root,
        files=files,
        manifest_sha256=manifest.get("self_sha256", _sha256(manifest_path)),
    )


def ensure(name: str, *, cache: Path | None = None) -> Path:
    """Return the cached directory of an upstream dataset, or explain how to fetch it."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ExternalDataError(f"unknown dataset {name!r}", detail=f"known: {sorted(REGISTRY)}")
    if spec.cache_dir is None:
        return curated_root()
    root = (cache or cache_root()) / spec.cache_dir
    if not root.is_dir():
        raise ExternalDataError(
            f"dataset {name!r} is not cached; fetch it with the documented command and rerun",
            detail=(
                f"expected {root}; licence {spec.licence}; source {spec.homepage}. "
                "The arm never downloads implicitly: run `answerdesk eval external --fetch "
                f"{name}` on a machine allowed to reach the publisher."
            ),
        )
    return root


def verify_upstream(name: str, *, cache: Path | None = None) -> dict[str, object]:
    """Check whatever ``sources.lock.json`` the cached dataset ships, when it has one."""
    root = ensure(name, cache=cache)
    lock = root / "sources.lock.json"
    if not lock.is_file():
        return {"dataset": name, "verified": False, "reason": "no sources.lock.json in cache"}
    payload = json.loads(lock.read_text(encoding="utf-8"))
    mismatched: list[str] = []
    for entry in payload.get("files", []):
        path = root / entry["path"]
        if not path.is_file():
            mismatched.append(f"missing {entry['path']}")
            continue
        actual = _sha256(path)
        if actual != entry.get("sha256"):
            mismatched.append(f"{entry['path']}: {actual} != {entry.get('sha256')}")
    return {"dataset": name, "verified": not mismatched, "problems": mismatched}


__all__ = [
    "CACHE_ENV",
    "CURATED_BENCH",
    "REGISTRY",
    "DatasetFile",
    "DatasetRecord",
    "DatasetSpec",
    "ExternalDataError",
    "cache_root",
    "curated_root",
    "ensure",
    "load_curated",
    "verify_upstream",
]
