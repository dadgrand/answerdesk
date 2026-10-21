"""Frozen, hashed document-level splits (FR-EVL-03, §10.2).

The split unit is the **document**, never the query. Two queries paraphrased from the same
paragraph in different halves of the split would leak the answer into the fitting half and
inflate every reported number; that failure is invisible in the metrics and obvious in the
manifest, so the manifest is where it is caught.

Determinism: documents are ordered by ``sha256(seed:doc_id)`` and the first
``ceil(ratio * n)`` become the test half. A hash-ordered cut is reproducible without
depending on dictionary order, filesystem order or Python's randomised ``hash()``, and it
yields exactly the requested ratio instead of a binomial approximation of it.

    # AT-EVL-03  # AT-DET-02
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from answerdesk.eval.corpus import stable_hash
from answerdesk.eval.errors import EvalDataError, SplitLeakageError
from answerdesk.eval.models import SplitManifest


def _rank_key(seed: int, doc_id: str) -> str:
    return hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()


def build_split(
    document_ids: Iterable[str],
    *,
    seed: int,
    corpus_version: str,
    test_ratio: float = 0.4,
    version: str = "v1",
) -> SplitManifest:
    """Deterministic document split with exactly ``ceil(test_ratio * n)`` test documents."""
    documents = sorted(set(document_ids))
    if not documents:
        raise EvalDataError("cannot split an empty document set")
    if not 0.0 < test_ratio < 1.0:
        raise EvalDataError(f"test_ratio must be in (0, 1), got {test_ratio}")
    ordered = sorted(documents, key=lambda doc_id: (_rank_key(seed, doc_id), doc_id))
    test_count = max(1, min(len(ordered) - 1, math.ceil(test_ratio * len(ordered))))
    test_docs = sorted(ordered[:test_count])
    dev_docs = sorted(ordered[test_count:])
    manifest = SplitManifest(
        version=version,
        seed=seed,
        corpus_version=corpus_version,
        test_ratio=test_ratio,
        dev_doc_ids=dev_docs,
        test_doc_ids=test_docs,
    )
    return manifest.model_copy(update={"sha256": split_hash(manifest)})


def build_stratified_split(
    corpus: object,
    *,
    seed: int,
    corpus_version: str,
    test_ratio: float = 0.4,
    quotas: dict[str, int] | None = None,
    version: str = "v1",
) -> SplitManifest:
    """Document split that also *feeds the category quotas* (§10.1 + §10.2 together).

    A pure hash cut keeps the 60/40 document ratio but can leave a locale with too few
    numeric/date questions in its test half — and then either the quota check fails or the
    generator silently under-fills a category. This function therefore picks test documents
    greedily by *deficit coverage*: each document is scored by how many still-unmet
    category needs its queries satisfy, ties break on the seeded hash, and the ratio still
    decides how many documents move. Document-disjointness is untouched: the unit is still
    the document.
    """
    from answerdesk.eval.corpus import Corpus

    if not isinstance(corpus, Corpus):  # pragma: no cover - defensive
        raise EvalDataError("build_stratified_split needs a Corpus")
    needs = dict(quotas or {})
    document_ids = corpus.document_ids
    if not document_ids:
        raise EvalDataError("cannot split an empty document set")
    if not 0.0 < test_ratio < 1.0:
        raise EvalDataError(f"test_ratio must be in (0, 1), got {test_ratio}")

    by_locale: dict[str, list[str]] = {}
    for doc_id in document_ids:
        chunks = corpus.doc_chunks(doc_id)
        locale = next((chunk.language for chunk in chunks if chunk.language), "unknown")
        by_locale.setdefault(locale, []).append(doc_id)

    #: Corpus query category → the golden category whose quota it can satisfy.
    supply_of = {
        "direct": "direct",
        "numeric_date": "numeric_date",
        "identifier": "numeric_date",
        "typo": "typo_code_switch",
        "code_switch": "typo_code_switch",
    }
    test_docs: list[str] = []
    dev_docs: list[str] = []
    for locale in sorted(by_locale):
        docs = sorted(by_locale[locale])
        target = max(1, min(len(docs) - 1, math.ceil(test_ratio * len(docs))))
        supplies: dict[str, dict[str, int]] = {doc: {} for doc in docs}
        for doc in docs:
            for query in corpus.queries_for_doc(doc):
                category = supply_of.get(query.category)
                if category is None:
                    continue
                supplies[doc][category] = supplies[doc].get(category, 0) + 1
        remaining_need = dict(needs)
        rank = {doc: _rank_key(seed, doc) for doc in docs}
        remaining = list(docs)
        chosen: list[str] = []
        while len(chosen) < target and remaining:
            best: str | None = None
            best_key: tuple[int, str] | None = None
            for doc in remaining:
                covered = sum(
                    min(count, remaining_need.get(category, 0))
                    for category, count in supplies[doc].items()
                )
                key = (-covered, rank[doc])
                if best_key is None or key < best_key:
                    best, best_key = doc, key
            if best is None:  # pragma: no cover - `remaining` is non-empty inside the loop
                break
            chosen.append(best)
            remaining.remove(best)
            for query in corpus.queries_for_doc(best):
                category = supply_of.get(query.category)
                if category and remaining_need.get(category, 0) > 0:
                    remaining_need[category] -= 1
        test_docs.extend(chosen)
        dev_docs.extend(remaining)

    manifest = SplitManifest(
        version=version,
        seed=seed,
        corpus_version=corpus_version,
        test_ratio=test_ratio,
        dev_doc_ids=sorted(dev_docs),
        test_doc_ids=sorted(test_docs),
    )
    return manifest.model_copy(update={"sha256": split_hash(manifest)})


def split_hash(manifest: SplitManifest) -> str:
    """sha256 over the canonical payload — the value printed in every report header."""
    canonical = json.dumps(
        manifest.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_no_leakage(
    *,
    test_document_ids: Sequence[str],
    dev_document_ids: Sequence[str],
    case_ids: Sequence[str] = (),
) -> None:
    """Raise when a test item derives from a document that also feeds dev (§10.2)."""
    overlap = sorted(set(test_document_ids) & set(dev_document_ids))
    if overlap:
        raise SplitLeakageError(
            case_ids or [f"document:{doc_id}" for doc_id in overlap],
            overlap,
        )


def verify_split(manifest: SplitManifest) -> SplitManifest:
    """Recompute the self-hash; a hand-edited split manifest is a hard failure."""
    expected = split_hash(manifest)
    if manifest.sha256 != expected:
        raise EvalDataError(
            f"split manifest {manifest.version!r} hash mismatch: file says {manifest.sha256}, "
            f"recomputed {expected}. The split is frozen (§10.2); regenerate it deliberately.",
        )
    return manifest


def write_split(manifest: SplitManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def read_split(path: str | Path) -> SplitManifest:
    source = Path(path)
    if not source.is_file():
        raise EvalDataError(
            f"split manifest not found at {source}; run `answerdesk eval --regenerate` once "
            "to freeze it (§10.2)."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalDataError(f"cannot read split manifest {source}: {exc}") from exc
    try:
        manifest = SplitManifest.model_validate(payload)
    except ValueError as exc:
        raise EvalDataError(f"split manifest {source} is not a valid SplitManifest: {exc}") from exc
    return verify_split(manifest)


def split_fingerprint(manifest: SplitManifest) -> str:
    """Short stable identity used in case ids and report tables."""
    return stable_hash(manifest.version, manifest.sha256)[:12]


__all__ = [
    "assert_no_leakage",
    "build_split",
    "build_stratified_split",
    "read_split",
    "split_fingerprint",
    "split_hash",
    "verify_split",
    "write_split",
]
