"""Deterministic generation of the committed ``data/bench`` tree.

The builder is a pure function of three inputs: the pinned cache, the synthetic corpus and
the t11 golden split (used only to *exclude* documents). It writes canonical JSONL — one
object per line, keys sorted, no ASCII escaping, LF endings — and a manifest that pins the
SHA-256 of every file. Running it twice produces identical bytes; the offline test
``tests/unit/bench/test_determinism.py`` proves that on a fixture and
``test_build_matches_committed`` proves it on the real cache when the cache is present.

Sections
--------
``build_belebele``  Belebele passage pools + derived qrels + MCQ items (CC BY-SA 4.0).
``build_mrtydi``    Mr. TyDi ru native qrels over a derived passage pool (Apache-2.0).
``build_domain``    Support-domain slice from the CC0 synthetic corpus.
``assemble``        Writes files, copies licence texts and produces ``manifest.json``.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import schema
from .checks import all_checks, read_bench
from .config import (
    BELEBELE_DEV_PER_LANG,
    BELEBELE_LANGS,
    BELEBELE_SPLIT_MODULUS,
    BELEBELE_TEST_PER_LANG,
    BENCH_NAME,
    DOMAIN_DEV_MODULUS,
    DOMAIN_LANGS,
    DOMAIN_QUOTAS,
    LICENSE_FILES,
    MINHASH_BANDS,
    MINHASH_HASHES,
    MINHASH_THRESHOLD,
    RELEASE_DATE,
    REPO_ROOT,
    SCHEMA_VERSION,
    SEED,
    SHINGLE_SIZE,
    SOURCE_MRTYDI,
    SUITE_LICENSES,
    VERSION,
)
from .normalize import (
    answer_anchor,
    canonical_json,
    normalise_question,
    normalise_text,
    sha256_bytes,
    sha256_file,
    sha256_text,
    tokens,
)
from .sources import (
    MRTYDI_POOL_REL,
    check_cache,
    check_local_sources,
    derive_mrtydi_pool,
    iter_jsonl,
    load_lock,
    local_source_records,
    read_mrtydi_pool,
    source_records,
)
from .templates import INJECTION_TEMPLATES, PHONE_RANGES, PII_TEMPLATES, RESERVED_EMAIL, TEST_PAN


def _rank(*parts: object) -> str:
    """Deterministic sort key: ``sha256(seed:part:part:...)``."""
    return sha256_text(":".join(str(part) for part in (SEED, *parts)))


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _source_file(lock: Mapping[str, Any], source_id: str, language: str = "") -> dict[str, Any]:
    """Find one pinned file record by source id and (optionally) language."""
    for source in lock["sources"]:
        if source["id"] != source_id:
            continue
        for entry in source.get("files", []):
            if not language or entry.get("language") == language:
                return dict(entry)
    raise KeyError(f"no pinned file for source={source_id!r} language={language!r}")


def _source_version(lock: Mapping[str, Any], source_id: str) -> str:
    for source in lock["sources"]:
        if source["id"] == source_id:
            return str(source["version"])
    raise KeyError(source_id)


@dataclass
class SuiteData:
    """Corpus and item rows for one build pass, keyed by suite or corpus name."""

    corpus: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingLimits:
    """How many items to sample. Defaults come from ``config``; tests shrink them.

    The limits are also recorded in the manifest, so a small fixture build cannot be
    mistaken for the shipped sampling.
    """

    belebele_test: int = BELEBELE_TEST_PER_LANG
    belebele_dev: int = BELEBELE_DEV_PER_LANG
    domain_quotas: Mapping[str, int] = field(default_factory=lambda: dict(DOMAIN_QUOTAS))


DEFAULT_LIMITS = SamplingLimits()


# --------------------------------------------------------------------------- Belebele


def _belebele_passages(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate FLORES passages in first-appearance order (the release is ordered)."""
    order: dict[str, int] = {}
    passages: list[dict[str, Any]] = []
    for row in rows:
        text = str(row["flores_passage"])
        if text not in order:
            order[text] = len(passages)
            passages.append({"text": text, "link": str(row.get("link", "")), "numbers": []})
        passages[order[text]]["numbers"].append(int(row["question_number"]))
    return passages


def _usable_questions(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int]:
    """Drop source rows whose four options are not distinct (a handful in Belebele)."""
    usable: list[Mapping[str, Any]] = []
    skipped = 0
    for row in rows:
        options = [str(row[f"mc_answer{index}"]).strip() for index in range(1, 5)]
        if len({normalise_question(option) for option in options}) == 4 and all(options):
            usable.append(row)
        else:
            skipped += 1
    return usable, skipped


def build_belebele(
    cache_dir: Path, lock: Mapping[str, Any], limits: SamplingLimits = DEFAULT_LIMITS
) -> SuiteData:
    """Build five passage pools, the derived retrieval suite and the MCQ suite."""
    version = _source_version(lock, "belebele")
    corpus_rows: list[dict[str, Any]] = []
    retrieval_items: list[dict[str, Any]] = []
    qa_items: list[dict[str, Any]] = []
    skipped_total = 0

    for language, variant in BELEBELE_LANGS.items():
        entry = _source_file(lock, "belebele", language)
        source_path = cache_dir / str(entry["cache_path"])
        rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        passages = _belebele_passages(rows)

        passage_id_by_text: dict[str, str] = {}
        split_by_text: dict[str, str] = {}
        for index, passage in enumerate(passages, start=1):
            passage_id = f"bb-{language}-p{index:04d}"
            numbers = sorted(passage["numbers"])
            split = (
                "dev"
                if int(_rank("bb-passage", language, passage_id)[:8], 16) % BELEBELE_SPLIT_MODULUS
                == 0
                else "test"
            )
            passage_id_by_text[passage["text"]] = passage_id
            split_by_text[passage["text"]] = split
            corpus_rows.append(
                {
                    "passage_id": passage_id,
                    "text": passage["text"],
                    "language": language,
                    "license": SUITE_LICENSES["belebele_retrieval"],
                    "source_link": passage["link"],
                    "question_numbers": numbers,
                    "provenance": {
                        "dataset": "facebook/belebele",
                        "version": version,
                        "source_file": str(entry.get("published_path", "")),
                        "source_sha256": str(entry["sha256"]),
                        "source_item_id": f"{variant}:questions:{numbers[0]}-{numbers[-1]}",
                        "derivation": "flores_passage_deduplicated_from_questions",
                        "modifications": [
                            "one pool entry per unique passage text",
                            "passage_id derived in first-appearance order",
                        ],
                    },
                }
            )

        usable, skipped = _usable_questions(rows)
        skipped_total += skipped
        for split, target in (("test", limits.belebele_test), ("dev", limits.belebele_dev)):
            eligible = [row for row in usable if split_by_text[str(row["flores_passage"])] == split]
            ranked = sorted(
                eligible, key=lambda row: _rank("bb-question", language, row["question_number"])
            )
            if len(ranked) < target:
                raise RuntimeError(
                    f"Belebele {language}/{split}: only {len(ranked)} questions, need {target}"
                )
            chosen = sorted(ranked[:target], key=lambda row: int(row["question_number"]))
            for offset, row in enumerate(chosen, start=1):
                number = int(row["question_number"])
                passage_text = str(row["flores_passage"])
                passage_id = passage_id_by_text[passage_text]
                correct_index = int(row["correct_answer_num"]) - 1
                options = [str(row[f"mc_answer{index}"]) for index in range(1, 5)]
                provenance = {
                    "dataset": "facebook/belebele",
                    "version": version,
                    "source_file": str(entry.get("published_path", "")),
                    "source_sha256": str(entry["sha256"]),
                    "source_item_id": f"{variant}:{number}",
                    "derivation": "gold_passage_question_pair",
                    "source_link": str(row.get("link", "")),
                    "modifications": [
                        "question and options kept verbatim",
                        "gold passage linked by derived passage_id",
                        "split inherited from the gold passage's seeded split",
                    ],
                }
                qa_items.append(
                    {
                        "item_id": f"bb-qa-{language}-{split}-{offset:03d}",
                        "suite": "belebele_qa",
                        "task": "multiple_choice",
                        "language": language,
                        "category": "reading_comprehension",
                        "split": split,
                        "license": SUITE_LICENSES["belebele_qa"],
                        "provenance": provenance,
                        "question": str(row["question"]),
                        "options": options,
                        "answer_index": correct_index,
                        "gold_evidence_ids": [passage_id],
                        "answer_anchor": answer_anchor(options[correct_index], passage_text),
                        "answerable": True,
                    }
                )
                retrieval_items.append(
                    {
                        "item_id": f"bb-ret-{language}-{split}-{offset:03d}",
                        "suite": "belebele_retrieval",
                        "task": "retrieval",
                        "language": language,
                        "category": "derived_retrieval",
                        "split": split,
                        "license": SUITE_LICENSES["belebele_retrieval"],
                        "provenance": {
                            **provenance,
                            "derivation": "derived_qrels_from_belebele_gold_passage",
                            "modifications": [
                                *provenance["modifications"],
                                "relevance label derived: the question's gold passage is "
                                "the only qrel",
                            ],
                        },
                        "query": str(row["question"]),
                        "qrels": [passage_id],
                        "gold_evidence_ids": [passage_id],
                    }
                )

    return SuiteData(
        corpus={"belebele": corpus_rows},
        items={"belebele_retrieval": retrieval_items, "belebele_qa": qa_items},
        stats={"skipped_questions_with_duplicate_options": skipped_total},
    )


# --------------------------------------------------------------------------- Mr. TyDi


def build_mrtydi(cache_dir: Path, lock: Mapping[str, Any]) -> SuiteData:
    """Build the Mr. TyDi ru suite: native qrels over the derived passage pool."""
    pool = read_mrtydi_pool(cache_dir / MRTYDI_POOL_REL)
    corpus_entry = _source_file(lock, "mrtydi-corpus")
    corpus_rows = [
        {
            "passage_id": f"mrtydi-ru:{docid}",
            "text": record["text"],
            "title": record["title"],
            "language": "ru",
            "license": SUITE_LICENSES["mrtydi_retrieval"],
            "provenance": {
                "dataset": "castorini/mr-tydi-corpus",
                "version": _source_version(lock, "mrtydi-corpus"),
                "source_file": str(corpus_entry.get("published_path", "")),
                "source_sha256": str(corpus_entry["sha256"]),
                "source_item_id": docid,
                "derivation": "corpus_passage_extracted_from_archive",
            },
        }
        for docid, record in sorted(pool.items())
    ]

    items: list[dict[str, Any]] = []
    dropped_duplicates = 0
    seen_queries: set[str] = set()
    for split in ("dev", "test"):
        entry = _source_file(lock, "mrtydi", language=split)
        path = cache_dir / str(entry["cache_path"])
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows.sort(key=lambda row: int(row["query_id"]))
        for row in rows:
            key = " ".join(tokens(str(row["query"])))
            if key in seen_queries:
                dropped_duplicates += 1
                continue
            seen_queries.add(key)
            qrels = [f"mrtydi-ru:{passage['docid']}" for passage in row["positive_passages"]]
            missing = [qrel for qrel in qrels if qrel.split(":", 1)[1] not in pool]
            if missing:
                raise RuntimeError(
                    f"Mr. TyDi {split}: qrels missing from the derived pool: {missing[:3]}"
                )
            items.append(
                {
                    "item_id": f"mrtydi-ru-{split}-{int(row['query_id']):05d}",
                    "suite": "mrtydi_retrieval",
                    "task": "retrieval",
                    "language": "ru",
                    "category": "native_qrels",
                    "split": split,
                    "license": SUITE_LICENSES["mrtydi_retrieval"],
                    "provenance": {
                        "dataset": SOURCE_MRTYDI,
                        "version": _source_version(lock, "mrtydi"),
                        "source_file": str(entry.get("published_path", "")),
                        "source_sha256": str(entry["sha256"]),
                        "source_item_id": str(row["query_id"]),
                        "derivation": "native_relevance_judgment",
                        "corpus_dataset": "castorini/mr-tydi-corpus",
                        "corpus_version": _source_version(lock, "mrtydi-corpus"),
                        "corpus_source_sha256": str(corpus_entry["sha256"]),
                    },
                    "query": str(row["query"]),
                    "qrels": qrels,
                    "gold_evidence_ids": list(qrels),
                }
            )

    return SuiteData(
        corpus={"mrtydi": corpus_rows},
        items={"mrtydi_retrieval": items},
        stats={
            "dropped_duplicate_queries": dropped_duplicates,
            "native_queries": len(items) + dropped_duplicates,
        },
    )


# --------------------------------------------------------------------------- domain slice

_FAQ_QUESTION = re.compile(r"^Q:\s*(.+)$")
_FAQ_ANSWER = re.compile(r"^A:\s*(.+)$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+\d[\d\s()\-]{6,}\d")
_ORDER = re.compile(r"([^\n:：]{1,40})[:：]\s*(ORD-\d{4}-\d+)")


def parse_faqs(text: str) -> list[tuple[str, str]]:
    """Extract ``Q:``/``A:`` pairs from an article's FAQ section, in document order."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        question = _FAQ_QUESTION.match(stripped)
        if question:
            pending = question.group(1).strip()
            continue
        answer = _FAQ_ANSWER.match(stripped)
        if answer and pending is not None:
            pairs.append((pending, answer.group(1).strip()))
            pending = None
    return pairs


def contact_values(text: str, language: str) -> dict[str, str]:
    """Reserved contact identifiers and the localized order number of one article."""
    phone_match = _PHONE.search(text)
    email_match = _EMAIL.search(text)
    order_match = _ORDER.search(text)
    return {
        "phone": normalise_text(phone_match.group(0)) if phone_match else PHONE_RANGES[language],
        "email": email_match.group(0) if email_match else RESERVED_EMAIL,
        "order_label": order_match.group(1).strip() if order_match else "",
        "order_id": order_match.group(2) if order_match else "",
    }


def golden_exclusions(
    golden_dir: Path, splits_file: Path
) -> tuple[set[str], set[str], dict[str, str]]:
    """Documents and question texts that t11's golden items use, with file hashes.

    Only documents that appear as evidence of a golden *item* are excluded. The frozen
    split file assigns every corpus document to ``dev`` or ``test`` by construction, so
    excluding split membership would exclude the whole corpus; the disjointness contract is
    at item level (``docs/benchmark.md`` §4.3).
    """
    documents: set[str] = set()
    questions: set[str] = set()
    hashes: dict[str, str] = {}
    candidates = [
        "test.jsonl",
        "dev.jsonl",
        "adversarial/near_miss.jsonl",
        "adversarial/injection.jsonl",
        "adversarial/pii.jsonl",
        "adversarial/perturbation.jsonl",
        "adversarial/claims.jsonl",
    ]
    for relative in candidates:
        path = golden_dir / relative
        if not path.is_file():
            continue
        hashes[path.relative_to(golden_dir.parent.parent.parent).as_posix()] = sha256_file(path)
        for record in iter_jsonl(path):
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            for key in ("doc_id", "document_id"):
                value = metadata.get(key) if isinstance(metadata, Mapping) else None
                if isinstance(value, str) and value:
                    documents.add(value)
            for value in metadata.get("document_ids", []) if isinstance(metadata, Mapping) else []:
                if isinstance(value, str) and value:
                    documents.add(value)
            question = record.get("question")
            if isinstance(question, str) and question.strip():
                questions.add(normalise_question(question))
    if splits_file.is_file():
        hashes[splits_file.relative_to(splits_file.parent.parent.parent).as_posix()] = sha256_file(
            splits_file
        )
    return documents, questions, hashes


def build_domain(
    corpus_dir: Path,
    golden_dir: Path,
    splits_file: Path,
    synthetic: Mapping[str, Any],
    limits: SamplingLimits = DEFAULT_LIMITS,
) -> tuple[SuiteData, dict[str, Any]]:
    """Derive the six-locale support-domain slice and record what was excluded to build it."""
    pinned = {str(entry["repo_path"]): str(entry["sha256"]) for entry in synthetic["files"]}
    queries_rel = "data/corpus/queries/queries.jsonl"
    manifest_rel = "data/corpus/manifest.json"
    queries_path = corpus_dir / "queries" / "queries.jsonl"
    corpus_manifest_path = corpus_dir / "manifest.json"
    for path, relative in ((queries_path, queries_rel), (corpus_manifest_path, manifest_rel)):
        if sha256_file(path) != pinned[relative]:
            raise RuntimeError(
                f"{path} does not match the pinned synthetic corpus hash ({relative})"
            )
    version = str(synthetic["version"])
    queries = list(iter_jsonl(queries_path))
    excluded_docs, excluded_questions, golden_hashes = golden_exclusions(golden_dir, splits_file)

    docs_by_lang: dict[str, dict[str, str]] = {language: {} for language in DOMAIN_LANGS}
    for language in DOMAIN_LANGS:
        for path in sorted((corpus_dir / "docs" / language).glob("*.md")):
            docs_by_lang[language][path.stem] = path.read_text(encoding="utf-8")

    provenance_base = {
        "dataset": "answerdesk-synthetic",
        "version": version,
        "source_file": queries_rel,
        "source_sha256": pinned[queries_rel],
        "corpus_manifest_sha256": pinned[manifest_rel],
    }

    per_language: dict[str, list[dict[str, Any]]] = {}
    for language in DOMAIN_LANGS:
        documents = docs_by_lang[language]
        if not documents:
            raise RuntimeError(f"domain {language}: no corpus documents")
        eligible = {doc_id for doc_id in documents if doc_id not in excluded_docs}
        language_queries = [query for query in queries if query["locale"] == language]
        items = _domain_items(
            language,
            language_queries,
            documents,
            eligible,
            excluded_questions,
            provenance_base,
            limits.domain_quotas,
        )
        per_language[language] = items

    all_items = [item for language in DOMAIN_LANGS for item in per_language[language]]
    exclusions = {
        "corpus_version": str(synthetic["version"]),
        "corpus_manifest_sha256": pinned[manifest_rel],
        "golden_snapshot_sha256": sha256_text(canonical_json(dict(sorted(golden_hashes.items())))),
        "policy": (
            "domain items must not share a document with a t11 golden item "
            "(evals/golden/v1 test, dev and adversarial files); the split file assigns every "
            "corpus document to dev or test by construction, so split membership is recorded "
            "but not used for exclusion"
        ),
        "excluded_document_ids": sorted(excluded_docs),
        "excluded_question_sha256": sorted(
            sha256_text(question) for question in excluded_questions
        ),
        "source_files": golden_hashes,
        "excluded_documents": len(excluded_docs),
        "excluded_questions": len(excluded_questions),
    }
    return SuiteData(corpus={}, items={"domain": all_items}), exclusions


def _domain_items(
    language: str,
    language_queries: Sequence[Mapping[str, Any]],
    documents: Mapping[str, str],
    eligible: set[str],
    excluded_questions: set[str],
    provenance_base: Mapping[str, str],
    quotas: Mapping[str, int] = DOMAIN_QUOTAS,
) -> list[dict[str, Any]]:
    """Select and construct the 40 domain items of one locale (see ``docs/benchmark.md`` §3.5)."""
    used_docs: set[str] = set()
    seen_questions: set[str] = set()
    items: list[dict[str, Any]] = []
    sequence = 0

    def q_docs(query: Mapping[str, Any]) -> set[str]:
        supporting = [str(doc) for doc in query.get("supporting_doc_ids") or [query["doc_id"]]]
        return set(supporting)

    def accept(question: str) -> bool:
        normalised = normalise_question(question)
        if normalised in excluded_questions or normalised in seen_questions:
            return False
        seen_questions.add(normalised)
        return True

    def add(item: dict[str, Any], category: str, position: int, quota: int) -> None:
        nonlocal sequence
        sequence += 1
        split = "dev" if position % DOMAIN_DEV_MODULUS == 0 else "test"
        item["item_id"] = f"dom-{language}-{sequence:04d}"
        item["split"] = split
        items.append(item)

    # --- corpus-backed categories: direct and numeric/date --------------------------
    for category in ("direct", "numeric_date"):
        quota = quotas[category]
        candidates = [
            query
            for query in language_queries
            if query["category"] == category
            and q_docs(query) <= eligible
            and not (q_docs(query) & used_docs)
        ]
        candidates.sort(key=lambda query: _rank("domain", language, category, query["query_id"]))
        picked: list[Mapping[str, Any]] = []
        for query in candidates:
            if not accept(str(query["text"])):
                continue
            if not q_docs(query) <= eligible or q_docs(query) & used_docs:
                continue
            picked.append(query)
            used_docs |= q_docs(query)
            if len(picked) == quota:
                break
        if len(picked) < quota:
            raise RuntimeError(
                f"domain {language}/{category}: found {len(picked)} usable queries, need {quota}"
            )
        for position, query in enumerate(picked):
            doc_id = str(query["doc_id"])
            reference = str(query["operator_reply"])
            _require_verbatim(reference, documents[doc_id], language, str(query["query_id"]))
            add(
                {
                    "suite": "domain",
                    "task": "qa",
                    "language": language,
                    "category": category,
                    "license": SUITE_LICENSES["domain"],
                    "provenance": {
                        **provenance_base,
                        "source_item_id": str(query["query_id"]),
                        "derivation": "corpus_query_row",
                        "modifications": [
                            "question and reference reply kept verbatim",
                            "split assigned by seeded rank",
                        ],
                    },
                    "question": str(query["text"]),
                    "answerable": True,
                    "expected_behavior": "answer_from_evidence",
                    "reference_answer": reference,
                    "evidence": [{"doc_id": doc_id, "quote": reference}],
                    "gold_evidence_ids": [doc_id],
                    "must_not_echo": [],
                    "pii_tokens": [],
                    "attack": None,
                },
                category,
                position,
                quota,
            )

    # --- multi-hop: two FAQ pairs of one article ------------------------------------
    quota = quotas["multi_hop"]
    faq_docs = sorted(
        (
            doc_id
            for doc_id in eligible
            if doc_id not in used_docs and len(parse_faqs(documents[doc_id])) >= 2
        ),
        key=lambda doc_id: _rank("domain", language, "multi_hop", doc_id),
    )
    picked_pairs: list[tuple[str, list[tuple[str, str]]]] = []
    for doc_id in faq_docs:
        pairs = parse_faqs(documents[doc_id])
        question = f"{pairs[0][0]} {pairs[1][0]}"
        if not accept(question):
            continue
        picked_pairs.append((doc_id, pairs[:2]))
        used_docs.add(doc_id)
        if len(picked_pairs) == quota:
            break
    if len(picked_pairs) < quota:
        raise RuntimeError(
            f"domain {language}/multi_hop: found {len(picked_pairs)} articles, need {quota}"
        )
    for position, (doc_id, pairs) in enumerate(picked_pairs):
        answer = f"{pairs[0][1]} {pairs[1][1]}"
        _require_verbatim(pairs[0][1], documents[doc_id], language, doc_id)
        _require_verbatim(pairs[1][1], documents[doc_id], language, doc_id)
        add(
            {
                "suite": "domain",
                "task": "qa",
                "language": language,
                "category": "multi_hop",
                "license": SUITE_LICENSES["domain"],
                "provenance": {
                    **provenance_base,
                    "source_item_id": doc_id,
                    "derivation": "two_faq_pairs_of_one_article",
                    "modifications": [
                        "question concatenates two FAQ questions of the same article",
                        "reference answer concatenates their verbatim FAQ answers",
                        "two-evidence shape mirrors the corpus multi_hop construction",
                    ],
                },
                "question": f"{pairs[0][0]} {pairs[1][0]}",
                "answerable": True,
                "expected_behavior": "answer_from_evidence",
                "reference_answer": answer,
                "evidence": [
                    {"doc_id": doc_id, "quote": pairs[0][1]},
                    {"doc_id": doc_id, "quote": pairs[1][1]},
                ],
                "gold_evidence_ids": [doc_id],
                "must_not_echo": [],
                "pii_tokens": [],
                "attack": None,
            },
            "multi_hop",
            position,
            quota,
        )

    # --- PII and prompt injection: template bank applied to a sampled article -------
    security_docs = sorted(
        (
            doc_id
            for doc_id in eligible
            if doc_id not in used_docs and parse_faqs(documents[doc_id])
        ),
        key=lambda doc_id: _rank("domain", language, "template_docs", doc_id),
    )
    cursor = 0
    for category in ("pii", "injection"):
        quota = quotas[category]
        produced = 0
        while produced < quota:
            while cursor < len(security_docs) and security_docs[cursor] in used_docs:
                cursor += 1
            if cursor >= len(security_docs):
                raise RuntimeError(f"domain {language}/{category}: out of unused articles")
            doc_id = security_docs[cursor]
            cursor += 1
            values = contact_values(documents[doc_id], language)
            template_index = produced % 2
            faqs = parse_faqs(documents[doc_id])
            value = ""
            if category == "pii":
                template = PII_TEMPLATES[language][template_index]
                value = values["phone"] if template_index == 0 else values["email"]
                if value not in documents[doc_id]:
                    continue
                answers = [answer for _, answer in faqs if value not in answer]
            else:
                answers = [answer for _, answer in faqs]
            if not answers:
                continue
            answer = answers[0]
            used_docs.add(doc_id)
            _require_verbatim(answer, documents[doc_id], language, doc_id)
            order_ref = (
                f"{values['order_label']}: {values['order_id']}" if values["order_id"] else ""
            )
            if category == "pii":
                question = template.format(phone=values["phone"], email=values["email"])
                if order_ref:
                    question = f"{question} {order_ref}"
                item = {
                    "question": question,
                    "answerable": True,
                    "expected_behavior": "answer_without_echo",
                    "reference_answer": answer,
                    "evidence": [{"doc_id": doc_id, "quote": answer}],
                    "gold_evidence_ids": [doc_id],
                    "must_not_echo": [value],
                    "pii_tokens": [value],
                    "attack": None,
                }
            else:
                template = INJECTION_TEMPLATES[language][template_index]
                question = f"{template} {order_ref}".strip()
                item = {
                    "question": question,
                    "answerable": True,
                    "expected_behavior": "answer_from_evidence",
                    "reference_answer": answer,
                    "evidence": [{"doc_id": doc_id, "quote": answer}],
                    "gold_evidence_ids": [doc_id],
                    "must_not_echo": [TEST_PAN] if TEST_PAN in template else [],
                    "pii_tokens": [],
                    "attack": {
                        "type": "prompt_injection",
                        "instruction": template,
                        "must_not_comply": True,
                    },
                }
            if not accept(str(item["question"])):
                raise RuntimeError(
                    f"domain {language}/{category}: constructed question duplicates an earlier one"
                )
            add(
                {
                    "suite": "domain",
                    "task": "qa",
                    "language": language,
                    "category": category,
                    "license": SUITE_LICENSES["domain"],
                    "provenance": {
                        **provenance_base,
                        "source_item_id": doc_id,
                        "derivation": f"corpus_{category}_template_applied_to_article",
                        "modifications": [
                            "question built from the corpus's localized template bank",
                            "reserved identifiers come from the article's own contact block",
                            "localized order number appended so no two items share a question",
                        ],
                    },
                    **item,
                },
                category,
                produced,
                quota,
            )
            produced += 1

    # --- unanswerable: four distinct out-of-domain questions -----------------------
    quota = quotas["unanswerable"]
    unanswerable = [
        query
        for query in language_queries
        if query["category"] == "unanswerable" and query["answerable"] is False
    ]
    unanswerable.sort(
        key=lambda query: _rank("domain", language, "unanswerable", query["query_id"])
    )
    picked_unanswerable: list[Mapping[str, Any]] = []
    for query in unanswerable:
        if not accept(str(query["text"])):
            continue
        picked_unanswerable.append(query)
        if len(picked_unanswerable) == quota:
            break
    if len(picked_unanswerable) < quota:
        raise RuntimeError(
            f"domain {language}/unanswerable: only {len(picked_unanswerable)} "
            f"distinct questions, need {quota}"
        )
    for position, query in enumerate(picked_unanswerable):
        add(
            {
                "suite": "domain",
                "task": "qa",
                "language": language,
                "category": "unanswerable",
                "license": SUITE_LICENSES["domain"],
                "provenance": {
                    **provenance_base,
                    "source_item_id": str(query["query_id"]),
                    "derivation": "corpus_unanswerable_query",
                    "modifications": [
                        "question kept verbatim",
                        "no evidence by design",
                        "split assigned by seeded rank",
                    ],
                },
                "question": str(query["text"]),
                "answerable": False,
                "expected_behavior": "escalate",
                "reference_answer": "",
                "evidence": [],
                "gold_evidence_ids": [],
                "must_not_echo": [],
                "pii_tokens": [],
                "attack": None,
            },
            "unanswerable",
            position,
            quota,
        )

    return items


def _require_verbatim(quote: str, document: str, language: str, identifier: str) -> None:
    """Every reference answer must be a verbatim slice of its own article."""
    if quote not in document:
        raise RuntimeError(
            f"domain {language}: reference not verbatim in {identifier}: {quote[:80]!r}"
        )


# --------------------------------------------------------------------------- assembly


def _license_for(path: str) -> str:
    if path.startswith("belebele/"):
        return SUITE_LICENSES["belebele_retrieval"]
    if path.startswith("mrtydi/"):
        return SUITE_LICENSES["mrtydi_retrieval"]
    if path.startswith("domain/"):
        return SUITE_LICENSES["domain"]
    return "MIT"


def _suite_paths(
    belebele_items: Mapping[str, list[dict[str, Any]]], domain_languages: Sequence[str]
) -> dict[str, list[str]]:
    belebele_languages = list(BELEBELE_LANGS)
    return {
        "belebele_retrieval": [
            f"belebele/retrieval/{language}/queries.jsonl" for language in belebele_languages
        ],
        "belebele_qa": [f"belebele/qa/{language}.jsonl" for language in belebele_languages],
        "mrtydi_retrieval": ["mrtydi/ru/queries.jsonl"],
        "domain": [f"domain/{language}.jsonl" for language in domain_languages],
    }


def assemble(
    *,
    cache_dir: Path,
    corpus_dir: Path,
    golden_dir: Path,
    splits_file: Path,
    lock_path: Path,
    out_dir: Path,
    lax_cache: bool = False,
    write: bool = True,
    enforce_checks: bool = True,
    limits: SamplingLimits = DEFAULT_LIMITS,
    log: Any = print,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build every suite, verify the invariants and write the release directory."""
    lock = load_lock(lock_path)
    if not lax_cache:
        problems = check_cache(cache_dir, lock)
        if problems:
            raise RuntimeError(
                "download cache is incomplete or corrupt — run --fetch:\n- " + "\n- ".join(problems)
            )
        local_problems = check_local_sources(lock, REPO_ROOT)
        if local_problems:
            raise RuntimeError(
                "local sources no longer match their pins:\n- " + "\n- ".join(local_problems)
            )
    if not (cache_dir / MRTYDI_POOL_REL).is_file():
        derive_mrtydi_pool(cache_dir, lock, log=log)

    synthetic = {record["id"]: record for record in local_source_records(lock)}[
        "answerdesk-synthetic"
    ]
    belebele = build_belebele(cache_dir, lock, limits)
    mrtydi = build_mrtydi(cache_dir, lock)
    domain, exclusions = build_domain(corpus_dir, golden_dir, splits_file, synthetic, limits)

    files: dict[str, bytes] = {}
    for language in BELEBELE_LANGS:
        corpus_rows = [row for row in belebele.corpus["belebele"] if row["language"] == language]
        files[f"belebele/retrieval/{language}/corpus.jsonl"] = _jsonl(corpus_rows)
        query_rows = [
            item for item in belebele.items["belebele_retrieval"] if item["language"] == language
        ]
        files[f"belebele/retrieval/{language}/queries.jsonl"] = _jsonl(query_rows)
    for language in BELEBELE_LANGS:
        files[f"belebele/qa/{language}.jsonl"] = _jsonl(
            [item for item in belebele.items["belebele_qa"] if item["language"] == language]
        )
    files["mrtydi/ru/corpus.jsonl"] = _jsonl(mrtydi.corpus["mrtydi"])
    files["mrtydi/ru/queries.jsonl"] = _jsonl(mrtydi.items["mrtydi_retrieval"])
    for language in DOMAIN_LANGS:
        files[f"domain/{language}.jsonl"] = _jsonl(
            [item for item in domain.items["domain"] if item["language"] == language]
        )

    files["domain/exclusions.json"] = (
        json.dumps(exclusions, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    for cache_relative, release_relative in LICENSE_FILES:
        files[release_relative] = (cache_dir / cache_relative).read_bytes()
    files["LICENSES/MIT.txt"] = (REPO_ROOT / "LICENSE").read_bytes()
    files["schema.json"] = (
        json.dumps(schema.json_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    items, passages, passage_files = read_bench(files)
    report = all_checks(items, passages, passage_files)
    if enforce_checks and not report["ok"]:
        detail = json.dumps(report, ensure_ascii=False)[:2000]
        raise RuntimeError(f"build produced a release that fails its own checks: {detail}")

    file_rows = [
        {
            "path": path,
            "sha256": sha256_bytes(files[path]),
            "bytes": len(files[path]),
            "license": _license_for(path),
            "items": files[path].count(b"\n") if path.endswith(".jsonl") else 0,
        }
        for path in sorted(files)
    ]
    counts = dict(report["counts"])
    counts["passages"] = {
        "belebele_retrieval": sum(1 for row in belebele.corpus["belebele"]),
        "mrtydi_retrieval": len(mrtydi.corpus["mrtydi"]),
    }
    counts["languages"] = sorted({str(item["language"]) for item in items})

    manifest: dict[str, Any] = {
        "benchmark": BENCH_NAME,
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "release_date": RELEASE_DATE,
        "seed": SEED,
        "languages": counts["languages"],
        "suites": {
            name: {
                "task": spec["task"],
                "license": spec["license"],
                "languages": list(spec["languages"]),
                "categories": list(spec["categories"]),
                "notes": spec["notes"],
                "paths": _suite_paths(belebele.items, DOMAIN_LANGS)[name],
                "items": report["counts"]["by_suite"].get(name, 0),
            }
            for name, spec in schema.SUITES.items()
        },
        "counts": counts,
        "sampling": {
            "seed": SEED,
            "belebele": {
                "queries_per_language": limits.belebele_test + limits.belebele_dev,
                "test_per_language": limits.belebele_test,
                "dev_per_language": limits.belebele_dev,
                "passage_split": (
                    f"one passage in {BELEBELE_SPLIT_MODULUS} is dev, "
                    "by sha256(seed, language, passage_id)"
                ),
                "question_split": "a question inherits the split of its gold passage",
                "pool": "every unique FLORES passage of the language variant",
            },
            "mrtydi": {
                "sampling": "none — the complete ru dev and test splits",
                "pool": "union of the positive passages of those queries",
            },
            "domain": {
                "quotas": dict(limits.domain_quotas),
                "dev_fraction": (
                    f"one item in {DOMAIN_DEV_MODULUS}, by seeded rank within each category"
                ),
                "unused_documents_only": True,
            },
            "minhash": {
                "hashes": MINHASH_HASHES,
                "bands": MINHASH_BANDS,
                "threshold": MINHASH_THRESHOLD,
                "shingle_size": SHINGLE_SIZE,
            },
        },
        "splits": {
            "policy": report["splits"]["policy"],
            "ids": report["splits"]["ids_by_split"],
            "domain_documents": report["splits"]["documents_by_split"],
            "t11_disjointness": (
                "domain documents exclude every document that appears as evidence of "
                "a t11 golden item (evals/golden/v1 test, dev and adversarial); the "
                "split assigns all corpus documents by construction, so split "
                "membership is recorded but not used for exclusion"
            ),
        },
        "license_map": {
            "belebele/": "CC-BY-SA-4.0",
            "mrtydi/": "Apache-2.0",
            "domain/": "CC0-1.0",
            "schema.json": "MIT",
            "LICENSES/MIT.txt": "MIT",
        },
        "sources": [*source_records(lock), *local_source_records(lock)],
        "files": file_rows,
        "checks": report,
        "adapter": {
            "version": "1",
            "corpus_files": {
                "belebele_retrieval": "belebele/retrieval/<language>/corpus.jsonl",
                "mrtydi_retrieval": "mrtydi/ru/corpus.jsonl",
            },
            "item_files": _suite_paths(belebele.items, DOMAIN_LANGS),
            "query_field": "query",
            "question_field": "question",
            "qrels_field": "qrels",
            "evidence_field": "gold_evidence_ids",
            "split_field": "split",
            "language_field": "language",
            "license_field": "license",
            "domain_note": (
                "domain items cite data/corpus document ids (gold_evidence_ids, "
                "evidence[].doc_id) and carry verbatim evidence quotes; they are not "
                "part of the retrieval passage pools"
            ),
        },
    }
    manifest["data_sha256"] = sha256_text(
        canonical_json([[row["path"], row["sha256"]] for row in file_rows])
    )
    manifest["self_sha256"] = ""
    manifest["self_sha256"] = sha256_text(canonical_json(manifest))
    files["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    if write:
        _write_tree(out_dir, files, log=log)
    return files, manifest


def _write_tree(out_dir: Path, files: Mapping[str, bytes], log: Any = print) -> None:
    """Write the release files, pruning stale generated files but never hand-written docs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_roots = ("belebele", "mrtydi", "domain", "LICENSES")
    for root in generated_roots:
        root_dir = out_dir / root
        if not root_dir.exists():
            continue
        for existing in sorted(root_dir.rglob("*")):
            if not existing.is_file():
                continue
            relative = existing.relative_to(out_dir).as_posix()
            if relative not in files and existing.suffix in (".jsonl", ".json", ".txt"):
                existing.unlink()
                log(f"pruned  {relative}")
    for relative in sorted(files):
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(files[relative])
    log(f"wrote   {len(files)} files to {out_dir}")
