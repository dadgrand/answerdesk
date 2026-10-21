"""Recompute the benchmark's invariants from the files themselves.

The builder and the verifier call the *same* functions, so ``manifest.json#checks`` is not
a claim written next to the data — it is the output of these functions over the bytes that
ship. ``--verify`` recomputes them and additionally enforces the acceptance floors
(``docs/benchmark.md`` §5).

All checks are pure and offline: they read the release's JSONL files, never the cache and
never the network.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from . import schema
from .config import (
    MIN_DOMAIN_ITEMS_PER_CATEGORY,
    MIN_QA_ITEMS_PER_LANG,
    MIN_RETRIEVAL_QUERIES_PER_LANG,
    MIN_SCRIPT_SHARE,
    MIN_TOTAL_ITEMS,
    MINHASH_BANDS,
    MINHASH_HASHES,
    MINHASH_THRESHOLD,
    SHINGLE_SIZE,
)
from .normalize import answer_anchor, near_duplicate_pairs, script_share

MAX_REPORTED = 10


def read_bench(
    files: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Split the release files into items, passages and passages grouped by file."""
    items: list[dict[str, Any]] = []
    passages: dict[str, dict[str, Any]] = {}
    passage_files: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(files):
        if not path.endswith(".jsonl"):
            continue
        payload = files[path].decode("utf-8")
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if path.endswith("/corpus.jsonl"):
            passage_files[path] = rows
            for row in rows:
                passages[str(row["passage_id"])] = row
        else:
            items.extend(rows)
    return items, passages, passage_files


def _text(item: Mapping[str, Any]) -> str:
    value = item.get("question") if "question" in item else item.get("query", "")
    return str(value)


def item_checks(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for item in items:
        errors.extend(schema.validate_item(item))
    duplicates = [
        item_id
        for item_id, count in Counter(str(item.get("item_id")) for item in items).items()
        if count > 1
    ]
    return {
        "ok": not errors and not duplicates,
        "count": len(items),
        "schema_errors": errors[:MAX_REPORTED],
        "schema_error_count": len(errors),
        "duplicate_item_ids": duplicates[:MAX_REPORTED],
    }


def count_checks(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_suite = Counter(str(item["suite"]) for item in items)
    by_language = Counter(str(item["language"]) for item in items)
    by_category = Counter(str(item["category"]) for item in items)
    by_split = Counter(str(item["split"]) for item in items)
    retrieval = by_suite["belebele_retrieval"] + by_suite["mrtydi_retrieval"]
    return {
        "ok": True,
        "items": len(items),
        "retrieval_queries": retrieval,
        "qa_items": by_suite["belebele_qa"],
        "domain_items": by_suite["domain"],
        "by_suite": dict(sorted(by_suite.items())),
        "by_language": dict(sorted(by_language.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_split": dict(sorted(by_split.items())),
    }


def floor_checks(items: Sequence[Mapping[str, Any]], counts: Mapping[str, Any]) -> dict[str, Any]:
    """Acceptance floors from the task contract and ``docs/requirements.md`` FR-EVL-01."""
    problems: list[str] = []
    if int(counts["items"]) < MIN_TOTAL_ITEMS:
        problems.append(f"only {counts['items']} items, need >= {MIN_TOTAL_ITEMS}")
    for suite, _label, floor in (
        ("belebele_retrieval", "retrieval", MIN_RETRIEVAL_QUERIES_PER_LANG),
        ("belebele_qa", "qa", MIN_QA_ITEMS_PER_LANG),
    ):
        per_language = Counter(str(item["language"]) for item in items if item["suite"] == suite)
        for language, count in sorted(per_language.items()):
            if count < floor:
                problems.append(f"{suite}/{language}: {count} items, need >= {floor}")
        for language in ("ru", "uz", "tg", "ky", "zh"):
            if language not in per_language:
                problems.append(f"{suite}: no items for {language}")
    domain_items = counts["domain_items"]
    if domain_items < 200:
        problems.append(f"domain slice has {domain_items} items, need >= 200")
    for category, count in sorted(domain_category_counts(items).items()):
        if count < MIN_DOMAIN_ITEMS_PER_CATEGORY:
            problems.append(f"domain category {category!r} has {count} items")
    return {"ok": not problems, "problems": problems[:MAX_REPORTED]}


def passage_checks(
    items: Sequence[Mapping[str, Any]], passages: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    problems: list[str] = []
    referenced = 0
    for item in items:
        identifiers: Iterable[str] = []
        if item["suite"] == "belebele_retrieval" or item["suite"] == "mrtydi_retrieval":
            identifiers = item.get("qrels", [])
        elif item["suite"] == "belebele_qa":
            identifiers = item.get("gold_evidence_ids", [])
        for identifier in identifiers:
            referenced += 1
            if identifier not in passages:
                problems.append(f"{item['item_id']}: unknown passage {identifier!r}")
    per_file = Counter()
    for passage in passages.values():
        per_file[str(passage["language"])] += 1
    return {
        "ok": not problems,
        "passages": len(passages),
        "references_checked": referenced,
        "by_language": dict(sorted(per_file.items())),
        "problems": problems[:MAX_REPORTED],
    }


def answer_checks(
    items: Sequence[Mapping[str, Any]], passages: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    problems: list[str] = []
    checked = 0

    for item in items:
        suite = item["suite"]
        if suite == "belebele_qa":
            checked += 1
            evidence = [str(identifier) for identifier in item["gold_evidence_ids"]]
            if len(evidence) != 1 or evidence[0] not in passages:
                problems.append(f"{item['item_id']}: gold evidence does not resolve")
                continue
            if not 0 <= int(item["answer_index"]) < len(item["options"]):
                problems.append(f"{item['item_id']}: answer_index outside the option list")
                continue
            recomputed = answer_anchor(
                str(item["options"][item["answer_index"]]), str(passages[evidence[0]]["text"])
            )
            if abs(recomputed - float(item["answer_anchor"])) > 1e-6:
                problems.append(
                    f"{item['item_id']}: answer_anchor {item['answer_anchor']} != recomputed {recomputed}"
                )
        elif suite == "domain":
            checked += 1
            quotes = [str(entry["quote"]) for entry in item.get("evidence", [])]
            reference = str(item.get("reference_answer", ""))
            if item.get("answerable"):
                if item["category"] == "multi_hop":
                    if reference != " ".join(quotes):
                        problems.append(
                            f"{item['item_id']}: multi_hop reference is not the quote concatenation"
                        )
                elif quotes and reference != quotes[0]:
                    problems.append(
                        f"{item['item_id']}: reference answer is not the evidence quote"
                    )
                if not any(str(gold) for gold in item.get("gold_evidence_ids", [])):
                    problems.append(f"{item['item_id']}: answerable item without gold evidence")
            for token in item.get("must_not_echo", []):
                if str(token) in reference:
                    problems.append(
                        f"{item['item_id']}: reference echoes forbidden token {token!r}"
                    )
            if item["category"] == "pii" and not any(
                str(token) in str(item["question"]) for token in item.get("pii_tokens", [])
            ):
                problems.append(f"{item['item_id']}: pii token absent from the question")
            if item["category"] == "injection":
                attack = item.get("attack") or {}
                if str(attack.get("instruction", "")) not in str(item["question"]):
                    problems.append(
                        f"{item['item_id']}: attack instruction absent from the question"
                    )
    return {"ok": not problems, "checked": checked, "problems": problems[:MAX_REPORTED]}


def dedup_checks(
    items: Sequence[Mapping[str, Any]],
    passage_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    duplicates: list[dict[str, Any]] = []
    groups = 0

    grouped_items: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped_items[(str(item["suite"]), str(item["language"]))].append(item)
    for (suite, language), members in sorted(grouped_items.items()):
        groups += 1
        texts = [_text(member) for member in members]
        for first, second, jaccard in near_duplicate_pairs(
            texts,
            threshold=MINHASH_THRESHOLD,
            num_hashes=MINHASH_HASHES,
            bands=MINHASH_BANDS,
            size=SHINGLE_SIZE,
        ):
            duplicates.append(
                {
                    "a": str(members[first]["item_id"]),
                    "b": str(members[second]["item_id"]),
                    "jaccard": jaccard,
                    "suite": suite,
                    "language": language,
                }
            )

    for path, rows in sorted(passage_files.items()):
        groups += 1
        texts = [str(row["text"]) for row in rows]
        for first, second, jaccard in near_duplicate_pairs(
            texts,
            threshold=MINHASH_THRESHOLD,
            num_hashes=MINHASH_HASHES,
            bands=MINHASH_BANDS,
            size=SHINGLE_SIZE,
        ):
            duplicates.append(
                {
                    "a": str(rows[first]["passage_id"]),
                    "b": str(rows[second]["passage_id"]),
                    "jaccard": jaccard,
                    "file": path,
                }
            )

    return {
        "ok": not duplicates,
        "threshold": MINHASH_THRESHOLD,
        "hashes": MINHASH_HASHES,
        "bands": MINHASH_BANDS,
        "shingle_size": SHINGLE_SIZE,
        "groups": groups,
        "duplicates": duplicates[:MAX_REPORTED],
        "duplicate_count": len(duplicates),
    }


def script_checks(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    minimum_by_language: dict[str, float] = {}
    for item in items:
        language = str(item["language"])
        text = _text(item)
        expected = {
            "ru": "Cyrillic",
            "ky": "Cyrillic",
            "tg": "Cyrillic",
            "uz": "Latin",
            "zh": "Han",
            "tr": "Latin",
        }[language]
        share = script_share(text, expected)
        minimum_by_language[language] = min(minimum_by_language.get(language, 1.0), share)
        if share < MIN_SCRIPT_SHARE:
            problems.append(f"{item['item_id']}: {language} text is only {share:.2f} {expected}")
    return {
        "ok": not problems,
        "minimum_share_by_language": dict(sorted(minimum_by_language.items())),
        "problems": problems[:MAX_REPORTED],
    }


def split_checks(
    items: Sequence[Mapping[str, Any]],
    passages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    problems: list[str] = []
    splits_by_passage: dict[str, set[str]] = defaultdict(set)
    docs_by_split: dict[str, set[str]] = defaultdict(set)
    ids_by_split: dict[str, set[str]] = defaultdict(set)

    for item in items:
        ids_by_split[str(item["split"])].add(str(item["item_id"]))
        if item["suite"] in ("belebele_retrieval", "mrtydi_retrieval"):
            evidence = list(item.get("qrels", []))
        elif item["suite"] == "belebele_qa":
            evidence = list(item.get("gold_evidence_ids", []))
        else:
            evidence = list(item.get("gold_evidence_ids", []))
        for identifier in evidence:
            splits_by_passage[str(identifier)].add(str(item["split"]))
        if item["suite"] == "domain":
            for doc_id in item.get("gold_evidence_ids", []):
                docs_by_split[str(item["split"])].add(str(doc_id))

    if ids_by_split["dev"] & ids_by_split["test"]:
        problems.append("an item id is present in both dev and test")
    for passage_id, splits in sorted(splits_by_passage.items()):
        if len(splits) > 1 and passage_id.startswith("bb-"):
            problems.append(f"passage {passage_id} is gold evidence in both splits")
    for doc_id in sorted(docs_by_split["dev"] & docs_by_split["test"]):
        problems.append(f"domain document {doc_id} appears in both splits")

    return {
        "ok": not problems,
        "policy": "seeded hash split; a passage (Belebele) or document (domain) is gold evidence in exactly one split",
        "ids_by_split": {split: len(values) for split, values in sorted(ids_by_split.items())},
        "documents_by_split": {
            split: len(values) for split, values in sorted(docs_by_split.items())
        },
        "problems": problems[:MAX_REPORTED],
    }


def domain_category_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(item["category"]) for item in items if item["suite"] == "domain").items()
        )
    )


def all_checks(
    items: Sequence[Mapping[str, Any]],
    passages: Mapping[str, Mapping[str, Any]],
    passage_files: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Every file-level invariant, in one JSON-serialisable report."""
    counts = count_checks(items)
    report = {
        "items": item_checks(items),
        "counts": counts,
        "floors": floor_checks(items, counts),
        "passages": passage_checks(items, passages),
        "answers": answer_checks(items, passages),
        "dedup": dedup_checks(items, passage_files),
        "scripts": script_checks(items),
        "splits": split_checks(items, passages),
    }
    report["ok"] = all(
        section.get("ok", False) for section in report.values() if isinstance(section, dict)
    )
    return report
