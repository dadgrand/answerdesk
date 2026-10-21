"""The transcreated fact set: parallel facts from corpus v2.1, one per locale.

A *fact* here is a (document, question) pair that exists in every locale: the corpus renders
the same archetype and document index in each language with its own locale-specific slot draw.
The set therefore tests **fact-level consistency across languages** (same question, same
archetype, same slot layout, each answered from its own locale's document) and not literal
value equality — values legitimately differ per locale because each document is its own
instance. Value fidelity is checked against each locale's *own* gold document, which is the
honest formulation of "the same fact" for this corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCALES = ("ru", "uz", "tg", "ky", "zh", "tr")
MIN_FACTS = 50
SET_VERSION = "v1"
#: archetypes that carry concrete values (amounts, dates, IDs) worth checking
VALUE_ARCHETYPES = (
    "tariff_schedule",
    "fee_table",
    "loan_credit",
    "deposit_savings",
    "remittance_tracking",
    "transfer_limits",
    "card_terms",
)
_FACT_ID = re.compile(r"syn-(?P<locale>[a-z]{2})-(?P<index>\d{4})")


@dataclass(frozen=True)
class FactInstance:
    locale: str
    query_id: str
    doc_id: str
    archetype: str
    question: str
    reference_answer: str
    gold_doc_ids: tuple[str, ...]
    style: str


@dataclass(frozen=True)
class FactGroup:
    fact_id: str
    archetype: str
    doc_index: str
    instances: dict[str, FactInstance]

    @property
    def locales_present(self) -> list[str]:
        return sorted(self.instances)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "archetype": self.archetype,
            "doc_index": self.doc_index,
            "locales": self.locales_present,
            "instances": {
                locale: instance_to_dict(inst) for locale, inst in self.instances.items()
            },
        }


def instance_to_dict(instance: FactInstance) -> dict[str, Any]:
    return {
        "locale": instance.locale,
        "query_id": instance.query_id,
        "doc_id": instance.doc_id,
        "archetype": instance.archetype,
        "question": instance.question,
        "reference_answer": instance.reference_answer,
        "gold_doc_ids": list(instance.gold_doc_ids),
        "style": instance.style,
    }


def _load_queries(corpus_dir: Path) -> list[dict[str, Any]]:
    path = corpus_dir / "queries" / "queries.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"queries file is missing: {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _index_of(doc_id: str) -> str:
    match = _FACT_ID.search(doc_id)
    return match.group("index") if match else doc_id


def build_groups(
    corpus_dir: Path, *, limit: int | None = None, locales: Sequence[str] = LOCALES
) -> list[FactGroup]:
    """One clean direct question per (archetype, document index) in every locale."""
    queries = _load_queries(corpus_dir)
    by_key: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in queries:
        if not row.get("answerable") or row.get("style"):
            continue
        locale = str(row.get("locale") or "")
        if locale not in locales:
            continue
        archetype = row.get("archetype") or row.get("intent") or "unknown"
        if archetype not in VALUE_ARCHETYPES:
            continue
        key = (archetype, _index_of(str(row.get("doc_id", ""))))
        by_key.setdefault(key, {}).setdefault(locale, []).append(dict(row))

    groups: list[FactGroup] = []
    for (archetype, doc_index), per_locale in sorted(by_key.items()):
        if len(per_locale) < len(locales):
            continue
        # prefer the direct question; fall back to any clean answerable question of the doc
        chosen: dict[str, dict[str, Any]] = {
            locale: min(
                rows,
                key=lambda row: (
                    0 if row.get("category") == "direct" else 1,
                    str(row.get("category")),
                ),
            )
            for locale, rows in per_locale.items()
        }
        instances = {
            locale: FactInstance(
                locale=locale,
                query_id=row["query_id"],
                doc_id=row["doc_id"],
                archetype=archetype,
                question=row["base_text"] or row["text"],
                reference_answer=row.get("operator_reply") or "",
                gold_doc_ids=tuple(sorted({row["doc_id"], *(row.get("supporting_doc_ids") or [])})),
                style=row.get("style") or "clean",
            )
            for locale, row in chosen.items()
        }
        groups.append(
            FactGroup(
                fact_id=f"fact-{archetype}-{doc_index}",
                archetype=archetype,
                doc_index=doc_index,
                instances=instances,
            )
        )
    # deterministic, archetype-balanced selection so every archetype is represented
    ordered: list[FactGroup] = []
    buckets: dict[str, list[FactGroup]] = {}
    for group in groups:
        buckets.setdefault(group.archetype, []).append(group)
    position = 0
    while len(ordered) < (limit or len(groups)):
        added = False
        for archetype in sorted(buckets):
            bucket = buckets[archetype]
            if position < len(bucket) and len(ordered) < (limit or len(groups)):
                ordered.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return ordered


def write_set(groups: Iterable[FactGroup], out_dir: Path, *, corpus_version: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [group.to_dict() for group in groups]
    facts_path = out_dir / "facts.jsonl"
    facts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    digest = hashlib.sha256(facts_path.read_bytes()).hexdigest()
    manifest = {
        "set_version": SET_VERSION,
        "corpus_version": corpus_version,
        "facts": len(rows),
        "locales": sorted({locale for row in rows for locale in row["locales"]}),
        "archetypes": sorted({row["archetype"] for row in rows}),
        "instances": sum(len(row["instances"]) for row in rows),
        "facts_sha256": digest,
        "definition": (
            "a fact is (archetype, document index) asked as the clean direct question in every "
            "locale; each locale answers from its own document, so value equality across "
            "languages is not assumed — value fidelity is checked against each locale's gold"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_set(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


__all__ = [
    "LOCALES",
    "MIN_FACTS",
    "SET_VERSION",
    "VALUE_ARCHETYPES",
    "FactGroup",
    "FactInstance",
    "build_groups",
    "load_set",
    "write_set",
]
