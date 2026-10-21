"""The item contract for ``answerdesk-bench``.

One declarative spec drives three things: the builder writes items that satisfy it, the
verifier rejects committed items that do not, and ``schema.json`` (generated into the
release directory) documents it for external readers. Keeping the spec in code means the
three cannot drift (``docs/benchmark.md`` §4).

Item shape: every item has the same core envelope, then a suite-specific payload. The
core envelope is what t19 needs to route an item without knowing its suite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

SCHEMA_ID: Final = "https://answerdesk.dev/bench/answerdesk-bench/v0.1/item.schema.json"

CORE_FIELDS: Final = (
    "item_id",
    "suite",
    "task",
    "language",
    "category",
    "split",
    "license",
    "provenance",
)

PROVENANCE_FIELDS: Final = (
    "dataset",
    "version",
    "source_file",
    "source_sha256",
    "source_item_id",
    "derivation",
)

SUITES: Final = {
    "belebele_retrieval": {
        "task": "retrieval",
        "license": "CC-BY-SA-4.0",
        "languages": ("ru", "uz", "tg", "ky", "zh"),
        "categories": ("derived_retrieval",),
        "directory": "belebele/retrieval",
        "notes": (
            "Retrieval labels derived from Belebele reading-comprehension triples "
            "(question, gold passage, one passage pool per language). Diagnostic: the "
            "question was written against a supplied passage, so it contains deictic "
            "references and is not an ad-hoc customer query."
        ),
    },
    "belebele_qa": {
        "task": "multiple_choice",
        "license": "CC-BY-SA-4.0",
        "languages": ("ru", "uz", "tg", "ky", "zh"),
        "categories": ("reading_comprehension",),
        "directory": "belebele/qa",
        "notes": (
            "Belebele four-option reading-comprehension items; the gold passage is the "
            "evidence and the labelled option is the answer."
        ),
    },
    "mrtydi_retrieval": {
        "task": "retrieval",
        "license": "Apache-2.0",
        "languages": ("ru",),
        "categories": ("native_qrels",),
        "directory": "mrtydi/ru",
        "notes": (
            "Native relevance judgments from Mr. TyDi v1.1 (dev + test); the passage pool "
            "is the union of the positive passages of those queries."
        ),
    },
    "domain": {
        "task": "qa",
        "license": "CC0-1.0",
        "languages": ("ru", "uz", "tg", "ky", "zh", "tr"),
        "categories": ("direct", "numeric_date", "multi_hop", "pii", "injection", "unanswerable"),
        "directory": "domain",
        "notes": (
            "Support-domain slice derived from the synthetic CC0 corpus: grounded answers, "
            "numeric/date, multi-evidence, unanswerable, PII and prompt-injection cases."
        ),
    },
}

SPLITS: Final = ("dev", "test")
LICENSES: Final = ("CC-BY-SA-4.0", "Apache-2.0", "CC0-1.0")
EXPECTED_BEHAVIOURS: Final = ("answer_from_evidence", "escalate", "answer_without_echo")
DOMAIN_CATEGORIES: Final = SUITES["domain"]["categories"]


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_id_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_text(item) for item in value)


def validate_item(item: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable violations; empty means the item conforms."""
    errors: list[str] = []
    label = str(item.get("item_id", "<no item_id>"))

    for field in CORE_FIELDS:
        _require(field in item, f"{label}: missing core field {field!r}", errors)
    if errors:
        return errors

    suite = item["suite"]
    if suite not in SUITES:
        return [f"{label}: unknown suite {suite!r}"]
    spec = SUITES[suite]
    _require(item["task"] == spec["task"], f"{label}: task must be {spec['task']!r}", errors)
    _require(
        item["license"] == spec["license"], f"{label}: license must be {spec['license']!r}", errors
    )
    _require(
        item["language"] in spec["languages"], f"{label}: language not allowed for {suite}", errors
    )
    _require(
        item["category"] in spec["categories"], f"{label}: category not allowed for {suite}", errors
    )
    _require(item["split"] in SPLITS, f"{label}: split must be one of {SPLITS}", errors)
    _require(_is_text(item["item_id"]), f"{label}: item_id must be a non-empty string", errors)

    provenance = item["provenance"]
    if not isinstance(provenance, Mapping):
        return [*errors, f"{label}: provenance must be an object"]
    for field in PROVENANCE_FIELDS:
        _require(_is_text(provenance.get(field)), f"{label}: provenance.{field} missing", errors)
    source_hash = provenance.get("source_sha256")
    if _is_text(source_hash):
        hex_digits = set("0123456789abcdef")
        ok = len(str(source_hash)) == 64 and set(str(source_hash)) <= hex_digits
        _require(ok, f"{label}: provenance.source_sha256 must be 64 lowercase hex chars", errors)

    if suite in ("belebele_retrieval", "mrtydi_retrieval"):
        _require(_is_text(item.get("query")), f"{label}: query missing", errors)
        _require(
            _is_id_list(item.get("qrels")), f"{label}: qrels must be a non-empty id list", errors
        )

    if suite == "belebele_qa":
        options = item.get("options")
        _require(_is_text(item.get("question")), f"{label}: question missing", errors)
        _require(
            isinstance(options, list) and len(options) == 4,
            f"{label}: need exactly 4 options",
            errors,
        )
        if isinstance(options, list) and len(options) == 4:
            distinct = {str(option).strip().casefold() for option in options}
            _require(len(distinct) == 4, f"{label}: options must be distinct", errors)
            _require(all(_is_text(option) for option in options), f"{label}: empty option", errors)
        answer_index = item.get("answer_index")
        _require(
            isinstance(answer_index, int)
            and not isinstance(answer_index, bool)
            and 0 <= answer_index <= 3,
            f"{label}: answer_index must be an int in 0..3",
            errors,
        )
        _require(
            _is_id_list(item.get("gold_evidence_ids")),
            f"{label}: gold_evidence_ids missing",
            errors,
        )
        _require(item.get("answerable") is True, f"{label}: qa items are answerable", errors)
        anchor = item.get("answer_anchor")
        _require(
            isinstance(anchor, (int, float))
            and not isinstance(anchor, bool)
            and 0.0 <= float(anchor) <= 1.0,
            f"{label}: answer_anchor must be a float in 0..1",
            errors,
        )

    if suite == "domain":
        _validate_domain_item(item, label, errors)

    return errors


def _validate_domain_item(item: Mapping[str, Any], label: str, errors: list[str]) -> None:
    category = item.get("category")
    answerable = item.get("answerable")
    behaviour = item.get("expected_behavior")
    reference = item.get("reference_answer")
    evidence = item.get("evidence")
    gold = item.get("gold_evidence_ids")

    _require(_is_text(item.get("question")), f"{label}: question missing", errors)
    _require(isinstance(answerable, bool), f"{label}: answerable must be bool", errors)
    _require(behaviour in EXPECTED_BEHAVIOURS, f"{label}: bad expected_behavior", errors)
    _require(isinstance(reference, str), f"{label}: reference_answer must be a string", errors)
    _require(isinstance(evidence, list), f"{label}: evidence must be a list", errors)
    _require(isinstance(gold, list), f"{label}: gold_evidence_ids must be a list", errors)
    _require(
        isinstance(item.get("must_not_echo"), list),
        f"{label}: must_not_echo must be a list",
        errors,
    )
    _require(
        isinstance(item.get("pii_tokens"), list), f"{label}: pii_tokens must be a list", errors
    )

    if isinstance(evidence, list):
        for entry in evidence:
            if not isinstance(entry, Mapping):
                errors.append(f"{label}: evidence entries must be objects")
                break
            if not (_is_text(entry.get("doc_id")) and _is_text(entry.get("quote"))):
                errors.append(f"{label}: evidence entry needs doc_id and quote")
                break

    if answerable is False:
        _require(
            behaviour == "escalate", f"{label}: unanswerable items must expect escalation", errors
        )
        _require(reference == "", f"{label}: unanswerable items carry no reference answer", errors)
        _require(evidence == [], f"{label}: unanswerable items carry no evidence", errors)
        _require(gold == [], f"{label}: unanswerable items carry no gold evidence ids", errors)
        _require(
            category == "unanswerable", f"{label}: only unanswerable items are unanswerable", errors
        )
    elif answerable is True:
        _require(_is_text(reference), f"{label}: answerable item needs a reference answer", errors)
        _require(bool(gold), f"{label}: answerable item needs gold evidence ids", errors)

    if category == "multi_hop":
        _require(
            isinstance(evidence, list) and len(evidence) >= 2,
            f"{label}: multi_hop needs 2+ evidence entries",
            errors,
        )
    if category in ("direct", "numeric_date"):
        _require(
            isinstance(evidence, list) and len(evidence) >= 1,
            f"{label}: needs an evidence quote",
            errors,
        )
    if category == "pii":
        _require(
            item.get("pii_tokens") not in ([], None), f"{label}: pii item needs pii_tokens", errors
        )
        _require(
            item.get("must_not_echo") not in ([], None),
            f"{label}: pii item needs must_not_echo",
            errors,
        )
        _require(
            behaviour == "answer_without_echo",
            f"{label}: pii items expect answer_without_echo",
            errors,
        )
    if category == "injection":
        attack = item.get("attack")
        _require(
            isinstance(attack, Mapping) and _is_text(attack.get("instruction")),
            f"{label}: injection needs attack.instruction",
            errors,
        )
        if isinstance(attack, Mapping):
            _require(
                attack.get("must_not_comply") is True,
                f"{label}: injection needs attack.must_not_comply",
                errors,
            )
        _require(
            behaviour == "answer_from_evidence",
            f"{label}: injection items expect answer_from_evidence",
            errors,
        )


def item_fields(suite: str) -> tuple[str, ...]:
    """Required fields for a suite (core envelope plus payload)."""
    payload = {
        "belebele_retrieval": ("query", "qrels", "gold_evidence_ids"),
        "mrtydi_retrieval": ("query", "qrels", "gold_evidence_ids"),
        "belebele_qa": (
            "question",
            "options",
            "answer_index",
            "gold_evidence_ids",
            "answer_anchor",
            "answerable",
        ),
        "domain": (
            "question",
            "answerable",
            "expected_behavior",
            "reference_answer",
            "evidence",
            "gold_evidence_ids",
            "must_not_echo",
            "pii_tokens",
            "attack",
        ),
    }[suite]
    return (*CORE_FIELDS, *payload)


def json_schema() -> dict[str, Any]:
    """A JSON Schema document describing the union of the four item kinds.

    Hand-built from the same tables ``validate_item`` uses; the verifier checks that the
    on-disk copy of this document equals a fresh generation.
    """
    string = {"type": "string", "minLength": 1}
    id_list = {"type": "array", "minItems": 1, "items": string}
    provenance = {
        "type": "object",
        "required": list(PROVENANCE_FIELDS),
        "additionalProperties": True,
        "properties": {field: string for field in PROVENANCE_FIELDS},
    }
    core = {
        "item_id": string,
        "suite": {"type": "string", "enum": list(SUITES)},
        "task": string,
        "language": {"type": "string", "enum": list(SUITES["domain"]["languages"])},
        "category": string,
        "split": {"type": "string", "enum": list(SPLITS)},
        "license": {"type": "string", "enum": list(LICENSES)},
        "provenance": provenance,
    }

    def variant(suite: str, payload: dict[str, Any], required: list[str]) -> dict[str, Any]:
        properties = dict(core)
        properties.update(payload)
        properties["suite"] = {"const": suite}
        properties["task"] = {"const": SUITES[suite]["task"]}
        properties["license"] = {"const": SUITES[suite]["license"]}
        properties["category"] = {"type": "string", "enum": list(SUITES[suite]["categories"])}
        properties["language"] = {"type": "string", "enum": list(SUITES[suite]["languages"])}
        return {
            "type": "object",
            "required": required,
            "additionalProperties": True,
            "properties": properties,
        }

    evidence = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["doc_id", "quote"],
            "additionalProperties": False,
            "properties": {"doc_id": string, "quote": string},
        },
    }
    retrieval_required = [*CORE_FIELDS, "query", "qrels", "gold_evidence_ids"]
    qa_required = [
        *CORE_FIELDS,
        "question",
        "options",
        "answer_index",
        "gold_evidence_ids",
        "answer_anchor",
        "answerable",
    ]
    domain_required = [
        *CORE_FIELDS,
        "question",
        "answerable",
        "expected_behavior",
        "reference_answer",
        "evidence",
        "gold_evidence_ids",
        "must_not_echo",
        "pii_tokens",
        "attack",
    ]

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "answerdesk-bench item (v0.1)",
        "description": (
            "One benchmark item. The suite discriminates the payload; see "
            "docs/benchmark.md §4 for field semantics and docs/benchmark.md §9 for the "
            "t19 adapter contract."
        ),
        "oneOf": [
            variant(
                "belebele_retrieval",
                {"query": string, "qrels": id_list, "gold_evidence_ids": id_list},
                retrieval_required,
            ),
            variant(
                "mrtydi_retrieval",
                {"query": string, "qrels": id_list, "gold_evidence_ids": id_list},
                retrieval_required,
            ),
            variant(
                "belebele_qa",
                {
                    "question": string,
                    "options": {"type": "array", "minItems": 4, "maxItems": 4, "items": string},
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "gold_evidence_ids": id_list,
                    "answer_anchor": {"type": "number", "minimum": 0, "maximum": 1},
                    "answerable": {"const": True},
                },
                qa_required,
            ),
            variant(
                "domain",
                {
                    "question": string,
                    "answerable": {"type": "boolean"},
                    "expected_behavior": {"type": "string", "enum": list(EXPECTED_BEHAVIOURS)},
                    "reference_answer": {"type": "string"},
                    "evidence": evidence,
                    "gold_evidence_ids": {"type": "array", "items": string},
                    "must_not_echo": {"type": "array", "items": string},
                    "pii_tokens": {"type": "array", "items": string},
                    "attack": {"type": ["object", "null"]},
                },
                domain_required,
            ),
        ],
    }
