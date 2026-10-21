"""The item contract: every committed item conforms, and its payload is internally consistent."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from bench import schema
from bench.config import SUITE_LICENSES
from bench.normalize import answer_anchor


def test_every_item_conforms_to_the_schema(items: list[dict[str, Any]]) -> None:
    errors = [error for item in items for error in schema.validate_item(item)]
    assert errors == []
    assert len(items) > 1000


def test_core_envelope_is_complete(items: list[dict[str, Any]]) -> None:
    for item in items:
        for field in schema.CORE_FIELDS:
            assert field in item, (item["item_id"], field)
        provenance = item["provenance"]
        for field in schema.PROVENANCE_FIELDS:
            assert isinstance(provenance[field], str) and provenance[field], (item["item_id"], field)
        assert len(provenance["source_sha256"]) == 64
        assert item["split"] in schema.SPLITS
        assert item["license"] == SUITE_LICENSES[item["suite"]]
        assert item["category"] in schema.SUITES[item["suite"]]["categories"]


def test_item_ids_are_unique_and_namespaced(items: list[dict[str, Any]]) -> None:
    counts = Counter(str(item["item_id"]) for item in items)
    assert [item_id for item_id, count in counts.items() if count > 1] == []
    prefixes = {"belebele_retrieval": "bb-ret-", "belebele_qa": "bb-qa-", "mrtydi_retrieval": "mrtydi-ru-", "domain": "dom-"}
    for item in items:
        assert str(item["item_id"]).startswith(prefixes[item["suite"]]), item["item_id"]


def test_retrieval_qrels_resolve_to_pool_passages(items, passages) -> None:
    checked = 0
    for item in items:
        if item["suite"] not in ("belebele_retrieval", "mrtydi_retrieval"):
            continue
        assert item["qrels"], item["item_id"]
        for qrel in item["qrels"]:
            assert qrel in passages, (item["item_id"], qrel)
            assert passages[qrel]["language"] == item["language"]
            checked += 1
    assert checked >= 3000


def test_qa_answers_are_verifiable_against_the_gold_passage(items, passages) -> None:
    checked = 0
    for item in items:
        if item["suite"] != "belebele_qa":
            continue
        assert item["answerable"] is True
        assert len(item["options"]) == 4
        assert len({option.strip().casefold() for option in item["options"]}) == 4
        assert 0 <= item["answer_index"] <= 3
        evidence = item["gold_evidence_ids"]
        assert len(evidence) == 1
        passage = passages[evidence[0]]["text"]
        expected = answer_anchor(item["options"][item["answer_index"]], passage)
        assert item["answer_anchor"] == expected, item["item_id"]
        checked += 1
    assert checked == 750


def test_domain_reference_is_a_verbatim_evidence_quote(items) -> None:
    for item in items:
        if item["suite"] != "domain":
            continue
        quotes = [entry["quote"] for entry in item["evidence"]]
        if item["answerable"]:
            if item["category"] == "multi_hop":
                assert len(quotes) >= 2
                assert item["reference_answer"] == " ".join(quotes)
            else:
                assert quotes and item["reference_answer"] == quotes[0]
        else:
            assert item["reference_answer"] == ""
            assert quotes == []
            assert item["gold_evidence_ids"] == []
            assert item["expected_behavior"] == "escalate"


def test_domain_adversarial_payload_is_well_formed(items) -> None:
    for item in items:
        if item["suite"] != "domain":
            continue
        if item["category"] == "pii":
            assert item["pii_tokens"] and item["must_not_echo"]
            assert item["expected_behavior"] == "answer_without_echo"
            for token in item["must_not_echo"]:
                assert token not in item["reference_answer"]
                assert any(token in item["question"] for token in item["pii_tokens"])
        elif item["category"] == "injection":
            attack = item["attack"]
            assert attack["must_not_comply"] is True
            assert attack["instruction"] in item["question"]
            for token in item["must_not_echo"]:
                assert token not in item["reference_answer"]
        else:
            assert item["attack"] is None


def test_schema_json_matches_the_generator(bench_dir: Path) -> None:
    committed = json.loads((bench_dir / "schema.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(schema.json_schema(), ensure_ascii=False))
