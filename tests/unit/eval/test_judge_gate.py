"""Judge-calibration gate tests: κ, sheet validation, gate transitions and fail-closed use.

The dry-run fixtures under ``fixtures/judge-gate/`` are synthetic and marked as such; they
exist to prove the computation and the mechanical flip, never to stand in for human labels.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import judge_gate  # noqa: E402

from answerdesk.eval.external.calibration import (  # noqa: E402
    ITEMS_PER_LANGUAGE,
    TEMPLATE_COLUMNS,
)
from answerdesk.eval.stats import cohen_kappa as reference_kappa  # noqa: E402

FIXTURES = judge_gate.FIXTURES_DIR
GATE_PATH = REPO_ROOT / "reports" / "judge-gate" / "gate.json"
SHEETS = REPO_ROOT / "reports" / "external-p0" / "calibration"
COLUMNS = judge_gate.Policy.load().required_columns


def _fixture_policy() -> judge_gate.Policy:
    return judge_gate.Policy.load(FIXTURES / "policy.json")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_sheet(
    directory: Path, language: str, rows: list[dict[str, str]], columns: list[str] = COLUMNS
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"calibration-{language}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _row(
    language: str, index: int, judge: str, left: str, right: str, adjudicated: str
) -> dict[str, str]:
    return {
        "item_id": f"t-{language}-{index:03d}",
        "language": language,
        "item_ref": "ref",
        "question": "",
        "answer_text": "",
        "judge_label": judge,
        "human_label_a": left,
        "human_label_b": right,
        "adjudicated_label": adjudicated,
        "notes": "",
    }


# --------------------------------------------------------------------------- statistics


def test_kappa_matches_textbook_values() -> None:
    assert judge_gate.cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    # po=0.5, both marginals 0.5 → pe=0.5 → κ=0
    assert judge_gate.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert judge_gate.cohen_kappa([True], [True]) is None
    assert judge_gate.cohen_kappa([True, True], [True, True]) is None  # expected agreement 1


def test_kappa_matches_the_harness_implementation() -> None:
    rng = random.Random(7)
    first = [rng.random() < 0.6 for _ in range(80)]
    second = [value if rng.random() < 0.8 else not value for value in first]
    assert judge_gate.cohen_kappa(first, second) == pytest.approx(
        reference_kappa(first, second), abs=1e-12
    )


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    first = ["supported"] * 20 + ["unsupported"] * 10
    second = ["supported"] * 18 + ["unsupported"] * 12
    one = judge_gate.bootstrap_kappa(first, second, samples=300, seed=11, confidence=0.95)
    two = judge_gate.bootstrap_kappa(first, second, samples=300, seed=11, confidence=0.95)
    assert one == two
    assert one is not None and -1.0 <= one[0] <= one[1] <= 1.0


# --------------------------------------------------------------------------- committed state


def test_committed_sheets_are_pending_and_unlabelled() -> None:
    artefact = judge_gate.evaluate(SHEETS, judge_gate.Policy.load())
    assert artefact["status"] == "pending"
    assert artefact["judged_metrics_label"] == "diagnostic"
    assert artefact["publishable"] is False
    assert artefact["synthetic"] is False
    assert artefact["pooled"]["labelled"] == 0
    for path in SHEETS.glob("calibration-*.csv"):
        for row in _read_rows(path):
            for column in ("human_label_a", "human_label_b", "adjudicated_label"):
                assert row[column].strip() == ""


def test_committed_gate_matches_the_committed_sheets() -> None:
    stored = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    recomputed = judge_gate.evaluate(SHEETS, judge_gate.Policy.load())
    assert stored["status"] == recomputed["status"] == "pending"
    assert stored["judged_metrics_label"] == "diagnostic"
    assert stored["inputs"] == recomputed["inputs"]
    for language, block in recomputed["per_language"].items():
        assert stored["per_language"][language]["items"] == block["items"]
        assert stored["per_language"][language]["status"] == block["status"]


def test_policy_matches_the_external_calibration_template() -> None:
    policy = judge_gate.Policy.load()
    assert set(policy.languages) == {"ru", "uz", "tg", "ky", "zh"}
    assert policy.items_per_language == ITEMS_PER_LANGUAGE == 150
    assert policy.required_columns == TEMPLATE_COLUMNS
    assert policy.judge_kappa_floor == 0.6
    assert policy.pairwise_agreement_floor == 0.7


# --------------------------------------------------------------------------- dry-run gate


def test_high_agreement_fixture_opens_the_gate_but_is_not_publishable() -> None:
    artefact = judge_gate.evaluate(FIXTURES / "high", _fixture_policy())
    assert artefact["status"] == "open"
    assert artefact["judged_metrics_label"] == "calibrated"
    assert artefact["synthetic"] is True
    assert artefact["publishable"] is False
    assert artefact["reasons"] == []
    assert artefact["pooled"]["judge_kappa"] is not None
    assert artefact["pooled"]["judge_kappa"] >= 0.6
    assert artefact["pooled"]["pairwise_agreement"] >= 0.7
    assert artefact["disagreements"] == 15  # three adjudicated disagreements per language
    assert all(block["judge_kappa"] >= 0.6 for block in artefact["per_language"].values())


def test_low_agreement_fixture_blocks_the_gate() -> None:
    artefact = judge_gate.evaluate(FIXTURES / "low", _fixture_policy())
    assert artefact["status"] == "blocked"
    assert artefact["judged_metrics_label"] == "diagnostic"
    text = " ".join(artefact["reasons"])
    assert "pairwise agreement" in text
    assert "judge–human κ" in text


def test_partial_fixture_blocks_the_gate() -> None:
    artefact = judge_gate.evaluate(FIXTURES / "partial", _fixture_policy())
    assert artefact["status"] == "blocked"
    assert any("zh: no labels" in reason for reason in artefact["reasons"])


def test_fixture_rows_are_marked_synthetic() -> None:
    marker = _fixture_policy().synthetic_marker
    for path in sorted(FIXTURES.rglob("*.csv")):
        rows = _read_rows(path)
        assert rows, path
        assert all(marker in row["notes"] for row in rows), path


def test_gate_is_fail_closed(tmp_path: Path) -> None:
    assert judge_gate.load_judged_label(tmp_path / "missing.json") == "diagnostic"
    corrupt = tmp_path / "gate.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert judge_gate.load_judged_label(corrupt) == "diagnostic"
    synthetic_open = tmp_path / "synthetic.json"
    synthetic_open.write_text(
        json.dumps({"status": "open", "publishable": True, "synthetic": True}), encoding="utf-8"
    )
    assert judge_gate.load_judged_label(synthetic_open) == "diagnostic"
    real_open = tmp_path / "open.json"
    real_open.write_text(
        json.dumps({"status": "open", "publishable": True, "synthetic": False}), encoding="utf-8"
    )
    assert judge_gate.load_judged_label(real_open) == "calibrated"


# --------------------------------------------------------------------------- validation


def test_invalid_sheets_are_rejected(tmp_path: Path) -> None:
    good_rows = [
        _row("ru", index, "supported", "supported", "supported", "") for index in range(1, 31)
    ]

    duplicate_dir = tmp_path / "duplicate"
    duplicate_rows = [dict(row) for row in good_rows]
    duplicate_rows[1]["item_id"] = duplicate_rows[0]["item_id"]
    _write_sheet(duplicate_dir, "ru", duplicate_rows)
    duplicate = judge_gate.evaluate(duplicate_dir, _fixture_policy())
    assert duplicate["status"] == "blocked"
    assert any("duplicate" in reason for reason in duplicate["reasons"])

    bad_label_dir = tmp_path / "bad-label"
    bad_rows = [dict(row) for row in good_rows]
    bad_rows[0]["judge_label"] = "probably"
    _write_sheet(bad_label_dir, "ru", bad_rows)
    bad_label = judge_gate.evaluate(bad_label_dir, _fixture_policy())
    assert bad_label["status"] == "blocked"
    assert any("not one of" in reason for reason in bad_label["reasons"])

    missing_column_dir = tmp_path / "missing-column"
    columns = [column for column in COLUMNS if column != "adjudicated_label"]
    _write_sheet(missing_column_dir, "ru", good_rows, columns=columns)
    missing = judge_gate.evaluate(missing_column_dir, _fixture_policy())
    assert missing["status"] == "blocked"
    assert any("missing columns" in reason for reason in missing["reasons"])

    short_dir = tmp_path / "short"
    _write_sheet(short_dir, "ru", good_rows[:-1])
    short = judge_gate.evaluate(short_dir, _fixture_policy())
    assert short["status"] == "blocked"
    assert any("row count" in reason for reason in short["reasons"])


def test_load_sheet_raises_with_structural_errors(tmp_path: Path) -> None:
    path = tmp_path / "calibration-ru.csv"
    path.write_text("item_id,language\nx,ru\n", encoding="utf-8")
    with pytest.raises(judge_gate.GateError) as error:
        judge_gate.load_sheet(path, _fixture_policy())
    assert any("missing columns" in message for message in error.value.errors)


# --------------------------------------------------------------------------- artefacts


def test_disagreements_are_written_for_adjudication(tmp_path: Path) -> None:
    artefact = judge_gate.evaluate(FIXTURES / "high", _fixture_policy())
    paths = judge_gate.write_artefacts(artefact, tmp_path)
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    assert "disagreement_rows" not in gate
    assert gate["status"] == "open"
    rows = _read_rows(paths["disagreements"])
    assert len(rows) == 15
    assert {row["reason"] for row in rows} == {"adjudicated"}
    assert paths["markdown"].is_file()


def test_self_check_passes() -> None:
    result = judge_gate.check()
    assert result["ok"] is True
    names = {item["check"] for item in result["checks"]}
    assert {
        "committed_sheets_pending",
        "committed_gate_matches",
        "dry_run_high_opens",
        "dry_run_low_blocks",
        "dry_run_partial_blocks",
        "fail_closed_without_gate",
        "invalid_sheet_rejected",
        "no_human_labels_committed",
    } <= names
    assert len(result["checks"]) >= 10
