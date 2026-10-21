"""Risk-control v3: persisted bench artefacts, union dev calibration, 5%-certificate attempt.

# AT-RISK-04  # AT-RISK-05  # AT-RISK-06
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.eval.risk import v3
from answerdesk.eval.risk.certificate import certify
from answerdesk.eval.risk.features import read_rows


def _answer_row(case_id: str, rubric: float, *, provider_error: bool = False) -> dict:
    return {
        "case_id": case_id,
        "locale": "ru",
        "bucket": "direct",
        "question": f"вопрос {case_id}",
        "answer": "Ответ на вопрос поддержки.",
        "claims": [{"text": "Ответ на вопрос поддержки.", "citations": [f"{case_id}::c0"]}],
        "citations": [f"{case_id}::c0"],
        "cited_doc_ids": [case_id],
        "retrieved_doc_ids": [case_id],
        "decision": "answer",
        "reason_code": "ANSWERED",
        "rubric": rubric,
        "claim_support": 1.0,
        "numeric_consistency": 1.0,
        "citation_coverage": 0.5,
        "provider_error": provider_error,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _mini_run(tmp_path: Path, *, leak: bool = False) -> Path:
    out = tmp_path / "risk-v3"
    out.mkdir()
    dev = [_answer_row(f"dev-{index:03d}", 5.0 if index % 3 else 1.0) for index in range(40)]
    test = [_answer_row(f"v1-{index:03d}", 5.0 if index % 4 else 1.0) for index in range(60)]
    if leak:
        test.append(_answer_row("dev-000", 5.0))
    _write_jsonl(out / "bench-answers-dev.jsonl", dev)
    _write_jsonl(out / "bench-answers-test.jsonl", test)
    return out


def test_features_come_from_persisted_answers_and_keep_splits_apart(tmp_path: Path) -> None:
    out = _mini_run(tmp_path)
    payload = v3.build_features(out_dir=out, consistency_features=tmp_path / "absent.csv")
    assert payload["dev_rows"] == 40
    assert payload["test_rows"] == 60
    assert payload["no_leakage"] is True
    rows = read_rows(out / "features.csv")
    assert {"claim_support", "numeric_consistency", "citation_coverage"} <= set(rows[0].features)
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["no_leakage"] is True


def test_leaking_split_is_rejected_by_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _mini_run(tmp_path, leak=True)
    v3.build_features(out_dir=out, consistency_features=tmp_path / "absent.csv")
    config = v3.V3Config(out_dir=out)
    v3.evaluate(config=config)
    assert v3.check(out, config=config) == 1
    assert "leak" in capsys.readouterr().out


def test_evaluate_fits_on_dev_only(tmp_path: Path) -> None:
    out = _mini_run(tmp_path)
    v3.build_features(out_dir=out, consistency_features=tmp_path / "absent.csv")
    summary = v3.evaluate(config=v3.V3Config(out_dir=out))
    bench = summary["populations"]["bench"]
    assert bench["dev_rows"] == 40 and bench["test_rows"] == 60
    assert bench["model"]["features"]  # the model records its feature list
    assert (out / "certificate.json").is_file()
    assert (out / "frontier.csv").is_file()
    # the certificate must be honest about what it covers
    assert "certificate covers the calibration draw" in " ".join(summary["caveats"])


def test_abstain_when_infeasible(tmp_path: Path) -> None:
    """A tiny, noisy calibration set cannot certify 5%: the artefact must say ABSENT."""
    out = tmp_path / "risk-v3"
    out.mkdir()
    _write_jsonl(
        out / "bench-answers-dev.jsonl",
        [_answer_row(f"dev-{index:03d}", 5.0 if index % 2 else 1.0) for index in range(6)],
    )
    _write_jsonl(
        out / "bench-answers-test.jsonl",
        [_answer_row(f"v1-{index:03d}", 5.0 if index % 2 else 1.0) for index in range(10)],
    )
    v3.build_features(out_dir=out, consistency_features=tmp_path / "absent.csv")
    summary = v3.evaluate(config=v3.V3Config(out_dir=out))
    certificate = summary["populations"]["bench"]["certificate"]
    assert certificate["feasible"] is False
    assert "ABSENT" in certificate["claims"]
    operating = summary["populations"]["bench"]["test_report"]["certified_operating_point"]
    assert operating["abstain"] is True
    assert operating["n_accepted"] == 0 and operating["coverage"] == 0.0


def test_check_reproduces_and_detects_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _mini_run(tmp_path)
    v3.build_features(out_dir=out, consistency_features=tmp_path / "absent.csv")
    config = v3.V3Config(out_dir=out)
    v3.evaluate(config=config)
    assert v3.check(out, config=config) == 0
    (out / "features.csv").write_text(
        (out / "features.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert v3.check(out, config=config) == 1


def test_certificate_rule_matches_the_bound() -> None:
    scores = [1.0 - index / 200 for index in range(100)]
    labels = [1] * 97 + [0] * 3
    certificate = certify(scores=scores, labels=labels, alpha=0.2, delta=0.1)
    if certificate.feasible:
        assert certificate.calibration_ucb <= 0.2
        feasible = [level for level in certificate.to_dict()["grid"] if level["feasible"]]
        assert certificate.calibration_coverage == max(level["coverage"] for level in feasible)
