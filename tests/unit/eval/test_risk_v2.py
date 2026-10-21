"""Risk-control v2: certificate rule, no-leakage split, metrics (offline tests).

# AT-RISK-01  # AT-RISK-02  # AT-RISK-03
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from answerdesk.eval.risk import metrics
from answerdesk.eval.risk.certificate import (
    bentkus_ucb,
    certify,
    hoeffding_ucb,
    ucb,
)
from answerdesk.eval.risk.features import FeatureRow, write_rows
from answerdesk.eval.risk.model import fit, fit_platt

REPO_ROOT = Path(__file__).resolve().parents[3]


def _script():
    spec = importlib.util.spec_from_file_location(
        "risk_v2_script", REPO_ROOT / "scripts" / "risk-v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# bounds and the certificate rule
# --------------------------------------------------------------------------------------


def test_ucb_is_above_the_point_estimate_and_shrinks_with_n() -> None:
    risk = 0.1
    assert bentkus_ucb(risk, 100, 0.1) > risk
    assert hoeffding_ucb(risk, 100, 0.1) > risk
    assert bentkus_ucb(risk, 1000, 0.1) < bentkus_ucb(risk, 100, 0.1)
    assert hoeffding_ucb(risk, 1000, 0.1) < hoeffding_ucb(risk, 100, 0.1)
    assert bentkus_ucb(risk, 100, 0.1) <= hoeffding_ucb(risk, 100, 0.1)


def test_ucb_is_clipped_and_rejects_bad_input() -> None:
    assert ucb(0.99, 1, 0.1, "bentkus") <= 1.0
    with pytest.raises(ValueError):
        bentkus_ucb(1.5, 10, 0.1)
    with pytest.raises(ValueError):
        ucb(0.1, 10, 0.1, "unknown")  # type: ignore[arg-type]


def test_certificate_absent_when_no_level_is_feasible() -> None:
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 0, 0]
    certificate = certify(scores=scores, labels=labels, alpha=0.05, delta=0.1)
    assert certificate.feasible is False
    assert certificate.calibration_coverage == 0.0
    assert "ABSENT" in certificate.to_dict()["claims"]


def test_certificate_selects_largest_feasible_coverage() -> None:
    # 40 rows, 38 correct: a zero-error region exists with enough rows for Bentkus at alpha=0.1
    scores = [1.0 - index / 100 for index in range(40)]
    labels = [1] * 38 + [0, 0]
    certificate = certify(scores=scores, labels=labels, alpha=0.1, delta=0.1)
    assert certificate.feasible is True
    feasible = [level for level in certificate.to_dict()["grid"] if level["feasible"]]
    assert certificate.calibration_coverage == max(level["coverage"] for level in feasible)
    assert certificate.calibration_ucb <= 0.1


def test_certificate_never_claims_more_than_the_bound() -> None:
    # a tiny calibration set cannot certify anything (Bentkus is clipped to 1.0 at n=2)
    tiny = certify(scores=[0.9, 0.8, 0.7], labels=[1, 1, 0], alpha=0.5, delta=0.1)
    assert tiny.feasible is False
    assert "ABSENT" in tiny.to_dict()["claims"]
    scores = [1.0 - index / 400 for index in range(200)]
    labels = [1] * 194 + [0] * 6
    certificate = certify(scores=scores, labels=labels, alpha=0.5, delta=0.1)
    payload = certificate.to_dict()
    assert certificate.feasible is True
    assert "with probability >= 1-delta" in payload["claims"]
    assert payload["rule"].startswith("select the largest calibration acceptance level")
    for level in payload["grid"]:
        assert level["ucb"] >= level["risk"]


def test_n_required_matches_the_closed_forms() -> None:
    script = _script()
    from answerdesk.eval.risk.runner import _n_required

    payload = _n_required(0.05, 0.1)
    assert payload["hoeffding_rows_at_zero_risk"] == pytest.approx(460.5, abs=0.1)
    assert payload["bentkus_rows_at_zero_risk"] == pytest.approx(46.1, abs=0.1)
    assert script.main is not None


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def test_auroc_known_values() -> None:
    assert metrics.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert metrics.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert metrics.auroc([0.9], [1]) == 0.5  # one class absent


def test_ece_zero_when_calibrated() -> None:
    scores = [0.0] * 10 + [1.0] * 10
    labels = [0] * 10 + [1] * 10
    assert metrics.ece_equal_mass(scores, labels)[0] == pytest.approx(0.0, abs=1e-9)
    assert metrics.ece_equal_width(scores, labels) == pytest.approx(0.0, abs=1e-9)


def test_frontier_is_monotone_and_aurc_matches_area() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [1, 0, 1, 0, 0]
    rows, aurc = metrics.frontier(scores, labels)
    assert [row["coverage"] for row in rows] == sorted(row["coverage"] for row in rows)
    assert rows[-1]["coverage"] == 1.0
    assert rows[-1]["risk"] == pytest.approx(3 / 5)
    assert 0.0 < aurc <= 1.0


def test_coverage_at_risk_picks_the_largest_feasible_prefix() -> None:
    rows = [
        {"coverage": 0.2, "risk": 0.0, "threshold": 0.9, "n_accepted": 2, "wrong": 0},
        {"coverage": 0.4, "risk": 0.06, "threshold": 0.8, "n_accepted": 4, "wrong": 0},
        {"coverage": 0.6, "risk": 0.2, "threshold": 0.7, "n_accepted": 6, "wrong": 1},
    ]
    point = metrics.coverage_at_risk(rows, 0.05)
    assert point["coverage"] == 0.2
    assert point["risk"] == 0.0


def test_platt_calibration_is_monotone_and_improves_log_loss() -> None:
    scores = [0.9, 0.85, 0.7, 0.4, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 0]
    calibrated = fit_platt(scores, labels)
    transformed = calibrated.apply(scores)
    assert transformed == sorted(transformed, reverse=True)
    assert all(0.0 < value < 1.0 for value in transformed)
    assert metrics.log_loss(transformed, labels) <= metrics.log_loss(scores, labels) + 1e-6


def test_logistic_fit_separates_a_learnable_signal() -> None:
    rows = []
    labels = []
    for index in range(40):
        value = index / 40
        rows.append({"x": value, "y": 0.0})
        labels.append(1 if value > 0.5 else 0)
    model = fit(rows, labels, features=("x",), l2=1.0, steps=2000)
    assert model.predict([{"x": 0.95}])[0] > model.predict([{"x": 0.05}])[0]


# --------------------------------------------------------------------------------------
# split and the check gate
# --------------------------------------------------------------------------------------


def _row(sample: str, fact: str, split: str, label: int, value: float) -> FeatureRow:
    return FeatureRow(
        sample_id=sample,
        fact_id=fact,
        locale="ru",
        bucket="tariff_schedule",
        split=split,
        features={
            "rerank_top": value,
            "retrieval_margin": value,
            "claim_support": value,
            "citation_coverage": value,
            "numeric_consistency": value,
            "overlap_score": value,
            "bm25_top": value,
            "bm25_margin": value,
            "bm25_above_gate": value,
            "citation_density": value,
            "log_claims": value,
        },
        label=label,
    )


def _synthetic_out_dir(tmp_path: Path, *, leak: bool = False) -> Path:
    out = tmp_path / "risk"
    out.mkdir()
    rows = []
    for index in range(30):
        label = 1 if index % 3 else 0
        value = 0.9 if label else 0.2
        rows.append(
            _row(f"f{index}::ru", f"f{index}", "dev" if index < 12 else "test", label, value)
        )
    if leak:
        # the same fact id appears in both splits
        rows.append(_row("f0::ru", "f0", "test", 1, 0.9))
    write_rows(out / "features.csv", rows)
    dev_facts = sorted({row.fact_id for row in rows if row.split == "dev"})
    test_facts = sorted({row.fact_id for row in rows if row.split == "test"})
    leakage = sorted(set(dev_facts) & set(test_facts))
    (out / "split.json").write_text(
        json.dumps(
            {
                "rule": "test",
                "dev_facts": dev_facts,
                "test_facts": test_facts,
                "dev_rows": sum(1 for row in rows if row.split == "dev"),
                "test_rows": sum(1 for row in rows if row.split == "test"),
                "leakage": leakage,
                "no_leakage": not leakage,
            }
        ),
        encoding="utf-8",
    )
    return out


def test_evaluate_and_check_round_trip(tmp_path: Path) -> None:
    from answerdesk.eval.risk.runner import check, evaluate

    out = _synthetic_out_dir(tmp_path)
    summary = evaluate(out_dir=out, steps=500)
    assert summary["primary_population"] == "consistency"
    assert (out / "certificate.json").is_file()
    assert (out / "frontier.csv").is_file()
    assert (out / "reliability.csv").is_file()
    assert check(out) == 0


def test_check_rejects_a_leaking_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from answerdesk.eval.risk.runner import evaluate

    out = _synthetic_out_dir(tmp_path)
    evaluate(out_dir=out, steps=500)
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    split["leakage"] = ["f0"]
    split["no_leakage"] = False
    (out / "split.json").write_text(json.dumps(split), encoding="utf-8")
    from answerdesk.eval.risk.runner import check

    assert check(out) == 1
    assert "leak" in capsys.readouterr().out


def test_check_rejects_feature_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from answerdesk.eval.risk.runner import check, evaluate

    out = _synthetic_out_dir(tmp_path)
    evaluate(out_dir=out, steps=500)
    text = (out / "features.csv").read_text(encoding="utf-8") + "\n"
    (out / "features.csv").write_text(text, encoding="utf-8")
    assert check(out) == 1
    assert "changed since" in capsys.readouterr().out
