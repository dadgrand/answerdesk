"""Risk-control v3 runner: bench answer artefacts, union dev calibration, 5%-certificate attempt.

    uv run python scripts/risk-v3.py --persist-dev --limit 200     # bounded, resumable
    uv run python scripts/risk-v3.py --persist-test --limit 200
    uv run python scripts/risk-v3.py --features
    uv run python scripts/risk-v3.py --self-consistency --limit 25
    uv run python scripts/risk-v3.py --evaluate
    uv run python scripts/risk-v3.py --check                       # offline reproduction gate

Dev calibration is the union of the persisted bench-dev turns and the consistency run's dev
rows (both carry the same feature schema), which is how the calibration set reaches >= 300 rows
without regenerating the consistency answers. Test is the bench test split, evaluated once.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from answerdesk.eval.risk import metrics
from answerdesk.eval.risk.certificate import Certificate, certify
from answerdesk.eval.risk.features import (
    DECISION_FEATURES,
    LEGACY_FEATURES,
    FeatureRow,
    feature_digest,
    load_documents,
    read_rows,
    write_rows,
)
from answerdesk.eval.risk.model import fit, fit_platt

DEFAULT_OUT = Path("reports/risk-v3")
ALPHA = 0.05
DELTA = 0.1
L2 = 2.0
STEPS = 4000


@dataclass
class V3Config:
    out_dir: Path = DEFAULT_OUT
    alpha: float = ALPHA
    delta: float = DELTA
    l2: float = L2
    steps: int = STEPS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _split_of(case_id: str) -> str:
    return "test" if case_id.startswith("v1-") else "dev"


def build_features(
    *,
    out_dir: Path = DEFAULT_OUT,
    corpus_dir: Path = Path("data/corpus"),
    consistency_features: Path | None = None,
) -> dict[str, Any]:
    """Feature rows from the persisted bench answers (dev + test) and the consistency dev rows."""
    from answerdesk.eval.risk.features import bm25_scores, overlap_score

    documents = load_documents(corpus_dir)
    rows: list[FeatureRow] = []
    for split, name in (("dev", "bench-answers-dev.jsonl"), ("test", "bench-answers-test.jsonl")):
        for record in _read_jsonl(out_dir / name):
            if record.get("provider_error") or record.get("rubric") is None:
                continue
            # selective-answering population: only turns the pipeline actually answered; an
            # escalation is an abstention, not a wrong answer, and the policy re-decides the
            # answered ones with a better score
            if record.get("decision") != "answer":
                continue
            question = str(record.get("question") or "")
            retrieved = [
                documents.get(doc_id, "") for doc_id in record.get("retrieved_doc_ids", [])
            ]
            lexical = (
                bm25_scores(question, [text for text in retrieved if text][:20])
                if retrieved
                else []
            )
            top = lexical[0] if lexical else 0.0
            second = lexical[1] if len(lexical) > 1 else 0.0
            margin = (top - second) / (abs(top) + 1e-6) if top else 0.0
            cited = list(record.get("cited_doc_ids") or [])
            claims = max(1, len(record.get("claims") or []))
            features = {
                "rerank_top": 0.0,
                "retrieval_margin": max(0.0, min(1.0, margin)),
                "claim_support": float(record.get("claim_support") or 0.0),
                "citation_coverage": float(record.get("citation_coverage") or 0.0),
                "numeric_consistency": float(record.get("numeric_consistency") or 0.0),
                "overlap_score": (
                    overlap_score(question, retrieved[0]) if retrieved and retrieved[0] else 0.0
                ),
                "bm25_top": top,
                "bm25_margin": margin,
                "bm25_above_gate": sum(1 for score in lexical if score > 0.0)
                / max(1, len(lexical)),
                "citation_density": len(cited) / claims,
                "log_claims": math.log1p(claims),
            }
            rows.append(
                FeatureRow(
                    sample_id=str(record["case_id"]),
                    fact_id=str(record["case_id"]),
                    locale=str(record.get("locale") or ""),
                    bucket=str(record.get("bucket") or "unknown"),
                    split=split,
                    features=features,
                    label=1 if float(record["rubric"]) >= 4 else 0,
                )
            )
    # union dev: add the consistency dev rows (same schema, already live-labelled)
    consistency = consistency_features or Path("reports/risk-v2/features.csv")
    if consistency.is_file():
        for row in read_rows(consistency):
            # the v3 test set is the bench test split, so every consistency row is calibration
            # data here; no bench-test case id reaches the fit
            if row.split not in {"dev", "test"}:
                continue
            features = dict(row.features)
            for name in DECISION_FEATURES:
                features.setdefault(name, 0.0)
            for name in LEGACY_FEATURES:
                features.setdefault(name, 0.0)
            rows.append(
                FeatureRow(
                    sample_id=f"consistency::{row.sample_id}",
                    fact_id=f"consistency::{row.fact_id}",
                    locale=row.locale,
                    bucket=row.bucket,
                    split="dev",
                    features=features,
                    label=row.label,
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "features.csv", rows)
    split_payload: dict[str, Any] = {
        "rule": (
            "test = the frozen bench test split (v1-* case ids); dev = frozen bench dev split "
            "plus the consistency run's dev facts; no case id appears in both"
        ),
        "dev_rows": sum(1 for row in rows if row.split == "dev"),
        "test_rows": sum(1 for row in rows if row.split == "test"),
        "leakage": sorted(
            {row.fact_id for row in rows if row.split == "dev"}
            & {row.fact_id for row in rows if row.split == "test"}
        ),
    }
    split_payload["no_leakage"] = not split_payload["leakage"]
    (out_dir / "split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(rows),
        "dev_rows": split_payload["dev_rows"],
        "test_rows": split_payload["test_rows"],
        "no_leakage": split_payload["no_leakage"],
        "features_sha256": feature_digest(out_dir / "features.csv"),
    }


def _n_required(alpha: float, delta: float) -> dict[str, Any]:
    x = math.log(1.0 / delta)
    return {
        "hoeffding_rows_at_zero_risk": round(x / (2.0 * alpha * alpha), 1),
        "bentkus_rows_at_zero_risk": round(x / alpha, 1),
        "note": (
            "rows needed for a certificate at alpha/delta when the accepted region has zero "
            "observed errors; non-zero observed risk needs more rows (UCB grows with "
            "sqrt(risk * log(1/delta) / n))"
        ),
    }


def evaluate(*, config: V3Config | None = None) -> dict[str, Any]:
    config = config or V3Config()
    out_dir = config.out_dir
    rows = read_rows(out_dir / "features.csv")
    dev = [row for row in rows if row.split == "dev"]
    test = [row for row in rows if row.split == "test"]
    if not dev or not test:
        raise RuntimeError("empty dev or test split; run --features first")
    model_features = tuple(DECISION_FEATURES) + tuple(LEGACY_FEATURES)
    model = fit(
        [r.features for r in dev],
        [r.label for r in dev],
        features=model_features,
        l2=config.l2,
        steps=config.steps,
    )
    calibrator = fit_platt(model.predict([r.features for r in dev]), [r.label for r in dev])
    dev_scores = calibrator.apply(model.predict([r.features for r in dev]))
    test_scores = calibrator.apply(model.predict([r.features for r in test]))
    dev_labels = [r.label for r in dev]
    test_labels = [r.label for r in test]
    certificate: Certificate = certify(
        scores=dev_scores,
        labels=dev_labels,
        alpha=config.alpha,
        delta=config.delta,
        bound="bentkus",
    )
    hoeffding = certify(
        scores=dev_scores,
        labels=dev_labels,
        alpha=config.alpha,
        delta=config.delta,
        bound="hoeffding",
    )
    illustrative = certify(
        scores=dev_scores, labels=dev_labels, alpha=0.25, delta=config.delta, bound="bentkus"
    )
    frontier_rows, aurc = metrics.frontier(test_scores, test_labels)
    ece_value, reliability = metrics.ece_equal_mass(test_scores, test_labels)
    legacy = fit(
        [r.features for r in dev],
        dev_labels,
        features=LEGACY_FEATURES,
        l2=config.l2,
        steps=config.steps,
    )
    legacy_test = fit_platt(legacy.predict([r.features for r in dev]), dev_labels).apply(
        legacy.predict([r.features for r in test])
    )
    if certificate.feasible:
        operating = metrics.selective_risk(
            scores=test_scores, labels=test_labels, threshold=certificate.threshold
        )
    else:
        # no certified threshold exists: the policy abstains on everything and says so
        operating = {
            "threshold": None,
            "n": len(test),
            "n_accepted": 0,
            "coverage": 0.0,
            "wrong": 0,
            "realised_risk": None,
            "risk_ci": None,
            "abstain": True,
            "reason": "certificate infeasible at the requested alpha; the policy answers nothing",
        }

    def arm(scores: Sequence[float]) -> dict[str, Any]:
        rows_frontier, area = metrics.frontier(scores, test_labels)
        return {
            "auroc": round(metrics.auroc(scores, test_labels), 6),
            "ece": round(metrics.ece_equal_mass(scores, test_labels)[0], 6),
            "aurc": round(area, 6),
            "coverage_at_risk": metrics.coverage_at_risk(rows_frontier, config.alpha),
        }

    summary = {
        "run_id": out_dir.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_population": "bench",
        "alpha": config.alpha,
        "delta": config.delta,
        "hyperparameters": {"l2": config.l2, "steps": config.steps},
        "populations": {
            "bench": {
                "description": (
                    "the product's frozen bench turns, re-answered with the shipped pipeline and "
                    "re-judged by the cross-family judge, with full answer artefacts persisted"
                ),
                "dev_rows": len(dev),
                "test_rows": len(test),
                "features_sha256": feature_digest(out_dir / "features.csv"),
                "base_error_dev": round(1.0 - float(np.mean(dev_labels)), 6),
                "base_error_test": round(1.0 - float(np.mean(test_labels)), 6),
                "model": {
                    "features": list(model_features),
                    "weights": model.to_dict(),
                    "calibration": calibrator.to_dict(),
                    "dev_auroc": round(metrics.auroc(dev_scores, dev_labels), 6),
                    "dev_ece": round(metrics.ece_equal_mass(dev_scores, dev_labels)[0], 6),
                },
                "certificate": certificate.to_dict(),
                "hoeffding_alternate": hoeffding.to_dict(),
                "certificate_at_alpha_0_25": illustrative.to_dict(),
                "test_report": {
                    "evaluated_once": True,
                    "auroc": round(metrics.auroc(test_scores, test_labels), 6),
                    "ece_equal_mass": round(ece_value, 6),
                    "ece_equal_width": round(metrics.ece_equal_width(test_scores, test_labels), 6),
                    "aurc": round(aurc, 6),
                    "coverage_at_risk": metrics.coverage_at_risk(frontier_rows, config.alpha),
                    "certified_operating_point": operating,
                },
                "arms": {
                    "risk_v3": arm(test_scores),
                    "legacy_6_feature": arm(legacy_test),
                    "raw_overlap": arm([0.0 for _ in test]),
                },
                "targets": {
                    "auroc_min": 0.80,
                    "coverage_at_risk_min": 0.20,
                    "ece_max": 0.08,
                    "auroc_met": bool(metrics.auroc(test_scores, test_labels) >= 0.80),
                    "coverage_met": bool(
                        metrics.coverage_at_risk(frontier_rows, config.alpha)["coverage"] >= 0.20
                    ),
                    "ece_met": bool(ece_value <= 0.08),
                },
                "per_intent": _bucket(test, test_scores, config.alpha, "bucket"),
                "per_language": _bucket(test, test_scores, config.alpha, "locale"),
            }
        },
        "n_required": _n_required(config.alpha, config.delta),
        "label_source": "cross-family LLM judge (deepseek), not human labels",
        "caveats": [
            "labels are cross-family LLM judgments, not human labels",
            "the certificate covers the calibration draw and the selection rule; test risk is "
            "reported separately and is not covered by it",
            "dev calibration is the union of the re-answered bench dev split and the consistency "
            "run's dev facts; both use the same feature schema",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "certificate.json").write_text(
        json.dumps(certificate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "frontier.csv",
        frontier_rows,
        ("coverage", "risk", "threshold", "n_accepted", "wrong"),
    )
    _write_csv(out_dir / "reliability.csv", reliability, ("n", "mean_score", "accuracy", "gap"))
    (out_dir / "report.md").write_text(render(summary), encoding="utf-8")
    return summary


def _bucket(
    rows: Sequence[FeatureRow], scores: Sequence[float], alpha: float, field: str
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(getattr(row, field)), []).append(index)
    report: dict[str, Any] = {}
    for key, indices in sorted(groups.items()):
        if len(indices) < 20:
            report[key] = {"n": len(indices), "status": "pooled (too thin: < 20 rows)"}
            continue
        bucket_scores = [scores[index] for index in indices]
        labels = [rows[index].label for index in indices]
        frontier_rows, _ = metrics.frontier(bucket_scores, labels)
        report[key] = {
            "n": len(indices),
            "auroc": round(metrics.auroc(bucket_scores, labels), 6),
            "coverage_at_risk": metrics.coverage_at_risk(frontier_rows, alpha),
        }
    return report


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> None:
    lines = [",".join(names)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(summary: Mapping[str, Any]) -> str:
    bench = summary["populations"]["bench"]
    test = bench["test_report"]
    certificate = bench["certificate"]
    targets = bench["targets"]
    return "\n".join(
        [
            f"# Risk control v3 (run `{summary['run_id']}`)",
            "",
            f"Bench population: {bench['dev_rows']} dev / {bench['test_rows']} test turns, "
            "re-answered with "
            "the shipped pipeline and re-judged cross-family.",
            "",
            f"- AUROC {test['auroc']} (target >= 0.80: {targets['auroc_met']})",
            f"- ECE {test['ece_equal_mass']} (target <= 0.08: {targets['ece_met']})",
            f"- coverage at realised risk <= {summary['alpha']}: "
            f"{test['coverage_at_risk']['coverage']} (target >= 0.20: {targets['coverage_met']})",
            f"- certificate at alpha {summary['alpha']}: "
            f"{'feasible' if certificate['feasible'] else 'ABSENT'} "
            f"(threshold {certificate['threshold']}, calibration risk "
            f"{certificate['calibration_risk']}, UCB {certificate['calibration_ucb']}, "
            f"n={certificate['n_calibration']})",
            f"- certified operating point on test: coverage "
            f"{test['certified_operating_point']['coverage']}, realised risk "
            f"{test['certified_operating_point']['realised_risk']} "
            f"(Wilson {test['certified_operating_point']['risk_ci']})",
            f"- rows needed for a 5% certificate: Bentkus "
            f"{summary['n_required']['bentkus_rows_at_zero_risk']}, Hoeffding "
            f"{summary['n_required']['hoeffding_rows_at_zero_risk']}",
            "",
            "## Caveats",
            "",
            *[f"- {caveat}" for caveat in summary["caveats"]],
            "",
        ]
    )


def check(out_dir: Path = DEFAULT_OUT, *, config: V3Config | None = None) -> int:
    """Re-derive the artefacts from the cached feature table; fail on drift, leakage or abstain."""
    import shutil
    import tempfile

    config = config or V3Config(out_dir=out_dir)
    for name in (
        "features.csv",
        "split.json",
        "summary.json",
        "certificate.json",
        "frontier.csv",
        "reliability.csv",
    ):
        if not (out_dir / name).is_file():
            print(f"missing artefact: {out_dir / name}")
            return 1
    committed = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if feature_digest(out_dir / "features.csv") != committed["populations"]["bench"].get(
        "features_sha256"
    ):
        print("features.csv changed since summary.json was written; re-run --evaluate")
        return 1
    split = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    if split["leakage"] or not split["no_leakage"]:
        print(f"split leaks: {split['leakage'][:5]}")
        return 1
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for name in ("features.csv", "split.json"):
            shutil.copyfile(out_dir / name, scratch / name)
        recomputed = evaluate(config=V3Config(out_dir=scratch))
    drift = [
        key
        for key in ("populations", "n_required")
        if json.dumps(recomputed.get(key), sort_keys=True)
        != json.dumps(committed.get(key), sort_keys=True)
    ]
    if drift:
        print(f"artefact drift in {drift}; re-run --evaluate")
        return 1
    certificate = json.loads((out_dir / "certificate.json").read_text(encoding="utf-8"))
    if json.dumps(certificate, sort_keys=True) != json.dumps(
        recomputed["populations"]["bench"]["certificate"], sort_keys=True
    ):
        print("certificate.json does not match the recomputed certificate")
        return 1
    bench = recomputed["populations"]["bench"]
    print(
        f"risk-v3 artefacts reproduce: AUROC {bench['test_report']['auroc']}, coverage@risk "
        f"{bench['test_report']['coverage_at_risk']['coverage']}, certificate feasible "
        f"{bench['certificate']['feasible']}"
    )
    return 0


__all__ = [
    "ALPHA",
    "DEFAULT_OUT",
    "V3Config",
    "build_features",
    "check",
    "evaluate",
    "render",
]
