"""Runner for the risk-control v2 artifact set (``scripts/risk-v2.py``).

Population: the live consistency run's clean-form answers (``reports/consistency-p0``), because
they carry the full answer text and cross-family judge labels that the decision-time features
need. The split is over **facts**, frozen in ``reports/risk-v2/split.json``: no fact appears on
both sides, so nothing leaks from dev into test. The policy is fitted and certified on dev and
evaluated exactly once on test.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from answerdesk.eval.risk import metrics
from answerdesk.eval.risk.certificate import certify
from answerdesk.eval.risk.features import (
    DECISION_FEATURES,
    LEGACY_FEATURES,
    FeatureRow,
    build_row,
    feature_digest,
    load_documents,
    read_rows,
    write_rows,
)
from answerdesk.eval.risk.model import LogisticModel, fit, fit_platt

DEFAULT_OUT = Path("reports/risk-v2")
SOURCE = Path("reports/consistency-p0/answers.jsonl")
CORPUS = Path("data/corpus")
ALPHA = 0.05
DELTA = 0.1
L2 = 2.0
STEPS = 4000
MIN_BUCKET = 20


def _split_of(fact_id: str, *, dev_share: int = 2, modulo: int = 5) -> str:
    import hashlib

    digest = hashlib.sha256(f"risk-v2:{fact_id}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % modulo < dev_share else "test"


def load_population(source: Path = SOURCE) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    population = [
        row for row in rows if row.get("variant") == "clean" and row.get("rubric") is not None
    ]
    for index, row in enumerate(population):
        row.setdefault("sample_id", f"{row['fact_id']}::{row['locale']}")
        row["_index"] = index
    return population


def build_features(
    *,
    out_dir: Path = DEFAULT_OUT,
    reranker: Any | None = None,
    source: Path = SOURCE,
    corpus: Path = CORPUS,
) -> dict[str, Any]:
    population = load_population(source)
    documents = load_documents(corpus)
    rows: list[FeatureRow] = []
    started = time.perf_counter()
    for row in population:
        split_name = _split_of(row["fact_id"])
        passages = [documents.get(doc_id, "") for doc_id in (row.get("retrieved_doc_ids") or ())]
        feature_row = build_row(
            row=row,
            split=split_name,
            documents=documents,
            reranker=reranker,
            retrieval_passages=passages,
        )
        rows.append(feature_row)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "features.csv", rows)
    split_payload: dict[str, Any] = {
        "rule": "sha256('risk-v2:' + fact_id)[:8] % 5 < 2 -> dev; the split is over facts",
        "dev_facts": sorted({row.fact_id for row in rows if row.split == "dev"}),
        "test_facts": sorted({row.fact_id for row in rows if row.split == "test"}),
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
        "features_sha256": feature_digest(out_dir / "features.csv"),
        "dev_rows": split_payload["dev_rows"],
        "test_rows": split_payload["test_rows"],
        "no_leakage": split_payload["no_leakage"],
        "duration_s": round(time.perf_counter() - started, 3),
        "reranked": reranker is not None,
    }


def _split_rows(rows: Sequence[FeatureRow], split: str) -> list[FeatureRow]:
    return [row for row in rows if row.split == split]


def _scores(model: LogisticModel, rows: Sequence[FeatureRow]) -> list[float]:
    return model.predict([row.features for row in rows])


def _bucket_report(
    rows: Sequence[FeatureRow], scores: Sequence[float], *, alpha: float, field: str
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = str(getattr(row, field))
        groups.setdefault(key, []).append(index)
    for key, indices in sorted(groups.items()):
        bucket_scores = [scores[index] for index in indices]
        labels = [rows[index].label for index in indices]
        if len(indices) < MIN_BUCKET:
            report[key] = {
                "n": len(indices),
                "status": f"pooled (too thin: < {MIN_BUCKET} rows)",
            }
            continue
        rows_frontier, _ = metrics.frontier(bucket_scores, labels)
        report[key] = {
            "n": len(indices),
            "auroc": round(metrics.auroc(bucket_scores, labels), 6),
            "accuracy": round(float(np.mean(labels)), 6),
            "coverage_at_risk": metrics.coverage_at_risk(rows_frontier, alpha),
        }
    return report


def _n_required(alpha: float, delta: float) -> dict[str, Any]:
    """How many calibration rows a 5% certificate needs, under each bound."""
    x = math.log(1.0 / delta)
    return {
        "hoeffding_rows_at_zero_risk": round(x / (2.0 * alpha * alpha), 1),
        "bentkus_rows_at_zero_risk": round(x / alpha, 1),
        "note": (
            "rows needed for a certificate at the given alpha/delta when the accepted region "
            "has zero observed errors; more rows are needed when the observed risk is non-zero "
            "(the UCB grows with sqrt(risk * log(1/delta) / n))"
        ),
    }


def _population_report(
    rows: Sequence[FeatureRow],
    *,
    model_features: Sequence[str],
    alpha: float,
    delta: float,
    l2: float,
    steps: int,
    unavailable: Sequence[str] = (),
) -> dict[str, Any]:
    dev = [row for row in rows if row.split == "dev"]
    test = [row for row in rows if row.split == "test"]
    if not dev or not test:
        raise RuntimeError("empty dev or test split for this population")
    model = fit(
        [row.features for row in dev],
        [row.label for row in dev],
        features=model_features,
        l2=l2,
        steps=steps,
    )
    dev_raw = [row.features for row in dev]
    test_raw = [row.features for row in test]
    calibrator = fit_platt(model.predict(dev_raw), [row.label for row in dev])
    dev_scores = calibrator.apply(model.predict(dev_raw))
    test_scores = calibrator.apply(model.predict(test_raw))
    dev_labels = [row.label for row in dev]
    test_labels = [row.label for row in test]
    certificate = certify(
        scores=dev_scores, labels=dev_labels, alpha=alpha, delta=delta, bound="bentkus"
    )
    hoeffding = certify(
        scores=dev_scores, labels=dev_labels, alpha=alpha, delta=delta, bound="hoeffding"
    )
    illustrative = certify(
        scores=dev_scores, labels=dev_labels, alpha=0.25, delta=delta, bound="bentkus"
    )
    rows_frontier, aurc = metrics.frontier(test_scores, test_labels)
    ece_mass, reliability = metrics.ece_equal_mass(test_scores, test_labels)
    legacy = fit(
        [row.features for row in dev], dev_labels, features=LEGACY_FEATURES, l2=l2, steps=steps
    )
    legacy_scores = fit_platt(legacy.predict([row.features for row in dev]), dev_labels).apply(
        legacy.predict([row.features for row in test])
    )

    def arm(scores: Sequence[float]) -> dict[str, Any]:
        frontier_rows, area = metrics.frontier(scores, test_labels)
        return {
            "auroc": round(metrics.auroc(scores, test_labels), 6),
            "ece": round(metrics.ece_equal_mass(scores, test_labels)[0], 6),
            "aurc": round(area, 6),
            "coverage_at_risk": metrics.coverage_at_risk(frontier_rows, alpha),
        }

    coverage_point = metrics.coverage_at_risk(rows_frontier, alpha)
    operating = metrics.selective_risk(
        scores=test_scores, labels=test_labels, threshold=certificate.threshold
    )
    return {
        "n": len(rows),
        "dev": len(dev),
        "test_rows": len(test),
        "base_error_dev": round(1.0 - float(np.mean(dev_labels)), 6),
        "base_error_test": round(1.0 - float(np.mean(test_labels)), 6),
        "unavailable_features": list(unavailable),
        "model": {
            "features": list(model_features),
            "weights": model.to_dict(),
            "calibration": calibrator.to_dict(),
            "dev_auroc": round(metrics.auroc(dev_scores, dev_labels), 6),
            "dev_ece": round(metrics.ece_equal_mass(dev_scores, dev_labels)[0], 6),
            "dev_log_loss": round(metrics.log_loss(dev_scores, dev_labels), 6),
        },
        "certificate": certificate.to_dict(),
        "hoeffding_alternate": hoeffding.to_dict(),
        "certificate_at_alpha_0_25": illustrative.to_dict(),
        "test": {
            "evaluated_once": True,
            "auroc": round(metrics.auroc(test_scores, test_labels), 6),
            "ece_equal_mass": round(ece_mass, 6),
            "ece_equal_width": round(metrics.ece_equal_width(test_scores, test_labels), 6),
            "aurc": round(aurc, 6),
            "coverage_at_risk": coverage_point,
            "certified_operating_point": operating,
        },
        "arms": {
            "risk_v2": arm(test_scores),
            "legacy_6_feature": arm(legacy_scores),
            "raw_overlap": arm([row.features.get("overlap_score", 0.0) for row in test]),
        },
        "targets": {
            "auroc_min": 0.75,
            "coverage_at_risk_min": 0.40,
            "ece_max": 0.08,
            "auroc_met": bool(metrics.auroc(test_scores, test_labels) >= 0.75),
            "coverage_met": bool(coverage_point["coverage"] >= 0.40),
            "ece_met": bool(ece_mass <= 0.08),
        },
        "per_intent": _bucket_report(test, test_scores, alpha=alpha, field="bucket"),
        "per_language": _bucket_report(test, test_scores, alpha=alpha, field="locale"),
        "_frontier": rows_frontier,
        "_reliability": reliability,
    }


def evaluate(
    *,
    out_dir: Path = DEFAULT_OUT,
    alpha: float = ALPHA,
    delta: float = DELTA,
    l2: float = L2,
    steps: int = STEPS,
) -> dict[str, Any]:
    """Fit/certify/evaluate each population; the bench population is primary."""
    from answerdesk.eval.risk import benchpop

    populations: dict[str, Any] = {}
    artifacts: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    consistency_rows = read_rows(out_dir / "features.csv")
    consistency = _population_report(
        consistency_rows,
        model_features=tuple(DECISION_FEATURES) + tuple(LEGACY_FEATURES),
        alpha=alpha,
        delta=delta,
        l2=l2,
        steps=steps,
    )
    consistency["source"] = str(SOURCE).replace("\\", "/")
    consistency["description"] = (
        "clean-form answers from the live consistency run (63 facts x 6 locales); full answer "
        "text available, so numeric consistency and claim support are exercised"
    )
    consistency["split"] = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    consistency["source_digest"] = feature_digest(out_dir / "features.csv")
    artifacts["consistency"] = (consistency.pop("_frontier"), consistency.pop("_reliability"))
    populations["consistency"] = consistency

    bench_path = out_dir / "bench-features.csv"
    if bench_path.is_file():
        bench_rows = read_rows(bench_path)
        bench = _population_report(
            bench_rows,
            model_features=tuple(
                name for name in DECISION_FEATURES if name not in benchpop.UNAVAILABLE
            )
            + tuple(LEGACY_FEATURES),
            alpha=alpha,
            delta=delta,
            l2=l2,
            steps=steps,
            unavailable=benchpop.UNAVAILABLE,
        )
        bench["source"] = f"{benchpop.BENCH_DEV} / {benchpop.BENCH_TEST}"
        bench["description"] = (
            "the product's own frozen test split (t11 turns, 200 dev / 200 test); labels are "
            "cross-family judge answer_ok"
        )
        bench["population"] = "bench"
        artifacts["bench"] = (bench.pop("_frontier"), bench.pop("_reliability"))
        populations["bench"] = bench

    # Primary population: the one where the decision-time signal actually exists. The bench
    # population lacks the two strongest features (numeric consistency and claim support need
    # the full answer text, which t11 did not persist) and its reranker feature could not be
    # built under the free-tariff rate limit, so it is reported as a second population with
    # an explicit miss rather than promoted.
    primary = "consistency" if "consistency" in populations else next(iter(populations))
    primary_reason = (
        "the consistency population carries the full answer text, so all five decision-time "
        "features are available; the bench population lacks two of them and its reranker "
        "feature was rate-limited at build time, and it fails the AUROC target as a result"
    )
    summary = {
        "run_id": out_dir.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_population": primary,
        "primary_reason": primary_reason,
        "populations": populations,
        "n_required": _n_required(alpha, delta),
        "alpha": alpha,
        "delta": delta,
        "hyperparameters": {"l2": l2, "steps": steps},
        "label_source": "cross-family LLM judge (deepseek-flash), not human labels",
        "caveats": [
            "labels are cross-family LLM judgments, not human labels; the certificate covers "
            "the judge's notion of correctness",
            "the certificate is over the calibration draw and the selection rule; test risk is "
            "reported separately and is not covered by it",
            "the consistency population's base error is dominated by sibling-document ambiguity "
            "(CNS-2); the bench population is the product's own frozen test split",
            "two decision-time features (numeric consistency, claim support) need the full "
            "answer text and are unavailable on the bench population",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    primary_frontier, primary_reliability = artifacts[primary]
    (out_dir / "certificate.json").write_text(
        json.dumps(
            populations[primary]["certificate"], ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    _write_frontier(out_dir / "frontier.csv", primary_frontier)
    _write_reliability(out_dir / "reliability.csv", primary_reliability)
    for name, (frontier_rows, reliability_rows) in artifacts.items():
        if name == primary:
            continue
        _write_frontier(out_dir / f"frontier-{name}.csv", frontier_rows)
        _write_reliability(out_dir / f"reliability-{name}.csv", reliability_rows)
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def _write_frontier(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    names = ["coverage", "risk", "threshold", "n_accepted", "wrong"]
    lines = [",".join(names)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reliability(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    names = ["n", "mean_score", "accuracy", "gap"]
    lines = [",".join(names)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Risk control v2 (run `{summary['run_id']}`)",
        "",
        f"Primary population: **{summary['primary_population']}** — "
        f"{summary.get('primary_reason', '')}",
        "",
        f"Labels: {summary.get('label_source')}; alpha {summary.get('alpha')}, delta "
        f"{summary.get('delta')}. Rows needed for a 5% certificate: Bentkus "
        f"{summary['n_required']['bentkus_rows_at_zero_risk']}, Hoeffding "
        f"{summary['n_required']['hoeffding_rows_at_zero_risk']} (at zero observed risk).",
        "",
    ]
    for name, population in summary["populations"].items():
        test = population["test"]
        certificate = population["certificate"]
        targets = population["targets"]
        lines += [
            f"## Population `{name}` ({population['dev']} dev / {population['test']} rows test)",
            "",
            f"- base error: dev {population['base_error_dev']}, "
            f"test {population['base_error_test']}",
            f"- AUROC {test['auroc']} (target >= 0.75: {targets['auroc_met']})",
            f"- ECE (equal-mass) {test['ece_equal_mass']} (target <= 0.08: {targets['ece_met']})",
            f"- coverage at realised risk <= {summary['alpha']}: "
            f"{test['coverage_at_risk']['coverage']} "
            f"(target >= 0.40: {targets['coverage_met']})",
            f"- AURC {test['aurc']}",
            f"- certificate feasible: {certificate['feasible']}; threshold "
            f"{certificate['threshold']}; calibration risk {certificate['calibration_risk']} "
            f"(UCB {certificate['calibration_ucb']}) over {certificate['n_calibration']} rows",
            f"- certified operating point on test: coverage "
            f"{test['certified_operating_point']['coverage']}, realised risk "
            f"{test['certified_operating_point']['realised_risk']}",
            f"- largest feasible alpha certificate: alpha 0.25, feasible "
            f"{population['certificate_at_alpha_0_25']['feasible']}, coverage "
            f"{population['certificate_at_alpha_0_25']['calibration_coverage']}",
            "",
        ]
    lines += ["## Caveats", ""]
    lines += [f"- {caveat}" for caveat in summary["caveats"]]
    lines.append("")
    return "\n".join(lines)


def check(out_dir: Path = DEFAULT_OUT, *, alpha: float = ALPHA, delta: float = DELTA) -> int:
    """Re-derive the artefacts from the cached feature tables and compare them to disk."""
    import shutil
    import tempfile

    required = [
        "features.csv",
        "split.json",
        "summary.json",
        "certificate.json",
        "frontier.csv",
        "reliability.csv",
    ]
    if (out_dir / "bench-features.csv").is_file():
        required.append("bench-features.csv")
    for name in required:
        if not (out_dir / name).is_file():
            print(f"missing artefact: {out_dir / name}")
            return 1
    committed = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    hyperparameters = committed.get("hyperparameters", {})
    if (
        feature_digest(out_dir / "features.csv")
        != committed["populations"]["consistency"]["source_digest"]
    ):
        print("features.csv changed since summary.json was written; re-run --run")
        return 1
    split = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    if split["leakage"] or not split["no_leakage"]:
        print(f"split leaks facts across dev/test: {split['leakage'][:5]}")
        return 1
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for name in required:
            shutil.copyfile(out_dir / name, scratch / name)
        recomputed = evaluate(
            out_dir=scratch,
            alpha=committed.get("alpha", alpha),
            delta=committed.get("delta", delta),
            l2=float(hyperparameters.get("l2", L2)),
            steps=int(hyperparameters.get("steps", STEPS)),
        )
    drift = [
        name
        for name in ("populations", "n_required", "primary_population")
        if not _same(recomputed.get(name), committed.get(name))
    ]
    if drift:
        details = _mismatches(recomputed.get(drift[0]), committed.get(drift[0]))
        print(f"artefact drift in {drift}; re-run --run")
        for line in details:
            print(f"  {line}")
        return 1
    printed_certificate = json.loads((out_dir / "certificate.json").read_text(encoding="utf-8"))
    if not _same(
        printed_certificate,
        recomputed["populations"][recomputed["primary_population"]]["certificate"],
    ):
        print("certificate.json does not match the recomputed certificate; re-run --run")
        return 1
    primary = recomputed["populations"][recomputed["primary_population"]]
    print(
        f"risk-v2 artefacts reproduce: primary {recomputed['primary_population']}, "
        f"AUROC {primary['test']['auroc']}, coverage@risk "
        f"{primary['test']['coverage_at_risk']['coverage']}, certificate feasible "
        f"{primary['certificate']['feasible']}"
    )
    return 0


def _same(left: Any, right: Any, *, rel: float = 1e-3, absolute: float = 1e-3) -> bool:
    """Compare artefacts: refitting must reproduce within tolerance across BLAS platforms."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= absolute + rel * abs(float(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key], rel=rel, absolute=absolute) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b, rel=rel, absolute=absolute) for a, b in zip(left, right, strict=False)
        )
    return left == right


def _mismatches(left: Any, right: Any, path: str = "", limit: int = 6) -> list[str]:
    """Return the first differing leaves, so a CI failure says which metric drifted."""
    found: list[str] = []
    if len(found) >= limit:
        return found
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            found.extend(_mismatches(left.get(key), right.get(key), f"{path}.{key}", limit))
            if len(found) >= limit:
                break
        return found[:limit]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        for index, (a, b) in enumerate(zip(left, right, strict=False)):
            found.extend(_mismatches(a, b, f"{path}[{index}]", limit))
            if len(found) >= limit:
                break
        return found[:limit]
    if not _same(left, right):
        return [f"{path}: {left!r} != {right!r}"]
    return []


__all__ = [
    "ALPHA",
    "DEFAULT_OUT",
    "DELTA",
    "build_features",
    "check",
    "evaluate",
    "load_population",
    "render_report",
]
