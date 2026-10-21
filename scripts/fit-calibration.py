"""Fit a *calibrated* answerability scorer on dev turns and re-decide the test turns offline.

The raw decision signal (`OverlapScorer.score`) saturates at 1.0 for most queries, so a
conformal quantile on it collapses to zero and the resulting threshold controls nothing. This
script replaces it with a small L2-regularised logistic regression over features that are
available at decision time, fitted **on dev answers only** (never on test), and then

1. fits the split-conformal q-hat on the *calibrated* dev scores,
2. re-decides every test turn offline with that threshold (no model calls),
3. reports realised risk / coverage with Wilson intervals, ECE, reliability bins and the full
   risk-coverage curve,
4. compares against the raw overlap scorer as an ablation.

Everything is reconstructed from artefacts that already exist: dev/test turn CSVs (with the
judge labels), the frozen corpus and a local BM25 index (no API calls). Retrieval scores are
recomputed with the same lexical channel the answer arms used.

    uv run python scripts/fit-calibration.py --run reports/bench --alpha 0.05

Output: ``reports/<run>/calibration/calibrated-scorer.json`` (+ ``reliability.csv``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.config import load_settings  # noqa: E402
from answerdesk.eval.arms import BM25Arm  # noqa: E402
from answerdesk.eval.corpus import load_corpus  # noqa: E402
from answerdesk.eval.stats import wilson_interval  # noqa: E402

FEATURES = (
    "overlap_score",
    "bm25_top",
    "bm25_margin",
    "bm25_above_gate",
    "citation_density",
    "log_claims",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    return float(value)  # type: ignore[arg-type]


def _features(row: dict[str, str], arm: BM25Arm, gate: float = 0.0) -> list[float]:
    question = row.get("question") or ""
    scored = []
    if question and not question.startswith("["):
        try:
            scored = [float(getattr(passage, "score", 0.0)) for passage in arm.search(question, 20)]
        except Exception:
            scored = []
    top = scored[0] if scored else 0.0
    second = scored[1] if len(scored) > 1 else 0.0
    above = float(sum(1 for score in scored if score > gate)) / 20.0
    claims = max(0.0, _float(row.get("claims")))
    cited = len([c for c in (row.get("cited") or "").split(";") if c])
    density = cited / claims if claims else 0.0
    return [
        _float(row.get("p_correct")),
        top,
        top - second,
        above,
        density,
        math.log1p(claims),
    ]


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, steps: int = 4000
) -> np.ndarray:
    """Plain gradient descent on the L2-regularised log-loss (numpy only, deterministic)."""
    n, d = x.shape
    weights = np.zeros(d + 1)
    design = np.hstack([np.ones((n, 1)), x])
    for _ in range(steps):
        logits = design @ weights
        probs = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probs - y) / n
        gradient[1:] += l2 * weights[1:] / n
        weights -= 0.5 * gradient  # fixed step; the loss is convex and the data scaled
    return weights


def _predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((x.shape[0], 1)), x])
    return 1.0 / (1.0 + np.exp(-(design @ weights)))


def _scaled(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def _conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    risks = 1.0 - scores
    ordered = np.sort(risks)
    rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * (1.0 - alpha))))
    return float(ordered[rank - 1])


def _ece(
    scores: np.ndarray, labels: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    total = 0.0
    rows: list[dict[str, float]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores < high if index < bins - 1 else scores <= high)
        if not mask.any():
            continue
        mean_score = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        share = float(mask.mean())
        total += share * abs(accuracy - mean_score)
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "n": int(mask.sum()),
                "mean_score": mean_score,
                "accuracy": accuracy,
            }
        )
    return total, rows


def _decide(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold


def _report(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    name: str,
    target: float,
) -> dict[str, Any]:
    answered = _decide(scores, threshold)
    n = len(scores)
    answered_n = int(answered.sum())
    wrong = int((labels[answered] < 0.5).sum())
    coverage = wilson_interval(answered_n, n)
    risk = wilson_interval(wrong, answered_n) if answered_n else None
    ece, bins = _ece(scores, labels.astype(float))
    curve = []
    order = np.argsort(-scores)
    cumulative_wrong = 0
    for position, index in enumerate(order, start=1):
        if not labels[index]:
            cumulative_wrong += 1
        if position % max(1, n // 20) == 0 or position == n:
            curve.append(
                {
                    "coverage": position / n,
                    "risk": cumulative_wrong / position,
                    "threshold": float(scores[index]),
                }
            )
    return {
        "scorer": name,
        "threshold": round(float(threshold), 6),
        "n": n,
        "answered": answered_n,
        "coverage": round(coverage.value, 6),
        "coverage_ci": [round(coverage.low, 6), round(coverage.high, 6)],
        "realised_risk": round(risk.value, 6) if risk else None,
        "risk_ci": [round(risk.low, 6), round(risk.high, 6)] if risk else None,
        "target_alpha": target,
        # the gate must clear alpha on the *upper* bound of the interval, not the point estimate
        "risk_control_held": bool(risk is not None and risk.high <= target),
        "rule": "risk_control_held = Wilson upper bound of realised risk <= alpha",
        "ece": round(ece, 6),
        "reliability": bins,
        "curve": curve,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "reports" / "bench")
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=2.0)
    args = parser.parse_args(argv)
    run: Path = args.run
    dev_path = args.dev or run / "_state" / "dev-turns.csv"
    test_path = args.test or run / "answer" / "primary" / "turns.csv"
    dev = _read(dev_path)
    test = _read(test_path)
    if not dev or not test:
        print("missing dev or test turns", file=sys.stderr)
        return 2
    settings = load_settings(profile=None)
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    arm = BM25Arm(corpus)

    x_dev = np.array([_features(row, arm) for row in dev])
    y_dev = np.array([1.0 if row.get("answer_ok") == "True" else 0.0 for row in dev])
    x_test = np.array([_features(row, arm) for row in test])
    y_test = np.array([1.0 if row.get("answer_ok") == "True" else 0.0 for row in test])

    x_dev_s, x_test_s = _scaled(x_dev, x_test)
    weights = _fit_logistic(x_dev_s, y_dev, l2=args.l2)
    dev_scores = _predict(weights, x_dev_s)
    test_scores = _predict(weights, x_test_s)

    qhat = _conformal_qhat(dev_scores, args.alpha)
    threshold = max(0.0, 1.0 - qhat)
    # operating point chosen on dev (risk <= 2%), never on test
    dev_target = 0.02
    dev_order = np.argsort(-dev_scores)
    dev_curve = []
    dev_wrong = 0
    for position, index in enumerate(dev_order, start=1):
        if y_dev[index] < 0.5:
            dev_wrong += 1
        dev_curve.append((position / len(dev_scores), dev_wrong / position))
    feasible = [point for point in dev_curve if point[1] <= dev_target]
    dev_operating_coverage = feasible[-1][0] if feasible else 0.0
    dev_operating_threshold = (
        float(
            np.sort(dev_scores)[::-1][max(0, round(dev_operating_coverage * len(dev_scores)) - 1)]
        )
        if feasible
        else 1.0
    )

    calibrated = _report(
        test_scores, y_test, threshold, name="calibrated-logistic", target=args.alpha
    )
    operating = _report(
        test_scores,
        y_test,
        dev_operating_threshold,
        name="calibrated (dev operating point)",
        target=dev_target,
    )
    raw = _report(
        np.array([_float(row.get("p_correct")) for row in test]),
        y_test,
        5e-1,
        name="raw-overlap@0.5",
        target=args.alpha,
    )
    raw_engine = _report(
        np.array([_float(row.get("p_correct")) for row in test]),
        y_test,
        1.0,
        name="raw-overlap@1.0 (what the engine actually did)",
        target=args.alpha,
    )
    payload = {
        "label_source": "judge labels (cross-family LLM judge: deepseek-flash on qwen answers)",
        "label_caveat": (
            "targets are judge labels, not human labels; the calibration inherits the judge's "
            "error and its per-claim rubric, so a human-labelled recalibration is required "
            "before any production gate"
        ),
        "risk_gate_rule": (
            "risk control holds only if the Wilson 95% upper bound of the realised risk on the "
            "answered set is <= alpha; a point estimate below alpha with an upper bound above "
            "it does not count"
        ),
        "features": list(FEATURES),
        "fit_split": "dev only",
        "n_dev": len(dev),
        "n_test": len(test),
        "alpha": args.alpha,
        "l2": args.l2,
        "coefficients": [round(float(value), 6) for value in weights],
        "dev_accuracy": round(float(((dev_scores >= 0.5).astype(float) == y_dev).mean()), 6),
        "q_hat": round(qhat, 6),
        "threshold": round(threshold, 6),
        "dev_operating_point": {
            "target": dev_target,
            "feasible": bool(feasible),
            "coverage": round(dev_operating_coverage, 6),
            "threshold": round(dev_operating_threshold, 6),
        },
        "test": {
            "calibrated_conformal": calibrated,
            "calibrated_operating": operating,
            "raw_overlap": raw,
            "raw_overlap_engine": raw_engine,
        },
        "verdict": {
            "risk_control_held": calibrated["risk_control_held"],
            "note": (
                "the calibrated scorer's realised risk is "
                f"{calibrated['realised_risk']} against alpha={args.alpha}; "
                + (
                    "the gate holds on the held-out split"
                    if calibrated["risk_control_held"]
                    else "the gate does NOT hold and no risk-control wording may ship"
                )
            ),
        },
    }
    (run / "calibration").mkdir(parents=True, exist_ok=True)
    (run / "calibration" / "calibrated-scorer.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (run / "calibration" / "reliability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["bin_low", "bin_high", "n", "mean_score", "accuracy"]
        )
        writer.writeheader()
        for row in calibrated["reliability"]:
            writer.writerow(row)
    print(
        json.dumps(
            {k: payload[k] for k in ("q_hat", "threshold", "dev_operating_point", "verdict")},
            ensure_ascii=False,
        )
    )
    print(
        "test:",
        {
            name: {
                "coverage": value["coverage"],
                "risk": value["realised_risk"],
                "ece": value["ece"],
                "held": value["risk_control_held"],
            }
            for name, value in payload["test"].items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
