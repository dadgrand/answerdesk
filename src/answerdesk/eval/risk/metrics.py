"""Selective-prediction metrics for the risk-controlled policy: AUROC, ECE, AURC, frontier."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from answerdesk.eval.stats import wilson_interval


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC (ties get average ranks); 0.5 when one class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = s[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2.0
            start = index
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def ece_equal_mass(
    scores: Sequence[float], labels: Sequence[int], bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error with equal-mass bins (the honest choice for skewed scores)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(s) == 0:
        return 0.0, []
    order = np.argsort(-s, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    total = 0.0
    rows: list[dict[str, float]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        mean_score = float(s[chunk].mean())
        accuracy = float(y[chunk].mean())
        share = len(chunk) / len(s)
        total += share * abs(accuracy - mean_score)
        rows.append(
            {
                "n": len(chunk),
                "mean_score": round(mean_score, 6),
                "accuracy": round(accuracy, 6),
                "gap": round(accuracy - mean_score, 6),
            }
        )
    return float(total), rows


def ece_equal_width(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(s) == 0:
        return 0.0
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (s >= low) & (s < high if index < bins - 1 else s <= high)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(s[mask].mean()))
    return float(total)


def frontier(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[dict[str, Any]], float]:
    """Risk-coverage curve and its area (AURC); lower AURC is better."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    rows: list[dict[str, Any]] = []
    wrong = 0
    area = 0.0
    previous_coverage = 0.0
    for position, (score, label) in enumerate(zip(s, y, strict=True), start=1):
        if label == 0:
            wrong += 1
        coverage = position / len(s)
        risk = wrong / position
        area += risk * (coverage - previous_coverage)
        previous_coverage = coverage
        rows.append(
            {
                "coverage": round(coverage, 6),
                "risk": round(risk, 6),
                "threshold": round(float(score), 6),
                "n_accepted": position,
                "wrong": wrong,
            }
        )
    return rows, float(area)


def coverage_at_risk(rows: Sequence[dict[str, Any]], alpha: float) -> dict[str, Any]:
    """Largest coverage whose realised risk is <= alpha (an oracle frontier point)."""
    feasible = [row for row in rows if float(row["risk"]) <= alpha]
    if not feasible:
        return {
            "alpha": alpha,
            "coverage": 0.0,
            "risk": None,
            "threshold": None,
            "n_accepted": 0,
            "label": "oracle frontier: no prefix meets the target",
        }
    best = max(feasible, key=lambda row: float(row["coverage"]))
    interval = wilson_interval(int(best["wrong"]), int(best["n_accepted"]))
    return {
        "alpha": alpha,
        "coverage": float(best["coverage"]),
        "risk": float(best["risk"]),
        "risk_ci": [round(interval.low, 6), round(interval.high, 6)],
        "threshold": float(best["threshold"]),
        "n_accepted": int(best["n_accepted"]),
        "label": "oracle frontier point (chosen on test for reporting only, not deployable)",
    }


def selective_risk(
    *, scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    accepted = s >= threshold
    n = len(s)
    n_accepted = int(accepted.sum())
    wrong = int((y[accepted] == 0).sum())
    interval = wilson_interval(wrong, n_accepted) if n_accepted else None
    return {
        "threshold": round(float(threshold), 6),
        "n": n,
        "n_accepted": n_accepted,
        "coverage": round(n_accepted / n, 6) if n else 0.0,
        "wrong": wrong,
        "realised_risk": round(wrong / n_accepted, 6) if n_accepted else None,
        "risk_ci": [round(interval.low, 6), round(interval.high, 6)] if interval else None,
    }


def reliability_table(
    scores: Sequence[float], labels: Sequence[int], bins: int = 10
) -> list[dict[str, Any]]:
    _, rows = ece_equal_mass(scores, labels, bins=bins)
    return rows


def log_loss(scores: Sequence[float], labels: Sequence[int]) -> float:
    epsilon = 1e-12
    total = 0.0
    for score, label in zip(scores, labels, strict=True):
        clipped = min(1.0 - epsilon, max(epsilon, float(score)))
        total += -math.log(clipped if label else 1.0 - clipped)
    return total / max(1, len(scores))


__all__ = [
    "auroc",
    "coverage_at_risk",
    "ece_equal_mass",
    "ece_equal_width",
    "frontier",
    "log_loss",
    "reliability_table",
    "selective_risk",
]
