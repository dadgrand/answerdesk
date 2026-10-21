"""L2-regularised logistic answerability model (numpy only, deterministic)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticModel:
    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    l2: float
    steps: int

    def predict(self, rows: Sequence[dict[str, float]]) -> list[float]:
        design = self._design(rows)
        logits = design @ np.asarray(self.weights, dtype=np.float64)
        return [float(value) for value in 1.0 / (1.0 + np.exp(-logits))]

    def _design(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row.get(name, 0.0)) for name in self.features] for row in rows],
            dtype=np.float64,
        )
        if matrix.size == 0:
            return np.ones((0, len(self.features) + 1), dtype=np.float64)
        scaled = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.hstack([np.ones((scaled.shape[0], 1)), scaled])

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "mean": [round(float(value), 9) for value in self.mean],
            "scale": [round(float(value), 9) for value in self.scale],
            "weights": [round(float(value), 9) for value in self.weights],
            "l2": self.l2,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LogisticModel:
        return cls(
            features=tuple(payload["features"]),  # type: ignore[arg-type]
            mean=tuple(payload["mean"]),  # type: ignore[arg-type]
            scale=tuple(payload["scale"]),  # type: ignore[arg-type]
            weights=tuple(payload["weights"]),  # type: ignore[arg-type]
            l2=float(payload["l2"]),  # type: ignore[arg-type]
            steps=int(str(payload["steps"])),
        )


def fit(
    rows: Sequence[dict[str, float]],
    labels: Sequence[int],
    *,
    features: Sequence[str],
    l2: float = 1.0,
    steps: int = 4000,
    learning_rate: float = 0.5,
) -> LogisticModel:
    """Gradient descent on the L2-regularised log-loss; deterministic, no random init."""
    matrix = np.asarray(
        [[float(row.get(name, 0.0)) for name in features] for row in rows], dtype=np.float64
    )
    y = np.asarray(labels, dtype=np.float64)
    if matrix.size == 0:
        raise ValueError("cannot fit on an empty design matrix")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0.0] = 1.0
    scaled = (matrix - mean) / scale
    design = np.hstack([np.ones((scaled.shape[0], 1)), scaled])
    weights = np.zeros(design.shape[1])
    n = design.shape[0]
    for _ in range(steps):
        probabilities = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (probabilities - y) / n
        gradient[1:] += l2 * weights[1:] / n
        weights -= learning_rate * gradient
    return LogisticModel(
        features=tuple(features),
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        l2=l2,
        steps=steps,
    )


@dataclass(frozen=True)
class PlattCalibrator:
    """One-dimensional logistic recalibration of a model's logits, fitted on dev only."""

    intercept: float
    slope: float

    def apply(self, probabilities: Sequence[float]) -> list[float]:
        calibrated: list[float] = []
        for probability in probabilities:
            clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
            logit = math.log(clipped / (1.0 - clipped))
            value = self.intercept + self.slope * logit
            calibrated.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))))
        return calibrated

    def to_dict(self) -> dict[str, float]:
        return {"intercept": round(self.intercept, 9), "slope": round(self.slope, 9)}


def fit_platt(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    steps: int = 3000,
    learning_rate: float = 0.5,
) -> PlattCalibrator:
    """Fit intercept+slope on dev logits (deterministic gradient descent)."""
    logits = []
    for probability in probabilities:
        clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
        logits.append(math.log(clipped / (1.0 - clipped)))
    y = list(labels)
    n = len(logits)
    if n == 0:
        return PlattCalibrator(0.0, 1.0)
    intercept, slope = 0.0, 1.0
    for _ in range(steps):
        grad_intercept = 0.0
        grad_slope = 0.0
        for logit, label in zip(logits, y, strict=True):
            value = intercept + slope * logit
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
            error = probability - label
            grad_intercept += error
            grad_slope += error * logit
        intercept -= learning_rate * grad_intercept / n
        slope -= learning_rate * grad_slope / n
    return PlattCalibrator(intercept=intercept, slope=slope)


__all__ = ["LogisticModel", "PlattCalibrator", "fit", "fit_platt"]
