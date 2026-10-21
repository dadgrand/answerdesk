"""Fitting fusion parameters on the dev split — and the leakage guard that enforces it.

INV-09 is blunt: *calibration and fusion artifacts are only ever fit on the dev split; the
test split is read-only for reporting.* This module is where that invariant is
mechanically enforced rather than promised:

* :func:`assert_disjoint` raises :class:`LeakageError` if any item id appears in both
  splits, and every fitting entry point calls it **before** evaluating anything;
* the artifact records the dev split hash, the test split hash (as an exclusion
  fingerprint) and the corpus hash, so a reviewer can re-derive what was fitted on what;
* the optimised metric is dev-only by construction: the caller passes an ``evaluate``
  callback, and the fitting code has no access to any other data source.

Two documented procedures are implemented (FR-RET-05):

``coordinate_search``
    Evaluate a grid of RRF ``k`` values, then a grid of convex-combination ``alpha``
    values, each on the dev split; keep the best arm per family and then the better
    family. Cheap, deterministic, and it reports the whole grid so a flat optimum is
    visible instead of hidden.

``logistic_regression``
    Treat fusion as a tiny ranker: features are the per-channel normalised scores, the
    label is "this candidate is gold", and the fitted coefficients become the convex
    weights. Implemented with plain numpy gradient descent (no scikit-learn dependency,
    fixed iteration count, zero initialisation) so it is reproducible on any machine.

The artifact is JSON with ``extra="forbid"`` validation on load: a hand-edited artifact
that gains a field is rejected, not silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from answerdesk.errors import AnswerdeskError
from answerdesk.index.store import EmbedderInfo
from answerdesk.retrieval.fusion import FusionParams, Normalization
from answerdesk.retrieval.query_analysis import QUERY_ANALYSIS_VERSION

FUSION_ARTIFACT_SCHEMA = "1.0.0"
DEFAULT_ARTIFACT_NAME = "fusion.json"

#: Grids the coordinate search walks. Deliberately small and fixed (reproducibility).
RRF_K_GRID: tuple[int, ...] = (10, 20, 40, 60, 80, 100)
ALPHA_GRID: tuple[float, ...] = (0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)


class LeakageError(AnswerdeskError):
    """A test-split item was about to influence a fitted parameter (INV-09)."""

    code = "split_leakage"
    http_status = 500


class SplitProvenance(BaseModel):
    """Identity of one split: name, size and a hash over its sorted item ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["dev", "test"]
    hash: str = Field(min_length=64, max_length=64)
    n_items: int = Field(ge=0)
    n_documents: int = Field(default=0, ge=0)
    salt: str = ""


class CorpusProvenance(BaseModel):
    """Which corpus the parameters were fitted against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    hash: str = Field(default="", min_length=0, max_length=64)


class FusionArtifact(BaseModel):
    """The persisted, reviewable result of fitting (FR-RET-05)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = FUSION_ARTIFACT_SCHEMA
    artifact_id: str = Field(min_length=1)
    method: Literal["coordinate_search", "logistic_regression"]
    metric: str = Field(min_length=1)
    metric_value: float
    params: FusionParams
    split: SplitProvenance
    excluded_split: SplitProvenance
    corpus: CorpusProvenance
    embedder: EmbedderInfo
    query_analysis: str = QUERY_ANALYSIS_VERSION
    seed: int = 1337
    fitted_at: str = ""
    notes: list[str] = Field(default_factory=list)
    search_history: list[dict[str, float | int | str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _splits_are_distinct(self) -> FusionArtifact:
        if self.split.hash == self.excluded_split.hash:
            raise ValueError("a fusion artifact cannot be fitted and reported on one split")
        if self.split.name == self.excluded_split.name:
            raise ValueError("the fitting split and the excluded split must differ by name")
        return self

    def fingerprint(self) -> str:
        """Stable identity of the *parameters*, excluding the wall-clock timestamp.

        Two fitting runs on the same data must produce the same fingerprint; only
        ``fitted_at`` may differ (NFR-DET-02).
        """
        payload = self.model_dump(mode="json")
        payload.pop("fitted_at", None)
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> FusionArtifact:
        source = Path(path)
        if not source.is_file():
            raise LeakageError(
                f"no fusion artifact at {source}; run `python -m answerdesk.retrieval.ablation` "
                "first (it fits on the dev split and writes this file)"
            )
        return cls.model_validate(json.loads(source.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FusionObservation:
    """One training row for the logistic fit: channel scores for one candidate."""

    features: dict[str, float]
    label: int


def split_hash(item_ids: Sequence[str]) -> str:
    """sha256 over the sorted item ids of a split — its reviewable identity."""
    digest = hashlib.sha256()
    for item_id in sorted(set(item_ids)):
        digest.update(item_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_disjoint(dev_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    """Raise :class:`LeakageError` if any item id is in both splits (INV-09)."""
    overlap = sorted(set(dev_ids) & set(test_ids))
    if overlap:
        raise LeakageError(
            "test-split items must never enter fusion fitting (INV-09): "
            f"{len(overlap)} overlapping ids, first: {overlap[:5]}"
        )


def coordinate_search(
    evaluate: Callable[[FusionParams], float],
    *,
    rrf_grid: Sequence[int] = RRF_K_GRID,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    normalization: Normalization = Normalization.MINMAX,
) -> tuple[FusionParams, float, list[dict[str, float | int | str]]]:
    """Grid-search RRF ``k`` and convex ``alpha``, keep the better family.

    Tie-break rule (documented, deterministic): convex combination wins ties, because the
    requirements ship the tuned CC as the path (D-03) and a tie means RRF bought nothing.
    """
    history: list[dict[str, float | int | str]] = []
    best_rrf: tuple[FusionParams, float] | None = None
    for k in rrf_grid:
        params = FusionParams(fuser="rrf", rrf_k=int(k))
        value = float(evaluate(params))
        history.append({"fuser": "rrf", "rrf_k": int(k), "value": round(value, 6)})
        if best_rrf is None or value > best_rrf[1]:
            best_rrf = (params, value)

    best_cc: tuple[FusionParams, float] | None = None
    for alpha in alpha_grid:
        params = FusionParams(fuser="cc", alpha=float(alpha), normalization=normalization)
        value = float(evaluate(params))
        history.append({"fuser": "cc", "alpha": round(float(alpha), 4), "value": round(value, 6)})
        if best_cc is None or value > best_cc[1]:
            best_cc = (params, value)

    if best_cc is None and best_rrf is None:  # pragma: no cover - grids are non-empty
        raise LeakageError("empty fusion grids: nothing to fit")
    if best_cc is not None and (best_rrf is None or best_cc[1] >= best_rrf[1]):
        return best_cc[0], best_cc[1], history
    assert best_rrf is not None  # for mypy; guaranteed by the branch above
    return best_rrf[0], best_rrf[1], history


def fit_logistic_weights(
    observations: Sequence[FusionObservation],
    *,
    l2: float = 1e-3,
    learning_rate: float = 0.5,
    iterations: int = 400,
    channels: Sequence[str] = ("bm25", "dense"),
) -> dict[str, float]:
    """Fit convex-channel weights by L2-regularised logistic regression (numpy only).

    The model is ``P(gold | x) = σ(b + w · x)`` with ``x`` the normalised channel scores. The
    **intercept ``b``** is essential: candidates are mostly non-gold (roughly 8 gold chunks in
    a 50-candidate window), so without a bias term the optimiser can only reach the base rate
    by pushing every feature weight negative — which, after the non-negativity clip, collapses
    to the equal-weight fallback and silently discards the learned signal. The returned weights
    are ``w`` clipped at 0 and normalised to sum to 1, ready for
    :class:`answerdesk.retrieval.fusion.ConvexFusion`.

    Why logistic regression at all: it answers "which channel's score actually predicts
    relevance" instead of assuming a 50/50 split. Why so small: 3,000 chunks do not support
    anything larger, and a deterministic 400-step gradient descent on a convex loss is
    reproducible in a way that a stochastic optimiser is not.
    """
    import numpy as np

    names = list(channels)
    if not observations:
        equal = 1.0 / len(names)
        return {name: equal for name in names}

    features = np.array(
        [[float(row.features.get(name, 0.0)) for name in names] for row in observations],
        dtype=np.float64,
    )
    # Design matrix with a leading intercept column; the intercept is fitted, never weighted.
    matrix = np.hstack([np.ones((features.shape[0], 1), dtype=np.float64), features])
    labels = np.array([1.0 if row.label else 0.0 for row in observations], dtype=np.float64)
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    for _ in range(max(1, iterations)):
        logits = matrix @ coefficients
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = matrix.T @ (probabilities - labels) / len(labels)
        gradient[1:] += l2 * coefficients[1:]
        coefficients -= learning_rate * gradient
    weights = coefficients[1:]
    clipped = np.clip(weights, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        equal = 1.0 / len(names)
        return {name: equal for name in names}
    return {
        name: round(float(value / total), 6) for name, value in zip(names, clipped, strict=True)
    }


def fit_fusion_artifact(
    *,
    evaluate: Callable[[FusionParams], float],
    dev_split: SplitProvenance,
    test_split: SplitProvenance,
    dev_item_ids: Sequence[str],
    test_item_ids: Sequence[str],
    corpus: CorpusProvenance,
    embedder: EmbedderInfo,
    observations: Sequence[FusionObservation] | None = None,
    method: Literal["coordinate_search", "logistic_regression"] = "coordinate_search",
    metric: str = "nDCG@10",
    metric_value: float | None = None,
    seed: int = 1337,
    rrf_grid: Sequence[int] = RRF_K_GRID,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    fitted_at: str | None = None,
    notes: Sequence[str] = (),
) -> FusionArtifact:
    """Fit on dev, record provenance, refuse if the dev/test splits overlap (FR-RET-05)."""
    assert_disjoint(dev_item_ids, test_item_ids)
    if dev_split.hash != split_hash(dev_item_ids):
        raise LeakageError(
            "the dev split provenance hash does not match the fitting ids: the artifact "
            "would claim to be fitted on data it never saw"
        )
    if test_split.hash != split_hash(test_item_ids):
        raise LeakageError("the excluded-split provenance hash does not match its item ids")

    history: list[dict[str, float | int | str]] = []
    if method == "logistic_regression":
        if observations is None:
            raise ValueError("logistic_regression needs observations")
        weights = fit_logistic_weights(observations)
        params = FusionParams(fuser="cc", alpha=None, weights=weights)
        value = float(evaluate(params)) if metric_value is None else float(metric_value)
        history = [{"fuser": "cc", **{f"w_{name}": weight for name, weight in weights.items()}}]
    else:
        params, value, history = coordinate_search(
            evaluate, rrf_grid=rrf_grid, alpha_grid=alpha_grid
        )

    timestamp = fitted_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    params_blob = json.dumps(params.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    params_digest = hashlib.sha256(params_blob).hexdigest()[:8]
    artifact_id = f"fusion-{method[:4]}-{params.fuser}-{params_digest}"
    return FusionArtifact(
        artifact_id=artifact_id,
        method=method,
        metric=metric,
        metric_value=round(float(value), 6),
        params=params,
        split=dev_split,
        excluded_split=test_split,
        corpus=corpus,
        embedder=embedder,
        seed=seed,
        fitted_at=timestamp,
        notes=[*notes],
        search_history=history,
    )


__all__ = [
    "ALPHA_GRID",
    "DEFAULT_ARTIFACT_NAME",
    "FUSION_ARTIFACT_SCHEMA",
    "RRF_K_GRID",
    "CorpusProvenance",
    "FusionArtifact",
    "FusionObservation",
    "LeakageError",
    "SplitProvenance",
    "assert_disjoint",
    "coordinate_search",
    "fit_fusion_artifact",
    "fit_logistic_weights",
    "split_hash",
]
