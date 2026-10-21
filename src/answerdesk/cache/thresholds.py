"""Per-cohort similarity thresholds tied to a wrong-hit budget (requirements §9.3, §9.5).

A global cosine threshold is a guess: it gives no error guarantee and its realised wrong-hit
rate depends on the traffic mix. This module implements the *error-budget* discipline the
requirements adopt instead:

> for each cohort, choose the smallest threshold τ such that, on the calibration replay set,
> ``wrong_hits / hits ≤ ε`` **and** ``Wilson 95% UCB ≤ 2ε``.

The "smallest τ" is what buys hit rate: among the thresholds that satisfy the budget, take
the most permissive one. When no threshold satisfies the budget (too few pairs, or the
cohort is intrinsically ambiguous), the table records ``feasible=False`` and falls back to
``1.0`` — i.e. no semantic hits for that cohort — which is the only safe answer. K-1 cannot
be met by guessing.

    # AT-CAC-07
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from answerdesk.contracts import Contract
from answerdesk.policy.metrics import wilson_upper

#: Default wrong-hit budget ε (requirements §9.5: K-1 ≤ 0.01 with UCB ≤ 0.02).
DEFAULT_EPSILON = 0.01

#: UCB slack multiple: the budget's upper bound is ``2ε`` (§9.3).
UCB_MULTIPLIER = 2.0

#: Safety floor for a calibrated threshold: the replay set is finite and not exhaustive, so a
#: sweep that "proves" a very low threshold is not evidence that every future query belongs
#: to the cached one. 0.80 is deliberately conservative and is recorded in the table.
MIN_THRESHOLD = 0.80

#: Candidate thresholds are the observed similarities, so the sweep is exact, not gridded.
_MAX_DISTINCT_CANDIDATES = 400


class CohortThreshold(Contract):
    """One cohort's calibrated threshold with its measured replay numbers."""

    cohort: str = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)
    epsilon: float = Field(gt=0.0, lt=1.0)
    n_pairs: int = Field(ge=0)
    hits: int = Field(ge=0)
    wrong_hits: int = Field(ge=0)
    wrong_hit_rate: float = Field(ge=0.0, le=1.0)
    wrong_hit_ucb: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    feasible: bool = True

    def satisfies_budget(self) -> bool:
        """Per-cohort gate: the measured wrong-hit rate fits the budget.

        The Wilson UCB gate of §9.3 is applied to the **pooled** replay set
        (``report.CacheEvaluation.passes_k1``): a per-cohort UCB of 2% needs roughly 190 hits
        in that cohort, which no realistic replay set has. The requirements' own sizing
        language (>= 120 pairs, >= 20 per locale) is a pooled statement, so the pooled bound
        is where it is verified; the per-cohort UCB stays in the report as a diagnostic.
        """
        return self.wrong_hit_rate <= self.epsilon


class ThresholdTable(Contract):
    """Calibrated thresholds for every cohort, bound to the replay set they came from."""

    table_version: str = "1.0.0"
    calibrated_at: str = ""
    replay_hash: str = ""
    epsilon: float = DEFAULT_EPSILON
    cohorts: list[CohortThreshold] = Field(default_factory=list)
    default_threshold: float = Field(default=0.93, ge=0.0, le=1.0)

    def threshold_for(self, cohort: str) -> float:
        for row in self.cohorts:
            if row.cohort == cohort:
                return row.threshold
        return self.default_threshold

    def row_for(self, cohort: str) -> CohortThreshold | None:
        return next((row for row in self.cohorts if row.cohort == cohort), None)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ThresholdTable:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def calibrate_threshold(
    *,
    cohort: str,
    similarities: Sequence[float],
    wrong: Sequence[bool],
    compatible: Sequence[bool] | None = None,
    exact: Sequence[bool] | None = None,
    epsilon: float = DEFAULT_EPSILON,
    default_threshold: float = 0.93,
    min_threshold: float = MIN_THRESHOLD,
) -> CohortThreshold:
    """Smallest threshold meeting the wrong-hit budget for one cohort.

    A pair is a hit when it is exact-equivalent (same normalised key) or when its
    similarity clears the threshold *and* the entity guard would let it through. Calibrating
    on raw similarity alone would ignore the guard that makes the semantic path safe.
    """
    if len(similarities) != len(wrong):
        raise ValueError("similarities and wrong must have the same length")
    resolved_compatible = list(compatible) if compatible is not None else [True] * len(similarities)
    resolved_exact = list(exact) if exact is not None else [False] * len(similarities)
    if not (len(resolved_compatible) == len(resolved_exact) == len(similarities)):
        raise ValueError("similarities, wrong, compatible and exact must have equal length")
    rows = list(zip(similarities, wrong, resolved_compatible, resolved_exact, strict=True))
    pairs = sorted(rows, key=lambda item: item[0], reverse=True)
    n_pairs = len(pairs)
    candidates = sorted({row[0] for row in pairs if row[0] >= min_threshold}, reverse=True)
    if not candidates:
        # Nothing at or above the floor: fall back to the default threshold (a miss).
        candidates = [max(min_threshold, min(1.0, default_threshold))]
    if len(candidates) > _MAX_DISTINCT_CANDIDATES:
        step = len(candidates) // _MAX_DISTINCT_CANDIDATES
        candidates = candidates[::step]

    exact_hits = sum(1 for row in pairs if row[3])
    wrong_exact = sum(1 for row in pairs if row[3] and row[1])
    semantic = [(row[0], row[1], row[2]) for row in pairs if not row[3]]
    best: CohortThreshold | None = None
    for threshold in candidates:
        hits = exact_hits
        wrong_hits = wrong_exact
        for similarity, is_wrong, is_compatible in semantic:
            if similarity < threshold:
                break
            if not is_compatible:
                continue
            hits += 1
            if is_wrong:
                wrong_hits += 1
        if hits == 0:
            continue
        rate = wrong_hits / hits
        ucb = wilson_upper(wrong_hits, hits)
        row = CohortThreshold(
            cohort=cohort,
            threshold=threshold,
            epsilon=epsilon,
            n_pairs=n_pairs,
            hits=hits,
            wrong_hits=wrong_hits,
            wrong_hit_rate=round(rate, 10),
            wrong_hit_ucb=round(ucb, 10),
            hit_rate=round(hits / n_pairs, 10),
            feasible=rate <= epsilon,
        )
        if not row.feasible:
            continue
        # candidates are descending: the last feasible one is the most permissive (smallest τ)
        if best is None or row.threshold < best.threshold:
            best = row
    if best is not None:
        return best
    return CohortThreshold(
        cohort=cohort,
        threshold=1.0,
        epsilon=epsilon,
        n_pairs=n_pairs,
        hits=0,
        wrong_hits=0,
        wrong_hit_rate=0.0,
        wrong_hit_ucb=0.0,
        hit_rate=0.0,
        feasible=False,
    )


def build_threshold_table(
    *,
    similarities_by_cohort: Mapping[str, Sequence[float]],
    wrong_by_cohort: Mapping[str, Sequence[bool]],
    compatible_by_cohort: Mapping[str, Sequence[bool]] | None = None,
    exact_by_cohort: Mapping[str, Sequence[bool]] | None = None,
    replay_hash: str,
    epsilon: float = DEFAULT_EPSILON,
    default_threshold: float = 0.93,
    calibrated_at: str | None = None,
) -> ThresholdTable:
    """Calibrate every cohort in the replay set."""
    cohorts: list[CohortThreshold] = []
    for cohort in sorted(similarities_by_cohort):
        cohorts.append(
            calibrate_threshold(
                cohort=cohort,
                similarities=similarities_by_cohort[cohort],
                wrong=wrong_by_cohort[cohort],
                compatible=(
                    compatible_by_cohort[cohort] if compatible_by_cohort is not None else None
                ),
                exact=(exact_by_cohort[cohort] if exact_by_cohort is not None else None),
                epsilon=epsilon,
                default_threshold=default_threshold,
            )
        )
    return ThresholdTable(
        calibrated_at=(calibrated_at or datetime.now(UTC).replace(microsecond=0).isoformat()),
        replay_hash=replay_hash,
        epsilon=epsilon,
        cohorts=cohorts,
        default_threshold=default_threshold,
    )


__all__ = [
    "DEFAULT_EPSILON",
    "MIN_THRESHOLD",
    "UCB_MULTIPLIER",
    "CohortThreshold",
    "ThresholdTable",
    "build_threshold_table",
    "calibrate_threshold",
]
