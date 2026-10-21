"""The calibration artifact: the only object allowed to say "we are calibrated".

Requirements §8.3, §8.6 and FR-ESC-06: every decision references a threshold that resolves
to a persisted artifact carrying ``split hash``, ``n``, ``α``, ``q̂``, the cost set, the
scorer identity, the model and the date. The artifact is content-addressed — its
:meth:`CalibrationArtifact.compute_hash` covers everything except the hash field itself — so
a decision citing ``cal-3f9a…`` can be reproduced from the file.

Two operational rules live here as code:

* **staleness** (§8.6): an artifact older than the configured TTL, or fit on a different
  corpus version, is *not* silently trusted; the decision engine escalates with
  ``STALE_CALIBRATION``.
* **honesty**: the artifact stores the measured metrics (risk with its Wilson interval,
  coverage, ECE both ways, Brier, AURC, E-2/E-3) next to the thresholds, so a release note
  cannot quote the threshold without the evidence.

    # AT-ESC-06  # AT-ESC-07
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from answerdesk.contracts import Contract, IntentBucket
from answerdesk.policy.conformal import ConformalSet
from answerdesk.policy.costs import CostSet
from answerdesk.policy.errors import ArtifactError
from answerdesk.policy.thresholds import ThresholdTable

ARTIFACT_VERSION = "1.0.0"


def canonical_hash(payload: Any) -> str:
    """sha256 over canonical JSON (sorted keys, no whitespace drift)."""
    material = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class BucketMetrics(Contract):
    """Measured behaviour of one bucket on the held-out test split."""

    bucket: IntentBucket
    n_calibration: int = Field(ge=0)
    n_test: int = Field(ge=0)
    alpha_bucket: float = Field(gt=0.0, lt=1.0)
    alpha_final: float = Field(gt=0.0, lt=1.0)
    q_hat: float
    answer_min_probability: float = Field(gt=0.0, lt=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    n_answered: int = Field(ge=0)
    n_answerable: int = Field(ge=0)
    n_answered_answerable: int = Field(ge=0)
    coverage_answerable: float = Field(ge=0.0, le=1.0)
    n_errors: int = Field(ge=0)
    realised_risk: float = Field(ge=0.0, le=1.0)
    risk_wilson_low: float = Field(ge=0.0, le=1.0)
    risk_wilson_high: float = Field(ge=0.0, le=1.0)
    ece_width: float = Field(ge=0.0, le=1.0)
    ece_mass: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)


class PooledMetrics(Contract):
    """The same numbers pooled across buckets (E-1 requires both views)."""

    n_calibration: int = Field(ge=0)
    n_test: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    n_answered: int = Field(ge=0)
    n_answerable: int = Field(ge=0)
    n_answered_answerable: int = Field(ge=0)
    coverage_answerable: float = Field(ge=0.0, le=1.0)
    n_errors: int = Field(ge=0)
    realised_risk: float = Field(ge=0.0, le=1.0)
    risk_wilson_low: float = Field(ge=0.0, le=1.0)
    risk_wilson_high: float = Field(ge=0.0, le=1.0)
    ece_width: float = Field(ge=0.0, le=1.0)
    ece_mass: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)
    aurc: float = Field(ge=0.0, le=1.0)
    must_escalate_recall: float = Field(ge=0.0, le=1.0)
    escalation_precision: float = Field(
        ge=0.0, le=1.0, description="escalations that were unsafe or would have been wrong"
    )
    escalation_precision_safety: float = Field(
        default=0.0, ge=0.0, le=1.0, description="escalations that were unsafe (literal E-3)"
    )
    realised_risk_target: float = Field(gt=0.0, lt=1.0)


class CalibrationArtifact(Contract):
    """Persisted calibration state; every decision's ``threshold_ref`` points here."""

    artifact_version: str = ARTIFACT_VERSION
    created_at: str = Field(min_length=10)
    corpus_version: str = ""
    calibration_hash: str = Field(min_length=8)
    test_hash: str = Field(min_length=8)
    split: str = ""
    scorer_name: str = ""
    scorer_version: str = ""
    model_id: str = ""
    decision_rule: str = "conformal+cost-bands (docs/escalation.md §3)"
    costs: CostSet
    cost_hash: str = Field(min_length=8)
    conformal: ConformalSet
    thresholds: ThresholdTable
    pooled: PooledMetrics
    buckets: list[BucketMetrics] = Field(default_factory=list)
    notes: str = ""
    artifact_hash: str = ""

    # -- content addressing -------------------------------------------------------
    def _body(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude={"artifact_hash"})
        return payload

    def compute_hash(self) -> str:
        return canonical_hash(self._body())

    def sealed(self) -> CalibrationArtifact:
        """Return a copy with ``artifact_hash`` set from the body."""
        return self.model_copy(update={"artifact_hash": self.compute_hash()})

    def verify(self) -> None:
        """Raise if the stored hash does not match the body (tamper/corruption check)."""
        expected = self.compute_hash()
        if not self.artifact_hash:
            raise ArtifactError("calibration artifact is unsealed (no artifact_hash)")
        if self.artifact_hash != expected:
            raise ArtifactError(
                f"calibration artifact hash mismatch: file says {self.artifact_hash[:12]}…, "
                f"body hashes to {expected[:12]}… — the artifact was edited after sealing"
            )

    def threshold_ref(self) -> str:
        return f"cal-{self.artifact_hash[:12]}"

    # -- persistence --------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        file_path = Path(path)
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ArtifactError(f"cannot read calibration artifact {file_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ArtifactError(
                f"calibration artifact {file_path} is not valid JSON: {exc}"
            ) from exc
        artifact = cls.model_validate(raw)
        artifact.verify()
        return artifact

    # -- staleness ----------------------------------------------------------------
    def age_days(self, now: datetime | None = None) -> float:
        """Age in days at ``now`` (a malformed date is an error, not a silent zero)."""
        try:
            created = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ArtifactError(
                f"artifact created_at is not ISO-8601: {self.created_at!r}"
            ) from exc
        reference = now or datetime.now(UTC)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return max(0.0, (reference - created).total_seconds() / 86400.0)

    def is_stale(
        self,
        *,
        now: datetime | None = None,
        max_age_days: int = 7,
        corpus_version: str | None = None,
    ) -> bool:
        """§8.6: too old, or fit on a different corpus version, is stale."""
        if self.age_days(now) > max_age_days:
            return True
        return corpus_version is not None and self.corpus_version != corpus_version

    def summary_row(self) -> dict[str, Any]:
        """Flat row for a report table (no nested contracts)."""
        return {
            "threshold_ref": self.threshold_ref(),
            "created_at": self.created_at,
            "scorer": f"{self.scorer_name}-{self.scorer_version}",
            "calibration_hash": self.calibration_hash[:12],
            "cost_hash": self.cost_hash[:12],
            "pooled_risk": round(self.pooled.realised_risk, 4),
            "pooled_risk_ucb": round(self.pooled.risk_wilson_high, 4),
            "coverage": round(self.pooled.coverage, 4),
            "ece": round(self.pooled.ece_width, 4),
            "brier": round(self.pooled.brier, 4),
            "aurc": round(self.pooled.aurc, 4),
        }


__all__ = [
    "ARTIFACT_VERSION",
    "BucketMetrics",
    "CalibrationArtifact",
    "PooledMetrics",
    "canonical_hash",
]
