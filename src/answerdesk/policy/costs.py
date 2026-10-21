"""The cost model: three actions, two ratios, no hard-coded thresholds.

Derivation (docs/escalation.md §2). With ``p̂ = P(answer correct | x, E)`` and three
actions — answer, clarify, escalate — the expected cost of each action is

```
E[cost | answer]   = (1 − p̂)·C_wrong
E[cost | clarify]  = C_delay
E[cost | escalate] = C_abstain
```

Minimising over actions gives the Bayes rule of requirements §8.3:

```
answer    iff  (1 − p̂)·C_wrong ≤ min(C_delay, C_abstain)
               ⇔  p̂ ≥ 1 − C_delay / C_wrong
clarify   iff  C_delay ≤ (1 − p̂)·C_wrong < C_abstain
               ⇔  1 − C_abstain/C_wrong ≤ p̂ < 1 − C_delay/C_wrong
escalate  otherwise  ⇔  p̂ < 1 − C_abstain/C_wrong
```

Costs are configuration, never code constants (§8.3): they live in
``configs/policy/costs.yaml`` **with provenance**, and a row without provenance fails
validation. The derived numbers are exposed as properties so no module ever compares
``p̂`` to a literal (CL-POL-01).

    # AT-ESC-03  # AT-ESC-06
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from answerdesk.config import Settings
from answerdesk.contracts import Contract
from answerdesk.policy.errors import CostModelError

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default cost file (a code constant, not a gated config key).
DEFAULT_COST_PATH = Path("configs") / "policy" / "costs.yaml"


class CostProvenance(Contract):
    """Where a number came from. A cost without this is invalid configuration (§8.3)."""

    source: str = Field(min_length=1, description="pilot measurement | assumption | contract")
    owner: str = Field(min_length=1)
    date: str = Field(min_length=10, description="ISO date the number was agreed")
    note: str = ""


class CostSet(Contract):
    """The three (plus one reserved) costs and the thresholds they imply."""

    c_wrong: float = Field(gt=0.0, description="cost of a wrong auto-answer")
    c_abstain: float = Field(ge=0.0, description="cost of escalating to a human")
    c_delay: float = Field(ge=0.0, description="cost of asking a clarifying question")
    c_review: float = Field(default=0.0, ge=0.0, description="reserved for a v1.1 review queue")
    provenance: CostProvenance

    @model_validator(mode="after")
    def _ordering_is_sane(self) -> CostSet:
        if self.c_abstain > self.c_wrong:
            raise ValueError(
                "c_abstain must be <= c_wrong: escalating could never be cheaper than a "
                "wrong answer, so the rule would answer everything"
            )
        if self.c_delay > self.c_abstain:
            raise ValueError(
                "c_delay must be <= c_abstain: the clarify band would be empty or inverted"
            )
        return self

    # -- derived thresholds (the only place these comparisons exist) --------------
    @property
    def answer_min_probability(self) -> float:
        """``1 − C_delay / C_wrong`` — the lower edge of the answer band."""
        return 1.0 - self.c_delay / self.c_wrong

    @property
    def clarify_min_probability(self) -> float:
        """``1 − C_abstain / C_wrong`` — below this, escalate instead of clarifying."""
        return 1.0 - self.c_abstain / self.c_wrong

    @property
    def alpha_from_costs(self) -> float:
        """``C_abstain / C_wrong`` — the risk the cost model is willing to carry."""
        return self.c_abstain / self.c_wrong

    @property
    def cost_ratio(self) -> float:
        """``C_wrong / C_abstain`` — the "how many escalations per wrong answer" ratio."""
        if self.c_abstain == 0.0:
            return float("inf")
        return self.c_wrong / self.c_abstain

    # -- provenance ---------------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        """Deterministic body used for the cost-set hash recorded in every decision."""
        return {
            "c_wrong": round(self.c_wrong, 10),
            "c_abstain": round(self.c_abstain, 10),
            "c_delay": round(self.c_delay, 10),
            "c_review": round(self.c_review, 10),
            "provenance_source": self.provenance.source,
            "provenance_date": self.provenance.date,
        }

    def hash(self) -> str:
        material = json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def threshold_summary(self) -> dict[str, float]:
        return {
            "answer_min_probability": round(self.answer_min_probability, 10),
            "clarify_min_probability": round(self.clarify_min_probability, 10),
            "alpha_from_costs": round(self.alpha_from_costs, 10),
        }


def default_cost_path() -> Path:
    return _REPO_ROOT / DEFAULT_COST_PATH


def load_cost_set(path: str | Path | None = None) -> CostSet:
    """Load ``configs/policy/costs.yaml``; provenance is mandatory (§8.3)."""
    file_path = Path(path) if path is not None else default_cost_path()
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CostModelError(f"cannot read cost file {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CostModelError(f"cost file {file_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise CostModelError(f"cost file {file_path} must be a mapping")
    costs = raw.get("costs")
    provenance = raw.get("provenance")
    if not isinstance(costs, dict):
        raise CostModelError(f"cost file {file_path} needs a 'costs' mapping")
    if not isinstance(provenance, dict) or not provenance:
        raise CostModelError(
            f"cost file {file_path} has no provenance. Requirements §8.3: a cost with no "
            "provenance is invalid config (state source, owner and date)."
        )
    try:
        return CostSet(**costs, provenance=CostProvenance(**provenance))
    except Exception as exc:
        raise CostModelError(f"cost file {file_path} is invalid: {exc}") from exc


def cost_set_from_settings(settings: Settings) -> CostSet:
    """Build a :class:`CostSet` from the frozen ``policy.costs`` configuration.

    Used when a caller has settings but no cost file at hand (unit tests, the console's
    read-only views). The provenance names the configuration source, so a decision made
    this way still records where the numbers came from.
    """
    costs = settings.policy.costs
    return CostSet(
        c_wrong=costs.c_wrong,
        c_abstain=costs.c_abstain,
        c_delay=costs.c_delay,
        provenance=CostProvenance(
            source="configs/default.yaml -> policy.costs (requirements Appendix C)",
            owner="t8",
            date="2026-10-20",
            note="reference defaults from requirements §8.3",
        ),
    )


def check_cost_drift(settings: Settings, cost_set: CostSet) -> list[str]:
    """Compare the YAML cost set with ``policy.costs``; returns human-readable drifts."""
    costs = settings.policy.costs
    problems: list[str] = []
    for name, configured, documented in (
        ("c_wrong", costs.c_wrong, cost_set.c_wrong),
        ("c_abstain", costs.c_abstain, cost_set.c_abstain),
        ("c_delay", costs.c_delay, cost_set.c_delay),
    ):
        if abs(configured - documented) > 1e-12:
            problems.append(f"{name}: settings={configured} but costs.yaml={documented}")
    return problems


__all__ = [
    "DEFAULT_COST_PATH",
    "CostProvenance",
    "CostSet",
    "check_cost_drift",
    "cost_set_from_settings",
    "default_cost_path",
    "load_cost_set",
]
