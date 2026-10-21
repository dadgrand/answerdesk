"""Output budget: compact schema, hard token cap, extractive-first instruction.

Generation latency is dominated by *decoding*: a 35B model at ~40 tok/s spends most of a turn
writing JSON boilerplate and a paragraph of prose around two facts. Three levers cut that without
touching grounding (all three are applied by `generate.pipeline.AnswerPipeline`):

1. **Compact schema** — the shipped draft schema documents itself richly (`description` on every
   field, a required ``why`` string). Descriptions cost *input* tokens on every call and the
   ``why`` field costs output tokens even when the answer is fine. The compact variant keeps the
   same shape, drops the prose, and makes ``why`` optional so refusal turns are not required to
   invent an explanation. It is validated by the same strict-schema checker, and the digest is
   recorded so a trace can prove which schema produced an answer.
2. **Token cap** — a hard ``max_output_tokens`` that budgets the answer, feeding the packer's
   reserved-token arithmetic (a smaller answer allowance leaves more room for evidence).
3. **Extractive-first instruction** — one short instruction appended to the user turn: quote the
   evidence sentence that answers the question, add no commentary, at most *N* claims. It is
   appended in code rather than in the versioned prompt registry on purpose: the registry's
   digest/lock contract (FR-GEN-05) is for *prompt wording*, and a per-deployment budget knob
   must not silently change the prompt version. The suffix is recorded in the artifact instead.

The budget is data (`AnswerBudget`), so the pipeline, the eval harness and the latency report all
name the same arm, and a test can assert the cap and the claim limit without a model.

    # AT-LAT-02  # AT-GEN-05  # AT-COST-01
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from pydantic import Field

from answerdesk.contracts import Contract

#: Instruction appended to the user turn in the budget arm (kept short on purpose: it is paid
#: for on every request).
EXTRACTIVE_INSTRUCTION = (
    "Answer in at most {max_claims} claims. Copy the answering sentence from the evidence and "
    "cite it verbatim; do not add commentary, examples or restatements."
)

#: Fields the compact schema removes from the *required* list (they stay allowed).
COMPACT_OPTIONAL_FIELDS: tuple[str, ...] = ("why",)


class AnswerBudget(Contract):
    """One generation arm's output budget."""

    name: str = "baseline"
    max_output_tokens: int = Field(default=512, ge=32, le=4096)
    max_claims: int = Field(default=6, ge=1, le=12)
    compact_schema: bool = False
    extractive: bool = False
    note: str = ""

    def instruction(self) -> str:
        """The suffix appended to the user turn, or ``""`` for the baseline arm."""
        if not self.extractive:
            return ""
        return EXTRACTIVE_INSTRUCTION.format(max_claims=self.max_claims)

    @classmethod
    def baseline(cls) -> AnswerBudget:
        return cls(name="baseline", max_output_tokens=512, max_claims=6)

    @classmethod
    def budget(cls) -> AnswerBudget:
        """The v2 arm: a third of the output allowance, two claims, compact schema."""
        return cls(
            name="budget",
            max_output_tokens=160,
            max_claims=2,
            compact_schema=True,
            extractive=True,
            note="TTFT-first arm from docs/latency.md",
        )


class BudgetSchema(Contract):
    """A schema plus its digest, so the artifact can name what the model was constrained by."""

    payload: dict[str, Any]
    digest: str
    compact: bool = False
    required_dropped: list[str] = Field(default_factory=list)


def _without_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _without_descriptions(value)
            for key, value in node.items()
            if key not in {"description", "$comment", "examples"}
        }
    if isinstance(node, list):
        return [_without_descriptions(item) for item in node]
    return node


def compact_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop documentation and make ``why`` optional; keep the shape strict.

    Returns the compact schema and the list of fields removed from ``required``.
    """
    compact = _without_descriptions(copy.deepcopy(schema))
    dropped: list[str] = []
    required = compact.get("required")
    if isinstance(required, list):
        kept = [name for name in required if name not in COMPACT_OPTIONAL_FIELDS]
        dropped = [name for name in required if name in COMPACT_OPTIONAL_FIELDS]
        compact["required"] = kept
    compact["title"] = f"{compact.get('title', 'draft')} (compact)"
    return compact, dropped


def budget_schema(schema: dict[str, Any], *, budget: AnswerBudget) -> BudgetSchema:
    """The schema a budget arm sends: the shipped one, or its compact variant."""
    payload = schema
    compacted: list[str] = []
    if budget.compact_schema:
        payload, compacted = compact_schema(schema)
    material = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
    return BudgetSchema(
        payload=payload, digest=digest, compact=budget.compact_schema, required_dropped=compacted
    )


def cap_claims(payload: dict[str, Any], *, max_claims: int) -> dict[str, Any]:
    """Truncate a *parsed* draft payload to ``max_claims`` claims (used before validation)."""
    claims = payload.get("claims")
    if isinstance(claims, list) and len(claims) > max_claims:
        payload = {**payload, "claims": claims[:max_claims]}
    return payload


__all__ = [
    "COMPACT_OPTIONAL_FIELDS",
    "EXTRACTIVE_INSTRUCTION",
    "AnswerBudget",
    "BudgetSchema",
    "budget_schema",
    "cap_claims",
    "compact_schema",
]
