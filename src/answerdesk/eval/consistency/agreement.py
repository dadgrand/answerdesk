"""Consistency metrics: agreement, invariance and the confidence intervals around them."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from answerdesk.eval.stats import paired_bootstrap, wilson_interval


@dataclass(frozen=True)
class Rate:
    n: int
    successes: int
    value: float
    low: float
    high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "successes": self.successes,
            "rate": round(self.value, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
        }


def rate(successes: Iterable[bool]) -> Rate:
    flags = list(successes)
    if not flags:
        return Rate(0, 0, 0.0, 0.0, 1.0)
    interval = wilson_interval(sum(1 for flag in flags if flag), len(flags))
    return Rate(
        len(flags),
        int(interval.value * len(flags)),
        float(interval.value),
        interval.low,
        interval.high,
    )


def _iterable(values: Sequence[bool]) -> Iterable[bool]:
    return values


def paired_delta(
    baseline: Sequence[float], candidate: Sequence[float], *, samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap on per-item 0/1 indicators (candidate − baseline)."""
    if len(baseline) != len(candidate):
        raise ValueError("paired comparison needs the same items")
    if not baseline:
        return {"n": 0, "delta": None, "ci_low": None, "ci_high": None, "ci_crosses_zero": True}
    result = paired_bootstrap(
        np.asarray(candidate, dtype=np.float64),
        np.asarray(baseline, dtype=np.float64),
        samples=samples,
        seed=seed,
    )
    return {
        "n": int(result.n),
        "baseline": round(float(result.mean_b), 6),
        "candidate": round(float(result.mean_a), 6),
        "delta": round(float(result.diff), 6),
        "ci_low": round(float(result.low), 6),
        "ci_high": round(float(result.high), 6),
        "ci_crosses_zero": bool(result.low <= 0.0 <= result.high),
        "samples": samples,
        "seed": seed,
    }


def fact_agreement(rows: Sequence[dict[str, Any]], *, key: str = "same_fact") -> dict[str, Any]:
    """Per-language agreement of the answer with the fact, plus the pooled rate."""
    per_language: dict[str, dict[str, Any]] = {}
    pooled: list[bool] = []
    for locale in sorted({str(row["locale"]) for row in rows}):
        subset = [row for row in rows if row["locale"] == locale]
        flags = [bool(row.get(key)) for row in subset]
        pooled.extend(flags)
        per_language[locale] = rate(flags).to_dict()
    return {"per_language": per_language, "pooled": rate(pooled).to_dict()}


def cross_language_pairs(
    rows: Sequence[dict[str, Any]], *, key: str = "same_fact"
) -> dict[str, Any]:
    """Agreement between every pair of locales on the facts they both answer.

    For each fact, a pair is *agreeing* when both locales answered it and both answers state
    the fact. This measures whether a fact survives translation, not whether the numbers are
    identical (the corpus instantiates each locale separately).
    """
    by_fact: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_fact.setdefault(str(row["fact_id"]), {})[str(row["locale"])] = bool(row.get(key))
    locales = sorted({row["locale"] for row in rows})
    pairs: dict[str, dict[str, Any]] = {}
    all_flags: list[bool] = []
    for left_index, left in enumerate(locales):
        for right in locales[left_index + 1 :]:
            flags = [
                values[left] and values[right]
                for values in by_fact.values()
                if left in values and right in values
            ]
            all_flags.extend(flags)
            pairs[f"{left}-{right}"] = rate(flags).to_dict()
    return {
        "pairs": pairs,
        "pooled": rate(all_flags).to_dict(),
        "definition": "both locales state the fact for the same fact_id",
    }


def citation_overlap(
    *,
    cited: Sequence[str],
    gold: Sequence[str],
) -> dict[str, Any]:
    cited_set = {value for value in cited if value}
    gold_set = {value for value in gold if value}
    if not cited_set:
        return {"hit": False, "precision": 0.0, "recall": 0.0, "cited": 0, "gold": len(gold_set)}
    overlap = cited_set & gold_set
    return {
        "hit": bool(overlap),
        "precision": len(overlap) / len(cited_set),
        "recall": (len(overlap) / len(gold_set)) if gold_set else 0.0,
        "cited": len(cited_set),
        "gold": len(gold_set),
    }


def invariance(rows: Sequence[dict[str, Any]], *, clean_kind: str = "clean") -> dict[str, Any]:
    """Surface-form invariance: same evidence and same decision as the clean form."""
    by_fact: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_fact.setdefault(str(row["fact_id"]), {})[str(row["variant"])] = row
    evidence_flags: list[bool] = []
    decision_flags: list[bool] = []
    per_kind: dict[str, dict[str, Any]] = {}
    for _fact_id, variants in by_fact.items():
        clean = variants.get(clean_kind)
        if clean is None or clean.get("decision") == "provider_error":
            continue
        for kind, row in variants.items():
            if kind == clean_kind or row.get("decision") == "provider_error":
                continue
            same_evidence = set(row.get("evidence_doc_ids") or []) == set(
                clean.get("evidence_doc_ids") or []
            )
            same_decision = row.get("decision") == clean.get("decision")
            evidence_flags.append(same_evidence)
            decision_flags.append(same_decision)
            block = per_kind.setdefault(kind, {"evidence": [], "decision": [], "facts": 0})
            block["evidence"].append(same_evidence)
            block["decision"].append(same_decision)
            block["facts"] += 1
    return {
        "pairs": len(evidence_flags),
        "evidence_invariance": rate(evidence_flags).to_dict(),
        "decision_invariance": rate(decision_flags).to_dict(),
        "per_variant": {
            kind: {
                "facts": block["facts"],
                "evidence": rate(block["evidence"]).to_dict(),
                "decision": rate(block["decision"]).to_dict(),
            }
            for kind, block in sorted(per_kind.items())
        },
    }


def language_discipline(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_language: dict[str, dict[str, Any]] = {}
    for locale in sorted({str(row["locale"]) for row in rows}):
        subset = [row for row in rows if row["locale"] == locale]
        per_language[locale] = {
            "answers": len(subset),
            "discipline": rate([bool(row.get("language_ok")) for row in subset]).to_dict(),
            "leakage": rate([bool(row.get("evidence_locale_ok")) for row in subset]).to_dict(),
            "detected": rate([bool(row.get("language_detected_ok")) for row in subset]).to_dict(),
        }
    return per_language


__all__ = [
    "Rate",
    "citation_overlap",
    "cross_language_pairs",
    "fact_agreement",
    "invariance",
    "language_discipline",
    "paired_delta",
    "rate",
]
