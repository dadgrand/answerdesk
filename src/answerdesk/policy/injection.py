"""Prompt-injection heuristics for the policy layer (FR-ESC-09, NFR-SEC-01).

The generation layer (t6) already refuses to answer when evidence carries instruction-shaped
text; this module is the *policy* half of the same defence and is used by the decision
engine whether or not a generation call happened:

* it reuses t6's directive rules (a public seam, so there is exactly one rule set to audit),
  plus two policy-level additions — answer coercion ("reply regardless of the evidence") and
  bidi/hidden-character obfuscation, which is how a payload hides from a text filter;
* it scans the customer turn **and** the retrieved passages, tagging the source so the trace
  can say where the payload came from;
* a hit is a hard guard: it forces escalate/refuse before any model call (INV-01, §8.4).

    # AT-ESC-09  # AT-SEC-01
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from answerdesk.contracts import RetrievedPassage
from answerdesk.generate.guards import INJECTION_RULES as GENERATION_RULES
from answerdesk.generate.guards import InjectionFinding, scan_passages, scan_query
from answerdesk.generate.guards import detect_injection as detect_generation_injection

#: Policy-level additions. Narrow on purpose: a guard that fires on ordinary support prose
#: is a guard that gets switched off (E-3 precision ≥ 0.55 depends on this).
POLICY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "answer-coercion",
        re.compile(
            r"\b(answer|reply|respond|confirm)\b[\s\S]{0,40}?"
            r"\b(regardless of (the )?(evidence|rules|policy)|no matter what|"
            r"even if (it is|it's) wrong|do not check)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden-unicode",
        re.compile(r"[\u202a-\u202e\u2066-\u2069]|\u200b\s*\u200b"),
    ),
    (
        "credential-request",
        re.compile(
            r"\b(send|share|provide|type|enter)\b[\s\S]{0,30}?"
            r"\b(password|pin|cvv|cvc|sms code|one-time code)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_policy_injection(text: str) -> list[str]:
    """Names of every rule (generation + policy) that ``text`` matches."""
    if not text:
        return []
    rules = list(detect_generation_injection(text))
    for name, pattern in POLICY_RULES:
        if pattern.search(text) and name not in rules:
            rules.append(name)
    return rules


def _excerpt(text: str, *, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def scan_query_policy(query: str) -> list[InjectionFinding]:
    """Policy-level scan of the customer turn (adds the policy rules to t6's findings)."""
    findings = scan_query(query)
    extra = [name for name in detect_policy_injection(query) if name not in _rules_in(findings)]
    if extra:
        findings.append(
            InjectionFinding(source="query", ref="query", rules=extra, excerpt=_excerpt(query))
        )
    return findings


def scan_passages_policy(passages: Sequence[RetrievedPassage]) -> list[InjectionFinding]:
    """Policy-level scan of the retrieved evidence (t6 findings merged with policy rules)."""
    merged: dict[str, InjectionFinding] = {
        finding.ref: finding for finding in scan_passages(passages)
    }
    for passage in passages:
        extra = [
            name
            for name in detect_policy_injection(passage.chunk.text)
            if name
            not in (
                merged[passage.chunk.chunk_id].rules if passage.chunk.chunk_id in merged else []
            )
        ]
        if not extra:
            continue
        existing = merged.get(passage.chunk.chunk_id)
        merged[passage.chunk.chunk_id] = InjectionFinding(
            source="passage",
            ref=passage.chunk.chunk_id,
            rules=list(dict.fromkeys([*(existing.rules if existing else []), *extra])),
            excerpt=existing.excerpt if existing else _excerpt(passage.chunk.text),
        )
    return [merged[key] for key in sorted(merged)]


def _rules_in(findings: Sequence[InjectionFinding]) -> set[str]:
    rules: set[str] = set()
    for finding in findings:
        rules.update(finding.rules)
    return rules


def scan_turn(query: str, passages: Sequence[RetrievedPassage]) -> list[InjectionFinding]:
    """Everything suspicious in one turn, query first then evidence."""
    return [*scan_query_policy(query), *scan_passages_policy(passages)]


def rule_names() -> list[str]:
    """Every rule name the policy scan can emit (documentation + tests)."""
    return [name for name, _ in POLICY_RULES] + [rule.name for rule in GENERATION_RULES]


__all__ = [
    "POLICY_RULES",
    "detect_policy_injection",
    "rule_names",
    "scan_passages_policy",
    "scan_query_policy",
    "scan_turn",
]
