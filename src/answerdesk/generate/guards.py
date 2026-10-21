"""Prompt-injection defence: spot the instruction, refuse to auto-answer.

INV-01 is architectural: retrieved text is data, never control flow. Two mechanisms
enforce it and they do not depend on the model behaving:

* **Spotlighting** — evidence is wrapped in ``<passage id="pN">`` blocks and the system
  prompt states that nothing inside them can change the task, the format, the language or
  the routing (FR-GEN-11). Passage-delimiter lookalikes in document text are neutralised
  by the packer so a document cannot close its own block.
* **Detection** — :func:`detect_injection` matches directive-shaped text (instruction
  overrides, role changes, exfiltration, delimiter escapes). A hit means the turn is
  escalated with ``INJECTION_SUSPECTED`` and **no model call is made**, so there is no
  prose to be talked into following the payload.

The heuristics are intentionally directive-only: ordinary support prose ("we can approve
refunds within 14 days") must never trip them, because a guard that fires on real content
is a guard that gets switched off. Precision/recall on the fixture set is measured by
NFR-SEC-01; the user-text half of the same problem (PII, prompt heuristics in the query)
is t8's ``policy`` module, which may reuse these rules.

    # AT-SEC-01  # AT-GEN-11
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from answerdesk.contracts import Chunk, Contract, RetrievedPassage


class InjectionRule(Contract):
    """One named directive pattern, so a finding is explainable in the trace."""

    name: str
    pattern: re.Pattern[str] = Field(exclude=True)
    description: str = ""


#: Directive-shaped patterns. Each is narrow on purpose: no bare "refund"/"approve".
INJECTION_RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        name="instruction-override",
        pattern=re.compile(
            r"\b(ignore|disregard|forget)\b[\s\S]{0,40}?\b(instruction|prompt|rule|directive)s?\b",
            re.IGNORECASE,
        ),
        description="tries to cancel the system prompt",
    ),
    InjectionRule(
        name="role-change",
        pattern=re.compile(
            r"\b(you are now|from now on you|act as (a|an|the)|new instructions?:)\b",
            re.IGNORECASE,
        ),
        description="tries to replace the assistant's role",
    ),
    InjectionRule(
        name="prompt-exfiltration",
        pattern=re.compile(
            r"\b(reveal|print|show|repeat|output)\b[\s\S]{0,30}?"
            r"\b(system prompt|instructions|your rules|prompt text)\b",
            re.IGNORECASE,
        ),
        description="tries to leak the system prompt",
    ),
    InjectionRule(
        name="format-override",
        pattern=re.compile(
            r"\b(respond|reply|answer|output)\b[\s\S]{0,25}?\b(only in|in plain|as xml|as html|"
            r"without json|in english only)\b",
            re.IGNORECASE,
        ),
        description="tries to change the output contract",
    ),
    InjectionRule(
        name="tool-invocation",
        pattern=re.compile(
            r"\b(call|invoke|execute|trigger)\b[\s\S]{0,25}?\b(tool|function|api|webhook|refund)\b",
            re.IGNORECASE,
        ),
        description="tries to make the assistant act",
    ),
    InjectionRule(
        name="delimiter-escape",
        pattern=re.compile(r"</?passage\b|</?evidence\b|<\|im_(start|end)\|>", re.IGNORECASE),
        description="tries to break the evidence delimiter",
    ),
    InjectionRule(
        name="exfiltration",
        pattern=re.compile(
            r"\b(send|post|upload|forward)\b[\s\S]{0,40}?\bhttps?://", re.IGNORECASE
        ),
        description="tries to exfiltrate data",
    ),
    InjectionRule(
        name="canary",
        pattern=re.compile(r"injection[-_ ]?canary", re.IGNORECASE),
        description="test canary: proves a payload reached the model or the answer",
    ),
)


class InjectionFinding(Contract):
    """One place where directive text was found, and which rules it matched."""

    source: Literal["query", "passage"]
    ref: str = ""
    rules: list[str] = Field(default_factory=list)
    excerpt: str = ""


def detect_injection(text: str) -> list[str]:
    """Names of the rules a piece of text matches (empty when clean)."""
    if not text:
        return []
    return [rule.name for rule in INJECTION_RULES if rule.pattern.search(text)]


def _excerpt(text: str, *, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def scan_query(query: str) -> list[InjectionFinding]:
    """Injection findings in the customer turn itself."""
    rules = detect_injection(query)
    if not rules:
        return []
    return [InjectionFinding(source="query", ref="query", rules=rules, excerpt=_excerpt(query))]


def scan_passages(passages: Sequence[RetrievedPassage]) -> list[InjectionFinding]:
    """Injection findings in retrieved evidence."""
    findings: list[InjectionFinding] = []
    for passage in passages:
        rules = detect_injection(passage.chunk.text)
        if rules:
            findings.append(
                InjectionFinding(
                    source="passage",
                    ref=passage.chunk.chunk_id,
                    rules=rules,
                    excerpt=_excerpt(passage.chunk.text),
                )
            )
    return findings


def scan_chunks(chunks: Sequence[Chunk]) -> list[InjectionFinding]:
    """Same scan for bare chunks (cache re-validation, t8)."""
    findings: list[InjectionFinding] = []
    for chunk in chunks:
        rules = detect_injection(chunk.text)
        if rules:
            findings.append(
                InjectionFinding(
                    source="passage", ref=chunk.chunk_id, rules=rules, excerpt=_excerpt(chunk.text)
                )
            )
    return findings


def scan(query: str, passages: Sequence[RetrievedPassage]) -> list[InjectionFinding]:
    """Full scan of one turn: the customer text plus every packed passage."""
    return [*scan_query(query), *scan_passages(passages)]


def contains_canary(text: str) -> bool:
    """Whether a test canary leaked into an answer (used by the injection fixture suite)."""
    return bool(re.search(r"injection[-_ ]?canary", text, re.IGNORECASE))


__all__ = [
    "INJECTION_RULES",
    "InjectionFinding",
    "InjectionRule",
    "contains_canary",
    "detect_injection",
    "scan",
    "scan_chunks",
    "scan_passages",
    "scan_query",
]
