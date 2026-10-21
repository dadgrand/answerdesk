"""The model's answer draft: parse it, validate it, never trust it.

The model is asked for JSON (schema in :mod:`answerdesk.generate.schema`) and this
module turns that text into typed drafts. Two decisions matter:

* **Tolerant parsing, strict validation.** Models wrap JSON in markdown fences, prepend
  "Here is the JSON:" or trail a period. We locate the outermost object and hand it to
  pydantic with ``extra="forbid"``; a shape we do not recognise is a
  :class:`MalformedModelOutput`, which the pipeline handles explicitly (one strict
  re-ask, then escalation) instead of crashing or hallucinating a fallback.
* **The refusal channel is part of the schema.** ``insufficient_evidence`` exists so the
  model never has to "try its best" (requirements §7.4, FR-GEN-10, CR-8).

    # AT-GEN-01  # AT-GEN-10
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field, model_validator

from answerdesk.contracts import Contract
from answerdesk.generate.errors import MalformedModelOutput

_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


class CitationDraft(Contract):
    """One claimed support link: a passage label plus the quote the model copied."""

    passage: str = Field(pattern=r"^p\d+$")
    quote: str = Field(min_length=1)


class ClaimDraft(Contract):
    """One atomic claim with at least one citation (CR-1 enforced at parse time)."""

    text: str = Field(min_length=1)
    citations: list[CitationDraft] = Field(min_length=1)


class AnswerDraft(Contract):
    """The full structured output of one generation (or repair) call."""

    insufficient_evidence: bool = False
    why: str = ""
    claims: list[ClaimDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusal_or_claims(self) -> AnswerDraft:
        if not self.insufficient_evidence and not self.claims:
            raise ValueError(
                "a draft that does not declare insufficient_evidence must carry at least "
                "one claim; the schema has no 'best guess' channel (CR-8)"
            )
        return self

    @property
    def is_refusal(self) -> bool:
        """A refusal is authoritative even when the model also emitted claims (INV-06)."""
        return self.insufficient_evidence

    def passage_labels(self) -> list[str]:
        seen: list[str] = []
        for claim in self.claims:
            for citation in claim.citations:
                if citation.passage not in seen:
                    seen.append(citation.passage)
        return seen


def _strip_fence(raw: str) -> str:
    match = _FENCE.match(raw)
    return match.group("body") if match is not None else raw


def _outermost_object(raw: str) -> str:
    """Locate the first ``{`` and its matching ``}`` (string-literal aware)."""
    text = _strip_fence(raw).strip()
    start = text.find("{")
    if start < 0:
        raise MalformedModelOutput(
            "model output contains no JSON object",
            detail=f"first 200 characters: {raw[:200]!r}",
        )
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise MalformedModelOutput(
        "model output has an unterminated JSON object",
        detail=f"first 200 characters: {raw[:200]!r}",
    )


def parse_answer_draft(raw: str) -> AnswerDraft:
    """Parse a model reply into an :class:`AnswerDraft` or raise ``MalformedModelOutput``."""
    payload_text = _outermost_object(raw)
    try:
        payload: Any = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise MalformedModelOutput(
            "model output is not valid JSON", detail=f"{exc}: {payload_text[:200]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise MalformedModelOutput("model output JSON is not an object")
    try:
        return AnswerDraft.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError, re-typed for the pipeline
        raise MalformedModelOutput(
            "model output does not match the answer-draft schema", detail=str(exc)
        ) from exc


def draft_to_json(draft: AnswerDraft) -> str:
    """Canonical JSON of a draft (used to build repair prompts deterministically)."""
    return json.dumps(draft.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


__all__ = [
    "AnswerDraft",
    "CitationDraft",
    "ClaimDraft",
    "draft_to_json",
    "parse_answer_draft",
]
