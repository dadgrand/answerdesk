"""The grounded-answer pipeline: pack → prompt → generate → cite → verify → repair.

State machine (requirements §7.3, FR-GEN-07/10/11/12):

```
question + passages
   ├─ injection guard hit ────────────────────────────► escalate(INJECTION_SUSPECTED)
   ├─ empty pack ─────────────────────────────────────► escalate(NO_EVIDENCE)
   ├─ generator/repair model call (one call per pass)
   │     ├─ malformed JSON ──► one strict re-ask ──────► escalate(PROVIDER_ERROR)
   │     ├─ insufficient_evidence ────────────────────► escalate(NO_EVIDENCE)
   │     └─ claims ──► citations ──► verifier
   │                        ├─ all supported ─────────► answer(ANSWERED)
   │                        └─ any unsupported ──► ONE repair pass
   │                                              ├─ now supported ► answer
   │                                              └─ still bad: drop claims
   │                                                    ├─ some survive ► answer
   │                                                    └─ none survive ► escalate
   └─ done
```

Four properties are load-bearing and tested:

* **No uncited claim is displayed** (INV-03/CR-1): a claim without a valid citation never
  reaches the result.
* **The numeric guard is not a model** (INV-04/CR-9): a claim quoting a number that is not
  in its cited chunk is dropped, and the pre-repair violation count is reported as a
  diagnostic (FR-GEN-12).
* **Repair happens at most once** (FR-GEN-07): unbounded repair loops are a latency and
  cost hazard.
* **A refusal carries no prose** (CR-8): an escalated result has empty text and no claims.

The released quality numbers (F-1 groundedness, F-2/F-3 citation precision/recall,
F-9 verifier quality) are measured on the frozen test split with a calibrated judge by
t11 — ``# AT-GEN-08`` marks the hook, not a claim that CI proves it.

    # AT-GEN-01  # AT-GEN-07  # AT-GEN-08  # AT-GEN-10  # AT-GEN-12
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import Field, model_validator

from answerdesk.config import Settings
from answerdesk.contracts import (
    Action,
    ChatProvider,
    Citation,
    Claim,
    ClaimVerifier,
    Completion,
    Contract,
    Message,
    ReasonCode,
    RetrievedPassage,
    Role,
    Support,
    Usage,
)
from answerdesk.generate.citations import (
    CitationBuild,
    CitationBuilder,
    cited_chunks,
    render_answer_text,
    validate_render,
)
from answerdesk.generate.draft import AnswerDraft, draft_to_json, parse_answer_draft
from answerdesk.generate.errors import MalformedModelOutput
from answerdesk.generate.guards import InjectionFinding, scan
from answerdesk.generate.language import (
    detect_language,
    language_name,
    language_notice,
    languages_agree,
)
from answerdesk.generate.packer import ContextPacker, PackedContext
from answerdesk.generate.prompts import Prompt, PromptRegistry
from answerdesk.generate.schema import load_answer_schema
from answerdesk.generate.verifier import ClaimCheck, LexicalVerifier
from answerdesk.llm.cost import PriceQuote, PricingTable, estimate_tokens

#: Prompt ids owned by this module (the registry is versioned; ids are stable).
ANSWER_PROMPT_ID = "grounded-answer"
REPAIR_PROMPT_ID = "repair-answer"


class GenerationSignal(Contract):
    """The generator's verdict for one turn, before the policy layer refines it.

    ``generate`` deliberately does not build a :class:`answerdesk.contracts.Decision`:
    the calibrated score and ``alpha`` belong to ``policy`` (t8), which composes the final
    decision from this signal plus its own conformal threshold.
    """

    action: Action
    reason_code: ReasonCode
    detail: str = ""
    evidence_score: float = Field(default=0.0, ge=0.0)

    @property
    def is_answer(self) -> bool:
        return self.action is Action.ANSWER


class GenerationResult(Contract):
    """Everything the answer path produced, with the provenance an audit needs."""

    question: str
    language: str
    text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    signal: GenerationSignal
    usage: Usage = Field(default_factory=Usage)
    prompt_version: str = ""
    verifier_version: str = ""
    repaired: bool = False
    repair_attempts: int = 0
    dropped_claims: int = Field(default=0, ge=0)
    numeric_violations_pre_repair: int = Field(default=0, ge=0)
    language_notice: bool = False
    injection: list[InjectionFinding] = Field(default_factory=list)
    packed: PackedContext = Field(default_factory=PackedContext)
    model: str = ""

    @model_validator(mode="after")
    def _answer_shape(self) -> GenerationResult:
        if self.signal.action is Action.ANSWER:
            if not self.claims:
                raise ValueError("an 'answer' result must carry at least one claim")
            for claim in self.claims:
                if not claim.citation_ids:
                    raise ValueError(
                        f"claim {claim.claim_id!r} is displayed without a citation (INV-03)"
                    )
                if claim.support is Support.UNSUPPORTED:
                    raise ValueError(
                        f"claim {claim.claim_id!r} is unsupported; it may not be displayed (CR-7)"
                    )
        elif self.claims:
            raise ValueError("a refusal/escalation carries no prose claims (CR-8)")
        return self

    def to_json(self) -> str:
        """Canonical JSON — the object the determinism test hashes (FR-GEN-09)."""
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
        )

    def to_answer_parts(self) -> tuple[str, list[Claim], list[Citation]]:
        """``(text, claims, citations)`` for the service layer to embed in an ``Answer``."""
        return self.text, list(self.claims), list(self.citations)


@dataclass
class _UsageAccumulator:
    """Sums token usage across the generation and repair calls of one answer."""

    pricing: PricingTable
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    model: str = ""
    quotes: list[PriceQuote] = field(default_factory=list)

    def add(self, completion: Completion) -> None:
        quote = self.pricing.price(
            completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.usd += quote.usd
        self.quotes.append(quote)
        self.model = completion.model or self.model

    def usage(self, *, prompt_version: str) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            usd=round(self.usd, 10),
            model=self.model,
            prompt_version=prompt_version,
        )


class AnswerPipeline:
    """Grounded generation with enforced citations, one repair pass and explicit refusals."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        registry: PromptRegistry | None = None,
        packer: ContextPacker | None = None,
        verifier: ClaimVerifier | None = None,
        pricing: PricingTable | None = None,
        schema: dict[str, object] | None = None,
        repair_passes: int = 1,
        max_answer_tokens: int = 512,
        language_notice_enabled: bool = True,
    ) -> None:
        if repair_passes < 0:
            raise ValueError("repair_passes must be >= 0")
        self.provider = provider
        self.registry = registry if registry is not None else PromptRegistry.load()
        self.packer = packer if packer is not None else ContextPacker()
        self.verifier = verifier if verifier is not None else LexicalVerifier()
        self.pricing = pricing if pricing is not None else PricingTable.default()
        self.schema = schema if schema is not None else load_answer_schema()
        self.repair_passes = repair_passes
        self.max_answer_tokens = max(1, max_answer_tokens)
        self.language_notice_enabled = language_notice_enabled
        self.builder = CitationBuilder()
        self.verifier_version = getattr(self.verifier, "version", "")

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_settings(cls, provider: ChatProvider, settings: Settings) -> AnswerPipeline:
        """Wire the pipeline from the frozen config (budget, k window, repair budget)."""
        return cls(
            provider,
            packer=ContextPacker.from_settings(settings),
            repair_passes=settings.generate.repair_passes,
            max_answer_tokens=settings.llm.max_tokens,
        )

    # -- public API ---------------------------------------------------------------
    def answer(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        *,
        language: str | None = None,
    ) -> GenerationResult:
        """Answer ``question`` from ``passages``; never raises for a normal failure mode."""
        resolved_language = language or detect_language(question)
        prompt = self.registry.get(ANSWER_PROMPT_ID)
        packed = self.pack(question, passages, prompt=prompt)
        accumulator = _UsageAccumulator(pricing=self.pricing)
        injection = scan(question, passages)
        if injection:
            return self._escalate(
                question=question,
                language=resolved_language,
                code=ReasonCode.INJECTION_SUSPECTED,
                detail="; ".join(
                    f"{finding.source}:{finding.ref}:{','.join(finding.rules)}"
                    for finding in injection
                ),
                packed=packed,
                prompt=prompt,
                accumulator=accumulator,
                injection=injection,
            )
        if packed.is_empty:
            return self._escalate(
                question=question,
                language=resolved_language,
                code=ReasonCode.NO_EVIDENCE,
                detail="no passage survived packing under the token budget",
                packed=packed,
                prompt=prompt,
                accumulator=accumulator,
                injection=injection,
            )

        draft, malformed = self._generate(
            prompt=prompt,
            question=question,
            language=resolved_language,
            packed=packed,
            accumulator=accumulator,
        )
        if draft is None:
            return self._escalate(
                question=question,
                language=resolved_language,
                code=ReasonCode.PROVIDER_ERROR,
                detail=malformed,
                packed=packed,
                prompt=prompt,
                accumulator=accumulator,
                injection=injection,
            )
        if draft.is_refusal:
            return self._escalate(
                question=question,
                language=resolved_language,
                code=ReasonCode.NO_EVIDENCE,
                detail=draft.why or "model declared the evidence insufficient",
                packed=packed,
                prompt=prompt,
                accumulator=accumulator,
                injection=injection,
            )

        build = self.builder.build(draft, packed)
        checks = self._check(build, packed)
        numeric_violations_pre_repair = sum(1 for check in checks if not check.numeric_ok)
        repaired = False
        repair_attempts = 0

        if self._needs_repair(build, checks) and self.repair_passes > 0:
            repair_attempts += 1
            repair_draft = self._repair(
                question=question,
                language=resolved_language,
                packed=packed,
                draft=draft,
                checks=checks,
                build=build,
                accumulator=accumulator,
            )
            if repair_draft is not None and not repair_draft.is_refusal:
                repaired = True
                draft = repair_draft
                build = self.builder.build(draft, packed)
                checks = self._check(build, packed)

        verdicts = {check.claim_id: check for check in checks}
        survivor_claims: list[Claim] = []
        for claim in build.claims:
            check = verdicts.get(claim.claim_id)
            if check is None:
                continue
            if check.verdict is Support.UNSUPPORTED:
                continue
            survivor_claims.append(claim.model_copy(update={"support": check.verdict}))
        dropped = len(build.claims) - len(survivor_claims)
        if not survivor_claims:
            code = (
                ReasonCode.CITATION_FAILED
                if any(not check.numeric_ok or not check.quote_ok for check in checks)
                else ReasonCode.VERIFIER_REJECT
            )
            return self._escalate(
                question=question,
                language=resolved_language,
                code=code,
                detail="no claim survived verification",
                packed=packed,
                prompt=prompt,
                accumulator=accumulator,
                injection=injection,
                repaired=repaired,
                repair_attempts=repair_attempts,
                dropped=dropped,
                numeric_violations=numeric_violations_pre_repair,
            )

        claims, citations = self._prune(survivor_claims, build.citations)
        spoken = " ".join(claim.text for claim in claims)
        notice = None
        language_notice_used = False
        if self.language_notice_enabled and not languages_agree(resolved_language, spoken):
            notice = language_notice(resolved_language)
            language_notice_used = True
        text = render_answer_text(claims, notice=notice)
        self._assert_render(text, claims, citations)
        return GenerationResult(
            question=question,
            language=resolved_language,
            text=text,
            claims=claims,
            citations=citations,
            signal=GenerationSignal(
                action=Action.ANSWER,
                reason_code=ReasonCode.ANSWERED,
                detail="verified",
                evidence_score=packed.evidence_score,
            ),
            usage=accumulator.usage(prompt_version=prompt.prompt_version),
            prompt_version=prompt.prompt_version,
            verifier_version=self.verifier_version,
            repaired=repaired,
            repair_attempts=repair_attempts,
            dropped_claims=dropped,
            numeric_violations_pre_repair=numeric_violations_pre_repair,
            language_notice=language_notice_used,
            injection=injection,
            packed=packed,
            model=accumulator.model,
        )

    # -- internals ----------------------------------------------------------------
    def reserved_tokens(self, prompt: Prompt, question: str) -> int:
        """Budget held back for the system prompt, the question, and the answer."""
        overhead = (
            estimate_tokens(prompt.system)
            + estimate_tokens(prompt.user)
            + estimate_tokens(question)
            + self.max_answer_tokens
            + 32
        )
        return overhead

    def pack(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        *,
        prompt: Prompt,
    ) -> PackedContext:
        return self.packer.pack(passages, reserved_tokens=self.reserved_tokens(prompt, question))

    def build_messages(
        self, prompt: Prompt, *, question: str, language: str, packed: PackedContext
    ) -> list[Message]:
        values = {
            "question": question,
            "language": language,
            "language_name": language_name(language),
            "evidence": packed.render_evidence(),
        }
        return [
            Message(role=Role.SYSTEM, content=prompt.render_system(**values)),
            Message(role=Role.USER, content=prompt.render_user(**values)),
        ]

    def _generate(
        self,
        *,
        prompt: Prompt,
        question: str,
        language: str,
        packed: PackedContext,
        accumulator: _UsageAccumulator,
    ) -> tuple[AnswerDraft | None, str]:
        """One generation call; on malformed output exactly one strict re-ask."""
        messages = self.build_messages(prompt, question=question, language=language, packed=packed)
        completion = self.provider.complete(messages, response_schema=self.schema)
        accumulator.add(completion)
        try:
            return parse_answer_draft(completion.text), ""
        except MalformedModelOutput as first_error:
            strict_messages = [
                messages[0],
                Message(
                    role=Role.USER,
                    content=(
                        messages[1].content
                        + "\n\nReturn ONLY the JSON object that matches the schema. "
                        "No markdown fences, no explanation."
                    ),
                ),
            ]
            completion = self.provider.complete(strict_messages, response_schema=self.schema)
            accumulator.add(completion)
            try:
                return parse_answer_draft(completion.text), ""
            except MalformedModelOutput as second_error:
                return None, f"{first_error.message}; after strict re-ask: {second_error.message}"

    def _check(self, build: CitationBuild, packed: PackedContext) -> list[ClaimCheck]:
        checker = self.verifier
        if isinstance(checker, LexicalVerifier):
            return checker.check(build.claims, build.citation_map(), packed.chunk_map())
        # Any other ClaimVerifier: fall back to the contract protocol, keyed by claim id.
        chunks = packed.chunk_map()
        citations = build.citation_map()
        mapping = {claim.claim_id: cited_chunks(claim, citations, chunks) for claim in build.claims}
        verdicts = checker.verify(build.claims, mapping)
        by_id = {verdict.claim_id: verdict for verdict in verdicts}
        checks: list[ClaimCheck] = []
        for claim in build.claims:
            verdict = by_id.get(claim.claim_id)
            support = verdict.verdict if verdict is not None else Support.UNSUPPORTED
            checks.append(
                ClaimCheck(
                    claim_id=claim.claim_id,
                    verdict=support,
                    coverage=1.0 if support is Support.SUPPORTED else 0.0,
                    rationale=verdict.rationale if verdict is not None else "no verdict",
                )
            )
        return checks

    @staticmethod
    def _needs_repair(build: CitationBuild, checks: Sequence[ClaimCheck]) -> bool:
        if build.issues:
            return True
        return any(check.verdict is Support.UNSUPPORTED for check in checks)

    def _repair(
        self,
        *,
        question: str,
        language: str,
        packed: PackedContext,
        draft: AnswerDraft,
        checks: Sequence[ClaimCheck],
        build: CitationBuild,
        accumulator: _UsageAccumulator,
    ) -> AnswerDraft | None:
        """Exactly one repair pass (FR-GEN-07); returns ``None`` if it cannot be parsed."""
        prompt = self.registry.get(REPAIR_PROMPT_ID)
        reasons = {check.claim_id: check.rationale for check in checks}
        rejected_lines: list[str] = []
        for claim in build.claims:
            check = next((c for c in checks if c.claim_id == claim.claim_id), None)
            if check is None or check.verdict is Support.SUPPORTED:
                continue
            reason = reasons.get(claim.claim_id, "unsupported")
            rejected_lines.append(f"- {claim.text} — {reason}")
        for issue in build.issues:
            rejected_lines.append(
                f"- citation rejected ({issue.code}): passage {issue.passage_label!r}, "
                f"quote {issue.quote!r}"
            )
        values = {
            "question": question,
            "language": language,
            "language_name": language_name(language),
            "evidence": packed.render_evidence(),
            "rejected": "\n".join(rejected_lines) or "- (no supported claim remained)",
            "previous_draft": draft_to_json(draft),
        }
        messages = [
            Message(role=Role.SYSTEM, content=prompt.render_system(**values)),
            Message(role=Role.USER, content=prompt.render_user(**values)),
        ]
        completion = self.provider.complete(messages, response_schema=self.schema)
        accumulator.add(completion)
        try:
            return parse_answer_draft(completion.text)
        except MalformedModelOutput:
            return None

    @staticmethod
    def _prune(
        claims: list[Claim], citations: list[Citation]
    ) -> tuple[list[Claim], list[Citation]]:
        """Drop unreferenced citations and renumber by first appearance (CR-5)."""
        by_id = {citation.citation_id: citation for citation in citations}
        order: dict[int, int] = {}
        for claim in claims:
            for citation_id in claim.citation_ids:
                if citation_id not in order and citation_id in by_id:
                    order[citation_id] = len(order) + 1
        kept_citations = [
            by_id[old_id].model_copy(update={"citation_id": new_id})
            for old_id, new_id in sorted(order.items(), key=lambda item: item[1])
        ]
        kept_claims = [
            claim.model_copy(
                update={
                    "citation_ids": [
                        order[citation_id]
                        for citation_id in claim.citation_ids
                        if citation_id in order
                    ]
                }
            )
            for claim in claims
        ]
        return kept_claims, kept_citations

    def _assert_render(self, text: str, claims: list[Claim], citations: list[Citation]) -> None:
        problems = validate_render(text, claims, citations)
        if problems:  # pragma: no cover - a bug, not a model failure
            raise AssertionError(f"rendered answer violates the citation contract: {problems}")

    def _escalate(
        self,
        *,
        question: str,
        language: str,
        code: ReasonCode,
        detail: str,
        packed: PackedContext,
        prompt: Prompt,
        accumulator: _UsageAccumulator,
        injection: list[InjectionFinding],
        repaired: bool = False,
        repair_attempts: int = 0,
        dropped: int = 0,
        numeric_violations: int = 0,
    ) -> GenerationResult:
        return GenerationResult(
            question=question,
            language=language,
            text="",
            claims=[],
            citations=[],
            signal=GenerationSignal(
                action=Action.ESCALATE,
                reason_code=code,
                detail=detail,
                evidence_score=packed.evidence_score,
            ),
            usage=accumulator.usage(prompt_version=prompt.prompt_version),
            prompt_version=prompt.prompt_version,
            verifier_version=self.verifier_version,
            repaired=repaired,
            repair_attempts=repair_attempts,
            dropped_claims=dropped,
            numeric_violations_pre_repair=numeric_violations,
            injection=injection,
            packed=packed,
            model=accumulator.model,
        )


__all__ = [
    "ANSWER_PROMPT_ID",
    "REPAIR_PROMPT_ID",
    "AnswerPipeline",
    "GenerationResult",
    "GenerationSignal",
]
