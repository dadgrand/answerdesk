"""Grounded generation: prompt registry, context packing, citations, verifier.

Owning task: **t6**. Public surface for the rest of the stack (t7 evaluation, t8 policy,
t9 service):

============================  ========================================================
:class:`AnswerPipeline`        pack → prompt → generate → cite → verify → repair
:class:`GenerationResult`      text, claims, citations, signal, usage, provenance
:class:`PromptRegistry`        versioned, hashed prompts (``configs/prompts``)
:class:`ContextPacker`         hard token budget, dedupe, lost-in-the-middle ordering
:class:`CitationBuilder`       span-level citation validation (CR-1 … CR-6)
:class:`LexicalVerifier`       deterministic claim verdicts (supported/partial/unsupported)
``detect_injection``           directive-shaped evidence guard (INV-01, NFR-SEC-01)
``numeric_violations``         the deterministic number/date guard (INV-04, CR-9)
``run_ask``                    the ``answerdesk ask`` body (offline demo pack)
============================  ========================================================

Requirements: FR-GEN-01 … FR-GEN-12, NFR-SEC-01; the contract is ``docs/requirements.md``
§7 and the operational walkthrough is ``docs/generation.md``.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.generate.ask import render_for_terminal, run_ask
from answerdesk.generate.citations import (
    CitationBuild,
    CitationBuilder,
    CitationIssue,
    find_quote_span,
    markers_in,
    normalise_whitespace,
    render_answer_text,
    validate_render,
)
from answerdesk.generate.demo import demo_cases, demo_corpus, demo_passages
from answerdesk.generate.draft import AnswerDraft, CitationDraft, ClaimDraft, parse_answer_draft
from answerdesk.generate.errors import (
    EvidenceError,
    GenerationError,
    MalformedModelOutput,
    PromptRegistryError,
    PromptRenderError,
    PromptVersionError,
)
from answerdesk.generate.guards import (
    INJECTION_RULES,
    InjectionFinding,
    contains_canary,
    detect_injection,
    scan,
    scan_passages,
    scan_query,
)
from answerdesk.generate.language import (
    SUPPORTED_LOCALES,
    detect_language,
    language_name,
    language_notice,
    languages_agree,
    text_language,
)
from answerdesk.generate.numeric import (
    canonical_number,
    canonical_numeric_tokens,
    numeric_violations,
)
from answerdesk.generate.packer import (
    ContextPacker,
    PackedContext,
    PackedPassage,
    lost_in_the_middle_order,
    token_cosine,
)
from answerdesk.generate.pipeline import (
    ANSWER_PROMPT_ID,
    REPAIR_PROMPT_ID,
    AnswerPipeline,
    GenerationResult,
    GenerationSignal,
)
from answerdesk.generate.prompts import (
    Prompt,
    PromptLock,
    PromptRegistry,
    content_digest,
    stamp_registry,
)
from answerdesk.generate.schema import (
    ANSWER_SCHEMA_NAME,
    assert_strict_schema,
    load_answer_schema,
    schema_digest,
)
from answerdesk.generate.verifier import ClaimCheck, LexicalVerifier, content_tokens, coverage

__all__ = [
    "ANSWER_PROMPT_ID",
    "ANSWER_SCHEMA_NAME",
    "INJECTION_RULES",
    "REPAIR_PROMPT_ID",
    "SUPPORTED_LOCALES",
    "AnswerDraft",
    "AnswerPipeline",
    "CitationBuild",
    "CitationBuilder",
    "CitationDraft",
    "CitationIssue",
    "ClaimCheck",
    "ClaimDraft",
    "ContextPacker",
    "EvidenceError",
    "GenerationError",
    "GenerationResult",
    "GenerationSignal",
    "InjectionFinding",
    "LexicalVerifier",
    "MalformedModelOutput",
    "PackedContext",
    "PackedPassage",
    "Prompt",
    "PromptLock",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptRenderError",
    "PromptVersionError",
    "assert_strict_schema",
    "canonical_number",
    "canonical_numeric_tokens",
    "contains_canary",
    "content_digest",
    "content_tokens",
    "coverage",
    "demo_cases",
    "demo_corpus",
    "demo_passages",
    "detect_injection",
    "detect_language",
    "find_quote_span",
    "language_name",
    "language_notice",
    "languages_agree",
    "load_answer_schema",
    "lost_in_the_middle_order",
    "markers_in",
    "normalise_whitespace",
    "numeric_violations",
    "parse_answer_draft",
    "render_answer_text",
    "render_for_terminal",
    "run_ask",
    "scan",
    "scan_passages",
    "scan_query",
    "schema_digest",
    "stamp_registry",
    "text_language",
    "token_cosine",
    "validate_render",
]
