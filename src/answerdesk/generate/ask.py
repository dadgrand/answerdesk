"""``answerdesk ask``: the offline-friendly question path behind CMD-H.

The command is a thin shell around :class:`~answerdesk.generate.pipeline.AnswerPipeline`:

* ``--fake`` uses the deterministic provider and the checked-in demo evidence pack
  (:mod:`answerdesk.generate.demo`), so it works from a clean checkout with no key, no
  network and no corpus (FR-OPS-03, INV-08);
* a real profile uses the configured provider and needs a passage source. The retrieval
  stack is owned by t5 and wired into the CLI/service by t9/t14; until then this function
  says so instead of silently answering from the demo corpus in production mode.

``run_ask`` is importable on its own so t7 (evaluation) and t9 (service) can drive the
pipeline without going through typer.

    # AT-OPS-03  # AT-OPS-04  # AT-GEN-10
"""

from __future__ import annotations

from collections.abc import Sequence

from answerdesk.config import Settings
from answerdesk.contracts import ChatProvider, RetrievedPassage
from answerdesk.errors import ConfigError
from answerdesk.generate.demo import demo_passages, demo_responses
from answerdesk.generate.pipeline import AnswerPipeline, GenerationResult
from answerdesk.llm.cost import CostLedger
from answerdesk.llm.factory import build_provider
from answerdesk.llm.presets import provider_config


def _fake_provider() -> ChatProvider:
    """The demo double: deterministic replies keyed on the demo evidence."""
    from answerdesk.testing.doubles import FakeProvider

    return FakeProvider(provider_config("fake"), responses=demo_responses())


def run_ask(
    question: str,
    *,
    settings: Settings,
    fake: bool = False,
    provider: ChatProvider | None = None,
    passages: Sequence[RetrievedPassage] | None = None,
    pipeline: AnswerPipeline | None = None,
    ledger: CostLedger | None = None,
) -> GenerationResult:
    """Answer one question and return the full :class:`GenerationResult`."""
    resolved_provider = provider
    resolved_passages = passages
    if resolved_provider is None:
        resolved_provider = _fake_provider() if fake else build_provider(settings, ledger=ledger)
    if resolved_passages is None:
        if not fake:
            raise ConfigError(
                "answerdesk ask needs a passage source: use --fake for the offline demo pack, "
                "or inject passages from the retrieval stack (wired by the service layer, t9)."
            )
        resolved_passages = demo_passages(question)
    resolved_pipeline = pipeline or AnswerPipeline.from_settings(resolved_provider, settings)
    return resolved_pipeline.answer(question, resolved_passages)


def render_for_terminal(result: GenerationResult) -> str:
    """Human-readable summary: answer, citations, decision, usage."""
    lines: list[str] = [f"Q: {result.question}"]
    if result.signal.is_answer:
        lines.append(f"A: {result.text}")
        if result.citations:
            lines.append("Citations:")
            for citation in result.citations:
                heading = " > ".join(citation.heading_path) or citation.doc_title
                lines.append(
                    f"  [{citation.citation_id}] {citation.doc_title} — {heading} "
                    f"({citation.chunk_id}) chars {citation.start_char}-{citation.end_char}"
                )
                lines.append(f"      “{citation.quote}”")
    else:
        lines.append(
            f"A: (no prose answer) {result.signal.action.value}: "
            f"{result.signal.reason_code.value} — {result.signal.detail}"
        )
    lines.append(
        f"Decision: {result.signal.action.value} ({result.signal.reason_code.value}); "
        f"language={result.language}; evidence_score={result.signal.evidence_score:.3f}; "
        f"repaired={result.repaired}; dropped={result.dropped_claims}"
    )
    lines.append(
        f"Usage: model={result.usage.model or '(none)'} input={result.usage.input_tokens} "
        f"output={result.usage.output_tokens} usd={result.usage.usd:.8f} "
        f"prompt={result.prompt_version or '(none)'} verifier={result.verifier_version or '(none)'}"
    )
    return "\n".join(lines)


__all__ = ["render_for_terminal", "run_ask"]
