"""Contract tests: the frozen seams must reject drift loudly.

These tests encode the invariants of ``docs/requirements.md`` that the skeleton owns:
CR-1/INV-03 (no uncited claim displayed), INV-02 (citations resolve), the closed reason
code enum (FR-ESC-05 parity with Appendix B) and the "no unknown fields" rule (ADR-006).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from answerdesk.contracts import (
    Action,
    Answer,
    CacheInfo,
    CacheStatus,
    Channel,
    Chunk,
    Citation,
    Claim,
    Confidence,
    Decision,
    Document,
    EscalationDecision,
    EvalCase,
    EvalCategory,
    EvalProvenance,
    IntentBucket,
    ProviderConfig,
    ReasonCode,
    RetrievalTrace,
    RetrievedPassage,
    Split,
    Support,
    TraceRecord,
    Usage,
)
from answerdesk.testing import (
    FakeJudge,
    FakeProvider,
    FakeVerifier,
    HashingEmbedder,
    InMemoryCache,
    InMemoryTraceSink,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _appendix_b_codes() -> set[str]:
    """Parse the closed reason-code enum out of docs/requirements.md Appendix B."""
    text = (REPO_ROOT / "docs" / "requirements.md").read_text(encoding="utf-8")
    start = text.index("## Appendix B")
    end = text.index("## Appendix C")
    return set(re.findall(r"^\| `([A-Z_]+)` \|", text[start:end], flags=re.MULTILINE))


def test_reason_codes_match_requirements() -> None:
    """FR-ESC-05: the code enum, the docs and the console share one closed set."""
    documented = _appendix_b_codes()
    assert len(documented) == 16, documented
    assert documented == {code.value for code in ReasonCode}


def test_answer_round_trips_through_json(make_answer: Callable[..., Answer]) -> None:
    answer = make_answer()
    restored = Answer.model_validate_json(answer.model_dump_json())
    assert restored == answer
    assert json.loads(answer.model_dump_json())["decision"]["action"] == "answer"


def test_unknown_fields_are_rejected(make_chunk: Callable[..., Chunk]) -> None:
    """ADR-006: a new cross-module field is a reviewable contract change."""
    chunk = make_chunk()
    payload = chunk.model_dump()
    payload["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        Chunk.model_validate(payload)


def test_claim_referencing_unknown_citation_is_rejected(
    make_answer: Callable[..., Answer],
) -> None:
    with pytest.raises(ValidationError, match="unknown citation ids"):
        make_answer(claims=[Claim(claim_id="c1", text="Refunds take 14 days.", citation_ids=[99])])


def test_answer_may_not_display_an_uncited_claim(make_answer: Callable[..., Answer]) -> None:
    """CR-1 / INV-03."""
    with pytest.raises(ValidationError, match="no citation"):
        make_answer(claims=[Claim(claim_id="c1", text="Refunds take 14 days.", citation_ids=[])])


def test_answer_may_not_display_an_unsupported_claim(
    make_answer: Callable[..., Answer],
) -> None:
    """CR-7 / INV-03."""
    with pytest.raises(ValidationError, match="unsupported"):
        make_answer(
            claims=[
                Claim(
                    claim_id="c1",
                    text="Refunds take 14 days.",
                    citation_ids=[1],
                    support=Support.UNSUPPORTED,
                )
            ]
        )


def test_duplicate_citation_ids_are_rejected(make_answer: Callable[..., Answer]) -> None:
    citation = Citation(
        citation_id=1,
        chunk_id="doc-1::0",
        doc_id="doc-1",
        quote="Refunds are processed within 14 days",
        start_char=0,
        end_char=38,
        doc_hash="a" * 64,
    )
    with pytest.raises(ValidationError, match="unique"):
        make_answer(citations=[citation, citation])


@pytest.mark.parametrize(
    ("action", "reason_code"),
    [
        (Action.ANSWER, ReasonCode.NO_EVIDENCE),
        (Action.ESCALATE, ReasonCode.ANSWERED),
        (Action.REFUSE, ReasonCode.ANSWERED),
    ],
)
def test_decision_reason_code_must_match_action(action: Action, reason_code: ReasonCode) -> None:
    """Appendix B: `ANSWERED` only for answers, and an answer always carries it."""
    with pytest.raises(ValidationError):
        Decision(action=action, reason_code=reason_code, score=0.5, alpha=0.05, threshold_ref="x")


def test_decision_accepts_a_consistent_pair() -> None:
    decision = Decision(
        action=Action.ESCALATE,
        reason_code=ReasonCode.LOW_CONFIDENCE,
        score=0.42,
        alpha=0.05,
        threshold_ref="calibration-dev-001",
    )
    assert decision.action is Action.ESCALATE


def test_escalation_decision_requires_escalation_action() -> None:
    answer_decision = Decision(
        action=Action.ANSWER,
        reason_code=ReasonCode.ANSWERED,
        score=0.9,
        alpha=0.05,
        threshold_ref="x",
    )
    with pytest.raises(ValidationError, match="escalate"):
        EscalationDecision(decision=answer_decision)


def test_cache_hit_requires_an_entry_id() -> None:
    with pytest.raises(ValidationError, match="entry_id"):
        CacheInfo(status=CacheStatus.HIT)


def test_citation_span_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="end_char"):
        Citation(
            citation_id=1,
            chunk_id="c",
            doc_id="d",
            quote="q",
            start_char=10,
            end_char=5,
            doc_hash="b" * 64,
        )


def test_chunk_span_and_doc_hash(make_chunk: Callable[..., Chunk]) -> None:
    chunk = make_chunk()
    assert len(chunk.doc_hash) == 64
    assert chunk.doc_hash == make_chunk().doc_hash  # stable, content addressed
    with pytest.raises(ValidationError, match="end_char"):
        make_chunk(start_char=5, end_char=5)


def test_document_requires_hex_sha256(make_document: Callable[..., Document]) -> None:
    document = make_document()
    assert document.doc_id == "doc-1"
    with pytest.raises(ValidationError, match="lowercase hex"):
        make_document(sha256="Z" * 64)


def test_provider_config_redacts_the_key() -> None:
    config = ProviderConfig(
        preset="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-super-secret-value",  # type: ignore[arg-type]
    )
    dumped = config.redacted()
    assert dumped["api_key"] == "***"
    assert "sk-super-secret-value" not in json.dumps(dumped)


def test_eval_case_label_rules() -> None:
    case = EvalCase(
        case_id="case-1",
        locale="ru",
        category=EvalCategory.DIRECT,
        question="Сколько идёт возврат?",
        answerable=True,
        gold_chunk_ids=["doc-1::0"],
        provenance=EvalProvenance(source="document", generator="synthetic-v1", seed=7),
    )
    assert case.split is Split.TEST
    assert case.intent_bucket is IntentBucket.INFORMATIONAL

    with pytest.raises(ValidationError, match="reference answer"):
        EvalCase(
            case_id="case-2",
            locale="ru",
            category=EvalCategory.UNANSWERABLE,
            question="?",
            answerable=False,
            reference_answer="nope",
        )
    with pytest.raises(ValidationError, match="gold chunk"):
        EvalCase(
            case_id="case-3",
            locale="tg",
            category=EvalCategory.DIRECT,
            question="?",
            answerable=True,
        )


def test_retrieval_trace_and_trace_record(make_chunk: Callable[..., Chunk]) -> None:
    passage = RetrievedPassage(chunk=make_chunk(), channel=Channel.DENSE, score=0.8, rank=1)
    trace = RetrievalTrace(query="refund", candidates=[passage], fused=[passage])
    record = TraceRecord(trace_id="trace-1", query="refund", retrieval=trace)
    assert record.retrieval is not None
    assert record.retrieval.fused[0].channel is Channel.DENSE
    assert record.created_at.tzinfo is not None


def test_all_doubles_satisfy_their_protocols() -> None:
    """The doubles must be drop-in replacements, not look-alikes (ADR-004)."""
    from answerdesk.contracts import (
        CacheBackend,
        ChatProvider,
        ClaimVerifier,
        Embedder,
        Judge,
        TraceSink,
    )

    embedder = HashingEmbedder()
    provider = FakeProvider()
    assert isinstance(embedder, Embedder)
    assert isinstance(provider, ChatProvider)
    assert isinstance(FakeJudge(), Judge)
    assert isinstance(FakeVerifier(), ClaimVerifier)
    assert isinstance(InMemoryCache(), CacheBackend)
    assert isinstance(InMemoryTraceSink(), TraceSink)
    assert provider.config.requires_api_key is False


def test_usage_and_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Usage(usd=-1.0)
    with pytest.raises(ValidationError):
        Confidence(p_correct=1.5, calibrator_version="x")
