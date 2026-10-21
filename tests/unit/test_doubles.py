"""Test doubles must be deterministic and honest — CI quality depends on it.

# AT-PORT-01
"""

from __future__ import annotations

import hashlib
import math

import pytest

from answerdesk.contracts import (
    Answer,
    Chunk,
    Claim,
    Message,
    Role,
    Support,
    TraceRecord,
)
from answerdesk.testing import (
    FakeJudge,
    FakeProvider,
    FakeVerifier,
    HashingEmbedder,
    InMemoryCache,
    InMemoryTraceSink,
    default_fake_provider_config,
    make_passage,
)


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=64)


def test_embedding_is_a_unit_vector(embedder: HashingEmbedder) -> None:
    vector = embedder.embed_query("Как вернуть деньги за карту?")
    assert len(vector) == 64
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-9)


def test_embedding_is_deterministic_across_instances() -> None:
    """Same text, new instance, same bytes — no PYTHONHASHSEED dependence."""
    first = HashingEmbedder(dimension=32).embed_query("refund policy")
    second = HashingEmbedder(dimension=32).embed_query("refund policy")
    assert first == second
    digest = hashlib.sha256(repr(first).encode()).hexdigest()
    assert digest  # stable representation available to fixtures


def test_documents_and_query_use_the_same_space(embedder: HashingEmbedder) -> None:
    vectors = embedder.embed_documents(["refund policy", "card fees"])
    assert vectors[0] == embedder.embed_query("refund policy")
    assert vectors[0] != vectors[1]


def test_identical_text_has_similarity_one(embedder: HashingEmbedder) -> None:
    assert embedder.similarity("refund policy", "refund policy") == pytest.approx(1.0, abs=1e-9)
    assert embedder.similarity("refund policy", "") == pytest.approx(0.0, abs=1e-9)


def test_empty_text_yields_a_zero_vector(embedder: HashingEmbedder) -> None:
    assert embedder.embed_query("") == [0.0] * 64


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbedder(dimension=0)


def test_fake_provider_is_deterministic() -> None:
    provider = FakeProvider()
    messages = [Message(role=Role.USER, content="Сколько идёт возврат?")]
    first = provider.complete(messages)
    second = provider.complete(messages)
    assert first.text == second.text
    assert first.output_tokens == second.output_tokens
    assert len(provider.calls) == 2
    assert provider.config.requires_api_key is False


def test_fake_provider_canned_responses_win() -> None:
    provider = FakeProvider(responses={"refund": "Refunds take 14 days [1]."})
    completion = provider.complete([Message(role=Role.USER, content="what is the refund time?")])
    assert completion.text == "Refunds take 14 days [1]."


def test_fake_provider_stream_reassembles_the_completion() -> None:
    provider = FakeProvider(chunk_words=2)
    messages = [Message(role=Role.USER, content="refund policy question")]
    chunks = provider.stream(messages)
    assert chunks[-1].done is True
    streamed = "".join(chunk.delta for chunk in chunks)
    assert provider.complete(messages).text.split()[0] in streamed


def test_fake_provider_records_the_response_schema() -> None:
    provider = FakeProvider()
    provider.complete(
        [Message(role=Role.USER, content="hi")],
        response_schema={"type": "object", "properties": {"claims": {"type": "array"}}},
    )
    assert provider.schemas[0] is not None
    assert "claims" in provider.schemas[0]["properties"]


def test_default_fake_config_is_keyless() -> None:
    config = default_fake_provider_config()
    assert config.preset == "fake"
    assert config.requires_api_key is False
    assert config.api_key is None


def test_fake_judge_prefers_a_matching_answer() -> None:
    judge = FakeJudge()
    reference = "Refunds are processed within 14 days of approval."
    perfect = judge.grade(
        question="q", reference_answer=reference, candidate_answer=reference, context=[]
    )
    wrong = judge.grade(
        question="q", reference_answer=reference, candidate_answer="Cats are nice.", context=[]
    )
    assert perfect.score == pytest.approx(5.0)
    assert wrong.score < 3.0
    assert wrong.score < perfect.score
    assert perfect.judge_model == "fake-judge"
    assert perfect.rubric_version


def test_fake_verifier_uses_the_marker() -> None:
    verifier = FakeVerifier()
    claims = [
        Claim(claim_id="c1", text="Refunds take 14 days.", citation_ids=[1]),
        Claim(claim_id="c2", text="[unsupported] Refunds are instant.", citation_ids=[1]),
    ]
    verdicts = verifier.verify(claims, {})
    assert [v.verdict for v in verdicts] == [Support.SUPPORTED, Support.UNSUPPORTED]
    assert all(v.verifier_version for v in verdicts)


def test_in_memory_cache_hit_rate_and_invalidation() -> None:
    cache = InMemoryCache()
    assert cache.stats()["hit_rate"] == 0.0
    assert cache.get("missing") is None

    cached_answer = _answer_with_version("corpus-1")
    cache.put("k1", cached_answer)
    assert cache.get("k1") == cached_answer
    assert cache.stats()["hit_rate"] == pytest.approx(0.5)

    assert cache.invalidate("corpus-2") == 1
    assert cache.get("k1") is None


def test_in_memory_trace_sink_keeps_records() -> None:
    sink = InMemoryTraceSink()
    sink.write(TraceRecord(trace_id="t1", query="q"))
    sink.write(TraceRecord(trace_id="t2", query="q"))
    assert sink.by_trace("t2") is not None
    assert sink.by_trace("nope") is None


def test_make_passage_helper() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="text",
        start_char=0,
        end_char=4,
        token_count=1,
    )
    passage = make_passage(chunk, score=0.5, rank=2)
    assert passage.chunk is chunk
    assert passage.rank == 2


def _answer_with_version(corpus_version: str) -> Answer:
    from answerdesk.contracts import (
        Action,
        CacheInfo,
        Citation,
        Confidence,
        Decision,
        Provenance,
        ReasonCode,
        Usage,
    )

    citation = Citation(
        citation_id=1,
        chunk_id="doc-1::0",
        doc_id="doc-1",
        quote="Refunds are processed within 14 days",
        start_char=0,
        end_char=38,
        doc_hash="c" * 64,
    )
    return Answer(
        answer_id="a1",
        trace_id="t1",
        language="en",
        text="Refunds take 14 days [1].",
        claims=[Claim(claim_id="c1", text="Refunds take 14 days.", citation_ids=[1])],
        citations=[citation],
        decision=Decision(
            action=Action.ANSWER,
            reason_code=ReasonCode.ANSWERED,
            score=0.9,
            alpha=0.05,
            threshold_ref="cal",
        ),
        confidence=Confidence(p_correct=0.9, calibrator_version="v1"),
        cache=CacheInfo(),
        usage=Usage(),
        provenance=Provenance(corpus_version=corpus_version),
    )
