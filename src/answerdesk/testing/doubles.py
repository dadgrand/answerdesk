"""Deterministic test doubles — the only reason CI can be offline and trustworthy.

``HashingEmbedder``, ``FakeProvider``, ``FakeJudge``, ``FakeVerifier`` and the in-memory
cache/trace sink implement the Protocols in :mod:`answerdesk.contracts` without network,
without model weights and without randomness. They are what the G-CI gate runs against
(requirements §0.3, ADR-004).

These doubles are deliberately *not* clever: they must be boring enough that a test
failure means the code under test broke, not the double.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from answerdesk.contracts import (
    Answer,
    Chunk,
    Claim,
    ClaimVerification,
    Completion,
    CompletionChunk,
    JudgeScore,
    Message,
    ProviderConfig,
    RetrievedPassage,
    Support,
    TraceRecord,
)

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Lowercased word unigrams plus bigrams — enough signal for ranking tests."""
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a} {b}" for a, b in itertools.pairwise(words)]
    return words + bigrams


class HashingEmbedder:
    """Deterministic hashed bag-of-features embedder (no model, no network, no numpy).

    Identical text always produces an identical unit vector, in any process, on any OS:
    hashes come from BLAKE2b, never from Python's randomised ``hash()``.
    """

    def __init__(self, dimension: int = 64, *, name: str | None = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.name = name or f"hashing-{dimension}"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity helper for tests and cache doubles."""
        a, b = self._vector(left), self._vector(right)
        return math.fsum(x * y for x, y in zip(a, b, strict=True))


FAKE_MODEL = "fake-model"


def default_fake_provider_config() -> ProviderConfig:
    return ProviderConfig(
        preset="fake",
        base_url="memory://fake",
        model=FAKE_MODEL,
        api_key=None,
        json_mode=False,
        requires_api_key=False,
    )


class FakeProvider:
    """Deterministic in-memory stand-in for an OpenAI-compatible provider (FR-GEN-09).

    The reply is derived only from the last user message, so a fixture that fixes the
    prompt fixes the completion — which is exactly what determinism tests need.
    """

    name = "fake"

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        responses: Mapping[str, str] | None = None,
        chunk_words: int = 4,
    ) -> None:
        self.config = config or default_fake_provider_config()
        self.responses = dict(responses or {})
        self.chunk_words = chunk_words
        self.calls: list[list[Message]] = []
        self.schemas: list[dict[str, Any] | None] = []

    # -- ChatProvider protocol ----------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        self.schemas.append(dict(response_schema) if response_schema is not None else None)
        text = self._reply(messages)
        return Completion(
            text=text,
            model=self.config.model,
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=len(text.split()),
        )

    def stream(self, messages: Sequence[Message]) -> Sequence[CompletionChunk]:
        self.calls.append(list(messages))
        text = self._reply(messages)
        words = text.split()
        chunks = [
            CompletionChunk(
                delta=" ".join(words[i : i + self.chunk_words]) + " ", model=self.config.model
            )
            for i in range(0, len(words), self.chunk_words)
        ]
        chunks.append(CompletionChunk(delta="", done=True, model=self.config.model))
        return chunks

    # -- helpers ------------------------------------------------------------------
    def _reply(self, messages: Sequence[Message]) -> str:
        last_user = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        for needle, canned in self.responses.items():
            if needle in last_user:
                return canned
        head = " ".join(last_user.split()[:20])
        return f"[{self.config.model}] deterministic reply to: {head}"


class FakeVerifier:
    """Marker-driven claim verifier: unsupported only when the fixture says so."""

    name = "fake-verifier"

    def __init__(self, *, version: str = "fake-verifier-1", marker: str = "[unsupported]") -> None:
        self.version = version
        self.marker = marker

    def verify(
        self, claims: Sequence[Claim], passages: Mapping[str, Sequence[Chunk]]
    ) -> list[ClaimVerification]:
        verdicts: list[ClaimVerification] = []
        for claim in claims:
            verdict = Support.UNSUPPORTED if self.marker in claim.text else Support.SUPPORTED
            verdicts.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    verdict=verdict,
                    rationale="fake-verifier marker rule",
                    verifier_version=self.version,
                )
            )
        return verdicts


class FakeJudge:
    """Lexical-overlap rubric judge with a deterministic 1–5 score (ADR-004)."""

    name = "fake-judge"

    def __init__(self, *, judge_model: str = "fake-judge", rubric_version: str = "fake-rubric-1"):
        self.judge_model = judge_model
        self.rubric_version = rubric_version

    def grade(
        self,
        *,
        question: str,
        reference_answer: str | None,
        candidate_answer: str,
        context: Sequence[str],
    ) -> JudgeScore:
        reference = reference_answer if reference_answer is not None else " ".join(context)
        overlap = _overlap_fraction(candidate_answer, reference)
        score = round(1.0 + 4.0 * overlap, 4)
        return JudgeScore(
            score=min(5.0, max(1.0, score)),
            rubric_version=self.rubric_version,
            judge_model=self.judge_model,
            rationale=f"lexical overlap {overlap:.3f}",
        )


class InMemoryCache:
    """Minimal cache backend used by tests; guard logic lives in ``answerdesk.cache``."""

    name = "in-memory"

    def __init__(self) -> None:
        self.entries: dict[str, Answer] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Answer | None:
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: Answer) -> None:
        self.entries[key] = value

    def invalidate(self, corpus_version: str) -> int:
        stale = [
            key
            for key, answer in self.entries.items()
            if answer.provenance.corpus_version != corpus_version
        ]
        for key in stale:
            del self.entries[key]
        return len(stale)

    def stats(self) -> Mapping[str, float]:
        total = self.hits + self.misses
        return {
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": (self.hits / total) if total else 0.0,
            "entries": float(len(self.entries)),
        }


class InMemoryTraceSink:
    """Collects :class:`TraceRecord` objects for assertions (t9 replaces with JSONL)."""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def write(self, record: TraceRecord) -> None:
        self.records.append(record)

    def by_trace(self, trace_id: str) -> TraceRecord | None:
        return next((r for r in self.records if r.trace_id == trace_id), None)


def _overlap_fraction(candidate: str, reference: str) -> float:
    candidate_tokens = set(_tokens(candidate))
    reference_tokens = set(_tokens(reference))
    if not candidate_tokens or not reference_tokens:
        return 0.0
    return len(candidate_tokens & reference_tokens) / len(candidate_tokens)


def make_passage(chunk: Chunk, score: float = 1.0, rank: int = 1) -> RetrievedPassage:
    """Small helper so tests do not hand-build passages."""
    from answerdesk.contracts import Channel

    return RetrievedPassage(chunk=chunk, channel=Channel.BM25, score=score, rank=rank)


__all__ = [
    "FAKE_MODEL",
    "FakeJudge",
    "FakeProvider",
    "FakeVerifier",
    "HashingEmbedder",
    "InMemoryCache",
    "InMemoryTraceSink",
    "default_fake_provider_config",
    "make_passage",
]
