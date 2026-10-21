"""Recorded-fixture provider: replay a captured transcript, deterministically, offline.

Why this exists (ADR-004): ``FakeProvider`` proves the *pipeline* is deterministic, but
``FakeProvider`` never produces a citation-shaped JSON answer. A recorded fixture holds
the real shape a model returned once, so generation tests can assert citation, repair
and refusal behaviour without a network call, an API key or a flaky model.

The file format is JSON, one object per exchange, keyed by a digest of the request
(messages + response-schema digest). Keys are digests rather than raw prompts so a
fixture can be committed without leaking a customer question, and so a prompt change
surfaces as a clean "fixture miss" instead of silently replaying the wrong answer.

    # AT-GEN-09  # AT-PORT-01
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from answerdesk.contracts import (
    ChatProvider,
    Completion,
    CompletionChunk,
    Contract,
    Message,
    ProviderConfig,
)
from answerdesk.llm.stream import ChunkStream

RECORDED_FIXTURE_VERSION = "1.0.0"
RECORDED_PRESET = "recorded"
RECORDED_MODEL = "recorded-model"


class RecordedFixtureMiss(LookupError):
    """No recorded exchange matches this request — the fixture is stale, not the code."""

    def __init__(self, digest: str, preview: str) -> None:
        super().__init__(
            f"no recorded exchange for request digest {digest[:12]}… (preview: {preview[:120]!r}). "
            "Re-record the fixture or update the prompt version."
        )
        self.digest = digest
        self.preview = preview


def canonical_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Role/content pairs in a stable order (the digest input)."""
    return [{"role": message.role.value, "content": message.content} for message in messages]


def schema_digest(response_schema: Mapping[str, Any] | None) -> str:
    if response_schema is None:
        return ""
    payload = json.dumps(dict(response_schema), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_digest(
    messages: Sequence[Message], response_schema: Mapping[str, Any] | None = None
) -> str:
    """Stable digest of everything that shapes a completion request."""
    material = json.dumps(canonical_messages(messages), sort_keys=True, ensure_ascii=False)
    combined = material + "\x00" + schema_digest(response_schema)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def request_preview(messages: Sequence[Message], *, limit: int = 160) -> str:
    """A content-free fingerprint of a request: roles and sizes, never prompt text.

    A fixture is committed to a public repository, so it must not carry a customer
    question (INV-05). The digest identifies the request; this preview only makes a
    fixture miss readable in a log line.
    """
    roles = ",".join(message.role.value for message in messages)
    chars = sum(len(message.content) for message in messages)
    return f"messages={len(messages)} roles={roles} chars={chars}"[:limit]


class RecordedResponse(Contract):
    """One captured completion, without latency (latency is not reproducible)."""

    text: str
    model: str = RECORDED_MODEL
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    finish_reason: str = "stop"


class RecordedExchange(Contract):
    request_digest: str = Field(min_length=64, max_length=64)
    request_preview: str = ""
    schema_digest: str = ""
    response: RecordedResponse


class RecordedFixture(Contract):
    version: str = RECORDED_FIXTURE_VERSION
    provider: str = RECORDED_PRESET
    model: str = RECORDED_MODEL
    exchanges: list[RecordedExchange] = Field(default_factory=list)


def default_recorded_config(*, model: str = RECORDED_MODEL) -> ProviderConfig:
    return ProviderConfig(
        preset=RECORDED_PRESET,
        base_url="memory://recorded",
        model=model,
        api_key=None,
        json_mode=True,
        requires_api_key=False,
    )


class RecordedProvider:
    """Replays a :class:`RecordedFixture`. Network-free by construction."""

    name = "recorded"

    def __init__(
        self,
        fixture: RecordedFixture,
        *,
        config: ProviderConfig | None = None,
        chunk_words: int = 4,
        allow_positional_fallback: bool = False,
    ) -> None:
        self.fixture = fixture
        self.config = config or default_recorded_config(model=fixture.model)
        self.chunk_words = max(1, chunk_words)
        self.allow_positional_fallback = allow_positional_fallback
        self._by_digest: dict[str, RecordedExchange] = {
            exchange.request_digest: exchange for exchange in fixture.exchanges
        }
        self._order = list(fixture.exchanges)
        self._cursor = 0
        self._used: set[str] = set()
        self.calls: list[dict[str, Any]] = []

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_file(
        cls, path: str | Path, *, config: ProviderConfig | None = None
    ) -> RecordedProvider:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(RecordedFixture.model_validate(raw), config=config)

    # -- ChatProvider -------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        exchange = self._lookup(messages, response_schema)
        response = exchange.response
        self.calls.append(
            {
                "digest": exchange.request_digest,
                "request_preview": request_preview(messages),
                "schema": None if response_schema is None else dict(response_schema),
            }
        )
        return Completion(
            text=response.text,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            finish_reason=_finish_reason(response.finish_reason),
        )

    def stream(self, messages: Sequence[Message]) -> ChunkStream:
        return ChunkStream(self._stream_chunks(messages))

    def _stream_chunks(self, messages: Sequence[Message]) -> Iterator[CompletionChunk]:
        completion = self.complete(messages)
        words = completion.text.split()
        for start in range(0, len(words), self.chunk_words):
            yield CompletionChunk(
                delta=" ".join(words[start : start + self.chunk_words]) + " ",
                done=False,
                model=completion.model,
            )
        yield CompletionChunk(delta="", done=True, model=completion.model)

    # -- lookup -------------------------------------------------------------------
    def _lookup(
        self, messages: Sequence[Message], response_schema: Mapping[str, Any] | None
    ) -> RecordedExchange:
        digest = request_digest(messages, response_schema)
        exchange = self._by_digest.get(digest)
        if exchange is not None:
            # An identical request may be replayed as often as the caller likes.
            self._used.add(digest)
            return exchange
        # Positional fallback is opt-in (hand-written fixtures only): by default a request
        # that does not match a recorded digest is a *stale fixture*, and replaying a
        # neighbouring answer for it would hide a prompt change.
        if self.allow_positional_fallback:
            while self._cursor < len(self._order):
                candidate = self._order[self._cursor]
                self._cursor += 1
                if candidate.request_digest not in self._used:
                    self._used.add(candidate.request_digest)
                    return candidate
        raise RecordedFixtureMiss(digest, request_preview(messages))


def _finish_reason(value: str) -> Any:
    return value if value in {"stop", "length", "content_filter", "error"} else "stop"


class RecordingProvider:
    """Wraps a live provider and captures its answers into a fixture (authoring only)."""

    name = "recording"

    def __init__(self, inner: ChatProvider) -> None:
        self.inner = inner
        self.config = inner.config
        self.exchanges: list[RecordedExchange] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        completion = self.inner.complete(messages, response_schema=response_schema)
        self.exchanges.append(
            RecordedExchange(
                request_digest=request_digest(messages, response_schema),
                request_preview=request_preview(messages),
                schema_digest=schema_digest(response_schema),
                response=RecordedResponse(
                    text=completion.text,
                    model=completion.model,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    finish_reason=completion.finish_reason,
                ),
            )
        )
        return completion

    def stream(self, messages: Sequence[Message]) -> ChunkStream:
        return ChunkStream(iter(self.inner.stream(messages)))

    def to_fixture(self, *, model: str | None = None) -> RecordedFixture:
        return RecordedFixture(
            model=model or (self.inner.config.model),
            exchanges=list(self.exchanges),
        )

    def save(self, path: str | Path, *, model: str | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fixture = self.to_fixture(model=model)
        target.write_text(
            json.dumps(fixture.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target


__all__ = [
    "RECORDED_FIXTURE_VERSION",
    "RECORDED_MODEL",
    "RECORDED_PRESET",
    "RecordedExchange",
    "RecordedFixture",
    "RecordedFixtureMiss",
    "RecordedProvider",
    "RecordedResponse",
    "RecordingProvider",
    "canonical_messages",
    "default_recorded_config",
    "request_digest",
    "request_preview",
    "schema_digest",
]
