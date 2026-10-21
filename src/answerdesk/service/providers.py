"""The provider seam the service owns: PII redaction on every outgoing message (INV-05).

``AnswerPipeline`` builds its own messages and calls ``provider.complete(...)``. The
pipeline is t6's and does not redact — it receives the (already redacted) question, but the
*packed evidence* is corpus text and a repair prompt repeats the previous draft, which can
contain a model paraphrase of a customer number. The service therefore wraps whatever
provider the factory produced in :class:`RedactingProvider`:

    build_provider(settings) -> RedactingProvider -> AnswerPipeline

The wrapper is a transparent ``ChatProvider``: same ``name``, same ``config``, same
``complete``/``stream`` behaviour, with :func:`answerdesk.telemetry.redaction.redact_messages`
applied first. Findings are recorded in the trace (``pii.provider_redactions``) but never
the values.

This is deliberately a *decorator*, not a fork of the provider: the single-HTTP-client rule
of INV-07 stays intact, and a test asserts the wrapper forwards the schema and the stream
unchanged.

    # AT-SEC-02  # AT-GEN-11  # AT-SVC-08
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from answerdesk.contracts import (
    ChatProvider,
    Completion,
    CompletionChunk,
    Message,
    ProviderConfig,
)
from answerdesk.telemetry.redaction import redact_messages


class RedactingProvider:
    """A ``ChatProvider`` that removes PII before the request leaves the process."""

    def __init__(self, inner: ChatProvider) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.config: ProviderConfig = inner.config
        self.redactions: int = 0

    # -- ChatProvider protocol ----------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        clean, findings = redact_messages(messages)
        if findings:
            self.redactions += len(findings)
        return self.inner.complete(clean, response_schema=response_schema)

    def stream(self, messages: Sequence[Message]) -> Sequence[CompletionChunk]:
        clean, findings = redact_messages(messages)
        if findings:
            self.redactions += len(findings)
        return self.inner.stream(clean)

    # -- introspection ------------------------------------------------------------
    def reset_counters(self) -> None:
        self.redactions = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"RedactingProvider(inner={self.name!r}, redactions={self.redactions})"


def wrap_provider(provider: ChatProvider) -> RedactingProvider:
    """Idempotent wrapping: a provider is redacted once, no matter how often it is passed."""
    if isinstance(provider, RedactingProvider):
        return provider
    return RedactingProvider(provider)


__all__ = ["RedactingProvider", "wrap_provider"]
