"""Deterministic test doubles and fixtures shared by every module's tests.

Owning task: **t3** (skeleton). Module tasks add their own doubles inside their
package, but anything reusable across modules belongs here.

Import them from here, never from a concrete module:

    from answerdesk.testing import HashingEmbedder, FakeProvider

See ADR-004 for why CI must run on doubles and why they must be boring.
"""

from __future__ import annotations

from answerdesk.testing.doubles import (
    FAKE_MODEL,
    FakeJudge,
    FakeProvider,
    FakeVerifier,
    HashingEmbedder,
    InMemoryCache,
    InMemoryTraceSink,
    default_fake_provider_config,
    make_passage,
)

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
