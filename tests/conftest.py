"""Shared fixtures for every answerdesk test suite.

The autouse ``_no_network`` guard is the enforcement point for INV-08 / NFR-DET-01: any
test that tries to open a socket fails with a message that names the requirement. That is
how "CI never needs network" stays true instead of merely being asserted in a README.

    # AT-DET-01
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from answerdesk.config import Settings
from answerdesk.contracts import (
    Action,
    Answer,
    CacheInfo,
    Chunk,
    Citation,
    Claim,
    Confidence,
    Decision,
    Document,
    Heading,
    Provenance,
    ReasonCode,
    Usage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ambient ``ANSWERDESK_*`` variables so tests do not depend on the shell.

    CI exports ``ANSWERDESK_PROFILE=offline`` for the gate; a test that loads
    ``configs/profiles/api.yaml`` must still see that file's profile.
    """
    for name in list(os.environ):
        if name.startswith("ANSWERDESK_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _no_network() -> Iterator[None]:
    """Fail any outbound connection attempt made by a test (INV-08, NFR-DET-01).

    Set ``ANSWERDESK_ALLOW_NETWORK=1`` only for a deliberate manual live-provider run.
    """
    if os.environ.get("ANSWERDESK_ALLOW_NETWORK") == "1":  # pragma: no cover - manual only
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _blocked(self: socket.socket, address: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "outbound network access is forbidden in tests (NFR-DET-01). "
            f"Attempted connection to {address!r}."
        )

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]


@pytest.fixture()
def settings() -> Settings:
    """Offline settings: deterministic doubles, no network, no key."""
    return Settings()


@pytest.fixture()
def make_document() -> Callable[..., Document]:
    def _make(doc_id: str = "doc-1", **overrides: object) -> Document:
        text = str(overrides.pop("text", "Refunds are processed within 14 days."))
        base: dict[str, object] = {
            "doc_id": doc_id,
            "title": "Refund policy",
            "source_path": f"{doc_id}.md",
            "source_format": "md",
            "text": text,
            "license": "CC0-1.0",
            "sha256": "0" * 64,
            "byte_size": len(text.encode("utf-8")),
            "headings": [Heading(level=2, text="Refunds", char_offset=0)],
        }
        base.update(overrides)
        return Document.model_validate(base)

    return _make


@pytest.fixture()
def make_chunk() -> Callable[..., Chunk]:
    def _make(chunk_id: str = "doc-1::0", **overrides: object) -> Chunk:
        text = str(overrides.pop("text", "Refunds are processed within 14 days of approval."))
        base: dict[str, object] = {
            "chunk_id": chunk_id,
            "doc_id": "doc-1",
            "text": text,
            "heading_path": ["Product", "Refunds"],
            "breadcrumb": "Product > Refunds",
            "start_char": 0,
            "end_char": len(text),
            "token_count": len(text.split()),
        }
        base.update(overrides)
        return Chunk.model_validate(base)

    return _make


@pytest.fixture()
def make_answer(make_chunk: Callable[..., Chunk]) -> Callable[..., Answer]:
    def _make(**overrides: object) -> Answer:
        chunk = make_chunk()
        citation = Citation(
            citation_id=1,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title="Refund policy",
            heading_path=list(chunk.heading_path),
            quote="Refunds are processed within 14 days",
            start_char=0,
            end_char=len("Refunds are processed within 14 days"),
            doc_hash=chunk.doc_hash,
        )
        base: dict[str, object] = {
            "answer_id": "answer-1",
            "trace_id": "trace-1",
            "language": "en",
            "text": "Refunds take 14 days [1].",
            "claims": [
                Claim(claim_id="c1", text="Refunds take 14 days.", citation_ids=[1]),
            ],
            "citations": [citation],
            "decision": Decision(
                action=Action.ANSWER,
                reason_code=ReasonCode.ANSWERED,
                score=0.97,
                alpha=0.05,
                threshold_ref="calibration-dev-001",
            ),
            "confidence": Confidence(p_correct=0.97, calibrator_version="platt-1"),
            "cache": CacheInfo(status="miss"),
            "usage": Usage(input_tokens=100, output_tokens=20, usd=0.0001, model="fake-model"),
            "provenance": Provenance(
                corpus_version="corpus-1",
                index_version="index-1",
                prompt_version="answer-v1",
                verifier_version="verifier-1",
                scorer_version="scorer-1",
            ),
        }
        base.update(overrides)
        return Answer.model_validate(base)

    return _make
