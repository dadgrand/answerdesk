"""F-4 (audit t32): error paths — a provider or validation error carrying PII/secrets.

The t24 suite sends hostile *user messages* and checks logs/traces; it never sends an error whose
**text** carries PII or a credential. A provider that echoes the request (or a validation error
that echoes the offending value) is a different leak path, and it is the one an attacker
influences indirectly by making the upstream fail.

Three tests:

1. ``test_provider_error_path_redaction_in_traces_and_logs`` — a provider error whose message
   contains an e-mail, a phone, a card number and a key: no raw value in any log record or in any
   written trace line. The failed turn **is** traced (the trace writer flushes at app shutdown);
   what it records is the outcome, not the upstream text, so this test pins the *containment*
   boundary. Where the text does escape is the HTTP envelope — that is finding F-5 / L-8, written
   below as a strict xfail.
2. ``test_validation_error_envelope_is_redacted`` — a 422 whose offending value is PII does not
   echo it (the service sanitises validation details).
3. ``test_provider_error_envelope_is_a_known_leak`` — **xfail(strict=True)**: the *HTTP error
   envelope* currently echoes the provider message verbatim, PII and secret included (audit
   finding F-5, documented in docs/robustness.md). The test is written as the fix's acceptance
   criterion; when the service redacts the envelope this strict xfail turns red and the marker is
   replaced by a positive assertion.

    # AT-SEC-02  # AT-OPS-07
"""

from __future__ import annotations

import logging

import pytest
from fuzz_support import ANSWERABLE, PII_CARD, PII_EMAIL, PII_PHONE, RaisingProvider, scenario

from answerdesk.llm.errors import ProviderHTTPError

pytestmark = pytest.mark.integration

SECRET = "sk-live-9f3a2b7c4d5e6f708192a3b4c5d6e7f8"
RAW_VALUES = (PII_EMAIL, PII_PHONE, PII_CARD, PII_CARD.replace(" ", ""), SECRET)
ERROR_TEXT = (
    f"upstream rejected the request for {PII_EMAIL} / {PII_PHONE} / card {PII_CARD} "
    f"and key {SECRET} (400)"
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_error_turn(app) -> tuple[object, list[logging.LogRecord]]:
    capture = _Capture()
    root = logging.getLogger()
    root.addHandler(capture)
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:

        async def _scenario(client, _runtime):
            return await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})

        response = scenario(app, _scenario)
    finally:
        root.removeHandler(capture)
        root.setLevel(previous)
    return response, capture.records


def test_provider_error_path_redaction_in_traces_and_logs(make_app) -> None:
    """No raw PII/secret from a provider error text reaches the log stream or a trace line."""
    from pathlib import Path

    app, runtime, _provider = make_app(
        provider=RaisingProvider(ProviderHTTPError(ERROR_TEXT, status_code=400))
    )
    response, records = _run_error_turn(app)

    assert response.status_code in {502, 400}, response.text[:300]
    assert records, "the failure must leave a log record, otherwise this test proves nothing"
    log_text = "\n".join(record.getMessage() for record in records)
    for raw in RAW_VALUES:
        assert raw not in log_text, f"{raw!r} leaked into a log record"

    trace_dir = Path(runtime.service.trace_dir)
    for path in sorted(trace_dir.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        for raw in RAW_VALUES:
            assert raw not in text, f"{raw!r} leaked into {path.name}"

    # Non-vacuity comes from the turn having failed (502) and the capture being non-empty; the
    # assertion above then says the payload is not in it. The service does not log the provider's
    # error text at all, and the failed turn's trace line records the outcome rather than the
    # upstream text — both are containment properties, recorded in docs/robustness.md (L-8).


def test_validation_error_envelope_is_redacted(make_app) -> None:
    """A 422 whose invalid value is PII must not echo it back in message or detail."""
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        return await client.post(
            "/v1/chat", json={"message": PII_CARD, "locale": 123, "stream": False}
        )

    response = scenario(app, _scenario)
    assert response.status_code == 422, response.text[:300]
    body = response.text
    for raw in (PII_CARD, PII_CARD.replace(" ", "")):
        assert raw not in body, body[:400]
    assert "validation_error" in body, body[:300]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-5 (audit t32): the HTTP error envelope echoes the provider message verbatim, so PII "
        "and credentials in that text reach the caller; fixed by value-level redaction of "
        "ErrorEnvelope.message/detail in the service (out of t33 scope)"
    ),
)
def test_provider_error_envelope_is_a_known_leak(make_app) -> None:
    """The fix's acceptance criterion: an error envelope must not carry raw PII/secrets."""
    app, _runtime, _provider = make_app(
        provider=RaisingProvider(ProviderHTTPError(ERROR_TEXT, status_code=400))
    )
    response, _records = _run_error_turn(app)
    body = response.text
    for raw in RAW_VALUES:
        assert raw not in body, f"{raw!r} leaked into the error envelope"
