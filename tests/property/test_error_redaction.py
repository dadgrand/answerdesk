"""F-5 (audit t32, closed in t35): an error envelope must not carry raw upstream text.

Two leak paths are pinned here:

* a **provider** error whose text quotes a customer e-mail/phone/card or an API key — the
  caller sees a generic message plus an HTTP status hint, while the trace and the log keep
  the full upstream wording, redacted;
* a **validation** error whose offending value is PII or a credential — the caller sees the
  field path and the pydantic error type, never the value.

The strict xfail that recorded F-5 is now a positive, parametrised assertion over the
4xx, 5xx, timeout, rate-limit and connection paths, plus the streaming transport.

    # AT-SEC-02  # AT-OPS-07  # INV-05
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest
from fuzz_support import (
    ANSWERABLE,
    PII_CARD,
    PII_EMAIL,
    PII_PHONE,
    RaisingProvider,
    parse_sse,
    scenario,
)

from answerdesk.llm.errors import ProviderHTTPError, ProviderTimeoutError

pytestmark = pytest.mark.integration

SECRET = "sk-live-9f3a2b7c4d5e6f708192a3b4c5d6e7f8"
RAW_VALUES = (PII_EMAIL, PII_PHONE, PII_CARD, PII_CARD.replace(" ", ""), SECRET)
ERROR_TEXT = (
    f"upstream rejected the request for {PII_EMAIL} / {PII_PHONE} / card {PII_CARD} "
    f"and key {SECRET} (status)"
)

#: Every provider failure path a client can reach, with the status and hint it must expose.
PROVIDER_PATHS = (
    ("http_4xx", ProviderHTTPError(ERROR_TEXT, status_code=400), 502, "HTTP 400", "provider_error"),
    ("http_5xx", ProviderHTTPError(ERROR_TEXT, status_code=500), 502, "HTTP 500", "provider_error"),
    (
        "rate_limit",
        ProviderHTTPError(ERROR_TEXT, status_code=429, retry_after_s=2.0),
        502,
        "HTTP 429",
        "provider_error",
    ),
    ("timeout", ProviderTimeoutError(ERROR_TEXT), 504, None, "provider_timeout"),
    ("connection", httpx.ConnectError(ERROR_TEXT), 502, None, "provider_unreachable"),
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

        response = scenario(app, _scenario, raise_app_exceptions=False)
    finally:
        root.removeHandler(capture)
        root.setLevel(previous)
    return response, capture.records


def _assert_no_raw_values(body: str, where: str) -> None:
    for raw in RAW_VALUES:
        assert raw not in body, f"{raw!r} leaked into {where}"


@pytest.mark.parametrize(
    ("label", "exc", "status", "hint", "code"),
    PROVIDER_PATHS,
    ids=[case[0] for case in PROVIDER_PATHS],
)
def test_provider_error_envelope_is_generic_and_redacted(
    make_app, label: str, exc: BaseException, status: int, hint: str | None, code: str
) -> None:
    """The fix's acceptance criterion: no upstream text, a stable code, an actionable hint."""
    app, _runtime, _provider = make_app(provider=RaisingProvider(exc))
    response, _records = _run_error_turn(app)

    assert response.status_code == status, response.text[:300]
    body = response.json()["error"]
    _assert_no_raw_values(response.text, f"the {label} envelope")
    assert body["code"] == code
    assert body["retryable"] is True
    assert body["trace_id"].startswith("tr-")
    # the caller keeps one actionable upstream fact: the HTTP status
    if hint is None:
        assert "detail" not in body or body["detail"] is None
    else:
        assert body["detail"] == hint
    # a generic message, never the provider's wording
    assert body["message"] in {
        "the upstream provider rejected the call",
        "the upstream call timed out",
        "the upstream provider could not be reached",
    }
    assert "upstream rejected the request" not in body["message"]


def test_provider_diagnostic_reaches_the_trace_and_the_log(make_app) -> None:
    """Containment must not cost observability: the operator keeps the redacted upstream text."""
    app, runtime, _provider = make_app(
        provider=RaisingProvider(ProviderHTTPError(ERROR_TEXT, status_code=400))
    )
    response, records = _run_error_turn(app)
    assert response.status_code == 502

    diagnostics = [getattr(record, "diagnostic", "") for record in records]
    joined = "\n".join(diagnostic for diagnostic in diagnostics if diagnostic)
    assert joined, "the service should log the upstream diagnostic"
    assert "[EMAIL]" in joined and "[SECRET]" in joined
    _assert_no_raw_values(joined, "the diagnostic log record")

    trace_id = response.json()["error"]["trace_id"]
    trace = runtime.tracer.read(trace_id, kind="turn")
    assert trace is not None, "the failed turn must be traced"
    error_text = str(trace.get("error"))
    assert "ProviderHTTPError" in error_text
    assert "[EMAIL]" in error_text and "[SECRET]" in error_text
    _assert_no_raw_values(error_text, "the trace error")

    trace_dir = Path(runtime.service.trace_dir)
    for path in sorted(trace_dir.glob("*.jsonl")):
        _assert_no_raw_values(path.read_text(encoding="utf-8"), path.name)


def test_unhandled_error_logs_a_redacted_diagnostic_and_no_traceback(make_app) -> None:
    """The generic handler must not copy an upstream message into the log stream.

    `JsonFormatter` masks secret-named keys but not free text, so a traceback would carry
    whatever the exception quotes (docs/robustness.md L-10). The service therefore logs the
    already-redacted diagnostic instead of `exc_info`.
    """
    app, _runtime, _provider = make_app(provider=RaisingProvider(httpx.ConnectError(ERROR_TEXT)))

    async def _scenario(client, _runtime):
        return await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})

    capture = _Capture()
    root = logging.getLogger()
    root.addHandler(capture)
    previous = root.level
    root.setLevel(logging.DEBUG)
    try:
        response = scenario(app, _scenario, raise_app_exceptions=False)
    finally:
        root.removeHandler(capture)
        root.setLevel(previous)

    assert response.status_code == 502
    records = [record for record in capture.records if record.name.startswith("answerdesk")]
    assert records, "the failure must leave a service log record"
    for record in records:
        assert record.exc_info is None, "no traceback may be logged for an arbitrary exception"
        _assert_no_raw_values(record.getMessage(), "the log message")
        diagnostic = getattr(record, "diagnostic", "")
        if diagnostic:
            _assert_no_raw_values(diagnostic, "the diagnostic log field")

    from answerdesk.logging import JsonFormatter

    record = logging.LogRecord(
        "answerdesk.service", logging.ERROR, __file__, 1, "service error diagnostic", None, None
    )
    record.diagnostic = "[EMAIL] [SECRET]"
    line = JsonFormatter().format(record)
    _assert_no_raw_values(line, "the formatted diagnostic line")


def test_json_formatter_alone_does_not_redact_tracebacks() -> None:
    """Characterises the gap that forces the rule above: formatter redaction is key-based.

    Recorded as L-10 in docs/robustness.md. If the formatter later runs value-level redaction,
    this test turns red and the service rule can be relaxed.
    """
    import sys

    from answerdesk.logging import JsonFormatter

    try:
        raise httpx.ConnectError(ERROR_TEXT)
    except httpx.ConnectError:
        record = logging.LogRecord(
            "answerdesk.service", logging.ERROR, __file__, 1, "x", None, sys.exc_info()
        )
    line = JsonFormatter().format(record)
    assert PII_EMAIL in line, "the formatter redacts values now; see L-10 for the follow-up"


def test_provider_error_path_redaction_in_traces_and_logs(make_app) -> None:
    """No raw PII/secret from a provider error text reaches the log stream or a trace line."""
    app, runtime, _provider = make_app(
        provider=RaisingProvider(ProviderHTTPError(ERROR_TEXT, status_code=400))
    )
    response, records = _run_error_turn(app)

    assert response.status_code in {502, 400}, response.text[:300]
    assert records, "the failure must leave a log record, otherwise this test proves nothing"
    log_text = "\n".join(record.getMessage() for record in records)
    _assert_no_raw_values(log_text, "the log stream")

    trace_dir = Path(runtime.service.trace_dir)
    for path in sorted(trace_dir.glob("*.jsonl")):
        _assert_no_raw_values(path.read_text(encoding="utf-8"), path.name)


def test_streaming_error_frame_is_also_generic_and_redacted(make_app) -> None:
    """The SSE transport uses the same envelope; a stream must not leak what HTTP would not."""
    app, _runtime, _provider = make_app(
        provider=RaisingProvider(ProviderHTTPError(ERROR_TEXT, status_code=429))
    )

    async def _scenario(client, _runtime):
        return await client.post("/v1/chat", json={"message": ANSWERABLE})

    response = scenario(app, _scenario)
    assert response.status_code == 200
    frames = parse_sse(response.text)
    error = next(frame for frame in frames if frame.event == "error")
    _assert_no_raw_values(response.text, "the SSE stream")
    assert error.data["error"]["code"] == "provider_error"
    assert error.data["error"]["detail"] == "HTTP 429"
    done = frames[-1]
    assert done.event == "done" and done.data["error"]["code"] == "provider_error"


def test_validation_error_envelope_is_redacted(make_app) -> None:
    """A 422 whose invalid value is PII must not echo it back in message or detail."""
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        return await client.post(
            "/v1/chat",
            json={"message": PII_CARD, "session_id": {"card": PII_CARD}, "locale": 123},
        )

    response = scenario(app, _scenario)
    assert response.status_code == 422, response.text[:300]
    body = response.json()["error"]
    _assert_no_raw_values(response.text, "the validation envelope")
    assert body["code"] == "validation_error"
    assert body["retryable"] is False
    # actionable without the value: field paths and pydantic error types only
    assert body["detail"]
    assert "body." in body["detail"]
    known_types = {
        "string_type",
        "model_type",
        "string_too_long",
        "missing",
        "extra_forbidden",
        "int_type",
        "string_unicode",
    }
    reported = {
        fragment.strip()
        for fragment in body["detail"].replace(";", ":").split(":")
        if fragment.strip()
    }
    assert reported & known_types, body["detail"]
    # no free-text provider/pydantic message either
    assert "Input should be" not in body["detail"]


def test_validation_error_for_an_oversized_pii_message_does_not_echo_it(make_app) -> None:
    """The max-length path is the one that historically included the input in its message."""
    app, _runtime, _provider = make_app()
    oversized = f"{PII_CARD} " + "x" * 8100

    async def _scenario(client, _runtime):
        return await client.post("/v1/chat", json={"message": oversized})

    response = scenario(app, _scenario)
    assert response.status_code == 422, response.text[:300]
    _assert_no_raw_values(response.text, "the oversized-message envelope")
    assert response.json()["error"]["detail"].startswith("body.message:")
