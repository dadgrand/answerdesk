"""L-10 (audit t35, closed in t36): log formatting redacts values, not only keys.

``answerdesk.logging.JsonFormatter`` used to mask secret-*named* mapping keys and copy every
free-text value — including an exception traceback — verbatim. Since t36 the message, the
traceback and every extra string pass through ``telemetry.redaction.redact_str`` (the same
detector the traces and the provider boundary use), so an upstream error that quotes a
customer e-mail, card or key cannot reach the log stream.

The log *schema* is unchanged: ``ts``/``level``/``logger``/``message`` plus extras, and
``exc`` only when the record carries exception info. Structural fields are never redacted —
a digit-run detector would otherwise turn the ISO timestamp into ``[PHONE]``.

    # AT-OPS-01  # AT-OBS-01  # INV-05
"""

from __future__ import annotations

import json
import logging
import re
import sys

import pytest

from answerdesk.logging import JsonFormatter, TextFormatter, redact

PII_EMAIL = "aisha.rahimova@example.com"
PII_PHONE = "+7 900 000-00-01"
PII_CARD = "4111 1111 1111 1111"
SECRET = "sk-live-9f3a2b7c4d5e6f708192a3b4c5d6e7f8"
PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAdeadbeefsecretbody\n"
    "-----END RSA PRIVATE KEY-----"
)
RAW_VALUES = (PII_EMAIL, PII_PHONE, PII_CARD, PII_CARD.replace(" ", ""), SECRET)
ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00$")


def _record(message: str, *, exc_info: tuple | None = None, level: int = logging.ERROR):
    return logging.LogRecord("answerdesk.test", level, __file__, 1, message, None, exc_info)


def _assert_clean(text: str, where: str) -> None:
    for raw in RAW_VALUES:
        assert raw not in text, f"{raw!r} leaked into {where}"


def test_message_values_are_redacted() -> None:
    record = _record(f"customer {PII_EMAIL} phone {PII_PHONE} card {PII_CARD} key {SECRET}")
    line = JsonFormatter().format(record)
    _assert_clean(line, "the formatted message")
    assert "[EMAIL]" in line and "[PHONE]" in line and "[CARD]" in line and "[SECRET]" in line


def test_pem_block_is_removed_entirely() -> None:
    """The header pattern alone would leave the base64 body; the block must go as a whole."""
    line = JsonFormatter().format(_record(f"leaked key: {PEM}"))
    assert "deadbeefsecretbody" not in line
    assert "BEGIN RSA PRIVATE KEY" not in line
    assert "[SECRET]" in line


def test_extra_values_are_redacted_including_nested_ones() -> None:
    record = _record("service error")
    record.diagnostic = f"upstream said {PII_EMAIL}"
    record.context = {"contact": PII_EMAIL, "card": PII_CARD, "list": [SECRET]}
    record.api_key = SECRET  # a secret-named key still becomes ***
    payload = json.loads(JsonFormatter().format(record))
    _assert_clean(json.dumps(payload), "the formatted extras")
    assert payload["api_key"] == "***"
    assert payload["context"]["contact"] == "[EMAIL]"
    assert payload["context"]["list"] == ["[SECRET]"]
    assert payload["diagnostic"] == "upstream said [EMAIL]"


def test_traceback_values_are_redacted() -> None:
    try:
        raise ValueError(f"upstream rejected {PII_EMAIL} / card {PII_CARD} / key {SECRET}")
    except ValueError:
        record = _record("unhandled service error", exc_info=sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "exc" in payload
    _assert_clean(payload["exc"], "the formatted traceback")
    assert "ValueError" in payload["exc"]  # diagnostics survive
    assert "[EMAIL]" in payload["exc"]


def test_log_schema_and_timestamp_are_unchanged() -> None:
    payload = json.loads(JsonFormatter().format(_record(f"mail {PII_EMAIL}")))
    assert set(payload) == {"ts", "level", "logger", "message"}
    assert ISO_TS.match(payload["ts"]), payload["ts"]
    assert payload["level"] == "ERROR" and payload["logger"] == "answerdesk.test"
    # a record without exc_info has no exc field at all
    assert "exc" not in payload


def test_redaction_is_idempotent_and_leaves_no_double_artefacts() -> None:
    once = redact(f"mail {PII_EMAIL} card {PII_CARD} key {SECRET}")
    twice = redact(once)
    assert once == twice, (once, twice)
    for artefact in ("[[", "]]", "[[EMAIL]]", "[[SECRET]]"):
        assert artefact not in once
    assert once.count("[EMAIL]") == 1 and once.count("[SECRET]") == 1
    # formatting the same record twice is byte-identical
    record = _record(f"mail {PII_EMAIL}")
    assert JsonFormatter().format(record) == JsonFormatter().format(record)


def test_text_formatter_redacts_the_message_but_not_the_clock() -> None:
    line = TextFormatter().format(_record(f"mail {PII_EMAIL} key {SECRET}"))
    _assert_clean(line, "the text log line")
    assert "[EMAIL]" in line
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ", line), line


def test_captured_log_records_never_carry_raw_values() -> None:
    """End-to-end over the logging pipeline: handler → formatter → emitted line."""

    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.lines: list[str] = []
            self.setFormatter(JsonFormatter())

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(self.format(record))

    handler = _ListHandler()
    logger = logging.getLogger("answerdesk.test.redaction")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        logger.error(
            "provider failed for %s, card %s, key %s",
            PII_EMAIL,
            PII_CARD,
            SECRET,
            extra={"diagnostic": f"upstream: {PII_PHONE}", "trace_id": "tr-abc"},
        )
        try:
            raise RuntimeError(f"upstream rejected {PII_EMAIL} / {SECRET}")
        except RuntimeError:
            logger.exception("unhandled %s", PII_EMAIL)
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    assert len(handler.lines) == 2
    for line in handler.lines:
        _assert_clean(line, "a captured log line")
        json.loads(line)
    assert "[EMAIL]" in handler.lines[0] and "[PHONE]" in handler.lines[0]
    assert "RuntimeError" in handler.lines[1]
    assert "[EMAIL]" in handler.lines[1]


def test_redact_keeps_non_string_scalars() -> None:
    assert redact(7) == 7
    assert redact(None) is None
    assert redact(True) is True
    assert redact({"count": 3, "ok": False}) == {"count": 3, "ok": False}
    assert redact(["a", 1]) == ["a", 1]


def test_import_is_stdlib_only_until_the_first_record() -> None:
    """The lazy redactor keeps ``answerdesk.logging`` import-light (no telemetry at import)."""
    import importlib

    import answerdesk.logging as logging_module

    reloaded = importlib.reload(logging_module)
    assert reloaded._REDACT_STR is None
    # formatting a clean message still initialises the shared detector
    reloaded.JsonFormatter().format(_record("plain message"))
    assert reloaded._REDACT_STR is not None


@pytest.mark.parametrize(
    "payload",
    [
        "-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----",
        "authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
        "ghp_0123456789abcdefghijklmnopqrstuvwx",
    ],
    ids=["pem", "bearer", "github_token"],
)
def test_credential_shapes_are_redacted(payload: str) -> None:
    line = JsonFormatter().format(_record(f"context: {payload}"))
    assert "MIIB" not in line
    assert "eyJhbGci" not in line
    assert "ghp_0123" not in line
