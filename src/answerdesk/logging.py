"""Structured logging with secret *and value* redaction.

Deliberately dependency-free at import time (stdlib only): the base install must stay light
(ADR-009) and the JSON shape is what t9's trace pipeline and any log shipper expect. The
value redactor is imported lazily on the first formatted record — ``answerdesk.telemetry``
imports this module through ``tracing``, so a top-level import would be circular.

Since t36 (L-10 closed) redaction is applied to **values**, not only to keys that look like
secrets: the formatted message, the exception traceback and every extra string pass through
``telemetry.redaction.redact_str`` (the same detector the traces use: e-mails, phones, cards,
IBAN-like and account-like runs, plus credential shapes such as ``sk-…``, ``Bearer …``,
``AKIA…`` and PEM blocks). Log keys/fields are unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.IGNORECASE)
_REDACTED = "***"

#: A whole PEM private-key block. The redactor replaces the header line, which would leave
#: the base64 body behind; a log line must lose the block, so this runs first.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)",
    re.DOTALL,
)

#: LogRecord attributes that are structural, not user supplied — never emitted as extras.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

#: Cached ``redact_str`` (lazy import; see the module docstring).
_REDACT_STR: Callable[[str], str] | None = None


def _redact_text(text: str) -> str:
    """Redact PII and credentials in free text (message, traceback, extra values)."""
    global _REDACT_STR
    if _REDACT_STR is None:  # pragma: no branch - set on the first formatted record
        from answerdesk.telemetry.redaction import redact_str

        _REDACT_STR = redact_str
    without_pem = _PEM_BLOCK.sub("[SECRET]", text)
    return _REDACT_STR(without_pem)


def redact(value: Any) -> Any:
    """Recursively redact a log payload: secret-named keys **and** free-text values (INV-05).

    Redaction is idempotent — the placeholders (``[EMAIL]``, ``[SECRET]``, ``***``) do not
    match the detector again — so applying it at the field level and again to the payload is
    safe and cannot produce ``[[EMAIL]]`` artefacts.
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _REDACTED if _SECRET_KEY.search(str(key)) else redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, extras.

    Structural fields (``ts``/``level``/``logger``) are never redacted: they are ours, and a
    digit-run detector would otherwise turn an ISO timestamp into ``[PHONE]``. Only the
    message, the traceback and the extras pass through the value redactor.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exc"] = _redact_text(self.formatException(record.exc_info))
        for key, item in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = _REDACTED if _SECRET_KEY.search(key) else redact(item)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable fallback for local terminals (redacted like the JSON formatter)."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        # A shallow copy keeps the shared LogRecord untouched; only its message is redacted,
        # so the human-readable timestamp survives.
        clone = copy.copy(record)
        clone.msg = _redact_text(record.getMessage())
        clone.args = None
        return super().format(clone)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger once; idempotent and safe to call from the CLI."""
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # httpx logs full request URLs at INFO; keep provider chatter at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "TextFormatter", "configure_logging", "get_logger", "redact"]
