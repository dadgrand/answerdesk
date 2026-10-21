"""Structured logging with secret redaction.

Deliberately dependency-free (stdlib only): the base install must stay light
(ADR-009) and the JSON shape is what t9's trace pipeline and any log shipper expect.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.IGNORECASE)
_REDACTED = "***"

#: LogRecord attributes that are structural, not user supplied — never emitted as extras.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def redact(value: Any) -> Any:
    """Recursively replace secret-looking mapping values with ``***`` (INV-05)."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = _REDACTED if _SECRET_KEY.search(str(key)) else redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, item in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact(item) if isinstance(item, Mapping) else item
        return json.dumps(redact(payload), ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable fallback for local terminals."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s")


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
