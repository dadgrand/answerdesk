"""PII and secret redaction for everything that leaves the process (INV-05).

Three boundaries are redacted here and nowhere else:

1. **logs** — :func:`answerdesk.logging.JsonFormatter` already masks secret-shaped *keys*;
   this module masks PII inside *values* (and secret values, which the formatter cannot see);
2. **traces** — every JSONL trace line is passed through :func:`redact_payload` before it
   is written, so a ticket number in a query never lands in a trace file;
3. **the provider request body** — the service wraps its provider in
   :class:`answerdesk.service.runtime.RedactingProvider`, which calls
   :func:`redact_messages` on every outgoing message.

The detector itself is ``answerdesk.policy.pii`` (t8): phones, e-mail, card numbers
(Luhn-checked), IBAN-like and account-like digit runs in the reserved synthetic ranges.
Keeping one detector means the trace, the log line and the provider request agree about
what "PII" means — a second regex set would drift.

Redaction is *replacement*, not hashing: an operator reading a trace must be able to see
that a field was redacted (`<EMAIL>`) without being able to recover it.

    # AT-SVC-08  # AT-SEC-02  # AT-OBS-01
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from answerdesk.contracts import Message
from answerdesk.policy.pii import PIIFinding, RedactionResult, detect, redact

#: Mapping keys whose *values* are secrets and must never be serialised (INV-05, §13.5).
SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|bearer|credential)",
    re.IGNORECASE,
)

#: The placeholder written instead of a secret value — never the length, never a prefix.
SECRET_PLACEHOLDER = "***"

#: Upper bound on redaction work per string; a 10 MB document is not a log line.
MAX_REDACT_CHARS = 100_000


@dataclass(frozen=True)
class Redacted:
    """A string after redaction, with what was removed (counts only, never the values)."""

    text: str
    kinds: tuple[str, ...] = ()
    count: int = 0

    @property
    def changed(self) -> bool:
        return self.count > 0

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


@dataclass(frozen=True)
class RedactionSummary:
    """Aggregate of a recursive redaction pass, for the trace's ``pii`` block."""

    count: int = 0
    kinds: tuple[str, ...] = ()
    fields: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {"redacted": self.count, "kinds": list(self.kinds), "fields": list(self.fields)}


def redact_text(text: str) -> Redacted:
    """Redact PII in a single string (no-op for empty input)."""
    if not text:
        return Redacted(text="")
    body = text[:MAX_REDACT_CHARS]
    result: RedactionResult = redact(body)
    kinds = tuple(sorted(result.kinds))
    count = result.redacted_count
    if len(text) > MAX_REDACT_CHARS:
        body = result.text + text[MAX_REDACT_CHARS:]
    else:
        body = result.text
    return Redacted(text=body, kinds=kinds, count=count)


def redact_str(text: str) -> str:
    """Convenience wrapper when the caller only needs the text."""
    return redact_text(text).text


def find_pii(text: str) -> list[PIIFinding]:
    """Detect without replacing — used by the guard and by tests."""
    if not text:
        return []
    return detect(text[:MAX_REDACT_CHARS])


def contains_pii(text: str) -> bool:
    return bool(find_pii(text))


def redact_messages(
    messages: Sequence[Message],
) -> tuple[list[Message], list[PIIFinding]]:
    """Redact every outgoing provider message (INV-05, the third boundary)."""
    cleaned: list[Message] = []
    findings: list[PIIFinding] = []
    for message in messages:
        findings.extend(detect(message.content))
        result = redact(message.content)
        cleaned.append(message.model_copy(update={"content": result.text}))
    return cleaned, findings


def redact_value(value: Any, *, key: str = "") -> tuple[Any, RedactionSummary]:
    """Recursively redact a JSON-like value; returns ``(clean_value, summary)``."""
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        counts = 0
        kinds: set[str] = set()
        fields: list[str] = []
        for raw_key, item in value.items():
            name = str(raw_key)
            if SECRET_KEY_PATTERN.search(name):
                clean[name] = SECRET_PLACEHOLDER
                continue
            child, summary = redact_value(item, key=name)
            clean[name] = child
            if summary.count:
                counts += summary.count
                kinds.update(summary.kinds)
                fields.extend(summary.fields or (name,))
        return clean, RedactionSummary(
            count=counts, kinds=tuple(sorted(kinds)), fields=tuple(dict.fromkeys(fields))
        )
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        counts = 0
        kinds: set[str] = set()
        for item in value:
            child, summary = redact_value(item, key=key)
            items.append(child)
            counts += summary.count
            kinds.update(summary.kinds)
        return items, RedactionSummary(count=counts, kinds=tuple(sorted(kinds)), fields=(key,) if counts else ())
    if isinstance(value, str):
        redacted = redact_text(value)
        if redacted.changed:
            return redacted.text, RedactionSummary(
                count=redacted.count, kinds=redacted.kinds, fields=(key,) if key else ()
            )
        return value, RedactionSummary()
    return value, RedactionSummary()


def redact_payload(value: Any) -> Any:
    """Redact a JSON-like value without the bookkeeping (the trace/log writer path)."""
    clean, _summary = redact_value(value)
    return clean


__all__ = [
    "MAX_REDACT_CHARS",
    "SECRET_KEY_PATTERN",
    "SECRET_PLACEHOLDER",
    "Redacted",
    "RedactionSummary",
    "contains_pii",
    "find_pii",
    "redact_messages",
    "redact_payload",
    "redact_str",
    "redact_text",
    "redact_value",
]
