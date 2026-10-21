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

#: Key names whose *values* are secrets and must never be serialised (INV-05, §13.5).
#: Matching is by name, not by substring: ``input_tokens`` is an observability number, while
#: ``admin_token`` is a credential. A naive ``token`` substring rule redacts the former and
#: silently destroys the cost trace (NFR-COST-03).
_SECRET_NAMES = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "api_key",
        "apikey",
        "private_key",
        "secret_key",
        "client_secret",
        "access_key",
    }
)

#: Key suffixes that make a key secret-bearing (``admin_token``, ``client_secret``).
_SECRET_SUFFIXES = ("_token", "_secret", "_password", "_passwd", "_credential", "_api_key")

#: The placeholder written instead of a secret value — never the length, never a prefix.
SECRET_PLACEHOLDER = "***"

#: Replacement for characters no UTF-8 sink can encode (unpaired surrogates, NUL).
INVALID_CHAR_REPLACEMENT = "\ufffd"


def sanitise_text(text: str) -> str:
    """Make a string safe for every UTF-8 sink (logs, traces, SQLite, JSON bodies).

    JSON allows unpaired surrogates (``"\\ud800"``) and Python carries them happily, but
    UTF-8 cannot encode them and ``sqlite3`` refuses them. A hostile client could therefore
    turn a stored turn into an ``UnicodeEncodeError`` — a 500 with no taxonomy code. The
    request boundary replaces them; telemetry repeats the replacement defensively because a
    trace write must never fail a turn (INV-05 is about *not leaking*, not about crashing).
    """
    if not text:
        return text
    if "\x00" in text:
        text = text.replace("\x00", INVALID_CHAR_REPLACEMENT)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", "replace").decode("utf-8")
    return text


def is_secret_key(name: str) -> bool:
    """Whether a mapping key names a credential (used by the trace and log writers)."""
    lowered = name.strip().lower().replace("-", "_")
    if lowered in _SECRET_NAMES:
        return True
    if any(lowered.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return True
    return "authorization" in lowered or lowered.endswith("bearer")


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


#: Well-known credential *shapes* found in free text (a customer pasting a key into the
#: chat, a stack trace quoting a header). Key-name matching cannot see these: the value sits
#: inside a query string, not behind a mapping key (fuzz finding F-06, docs/robustness.md).
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|password|passwd|secret|token)\s*[=:]\s*[^\s,;]{8,}"),
)

#: What a redacted credential becomes. One placeholder for every shape: the operator only
#: needs to know that something was removed.
SECRET_VALUE_PLACEHOLDER = "[SECRET]"


def _redact_secret_values(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings; returns the text and how many were replaced."""
    count = 0
    for pattern in _SECRET_VALUE_PATTERNS:
        text, replaced = pattern.subn(SECRET_VALUE_PLACEHOLDER, text)
        count += replaced
    return text, count


def redact_text(text: str) -> Redacted:
    """Redact PII and credential-shaped values in a single string (no-op for empty input)."""
    if not text:
        return Redacted(text="")
    text = sanitise_text(text)
    body = text[:MAX_REDACT_CHARS]
    result: RedactionResult = redact(body)
    kinds = set(result.kinds)
    count = result.redacted_count
    body, secrets = _redact_secret_values(result.text)
    if secrets:
        kinds.add("secret")
        count += secrets
    if len(text) > MAX_REDACT_CHARS:
        tail, tail_secrets = _redact_secret_values(text[MAX_REDACT_CHARS:])
        body += tail
        if tail_secrets:
            kinds.add("secret")
            count += tail_secrets
    return Redacted(text=body, kinds=tuple(sorted(kinds)), count=count)


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
        content = sanitise_text(message.content)
        findings.extend(detect(content))
        result = redact(content)
        text, _secrets = _redact_secret_values(result.text)
        cleaned.append(message.model_copy(update={"content": text}))
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
            if is_secret_key(name):
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
        list_kinds: set[str] = set()
        for item in value:
            child, summary = redact_value(item, key=key)
            items.append(child)
            counts += summary.count
            list_kinds.update(summary.kinds)
        return items, RedactionSummary(
            count=counts, kinds=tuple(sorted(list_kinds)), fields=(key,) if counts else ()
        )
    if isinstance(value, str):
        redacted = redact_text(value)
        if redacted.changed:
            return redacted.text, RedactionSummary(
                count=redacted.count, kinds=redacted.kinds, fields=(key,) if key else ()
            )
        # Even an unchanged value is returned through :func:`sanitise_text`, so a surrogate
        # that survived the request boundary cannot break the JSONL write.
        return redacted.text, RedactionSummary()
    return value, RedactionSummary()


def redact_payload(value: Any) -> Any:
    """Redact a JSON-like value without the bookkeeping (the trace/log writer path)."""
    clean, _summary = redact_value(value)
    return clean


__all__ = [
    "MAX_REDACT_CHARS",
    "SECRET_PLACEHOLDER",
    "Redacted",
    "RedactionSummary",
    "contains_pii",
    "find_pii",
    "is_secret_key",
    "redact_messages",
    "redact_payload",
    "redact_str",
    "redact_text",
    "redact_value",
]
