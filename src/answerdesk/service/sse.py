"""Server-Sent Events: framing, the fixed event order and token chunking.

FR-SVC-01 fixes the order — ``retrieval`` → ``tokens``* → ``citations`` → ``decision`` →
``done`` — and requires ``trace_id`` on every event. This module owns the *framing*
(``event:``/``data:``/blank line) and the chunking of the verified answer into token-sized
deltas; the route owns the sequencing.

Why tokens arrive *after* generation rather than during it (documented in ``docs/api.md``
§5.1): the answer is a structured object whose claims must be verified against the packed
evidence (CR-1, INV-03). Streaming raw model tokens would display unverified text, which
the citation contract forbids. The service therefore emits the verified text progressively
— the client renders exactly as it would for a live stream — while the model call itself
stays a single structured request. A future provider-level streaming path can reuse this
framing unchanged.

Framing details that matter in practice:

* ``data:`` is a single line of compact JSON (``ensure_ascii=False`` so Cyrillic/CJK stay
  readable; a JSON encoder never emits a raw newline);
* each frame ends with exactly one blank line;
* no ``id:`` field: resumption would replay model calls, which the idempotency store
  already covers for the non-streaming path;
* comment frames (``: ping``) are available for long-lived connections behind proxies.

    # AT-SVC-01  # AT-UI-08
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

#: The event names in their contractual order.
EVENT_ORDER: tuple[str, ...] = ("retrieval", "tokens", "citations", "decision", "done")

#: Content type and the headers a proxy must not buffer.
SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: A whitespace-preserving splitter; tokens keep their trailing separator.
_TOKEN_RE = re.compile(r"\S+\s*")

#: Unicode line/paragraph separators (and NEL) are *not* escaped by ``json.dumps`` and are
#: treated as line breaks by some SSE clients (Python's ``str.splitlines`` is one). A raw one
#: inside ``data:`` splits the frame and the client fails to parse the JSON — fuzz finding
#: F-07, docs/robustness.md. Escaping them keeps every frame on exactly one physical line.
_UNICODE_LINE_BREAKS = {
    0x0085: "\\u0085",  # NEXT LINE
    0x2028: "\\u2028",  # LINE SEPARATOR
    0x2029: "\\u2029",  # PARAGRAPH SEPARATOR
}


def _escape_line_breaks(payload: str) -> str:
    return payload.translate(_UNICODE_LINE_BREAKS)


def sse_event(event: str, data: Mapping[str, Any]) -> str:
    """Format one SSE frame."""
    if not event or "\n" in event:
        raise ValueError("SSE event names must be a single non-empty line")
    payload = json.dumps(dict(data), ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {_escape_line_breaks(payload)}\n\n"


def sse_comment(text: str = "ping") -> str:
    """A comment frame: keeps the connection alive without inventing an event."""
    return f": {text}\n\n"


def token_chunks(text: str, size: int = 24) -> list[str]:
    """Split text into deltas of roughly ``size`` characters on word boundaries.

    The concatenation of the returned chunks is always exactly ``text`` — a test asserts
    it for every documented locale, because a lost space is a rendering bug.
    """
    if not text:
        return []
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    tokens = _TOKEN_RE.findall(text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for token in tokens:
        if current and length + len(token) > size:
            chunks.append("".join(current))
            current = []
            length = 0
        current.append(token)
        length += len(token)
    if current:
        chunks.append("".join(current))
    return chunks


def iter_events(frames: Iterable[str]) -> Iterable[str]:
    """Identity helper that documents the route's generator contract."""
    yield from frames


__all__ = [
    "EVENT_ORDER",
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "iter_events",
    "sse_comment",
    "sse_event",
    "token_chunks",
]
