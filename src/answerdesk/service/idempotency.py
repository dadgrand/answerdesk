"""Idempotency keys: replay a stored response instead of paying for a second turn.

FR-SVC-02: "requests accept an idempotency key and replaying a key returns the stored
response without a second model call". The rule is small but easy to get wrong, so it is
isolated here:

* the key is scoped to an endpoint (``chat``, ``feedback``, ``escalations``); a client may
  reuse the same key string for different endpoints without collision;
* the *request fingerprint* is a canonical hash of the meaningful body fields. Replaying a
  key with a different body is a ``409 idempotency_conflict`` — silently returning the old
  answer to a different question is exactly the failure the header exists to prevent;
* the stored response is the exact JSON the first call returned (including its
  ``trace_id``), so a retried request is indistinguishable from the original;
* the replay window is bounded (``ANSWERDESK_SERVICE_IDEMPOTENCY_TTL_S``), after which the
  key is forgotten and a new turn is allowed.

    # AT-SVC-02
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from answerdesk.service.errors import IdempotencyConflict
from answerdesk.service.sessions import SQLiteSessionStore, StoredResponse


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    """Canonical hash of a request body (sorted keys, no whitespace drift)."""
    material = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def replay(
    store: SQLiteSessionStore,
    *,
    key: str | None,
    endpoint: str,
    fingerprint: str,
    ttl_s: int,
) -> StoredResponse | None:
    """Return the stored response for ``key``, or ``None``; conflict on a different body."""
    if not key:
        return None
    stored = store.lookup_idempotent(key, endpoint, ttl_s=ttl_s)
    if stored is None:
        return None
    if stored.request_hash != fingerprint:
        raise IdempotencyConflict(key)
    return stored


def remember(
    store: SQLiteSessionStore,
    *,
    key: str | None,
    endpoint: str,
    fingerprint: str,
    payload: Mapping[str, Any],
    trace_id: str,
    status: int = 200,
) -> None:
    if not key:
        return
    store.save_idempotent(
        key=key,
        endpoint=endpoint,
        request_hash=fingerprint,
        status=status,
        payload=payload,
        trace_id=trace_id,
    )


__all__ = ["remember", "replay", "request_fingerprint"]
