"""SQLite persistence for sessions, turns, feedback, tickets and idempotency keys.

The service is stateless unless it remembers the conversation, so this module is the
product's memory. Four tables carry it (schema in :data:`SCHEMA`):

``sessions``
    one row per conversation, with the locale hint and free-form metadata;
``turns``
    one row per customer turn: the redacted message, the answered text, the trace id, the
    decision (action + reason + detail) and the idempotency key that produced it. The
    ``(session_id, seq)`` unique constraint makes turn order explicit and race-free;
``feedback`` / ``escalation_tickets``
    the operator actions of FR-SVC-05, always linked by ``trace_id``;
``idempotency``
    the replay store: ``(key, endpoint) → request_hash, status, response_json``. Replaying a
    key returns the stored response **without a second model call** (FR-SVC-02), and using
    the same key with a different body is a ``409`` rather than a silent overwrite.

Design choices that are deliberate:

* **PII is redacted before it is stored.** The message and the answer text go through
  :func:`answerdesk.telemetry.redaction.redact_str`. A support DB is exactly where PII
  would otherwise accumulate forever (INV-05).
* **SQLite, not a server.** A reference stack must run from a clean checkout with no
  service to install; WAL mode plus a process-local lock is enough for the documented
  single-process deployment (multiple workers are discussed in ``docs/ops.md``).
* **Synchronous API.** Calls are microseconds; the routes call this directly rather than
  paying for a thread hop per turn. The lock keeps a future async caller honest.
* **Every read is bounded.** ``history`` returns the last N turns; nothing here can load
  an unbounded conversation into memory.

    # AT-SVC-02  # AT-SVC-05
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from answerdesk.telemetry.redaction import redact_str

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    locale TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'customer',
    message TEXT NOT NULL,
    answer_text TEXT NOT NULL DEFAULT '',
    trace_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS turns_trace ON turns(trace_id);
CREATE UNIQUE INDEX IF NOT EXISTS turns_idempotency
    ON turns(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    session_id TEXT,
    rating TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_trace ON feedback(trace_id);
CREATE TABLE IF NOT EXISTS escalation_tickets (
    ticket_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    session_id TEXT,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    reason TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tickets_trace ON escalation_tickets(trace_id);
CREATE INDEX IF NOT EXISTS tickets_status ON escalation_tickets(status);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status INTEGER NOT NULL,
    response TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (key, endpoint)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    locale: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TurnRow:
    turn_id: str
    session_id: str
    seq: int
    message: str
    answer_text: str
    trace_id: str
    action: str
    reason_code: str
    detail: str
    created_at: str


@dataclass(frozen=True)
class FeedbackRow:
    feedback_id: str
    trace_id: str
    session_id: str | None
    rating: str
    comment: str
    created_at: str


@dataclass(frozen=True)
class TicketRow:
    ticket_id: str
    trace_id: str
    session_id: str | None
    status: str
    priority: str
    reason: str
    note: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredResponse:
    """A replayable response previously stored under an idempotency key."""

    key: str
    endpoint: str
    request_hash: str
    status: int
    payload: dict[str, Any]
    trace_id: str
    created_at: str


class SQLiteSessionStore:
    """The conversation store (see the module docstring for the contract)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteSessionStore:  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: object) -> None:  # pragma: no cover - convenience
        self.close()

    # -- sessions ------------------------------------------------------------------
    def create_session(
        self,
        *,
        session_id: str | None = None,
        locale: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionRecord:
        resolved = session_id or _new_id("s")
        created = _now()
        payload = json.dumps(dict(metadata or {}), sort_keys=True, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, locale, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (resolved, created, created, locale, payload),
            )
        return SessionRecord(resolved, created, created, locale, dict(metadata or {}))

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return _session(row) if row is not None else None

    def ensure_session(
        self, session_id: str | None, *, locale: str | None = None
    ) -> tuple[SessionRecord, bool]:
        """Return ``(session, created)``; a supplied id that does not exist is created."""
        if session_id:
            existing = self.get_session(session_id)
            if existing is not None:
                return existing, False
            return self.create_session(session_id=session_id, locale=locale), True
        return self.create_session(locale=locale), True

    # -- turns ---------------------------------------------------------------------
    def append_turn(
        self,
        *,
        session_id: str,
        message: str,
        trace_id: str,
        action: str,
        reason_code: str,
        answer_text: str = "",
        detail: str = "",
        idempotency_key: str | None = None,
    ) -> TurnRow:
        """Append one turn, assigning the next sequence number atomically."""
        clean_message = redact_str(message)
        clean_answer = redact_str(answer_text)
        clean_detail = redact_str(detail)
        created = _now()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS last FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row["last"]) + 1
            turn_id = _new_id("t")
            self._conn.execute(
                "INSERT INTO turns(turn_id, session_id, seq, message, answer_text, trace_id, "
                "action, reason_code, detail, idempotency_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn_id,
                    session_id,
                    seq,
                    clean_message,
                    clean_answer,
                    trace_id,
                    action,
                    reason_code,
                    clean_detail,
                    idempotency_key,
                    created,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (created, session_id)
            )
        return TurnRow(
            turn_id=turn_id,
            session_id=session_id,
            seq=seq,
            message=clean_message,
            answer_text=clean_answer,
            trace_id=trace_id,
            action=action,
            reason_code=reason_code,
            detail=clean_detail,
            created_at=created,
        )

    def turns(self, session_id: str, *, limit: int | None = None) -> list[TurnRow]:
        query = "SELECT * FROM turns WHERE session_id = ? ORDER BY seq ASC"
        params: list[Any] = [session_id]
        if limit is not None:
            query = (
                "SELECT * FROM (SELECT * FROM turns WHERE session_id = ? "
                "ORDER BY seq DESC LIMIT ?) ORDER BY seq ASC"
            )
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_turn(row) for row in rows]

    def history(self, session_id: str | None, *, limit: int = 6) -> list[TurnRow]:
        """The last ``limit`` turns, oldest first — the pipeline's conversation replay."""
        if not session_id:
            return []
        return self.turns(session_id, limit=limit)

    def turn_by_trace(self, trace_id: str) -> TurnRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM turns WHERE trace_id = ? ORDER BY created_at DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
        return _turn(row) if row is not None else None

    def get_turn(self, turn_id: str) -> TurnRow | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        return _turn(row) if row is not None else None

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row["n"])

    # -- feedback ------------------------------------------------------------------
    def add_feedback(
        self,
        *,
        trace_id: str,
        rating: str,
        comment: str | None = None,
        session_id: str | None = None,
    ) -> FeedbackRow:
        created = _now()
        feedback_id = _new_id("fb")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO feedback(feedback_id, trace_id, session_id, rating, comment, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (feedback_id, trace_id, session_id, rating, redact_str(comment or ""), created),
            )
        return FeedbackRow(feedback_id, trace_id, session_id, rating, redact_str(comment or ""), created)

    def feedback_for(self, trace_id: str) -> list[FeedbackRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM feedback WHERE trace_id = ? ORDER BY created_at ASC", (trace_id,)
            ).fetchall()
        return [
            FeedbackRow(
                row["feedback_id"],
                row["trace_id"],
                row["session_id"],
                row["rating"],
                row["comment"],
                row["created_at"],
            )
            for row in rows
        ]

    # -- escalation tickets --------------------------------------------------------
    def create_ticket(
        self,
        *,
        trace_id: str,
        reason: str,
        priority: str = "normal",
        note: str | None = None,
        session_id: str | None = None,
        status: str = "open",
    ) -> TicketRow:
        created = _now()
        ticket_id = _new_id("tk")
        clean_note = redact_str(note or "")
        clean_reason = redact_str(reason)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO escalation_tickets(ticket_id, trace_id, session_id, status, "
                "priority, reason, note, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket_id,
                    trace_id,
                    session_id,
                    status,
                    priority,
                    clean_reason,
                    clean_note,
                    created,
                    created,
                ),
            )
        return TicketRow(
            ticket_id,
            trace_id,
            session_id,
            status,
            priority,
            clean_reason,
            clean_note,
            created,
            created,
        )

    def get_ticket(self, ticket_id: str) -> TicketRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return _ticket(row) if row is not None else None

    def list_tickets(self, *, status: str | None = None, limit: int = 50) -> list[TicketRow]:
        query = "SELECT * FROM escalation_tickets"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_ticket(row) for row in rows]

    def update_ticket_status(self, ticket_id: str, status: str) -> TicketRow | None:
        updated = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE escalation_tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
                (status, updated, ticket_id),
            )
        return self.get_ticket(ticket_id)

    def ticket_for_trace(self, trace_id: str) -> TicketRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM escalation_tickets WHERE trace_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
        return _ticket(row) if row is not None else None

    # -- idempotency ---------------------------------------------------------------
    def save_idempotent(
        self,
        *,
        key: str,
        endpoint: str,
        request_hash: str,
        status: int,
        payload: Mapping[str, Any],
        trace_id: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO idempotency(key, endpoint, request_hash, status, response, "
                "trace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    endpoint,
                    request_hash,
                    int(status),
                    json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                    trace_id,
                    _now(),
                ),
            )

    def lookup_idempotent(
        self, key: str, endpoint: str, *, ttl_s: int | None = None
    ) -> StoredResponse | None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM idempotency WHERE key = ? AND endpoint = ?", (key, endpoint)
            ).fetchone()
            if row is None:
                return None
            if ttl_s is not None and ttl_s >= 0:
                created = datetime.fromisoformat(row["created_at"])
                age = (datetime.now(UTC) - created).total_seconds()
                if age > ttl_s:
                    self._conn.execute(
                        "DELETE FROM idempotency WHERE key = ? AND endpoint = ?", (key, endpoint)
                    )
                    return None
        return StoredResponse(
            key=row["key"],
            endpoint=row["endpoint"],
            request_hash=row["request_hash"],
            status=int(row["status"]),
            payload=json.loads(row["response"]),
            trace_id=row["trace_id"],
            created_at=row["created_at"],
        )

    def purge_idempotent(self, *, ttl_s: int) -> int:
        cutoff = datetime.now(UTC).timestamp() - ttl_s
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT key, endpoint, created_at FROM idempotency").fetchall()
            stale = [
                (row["key"], row["endpoint"])
                for row in rows
                if datetime.fromisoformat(row["created_at"]).timestamp() < cutoff
            ]
            for key, endpoint in stale:
                self._conn.execute(
                    "DELETE FROM idempotency WHERE key = ? AND endpoint = ?", (key, endpoint)
                )
        return len(stale)

    # -- diagnostics ---------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        with self._lock:
            counts = {}
            for table in ("sessions", "turns", "feedback", "escalation_tickets", "idempotency"):
                row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                counts[table] = int(row["n"])
        counts["schema_version"] = SCHEMA_VERSION
        return counts


def _session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        locale=row["locale"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _turn(row: sqlite3.Row) -> TurnRow:
    return TurnRow(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        seq=int(row["seq"]),
        message=row["message"],
        answer_text=row["answer_text"],
        trace_id=row["trace_id"],
        action=row["action"],
        reason_code=row["reason_code"],
        detail=row["detail"],
        created_at=row["created_at"],
    )


def _ticket(row: sqlite3.Row) -> TicketRow:
    return TicketRow(
        ticket_id=row["ticket_id"],
        trace_id=row["trace_id"],
        session_id=row["session_id"],
        status=row["status"],
        priority=row["priority"],
        reason=row["reason"],
        note=row["note"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "FeedbackRow",
    "SQLiteSessionStore",
    "SessionRecord",
    "StoredResponse",
    "TicketRow",
    "TurnRow",
]
