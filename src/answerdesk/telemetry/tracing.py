"""Structured JSONL tracing: one line per request, PII redacted, append-only.

The trace is the product's audit surface. Requirements FR-SVC-08 fixes its content and
NFR-COST-03 adds the per-stage attribution:

======================  =========================================================
field                   provenance
======================  =========================================================
``trace_id``            minted by the service before any work happens
``query``               the customer message, **redacted** (INV-05)
``retrieval``           :class:`answerdesk.contracts.RetrievalTrace` — scores,
                        channels, fusion info, packed order (FR-UI-04 reads this)
``timings_ms``          one entry per stage: analysis, retrieval, rerank,
                        generation, verify, cache, total (NFR-COST-03)
``decision``            action, reason code, ``p̂``, ``alpha``, ``threshold_ref``
``usage``               input/output tokens, USD, model, prompt version
``cache_status``        hit / miss / bypassed, with the guard reason
``corpus_version``      which corpus produced the evidence
``index_version``       which index produced the ranking
======================  =========================================================

A trace line is a *document*, not a frozen pydantic contract: :class:`TraceRecord` is the
validated core and :func:`trace_line` adds the service-owned extension block
(``schema_version``, ``service``, ``pii``, ``error``, ``escalation``). That keeps
``contracts.py`` frozen while still letting the console render everything an operator
needs. The extension is versioned and documented in ``docs/api.md``.

Writers are synchronous and append-only. A test can pass an :class:`InMemoryTraceWriter`;
production uses :class:`JsonlTraceWriter`, which rotates by size and never overwrites a
previous line.

    # AT-SVC-08  # AT-OBS-01  # AT-OBS-02  # AT-COST-03  # AT-LAT-01
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from answerdesk.contracts import TraceRecord
from answerdesk.logging import get_logger
from answerdesk.telemetry.redaction import (
    RedactionSummary,
    redact_payload,
    redact_value,
    sanitise_text,
)

_LOGGER = get_logger(__name__)

#: Version of the trace *document* (contracts' ``SCHEMA_VERSION`` covers the models).
TRACE_SCHEMA_VERSION = "1.1.0"

#: The file name inside the trace directory.
TRACE_FILE_NAME = "traces.jsonl"


@runtime_checkable
class TraceWriter(Protocol):
    """The sink seam: anything that can persist a trace document."""

    def write(self, line: Mapping[str, Any]) -> None: ...

    def read(self, trace_id: str, *, kind: str | None = None) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


def trace_line(
    record: TraceRecord,
    *,
    extra: Mapping[str, Any] | None = None,
    redact: bool = True,
) -> tuple[dict[str, Any], RedactionSummary]:
    """Turn a validated :class:`TraceRecord` into the persisted trace document.

    Returns ``(line, summary)`` so the caller can record *that* redaction happened
    (the trace never contains the redacted values, only counts and kinds).
    """
    payload = record.model_dump(mode="json")
    extension: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "p_hat": record.confidence,
        "reason_code": record.decision.reason_code.value if record.decision else None,
        "action": record.decision.action.value if record.decision else None,
        "stage_latency_ms": dict(record.timings_ms),
    }
    if extra:
        extension.update(extra)
    payload.update(extension)
    if not redact:
        return payload, RedactionSummary()
    clean, summary = redact_value(payload)
    assert isinstance(clean, dict)
    clean["pii"] = summary.as_dict()
    return clean, summary


class InMemoryTraceWriter:
    """A trace sink for tests and for ``--fake`` runs that must not touch disk."""

    def __init__(self, *, redact: bool = True) -> None:
        self.lines: list[dict[str, Any]] = []
        self.redact = redact
        self._lock = threading.Lock()

    def write(self, line: Mapping[str, Any]) -> None:
        payload = redact_payload(dict(line)) if self.redact else dict(line)
        with self._lock:
            self.lines.append(payload)

    def read(self, trace_id: str, *, kind: str | None = None) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        with self._lock:
            for line in reversed(self.lines):
                if line.get("trace_id") != trace_id:
                    continue
                if kind is None or line.get("kind") == kind:
                    return line
                fallback = fallback or line
        return fallback

    def close(self) -> None:  # pragma: no cover - nothing to flush
        return None

    # -- test conveniences --------------------------------------------------------
    def messages(self) -> Iterable[dict[str, Any]]:
        return list(self.lines)

    def __len__(self) -> int:
        return len(self.lines)


class JsonlTraceWriter:
    """Append-only JSONL file writer with size-based rotation (``traces.1.jsonl`` …)."""

    def __init__(
        self,
        directory: str | Path,
        *,
        filename: str = TRACE_FILE_NAME,
        max_bytes: int = 32 * 1024 * 1024,
        backups: int = 3,
        redact: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename
        self.max_bytes = max(0, max_bytes)
        self.backups = max(0, backups)
        self.redact = redact
        self._lock = threading.Lock()
        self._handle: Any = None

    # -- protocol ------------------------------------------------------------------
    def write(self, line: Mapping[str, Any]) -> None:
        payload = redact_payload(dict(line)) if self.redact else dict(line)
        serialised = sanitise_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        encoded = serialised.encode("utf-8")
        with self._lock:
            self._rotate_if_needed(len(encoded) + 1)
            handle = self._open()
            handle.write(serialised + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self, trace_id: str, *, kind: str | None = None) -> dict[str, Any] | None:
        """Scan newest-first for a trace id, preferring ``kind`` when one is given."""
        fallback: dict[str, Any] | None = None
        for path in self._files_newest_first():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:  # pragma: no cover - racing rotation
                continue
            for raw in reversed(lines):
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:  # pragma: no cover - corrupt line
                    continue
                if not isinstance(payload, dict) or payload.get("trace_id") != trace_id:
                    continue
                if kind is None or payload.get("kind") == kind:
                    return payload
                fallback = fallback or payload
        return fallback

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    # -- internals -----------------------------------------------------------------
    def _open(self) -> Any:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self._handle

    def _rotate_if_needed(self, incoming: int) -> None:
        if self.max_bytes <= 0 or self._handle is None:
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size + incoming <= self.max_bytes:
            return
        self._handle.close()
        self._handle = None
        for index in range(self.backups, 0, -1):
            source = self.path.with_name(f"traces.{index}.jsonl")
            target = self.path.with_name(f"traces.{index + 1}.jsonl")
            if index == self.backups and source.exists():
                source.unlink()
                continue
            if source.exists():
                source.replace(target)

    def _files_newest_first(self) -> list[Path]:
        files = [self.path]
        for index in range(1, self.backups + 1):
            candidate = self.path.with_name(f"traces.{index}.jsonl")
            if candidate.is_file():
                files.append(candidate)
        return files


@dataclass
class TraceStats:
    """Counters the writer exposes for /healthz and /metrics."""

    written: int = 0
    errors: int = 0


__all__ = [
    "TRACE_FILE_NAME",
    "TRACE_SCHEMA_VERSION",
    "InMemoryTraceWriter",
    "JsonlTraceWriter",
    "TraceStats",
    "TraceWriter",
    "trace_line",
]
