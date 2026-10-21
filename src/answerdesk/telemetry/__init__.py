"""Structured traces and Prometheus metrics (t9).

Owning task: **t9** (service + observability). The package owns two seams the whole stack
writes through: the JSONL trace document and the Prometheus registry.

Public surface for the service, the console and any operator tooling:

=================================  =====================================================
:class:`Metrics`                   dependency-free Prometheus registry (counters,
                                   histograms, gauges) and its exposition
:class:`JsonlTraceWriter`          append-only, rotating JSONL trace sink
:class:`InMemoryTraceWriter`       the same seam for tests and offline runs
``trace_line``                     validated :class:`TraceRecord` → trace document
``redact_text`` / ``redact_payload``  PII + secret redaction at every boundary (INV-05)
=================================  =====================================================

Requirements: NFR-OBS-01, NFR-COST-03, FR-SVC-08; contract in ``docs/api.md`` §6 and
``docs/ops.md`` §5.

Every type that crosses this module's boundary comes from
:mod:`answerdesk.contracts` (the trace document itself is a versioned extension of
:class:`answerdesk.contracts.TraceRecord`).
"""

from __future__ import annotations

from answerdesk.telemetry.metrics import (
    CONTENT_TYPE,
    DEFAULT_BUCKETS,
    STAGES,
    Counter,
    Gauge,
    Histogram,
    Metrics,
    parse_exposition,
    render_metrics,
)
from answerdesk.telemetry.redaction import (
    Redacted,
    RedactionSummary,
    contains_pii,
    find_pii,
    redact_messages,
    redact_payload,
    redact_str,
    redact_text,
    redact_value,
)
from answerdesk.telemetry.tracing import (
    TRACE_FILE_NAME,
    TRACE_SCHEMA_VERSION,
    InMemoryTraceWriter,
    JsonlTraceWriter,
    TraceStats,
    TraceWriter,
    trace_line,
)

__all__ = [
    "CONTENT_TYPE",
    "DEFAULT_BUCKETS",
    "STAGES",
    "TRACE_FILE_NAME",
    "TRACE_SCHEMA_VERSION",
    "Counter",
    "Gauge",
    "Histogram",
    "InMemoryTraceWriter",
    "JsonlTraceWriter",
    "Metrics",
    "Redacted",
    "RedactionSummary",
    "TraceStats",
    "TraceWriter",
    "contains_pii",
    "find_pii",
    "parse_exposition",
    "redact_messages",
    "redact_payload",
    "redact_str",
    "redact_text",
    "redact_value",
    "render_metrics",
    "trace_line",
]
