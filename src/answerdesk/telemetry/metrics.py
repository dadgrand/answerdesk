"""Prometheus metrics without a Prometheus dependency.

``prometheus-client`` is a fine library, but it is not part of the frozen dependency
allowlist (§13.3) and the exposition format is a stable, trivial text protocol. This
module therefore owns a small, deterministic registry:

* counters, gauges and histograms with a fixed label set per family;
* the exact ``text/plain; version=0.0.4`` exposition (``# HELP`` / ``# TYPE`` / samples),
  sorted by family name and label values so two scrapes of the same state are identical;
* no timers, no background threads, no import-time registration — the registry is an
  object the service owns, which is what makes it testable.

The metric set is the one requirements §8.6/§9.5 and NFR-OBS-01 name: request, latency,
error, cache hit/miss/bypass/wrong-hit, escalation, refusal, plus the cost/latency
attribution NFR-COST-03 needs (``answerdesk_stage_duration_seconds``, tokens, USD).

    # AT-OBS-01  # AT-COST-03  # AT-CAC-08  # AT-ESC-10
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

#: Prometheus' conventional HTTP latency buckets, plus a 10 s overflow bucket.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

#: Stage names the trace and the histogram agree on (NFR-COST-03).
STAGES: tuple[str, ...] = (
    "analysis",
    "retrieval",
    "rerank",
    "generation",
    "verify",
    "cache",
    "total",
)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _format_number(value: float) -> str:
    """Render a float the way Prometheus expects (no exponent for normal magnitudes)."""
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _labels(names: Sequence[str], values: Sequence[str]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{name}="{_escape_label(value)}"' for name, value in zip(names, values, strict=True))
    return "{" + pairs + "}"


class _Family:
    """Common surface of every metric family."""

    kind = "untyped"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help_text
        self.label_names = tuple(label_names)

    def _key(self, labels: Sequence[str]) -> tuple[str, ...]:
        if len(labels) != len(self.label_names):
            raise ValueError(
                f"{self.name} expects {len(self.label_names)} label(s) "
                f"{self.label_names}, got {tuple(labels)}"
            )
        return tuple(str(label) for label in labels)

    def render(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(_Family):
    """Monotonic counter (a float so tests can compare exactly)."""

    kind = "counter"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, *labels: str) -> None:
        if amount < 0:
            raise ValueError(f"{self.name} is a counter; it cannot decrease")
        key = self._key(labels)
        self._values[key] = self._values.get(key, 0.0) + float(amount)

    def value(self, *labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def samples(self) -> dict[tuple[str, ...], float]:
        return dict(self._values)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} counter"]
        for key in sorted(self._values):
            lines.append(f"{self.name}{_labels(self.label_names, key)} {_format_number(self._values[key])}")
        return lines


class Gauge(_Family):
    """A value that can go up and down (cache entries, in-flight requests)."""

    kind = "gauge"

    def __init__(self, name: str, help_text: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, label_names)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, *labels: str) -> None:
        self._values[self._key(labels)] = float(value)

    def inc(self, amount: float = 1.0, *labels: str) -> None:
        key = self._key(labels)
        self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1.0, *labels: str) -> None:
        self.inc(-amount, *labels)

    def value(self, *labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} gauge"]
        for key in sorted(self._values):
            lines.append(f"{self.name}{_labels(self.label_names, key)} {_format_number(self._values[key])}")
        return lines


@dataclass
class _Observation:
    buckets: dict[float, int] = field(default_factory=dict)
    count: int = 0
    total: float = 0.0


class Histogram(_Family):
    """Cumulative histogram with explicit bucket bounds (Prometheus semantics)."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        label_names: Sequence[str] = (),
        *,
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, label_names)
        self.bounds = tuple(sorted({float(bound) for bound in buckets}))
        if not self.bounds or self.bounds[-1] == math.inf:
            raise ValueError("histogram buckets must be finite; +Inf is implicit")
        self._observations: dict[tuple[str, ...], _Observation] = {}

    def observe(self, value: float, *labels: str) -> None:
        key = self._key(labels)
        observation = self._observations.setdefault(key, _Observation())
        numeric = float(value)
        if math.isnan(numeric):
            return
        observation.count += 1
        observation.total += numeric
        for bound in self.bounds:
            if numeric <= bound:
                observation.buckets[bound] = observation.buckets.get(bound, 0) + 1

    def count(self, *labels: str) -> int:
        observation = self._observations.get(self._key(labels))
        return observation.count if observation else 0

    def sum(self, *labels: str) -> float:
        observation = self._observations.get(self._key(labels))
        return observation.total if observation else 0.0

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} histogram"]
        for key in sorted(self._observations):
            observation = self._observations[key]
            for bound in self.bounds:
                cumulative = observation.buckets.get(bound, 0)
                labels = (*key, _format_number(bound))
                names = (*self.label_names, "le")
                lines.append(f"{self.name}_bucket{_labels(names, labels)} {cumulative}")
            labels_inf = (*key, "+Inf")
            names_inf = (*self.label_names, "le")
            lines.append(f"{self.name}_bucket{_labels(names_inf, labels_inf)} {observation.count}")
            lines.append(f"{self.name}_sum{_labels(self.label_names, key)} {_format_number(observation.total)}")
            lines.append(f"{self.name}_count{_labels(self.label_names, key)} {observation.count}")
        return lines


class Metrics:
    """The service's metric registry.

    One instance per application (``app.state.metrics``); tests construct their own so a
    test never inherits counters from another test.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.http_requests = Counter(
            "answerdesk_http_requests_total",
            "HTTP requests handled, by method, route template and status code.",
            ("method", "route", "status"),
        )
        self.http_duration = Histogram(
            "answerdesk_http_request_duration_seconds",
            "Wall-clock time from request arrival to final byte.",
            ("method", "route"),
        )
        self.http_errors = Counter(
            "answerdesk_http_errors_total",
            "Error responses by stable error code (docs/api.md taxonomy).",
            ("code",),
        )
        self.stage_duration = Histogram(
            "answerdesk_stage_duration_seconds",
            "Per-stage latency (analysis, retrieval, rerank, generation, verify, cache, total).",
            ("stage",),
        )
        self.decisions = Counter(
            "answerdesk_decisions_total",
            "Policy decisions by action and reason code.",
            ("action", "reason"),
        )
        self.escalations = Counter(
            "answerdesk_escalations_total",
            "Turns escalated to a human, by reason code.",
            ("reason",),
        )
        self.refusals = Counter(
            "answerdesk_refusals_total",
            "Turns refused (never auto-answered), by reason code.",
            ("reason",),
        )
        self.cache_hits = Counter(
            "answerdesk_cache_hits_total",
            "Semantic-cache hits, by kind (exact, semantic, negative).",
            ("kind",),
        )
        self.cache_misses = Counter(
            "answerdesk_cache_misses_total",
            "Semantic-cache misses (including guard demotions).",
        )
        self.cache_bypasses = Counter(
            "answerdesk_cache_bypass_total",
            "Cache lookups bypassed (kill switch or high-risk intent).",
        )
        self.cache_wrong_hits = Counter(
            "answerdesk_cache_wrong_hits_total",
            "Cache hits later labelled wrong by feedback (the error budget numerator).",
        )
        self.cache_hit_latency = Histogram(
            "answerdesk_cache_hit_latency_seconds",
            "Latency of a cache hit (NFR-LAT-04).",
        )
        self.cache_entries = Gauge(
            "answerdesk_cache_entries",
            "Entries currently held by the semantic cache.",
        )
        self.model_tokens = Counter(
            "answerdesk_model_tokens_total",
            "Provider tokens billed, by kind (input, output).",
            ("kind",),
        )
        self.model_cost = Counter(
            "answerdesk_model_cost_usd_total",
            "Provider spend in USD, by model.",
            ("model",),
        )
        self.sessions_created = Counter(
            "answerdesk_sessions_created_total",
            "Sessions created by the service.",
        )
        self.feedback = Counter(
            "answerdesk_feedback_total",
            "Feedback submissions by rating.",
            ("rating",),
        )
        self.rate_limited = Counter(
            "answerdesk_rate_limited_total",
            "Requests rejected by the per-client rate limiter.",
        )
        self.traces_written = Counter(
            "answerdesk_traces_written_total",
            "JSONL trace lines written, by outcome (ok, error).",
            ("outcome",),
        )
        self.idempotent_replays = Counter(
            "answerdesk_idempotent_replays_total",
            "Requests served from the idempotency store instead of the pipeline.",
        )
        self._families: tuple[_Family, ...] = (
            self.http_requests,
            self.http_duration,
            self.http_errors,
            self.stage_duration,
            self.decisions,
            self.escalations,
            self.refusals,
            self.cache_hits,
            self.cache_misses,
            self.cache_bypasses,
            self.cache_wrong_hits,
            self.cache_hit_latency,
            self.cache_entries,
            self.model_tokens,
            self.model_cost,
            self.sessions_created,
            self.feedback,
            self.rate_limited,
            self.traces_written,
            self.idempotent_replays,
        )

    # -- recording helpers (keep the call sites readable) -------------------------
    def observe_request(
        self,
        *,
        method: str,
        route: str,
        status: int,
        duration_s: float,
    ) -> None:
        with self._lock:
            self.http_requests.inc(1.0, method, route, str(status))
            self.http_duration.observe(duration_s, method, route)

    def observe_error(self, code: str) -> None:
        with self._lock:
            self.http_errors.inc(1.0, code)

    def observe_stage(self, stage: str, duration_s: float) -> None:
        with self._lock:
            self.stage_duration.observe(duration_s, stage)

    def observe_decision(self, action: str, reason: str) -> None:
        with self._lock:
            self.decisions.inc(1.0, action, reason)
            if action == "escalate":
                self.escalations.inc(1.0, reason)
            elif action == "refuse":
                self.refusals.inc(1.0, reason)

    def observe_cache_status(self, status: str, *, kind: str = "exact") -> None:
        with self._lock:
            if status == "hit":
                self.cache_hits.inc(1.0, kind)
            elif status == "miss":
                self.cache_misses.inc(1.0)
            elif status == "bypassed":
                self.cache_bypasses.inc(1.0)

    def observe_usage(self, *, input_tokens: int, output_tokens: int, usd: float, model: str) -> None:
        with self._lock:
            if input_tokens:
                self.model_tokens.inc(float(input_tokens), "input")
            if output_tokens:
                self.model_tokens.inc(float(output_tokens), "output")
            if usd:
                self.model_cost.inc(float(usd), model or "unknown")

    def observe_trace(self, *, outcome: str) -> None:
        with self._lock:
            self.traces_written.inc(1.0, outcome)

    def set_cache_entries(self, count: int) -> None:
        with self._lock:
            self.cache_entries.set(float(count))

    def refresh_cache_stats(self, stats: Mapping[str, object]) -> None:
        """Mirror the cache's own counters into the exposed families (idempotent)."""
        with self._lock:
            self.cache_entries.set(float(int(stats.get("entries", 0))))
            self.cache_wrong_hits.inc(0.0)  # materialise the family with its HELP line
            self.cache_hits.inc(0.0, "exact")
            self.cache_hits.inc(0.0, "semantic")
            self.cache_hits.inc(0.0, "negative")
            self.cache_misses.inc(0.0)
            self.cache_bypasses.inc(0.0)

    # -- exposition ----------------------------------------------------------------
    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for family in sorted(self._families, key=lambda item: item.name):
                lines.extend(family.render())
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, object]:
        """Debug/diagnostic view (never scraped; used by tests and /metrics?format=json)."""
        with self._lock:
            return {
                "http_requests": {str(key): value for key, value in self.http_requests.samples().items()},
                "http_errors": {str(key): value for key, value in self.http_errors.samples().items()},
                "decisions": {str(key): value for key, value in self.decisions.samples().items()},
                "escalations": {str(key): value for key, value in self.escalations.samples().items()},
                "refusals": {str(key): value for key, value in self.refusals.samples().items()},
                "cache_hits": {str(key): value for key, value in self.cache_hits.samples().items()},
                "cache_misses": self.cache_misses.value(),
                "cache_bypasses": self.cache_bypasses.value(),
                "cache_wrong_hits": self.cache_wrong_hits.value(),
                "model_tokens": {str(key): value for key, value in self.model_tokens.samples().items()},
                "model_cost_usd": {
                    str(key): value for key, value in self.model_cost.samples().items()
                },
                "sessions_created": self.sessions_created.value(),
                "feedback": {str(key): value for key, value in self.feedback.samples().items()},
                "rate_limited": self.rate_limited.value(),
                "traces_written": {
                    str(key): value for key, value in self.traces_written.samples().items()
                },
            }


def render_metrics(metrics: Metrics) -> str:
    return metrics.render()


def parse_exposition(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse the exposition format back into samples — the test's independent oracle.

    Deliberately simple (the service only ever emits this subset): ``name{labels} value``,
    comments skipped, ``+Inf``/``NaN`` handled. Returns samples grouped by family name.
    """
    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        head, _, raw_value = line.rpartition(" ")
        labels: dict[str, str] = {}
        if "{" in head:
            name, _, raw_labels = head.partition("{")
            raw_labels = raw_labels.rstrip("}")
            for pair in _split_labels(raw_labels):
                key, _, value = pair.partition("=")
                labels[key] = value.strip('"').replace('\\"', '"').replace("\\n", "\n")
        else:
            name = head
        try:
            value = float(raw_value)
        except ValueError:  # pragma: no cover - malformed input
            continue
        samples.setdefault(name, []).append((labels, value))
    return samples


def _split_labels(raw: str) -> Iterable[str]:
    """Split ``a="1",b="2,3"`` on commas that are outside quotes."""
    current: list[str] = []
    in_quotes = False
    escaped = False
    for char in raw:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == "," and not in_quotes:
            yield "".join(current)
            current = []
            continue
        current.append(char)
    if current:
        yield "".join(current)


__all__ = [
    "CONTENT_TYPE",
    "DEFAULT_BUCKETS",
    "STAGES",
    "Counter",
    "Gauge",
    "Histogram",
    "Metrics",
    "parse_exposition",
    "render_metrics",
]
