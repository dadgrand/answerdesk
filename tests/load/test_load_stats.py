"""Statistics, workload and assertion logic of the load harness (fast, no app).

These tests pin the parts a wrong number would make meaningless: the percentile
implementation, the determinism of the workload, the cache-phase construction and the
pass/fail rules the CLI exits on.

    # AT-OPS-07
"""

from __future__ import annotations

import pytest


def test_percentile_matches_the_linear_interpolation_definition(load_harness) -> None:
    values = [float(value) for value in range(1, 101)]  # 1..100
    assert load_harness.percentile(values, 50) == pytest.approx(50.5)
    assert load_harness.percentile(values, 95) == pytest.approx(95.05)
    assert load_harness.percentile(values, 99) == pytest.approx(99.01)
    assert load_harness.percentile([7.0], 99) == 7.0
    assert load_harness.percentile([], 95) is None


def test_latency_summary_reports_the_documented_fields(load_harness) -> None:
    summary = load_harness.latency_summary([1.0, 2.0, 3.0, 4.0])
    assert set(summary) == {"p50", "p95", "p99", "max", "mean"}
    assert summary["max"] == 4.0
    assert summary["mean"] == 2.5
    assert load_harness.latency_summary([])["p50"] is None


def test_workload_is_deterministic_and_mixed(load_harness) -> None:
    first = load_harness.build_workload(40)
    second = load_harness.build_workload(40)
    assert first == second
    kinds = {request.kind for request in first}
    assert kinds == {
        "answerable-stream",
        "answerable-json",
        "unanswerable-stream",
        "unanswerable-json",
    }
    assert any(request.stream for request in first) and any(not request.stream for request in first)
    # cold phase: distinct surface forms, so the first pass is genuinely cold
    answerable = [request.message for request in first if request.kind.startswith("answerable")]
    assert len(set(answerable)) == len(answerable)


def test_cache_hits_are_built_from_the_queries_that_were_actually_answered(load_harness) -> None:
    sink = load_harness.CollectingTraceWriter()
    sink.write(
        {
            "trace_id": "tr-1",
            "kind": "turn",
            "action": "answer",
            "query": "обслуживание карты",
        }
    )
    sink.write(
        {
            "trace_id": "tr-2",
            "kind": "turn",
            "action": "escalate",
            "query": "Сколько стоит билет на Марс?",
        }
    )
    samples = [
        load_harness.Sample(kind="answerable-json", status=200, duration_ms=1.0, trace_id="tr-1"),
        load_harness.Sample(kind="unanswerable-json", status=200, duration_ms=1.0, trace_id="tr-2"),
    ]
    assert load_harness.answered_queries(samples, sink) == ["обслуживание карты"]
    warm = load_harness.build_cache_hits(["обслуживание карты"], 4)
    assert {request.message for request in warm} == {"обслуживание карты"}
    assert {request.kind for request in warm} == {"answerable-stream", "answerable-json"}


def test_evaluate_catches_every_documented_failure(load_harness) -> None:
    clean = {
        "levels": [
            {
                "level": {"concurrency": 8},
                "cold": {
                    "status_counts": {"200": 10},
                    "timeouts": 0,
                    "errors": 0,
                    "done_frames_missing": 0,
                    "stages_ms": {"total": {"p50": 1.0}},
                },
                "warm": {
                    "status_counts": {"200": 10},
                    "timeouts": 0,
                    "errors": 0,
                    "done_frames_missing": 0,
                    "stages_ms": {"total": {"p50": 1.0}},
                },
            }
        ],
        "leaks": {
            "rss_growth_mb": 1.0,
            "handle_growth": 0,
            "thread_growth": 0,
            "idempotency_locks": 0,
        },
        "rate_limit": {
            "http_429": 5,
            "retry_after_present": True,
            "server_errors": 0,
            "timeouts": 0,
            "health_status": 200,
            "metrics_status": 200,
        },
    }
    assert load_harness.evaluate(clean) == {"passed": True, "failures": []}

    broken = load_harness.json.loads(load_harness.json.dumps(clean))
    broken["levels"][0]["cold"]["status_counts"] = {"200": 9, "500": 1}
    broken["leaks"]["rss_growth_mb"] = 500.0
    broken["leaks"]["idempotency_locks"] = 2
    broken["rate_limit"]["http_429"] = 0
    failures = load_harness.evaluate(broken)["failures"]
    assert any("5xx" in failure for failure in failures)
    assert any("RSS" in failure for failure in failures)
    assert any("idempotency" in failure for failure in failures)
    assert any("429" in failure for failure in failures)


def test_process_probe_reports_live_resources(load_harness) -> None:
    probe = load_harness.ProcessProbe()
    snapshot = probe.snapshot()
    assert set(snapshot) == {"rss_mb", "handles", "threads"}
    assert snapshot["threads"] >= 1
    assert probe.peak_rss_mb() is None or probe.peak_rss_mb() > 0


def test_stage_summary_groups_by_stage(load_harness) -> None:
    samples = [
        load_harness.Sample(
            kind="k", status=200, duration_ms=3.0, stage_ms={"retrieval": 1.0, "generation": 1.5}
        ),
        load_harness.Sample(
            kind="k", status=200, duration_ms=4.0, stage_ms={"retrieval": 2.0, "generation": 2.5}
        ),
    ]
    summary = load_harness._stage_summary(samples)
    assert set(summary) == {"retrieval", "generation"}
    assert summary["retrieval"]["p50"] == pytest.approx(1.5)
    assert summary["generation"]["max"] == 2.5
