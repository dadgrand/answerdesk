"""End-to-end load smoke: the real ASGI app, four workloads, cold and warm cache.

This is the CI gate for the load harness: it must finish quickly, report per-stage
percentiles, separate the cache paths, respect the rate limiter and leave no resources
behind. A failure here means the *harness* or the *service* regressed, not that a percentile
moved by a millisecond.

    # AT-OPS-07  # NFR-LAT-01  # NFR-OPS-01
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.integration


def test_smoke_load_pass(load_harness) -> None:
    report = load_harness.run(mode="smoke", levels=[1, 4], requests_per_phase=12, leak_waves=2)

    verdict = report["assertions"]
    assert verdict["passed"], verdict["failures"]
    assert report["mode"] == "smoke"
    assert report["service"]["fake"] is True
    assert report["executor"]["max_workers"] >= 1

    levels = {level["level"]["concurrency"]: level for level in report["levels"]}
    assert set(levels) == {1, 4}
    for concurrency, level in levels.items():
        for phase in ("cold", "warm"):
            data = level[phase]
            assert data["requests"] == 12
            assert data["errors"] == 0
            assert data["timeouts"] == 0
            assert all(int(status) < 500 for status in data["status_counts"]), data["status_counts"]
            assert data["done_frames_missing"] == 0
            assert data["throughput_rps"] and data["throughput_rps"] > 0
            latency = data["latency_ms"]
            assert latency["p50"] is not None
            assert latency["p99"] >= latency["p95"] >= latency["p50"]
            # per-stage attribution is the point of the harness
            for stage in ("retrieval", "generation", "verify", "policy", "cache", "total"):
                assert stage in data["stages_ms"], (concurrency, phase, stage, data["stages_ms"])
            assert data["pending_tasks"] <= 2, data["pending_tasks"]

    # cold path: nothing was answered before, so no hit is expected for these surface forms
    assert levels[1]["cold"]["cache"].get("hit", 0) == 0
    # warm path: the prefill answered the pool, so *every* replay is an exact hit
    warm_hits = levels[1]["warm"]["cache"].get("hit", 0)
    assert warm_hits == 12, levels[1]["warm"]["cache"]
    # a cache hit skips retrieval and generation
    warm_stages = levels[1]["warm"]["stages_ms"]
    cold_stages = levels[1]["cold"]["stages_ms"]
    assert warm_stages["generation"]["p50"] <= cold_stages["generation"]["p50"]


def test_smoke_load_is_fast_enough_for_ci(load_harness) -> None:
    """Contract: the smoke mode stays under 30 s."""
    started = time.perf_counter()
    report = load_harness.run(mode="smoke", levels=[1], requests_per_phase=8, leak_waves=2)
    elapsed = time.perf_counter() - started
    assert report["assertions"]["passed"], report["assertions"]["failures"]
    assert elapsed < 30.0, f"smoke load took {elapsed:.1f}s"


def test_rate_limiting_under_load(load_harness, tmp_path) -> None:
    rate = load_harness.run_rate_limit(tmp_path, concurrency=8, count=96)
    assert rate["server_errors"] == 0, rate
    assert rate["timeouts"] == 0, rate
    assert rate["http_429"] > 0, rate
    assert rate["retry_after_present"] is True, rate
    assert rate["health_status"] == 200
    assert rate["metrics_status"] == 200
    assert all(int(status) != 500 for status in rate["status_counts"]), rate["status_counts"]


def test_leak_waves_leave_no_locks_or_handles(load_harness, tmp_path) -> None:
    leaks = load_harness.run_leak_waves(
        workdir=tmp_path / "leaks",
        concurrency=8,
        count=16,
        waves=3,
        probe=load_harness.ProcessProbe(),
    )
    assert leaks["idempotency_locks"] == 0, leaks
    assert leaks["thread_growth"] <= 4, leaks
    if leaks["handle_growth"] is not None:
        assert leaks["handle_growth"] <= 32, leaks
    if leaks["rss_growth_mb"] is not None:
        assert leaks["rss_growth_mb"] <= 64.0, leaks
    assert leaks["traces_collected"] >= 3 * 16 * 0.5, leaks["traces_collected"]
