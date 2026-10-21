/** Analytics math tests: the exposition parser and every KPI derivation. */

import { describe, expect, it } from "vitest";

import { mockMetrics } from "./mock/fixtures";
import {
  deriveKpis,
  histogramBuckets,
  histogramQuantile,
  parsePrometheus,
  samples,
  sum,
} from "./metrics";

describe("parsePrometheus", () => {
  it("parses labels, quoted values and numbers", () => {
    const family = parsePrometheus(
      [
        "# HELP answerdesk_http_requests_total requests",
        "# TYPE answerdesk_http_requests_total counter",
        'answerdesk_http_requests_total{method="POST",route="/v1/chat",status="200"} 64',
        'answerdesk_http_requests_total{method="GET",route="/healthz",status="200"} 12',
        "answerdesk_cache_misses_total 22",
      ].join("\n"),
    );
    expect(sum(family, "answerdesk_http_requests_total")).toBe(76);
    expect(sum(family, "answerdesk_http_requests_total", { route: "/v1/chat" })).toBe(64);
    expect(sum(family, "answerdesk_cache_misses_total")).toBe(22);
  });

  it("keeps commas inside quoted label values", () => {
    const family = parsePrometheus('metric{a="x,y",b="z"} 1');
    expect(samples(family, "metric")[0]?.labels).toEqual({ a: "x,y", b: "z" });
  });

  it("ignores comments and malformed lines", () => {
    const family = parsePrometheus("# comment\nnot a sample\nmetric 3\ngarbage");
    expect(sum(family, "metric")).toBe(3);
  });
});

describe("histogramQuantile", () => {
  const text = [
    'metric_bucket{le="0.1"} 10',
    'metric_bucket{le="0.5"} 40',
    'metric_bucket{le="1"} 80',
    'metric_bucket{le="+Inf"} 100',
    "metric_count 100",
    "metric_sum 42",
  ].join("\n");
  const family = parsePrometheus(text);
  const histogram = histogramBuckets(family, "metric");

  it("extracts cumulative buckets and totals", () => {
    expect(histogram.count).toBe(100);
    expect(histogram.sum).toBe(42);
    expect(histogram.buckets).toHaveLength(4);
    expect(histogram.buckets[3]?.le).toBe(Number.POSITIVE_INFINITY);
  });

  it("interpolates within a bucket", () => {
    // p50 = rank 50 → between 0.5 (40) and 1 (80): 0.5 + (10/40)*0.5 = 0.625
    expect(histogramQuantile(histogram.buckets, 0.5, histogram.count)).toBeCloseTo(0.625, 5);
    // p95 falls in the +Inf bucket: clamp to the last finite bound (Prometheus returns +Inf).\n    expect(histogramQuantile(histogram.buckets, 0.95, histogram.count)).toBe(1);
  });

  it("returns null for an empty histogram", () => {
    expect(histogramQuantile([], 0.5)).toBeNull();
  });
});

describe("deriveKpis", () => {
  const kpis = deriveKpis(parsePrometheus(mockMetrics()));

  it("counts requests and decisions", () => {
    expect(kpis.requests).toBe(64 + 9 + 12 + 8 + 14 + 6);
    expect(kpis.answers).toBe(41);
    expect(kpis.escalations).toBe(15);
    expect(kpis.clarifications).toBe(6);
    expect(kpis.refusals).toBe(2);
    expect(kpis.errorCount).toBe(3);
  });

  it("derives escalation rate, cost per answer and cache hit rate", () => {
    expect(kpis.escalationRate).toBeCloseTo(15 / 64, 6);
    expect(kpis.usdPerAnswer).toBeCloseTo(0.1834 / 41, 6);
    expect(kpis.cacheHitRate).toBeCloseTo(13 / 35, 6);
    expect(kpis.cacheHits).toBe(13);
    expect(kpis.cacheMisses).toBe(22);
    expect(kpis.cacheBypasses).toBe(3);
  });

  it("computes latency percentiles from the stage histogram", () => {
    expect(kpis.latencyP50).not.toBeNull();
    expect(kpis.latencyP95).not.toBeNull();
    expect(kpis.latencyP95!).toBeGreaterThan(kpis.latencyP50!);
    expect(kpis.latencyDistribution.length).toBeGreaterThan(5);
  });

  it("reports per-stage p95 and model spend", () => {
    const stages = kpis.stageLatency.map((row) => row.stage);
    expect(stages).toContain("retrieval");
    expect(stages).toContain("generation");
    const generation = kpis.stageLatency.find((row) => row.stage === "generation");
    const retrieval = kpis.stageLatency.find((row) => row.stage === "retrieval");
    expect(generation!.p95).toBeGreaterThan(retrieval!.p95);
    expect(kpis.models[0]).toEqual({ model: "gpt-4o-mini", usd: 0.1834 });
    expect(kpis.tokensIn).toBe(51968);
    expect(kpis.tokensOut).toBe(6144);
  });

  it("degrades gracefully on an empty scrape", () => {
    const empty = deriveKpis(parsePrometheus(""));
    expect(empty.requests).toBe(0);
    expect(empty.latencyP50).toBeNull();
    expect(empty.usdPerAnswer).toBeNull();
    expect(empty.escalationRate).toBeNull();
    expect(empty.cacheHitRate).toBeNull();
    expect(empty.decisions).toEqual([]);
  });
});
