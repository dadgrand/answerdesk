/**
 * Analytics: latency, cost per answer, escalation rate and cache hit rate from `/metrics`.
 *
 * NFR-OBS-02 requires the console to read real API series, so this page parses the
 * Prometheus exposition the service exposes (docs/ops.md §5.1) instead of shipping canned
 * numbers. Every KPI states the metric family it comes from in its hint, which makes a
 * suspicious number traceable in one step.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, fetchMetricsText } from "../api/client";
import { actionLabel, deriveKpis, parsePrometheus, type Kpis } from "../api/metrics";
import { BarList, Histogram, KpiCard, RingGauge, StackedBar } from "../components/charts";
import { ErrorBanner, Spinner } from "../components/Badges";
import { formatInt, formatMs, formatPct, formatTokens, formatUsd } from "../lib/format";

const LATENCY_BUDGET_MS = 4000;

interface Loaded {
  kpis: Kpis;
  fetchedAt: Date;
}

export function AnalyticsPage() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const text = await fetchMetricsText();
      setData({ kpis: deriveKpis(parsePrometheus(text)), fetchedAt: new Date() });
    } catch (cause) {
      setError(cause instanceof ApiError ? `${cause.code}: ${cause.message}` : String(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const kpis = data?.kpis ?? null;

  const decisionSegments = useMemo(() => {
    if (!kpis) return [];
    return [
      { label: "answered", value: kpis.answers, tone: "answer" },
      { label: "clarified", value: kpis.clarifications, tone: "clarify" },
      { label: "escalated", value: kpis.escalations, tone: "escalate" },
      { label: "refused", value: kpis.refusals, tone: "refuse" },
    ].filter((segment) => segment.value > 0);
  }, [kpis]);

  const p95Tone = kpis?.latencyP95 === null || kpis?.latencyP95 === undefined
    ? "neutral"
    : kpis.latencyP95 > LATENCY_BUDGET_MS
      ? "bad"
      : kpis.latencyP95 > LATENCY_BUDGET_MS * 0.6
        ? "warn"
        : "good";

  return (
    <section className="analytics" aria-labelledby="analytics-title">
      <header className="analytics__head">
        <div>
          <h2 id="analytics-title" className="analytics__title">
            Service analytics
          </h2>
          <p className="analytics__sub">
            Live scrape of <code>GET /metrics</code> — Prometheus exposition, parsed in the browser. Budgets from
            requirements NFR-LAT-01 (p95 ≤ 4 s) and NFR-OBS-02.
          </p>
        </div>
        <div className="analytics__actions">
          <span className="analytics__stamp">
            {data ? `scraped ${data.fetchedAt.toLocaleTimeString()}` : loading ? "scraping…" : "no scrape yet"}
          </span>
          <button type="button" className="button button--ghost" onClick={() => void load()} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      {error ? <ErrorBanner title="Metrics unavailable" message={error} onRetry={() => void load()} /> : null}

      {!data && loading ? (
        <p className="analytics__loading">
          <Spinner label="Loading metrics" /> reading the metric registry…
        </p>
      ) : null}

      {kpis ? (
        <>
          <div className="kpi-grid">
            <KpiCard
              label="Requests"
              value={formatInt(kpis.requests)}
              hint="answerdesk_http_requests_total"
              icon="⇄"
            />
            <KpiCard
              label="p50 latency"
              value={formatMs(kpis.latencyP50)}
              hint="stage total · histogramQuantile(0.5)"
              icon="◷"
            />
            <KpiCard
              label="p95 latency"
              value={formatMs(kpis.latencyP95)}
              hint={`stage total · budget ${formatMs(LATENCY_BUDGET_MS)}`}
              tone={p95Tone === "neutral" ? "neutral" : p95Tone}
              icon="⏱"
            />
            <KpiCard
              label="USD / answer"
              value={formatUsd(kpis.usdPerAnswer)}
              hint={
                kpis.usdTotal === 0
                  ? "no priced calls in this scrape (offline fake provider)"
                  : `${formatUsd(kpis.usdTotal)} total · decision-weighted`
              }
              icon="$"
            />
            <KpiCard
              label="Escalation rate"
              value={formatPct(kpis.escalationRate)}
              hint={`${formatInt(kpis.escalations)} escalations / ${formatInt(kpis.answers + kpis.escalations + kpis.clarifications + kpis.refusals)} decisions`}
              tone={kpis.escalationRate !== null && kpis.escalationRate > 0.4 ? "warn" : "good"}
              icon="⚑"
            />
            <KpiCard
              label="Cache hit rate"
              value={formatPct(kpis.cacheHitRate)}
              hint={`${formatInt(kpis.cacheHits)} hits · ${formatInt(kpis.cacheMisses)} misses`}
              tone={kpis.cacheHitRate !== null && kpis.cacheHitRate >= 0.15 ? "good" : "neutral"}
              icon="⚡"
            />
          </div>

          <div className="analytics__grid">
            <article className="panel panel--wide">
              <header className="panel__head">
                <h3 className="panel__title">Latency distribution</h3>
                <span className="panel__hint">answerdesk_stage_duration_seconds · stage=total</span>
              </header>
              <Histogram
                buckets={kpis.latencyDistribution}
                format={(value) => (value >= 1 ? `${value}s` : `${Math.round(value * 1000)}ms`)}
                ariaLabel="Latency distribution histogram"
              />
              <BarList
                ariaLabel="p95 against the latency budget"
                items={[
                  {
                    label: "p95",
                    value: kpis.latencyP95 ?? 0,
                    display: formatMs(kpis.latencyP95),
                    tone: p95Tone === "bad" ? "bad" : "good",
                  },
                  { label: "budget", value: LATENCY_BUDGET_MS, display: formatMs(LATENCY_BUDGET_MS), tone: "muted" },
                ]}
              />
              {kpis.latencyP95 !== null && kpis.latencyP95 < 5 ? (
                <p className="panel__note">
                  Every sample falls below the first 5 ms bucket: the offline profile answers with deterministic
                  doubles. A model profile fills the upper buckets.
                </p>
              ) : null}
              <div className="panel__stats">
                <div>
                  <span className="panel__stat-label">p50</span>
                  <span className="panel__stat-value">{formatMs(kpis.latencyP50)}</span>
                </div>
                <div>
                  <span className="panel__stat-label">p95</span>
                  <span className="panel__stat-value">{formatMs(kpis.latencyP95)}</span>
                </div>
                <div>
                  <span className="panel__stat-label">errors</span>
                  <span className={`panel__stat-value ${kpis.errorCount > 0 ? "panel__stat-value--warn" : ""}`}>
                    {formatInt(kpis.errorCount)}
                  </span>
                </div>
              </div>
            </article>

            <article className="panel">
              <header className="panel__head">
                <h3 className="panel__title">Decision mix</h3>
                <span className="panel__hint">answerdesk_decisions_total</span>
              </header>
              <StackedBar segments={decisionSegments} ariaLabel="Decision mix" />
              <table className="mini-table">
                <caption>Decisions by action</caption>
                <thead>
                  <tr>
                    <th scope="col">Action</th>
                    <th scope="col">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {kpis.decisions.map((row) => (
                    <tr key={row.action}>
                      <td>{actionLabel(row.action)}</td>
                      <td>{formatInt(row.count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </article>

            <article className="panel">
              <header className="panel__head">
                <h3 className="panel__title">Semantic cache</h3>
                <span className="panel__hint">cache_hits · cache_misses · cache_bypass</span>
              </header>
              <div className="gauge-wrap">
                <RingGauge
                  value={kpis.cacheHitRate}
                  label="hit rate"
                  display={formatPct(kpis.cacheHitRate, 0)}
                  tone={kpis.cacheHitRate !== null && kpis.cacheHitRate >= 0.15 ? "good" : "warn"}
                />
              </div>
              <BarList
                ariaLabel="Cache hits by kind"
                items={[
                  ...kpis.cacheBreakdown.map((row) => ({
                    label: `${row.kind} hits`,
                    value: row.count,
                    display: formatInt(row.count),
                    tone: "good",
                  })),
                  { label: "misses", value: kpis.cacheMisses, display: formatInt(kpis.cacheMisses), tone: "warn" },
                  { label: "bypassed", value: kpis.cacheBypasses, display: formatInt(kpis.cacheBypasses), tone: "muted" },
                ]}
              />
            </article>

            <article className="panel">
              <header className="panel__head">
                <h3 className="panel__title">Stage p95</h3>
                <span className="panel__hint">stage_duration_seconds · per stage</span>
              </header>
              <BarList
                ariaLabel="Latency p95 by stage"
                items={kpis.stageLatency.map((row) => ({
                  label: row.stage,
                  value: row.p95,
                  display: formatMs(row.p95),
                  tone: row.stage === "generation" ? "warn" : "accent",
                }))}
                emptyHint="No stage timings in this scrape."
              />
            </article>

            <article className="panel panel--wide">
              <header className="panel__head">
                <h3 className="panel__title">Cost &amp; throughput</h3>
                <span className="panel__hint">model_cost_usd_total · model_tokens_total</span>
              </header>
              <div className="analytics__split">
                <div>
                  <h4 className="panel__subtitle">Spend by model</h4>
                  {kpis.models.length > 0 ? (
                    <BarList
                      ariaLabel="Spend by model"
                      items={kpis.models.map((row) => ({
                        label: row.model,
                        value: row.usd,
                        display: formatUsd(row.usd, 4),
                        tone: "accent",
                      }))}
                      emptyHint="No model cost recorded yet."
                    />
                  ) : (
                    <p className="chart__empty">
                      No billable price for this model in this scrape — the offline fake provider is free.
                    </p>
                  )}
                  <h4 className="panel__subtitle">Tokens</h4>
                  <BarList
                    ariaLabel="Tokens by direction"
                    items={[
                      { label: "input", value: kpis.tokensIn, display: formatTokens(kpis.tokensIn), tone: "accent" },
                      { label: "output", value: kpis.tokensOut, display: formatTokens(kpis.tokensOut), tone: "good" },
                    ]}
                  />
                </div>
                <table className="mini-table">
                  <caption>Cost summary</caption>
                  <tbody>
                    <tr>
                      <th scope="row">Total spend</th>
                      <td>{formatUsd(kpis.usdTotal)}</td>
                    </tr>
                    <tr>
                      <th scope="row">Answered turns</th>
                      <td>{formatInt(kpis.answers)}</td>
                    </tr>
                    <tr>
                      <th scope="row">USD per answered turn</th>
                      <td>{formatUsd(kpis.usdPerAnswer)}</td>
                    </tr>
                    <tr>
                      <th scope="row">Escalated turns</th>
                      <td>{formatInt(kpis.escalations)}</td>
                    </tr>
                    <tr>
                      <th scope="row">Clarifications</th>
                      <td>{formatInt(kpis.clarifications)}</td>
                    </tr>
                    <tr>
                      <th scope="row">Refusals (guards)</th>
                      <td>{formatInt(kpis.refusals)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </article>
          </div>

          <p className="analytics__source">
            Source: <code>GET /metrics</code> · service profile and versions on the Console screen · numbers update on
            every scrape, no history is stored by the console.
          </p>
        </>
      ) : null}
    </section>
  );
}
