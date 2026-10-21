/**
 * Diagnostics drawer: everything the service knows about the selected answer.
 *
 * The drawer reads the persisted trace (`GET /v1/traces/{id}`), falling back to the SSE
 * retrieval payload when the trace is not available yet. That is the product's honesty
 * mechanism: the operator inspects the same numbers the release evidence is computed from
 * — retrieval scores and channel, cache status and guard reason, the policy decision with
 * α and `threshold_ref`, per-stage latency and the token/USD cost.
 */

import { useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";
import type { RetrievalEvent, TraceResponse } from "../api/types";
import { formatInt, formatMs, formatPct, formatTokens, formatUsd, humanizeReason, shortId } from "../lib/format";
import { CacheChip, DecisionBadge, Spinner } from "./Badges";

export interface DiagnosticsDrawerProps {
  open: boolean;
  traceId: string | null;
  retrieval: RetrievalEvent | null;
  onClose: () => void;
  loadTrace: (traceId: string) => Promise<TraceResponse>;
}

function ScoreBar({ value, max }: { value: number; max: number }) {
  const width = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0;
  return (
    <span className="score-bar" role="img" aria-label={`score ${value.toFixed(4)}`}>
      <span className="score-bar__fill" style={{ width: `${width}%` }} />
    </span>
  );
}

/** ``· k=60`` from whatever parameters the fuser actually reported. */
function fusionParams(params: Record<string, number | string> | null | undefined): string {
  if (!params) return "";
  const entries = Object.entries(params);
  return entries.length > 0 ? ` · ${entries.map(([key, value]) => `${key}=${value}`).join(", ")}` : "";
}

export function DiagnosticsDrawer({ open, traceId, retrieval, onClose, loadTrace }: DiagnosticsDrawerProps) {
  const [data, setData] = useState<TraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open || !traceId) {
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadTrace(traceId)
      .then((response) => {
        if (!cancelled) setData(response);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? `${cause.code}: ${cause.message}` : String(cause));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, traceId, loadTrace]);

  if (!open) return null;

  const trace = data?.trace ?? null;
  const passages = trace?.retrieval?.packed ?? null;
  const scores = trace?.retrieval_scores ?? null;
  const cache = trace?.cache ?? null;
  const decision = trace?.decision ?? null;
  const timings = trace?.stage_latency_ms ?? trace?.timings_ms ?? null;
  const timingsEntries = timings
    ? Object.entries(timings)
        .filter(([stage]) => stage !== "total")
        .sort((a, b) => b[1] - a[1])
    : [];
  const maxTiming = timingsEntries.reduce((max, [, value]) => Math.max(max, value), 0);
  const maxScore = scores?.reduce((max, row) => Math.max(max, row.score), 0) ?? 0;

  const fallbackPassages = !passages && retrieval
    ? retrieval.passages.map((passage) => ({
        chunk: {
          chunk_id: passage.chunk_id,
          doc_id: passage.doc_id,
          text: passage.snippet,
          heading_path: passage.heading_path,
          metadata: { doc_title: passage.doc_title },
        },
        channel: passage.channel,
        score: passage.score,
        rank: passage.rank,
        stage: passage.stage,
      }))
    : null;
  const shownPassages = passages ?? fallbackPassages ?? [];
  const effectiveCache = cache ?? (retrieval ? { ...retrieval.cache, entry_id: retrieval.cache.entry_id } : null);

  return (
    <aside className="drawer" aria-label="Answer diagnostics">
      <header className="drawer__head">
        <div>
          <h2 className="drawer__title">Diagnostics</h2>
          <p className="drawer__sub">
            {traceId ? (
              <span title={traceId}>trace {shortId(traceId, 10, 6)}</span>
            ) : (
              "no completed turn yet"
            )}
          </p>
        </div>
        <button ref={closeRef} type="button" className="icon-button" onClick={onClose} aria-label="Close diagnostics">
          ×
        </button>
      </header>

      <div className="drawer__body">
        {!traceId ? (
          <p className="drawer__empty">Ask a question first — diagnostics attach to a completed turn and its trace.</p>
        ) : null}

        {loading && !trace ? (
          <p className="drawer__loading">
            <Spinner label="Loading trace" /> reading the trace…
          </p>
        ) : null}

        {error ? (
          <p className="drawer__error" role="alert">
            Trace unavailable ({error}). Showing the SSE retrieval payload instead.
          </p>
        ) : null}

        {decision ? (
          <section className="diag">
            <h3 className="diag__title">Policy decision</h3>
            <div className="diag__row">
              <DecisionBadge decision={decision} />
              <span className="mono">{decision.reason_code}</span>
            </div>
            <dl className="diag__grid">
              <div>
                <dt>Calibrated p̂</dt>
                <dd>{decision.score === null ? "—" : formatPct(decision.score)}</dd>
              </div>
              <div>
                <dt>α (risk tier)</dt>
                <dd>{formatPct(decision.alpha, 0)}</dd>
              </div>
              <div>
                <dt>threshold_ref</dt>
                <dd title={decision.threshold_ref}>{decision.threshold_ref}</dd>
              </div>
              <div>
                <dt>Required to answer</dt>
                <dd>{formatPct(1 - decision.alpha)}</dd>
              </div>
            </dl>
            <p className="diag__note">{humanizeReason(decision.reason_code)}</p>
            {trace?.escalation ? (
              <p className="diag__note">
                Escalated · ticket <span className="mono">{trace.escalation.ticket_id}</span> ({trace.escalation.reason})
              </p>
            ) : null}
          </section>
        ) : null}

        <section className="diag">
          <h3 className="diag__title">Retrieval candidates</h3>
          {shownPassages.length === 0 ? (
            <p className="diag__empty">No passage survived packing for this turn.</p>
          ) : (
            <ol className="retrieval" aria-label="Retrieval candidates">
              {shownPassages.map((passage) => {
                const title = String(passage.chunk.metadata?.doc_title ?? passage.chunk.doc_id);
                return (
                  <li key={passage.chunk.chunk_id} className="retrieval__item">
                    <div className="retrieval__top">
                      <span className="retrieval__rank">#{passage.rank}</span>
                      <span className={`channel channel--${passage.channel}`}>{passage.channel}</span>
                      <span className="retrieval__score">{passage.score.toFixed(4)}</span>
                    </div>
                    <ScoreBar value={passage.score} max={maxScore || passage.score} />
                    <p className="retrieval__doc">{title}</p>
                    <p className="retrieval__path">{passage.chunk.heading_path.join(" › ")}</p>
                    <p className="retrieval__snippet" dir="auto">
                      {passage.chunk.text}
                    </p>
                  </li>
                );
              })}
            </ol>
          )}
          {scores && scores.length > 0 ? (
            <table className="mini-table">
              <caption>Channels and scores (packed order)</caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Chunk</th>
                  <th scope="col">Channel</th>
                  <th scope="col">Score</th>
                </tr>
              </thead>
              <tbody>
                {scores.map((row) => (
                  <tr key={`${row.chunk_id}-${row.rank}`}>
                    <td>{row.rank}</td>
                    <td className="mono" title={row.chunk_id}>
                      {shortId(row.chunk_id, 12, 4)}
                    </td>
                    <td>{row.channel}</td>
                    <td>{row.score.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          <dl className="diag__grid">
            <div>
              <dt>Fusion</dt>
              <dd>
                {(trace?.fusion?.fuser ?? retrieval?.fusion?.fuser ?? "—") +
                  fusionParams(trace?.fusion?.params ?? retrieval?.fusion?.params)}
              </dd>
            </div>
            <div>
              <dt>Candidates</dt>
              <dd>{formatInt(trace?.retrieval?.packed.length ?? retrieval?.candidates ?? 0)} packed</dd>
            </div>
            <div>
              <dt>Duplicates dropped</dt>
              <dd>{formatInt(trace?.retrieval?.dropped_duplicates ?? retrieval?.dropped_duplicates ?? 0)}</dd>
            </div>
          </dl>
        </section>

        <section className="diag">
          <h3 className="diag__title">Semantic cache</h3>
          {effectiveCache ? (
            <>
              <div className="diag__row">
                <CacheChip status={effectiveCache.status} similarity={effectiveCache.similarity} />
                <span className="mono">{effectiveCache.reason || "—"}</span>
              </div>
              <dl className="diag__grid">
                <div>
                  <dt>Entry</dt>
                  <dd title={effectiveCache.entry_id ?? ""}>
                    {effectiveCache.entry_id ? shortId(effectiveCache.entry_id, 10, 4) : "—"}
                  </dd>
                </div>
                <div>
                  <dt>Similarity</dt>
                  <dd>{effectiveCache.similarity === null ? "—" : formatPct(effectiveCache.similarity)}</dd>
                </div>
              </dl>
            </>
          ) : (
            <p className="diag__empty">No cache information for this turn.</p>
          )}
        </section>

        <section className="diag">
          <h3 className="diag__title">Latency by stage</h3>
          {timingsEntries.length === 0 ? (
            <p className="diag__empty">The trace has no stage timings.</p>
          ) : (
            <ul className="timings" aria-label="Latency by stage">
              {timingsEntries.map(([stage, value]) => (
                <li key={stage} className="timings__row">
                  <span className="timings__stage">{stage}</span>
                  <span className="timings__bar">
                    <span
                      className="timings__fill"
                      style={{ width: `${maxTiming > 0 ? Math.max(2, (value / maxTiming) * 100) : 0}%` }}
                    />
                  </span>
                  <span className="timings__value">{formatMs(value)}</span>
                </li>
              ))}
              <li className="timings__row timings__row--total">
                <span className="timings__stage">total</span>
                <span className="timings__bar">
                  <span className="timings__fill" style={{ width: "100%" }} />
                </span>
                <span className="timings__value">{formatMs(timings?.total ?? null)}</span>
              </li>
            </ul>
          )}
        </section>

        <section className="diag">
          <h3 className="diag__title">Cost &amp; provenance</h3>
          <dl className="diag__grid">
            <div>
              <dt>Model</dt>
              <dd>{trace?.usage.model ?? "—"}</dd>
            </div>
            <div>
              <dt>Tokens in / out</dt>
              <dd>
                {formatTokens(trace?.usage.input_tokens ?? 0)} / {formatTokens(trace?.usage.output_tokens ?? 0)}
              </dd>
            </div>
            <div>
              <dt>USD</dt>
              <dd>{formatUsd(trace?.usage.usd ?? null)}</dd>
            </div>
            <div>
              <dt>Prompt</dt>
              <dd title={trace?.prompt_version ?? ""}>{trace?.prompt_version || "—"}</dd>
            </div>
            <div>
              <dt>Corpus</dt>
              <dd title={trace?.corpus_version ?? ""}>{shortId(trace?.corpus_version ?? "—", 12, 4)}</dd>
            </div>
            <div>
              <dt>Index</dt>
              <dd title={trace?.index_version ?? ""}>{shortId(trace?.index_version ?? "—", 12, 4)}</dd>
            </div>
            <div>
              <dt>Verifier</dt>
              <dd>{trace?.generation?.verifier_version ?? "—"}</dd>
            </div>
            <div>
              <dt>Calibration</dt>
              <dd title={trace?.service?.calibration?.source ?? ""}>
                {trace?.service?.calibration?.threshold_ref ?? "—"}
                {trace?.service?.calibration?.stale ? " · stale" : ""}
              </dd>
            </div>
          </dl>
          {trace?.generation ? (
            <p className="diag__note">
              repaired {trace.generation.repair_attempts}× · dropped claims {trace.generation.dropped_claims} · numeric
              violations pre-repair {trace.generation.numeric_violations_pre_repair}
            </p>
          ) : null}
        </section>

        {trace ? (
          <details className="raw">
            <summary>Raw trace document (redacted JSONL)</summary>
            <pre className="raw__pre">{JSON.stringify(trace, null, 2)}</pre>
          </details>
        ) : null}
      </div>
    </aside>
  );
}
