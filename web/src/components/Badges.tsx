/** Small presentational primitives shared by both screens: badges, meter, chips. */

import { formatPct } from "../lib/format";
import type { CacheStatus, Decision } from "../api/types";
import { humanizeReason, languageLabel } from "../lib/format";

export function LanguageBadge({ locale }: { locale: string | null | undefined }) {
  return (
    <span className="badge badge--lang" title={`Locale: ${locale ?? "unknown"}`}>
      {languageLabel(locale)}
    </span>
  );
}

const ACTION_CLASS: Record<string, string> = {
  answer: "badge--answer",
  clarify: "badge--clarify",
  escalate: "badge--escalate",
  refuse: "badge--refuse",
};

export function DecisionBadge({ decision }: { decision: Decision | null }) {
  if (!decision) return null;
  return (
    <span
      className={`badge ${ACTION_CLASS[decision.action] ?? "badge--neutral"}`}
      title={`Reason code: ${decision.reason_code}`}
    >
      <span className="badge__dot" aria-hidden="true" />
      {decision.action}
      <span className="badge__detail">{humanizeReason(decision.reason_code)}</span>
    </span>
  );
}

export function CacheChip({ status, similarity }: { status: CacheStatus; similarity?: number | null }) {
  const label = status === "hit" ? "cache hit" : status === "bypassed" ? "cache bypassed" : "cache miss";
  return (
    <span className={`chip chip--cache chip--cache-${status}`} title={label}>
      {status === "hit" ? "⚡" : status === "bypassed" ? "⤼" : "○"} {label}
      {status === "hit" && similarity !== null && similarity !== undefined ? (
        <span className="chip__meta">{formatPct(similarity, 0)}</span>
      ) : null}
    </span>
  );
}

export interface ConfidenceMeterProps {
  value: number | null;
  threshold?: number;
  label?: string;
}

/** The calibrated confidence bar with the decision threshold marked. */
export function ConfidenceMeter({ value, threshold = 0.95, label = "Calibrated p̂" }: ConfidenceMeterProps) {
  const pct = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  const thresholdPct = Math.max(0, Math.min(1, threshold)) * 100;
  const state = value === null ? "unknown" : value >= threshold ? "confident" : "below";
  return (
    <div className={`meter meter--${state}`} aria-label={`${label}: ${value === null ? "unknown" : formatPct(value)}`}>
      <div className="meter__head">
        <span className="meter__label">{label}</span>
        <span className="meter__value">{value === null ? "—" : formatPct(value)}</span>
      </div>
      <div className="meter__track" role="img" aria-hidden="true">
        <div className="meter__fill" style={{ width: `${pct}%` }} />
        <div className="meter__threshold" style={{ left: `${thresholdPct}%` }} title={`Answer threshold ${formatPct(threshold)}`} />
      </div>
      <div className="meter__foot">
        <span>0%</span>
        <span className="meter__target">answer ≥ {formatPct(threshold)}</span>
        <span>100%</span>
      </div>
    </div>
  );
}

export function PriorityDot({ priority }: { priority: "normal" | "high" | "urgent" }) {
  return <span className={`priority priority--${priority}`} title={`${priority} priority`} aria-label={`${priority} priority`} />;
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="spinner" role="status" aria-label={label}>
      <span className="spinner__dot" />
      <span className="spinner__dot" />
      <span className="spinner__dot" />
    </span>
  );
}

export function ErrorBanner({ title, message, onRetry }: { title: string; message: string; onRetry?: () => void }) {
  return (
    <div className="banner banner--error" role="alert">
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      {onRetry ? (
        <button type="button" className="button button--ghost" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <div className="empty__glyph" aria-hidden="true">
        ◎
      </div>
      <p className="empty__title">{title}</p>
      {hint ? <p className="empty__hint">{hint}</p> : null}
    </div>
  );
}
