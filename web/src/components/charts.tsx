/** Dependency-free chart primitives (SVG/divs) with accessible labels and titles. */

import type { ReactNode } from "react";

export function KpiCard({
  label,
  value,
  hint,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  icon?: ReactNode;
}) {
  return (
    <article className={`kpi kpi--${tone}`}>
      <header className="kpi__head">
        <span className="kpi__label">{label}</span>
        {icon ? <span className="kpi__icon" aria-hidden="true">{icon}</span> : null}
      </header>
      <p className="kpi__value">{value}</p>
      {hint ? <p className="kpi__hint">{hint}</p> : null}
    </article>
  );
}

export interface BarDatum {
  label: string;
  value: number;
  display?: string;
  tone?: string;
}

export function BarList({
  items,
  ariaLabel,
  emptyHint = "No data yet",
}: {
  items: BarDatum[];
  ariaLabel: string;
  emptyHint?: string;
}) {
  const max = items.reduce((top, item) => Math.max(top, item.value), 0);
  if (items.length === 0) return <p className="chart__empty">{emptyHint}</p>;
  return (
    <ul className="barlist" aria-label={ariaLabel}>
      {items.map((item) => (
        <li key={item.label} className="barlist__row">
          <span className="barlist__label">{item.label}</span>
          <span className="barlist__track">
            <span
              className={`barlist__fill barlist__fill--${item.tone ?? "accent"}`}
              style={{ width: `${max > 0 ? Math.max(2, (item.value / max) * 100) : 0}%` }}
            />
          </span>
          <span className="barlist__value">{item.display ?? item.value.toFixed(0)}</span>
        </li>
      ))}
    </ul>
  );
}

export function StackedBar({
  segments,
  ariaLabel,
}: {
  segments: BarDatum[];
  ariaLabel: string;
}) {
  const total = segments.reduce((sum, segment) => sum + segment.value, 0);
  if (total === 0) return <p className="chart__empty">No decisions recorded yet.</p>;
  return (
    <div className="stacked" role="img" aria-label={ariaLabel}>
      <div className="stacked__bar">
        {segments.map((segment) => (
          <span
            key={segment.label}
            className={`stacked__seg stacked__seg--${segment.tone ?? "accent"}`}
            style={{ width: `${(segment.value / total) * 100}%` }}
            title={`${segment.label}: ${segment.value}`}
          />
        ))}
      </div>
      <ul className="legend">
        {segments.map((segment) => (
          <li key={segment.label} className="legend__item">
            <span className={`legend__dot stacked__seg--${segment.tone ?? "accent"}`} aria-hidden="true" />
            <span className="legend__label">{segment.label}</span>
            <span className="legend__value">
              {segment.value} · {((segment.value / total) * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export interface HistogramBucket {
  le: number;
  count: number;
}

/** Latency distribution: cumulative Prometheus buckets turned into per-bucket counts. */
export function Histogram({
  buckets,
  format,
  ariaLabel,
}: {
  buckets: HistogramBucket[];
  format: (value: number) => string;
  ariaLabel: string;
}) {
  const finite = buckets.filter((bucket) => Number.isFinite(bucket.le));
  const counts = finite.map((bucket, index) => Math.max(0, bucket.count - (finite[index - 1]?.count ?? 0)));
  const max = counts.reduce((top, count) => Math.max(top, count), 0);
  if (finite.length === 0 || max === 0) return <p className="chart__empty">No latency samples yet.</p>;
  const width = 640;
  const height = 180;
  const padding = { top: 12, right: 8, bottom: 34, left: 8 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const barWidth = innerWidth / counts.length;
  return (
    <figure className="histogram">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
        {[0.25, 0.5, 0.75, 1].map((fraction) => (
          <line
            key={fraction}
            x1={padding.left}
            x2={width - padding.right}
            y1={padding.top + innerHeight * (1 - fraction)}
            y2={padding.top + innerHeight * (1 - fraction)}
            className="histogram__grid"
          />
        ))}
        {counts.map((count, index) => {
          const heightPx = max > 0 ? (count / max) * innerHeight : 0;
          return (
            <g key={finite[index]?.le}>
              <rect
                x={padding.left + index * barWidth + 2}
                y={padding.top + innerHeight - heightPx}
                width={Math.max(1, barWidth - 4)}
                height={heightPx}
                className="histogram__bar"
                rx={3}
              >
                <title>{`≤ ${format(finite[index]?.le ?? 0)}: ${count} samples`}</title>
              </rect>
              {index % 2 === 0 ? (
                <text
                  x={padding.left + index * barWidth + barWidth / 2}
                  y={height - 12}
                  textAnchor="middle"
                  className="histogram__label"
                >
                  {format(finite[index]?.le ?? 0)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

export function RingGauge({
  value,
  label,
  display,
  tone = "good",
}: {
  value: number | null;
  label: string;
  display: string;
  tone?: "good" | "warn" | "bad";
}) {
  const ratio = value === null ? 0 : Math.max(0, Math.min(1, value));
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  return (
    <div className={`gauge gauge--${tone}`} role="img" aria-label={`${label}: ${display}`}>
      <svg viewBox="0 0 120 120" width="140" height="140">
        <circle cx="60" cy="60" r={radius} className="gauge__track" fill="none" strokeWidth="9" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          className="gauge__value"
          fill="none"
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - ratio)}
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="gauge__center">
        <span className="gauge__display">{display}</span>
        <span className="gauge__label">{label}</span>
      </div>
    </div>
  );
}
