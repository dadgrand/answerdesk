/** Formatting helpers shared by the console screens (pure functions, unit-tested). */

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 1) return "<1 ms";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

export function formatUsd(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "$0";
  if (value < 0.0001) return `$${value.toExponential(2)}`;
  return `$${value.toFixed(digits)}`;
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.round(value).toLocaleString("en-US");
}

export function formatTokens(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 1000) return String(Math.round(value));
  return `${(value / 1000).toFixed(1)}k`;
}

/** mm:ss for the SLA timer. */
export function formatClock(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

export function languageLabel(code: string | null | undefined): string {
  if (!code) return "??";
  const map: Record<string, string> = {
    ru: "RU",
    tg: "TG",
    uz: "UZ",
    ky: "KY",
    tr: "TR",
    zh: "ZH",
    en: "EN",
  };
  return map[code.slice(0, 2).toLowerCase()] ?? code.slice(0, 2).toUpperCase();
}

export function languageName(code: string | null | undefined): string {
  const map: Record<string, string> = {
    ru: "Russian",
    tg: "Tajik",
    uz: "Uzbek",
    ky: "Kyrgyz",
    tr: "Turkish",
    zh: "Chinese",
    en: "English",
  };
  return map[(code ?? "").slice(0, 2).toLowerCase()] ?? "Unknown";
}

export function shortId(id: string, front = 6, back = 4): string {
  if (id.length <= front + back + 1) return id;
  return `${id.slice(0, front)}…${id.slice(-back)}`;
}

export function humanizeReason(code: string | null | undefined): string {
  if (!code) return "Unknown";
  const map: Record<string, string> = {
    ANSWERED: "Answered from verified evidence",
    CLARIFY_AMBIGUOUS: "Question is ambiguous — clarify",
    NO_EVIDENCE: "No passage passed the evidence threshold",
    LOW_CONFIDENCE: "Below the calibrated confidence band",
    VERIFIER_REJECT: "Every claim failed verification",
    CITATION_FAILED: "Citation or numeric check failed",
    HIGH_RISK_INTENT: "High-risk intent (account/legal) requires a human",
    PII_DETECTED: "Personal data detected — refused automatically",
    INJECTION_SUSPECTED: "Prompt-injection attempt suspected",
    OUT_OF_SCOPE: "Outside the knowledge base",
    UNSUPPORTED_LANGUAGE: "Unsupported language",
    CORPUS_MISSING: "Corpus unavailable",
    PROVIDER_ERROR: "Model provider failed",
    STALE_CALIBRATION: "Calibration artifact is stale",
    OPERATOR_REQUEST: "Operator requested escalation",
    USER_REQUESTED_HUMAN: "Customer asked for a human",
  };
  return map[code] ?? code.replaceAll("_", " ").toLowerCase();
}

export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return "—";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}
