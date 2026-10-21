/**
 * Prometheus text-format parser and the analytics derivations.
 *
 * The analytics page reads `GET /metrics` from the real service (NFR-OBS-02: "the console
 * analytics view reads real API series"). Parsing the exposition ourselves keeps the
 * console dependency-free and, more importantly, makes every number on the page traceable
 * to a metric family — a test feeds a recorded exposition and asserts the KPIs.
 */

export interface Sample {
  labels: Record<string, string>;
  value: number;
}

export type MetricFamily = Map<string, Sample[]>;

/** Parse the text exposition format (`# HELP`/`# TYPE` comments are ignored). */
export function parsePrometheus(text: string): MetricFamily {
  const families: MetricFamily = new Map();
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const space = line.lastIndexOf(" ");
    if (space === -1) continue;
    const head = line.slice(0, space);
    const value = Number(line.slice(space + 1));
    if (!Number.isFinite(value)) continue;
    let name = head;
    const labels: Record<string, string> = {};
    const brace = head.indexOf("{");
    if (brace !== -1) {
      name = head.slice(0, brace);
      const rawLabels = head.slice(brace + 1, head.lastIndexOf("}"));
      for (const pair of splitLabels(rawLabels)) {
        const eq = pair.indexOf("=");
        if (eq === -1) continue;
        const key = pair.slice(0, eq).trim();
        let val = pair.slice(eq + 1).trim();
        if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1);
        labels[key] = val.replace(/\\"/g, '"').replace(/\\n/g, "\n");
      }
    }
    const bucket = families.get(name) ?? [];
    bucket.push({ labels, value });
    families.set(name, bucket);
  }
  return families;
}

function splitLabels(raw: string): string[] {
  const parts: string[] = [];
  let current = "";
  let inQuotes = false;
  let escaped = false;
  for (const char of raw) {
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (char === "\\") {
      current += char;
      escaped = true;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      current += char;
      continue;
    }
    if (char === "," && !inQuotes) {
      parts.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  if (current.trim()) parts.push(current);
  return parts;
}

export function samples(family: MetricFamily, name: string): Sample[] {
  return family.get(name) ?? [];
}

export function sum(family: MetricFamily, name: string, match: Record<string, string> = {}): number {
  return samples(family, name)
    .filter((sample) => Object.entries(match).every(([key, value]) => sample.labels[key] === value))
    .reduce((total, sample) => total + sample.value, 0);
}

export interface Bucket {
  le: number;
  count: number;
}

/** Extract cumulative histogram buckets, sorted by `le` (the implicit +Inf is included). */
export function histogramBuckets(
  family: MetricFamily,
  name: string,
  match: Record<string, string> = {},
): { buckets: Bucket[]; count: number; sum: number } {
  const matching = samples(family, `${name}_bucket`).filter((sample) =>
    Object.entries(match).every(([key, value]) => sample.labels[key] === value),
  );
  const buckets: Bucket[] = [];
  for (const sample of matching) {
    const le = sample.labels.le;
    if (le === undefined) continue;
    const bound = le === "+Inf" ? Number.POSITIVE_INFINITY : Number(le);
    if (!Number.isFinite(bound) && le !== "+Inf") continue;
    buckets.push({ le: bound, count: sample.value });
  }
  buckets.sort((a, b) => a.le - b.le);
  const count = sum(family, `${name}_count`, match);
  const total = sum(family, `${name}_sum`, match);
  return { buckets, count, sum: total };
}

/** Prometheus-style `histogram_quantile` over cumulative buckets. */
export function histogramQuantile(
  buckets: Bucket[],
  q: number,
  count?: number,
): number | null {
  const total = count ?? buckets[buckets.length - 1]?.count ?? 0;
  if (!total || q <= 0 || q > 1) return null;
  const rank = q * total;
  let previousBound = 0;
  let previousCount = 0;
  for (const bucket of buckets) {
    if (bucket.count >= rank) {
      if (!Number.isFinite(bucket.le)) return previousBound || null;
      const span = bucket.count - previousCount;
      if (span <= 0) return bucket.le;
      return previousBound + ((rank - previousCount) / span) * (bucket.le - previousBound);
    }
    previousBound = Number.isFinite(bucket.le) ? bucket.le : previousBound;
    previousCount = bucket.count;
  }
  return previousBound || null;
}

export interface DecisionCount {
  action: string;
  count: number;
}

export interface CacheCount {
  kind: string;
  count: number;
}

export interface ModelCost {
  model: string;
  usd: number;
}

export interface LatencyRow {
  stage: string;
  p95: number;
}

export interface Kpis {
  requests: number;
  answers: number;
  escalations: number;
  clarifications: number;
  refusals: number;
  latencyP50: number | null;
  latencyP95: number | null;
  usdPerAnswer: number | null;
  usdTotal: number;
  escalationRate: number | null;
  cacheHitRate: number | null;
  cacheHits: number;
  cacheMisses: number;
  cacheBypasses: number;
  tokensIn: number;
  tokensOut: number;
  latencyDistribution: Bucket[];
  decisions: DecisionCount[];
  cacheBreakdown: CacheCount[];
  models: ModelCost[];
  stageLatency: LatencyRow[];
  errorCount: number;
}

const ACTION_LABELS: Record<string, string> = {
  answer: "answered",
  clarify: "clarified",
  escalate: "escalated",
  refuse: "refused",
};

export function actionLabel(action: string): string {
  return ACTION_LABELS[action] ?? action;
}

/** Derive every analytics KPI from one scrape. */
export function deriveKpis(family: MetricFamily): Kpis {
  const decisions: DecisionCount[] = samples(family, "answerdesk_decisions_total").map((sample) => ({
    action: sample.labels.action ?? "unknown",
    count: sample.value,
  }));
  const byAction = (action: string) =>
    decisions.filter((row) => row.action === action).reduce((total, row) => total + row.count, 0);
  const answers = byAction("answer");
  const escalations = byAction("escalate");
  const clarifications = byAction("clarify");
  const refusals = byAction("refuse");
  const totalDecisions = answers + escalations + clarifications + refusals;

  const stageTotal = histogramBuckets(family, "answerdesk_stage_duration_seconds", { stage: "total" });
  const httpTotal = histogramBuckets(family, "answerdesk_http_request_duration_seconds", {
    route: "/v1/chat",
  });
  const chosen = stageTotal.count > 0 ? stageTotal : httpTotal;

  const latencyP50 = histogramQuantile(chosen.buckets, 0.5, chosen.count);
  const latencyP95 = histogramQuantile(chosen.buckets, 0.95, chosen.count);

  const usdTotal = samples(family, "answerdesk_model_cost_usd_total").reduce(
    (total, sample) => total + sample.value,
    0,
  );
  const usdPerAnswer = answers > 0 ? usdTotal / answers : null;

  const cacheHits = samples(family, "answerdesk_cache_hits_total").reduce(
    (total, sample) => total + sample.value,
    0,
  );
  const cacheMisses = sum(family, "answerdesk_cache_misses_total");
  const cacheBypasses = sum(family, "answerdesk_cache_bypass_total");
  const cacheLookups = cacheHits + cacheMisses;
  const cacheHitRate = cacheLookups > 0 ? cacheHits / cacheLookups : null;

  const cacheBreakdown = samples(family, "answerdesk_cache_hits_total").map((sample) => ({
    kind: sample.labels.kind ?? "exact",
    count: sample.value,
  }));

  const tokensIn = sum(family, "answerdesk_model_tokens_total", { kind: "input" });
  const tokensOut = sum(family, "answerdesk_model_tokens_total", { kind: "output" });

  const models = samples(family, "answerdesk_model_cost_usd_total").map((sample) => ({
    model: sample.labels.model ?? "unknown",
    usd: sample.value,
  }));

  const stageLatency: LatencyRow[] = [];
  for (const stage of ["analysis", "retrieval", "rerank", "generation", "verify", "policy", "cache"]) {
    const histogram = histogramBuckets(family, "answerdesk_stage_duration_seconds", { stage });
    const p95 = histogramQuantile(histogram.buckets, 0.95, histogram.count);
    if (p95 !== null) stageLatency.push({ stage, p95 });
  }

  const errorCount = samples(family, "answerdesk_http_errors_total").reduce(
    (total, sample) => total + sample.value,
    0,
  );

  return {
    requests: sum(family, "answerdesk_http_requests_total"),
    answers,
    escalations,
    clarifications,
    refusals,
    latencyP50,
    latencyP95,
    usdPerAnswer,
    usdTotal,
    escalationRate: totalDecisions > 0 ? escalations / totalDecisions : null,
    cacheHitRate,
    cacheHits,
    cacheMisses,
    cacheBypasses,
    tokensIn,
    tokensOut,
    latencyDistribution: chosen.buckets,
    decisions,
    cacheBreakdown,
    models,
    stageLatency,
    errorCount,
  };
}
