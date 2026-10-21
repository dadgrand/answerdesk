/**
 * Synthetic fixtures for MSW mode (offline development, tests, screenshots fallback).
 *
 * Shapes are copied from the real service responses (`docs/api.md` §3/§5/§6) so the mock
 * exercises the same code paths as production: the same SSE frames, the same trace
 * document, the same Prometheus families. Data is fictional (requirements C-10).
 */

import type {
  Answer,
  Citation,
  Claim,
  Decision,
  HealthResponse,
  InfoResponse,
  Passage,
  RetrievalEvent,
  TraceLine,
  Usage,
} from "../types";

export function answerable(message: string): boolean {
  return /обслуживание|card servicing|annual fee|тариф/i.test(message);
}

export const MOCK_INFO: InfoResponse = {
  service: "answerdesk",
  version: "0.1.0",
  profile: "offline",
  fake: true,
  corpus_version: "corpus-demo-6cb29bf77b57",
  index_version: "index-87424537ecf8b33c",
  chunks: 8,
  model: "fake-model",
  preset: "fake",
  cache_enabled: true,
  admin_enabled: false,
};

export const MOCK_HEALTH: HealthResponse = {
  status: "ok",
  service: "answerdesk",
  version: "0.1.0",
  schema_version: "1.0.0",
  uptime_s: 1843.2,
  checks: {
    index: { status: "ok", detail: "8 chunks (hashing-64)", version: "index-87424537ecf8b33c" },
    calibration: { status: "ok", detail: "age 0.2d", version: "cal-f70b9bf8edb0" },
    cache: { status: "ok", detail: "12 entries" },
    database: { status: "ok", detail: "38 sessions, 64 turns" },
    provider: { status: "ok", detail: "preset=fake api_key=absent" },
  },
};

const TRACE_REF = "cal-f70b9bf8edb0";
const CORPUS = "corpus-demo-6cb29bf77b57";
const INDEX = "index-87424537ecf8b33c";
const PROMPT = "grounded-answer-1.0.0";

export const ANSWER_TEXT =
  "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. [1] " +
  "Плата списывается в первый день каждого месяца; при оплате за год действует скидка 10 %. [2]";

export const ANSWER_CITATIONS: Citation[] = [
  {
    citation_id: 1,
    chunk_id: "chunk-card-fee-01",
    doc_id: "doc-card-fee",
    doc_title: "Тарифы по карте «Вектор»",
    heading_path: ["Тарифы", "Обслуживание"],
    quote: "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
    start_char: 0,
    end_char: 63,
    doc_hash: "8f14e45fceea167a5a36dedd4bea2543a1b2c3d4e5f60718293a4b5c6d7e8f90",
  },
  {
    citation_id: 2,
    chunk_id: "chunk-card-fee-02",
    doc_id: "doc-card-fee",
    doc_title: "Тарифы по карте «Вектор»",
    heading_path: ["Тарифы", "Периодичность списания"],
    quote:
      "Плата списывается в первый день каждого месяца. При оплате за год действует скидка 10 %.",
    start_char: 72,
    end_char: 160,
    doc_hash: "7c9e686f3c2a1b0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d",
  },
];

export const ANSWER_CLAIMS: Claim[] = [
  {
    claim_id: "c1",
    text: "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
    citation_ids: [1],
    support: "supported",
  },
  {
    claim_id: "c2",
    text: "Плата списывается в первый день каждого месяца; при оплате за год действует скидка 10 %.",
    citation_ids: [2],
    support: "supported",
  },
];

export const ANSWER_PASSAGES: Passage[] = [
  {
    chunk_id: "chunk-card-fee-01",
    doc_id: "doc-card-fee",
    doc_title: "Тарифы по карте «Вектор»",
    heading_path: ["Тарифы", "Обслуживание"],
    breadcrumb: "Тарифы по карте «Вектор» > Тарифы > Обслуживание",
    channel: "bm25",
    stage: "pack",
    score: 0.0328,
    rank: 1,
    snippet: "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. Плата списывается…",
  },
  {
    chunk_id: "chunk-card-fee-02",
    doc_id: "doc-card-fee",
    doc_title: "Тарифы по карте «Вектор»",
    heading_path: ["Тарифы", "Периодичность списания"],
    breadcrumb: "Тарифы по карте «Вектор» > Тарифы > Периодичность списания",
    channel: "dense",
    stage: "pack",
    score: 0.0301,
    rank: 2,
    snippet: "Плата списывается в первый день каждого месяца. При оплате за год действует скидка…",
  },
  {
    chunk_id: "chunk-transfer-01",
    doc_id: "doc-transfer",
    doc_title: "Лимиты переводов",
    heading_path: ["Переводы", "Суточные лимиты"],
    breadcrumb: "Лимиты переводов > Переводы > Суточные лимиты",
    channel: "bm25",
    stage: "pack",
    score: 0.0164,
    rank: 3,
    snippet: "Суточный лимит переводов — 150 000 рублей, месячный — 1 500 000 рублей.",
  },
  {
    chunk_id: "chunk-refund-01",
    doc_id: "doc-refund",
    doc_title: "Возврат средств",
    heading_path: ["Возвраты", "Сроки"],
    breadcrumb: "Возврат средств > Возвраты > Сроки",
    channel: "dense",
    stage: "pack",
    score: 0.0142,
    rank: 4,
    snippet: "Возврат обрабатывается в течение 14 дней с момента одобрения.",
  },
  {
    chunk_id: "chunk-replace-01",
    doc_id: "doc-replace",
    doc_title: "Перевыпуск карты",
    heading_path: ["Карта", "Перевыпуск"],
    breadcrumb: "Перевыпуск карты > Карта > Перевыпуск",
    channel: "bm25",
    stage: "pack",
    score: 0.0129,
    rank: 5,
    snippet: "Перевыпуск карты занимает 5 рабочих дней; стоимость — 300 рублей.",
  },
];

export const ANSWER_USAGE: Usage = {
  input_tokens: 812,
  output_tokens: 96,
  usd: 0.00442,
  model: "gpt-4o-mini",
  prompt_version: PROMPT,
};

export const ANSWER_DECISION: Decision = {
  action: "answer",
  reason_code: "ANSWERED",
  score: 0.983,
  alpha: 0.05,
  threshold_ref: TRACE_REF,
};

export function buildAnswer(options: { traceId: string; cacheHit: boolean; similarity?: number | null }): Answer {
  return {
    answer_id: "ans-9a8b7c6d5e4f3a2b",
    trace_id: options.traceId,
    language: "ru",
    text: ANSWER_TEXT,
    claims: ANSWER_CLAIMS,
    citations: ANSWER_CITATIONS,
    decision: ANSWER_DECISION,
    confidence: { p_correct: ANSWER_DECISION.score, calibrator_version: TRACE_REF },
    cache: {
      status: options.cacheHit ? "hit" : "miss",
      entry_id: options.cacheHit ? "ce-51c3a20f9d0b7e44" : null,
      similarity: options.cacheHit ? (options.similarity ?? 1) : null,
    },
    usage: options.cacheHit
      ? { input_tokens: 0, output_tokens: 0, usd: 0, model: "gpt-4o-mini", prompt_version: PROMPT }
      : ANSWER_USAGE,
    provenance: {
      corpus_version: CORPUS,
      index_version: INDEX,
      prompt_version: PROMPT,
      verifier_version: "lexical-1.0.0",
      scorer_version: "overlap-1.0.0",
    },
  };
}

export function retrievalEvent(traceId: string, cacheHit: boolean): RetrievalEvent {
  return {
    event: "retrieval",
    trace_id: traceId,
    query: "обслуживание карты",
    normalized_query: "обслуживание карты",
    language: "ru",
    passages: ANSWER_PASSAGES,
    channels: { bm25: 8, dense: 8 },
    fusion: { fuser: "rrf", params: { k: 60 } },
    candidates: 16,
    dropped_duplicates: 1,
    cache: {
      status: cacheHit ? "hit" : "miss",
      entry_id: cacheHit ? "ce-51c3a20f9d0b7e44" : null,
      similarity: cacheHit ? 1 : null,
      reason: cacheHit ? "exact match" : "no candidates",
    },
    timings_ms: { cache: 0.08, retrieval: 1.42, generation: 0, total: 1.5 },
  };
}

export function buildAnsweredTrace(traceId: string, cacheHit: boolean): TraceLine {
  return {
    kind: "turn",
    schema_version: "1.1.0",
    trace_id: traceId,
    session_id: "s-console-1042",
    turn_id: "t-7c1e2d3f4a5b6c7d",
    query: "обслуживание карты",
    query_pii: { redacted: 0, kinds: [] },
    language: "ru",
    intent_bucket: "informational",
    decision: ANSWER_DECISION,
    p_hat: ANSWER_DECISION.score,
    confidence: ANSWER_DECISION.score,
    retrieval: {
      query: "обслуживание карты",
      normalized_query: "обслуживание карты",
      language: "ru",
      packed: ANSWER_PASSAGES.map((passage) => ({
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
      })),
      fusion: { fuser: "rrf", params: { k: 60 } },
      dropped_duplicates: 1,
    },
    retrieval_scores: ANSWER_PASSAGES.map((passage) => ({
      chunk_id: passage.chunk_id,
      doc_id: passage.doc_id,
      channel: passage.channel,
      score: passage.score,
      rank: passage.rank,
      stage: passage.stage,
    })),
    fusion: { fuser: "rrf", params: { k: 60 } },
    cache: {
      status: cacheHit ? "hit" : "miss",
      entry_id: cacheHit ? "ce-51c3a20f9d0b7e44" : null,
      similarity: cacheHit ? 1 : null,
      reason: cacheHit ? "exact match" : "no candidates",
    },
    cache_status: cacheHit ? "hit" : "miss",
    history: { follow_up: false, turns_used: 1, cues: [] },
    generation: {
      repaired: false,
      repair_attempts: 0,
      dropped_claims: 0,
      numeric_violations_pre_repair: 0,
      verifier_version: "lexical-1.0.0",
      language_notice: false,
    },
    escalation: null,
    usage: cacheHit
      ? { input_tokens: 0, output_tokens: 0, usd: 0, model: "gpt-4o-mini", prompt_version: PROMPT }
      : ANSWER_USAGE,
    timings_ms: { history: 0.04, cache: 0.08, retrieval: 1.42, generation: 214.6, verify: 0.9, policy: 0.3, total: 217.5 },
    stage_latency_ms: { history: 0.04, cache: 0.08, retrieval: 1.42, generation: 214.6, verify: 0.9, policy: 0.3, total: 217.5 },
    corpus_version: CORPUS,
    index_version: INDEX,
    prompt_version: PROMPT,
    service: {
      profile: "offline",
      fake: true,
      model: "gpt-4o-mini",
      preset: "fake",
      calibration: { source: "reports/policy-smoke/calibration-artifact.json", threshold_ref: TRACE_REF, stale: false, age_days: 0.2 },
      scorer: { name: "overlap-baseline", version: "overlap-1.0.0" },
    },
    provider_redactions: 0,
    error: null,
    created_at: new Date().toISOString(),
  };
}

export const ESCALATION_REASON = "LOW_CONFIDENCE";

export function buildEscalatedTrace(traceId: string, query: string, reason: string): TraceLine {
  const answerTrace = buildAnsweredTrace(traceId, false);
  return {
    ...answerTrace,
    query,
    language: "tg",
    decision: {
      action: "escalate",
      reason_code: reason,
      score: 0.62,
      alpha: 0.05,
      threshold_ref: TRACE_REF,
    },
    p_hat: 0.62,
    confidence: 0.62,
    generation: {
      repaired: false,
      repair_attempts: 0,
      dropped_claims: 0,
      numeric_violations_pre_repair: 0,
      verifier_version: "lexical-1.0.0",
      language_notice: false,
    },
    escalation: { ticket_id: "tk-3f2a9b1c8d7e6f50", reason },
    cache: { status: "miss", entry_id: null, similarity: null, reason: "no candidates" },
    cache_status: "miss",
    timings_ms: { history: 0.03, cache: 0.06, retrieval: 1.1, generation: 188.4, verify: 0.4, policy: 0.2, total: 190.2 },
    stage_latency_ms: { history: 0.03, cache: 0.06, retrieval: 1.1, generation: 188.4, verify: 0.4, policy: 0.2, total: 190.2 },
  };
}

export const MOCK_TICKET = {
  ticket_id: "tk-3f2a9b1c8d7e6f50",
  trace_id: "tr-demo-escalated",
  session_id: "s-console-1043",
  status: "open",
  priority: "high" as const,
  reason: "Customer asked for a human",
  note: "Console escalation",
  created_at: new Date().toISOString(),
};

export function escalationAnswer(traceId: string, reason: string): Answer {
  return {
    answer_id: "ans-escalated",
    trace_id: traceId,
    language: "tg",
    text: "",
    claims: [],
    citations: [],
    decision: {
      action: "escalate",
      reason_code: reason,
      score: 0.62,
      alpha: 0.05,
      threshold_ref: TRACE_REF,
    },
    confidence: { p_correct: 0.62, calibrator_version: TRACE_REF },
    cache: { status: "miss", entry_id: null, similarity: null },
    usage: { input_tokens: 640, output_tokens: 0, usd: 0.0031, model: "gpt-4o-mini", prompt_version: PROMPT },
    provenance: {
      corpus_version: CORPUS,
      index_version: INDEX,
      prompt_version: PROMPT,
      verifier_version: "lexical-1.0.0",
      scorer_version: "overlap-1.0.0",
    },
  };
}

const BUCKETS = ["0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10"];

/** Build a cumulative histogram block the way the Python registry renders it. */
export function histogram(
  name: string,
  labels: Record<string, string>,
  counts: number[],
  total: number,
  sumValue: number,
): string[] {
  const lines: string[] = [];
  BUCKETS.forEach((bound, index) => {
    const labelText = Object.entries({ ...labels, le: bound })
      .map(([key, value]) => `${key}="${value}"`)
      .join(",");
    lines.push(`${name}_bucket{${labelText}} ${counts[index] ?? 0}`);
  });
  const infLabels = Object.entries({ ...labels, le: "+Inf" })
    .map(([key, value]) => `${key}="${value}"`)
    .join(",");
  lines.push(`${name}_bucket{${infLabels}} ${total}`);
  const labelText = Object.entries(labels)
    .map(([key, value]) => `${key}="${value}"`)
    .join(",");
  const suffix = labelText ? `{${labelText}}` : "";
  lines.push(`${name}_sum${suffix} ${sumValue}`);
  lines.push(`${name}_count${suffix} ${total}`);
  return lines;
}

/** A full scrape with the same families the service exposes (docs/ops.md §5.1). */
export function mockMetrics(): string {
  const lines: string[] = [
    "# HELP answerdesk_http_requests_total HTTP requests handled, by method, route template and status code.",
    "# TYPE answerdesk_http_requests_total counter",
    'answerdesk_http_requests_total{method="POST",route="/v1/chat",status="200"} 64',
    'answerdesk_http_requests_total{method="GET",route="/v1/traces/{trace_id}",status="200"} 9',
    'answerdesk_http_requests_total{method="GET",route="/healthz",status="200"} 12',
    'answerdesk_http_requests_total{method="GET",route="/metrics",status="200"} 8',
    'answerdesk_http_requests_total{method="POST",route="/v1/feedback",status="201"} 14',
    'answerdesk_http_requests_total{method="POST",route="/v1/escalations",status="201"} 6',
    "# HELP answerdesk_http_request_duration_seconds Wall-clock time from request arrival to final byte.",
    "# TYPE answerdesk_http_request_duration_seconds histogram",
    ...histogram(
      "answerdesk_http_request_duration_seconds",
      { method: "POST", route: "/v1/chat" },
      [0, 1, 4, 9, 18, 41, 58, 62, 64, 64, 64],
      64,
      11.52,
    ),
    "# HELP answerdesk_http_errors_total Error responses by stable error code (docs/api.md taxonomy).",
    "# TYPE answerdesk_http_errors_total counter",
    'answerdesk_http_errors_total{code="rate_limited"} 2',
    'answerdesk_http_errors_total{code="provider_error"} 1',
    "# HELP answerdesk_stage_duration_seconds Per-stage latency.",
    "# TYPE answerdesk_stage_duration_seconds histogram",
    ...histogram("answerdesk_stage_duration_seconds", { stage: "total" }, [0, 1, 5, 11, 22, 47, 60, 63, 64, 64, 64], 64, 11.52),
    ...histogram("answerdesk_stage_duration_seconds", { stage: "retrieval" }, [44, 58, 63, 64, 64, 64, 64, 64, 64, 64, 64], 64, 0.384),
    ...histogram("answerdesk_stage_duration_seconds", { stage: "generation" }, [0, 0, 2, 7, 15, 38, 55, 62, 64, 64, 64], 64, 10.88),
    ...histogram("answerdesk_stage_duration_seconds", { stage: "verify" }, [30, 52, 61, 64, 64, 64, 64, 64, 64, 64, 64], 64, 0.192),
    ...histogram("answerdesk_stage_duration_seconds", { stage: "policy" }, [38, 57, 63, 64, 64, 64, 64, 64, 64, 64, 64], 64, 0.128),
    ...histogram("answerdesk_stage_duration_seconds", { stage: "cache" }, [60, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], 64, 0.032),
    "# HELP answerdesk_decisions_total Policy decisions by action and reason code.",
    "# TYPE answerdesk_decisions_total counter",
    'answerdesk_decisions_total{action="answer",reason="ANSWERED"} 41',
    'answerdesk_decisions_total{action="escalate",reason="LOW_CONFIDENCE"} 11',
    'answerdesk_decisions_total{action="escalate",reason="NO_EVIDENCE"} 4',
    'answerdesk_decisions_total{action="clarify",reason="CLARIFY_AMBIGUOUS"} 6',
    'answerdesk_decisions_total{action="refuse",reason="PII_DETECTED"} 2',
    "# HELP answerdesk_escalations_total Turns escalated to a human, by reason code.",
    "# TYPE answerdesk_escalations_total counter",
    'answerdesk_escalations_total{reason="LOW_CONFIDENCE"} 11',
    'answerdesk_escalations_total{reason="NO_EVIDENCE"} 4',
    "# HELP answerdesk_refusals_total Turns refused, by reason code.",
    "# TYPE answerdesk_refusals_total counter",
    'answerdesk_refusals_total{reason="PII_DETECTED"} 2',
    "# HELP answerdesk_cache_hits_total Semantic-cache hits, by kind.",
    "# TYPE answerdesk_cache_hits_total counter",
    'answerdesk_cache_hits_total{kind="exact"} 9',
    'answerdesk_cache_hits_total{kind="semantic"} 4',
    'answerdesk_cache_hits_total{kind="negative"} 0',
    "# HELP answerdesk_cache_misses_total Semantic-cache misses.",
    "# TYPE answerdesk_cache_misses_total counter",
    "answerdesk_cache_misses_total 22",
    "# HELP answerdesk_cache_bypass_total Cache lookups bypassed.",
    "# TYPE answerdesk_cache_bypass_total counter",
    "answerdesk_cache_bypass_total 3",
    "# HELP answerdesk_cache_wrong_hits_total Cache hits later labelled wrong.",
    "# TYPE answerdesk_cache_wrong_hits_total counter",
    "answerdesk_cache_wrong_hits_total 0",
    "# HELP answerdesk_cache_entries Entries currently held by the semantic cache.",
    "# TYPE answerdesk_cache_entries gauge",
    "answerdesk_cache_entries 12",
    "# HELP answerdesk_model_tokens_total Provider tokens billed.",
    "# TYPE answerdesk_model_tokens_total counter",
    'answerdesk_model_tokens_total{kind="input"} 51968',
    'answerdesk_model_tokens_total{kind="output"} 6144',
    "# HELP answerdesk_model_cost_usd_total Provider spend in USD, by model.",
    "# TYPE answerdesk_model_cost_usd_total counter",
    'answerdesk_model_cost_usd_total{model="gpt-4o-mini"} 0.1834',
    "# HELP answerdesk_sessions_created_total Sessions created by the service.",
    "# TYPE answerdesk_sessions_created_total counter",
    "answerdesk_sessions_created_total 38",
    "# HELP answerdesk_feedback_total Feedback submissions by rating.",
    "# TYPE answerdesk_feedback_total counter",
    'answerdesk_feedback_total{rating="up"} 12',
    'answerdesk_feedback_total{rating="down"} 2',
    "# HELP answerdesk_rate_limited_total Requests rejected by the per-client rate limiter.",
    "# TYPE answerdesk_rate_limited_total counter",
    "answerdesk_rate_limited_total 2",
    "# HELP answerdesk_traces_written_total JSONL trace lines written.",
    "# TYPE answerdesk_traces_written_total counter",
    'answerdesk_traces_written_total{outcome="ok"} 64',
    'answerdesk_traces_written_total{outcome="error"} 1',
    "# HELP answerdesk_idempotent_replays_total Requests served from the idempotency store.",
    "# TYPE answerdesk_idempotent_replays_total counter",
    "answerdesk_idempotent_replays_total 5",
  ];
  return `${lines.join("\n")}\n`;
}
