/**
 * Wire types for the answerdesk v1 API (docs/api.md).
 *
 * These mirror the service's pydantic contracts but only the fields the console renders.
 * Keeping them in one file makes a schema drift visible in review: if `docs/api.md`
 * changes, this is the only client file that has to follow.
 */

export type Action = "answer" | "clarify" | "escalate" | "refuse";
export type CacheStatus = "hit" | "miss" | "bypassed";
export type Priority = "normal" | "high" | "urgent";
export type IntentBucket = "informational" | "transactional" | "account_legal";

export interface Citation {
  citation_id: number;
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  heading_path: string[];
  quote: string;
  start_char: number;
  end_char: number;
  doc_hash: string;
}

export interface Claim {
  claim_id: string;
  text: string;
  citation_ids: number[];
  support: string;
}

export interface Decision {
  action: Action;
  reason_code: string;
  score: number;
  alpha: number;
  threshold_ref: string;
}

export interface Confidence {
  p_correct: number;
  calibrator_version: string;
}

export interface CacheInfo {
  status: CacheStatus;
  entry_id: string | null;
  similarity: number | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  usd: number;
  model: string;
  prompt_version: string;
}

export interface Provenance {
  corpus_version: string;
  index_version: string;
  prompt_version: string;
  verifier_version: string;
  scorer_version: string;
}

export interface Answer {
  answer_id: string;
  trace_id: string;
  language: string;
  text: string;
  claims: Claim[];
  citations: Citation[];
  decision: Decision;
  confidence: Confidence;
  cache: CacheInfo;
  usage: Usage;
  provenance: Provenance;
}

export interface Passage {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  heading_path: string[];
  breadcrumb: string;
  channel: string;
  stage: string;
  score: number;
  rank: number;
  snippet: string;
}

export interface Fusion {
  fuser: string;
  params: Record<string, number | string>;
}

export interface RetrievalEvent {
  event: "retrieval";
  trace_id: string;
  query: string;
  normalized_query: string | null;
  language: string | null;
  passages: Passage[];
  channels: Record<string, number>;
  fusion: Fusion | null;
  candidates: number;
  dropped_duplicates: number;
  cache: { status: CacheStatus; entry_id: string | null; similarity: number | null; reason: string };
  timings_ms: Record<string, number>;
}

export interface TokenEvent {
  event: "tokens";
  trace_id: string;
  delta: string;
  index: number;
}

export interface CitationsEvent {
  event: "citations";
  trace_id: string;
  citations: Citation[];
  claims: Claim[];
}

export interface EscalationSummary {
  ticket_id: string | null;
  priority: Priority;
  detail: string;
  reason: string;
}

export interface DecisionEvent {
  event: "decision";
  trace_id: string;
  action: Action;
  reason_code: string;
  score: number;
  alpha: number;
  threshold_ref: string;
  escalation: EscalationSummary | null;
}

export interface DoneEvent {
  event: "done";
  trace_id: string;
  session_id: string;
  turn_id: string;
  answer: Answer;
  usage: Usage;
  timings_ms: Record<string, number>;
  history_turns: number;
  follow_up: boolean;
  ticket_id: string | null;
  idempotent_replay?: boolean;
}

export interface ErrorPayload {
  code: string;
  message: string;
  trace_id: string;
  retryable: boolean;
  detail?: string | null;
  retry_after_s?: number | null;
}

export interface StreamErrorEvent {
  event: "error";
  trace_id: string;
  error: ErrorPayload;
}

export type ChatEvent =
  | RetrievalEvent
  | TokenEvent
  | CitationsEvent
  | DecisionEvent
  | DoneEvent
  | StreamErrorEvent;

export interface ChatRequest {
  message: string;
  session_id?: string;
  locale?: string;
  intent_bucket?: IntentBucket;
  intent_label?: string | null;
  stream?: boolean;
}

export interface InfoResponse {
  service: string;
  version: string;
  profile: string;
  fake: boolean;
  corpus_version: string;
  index_version: string;
  chunks: number;
  model: string;
  preset: string;
  cache_enabled: boolean;
  admin_enabled: boolean;
}

export interface HealthComponent {
  status: "ok" | "degraded" | "error" | "disabled";
  detail: string;
  version?: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  version: string;
  schema_version: string;
  uptime_s: number;
  checks: Record<string, HealthComponent>;
}

export interface TraceLine {
  kind: string;
  schema_version?: string;
  trace_id: string;
  session_id: string | null;
  turn_id: string | null;
  query: string;
  query_pii?: { redacted: number; kinds: string[] };
  language: string | null;
  intent_bucket: IntentBucket | null;
  decision: Decision | null;
  p_hat: number | null;
  confidence: number | null;
  retrieval?: {
    query: string;
    normalized_query: string | null;
    language: string | null;
    packed: Array<{ chunk: { chunk_id: string; doc_id: string; text: string; heading_path: string[]; metadata?: Record<string, unknown> }; channel: string; score: number; rank: number; stage: string }>;
    fusion: Fusion | null;
    dropped_duplicates?: number;
  } | null;
  retrieval_scores?: Array<{ chunk_id: string; doc_id: string; channel: string; score: number; rank: number; stage: string }>;
  fusion?: Fusion | null;
  cache?: { status: CacheStatus; entry_id: string | null; similarity: number | null; reason: string };
  cache_status?: CacheStatus;
  history?: { follow_up: boolean; turns_used: number; cues?: string[] };
  generation?: {
    repaired: boolean;
    repair_attempts: number;
    dropped_claims: number;
    numeric_violations_pre_repair: number;
    verifier_version: string;
    language_notice: boolean;
  } | null;
  escalation?: { ticket_id: string; reason: string } | null;
  usage: Usage;
  timings_ms?: Record<string, number>;
  stage_latency_ms?: Record<string, number>;
  corpus_version: string;
  index_version: string;
  prompt_version: string;
  service?: {
    profile: string;
    fake: boolean;
    model: string;
    preset: string;
    calibration: { source: string; threshold_ref: string; stale: boolean; age_days: number };
    scorer: { name: string; version: string };
  };
  provider_redactions?: number;
  error: string | null;
  created_at?: string;
}

export interface TraceResponse {
  trace_id: string;
  trace: TraceLine;
}

export interface EscalationTicket {
  ticket_id: string;
  trace_id: string;
  session_id: string | null;
  status: string;
  priority: Priority;
  reason: string;
  note: string;
  created_at: string;
}

export interface EscalationRequest {
  trace_id: string;
  reason: string;
  priority?: Priority;
  note?: string | null;
  session_id?: string | null;
}

export interface FeedbackRequest {
  trace_id: string;
  rating: "up" | "down" | "neutral";
  comment?: string;
  session_id?: string;
}

export interface ApiErrorBody {
  error: ErrorPayload;
}
