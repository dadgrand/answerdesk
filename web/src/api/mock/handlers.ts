/**
 * MSW handlers: the whole console works offline against these.
 *
 * The handlers mirror the real service contract (t9): SSE frames in the documented order,
 * the trace document, the Prometheus exposition, and the taxonomy error envelope. The
 * stream is a real `ReadableStream` with small inter-frame delays, so the mock exercises
 * the same progressive-rendering path as the live service — a canned JSON blob would hide
 * exactly the bug class the streaming tests exist to catch.
 */

import { HttpResponse, http } from "msw";

import type { ChatRequest, EscalationRequest, TraceLine } from "../types";
import {
  ANSWER_DECISION,
  ANSWER_TEXT,
  ANSWER_USAGE,
  answerable,
  buildAnswer,
  buildAnsweredTrace,
  buildEscalatedTrace,
  escalationAnswer,
  MOCK_HEALTH,
  MOCK_INFO,
  MOCK_TICKET,
  mockMetrics,
  retrievalEvent,
} from "./fixtures";
import { ANSWER_CITATIONS, ANSWER_CLAIMS } from "./fixtures";

const TRACE_TTL_MS = 60 * 60 * 1000;
const traces = new Map<string, { trace: TraceLine; createdAt: number }>();
const seenQueries = new Set<string>();
let sequence = 0;

export function resetMockState(): void {
  traces.clear();
  seenQueries.clear();
  sequence = 0;
}

function nextTraceId(): string {
  sequence += 1;
  return `tr-demo-${String(sequence).padStart(4, "0")}`;
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function tokenDeltas(text: string, wordsPerChunk = 4): string[] {
  const words = text.split(/(\s+)/).filter((part) => part.length > 0);
  const chunks: string[] = [];
  let current: string[] = [];
  let wordsInChunk = 0;
  for (const word of words) {
    current.push(word);
    if (word.trim()) wordsInChunk += 1;
    if (wordsInChunk >= wordsPerChunk) {
      chunks.push(current.join(""));
      current = [];
      wordsInChunk = 0;
    }
  }
  if (current.length) chunks.push(current.join(""));
  return chunks;
}

const isTestEnvironment = (): boolean =>
  typeof process !== "undefined" && (process.env.VITEST === "true" || process.env.NODE_ENV === "test");

/** A streaming HTTP response; one frame per chunk, separated by a short delay. */
function sseResponse(frames: string[], intervalMs = 18) {
  const encoder = new TextEncoder();
  const delay = isTestEnvironment() ? 0 : intervalMs;
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (index >= frames.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(frames[index]));
      index += 1;
      if (delay > 0 && index < frames.length) {
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    },
  });
  return new HttpResponse(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Trace-Id": "tr-demo-sse",
    },
  });
}

function answerFrames(traceId: string, cacheHit: boolean): string[] {
  const answer = buildAnswer({ traceId, cacheHit });
  const frames = [frame("retrieval", retrievalEvent(traceId, cacheHit))];
  for (const [index, delta] of tokenDeltas(ANSWER_TEXT).entries()) {
    frames.push(frame("tokens", { event: "tokens", trace_id: traceId, delta, index }));
  }
  frames.push(frame("citations", { event: "citations", trace_id: traceId, citations: ANSWER_CITATIONS, claims: ANSWER_CLAIMS }));
  frames.push(
    frame("decision", {
      event: "decision",
      trace_id: traceId,
      action: "answer",
      reason_code: "ANSWERED",
      score: ANSWER_DECISION.score,
      alpha: ANSWER_DECISION.alpha,
      threshold_ref: ANSWER_DECISION.threshold_ref,
      escalation: null,
    }),
  );
  frames.push(
    frame("done", {
      event: "done",
      trace_id: traceId,
      session_id: "s-console-1042",
      turn_id: "t-7c1e2d3f4a5b6c7d",
      answer,
      usage: answer.usage,
      timings_ms: { retrieval: 1.42, generation: cacheHit ? 0 : 214.6, verify: 0.9, policy: 0.3, total: cacheHit ? 2.1 : 217.5 },
      history_turns: 1,
      follow_up: false,
      ticket_id: null,
      idempotent_replay: false,
    }),
  );
  return frames;
}

function escalationFrames(traceId: string, query: string, reason: string): string[] {
  const answer = escalationAnswer(traceId, reason);
  const ticketId = reason === "PII_DETECTED" ? "tk-pii-7a1c3e5f" : "tk-3f2a9b1c8d7e6f50";
  const detail =
    reason === "PII_DETECTED"
      ? "PII guard: card number detected before any provider call"
      : "p_hat 0.6200 < escalate threshold 0.9500";
  const retrieval = {
    ...retrievalEvent(traceId, false),
    query,
    language: query.match(/[\u0600-\u06FF]/) ? "tg" : "ru",
    passages: [],
    channels: {},
    candidates: 0,
    cache: { status: "miss" as const, entry_id: null, similarity: null, reason: "no candidates" },
  };
  const pII = reason === "PII_DETECTED";
  const frames = [frame("retrieval", retrieval)];
  frames.push(frame("citations", { event: "citations", trace_id: traceId, citations: [], claims: [] }));
  frames.push(
    frame("decision", {
      event: "decision",
      trace_id: traceId,
      action: pII ? "refuse" : "escalate",
      reason_code: reason,
      score: answer.decision.score,
      alpha: answer.decision.alpha,
      threshold_ref: answer.decision.threshold_ref,
      escalation: { ticket_id: ticketId, priority: pII ? "high" : "normal", detail, reason },
    }),
  );
  frames.push(
    frame("done", {
      event: "done",
      trace_id: traceId,
      session_id: "s-console-1043",
      turn_id: "t-1a2b3c4d5e6f7081",
      answer,
      usage: answer.usage,
      timings_ms: { retrieval: 1.1, generation: pII ? 0 : 188.4, verify: 0.4, policy: 0.2, total: pII ? 1.8 : 190.2 },
      history_turns: 1,
      follow_up: false,
      ticket_id: ticketId,
      idempotent_replay: false,
    }),
  );
  return frames;
}

export const mockHandlers = [
  http.get("*/healthz", () => HttpResponse.json(MOCK_HEALTH)),

  http.get("*/v1/info", () => HttpResponse.json(MOCK_INFO)),

  http.get("*/metrics", async () => {
    await new Promise((resolve) => setTimeout(resolve, isTestEnvironment() ? 0 : 60));
    return HttpResponse.text(mockMetrics(), {
      headers: { "Content-Type": "text/plain; version=0.0.4; charset=utf-8" },
    });
  }),

  http.post("*/v1/chat", async ({ request }) => {
    const body = (await request.json()) as ChatRequest;
    const message = body.message ?? "";
    const traceId = nextTraceId();
    const pii = /\b(?:\d[ -]?){13,19}\b/.test(message);

    if (pii) {
      traces.set(traceId, { trace: buildEscalatedTrace(traceId, message, "PII_DETECTED"), createdAt: Date.now() });
      return sseResponse(escalationFrames(traceId, message, "PII_DETECTED"));
    }

    if (answerable(message)) {
      const key = message.trim().toLowerCase();
      const cacheHit = seenQueries.has(key);
      seenQueries.add(key);
      traces.set(traceId, { trace: buildAnsweredTrace(traceId, cacheHit), createdAt: Date.now() });
      return sseResponse(answerFrames(traceId, cacheHit));
    }

    const reason = /возврат|qaytarish|退款|refund|qaytar/i.test(message) ? "NO_EVIDENCE" : "LOW_CONFIDENCE";
    traces.set(traceId, { trace: buildEscalatedTrace(traceId, message, reason), createdAt: Date.now() });
    return sseResponse(escalationFrames(traceId, message, reason));
  }),

  http.get("*/v1/traces/:traceId", ({ params }) => {
    const traceId = String(params.traceId);
    purgeExpired();
    const entry = traces.get(traceId);
    if (!entry) {
      return HttpResponse.json(
        {
          error: {
            code: "not_found",
            message: `trace ${traceId} was not found`,
            trace_id: traceId,
            retryable: false,
          },
        },
        { status: 404 },
      );
    }
    return HttpResponse.json({ trace_id: traceId, trace: entry.trace });
  }),

  http.post("*/v1/escalations", async ({ request }) => {
    const body = (await request.json()) as EscalationRequest;
    return HttpResponse.json(
      {
        ...MOCK_TICKET,
        trace_id: body.trace_id,
        reason: body.reason,
        priority: body.priority ?? "normal",
        note: body.note ?? "",
        created_at: new Date().toISOString(),
      },
      { status: 201 },
    );
  }),

  http.post("*/v1/feedback", async ({ request }) => {
    const body = (await request.json()) as { trace_id: string; rating: string };
    return HttpResponse.json(
      {
        feedback_id: "fb-9e8d7c6b5a4f",
        trace_id: body.trace_id,
        rating: body.rating,
        created_at: new Date().toISOString(),
      },
      { status: 201 },
    );
  }),

  http.get("*/v1/sessions/:sessionId", ({ params }) =>
    HttpResponse.json({
      session_id: String(params.sessionId),
      created_at: new Date(Date.now() - 600_000).toISOString(),
      updated_at: new Date().toISOString(),
      locale: "ru",
      turn_count: 1,
      turns: [],
    }),
  ),
];

function purgeExpired(): void {
  const now = Date.now();
  for (const [key, entry] of traces) {
    if (now - entry.createdAt > TRACE_TTL_MS) traces.delete(key);
  }
}

export { ANSWER_TEXT, ANSWER_USAGE };
