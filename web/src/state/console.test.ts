/** State-machine tests: progressive streaming without a transport in the way. */

import { describe, expect, it } from "vitest";

import { buildAnswer, buildAnsweredTrace, escalationAnswer, retrievalEvent } from "../api/mock/fixtures";
import type { ChatEvent, DoneEvent, RetrievalEvent, TokenEvent } from "../api/types";
import { buildInbox } from "../demo/inbox";
import {
  consoleReducer,
  createInitialState,
  draftText,
  emptyTurn,
  turnFor,
  visibleConversations,
  type ConsoleAction,
  type ConsoleState,
} from "./console";

const NOW = Date.UTC(2026, 9, 21, 12, 0, 0);

function apply(state: ConsoleState, actions: ConsoleAction[]): ConsoleState {
  return actions.reduce(consoleReducer, state);
}

function token(delta: string, index = 0): TokenEvent {
  return { event: "tokens", trace_id: "tr-1", delta, index };
}

function doneAnswered(traceId = "tr-1"): DoneEvent {
  return {
    event: "done",
    trace_id: traceId,
    session_id: "s-console-c-1042",
    turn_id: "t-1",
    answer: buildAnswer({ traceId, cacheHit: false }),
    usage: { input_tokens: 10, output_tokens: 4, usd: 0.001, model: "m", prompt_version: "p" },
    timings_ms: { total: 214, retrieval: 1.2 },
    history_turns: 1,
    follow_up: false,
    ticket_id: null,
  };
}

describe("consoleReducer", () => {
  it("selects the first conversation and starts idle", () => {
    const state = createInitialState("dark", buildInbox(NOW));
    expect(state.selectedId).toBe("c-1042");
    expect(state.turns).toEqual({});
    expect(state.theme).toBe("dark");
  });

  it("accumulates streamed tokens one delta at a time", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      { type: "turn/event", conversationId: "c-1042", event: token("Обслуживание ") },
    ]);
    expect(turnFor(state, "c-1042")?.streamingText).toBe("Обслуживание ");
    state = consoleReducer(state, {
      type: "turn/event",
      conversationId: "c-1042",
      event: token("карты ", 1),
    });
    expect(turnFor(state, "c-1042")?.streamingText).toBe("Обслуживание карты ");
    expect(turnFor(state, "c-1042")?.tokenCount).toBe(2);
    expect(turnFor(state, "c-1042")?.status).toBe("streaming");
  });

  it("records retrieval and only exposes citations once the verifier emits them", () => {
    const retrieval: RetrievalEvent = retrievalEvent("tr-1", false);
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      { type: "turn/event", conversationId: "c-1042", event: retrieval },
    ]);
    expect(turnFor(state, "c-1042")?.traceId).toBe("tr-1");
    expect(turnFor(state, "c-1042")?.retrieval?.passages).toHaveLength(5);
    expect(turnFor(state, "c-1042")?.citations).toHaveLength(0);

    state = consoleReducer(state, {
      type: "turn/event",
      conversationId: "c-1042",
      event: {
        event: "citations",
        trace_id: "tr-1",
        citations: buildAnswer({ traceId: "tr-1", cacheHit: false }).citations,
        claims: buildAnswer({ traceId: "tr-1", cacheHit: false }).claims,
      },
    });
    expect(turnFor(state, "c-1042")?.citations).toHaveLength(2);
    expect(turnFor(state, "c-1042")?.claims).toHaveLength(2);
  });

  it("completes an answered turn and prefers the final answer text", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      { type: "turn/event", conversationId: "c-1042", event: token("truncated") },
      { type: "turn/event", conversationId: "c-1042", event: doneAnswered() },
    ]);
    const turn = turnFor(state, "c-1042");
    expect(turn?.status).toBe("complete");
    expect(turn?.answer?.text).toContain("120 рублей");
    expect(turn?.streamingText).toBe(turn?.answer?.text);
    expect(draftText(turn)).toBe(turn?.answer?.text);
    // The done event carries no explicit decision frame here: one is derived.
    expect(turn?.decision?.action).toBe("answer");
  });

  it("marks an escalated turn and keeps its ticket", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    const traceId = "tr-esc";
    const done: DoneEvent = {
      ...doneAnswered(traceId),
      answer: escalationAnswer(traceId, "LOW_CONFIDENCE"),
      ticket_id: "tk-3f2a9b1c8d7e6f50",
    };
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1043" },
      { type: "turn/event", conversationId: "c-1043", event: done },
    ]);
    const turn = turnFor(state, "c-1043");
    expect(turn?.status).toBe("escalated");
    expect(turn?.ticketId).toBe("tk-3f2a9b1c8d7e6f50");
    expect(turn?.answer?.text).toBe("");
    expect(turn?.decision?.reason_code).toBe("LOW_CONFIDENCE");
  });

  it("maps a stream error event to an error state", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      {
        type: "turn/event",
        conversationId: "c-1042",
        event: {
          event: "error",
          trace_id: "tr-1",
          error: { code: "provider_timeout", message: "timed out", trace_id: "tr-1", retryable: true },
        },
      },
    ]);
    expect(turnFor(state, "c-1042")?.status).toBe("error");
    expect(turnFor(state, "c-1042")?.error?.code).toBe("provider_timeout");
    expect(turnFor(state, "c-1042")?.error?.retryable).toBe(true);
  });

  it("treats a terminal done frame without an answer as an error, never a blank answer", () => {
    // docs/api.md §5.4: a mid-stream failure ends with `done` carrying `error` and no answer.
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      {
        type: "turn/event",
        conversationId: "c-1042",
        event: {
          event: "done",
          trace_id: "tr-fail",
          session_id: "s-console-c-1042",
          turn_id: "",
          error: { code: "internal_error", message: "internal error", trace_id: "tr-fail", retryable: false },
        } as unknown as ChatEvent,
      },
    ]);
    const turn = turnFor(state, "c-1042");
    expect(turn?.status).toBe("error");
    expect(turn?.error).toEqual({ code: "internal_error", message: "internal error", retryable: false });
    expect(turn?.traceId).toBe("tr-fail");
    expect(turn?.answer).toBeNull();
  });

  it("does not restart a completed turn but restarts a failed one", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "turn/start", conversationId: "c-1042" },
      { type: "turn/event", conversationId: "c-1042", event: doneAnswered() },
    ]);
    const afterComplete = consoleReducer(state, { type: "turn/start", conversationId: "c-1042" });
    expect(afterComplete).toBe(state);

    const failed = apply(state, [
      { type: "turn/error", conversationId: "c-1043", error: { message: "x", code: "y", retryable: true } },
    ]);
    const retried = consoleReducer(failed, { type: "turn/start", conversationId: "c-1043" });
    expect(turnFor(retried, "c-1043")?.status).toBe("streaming");
    expect(turnFor(retried, "c-1043")?.error).toBeNull();
  });

  it("records operator actions: sent reply, feedback and escalation", () => {
    let state = createInitialState("dark", buildInbox(NOW));
    state = apply(state, [
      { type: "draft/send", conversationId: "c-1042", text: "Edited reply", at: NOW },
      { type: "feedback", conversationId: "c-1042", rating: "up" },
      { type: "escalation/opened", conversationId: "c-1042", ticketId: "tk-1", reason: "OPERATOR_REQUEST" },
    ]);
    expect(state.sent["c-1042"]?.text).toBe("Edited reply");
    expect(state.feedback["c-1042"]).toBe("up");
    expect(state.tickets["c-1042"]?.ticketId).toBe("tk-1");
  });

  it("filters by language and SLA risk", () => {
    const state = createInitialState("dark", buildInbox(NOW));
    expect(visibleConversations(state, NOW)).toHaveLength(7);
    const ruOnly = consoleReducer(state, { type: "filter/locale", locale: "ru" });
    expect(visibleConversations(ruOnly, NOW).map((c) => c.customer.locale)).toEqual(["ru", "ru"]);
    const atRisk = consoleReducer(state, { type: "filter/sla", sla: "at-risk" });
    const risky = visibleConversations(atRisk, NOW);
    expect(risky.length).toBeGreaterThan(0);
    expect(risky.every((c) => (NOW - c.waitingSince) / 1000 >= 75)).toBe(true);
  });

  it("starts from an empty turn with no fabricated state", () => {
    const turn = emptyTurn("c-1");
    expect(turn.status).toBe("idle");
    expect(turn.answer).toBeNull();
    expect(turn.traceId).toBeNull();
  });

  it("keeps a trace line builder consistent with the answer decision", () => {
    const trace = buildAnsweredTrace("tr-1", true);
    expect(trace.cache?.status).toBe("hit");
    expect(trace.decision?.action).toBe("answer");
    expect(trace.retrieval_scores).toHaveLength(5);
    expect(trace.timings_ms?.generation).toBeGreaterThan(0);
  });
});
