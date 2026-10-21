/**
 * The console's state machine (pure reducer — no React, no network).
 *
 * Keeping the transitions pure is what makes progressive streaming testable: a test feeds
 * the same `ChatEvent` sequence the SSE reader produces and asserts the rendered state
 * after each event, with no timers and no transport involved.
 */

import type {
  Answer,
  ChatEvent,
  Citation,
  Claim,
  DecisionEvent,
  DoneEvent,
  RetrievalEvent,
  Usage,
} from "../api/types";
import type { Conversation } from "../demo/inbox";
import { buildInbox } from "../demo/inbox";

export type Theme = "dark" | "light";
export type TurnStatus = "idle" | "streaming" | "complete" | "escalated" | "error";
export type Rating = "up" | "down";

export interface TurnError {
  message: string;
  code: string;
  retryable: boolean;
}

export interface TurnState {
  conversationId: string;
  status: TurnStatus;
  traceId: string | null;
  retrieval: RetrievalEvent | null;
  streamingText: string;
  citations: Citation[];
  claims: Claim[];
  decision: DecisionEvent | null;
  answer: Answer | null;
  usage: Usage | null;
  timings: Record<string, number>;
  ticketId: string | null;
  error: TurnError | null;
  tokenCount: number;
}

export interface SentReply {
  text: string;
  at: number;
}

export interface ConsoleState {
  inbox: Conversation[];
  selectedId: string | null;
  turns: Record<string, TurnState>;
  diagnosticsOpen: boolean;
  theme: Theme;
  localeFilter: "all" | string;
  slaFilter: "all" | "at-risk";
  sent: Record<string, SentReply>;
  feedback: Record<string, Rating>;
  tickets: Record<string, { ticketId: string; reason: string }>;
}

export type ConsoleAction =
  | { type: "select"; conversationId: string }
  | { type: "turn/start"; conversationId: string }
  | { type: "turn/event"; conversationId: string; event: ChatEvent }
  | { type: "turn/error"; conversationId: string; error: TurnError }
  | { type: "diagnostics"; open?: boolean }
  | { type: "theme"; theme: Theme }
  | { type: "filter/locale"; locale: "all" | string }
  | { type: "filter/sla"; sla: "all" | "at-risk" }
  | { type: "draft/send"; conversationId: string; text: string; at: number }
  | { type: "feedback"; conversationId: string; rating: Rating }
  | { type: "escalation/opened"; conversationId: string; ticketId: string; reason: string };

export function emptyTurn(conversationId: string): TurnState {
  return {
    conversationId,
    status: "idle",
    traceId: null,
    retrieval: null,
    streamingText: "",
    citations: [],
    claims: [],
    decision: null,
    answer: null,
    usage: null,
    timings: {},
    ticketId: null,
    error: null,
    tokenCount: 0,
  };
}

export function createInitialState(
  theme: Theme = "dark",
  inbox?: Conversation[],
  selectedId?: string | null,
): ConsoleState {
  const conversations = inbox ?? buildInbox();
  const requested =
    selectedId && conversations.some((item) => item.id === selectedId) ? selectedId : null;
  return {
    inbox: conversations,
    selectedId: requested ?? conversations[0]?.id ?? null,
    turns: {},
    diagnosticsOpen: false,
    theme,
    localeFilter: "all",
    slaFilter: "all",
    sent: {},
    feedback: {},
    tickets: {},
  };
}

export function consoleReducer(state: ConsoleState, action: ConsoleAction): ConsoleState {
  switch (action.type) {
    case "select":
      return { ...state, selectedId: action.conversationId };

    case "turn/start": {
      const existing = state.turns[action.conversationId];
      if (existing && (existing.status === "streaming" || existing.status === "complete")) {
        return state;
      }
      return {
        ...state,
        turns: {
          ...state.turns,
          [action.conversationId]: { ...emptyTurn(action.conversationId), status: "streaming" },
        },
      };
    }

    case "turn/event": {
      const current = state.turns[action.conversationId] ?? emptyTurn(action.conversationId);
      const next = applyEvent(current, action.event);
      if (next === current) return state;
      return { ...state, turns: { ...state.turns, [action.conversationId]: next } };
    }

    case "turn/error": {
      const current = state.turns[action.conversationId] ?? emptyTurn(action.conversationId);
      return {
        ...state,
        turns: {
          ...state.turns,
          [action.conversationId]: { ...current, status: "error", error: action.error },
        },
      };
    }

    case "diagnostics":
      return { ...state, diagnosticsOpen: action.open ?? !state.diagnosticsOpen };

    case "theme":
      return { ...state, theme: action.theme };

    case "filter/locale":
      return { ...state, localeFilter: action.locale };

    case "filter/sla":
      return { ...state, slaFilter: action.sla };

    case "draft/send":
      return {
        ...state,
        sent: { ...state.sent, [action.conversationId]: { text: action.text, at: action.at } },
      };

    case "feedback":
      return { ...state, feedback: { ...state.feedback, [action.conversationId]: action.rating } };

    case "escalation/opened":
      return {
        ...state,
        tickets: {
          ...state.tickets,
          [action.conversationId]: { ticketId: action.ticketId, reason: action.reason },
        },
      };

    default:
      return state;
  }
}

function applyEvent(turn: TurnState, event: ChatEvent): TurnState {
  switch (event.event) {
    case "retrieval":
      return {
        ...turn,
        status: "streaming",
        traceId: event.trace_id,
        retrieval: event,
        timings: { ...turn.timings, ...event.timings_ms },
      };

    case "tokens": {
      // A stream that starts mid-way is still a valid stream: append, never replace.
      return {
        ...turn,
        status: "streaming",
        streamingText: turn.streamingText + event.delta,
        tokenCount: turn.tokenCount + 1,
      };
    }

    case "citations":
      return { ...turn, citations: event.citations, claims: event.claims };

    case "decision":
      return { ...turn, decision: event, status: "streaming" };

    case "done": {
      // The service's last-resort frame after a mid-stream failure carries `error` and no
      // `answer` (docs/api.md §5.4). Treat it as an error, never as a blank answer.
      const terminal = event as DoneEvent & { error?: { code: string; message: string; retryable: boolean } };
      if (!terminal.answer) {
        return {
          ...turn,
          status: "error",
          traceId: terminal.trace_id ?? turn.traceId,
          error: terminal.error
            ? {
                message: terminal.error.message,
                code: terminal.error.code,
                retryable: terminal.error.retryable,
              }
            : { message: "the stream ended without an answer", code: "stream_error", retryable: true },
        };
      }
      const action = terminal.answer.decision.action;
      return {
        ...turn,
        status: action === "answer" ? "complete" : "escalated",
        traceId: terminal.trace_id,
        answer: terminal.answer,
        citations: terminal.answer.citations,
        claims: terminal.answer.claims,
        // The final answer text is authoritative: a dropped token frame cannot truncate it.
        streamingText: terminal.answer.text || turn.streamingText,
        usage: terminal.usage,
        timings: { ...turn.timings, ...terminal.timings_ms },
        ticketId: terminal.ticket_id,
        decision:
          turn.decision ??
          {
            event: "decision",
            trace_id: terminal.trace_id,
            action,
            reason_code: terminal.answer.decision.reason_code,
            score: terminal.answer.decision.score,
            alpha: terminal.answer.decision.alpha,
            threshold_ref: terminal.answer.decision.threshold_ref,
            escalation: null,
          },
      };
    }

    case "error":
      return {
        ...turn,
        status: "error",
        error: {
          message: event.error.message,
          code: event.error.code,
          retryable: event.error.retryable,
        },
      };

    default:
      return turn;
  }
}

export function selectedConversation(state: ConsoleState): Conversation | null {
  return state.inbox.find((conversation) => conversation.id === state.selectedId) ?? null;
}

export function turnFor(state: ConsoleState, conversationId: string | null): TurnState | null {
  if (!conversationId) return null;
  return state.turns[conversationId] ?? null;
}

export function visibleConversations(state: ConsoleState, now: number): Conversation[] {
  return state.inbox.filter((conversation) => {
    if (state.localeFilter !== "all" && conversation.customer.locale !== state.localeFilter) {
      return false;
    }
    if (state.slaFilter === "at-risk") {
      const ageSeconds = (now - conversation.waitingSince) / 1000;
      if (ageSeconds < 75) return false;
    }
    return true;
  });
}

/** The displayed draft: the streamed/verified text, or the operator's edited version. */
export function draftText(turn: TurnState | null): string {
  return turn?.answer?.text || turn?.streamingText || "";
}
