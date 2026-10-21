/**
 * The `useConsole` hook: state machine + the network effects around it.
 *
 * Everything that talks to the service lives here, so components stay presentational and
 * the streaming path is testable through MSW. Three behaviours are deliberate:
 *
 * 1. **One stream per conversation.** Selecting a conversation that already has a
 *    complete/escalated turn does not re-ask; the "Ask again" action forces a new turn.
 * 2. **Abortable.** Switching conversations aborts the in-flight request, so a slow turn
 *    cannot paint into the wrong conversation (the reducer is keyed by conversation id).
 * 3. **Trace-driven diagnostics.** The drawer reads `GET /v1/traces/{trace_id}` — the same
 *    redacted JSONL line the operators see in the log — instead of trusting the SSE payload
 *    alone. That is what makes the diagnostics panel an audit tool, not a debug print.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { ApiError, fetchTrace, postEscalation, postFeedback, streamChat } from "../api/client";
import type { ChatEvent, EscalationTicket, TraceResponse } from "../api/types";
import { buildInbox, type Conversation } from "../demo/inbox";
import { urlParam } from "../lib/url";
import {
  consoleReducer,
  createInitialState,
  selectedConversation,
  type ConsoleState,
  type Rating,
  type Theme,
  type TurnState,
  turnFor,
  visibleConversations,
} from "./console";

export const THEME_STORAGE_KEY = "answerdesk.theme";

export function readStoredTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage can be unavailable (private mode); the default is fine */
  }
  return "dark";
}

/**
 * `?conversation=<id>` deep-links a conversation.
 *
 * The screenshot script and an operator's bookmark both need this: opening the console on
 * a specific customer must not require clicking through the queue first.
 */
export function conversationFromUrl(inbox: Conversation[]): string | null {
  try {
    const requested = urlParam("conversation");
    return requested && inbox.some((item) => item.id === requested) ? requested : null;
  } catch {
    return null;
  }
}

export interface Notice {
  kind: "info" | "success" | "error";
  message: string;
}

export interface UseConsole {
  state: ConsoleState;
  conversations: Conversation[];
  conversation: Conversation | null;
  turn: TurnState | null;
  selectConversation: (conversationId: string) => void;
  retryTurn: (conversationId: string) => void;
  toggleDiagnostics: (open?: boolean) => void;
  setTheme: (theme: Theme) => void;
  setLocaleFilter: (locale: "all" | string) => void;
  setSlaFilter: (sla: "all" | "at-risk") => void;
  sendDraft: (conversationId: string, text: string) => void;
  escalate: (conversationId: string) => Promise<void>;
  rate: (conversationId: string, rating: Rating) => Promise<void>;
  loadTrace: (traceId: string) => Promise<TraceResponse>;
  ticket: EscalationTicket | null;
  notice: Notice | null;
  clearNotice: () => void;
  now: number;
}

/**
 * `?session=<slug>` opens a fresh conversation id for the same queue item.
 *
 * Cache entries are keyed on the query, not the session, so a returning customer with a new
 * session is exactly the case the exact cache exists for — and it is how the screenshot
 * script demonstrates a cache hit without faking one.
 */
export function sessionFromUrl(): string | null {
  try {
    const requested = urlParam("session");
    return requested ? requested.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 40) : null;
  } catch {
    return null;
  }
}

export function useConsole(options: { inbox?: Conversation[]; now?: number } = {}): UseConsole {
  const inbox = useMemo(() => options.inbox ?? buildInbox(), [options.inbox]);
  const sessionSlug = sessionFromUrl();
  const sessionIdFor = useCallback(
    (conversationId: string) =>
      sessionSlug ? `s-${sessionSlug}-${conversationId}`.slice(0, 64) : `s-console-${conversationId}`,
    [sessionSlug],
  );
  const [state, dispatch] = useReducer(consoleReducer, undefined, () =>
    createInitialState(readStoredTheme(), inbox, conversationFromUrl(inbox)),
  );
  const [notice, setNotice] = useState<Notice | null>(null);
  const [ticket, setTicket] = useState<EscalationTicket | null>(null);
  const [now, setNow] = useState(() => options.now ?? Date.now());
  const controllers = useRef(new Map<string, AbortController>());
  const started = useRef(new Set<string>());

  // A one-second clock drives the SLA timers without re-rendering on every event. The
  // vitest suite freezes the clock instead: a 1 Hz state tick inside an act() scope only
  // adds warnings, and the SLA arithmetic itself is unit-tested in lib/sla.
  const clockIsFrozen = options.now !== undefined || import.meta.env.MODE === "test";
  useEffect(() => {
    if (clockIsFrozen) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [clockIsFrozen]);

  useEffect(() => {
    document.documentElement.dataset.theme = state.theme;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, state.theme);
    } catch {
      /* non-fatal */
    }
  }, [state.theme]);

  const startTurn = useCallback(
    (conversation: Conversation) => {
      if (started.current.has(conversation.id)) return;
      started.current.add(conversation.id);
      const controller = new AbortController();
      controllers.current.get(conversation.id)?.abort();
      controllers.current.set(conversation.id, controller);
      dispatch({ type: "turn/start", conversationId: conversation.id });

      void (async () => {
        try {
          for await (const event of streamChat(
            {
              message: conversation.latestMessage,
              session_id: sessionIdFor(conversation.id),
              locale: conversation.customer.locale,
              intent_bucket: conversation.intentBucket,
              intent_label: conversation.intentLabel,
            },
            { signal: controller.signal },
          )) {
            if (controller.signal.aborted) return;
            dispatch({ type: "turn/event", conversationId: conversation.id, event });
          }
        } catch (error) {
          if (controller.signal.aborted) return;
          const apiError = error instanceof ApiError ? error : new ApiError(String(error), { status: 0 });
          dispatch({
            type: "turn/error",
            conversationId: conversation.id,
            error: { message: apiError.message, code: apiError.code, retryable: apiError.retryable },
          });
        }
      })();
    },
    [sessionIdFor],
  );

  const selectConversation = useCallback(
    (conversationId: string) => {
      dispatch({ type: "select", conversationId });
      const conversation = inbox.find((item) => item.id === conversationId);
      if (conversation) startTurn(conversation);
    },
    [inbox, startTurn],
  );

  const retryTurn = useCallback(
    (conversationId: string) => {
      started.current.delete(conversationId);
      const conversation = inbox.find((item) => item.id === conversationId);
      if (conversation) startTurn(conversation);
    },
    [inbox, startTurn],
  );

  // The selected conversation (deep link or first in the queue) is opened on mount.
  useEffect(() => {
    const selected = inbox.find((item) => item.id === state.selectedId) ?? inbox[0];
    if (selected && !started.current.has(selected.id)) startTurn(selected);
  }, [inbox, startTurn, state.selectedId]);

  useEffect(() => {
    const active = controllers.current;
    return () => {
      for (const controller of active.values()) controller.abort();
      active.clear();
    };
  }, []);

  const sendDraft = useCallback(
    (conversationId: string, text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      dispatch({ type: "draft/send", conversationId, text: trimmed, at: Date.now() });
      setNotice({
        kind: "success",
        message: "Reply marked as sent. v1.0 has no outbound connector, so this records the action in the console.",
      });
    },
    [],
  );

  const escalate = useCallback(
    async (conversationId: string) => {
      const turn = turnFor(state, conversationId);
      const conversation = inbox.find((item) => item.id === conversationId);
      if (!turn?.traceId || !conversation) {
        setNotice({ kind: "error", message: "There is no completed turn to escalate yet." });
        return;
      }
      try {
        const created = await postEscalation({
          trace_id: turn.traceId,
          reason: "OPERATOR_REQUEST",
          priority: conversation.priority,
          note: `Escalated from the operator console · ${conversation.subject}`,
          session_id: sessionIdFor(conversationId),
        });
        setTicket(created);
        dispatch({
          type: "escalation/opened",
          conversationId,
          ticketId: created.ticket_id,
          reason: created.reason,
        });
        setNotice({ kind: "success", message: `Escalation ticket ${created.ticket_id} created.` });
      } catch (error) {
        const apiError = error instanceof ApiError ? error : new ApiError(String(error), { status: 0 });
        setNotice({ kind: "error", message: `Escalation failed: ${apiError.message}` });
      }
    },
    [inbox, state, sessionIdFor],
  );

  const rate = useCallback(
    async (conversationId: string, rating: Rating) => {
      const turn = turnFor(state, conversationId);
      if (!turn?.traceId) return;
      try {
        await postFeedback({ trace_id: turn.traceId, rating, session_id: sessionIdFor(conversationId) });
        dispatch({ type: "feedback", conversationId, rating });
        setNotice({ kind: "info", message: "Feedback recorded against the trace." });
      } catch (error) {
        const apiError = error instanceof ApiError ? error : new ApiError(String(error), { status: 0 });
        setNotice({ kind: "error", message: `Feedback failed: ${apiError.message}` });
      }
    },
    [state, sessionIdFor],
  );

  const loadTrace = useCallback(async (traceId: string) => fetchTrace(traceId), []);

  return {
    state,
    conversations: visibleConversations(state, now),
    conversation: selectedConversation(state),
    turn: turnFor(state, state.selectedId),
    selectConversation,
    retryTurn,
    toggleDiagnostics: (open?: boolean) => dispatch({ type: "diagnostics", open }),
    setTheme: (theme: Theme) => dispatch({ type: "theme", theme }),
    setLocaleFilter: (locale) => dispatch({ type: "filter/locale", locale }),
    setSlaFilter: (sla) => dispatch({ type: "filter/sla", sla }),
    sendDraft,
    escalate,
    rate,
    loadTrace,
    ticket,
    notice,
    clearNotice: () => setNotice(null),
    now,
  };
}

export type { ChatEvent };
