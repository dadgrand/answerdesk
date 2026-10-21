/**
 * The API client: one place that knows the answerdesk wire format.
 *
 * `VITE_API_BASE` is empty by default (same origin: the service serves the built console,
 * and the Vite dev server proxies `/v1`, `/metrics`, `/healthz`). The Playwright script
 * sets it to the real service it starts.
 *
 * Streaming is the point of the console, so `streamChat` returns an async generator over
 * parsed events and never buffers the whole answer: the UI paints each frame as it lands.
 */

import { SseDecoder, type SseFrame } from "./sse";
import type {
  ApiErrorBody,
  ChatEvent,
  ChatRequest,
  EscalationRequest,
  EscalationTicket,
  FeedbackRequest,
  HealthResponse,
  InfoResponse,
  TraceResponse,
} from "./types";

const ENV_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
export const API_BASE = ENV_BASE.replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;
  readonly traceId: string | null;

  constructor(message: string, options: { status: number; code?: string; retryable?: boolean; traceId?: string | null }) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code ?? "http_error";
    this.retryable = options.retryable ?? (options.status >= 500 || options.status === 429);
    this.traceId = options.traceId ?? null;
  }
}

async function readError(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `${response.status} ${response.statusText}`.trim();
  let retryable = response.status >= 500 || response.status === 429;
  let traceId: string | null = response.headers.get("x-trace-id");
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (body.error) {
      code = body.error.code;
      message = body.error.message;
      retryable = body.error.retryable;
      traceId = body.error.trace_id ?? traceId;
    }
  } catch {
    /* a non-JSON error body is still an error; the status line is enough */
  }
  return new ApiError(message, { status: response.status, code, retryable, traceId });
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), { headers: { Accept: "application/json" }, ...init });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as T;
}

export async function fetchInfo(): Promise<InfoResponse> {
  return getJson<InfoResponse>("/v1/info");
}

export async function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/healthz");
}

export async function fetchMetricsText(): Promise<string> {
  const response = await fetch(apiUrl("/metrics"), { headers: { Accept: "text/plain" } });
  if (!response.ok) throw await readError(response);
  return response.text();
}

export async function fetchTrace(traceId: string): Promise<TraceResponse> {
  return getJson<TraceResponse>(`/v1/traces/${encodeURIComponent(traceId)}`);
}

export async function postEscalation(body: EscalationRequest): Promise<EscalationTicket> {
  return getJson<EscalationTicket>("/v1/escalations", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}

export async function postFeedback(body: FeedbackRequest): Promise<void> {
  await getJson<unknown>("/v1/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}

export interface StreamOptions {
  signal?: AbortSignal;
  /** Called for every parsed frame before it is yielded (used by the reducer). */
  onEvent?: (event: ChatEvent) => void;
}

/**
 * Start one chat turn and yield its SSE events.
 *
 * Cancellation is implemented by cancelling the *reader* rather than by passing an
 * `AbortSignal` to `fetch`: in a jsdom test environment the jsdom `AbortSignal` is not the
 * realm's own class that undici validates, so forwarding it turns every request into a
 * `TypeError`. Cancelling the reader closes the stream just as well and keeps the client
 * environment-agnostic.
 *
 * Message-level failures arrive as an `error` event followed by `done` (docs/api.md §5.4);
 * transport-level failures throw `ApiError` before the first event.
 */
export async function* streamChat(
  request: ChatRequest,
  options: StreamOptions = {},
): AsyncGenerator<ChatEvent> {
  const response = await fetch(apiUrl("/v1/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ ...request, stream: true }),
  });
  if (!response.ok) throw await readError(response);
  if (!response.body) throw new ApiError("the response has no body", { status: 500 });

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    // A proxy converted the stream to a JSON answer; still surface the same UI.
    const body = (await response.json()) as ChatEvent;
    options.onEvent?.(body);
    yield body;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const sse = new SseDecoder();
  const onAbort = () => {
    void reader.cancel().catch(() => undefined);
  };
  options.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    for (;;) {
      if (options.signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      for (const frame of sse.push(decoder.decode(value, { stream: true }))) {
        const event = parseFrame(frame);
        if (!event) continue;
        options.onEvent?.(event);
        yield event;
      }
    }
    for (const frame of sse.flush()) {
      const event = parseFrame(frame);
      if (!event) continue;
      options.onEvent?.(event);
      yield event;
    }
  } finally {
    options.signal?.removeEventListener("abort", onAbort);
    try {
      reader.releaseLock();
    } catch {
      /* already released by cancel() */
    }
  }
}

export function parseFrame(frame: SseFrame): ChatEvent | null {
  if (!frame.data) return null;
  try {
    return JSON.parse(frame.data) as ChatEvent;
  } catch {
    return null;
  }
}
