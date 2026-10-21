/**
 * SSE decoder tests: the streaming contract is a parser contract.
 *
 * A chunk boundary can land anywhere — inside `event:`, between `data:` lines, between
 * frames. These tests feed the decoder the awkward splits a real proxy produces.
 */

import { describe, expect, it } from "vitest";

import { SseDecoder } from "./sse";

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

describe("SseDecoder", () => {
  it("parses a complete frame", () => {
    const decoder = new SseDecoder();
    const frames = decoder.push(frame("retrieval", { trace_id: "tr-1" }));
    expect(frames).toHaveLength(1);
    expect(frames[0]?.event).toBe("retrieval");
    expect(JSON.parse(frames[0]?.data ?? "{}")).toEqual({ trace_id: "tr-1" });
  });

  it("reassembles frames split at arbitrary byte boundaries", () => {
    const stream = `${frame("retrieval", { trace_id: "tr-1" })}${frame("tokens", { delta: "Привет " })}${frame(
      "done",
      { trace_id: "tr-1" },
    )}`;
    const decoder = new SseDecoder();
    const events: string[] = [];
    // Chunk sizes that cut through field names and JSON.
    for (let index = 0; index < stream.length; index += 7) {
      for (const parsed of decoder.push(stream.slice(index, index + 7))) events.push(parsed.event);
    }
    for (const parsed of decoder.flush()) events.push(parsed.event);
    expect(events).toEqual(["retrieval", "tokens", "done"]);
  });

  it("tolerates CRLF line endings and comment heartbeats", () => {
    const decoder = new SseDecoder();
    const frames = decoder.push(": ping\r\n\r\nevent: tokens\r\ndata: {\"delta\":\"x\"}\r\n\r\n");
    expect(frames).toHaveLength(1);
    expect(frames[0]?.event).toBe("tokens");
  });

  it("joins multi-line data fields", () => {
    const decoder = new SseDecoder();
    const frames = decoder.push("event: note\ndata: line one\ndata: line two\n\n");
    expect(frames[0]?.data).toBe("line one\nline two");
  });

  it("keeps a trailing frame without a blank line on flush", () => {
    const decoder = new SseDecoder();
    expect(decoder.push("event: done\ndata: {}")).toHaveLength(0);
    const flushed = decoder.flush();
    expect(flushed).toHaveLength(1);
    expect(flushed[0]?.event).toBe("done");
  });

  it("ignores empty blocks and keeps event ids", () => {
    const decoder = new SseDecoder();
    const frames = decoder.push("\n\nevent: tokens\nid: 42\ndata: {\"delta\":\"a\"}\n\n");
    expect(frames).toHaveLength(1);
    expect(frames[0]?.id).toBe("42");
  });
});
