/**
 * Server-Sent Events decoder.
 *
 * The service emits `event: <name>` + one `data:` line + a blank line per frame
 * (docs/api.md §5.2). A browser reader hands us arbitrary chunk boundaries, so this
 * decoder is incremental: it keeps a buffer, splits complete frames on a blank line, and
 * never assumes that a frame arrives in one chunk. It also tolerates CRLF, comment frames
 * (`: ping`) and multi-line `data:` fields, which makes it usable against proxies that
 * coalesce or split writes.
 *
 * The decoder is deliberately dependency-free and synchronous so the streaming tests can
 * feed it scripted chunks and observe exactly what the UI would observe.
 */

export interface SseFrame {
  event: string;
  data: string;
  id?: string;
}

export class SseDecoder {
  private buffer = "";

  /** Feed one decoded text chunk; returns the frames it completed. */
  push(chunk: string): SseFrame[] {
    this.buffer += chunk;
    const frames: SseFrame[] = [];
    // Frames are separated by a blank line. Normalise CRLF/CR first so a proxy that
    // rewrites line endings cannot hide a frame boundary.
    this.buffer = this.buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const frame = parseBlock(block);
      if (frame) frames.push(frame);
      boundary = this.buffer.indexOf("\n\n");
    }
    return frames;
  }

  /** Feed the last chunk; returns any trailing frame without a blank line. */
  flush(): SseFrame[] {
    const block = this.buffer.trim();
    this.buffer = "";
    if (!block) return [];
    const frame = parseBlock(block);
    return frame ? [frame] : [];
  }
}

function parseBlock(block: string): SseFrame | null {
  if (!block.trim()) return null;
  let event = "";
  let id: string | undefined;
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // comment / heartbeat
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id") id = value;
  }
  if (!event && dataLines.length === 0) return null;
  const frame: SseFrame = { event: event || "message", data: dataLines.join("\n") };
  if (id !== undefined) frame.id = id;
  return frame;
}

/** Read an SSE response body, yielding parsed frames as they arrive. */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const sse = new SseDecoder();
  try {
    for (;;) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      for (const frame of sse.push(text)) yield frame;
    }
    for (const frame of sse.flush()) yield frame;
  } finally {
    reader.releaseLock();
  }
}
