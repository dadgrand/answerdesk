/**
 * Entry point.
 *
 * The mock service worker (when enabled) must be installed *before* the first request, so
 * the bootstrap awaits it and only then renders. That is the difference between an offline
 * demo that works and one that flashes "service unreachable" for a second.
 *
 * Deliberately **not** wrapped in `<StrictMode>`: React 18's development double-mount
 * aborts the first SSE stream and immediately restarts it, doubling the model calls a demo
 * makes. The streaming path is covered by the component tests instead.
 */

import { createRoot } from "react-dom/client";

import { App } from "./App";
import { maybeStartMock } from "./mock";
import "./styles.css";

async function bootstrap(): Promise<void> {
  try {
    await maybeStartMock();
  } catch (error) {
    console.warn("mock service worker failed to start; the console will talk to the real service", error);
  }
  const container = document.getElementById("root");
  if (!container) throw new Error("missing #root");
  createRoot(container).render(<App />);
}

void bootstrap();
