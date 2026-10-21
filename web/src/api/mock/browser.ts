/** MSW browser worker: starts mock mode in `npm run dev` (no backend needed). */

import { setupWorker } from "msw/browser";

import { mockHandlers } from "./handlers";

let worker: ReturnType<typeof setupWorker> | null = null;

export async function startMockWorker(): Promise<void> {
  if (worker) return;
  worker = setupWorker(...mockHandlers);
  await worker.start({
    serviceWorker: { url: "/mockServiceWorker.js" },
    onUnhandledRequest: "bypass",
    quiet: true,
  });
}

export function stopMockWorker(): void {
  worker?.stop();
  worker = null;
}
