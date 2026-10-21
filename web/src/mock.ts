/**
 * Optional MSW mock mode.
 *
 * Enabled by `VITE_USE_MOCK=true` (build/dev) or `?mock=1` (any build, inside the hash
 * route as well), so the console can be shown and screenshotted without a Python service.
 * In mock mode every request is answered by `src/api/mock/handlers.ts` — the same handlers
 * the vitest suite uses.
 */

import { urlParam } from "./lib/url";

export function mockRequested(): boolean {
  const fromEnv = (import.meta.env.VITE_USE_MOCK as string | undefined) === "true";
  if (fromEnv) return true;
  try {
    return urlParam("mock") === "1";
  } catch {
    return false;
  }
}

export async function maybeStartMock(): Promise<boolean> {
  if (!mockRequested()) return false;
  // Imported lazily so the MSW browser bundle is not shipped in the production chunk.
  const { startMockWorker } = await import("./api/mock/browser");
  await startMockWorker();
  return true;
}
