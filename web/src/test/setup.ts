/**
 * Vitest setup: jsdom polyfills + the MSW node server.
 *
 * Each test starts from the same state: handlers reset, mock traces cleared, storage
 * cleared. That is what makes the streaming tests deterministic — a leftover cached answer
 * from a previous test would flip a `miss` into a `hit` and the assertion would be about
 * the test order instead of the code.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, beforeEach, vi } from "vitest";

import { resetMockState } from "../api/mock/handlers";
import { mockServer } from "../api/mock/server";

beforeAll(() => {
  mockServer.listen({ onUnhandledRequest: "error" });
});

afterEach(() => {
  cleanup();
  mockServer.resetHandlers();
  resetMockState();
  window.localStorage.clear();
  window.location.hash = "";
});

afterAll(() => {
  mockServer.close();
});

beforeEach(() => {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList,
  );
  Element.prototype.scrollIntoView = vi.fn();
});
