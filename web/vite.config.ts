/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

/**
 * The console talks to the answerdesk service on the same origin in production and through
 * the Vite dev proxy in development, so no CORS configuration is needed for `npm run dev`.
 * `npm run preview` serves the built app; the Playwright screenshot script starts the real
 * Python service and points the console at it through `VITE_API_BASE`.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1": { target: "http://127.0.0.1:8765", changeOrigin: false },
      "/healthz": { target: "http://127.0.0.1:8765", changeOrigin: false },
      "/metrics": { target: "http://127.0.0.1:8765", changeOrigin: false },
    },
  },
  preview: { port: 4173, strictPort: true },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2022",
    reportCompressedSize: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
    unstubGlobals: true,
    include: ["src/**/*.test.{ts,tsx}"],
    reporters: ["default"],
  },
});
