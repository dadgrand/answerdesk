#!/usr/bin/env node
/**
 * Playwright screenshots of the operator console (CL-UI-02).
 *
 * One command, six screens, reproducible output:
 *
 *   npm run screenshots            # prefers the REAL Python service (offline --fake profile)
 *   npm run screenshots -- --mock  # MSW fixtures, no Python needed
 *   npm run screenshots -- --base http://127.0.0.1:8765  # an already-running service
 *
 * The script:
 *  1. builds `dist/` if it is missing;
 *  2. starts `answerdesk serve --fake` when `uv` is available (real SSE, real traces, real
 *     metrics), otherwise falls back to MSW mock mode and says so;
 *  3. serves `dist/` with a tiny static server that proxies `/v1`, `/healthz` and
 *     `/metrics` to the service, so the console runs same-origin;
 *  4. drives Chromium through the console (answer → citation → diagnostics → escalation →
 *     light theme → analytics) and writes PNGs + a manifest to `docs/assets/`.
 *
 * The service state (SQLite, traces) goes to `web/.screenshots/`, which is git-ignored:
 * a screenshot run must never leave runtime state in the repository.
 */

import { spawn, spawnSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, "..");
const REPO_ROOT = resolve(WEB_ROOT, "..");
const DIST = join(WEB_ROOT, "dist");
const STATE_DIR = join(WEB_ROOT, ".screenshots");
const VIEWPORT = { width: 1560, height: 980 };
const SCALE = 1.5;

const args = process.argv.slice(2);
const argValue = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index === -1 ? fallback : args[index + 1] ?? fallback;
};

const OUT_DIR = argValue("out", join(REPO_ROOT, "docs", "assets"));
const forceMock = args.includes("--mock");
const externalBase = argValue("base", null);
const servicePort = Number(argValue("service-port", 8765));
const serverPort = Number(argValue("port", 4173));

const log = (message) => console.log(`[screenshots] ${message}`);
let activeService = null;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".map": "application/json",
};

function ensureBuild() {
  if (existsSync(join(DIST, "index.html"))) return;
  log("dist/ is missing — building the console first");
  const result = spawnSync("npm", ["run", "build"], {
    cwd: WEB_ROOT,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  if (result.status !== 0) {
    throw new Error("npm run build failed; cannot take screenshots");
  }
}

async function ping(url, timeoutMs = 1500) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    return response.ok;
  } catch {
    return false;
  }
}

async function waitFor(url, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await ping(url)) return true;
    await new Promise((done) => setTimeout(done, 250));
  }
  throw new Error(`timed out waiting for ${label} at ${url}`);
}

function startService() {
  const env = { ...process.env };
  for (const key of ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]) {
    delete env[key];
  }
  env.ANSWERDESK_SERVICE_DB_PATH = join(STATE_DIR, "service.db");
  env.ANSWERDESK_SERVICE_TRACE_DIR = join(STATE_DIR, "traces");
  env.ANSWERDESK_LOGGING__LEVEL = "WARNING";
  mkdirSync(STATE_DIR, { recursive: true });

  const child = spawn(
    "uv",
    ["run", "--extra", "api", "answerdesk", "serve", "--fake", "--port", String(servicePort)],
    { cwd: REPO_ROOT, env, stdio: ["ignore", "pipe", "pipe"], shell: process.platform === "win32" },
  );
  let stderr = "";
  child.stderr?.on("data", (chunk) => {
    stderr += String(chunk);
    if (stderr.length > 4000) stderr = stderr.slice(-4000);
  });
  child.on("error", () => undefined);
  return { child, getStderr: () => stderr };
}

function stopService(service) {
  if (!service) return;
  const { child } = service;
  if (!child || child.killed) return;
  try {
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      child.kill("SIGTERM");
    }
  } catch {
    /* best effort */
  }
}

/** Static file server for dist/ with an optional same-origin proxy to the service. */
function startStaticServer(port, proxyTarget) {
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    const pathname = url.pathname;

    if (proxyTarget && (pathname.startsWith("/v1") || pathname === "/healthz" || pathname === "/metrics")) {
      const upstream = new URL(proxyTarget);
      const proxyReq = {
        hostname: upstream.hostname,
        port: upstream.port || 80,
        path: `${pathname}${url.search}`,
        method: request.method,
        headers: { ...request.headers, host: upstream.host },
      };
      import("node:http").then(({ request: httpRequest }) => {
        const forwarded = httpRequest(proxyReq, (upstreamResponse) => {
          response.writeHead(upstreamResponse.statusCode ?? 502, upstreamResponse.headers);
          upstreamResponse.pipe(response);
        });
        forwarded.on("error", () => {
          response.writeHead(502, { "content-type": "application/json" });
          response.end(JSON.stringify({ error: { code: "bad_gateway", message: "service unreachable" } }));
        });
        request.pipe(forwarded);
      });
      return;
    }

    let filePath = normalize(join(DIST, decodeURIComponent(pathname)));
    if (!filePath.startsWith(DIST)) {
      response.writeHead(403).end("forbidden");
      return;
    }
    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      filePath = join(DIST, "index.html");
    }
    const type = MIME[extname(filePath)] ?? "application/octet-stream";
    response.writeHead(200, { "content-type": type, "cache-control": "no-store" });
    createReadStream(filePath).pipe(response);
  });
  return new Promise((done) => {
    server.listen(port, "127.0.0.1", () => done(server));
  });
}

async function launchBrowser() {
  const channels = [process.env.PLAYWRIGHT_CHANNEL, "chrome", "msedge"].filter(Boolean);
  let lastError;
  for (const channel of channels) {
    try {
      const browser = await chromium.launch({ channel, headless: true });
      log(`browser: ${channel}`);
      return browser;
    } catch (error) {
      lastError = error;
    }
  }
  try {
    const browser = await chromium.launch({ headless: true });
    log("browser: bundled chromium");
    return browser;
  } catch (error) {
    throw new Error(
      `no Chromium available (${lastError?.message ?? error.message}). ` +
        "Install Chrome/Edge, or run `npx playwright install chromium`.",
    );
  }
}

async function shoot(page, name, description, manifest) {
  const target = join(OUT_DIR, `${name}.png`);
  // Move the pointer away so a hover tooltip (e.g. the citation preview) does not sit on
  // top of the content in the committed screenshot.
  await page.mouse.move(4, 4);
  await page.waitForTimeout(150);
  await page.screenshot({ path: target, fullPage: false });
  manifest.screens.push({ file: relative(REPO_ROOT, target).split(sep).join("/"), description });
  log(`saved ${relative(REPO_ROOT, target).split(sep).join("/")}`);
}

function writeManifest(manifest) {
  writeFileSync(join(OUT_DIR, "screenshots.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

async function main() {
  ensureBuild();
  mkdirSync(OUT_DIR, { recursive: true });
  rmSync(STATE_DIR, { recursive: true, force: true });

  let mode = "mock";
  let service = null;
  let baseUrl = externalBase;

  if (baseUrl) {
    await waitFor(`${baseUrl}/healthz`, 5000, "the provided service");
    mode = "live";
    log(`using the provided service at ${baseUrl}`);
  } else if (!forceMock) {
    const alreadyRunning = await ping(`http://127.0.0.1:${servicePort}/healthz`);
    if (alreadyRunning) {
      mode = "live";
      baseUrl = `http://127.0.0.1:${servicePort}`;
      log(`reusing the service already listening on ${baseUrl}`);
    } else {
      service = startService();
      activeService = service;
      try {
        await waitFor(`http://127.0.0.1:${servicePort}/healthz`, 90_000, "answerdesk serve");
        mode = "live";
        baseUrl = `http://127.0.0.1:${servicePort}`;
        log("started the real service (offline --fake profile)");
      } catch (error) {
        stopService(service);
        service = null;
        log(`real service unavailable (${String(error.message).split("\n")[0]}); using MSW mock mode`);
      }
    }
  } else {
    log("--mock requested: using MSW fixtures");
  }

  const appPort = serverPort;
  const server = await startStaticServer(appPort, mode === "live" ? baseUrl : null);
  const appBase = `http://127.0.0.1:${appPort}`;
  const consoleUrl = (conversation) => {
    const params = new URLSearchParams({ conversation });
    if (mode !== "live") params.set("mock", "1");
    return `${appBase}/#/console?${params.toString()}`;
  };

  const browser = await launchBrowser();
  const manifest = {
    generatedAt: new Date().toISOString(),
    mode,
    service: mode === "live" ? baseUrl : "msw",
    app: appBase,
    viewport: VIEWPORT,
    deviceScaleFactor: SCALE,
    screens: [],
  };

  try {
    const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: SCALE, locale: "en-US" });
    page.setDefaultTimeout(20_000);
    page.on("pageerror", (error) => log(`pageerror: ${error.message}`));

    // 1. The console with a verified, cited answer (dark theme), deep-linked to the queue item.
    await page.goto(`${consoleUrl("c-1042")}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector(".answer--complete", { timeout: 30_000 });
    await page.waitForSelector(".cite", { timeout: 15_000 });
    await page.waitForTimeout(400);
    await shoot(page, "console-answer", "Queue + conversation with a verified answer and inline citation chips", manifest);

    // 2. Citation → source passage.
    await page.locator(".cite").first().click();
    await page.waitForSelector(".passage", { timeout: 10_000 });
    await page.waitForTimeout(250);
    await shoot(page, "console-citation", "Citation clicked: the source passage with document, heading and span", manifest);

    // 3. Diagnostics drawer (retrieval scores, cache, decision, latency, cost).
    await page.getByRole("button", { name: "Diagnostics", exact: true }).click();
    await page.waitForSelector(".drawer", { timeout: 10_000 });
    await page.waitForSelector(".retrieval__item", { timeout: 15_000 });
    await page.waitForTimeout(400);
    await shoot(page, "console-diagnostics", "Diagnostics drawer: decision, retrieval candidates with scores, cache, latency, cost", manifest);
    await page.getByRole("button", { name: "Close diagnostics" }).click();

    // 4. An escalated conversation (reason, p-hat, ticket, actions).
    await page.getByRole("button", { name: /Mehrzod Sattorov/ }).click();
    await page.waitForSelector(".answer--escalated", { timeout: 30_000 });
    await page.waitForTimeout(400);
    await shoot(page, "console-escalation", "Escalated turn: reason code, p-hat against the band, ticket and manual reply", manifest);

    // 5. Light theme, back on the answered conversation as a *returning* session (`session`
    // deep link): the service answers the same question from the exact cache.
    await page.goto(`${consoleUrl("c-1042")}&session=returning`, { waitUntil: "domcontentloaded" });
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForSelector(".answer--complete", { timeout: 30_000 });
    await page.getByRole("button", { name: /Light/ }).click();
    await page.waitForTimeout(400);
    log(`cache chip after reload: ${await page.locator(".chip--cache").first().innerText()}`);
    await shoot(page, "console-light", "Light theme; a returning session gets the same question from the exact cache hit", manifest);

    if (mode === "live") {
      try {
        const text = await (await fetch(`${baseUrl}/metrics`)).text();
        manifest.metrics = {
          cacheHits: text.match(/^answerdesk_cache_hits_total.*$/gm) ?? [],
          cacheMisses: text.match(/^answerdesk_cache_misses_total.*$/gm) ?? [],
          decisions: text.match(/^answerdesk_decisions_total.*$/gm) ?? [],
        };
      } catch {
        /* the manifest is optional evidence */
      }
    }

    // 6. Analytics from the live metrics scrape.
    await page.getByRole("button", { name: "Analytics", exact: true }).click();
    await page.waitForSelector(".kpi", { timeout: 20_000 });
    await page.waitForTimeout(500);
    await shoot(page, "analytics", "Analytics: latency distribution, p50/p95, USD per answer, escalation rate, cache hit rate", manifest);

    writeManifest(manifest);
    log(`done: ${manifest.screens.length} screens in docs/assets (mode=${mode})`);
  } finally {
    await browser.close();
    server.close();
    stopService(service);
    if (!process.exitCode) rmSync(STATE_DIR, { recursive: true, force: true });
    else log(`keeping ${STATE_DIR} for debugging`);
  }
}

main().catch((error) => {
  console.error(`[screenshots] failed: ${error.stack ?? error.message}`);
  console.error(`[screenshots] service stderr tail: ${serviceStderr()}`);
  process.exitCode = 1;
});

function serviceStderr() {
  return activeService?.getStderr() ?? "(no service started)";
}
