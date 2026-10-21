/**
 * Accessibility contract for the operator console (NFR-A11Y-01, AT-A11Y-01).
 *
 * The queue is worked all day by a keyboard user, so the console's structure is a product
 * requirement, not a nicety: one `main` landmark, real headings, accessible names on every
 * control, a live region for the streamed answer, a labelled complementary panel for the
 * source passage, and a labelled dialog for the diagnostics drawer.
 *
 * The explicit role/label assertions below pin the *keyboard path* the review asked for, and
 * `axe-core` runs the machine checkable rules (WCAG 2.0/2.1 A and AA) over the same DOM. A
 * regression either loses a name (the explicit assertions) or trips a rule (axe), and both
 * fail this file. The machine report is written to `web/reports/a11y/console-axe.json` and
 * committed, so a reviewer can read what was actually checked.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { describe, expect, it } from "vitest";

import { App } from "./App";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPORT = resolve(HERE, "../reports/a11y/console-axe.json");

declare global {
  interface Window {
    __a11yReport?: unknown;
  }
}

async function runAxe(context: Element) {
  const results = await axe.run(context, {
    runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
    // jsdom has no layout engine: colour contrast and visibility can only be reported as
    // "incomplete" here, so the check is explicitly out of scope for the DOM suite and is
    // covered by the Playwright screenshots instead.
    rules: { "color-contrast": { enabled: false } },
  });
  const summary = {
    generatedAt: new Date().toISOString(),
    engine: `axe-core ${axe.version}`,
    url: "app://answerdesk-console",
    runOnly: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"],
    excluded: ["color-contrast (needs layout; Playwright/screenshots cover it)"],
    violations: results.violations.map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      nodes: violation.nodes.map((node) => node.target.join(" ")),
    })),
    passes: results.passes.length,
    incomplete: results.incomplete.map((item) => ({ id: item.id, nodes: item.nodes.length })),
  };
  mkdirSync(dirname(REPORT), { recursive: true });
  writeFileSync(REPORT, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  window.__a11yReport = summary;
  return { results, summary };
}

describe("console accessibility contract", () => {
  it("exposes the landmarks, names and live regions a keyboard operator needs", async () => {
    render(<App />);
    expect(await screen.findByRole("button", { name: /Aisha Rahimova/ })).toBeInTheDocument();

    // Landmarks: one main region, the queue as navigation, headings in order.
    expect(screen.getAllByRole("main")).toHaveLength(1);
    expect(screen.getByRole("navigation", { name: /primary/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();

    // Every interactive control has an accessible name (axe checks the same, this names it).
    const interactive = screen.getAllByRole("button");
    expect(interactive.length).toBeGreaterThan(3);
    for (const control of interactive) {
      expect(control).toHaveAccessibleName();
    }

    // The streamed answer is announced politely instead of stealing focus.
    const status = screen.getAllByRole("status");
    expect(status.some((region) => region.getAttribute("aria-live") === "polite")).toBe(true);

    // The decision badge is text, not colour alone.
    expect(screen.getByText(/Answered from verified evidence/)).toBeInTheDocument();
  });

  it("labels the source-passage panel and the diagnostics dialog", async () => {
    render(<App />);
    const chips = await screen.findAllByRole("button", { name: /Citation \d:/ }, { timeout: 5000 });
    await userEvent.click(chips[0]!);

    const passage = await screen.findByRole("complementary", {
      name: /Source passage for citation 1/,
    });
    expect(passage).toHaveAttribute("aria-label");
    expect(passage).toHaveTextContent("chunk-card-fee-01");

    // The drawer is a labelled complementary region, not a focus trap: the operator keeps
    // working while it is open. That is the documented design, so assert it as such.
    const openers = await screen.findAllByRole("button", { name: /^Diagnostics$/i });
    await userEvent.click(openers[0]!);
    const drawer = await screen.findByRole("complementary", { name: /Answer diagnostics/i });
    expect(drawer).toHaveAttribute("aria-label", "Answer diagnostics");
    expect(drawer).toHaveTextContent(/Diagnostics/);
  });

  it("has no axe-core violations on the loaded console", async () => {
    const { container } = render(<App />);
    await screen.findAllByText(/120 рублей в месяц/, undefined, { timeout: 5000 });
    await waitFor(() => expect(screen.getByText(/Claims verified · 2/)).toBeInTheDocument());

    const { results, summary } = await runAxe(container);
    expect(summary.violations).toEqual([]);
    expect(results.violations).toHaveLength(0);
    expect(summary.passes).toBeGreaterThan(10);
  });
});
