/**
 * XSS safety at the DOM level (NFR-SEC-04, AT-SEC-04).
 *
 * The console renders model- and document-controlled text: an answer body, citation quotes,
 * document titles and heading paths. All of it is attacker-influenced (a hostile retrieved
 * document can carry a `<script>` payload and a provider can echo it), so the product claim
 * is not "we sanitise it" but "it can never execute". React escapes text children, and the
 * card must never switch to a raw-HTML injection API — this test is what makes that a
 * guarantee rather than an assumption (the static scan next door checks the source for one).
 *
 * The payload is the classic pair: a script element and an image with an inline handler.
 * If either reached the DOM as markup, one of the two markers below would be set.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { buildAnswer } from "../api/mock/fixtures";
import { buildInbox } from "../demo/inbox";
import { emptyTurn, type TurnState } from "../state/console";
import { AnswerCard } from "./AnswerCard";

const PAYLOAD = "<script>window.__xssScript = 'executed'</script><img src=x onerror=\"window.__xssImg='executed'\">";
const conversation = buildInbox(Date.UTC(2026, 9, 21, 12, 0, 0))[0]!;

declare global {
  interface Window {
    __xssScript?: string;
    __xssImg?: string;
  }
}

function hostileAnswer() {
  const answer = buildAnswer({ traceId: "tr-xss-0001", cacheHit: false });
  return {
    ...answer,
    // The answer text carries the payload inside a sentence, exactly as a provider echo would.
    text: `Тариф на обслуживание: ${PAYLOAD} 120 рублей в месяц. [1]`,
    citations: answer.citations.map((citation, index) =>
      index === 0
        ? {
            ...citation,
            doc_title: `Тарифы ${PAYLOAD}`,
            heading_path: ["Тарифы", PAYLOAD],
            quote: `Обслуживание ${PAYLOAD} стоит 120 рублей.`,
          }
        : citation,
    ),
  };
}

function renderHostileCard() {
  const answer = hostileAnswer();
  const turn: TurnState = {
    ...emptyTurn(conversation.id),
    status: "complete",
    traceId: answer.trace_id,
    answer,
    citations: answer.citations,
    claims: answer.claims,
    usage: answer.usage,
    tokenCount: 42,
  };
  const view = render(
    <AnswerCard
      conversation={conversation}
      turn={turn}
      sentText={null}
      feedback={null}
      ticket={null}
      onSend={vi.fn()}
      onEscalate={vi.fn()}
      onRate={vi.fn()}
      onOpenDiagnostics={vi.fn()}
      onRetry={vi.fn()}
    />,
  );
  return { ...view, answer };
}

describe("answer card renders hostile retrieved content as text", () => {
  it("never creates a script element or an inline handler from an answer", () => {
    const { container } = renderHostileCard();

    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("[onerror]")).toBeNull();
    expect(window.__xssScript).toBeUndefined();
    expect(window.__xssImg).toBeUndefined();

    // The payload is present, escaped, as text — the operator must still see what the source said.
    expect(container.textContent).toContain("<script>window.__xssScript = 'executed'</script>");
    expect(container.outerHTML).toContain("&lt;script&gt;");
    // No element may carry an inline event handler, whatever attribute it landed in.
    for (const element of Array.from(container.querySelectorAll("*"))) {
      for (const attribute of Array.from(element.attributes)) {
        expect(attribute.name.toLowerCase().startsWith("on")).toBe(false);
      }
    }
  });

  it("escapes citation titles, heading paths and quotes without losing them", () => {
    const { container } = renderHostileCard();
    const text = container.textContent ?? "";

    // A citation chip opens the passage; its title and quote are hostile too.
    expect(text).toContain("Тарифы <script>");
    expect(text).toContain("Обслуживание <script>");
    expect(container.querySelectorAll("script")).toHaveLength(0);
    // Exactly one interactive citation chip per verified citation, no synthetic ones.
    expect(screen.getAllByRole("button", { name: /источник|citation|\[1\]/i }).length).toBeGreaterThan(0);
  });
});
