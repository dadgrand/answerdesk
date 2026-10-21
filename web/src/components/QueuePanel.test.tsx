/** Queue rendering: language badge, SLA state, selection and filters. */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { buildInbox } from "../demo/inbox";
import { QueuePanel } from "./QueuePanel";

const NOW = Date.UTC(2026, 9, 21, 12, 0, 0);
const inbox = buildInbox(NOW);

function renderQueue(overrides: Partial<React.ComponentProps<typeof QueuePanel>> = {}) {
  const onSelect = vi.fn();
  const onLocaleFilter = vi.fn();
  const onSlaFilter = vi.fn();
  render(
    <QueuePanel
      conversations={inbox}
      selectedId="c-1042"
      now={NOW}
      turns={{}}
      localeFilter="all"
      slaFilter="all"
      onSelect={onSelect}
      onLocaleFilter={onLocaleFilter}
      onSlaFilter={onSlaFilter}
      {...overrides}
    />,
  );
  return { onSelect, onLocaleFilter, onSlaFilter };
}

describe("QueuePanel", () => {
  it("lists every conversation with a language badge and channel", () => {
    renderQueue();
    expect(screen.getByText("Aisha Rahimova")).toBeInTheDocument();
    expect(screen.getByText("Mehrzod Sattorov")).toBeInTheDocument();
    expect(screen.getByText("Li Wei")).toBeInTheDocument();
    expect(screen.getAllByText("Telegram").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByTitle(/Locale:/).map((node) => node.textContent)).toEqual([
      "RU",
      "TG",
      "UZ",
      "KY",
      "TR",
      "ZH",
      "RU",
    ]);
  });

  it("shows the SLA age and state", () => {
    renderQueue();
    expect(screen.getByText("00:42")).toBeInTheDocument(); // 42 s, within SLA
    expect(screen.getByText("01:28")).toBeInTheDocument(); // 88 s, at risk
    expect(screen.getAllByTitle("SLA breached").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByTitle("SLA at risk").length).toBeGreaterThanOrEqual(1);
  });

  it("marks the selected conversation for assistive technology", () => {
    renderQueue();
    const selected = screen.getByRole("button", { name: /Aisha Rahimova/ });
    expect(selected).toHaveAttribute("aria-current", "true");
    expect(selected.textContent).toContain("Aisha Rahimova");
  });

  it("selects a conversation on click", async () => {
    const { onSelect } = renderQueue();
    await userEvent.click(screen.getByRole("button", { name: /Mehrzod Sattorov/ }));
    expect(onSelect).toHaveBeenCalledWith("c-1043");
  });

  it("renders per-conversation turn status tags", () => {
    renderQueue({
      turns: {
        "c-1042": {
          conversationId: "c-1042",
          status: "complete",
          traceId: "tr-1",
          retrieval: null,
          streamingText: "",
          citations: [],
          claims: [],
          decision: null,
          answer: null,
          usage: null,
          timings: {},
          ticketId: null,
          error: null,
          tokenCount: 0,
        },
      },
    });
    expect(screen.getAllByText("answered").length).toBeGreaterThan(0);
  });

  it("exposes filter controls as pressed buttons and a language select", async () => {
    const { onSlaFilter, onLocaleFilter } = renderQueue();
    const atRisk = screen.getByRole("button", { name: "At risk" });
    expect(atRisk).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(atRisk);
    expect(onSlaFilter).toHaveBeenCalledWith("at-risk");

    await userEvent.selectOptions(screen.getByLabelText("Filter by language"), "tg");
    expect(onLocaleFilter).toHaveBeenCalledWith("tg");
  });

  it("shows an empty state when the filter matches nothing", () => {
    renderQueue({ conversations: [] });
    expect(screen.getByText("Nothing in this filter")).toBeInTheDocument();
  });
});
