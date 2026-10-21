/** Answer card: citations open passages, actions are one click, states are explicit. */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ANSWER_CITATIONS, ANSWER_TEXT, buildAnswer, escalationAnswer, retrievalEvent } from "../api/mock/fixtures";
import { buildInbox } from "../demo/inbox";
import { emptyTurn, type TurnState } from "../state/console";
import { AnswerCard } from "./AnswerCard";

const conversation = buildInbox(Date.UTC(2026, 9, 21, 12, 0, 0))[0]!;

function baseTurn(overrides: Partial<TurnState> = {}): TurnState {
  return { ...emptyTurn(conversation.id), ...overrides };
}

function renderCard(turn: TurnState, props: Partial<React.ComponentProps<typeof AnswerCard>> = {}) {
  const onSend = vi.fn();
  const onEscalate = vi.fn();
  const onRate = vi.fn();
  const onOpenDiagnostics = vi.fn();
  const onRetry = vi.fn();
  render(
    <AnswerCard
      conversation={conversation}
      turn={turn}
      sentText={null}
      feedback={null}
      ticket={null}
      onSend={onSend}
      onEscalate={onEscalate}
      onRate={onRate}
      onOpenDiagnostics={onOpenDiagnostics}
      onRetry={onRetry}
      {...props}
    />,
  );
  return { onSend, onEscalate, onRate, onOpenDiagnostics, onRetry };
}

describe("AnswerCard", () => {
  it("renders streamed text progressively without citation chips", () => {
    const retrieval = retrievalEvent("tr-1", false);
    renderCard(
      baseTurn({
        status: "streaming",
        traceId: "tr-1",
        retrieval,
        streamingText: "Обслуживание карты стоит 120 рублей [1]",
        tokenCount: 6,
      }),
    );
    expect(screen.getByText(/Обслуживание карты стоит 120 рублей/)).toBeInTheDocument();
    expect(screen.getByText("Drafting and verifying claims…")).toBeInTheDocument();
    expect(screen.getByText("6 tokens")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Citation 1/ })).not.toBeInTheDocument();
    expect(screen.getByText("[1]")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send reply" })).toBeDisabled();
  });

  it("shows verified citations, confidence, decision and cache state", () => {
    const answer = buildAnswer({ traceId: "tr-1", cacheHit: false });
    renderCard(baseTurn({ status: "complete", traceId: "tr-1", answer, citationList: undefined } as Partial<TurnState>), {
      feedback: null,
    });
    // The card derives citations from the answer on completion.
    expect(screen.getByRole("button", { name: /Citation 1: Тарифы по карте/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Citation 2:/ })).toBeInTheDocument();
    expect(screen.getByText("98.3%")).toBeInTheDocument();
    expect(screen.getByText(/cache miss/)).toBeInTheDocument();
    expect(screen.getByText(/verified citations/)).toBeInTheDocument();
    expect(screen.getByText("Answered from verified evidence")).toBeInTheDocument();
  });

  it("opens the source passage on click and closes it with Escape", async () => {
    const answer = buildAnswer({ traceId: "tr-1", cacheHit: false });
    renderCard(baseTurn({ status: "complete", traceId: "tr-1", answer }));
    const chip = screen.getByRole("button", { name: /Citation 1:/ });
    expect(chip).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(chip);
    expect(chip).toHaveAttribute("aria-expanded", "true");
    const passage = screen.getByRole("complementary", { name: /Source passage for citation 1/ });
    expect(passage).toHaveTextContent(ANSWER_CITATIONS[0]!.quote);
    expect(passage).toHaveTextContent("chunk-card-fee-01");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("complementary", { name: /Source passage/ })).not.toBeInTheDocument();
  });

  it("supports edit, save and send", async () => {
    const answer = buildAnswer({ traceId: "tr-1", cacheHit: false });
    const { onSend } = renderCard(baseTurn({ status: "complete", traceId: "tr-1", answer }));
    await userEvent.click(screen.getByRole("button", { name: "Edit" }));
    const editor = screen.getByLabelText("Edit the reply");
    expect(editor).toHaveValue(ANSWER_TEXT);
    await userEvent.clear(editor);
    await userEvent.type(editor, "Обслуживание стоит 120 рублей в месяц.");
    await userEvent.click(screen.getByRole("button", { name: "Save edit" }));
    expect(screen.getByText("edited by operator")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Send reply" }));
    expect(onSend).toHaveBeenCalledWith("Обслуживание стоит 120 рублей в месяц.");
  });

  it("marks a sent reply and disables the send action", () => {
    const answer = buildAnswer({ traceId: "tr-1", cacheHit: false });
    renderCard(baseTurn({ status: "complete", traceId: "tr-1", answer }), {
      sentText: "Отправлено",
    });
    expect(screen.getByRole("button", { name: "Send reply" })).toBeDisabled();
    expect(screen.getByText(/sent · marked resolved/)).toBeInTheDocument();
  });

  it("presents an escalation with reason, p-hat and ticket", () => {
    const answer = escalationAnswer("tr-esc", "LOW_CONFIDENCE");
    renderCard(
      baseTurn({
        status: "escalated",
        traceId: "tr-esc",
        answer,
        decision: {
          event: "decision",
          trace_id: "tr-esc",
          action: "escalate",
          reason_code: "LOW_CONFIDENCE",
          score: 0.62,
          alpha: 0.05,
          threshold_ref: "cal-f70b9bf8edb0",
          escalation: {
            ticket_id: "tk-3f2a9b1c8d7e6f50",
            priority: "normal",
            detail: "p_hat 0.6200 < escalate threshold 0.9500",
            reason: "LOW_CONFIDENCE",
          },
        },
        ticketId: "tk-3f2a9b1c8d7e6f50",
      }),
      { ticket: { ticketId: "tk-3f2a9b1c8d7e6f50", reason: "LOW_CONFIDENCE" } },
    );
    expect(screen.getByText(/Needs a human/)).toBeInTheDocument();
    expect(screen.getAllByText(/Below the calibrated confidence band/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("tk-3f2a9b1c8d7e6f50")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ticket/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Write a reply" })).toBeEnabled();
  });

  it("shows a retryable error with a Retry action", async () => {
    const { onRetry } = renderCard(
      baseTurn({
        status: "error",
        traceId: "tr-1",
        error: { code: "provider_timeout", message: "the provider timed out", retryable: true },
      }),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("provider_timeout");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("records feedback and opens diagnostics", async () => {
    const answer = buildAnswer({ traceId: "tr-1", cacheHit: false });
    const { onRate, onOpenDiagnostics } = renderCard(baseTurn({ status: "complete", traceId: "tr-1", answer }));
    await userEvent.click(screen.getByRole("button", { name: "Rate the answer as correct" }));
    expect(onRate).toHaveBeenCalledWith("up");
    await userEvent.click(screen.getByRole("button", { name: "Diagnostics" }));
    expect(onOpenDiagnostics).toHaveBeenCalled();
  });
});
