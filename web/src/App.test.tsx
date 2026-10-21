/**
 * End-to-end console behaviour against MSW (the same handlers as mock mode).
 *
 * This is the test that proves the acceptance path: queue → streamed grounded answer →
 * clickable citation → source passage → diagnostics from the persisted trace → operator
 * actions → analytics. The transport is intercepted at the fetch boundary, so the React
 * code, the SSE decoder and the API client are all real.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { App } from "./App";
import { mockServer } from "./api/mock/server";

describe("operator console", () => {
  it("streams a grounded answer with verified citation chips that open the passage", async () => {
    render(<App />);

    expect(await screen.findByRole("button", { name: /Aisha Rahimova/ })).toBeInTheDocument();
    // The draft streams in and settles on the verified answer.
    expect((await screen.findAllByText(/120 рублей в месяц/, undefined, { timeout: 5000 })).length).toBeGreaterThan(0);
    expect(await screen.findByText(/sent · marked resolved|2 verified citations/)).toBeInTheDocument();

    const chips = await screen.findAllByRole("button", { name: /Citation \d:/ });
    expect(chips).toHaveLength(2);

    await userEvent.click(chips[0]!);
    const passage = await screen.findByRole("complementary", { name: /Source passage for citation 1/ });
    expect(passage).toHaveTextContent("Тарифы по карте «Вектор»");
    expect(passage).toHaveTextContent("Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.");
    expect(passage).toHaveTextContent("chunk-card-fee-01");
  });

  it("shows the decision, confidence, cache state and the diagnostics from the trace", async () => {
    render(<App />);
    await screen.findAllByText(/120 рублей в месяц/, undefined, { timeout: 5000 });

    expect(screen.getByText("98.3%")).toBeInTheDocument();
    expect(screen.getByText(/Claims verified · 2/)).toBeInTheDocument();
    expect(screen.getByText("Answered from verified evidence")).toBeInTheDocument();
    expect(screen.getByText(/cache miss/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Diagnostics" }));
    const drawer = await screen.findByRole("complementary", { name: "Answer diagnostics" });

    expect(within(drawer).getByText("Policy decision")).toBeInTheDocument();
    expect(within(drawer).getByText("ANSWERED")).toBeInTheDocument();
    expect(within(drawer).getAllByText("cal-f70b9bf8edb0").length).toBeGreaterThanOrEqual(1);

    // The trace arrives asynchronously: candidates, cache and stage latency come from it.
    expect(await within(drawer).findByText("chunk-card-fee-01", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(within(drawer).getByText("Retrieval candidates")).toBeInTheDocument();
    expect(within(drawer).getByText("Semantic cache")).toBeInTheDocument();
    expect(within(drawer).getByText("Latency by stage")).toBeInTheDocument();
    expect(within(drawer).getByText("generation")).toBeInTheDocument();
    expect(within(drawer).getByText(/repaired 0×/)).toBeInTheDocument();
  });

  it("escalates and records feedback with one action each", async () => {
    render(<App />);
    await screen.findAllByText(/120 рублей в месяц/, undefined, { timeout: 5000 });

    await userEvent.click(screen.getByRole("button", { name: "Rate the answer as correct" }));
    expect(await screen.findByText(/Feedback recorded against the trace/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Escalate" }));
    expect(await screen.findByText(/Escalation ticket tk-3f2a9b1c8d7e6f50 created/)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /Ticket tk-3f2/ })).toBeInTheDocument();
  });

  it("switches conversations and shows an escalation for a low-confidence turn", async () => {
    render(<App />);
    await screen.findByRole("button", { name: /Aisha Rahimova/ });
    await userEvent.click(await screen.findByRole("button", { name: /Mehrzod Sattorov/ }));

    expect(await screen.findByText(/Needs a human/, undefined, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByText(/62\.0% is below the 95\.0% band/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Write a reply" })).toBeInTheDocument();
  });

  it("navigates to analytics and charts the metrics from the API", async () => {
    render(<App />);
    await screen.findByRole("button", { name: /Aisha Rahimova/ });
    await userEvent.click(screen.getByRole("button", { name: "Analytics" }));

    expect(await screen.findByRole("heading", { name: "Service analytics" })).toBeInTheDocument();
    expect(await screen.findByText("Escalation rate", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText("23.4%")).toBeInTheDocument();
    expect(screen.getByText("37.1%")).toBeInTheDocument();
    expect(screen.getAllByText("$0.0045").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Decision mix")).toBeInTheDocument();
    expect(screen.getByText("Semantic cache")).toBeInTheDocument();
    expect(screen.getByText("Latency distribution")).toBeInTheDocument();
  });

  it("toggles the theme and persists it", async () => {
    render(<App />);
    await screen.findByRole("button", { name: /Aisha Rahimova/ });
    expect(document.documentElement.dataset.theme).toBe("dark");
    await userEvent.click(screen.getByRole("button", { name: /Light/ }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem("answerdesk.theme")).toBe("light");
  });

  it("shows a retryable error banner when the service fails", async () => {
    mockServer.use(
      http.post("*/v1/chat", () =>
        HttpResponse.json(
          {
            error: {
              code: "provider_error",
              message: "provider returned 500",
              trace_id: "tr-x",
              retryable: true,
            },
          },
          { status: 502 },
        ),
      ),
    );
    render(<App />);
    const alert = await screen.findByRole("alert", {}, { timeout: 5000 });
    expect(alert).toHaveTextContent("provider_error");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("shows the service identity in the top bar", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("fake-model")).toBeInTheDocument());
    expect(screen.getByTitle(/answerdesk 0\.1\.0/)).toBeInTheDocument();
  });
});
