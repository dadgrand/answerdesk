/** Analytics page: KPIs from the exposition, refresh, and a clear failure state. */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { mockServer } from "../api/mock/server";
import { AnalyticsPage } from "./AnalyticsPage";

describe("AnalyticsPage", () => {
  it("renders the KPI cards derived from /metrics", async () => {
    render(<AnalyticsPage />);
    expect(await screen.findByText("Requests", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText("113")).toBeInTheDocument(); // requests total
    expect(screen.getByText("Escalation rate")).toBeInTheDocument();
    expect(screen.getByText("23.4%")).toBeInTheDocument();
    expect(screen.getByText("Cache hit rate")).toBeInTheDocument();
    expect(screen.getByText("37.1%")).toBeInTheDocument();
    expect(screen.getByText("USD / answer")).toBeInTheDocument();
    expect(screen.getAllByText("$0.0045").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("p95 latency")).toBeInTheDocument();
  });

  it("renders the charts (latency distribution, decision mix, cache, stages)", async () => {
    render(<AnalyticsPage />);
    await screen.findByText("Latency distribution", {}, { timeout: 3000 });
    expect(screen.getByRole("img", { name: "Latency distribution histogram" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Decision mix" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /hit rate/ })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Latency p95 by stage" })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Spend by model" })).toBeInTheDocument();
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
  });

  it("refreshes the scrape on demand", async () => {
    render(<AnalyticsPage />);
    await screen.findByText("Requests", {}, { timeout: 3000 });
    const refresh = screen.getByRole("button", { name: "Refresh" });
    await userEvent.click(refresh);
    expect(await screen.findByText("Requests", {}, { timeout: 3000 })).toBeInTheDocument();
  });

  it("surfaces a metrics failure with a retry", async () => {
    mockServer.use(
      http.get("*/metrics", () =>
        HttpResponse.json(
          { error: { code: "service_unavailable", message: "no metrics", trace_id: "tr-x", retryable: true } },
          { status: 503 },
        ),
      ),
    );
    render(<AnalyticsPage />);
    expect(await screen.findByRole("alert", {}, { timeout: 3000 })).toHaveTextContent("service_unavailable");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
