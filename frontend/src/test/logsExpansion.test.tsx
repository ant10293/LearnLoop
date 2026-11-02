import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../lib/api";
import { BundleContext } from "../lib/bundleContext";
import type { Bundle } from "../lib/types";
import { LogsPage } from "../pages/RunPages";

const bundle = { run: { id: "run-1" } } as unknown as Bundle;

const requests = [
  { id: "req-1", stage: "first", model_id: "m", status: "completed", latency_ms: 1, response_text: "first body" },
  { id: "req-2", stage: "second", model_id: "m", status: "completed", latency_ms: 1, response_text: "second body" },
];

describe("logs table expansion", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("keeps one request expanded at a time", async () => {
    vi.spyOn(api, "requests").mockResolvedValue(requests as never);
    render(
      <QueryClientProvider client={new QueryClient()}>
        <BundleContext.Provider value={bundle}>
          <LogsPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );

    const first = await screen.findByRole("button", { name: "First" });
    const second = screen.getByRole("button", { name: "Second" });

    fireEvent.click(first);
    expect(screen.getByText("first body")).toBeInTheDocument();

    fireEvent.click(second);
    expect(screen.queryByText("first body")).not.toBeInTheDocument();
    expect(screen.getByText("second body")).toBeInTheDocument();
    expect(first).toHaveAttribute("aria-expanded", "false");
    expect(second).toHaveAttribute("aria-expanded", "true");

    fireEvent.click(second);
    expect(screen.queryByText("second body")).not.toBeInTheDocument();
    expect(second).toHaveAttribute("aria-expanded", "false");
  });
});
