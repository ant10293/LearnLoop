import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { BundleContext } from "../lib/bundleContext";
import type { Bundle } from "../lib/types";
import { ObjectsPage } from "../pages/RunPages";

const bundle = {
  run: { id: "run-1" },
  objects: [
    {
      id: "obj-1",
      local_id: "Object_1",
      label: "rod",
      description: "",
      initial_state: "",
      final_state: "",
      uncertain_identity: false,
      observation_count: 2,
      color: "#fff",
      primary_image_url: "/thumb.png",
      images: [{ id: "img-1", kind: "initial", timestamp: 0, url: "/thumb.png", score: 1 }],
    },
  ],
  observations: [
    {
      id: "obs-1",
      object_local_id: "Object_1",
      timestamp: 0,
      bbox: { x: 0, y: 0, w: 1, h: 1 },
      frame_id: "f1",
      visibility: "visible",
      confidence: 0.42,
    },
    {
      id: "obs-2",
      object_local_id: "Object_1",
      timestamp: 2,
      bbox: { x: 0, y: 0, w: 1, h: 1 },
      frame_id: "f2",
      visibility: "visible",
      confidence: null,
    },
  ],
} as unknown as Bundle;

describe("confidence on the object image grid", () => {
  afterEach(() => cleanup());

  it("overlays the percent on the top right and skips a still with no score", () => {
    const client = new QueryClient();
    render(
      <QueryClientProvider client={client}>
        <BundleContext.Provider value={bundle}>
          <ObjectsPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: /Rod/ }));

    const score = screen.getByText("42%");
    expect(score).toHaveClass("shot-score");
    expect(score.parentElement).toHaveClass("shot-frame");
    expect(score.previousElementSibling?.tagName).toBe("IMG");
    expect(screen.getByRole("button", { name: "View Initial · 0.00s, 42% confidence" })).toBeInTheDocument();

    const quiet = screen.getByRole("button", { name: "View 2.00s" });
    expect(quiet.querySelector(".shot-score")).toBeNull();
  });

  it("shows a square crop for each frame and a blank box where the object is missing", () => {
    const client = new QueryClient();
    const withFrames = {
      ...bundle,
      frames: [
        { id: "f1", timestamp: 0, reason: "uniform", url: "/f1" },
        { id: "gap", timestamp: 1, reason: "uniform", url: "/gap" },
        { id: "f2", timestamp: 2, reason: "uniform", url: "/f2" },
      ],
    } as unknown as Bundle;
    render(
      <QueryClientProvider client={client}>
        <BundleContext.Provider value={withFrames}>
          <ObjectsPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: /Rod/ }));

    const strip = screen.getByRole("list", { name: "Rod timeline" });
    const firstShot = screen.getByRole("button", { name: "View Initial · 0.00s, 42% confidence" });
    expect(strip.compareDocumentPosition(firstShot) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    const cells = within(strip).getAllByRole("listitem");
    expect(cells).toHaveLength(3);
    expect(cells[0]).toHaveAttribute("aria-label", "Frame at 0:00.00");
    expect(cells[0]).toHaveTextContent("0:00.00");
    const shot = cells[0].querySelector("img");
    expect(shot).toHaveAttribute("src", "/api/media/observations/obs-1/crop");
    expect(shot).toHaveClass("object-cover", "object-center");
    expect(shot?.parentElement).toHaveClass("h-14", "w-14", "border-0");
    expect(shot?.parentElement?.className).not.toMatch(/rounded/);
    expect(cells[1]).toHaveAttribute("aria-label", "Missing at 0:01.00");
    expect(cells[1]).toHaveTextContent("0:01.00");
    expect(cells[1].querySelector("img")).toBeNull();
    expect(cells[1].querySelector("div")?.className).not.toMatch(/rounded/);
    expect(cells[2]).toHaveTextContent("0:02.00");
    expect(cells[2].querySelector("img")).toHaveAttribute("src", "/api/media/observations/obs-2/crop");
  });

  it("closes the detail when the selected object card is clicked again", () => {
    const client = new QueryClient();
    render(
      <QueryClientProvider client={client}>
        <BundleContext.Provider value={bundle}>
          <ObjectsPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );

    const card = screen.getByRole("button", { name: /Rod/ });
    fireEvent.click(card);
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(card).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(card);
    expect(screen.queryByText("Dataset")).not.toBeInTheDocument();
    expect(card).toHaveAttribute("aria-pressed", "false");
  });
});
