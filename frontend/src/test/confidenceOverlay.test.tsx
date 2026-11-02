import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoStage } from "../components/VideoStage";
import type { Bundle } from "../lib/types";
import { useInspector } from "../stores/ui";

const bundle = {
  run: { id: "run-1" },
  video: null,
  frames: [],
  objects: [{ local_id: "Object_1", label: "rod", color: "#7EB6FF" }],
  observations: [
    {
      id: "obs-1",
      object_local_id: "Object_1",
      timestamp: 1,
      bbox: { x: 0.2, y: 0.2, w: 0.1, h: 0.1 },
      visibility: "visible",
      continuity: "continuous",
      confidence: 0.42,
    },
  ],
  tracks: { points: [], truncated: false, total: 0 },
} as unknown as Bundle;

function stage() {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <VideoStage bundle={bundle} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  HTMLCanvasElement.prototype.getContext = (() => null) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  HTMLMediaElement.prototype.pause = () => {};
  Object.defineProperty(document, "fonts", {
    configurable: true,
    value: { load: () => Promise.resolve([]) },
  });
  useInspector.setState({
    time: 1,
    playing: false,
    overlay: true,
    layers: { tracks: false, trajectory: false, uncertain: false, confidence: true, conflicts: false, drops: false },
  });
});

afterEach(() => {
  useInspector.setState({ time: 0, playing: false, overlay: true });
});

describe("confidence on the video overlay", () => {
  it("draws the model score as a percent and hides it without changing the box", () => {
    const view = stage();
    const score = screen.getByText("42%");
    const rect = view.container.querySelector("rect");
    expect(score).toBeInTheDocument();
    expect(rect).not.toBeNull();
    expect(Number(score.getAttribute("x"))).toBe(Number(rect?.getAttribute("x")) + 4);

    fireEvent.click(screen.getByRole("button", { name: "Hide confidence" }));

    expect(screen.queryByText("42%")).not.toBeInTheDocument();
    expect(view.container.querySelector("rect")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Show confidence" })).toBeInTheDocument();
  });
});

describe("objects in the current frame", () => {
  it("shows a labeled crop for each object at the playhead", () => {
    const scene = {
      ...bundle,
      objects: [
        { local_id: "Object_1", label: "rod", color: "#7EB6FF" },
        { local_id: "Object_2", label: "nut", color: "#e6b450" },
      ],
      observations: [
        {
          id: "obs-1",
          object_local_id: "Object_1",
          timestamp: 1,
          bbox: { x: 0.2, y: 0.2, w: 0.1, h: 0.1 },
          frame_id: "f1",
          visibility: "visible",
          confidence: 0.42,
        },
        {
          id: "obs-2",
          object_local_id: "Object_2",
          timestamp: 8,
          bbox: { x: 0.4, y: 0.4, w: 0.1, h: 0.1 },
          frame_id: "f2",
          visibility: "visible",
        },
      ],
    } as unknown as Bundle;
    const client = new QueryClient();
    const view = render(
      <QueryClientProvider client={client}>
        <VideoStage bundle={scene} />
      </QueryClientProvider>,
    );
    const frame = within(view.container);
    expect(frame.getByText("Objects in frame")).toBeInTheDocument();
    const rod = frame.getByRole("img", { name: "Rod" });
    expect(rod).toHaveAttribute("src", "/api/media/observations/obs-1/crop");
    expect(rod.closest(".frame-objects-row")).not.toBeNull();
    expect(frame.queryByRole("button", { name: "View Nut" })).not.toBeInTheDocument();

    fireEvent.click(frame.getByRole("button", { name: "View Rod" }));
    const dialog = screen.getByRole("dialog", { name: "Rod" });
    expect(dialog.querySelector("img")).toHaveAttribute("src", "/api/media/observations/obs-1/crop");
    const score = within(dialog).getByText("42%");
    expect(score).toHaveClass("shot-score");
    expect(score.parentElement).toHaveClass("shot-frame");
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    fireEvent.click(frame.getByRole("button", { name: "Hide boxes" }));
    expect(frame.getByRole("img", { name: "Rod" })).toBeInTheDocument();
    expect(view.container.querySelector("rect")).toBeNull();

    fireEvent.click(frame.getByRole("button", { name: "Hide objects in frame" }));
    expect(frame.queryByRole("img", { name: "Rod" })).not.toBeInTheDocument();
    expect(frame.getByText("Objects in frame")).toBeInTheDocument();

    fireEvent.click(frame.getByRole("button", { name: "Show objects in frame" }));
    expect(frame.getByRole("img", { name: "Rod" })).toBeInTheDocument();
  });
});

describe("frame stepping on the picture", () => {
  it("uses the preview chevrons and leaves them off the button row", () => {
    const view = stage();
    const frame = within(view.container);
    const previous = frame.getByRole("button", { name: "Previous frame" });
    const next = frame.getByRole("button", { name: "Next frame" });
    expect(previous).toHaveClass("asset-preview-nav", "is-prev");
    expect(next).toHaveClass("asset-preview-nav", "is-next");
    expect(previous.closest(".stage-frame")).not.toBeNull();
    expect(view.container.querySelector(".flex-wrap")).not.toContainElement(previous);
    expect(frame.queryByRole("button", { name: "Prev frame" })).not.toBeInTheDocument();

    fireEvent.click(next);
    expect(useInspector.getState().time).toBeCloseTo(1.1);
  });
});

describe("dropped box layer", () => {
  const withDrops = {
    ...bundle,
    box_decisions: {
      total: 1,
      truncated: false,
      counts: [],
      rows: [{
        step: "locate",
        reason_code: "low_confidence",
        outcome: "dropped",
        object_local_id: "Object_1",
        timestamp: 1,
        bbox: { x: 0.5, y: 0.5, w: 0.1, h: 0.1 },
        confidence: 0.2,
        other_object_id: null,
        detail: "",
      }],
    },
  } as unknown as Bundle;

  it("has no toggle when the run recorded nothing", () => {
    useInspector.setState({ time: 1 });
    stage();
    expect(screen.queryByRole("button", { name: "Show dropped" })).toBeNull();
  });

  it("draws the reason only after the toggle is on", () => {
    useInspector.setState({ time: 1 });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <VideoStage bundle={withDrops} />
      </QueryClientProvider>,
    );
    expect(screen.queryByTestId("dropped-box")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show dropped" }));
    expect(within(screen.getByTestId("dropped-box")).getByText(/Object_1 · Low confidence/)).toBeInTheDocument();
  });
});
