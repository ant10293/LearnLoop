import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BundleContext } from "../lib/bundleContext";
import { boxAt, canApproveLive, dragBox, layerBoxes, manualTargets, nextStep, placeSeed, spansOf, stepTimes, trackRange, usage, type RecoveryPlan, type RecoveryProposal } from "../lib/recovery";
import { api } from "../lib/api";
import { RecoveryPage } from "../pages/Recovery";
import { useInspector } from "../stores/ui";
import type { Bundle } from "../lib/types";

const box = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 };

const row = (extra: Record<string, unknown>) => ({ object_local_id: "Object_1", timestamp: 1, bbox: box, ...extra });

const bundle = {
  run: { id: "run-1" },
  video: { playback_url: "", duration_sec: 10 },
  frames: [{ id: "f1", timestamp: 0.5, reason: "uniform", scene_change: false }, { id: "f2", timestamp: 1, reason: "uniform", scene_change: false }],
  objects: [],
  actions: [],
  transitions: [],
  ambiguities: [],
  steps: [],
  observations: [row({ geometry_origin: "vlm_estimated" }), row({ source_model: "tracker", observation_role: "canonical" })],
  raw_observations: { total: 1, truncated: false, rows: [row({ source_model: "sam-3.1", association_status: "conflict" })] },
  tracks: { total: 1, truncated: false, points: [{ track_ref: "track_9", timestamp: 1, bbox: box, detector_id: "motion", status: "active" }] },
} as unknown as Bundle;

const proposal = {
  id: "p1",
  current: true,
  object_local_id: "Object_1",
  rows: [
    { timestamp: 1, bbox: box, outcome: "provisional", association: "associated", direction: "forward", seed_trust: "human_verified", partial: false, context: false, confidence: 0.9, checks: [], frame_index: 0 },
  ],
} as unknown as RecoveryProposal;

const none = { vlm: false, tracker: false, sam: false, recovered: false, reacquired: false, raw: false, canonical: false };

describe("recovery layers", () => {
  it("draws only the selected layer", () => {
    expect(layerBoxes(bundle, [], 1, none)).toEqual([]);
    const vlm = layerBoxes(bundle, [], 1, { ...none, vlm: true });
    expect(vlm.map((item) => item.layer)).toEqual(["vlm"]);
  });

  it("compares original and recovered boxes on the same frame", () => {
    const shown = layerBoxes(bundle, [proposal], 1, { ...none, vlm: true, recovered: true });
    expect(shown.map((item) => item.layer).sort()).toEqual(["recovered", "vlm"]);
  });

  it("never draws an estimate, conflict, or unconfirmed box as solid", () => {
    expect(layerBoxes(bundle, [], 1, { ...none, vlm: true })[0].dashed).toBe(true);
    expect(layerBoxes(bundle, [], 1, { ...none, sam: true })[0].dashed).toBe(true);
    expect(layerBoxes(bundle, [proposal], 1, { ...none, recovered: true })[0].dashed).toBe(true);
  });

  it("shows a canonical layer and keeps the row on its source layer", () => {
    const shown = layerBoxes(bundle, [], 1, { ...none, canonical: true, tracker: true });
    expect(shown.map((item) => item.layer).sort()).toEqual(["canonical", "tracker", "tracker"]);
  });

  it("keeps a box up between stored times, as Inspect does", () => {
    expect(layerBoxes(bundle, [], 1.4, { ...none, vlm: true })).toHaveLength(1);
  });

  it("steps through frames, stored boxes, and recovered boxes inside the interval", () => {
    const later = { ...proposal, rows: [{ ...proposal.rows[0], timestamp: 2 }, { ...proposal.rows[0], timestamp: 3, context: true }] } as RecoveryProposal;
    const times = stepTimes(bundle, [later]);
    expect(times).toEqual([0.5, 1, 2]);
    expect(nextStep(times, 1, 1)).toBe(2);
    expect(nextStep(times, 1, -1)).toBe(0.5);
    expect(nextStep(times, 2, 1)).toBeNull();
  });

  it("does not hold a box past its time", () => {
    expect(layerBoxes(bundle, [proposal], 5, { vlm: true, tracker: true, sam: true, recovered: true, reacquired: true, raw: true, canonical: true })).toEqual([]);
  });

  it("draws every raw SAM box, keeps two tracks apart, and marks unassigned ones", () => {
    const raw = (track: string, object: string, status: string) => ({
      request_key: "k", raw_track_id: track, object_local_id: object, frame_index: 0, timestamp: 1, bbox: box, pixels: [], confidence: 0.2, has_mask: true, status,
    });
    const rows = [raw("1", "Object_1", "associated"), raw("7", "", "unassigned"), { ...raw("8", "", "frame_out_of_range"), timestamp: null, bbox: null }];
    expect(layerBoxes(bundle, [], 1, none, rows)).toEqual([]);
    const shown = layerBoxes(bundle, [], 1, { ...none, raw: true }, rows);
    expect(shown.map((item) => [item.objectId, item.dashed])).toEqual([["Object_1", false], ["SAM track 7", true]]);
    expect(stepTimes({ frames: [], observations: [] }, [], rows)).toEqual([1]);
  });

  it("exposes provenance on a recovered box and labels the score uncalibrated", () => {
    const details = layerBoxes(bundle, [proposal], 1, { ...none, recovered: true })[0].details.join(" ");
    expect(details).toContain("outcome provisional");
    expect(details).toContain("uncalibrated");
  });
});

describe("seed drawing and approval", () => {
  it("normalizes a drag in any direction and clamps it to the frame", () => {
    expect(dragBox(0.5, 0.5, 0.2, 0.1)).toEqual({ x: 0.2, y: 0.1, w: 0.3, h: 0.4 });
    expect(dragBox(0.9, 0.9, 1.4, 1.2)?.w).toBeCloseTo(0.1);
    expect(dragBox(0.5, 0.5, 0.5, 0.5)).toBeNull();
  });

  const plan = (remote: number, violations: string[] = []) => ({
    plan_hash: "h",
    requests: [],
    blocked: [],
    notes: [],
    violations,
    totals: { requests: remote, remote_requests: remote, cached_requests: 0, frames: 10, billed_frames: 50, estimated_cost_usd: null, cost_note: "" },
  }) as RecoveryPlan;

  it("offers a live run only for a plan with requests and no broken limit", () => {
    expect(canApproveLive(null)).toBe(false);
    expect(canApproveLive(plan(0))).toBe(false);
    expect(canApproveLive(plan(1, ["over budget"]))).toBe(false);
    expect(canApproveLive(plan(1))).toBe(true);
  });

  it("says when the cost is unknown", () => {
    expect(usage(plan(1).totals)).toContain("cost unknown");
  });
});

describe("seed drafts", () => {
  const draft = (objectId: string, timestamp: number) => ({
    objectId,
    timestamp,
    bbox: box,
    points: [],
    ...trackRange(timestamp, 2, "forward"),
    direction: "forward" as const,
    origin: "drawn",
  });

  it("keeps many seeds in time order and replaces one on the same frame", () => {
    let drafts = placeSeed([], draft("Object_1", 5));
    drafts = placeSeed(drafts, draft("Object_2", 5));
    drafts = placeSeed(drafts, draft("Object_1", 1));
    expect(drafts.map((item) => [item.objectId, item.timestamp])).toEqual([["Object_1", 1], ["Object_1", 5], ["Object_2", 5]]);
    expect(placeSeed(drafts, { ...draft("Object_1", 5), origin: "redrawn" })).toHaveLength(3);
  });

  it("sets the tracked stretch from the direction", () => {
    expect(trackRange(5, 2, "forward")).toEqual({ start: 5, end: 7 });
    expect(trackRange(5, 2, "backward")).toEqual({ start: 3, end: 5 });
    expect(trackRange(1, 2, "both")).toEqual({ start: 0, end: 3 });
  });

  it("sends every seed as its own target", () => {
    const targets = manualTargets([{ ...draft("Object_1", 5), id: "a" }, { ...draft("Object_2", 6), id: "b" }]);
    expect(targets.map((item) => item.object_local_id)).toEqual(["Object_1", "Object_2"]);
    expect(targets[0].seed).toMatchObject({ timestamp: 5, bbox: box });
  });

  it("picks the smallest known object's box under a click", () => {
    const shape = (objectId: string, w: number) => ({ key: objectId, layer: "vlm" as const, objectId, bbox: { x: 0, y: 0, w, h: w }, timestamp: 1, dashed: true, details: [] });
    const boxes = [shape("Object_1", 0.5), shape("Object_2", 0.2), shape("track_9", 0.1)];
    expect(boxAt(boxes, new Set(["Object_1", "Object_2"]), 0.1, 0.1)?.objectId).toBe("Object_2");
    expect(boxAt(boxes, new Set(["Object_1", "Object_2"]), 0.9, 0.9)).toBeNull();
  });

  it("joins times into lane spans and leaves holes visible", () => {
    expect(spansOf([0, 1, 2, 8, 9], "stored")).toEqual([{ start: 0, end: 2, kind: "stored" }, { start: 8, end: 9, kind: "stored" }]);
  });
});

describe("recovery page", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      disconnect() {}
    });
    // jsdom has no PointerEvent, and without one the event carries no coordinates.
    vi.stubGlobal("PointerEvent", class extends MouseEvent {});
    useInspector.getState().setTime(1);
    vi.spyOn(api, "recoveryCandidates").mockResolvedValue({ run_id: "run-1", revision_id: "r", targets: [target], seeds: [], seed_counts: {}, objects: [], price_per_frame_usd: null } as never);
    vi.spyOn(api, "recoveryExperiments").mockResolvedValue([]);
    vi.spyOn(api, "recoveryDiagnostics").mockResolvedValue({ experiment_id: "", generation: 0, note: "", requests: [] });
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue({ run_id: "run-1", objects: [], note: "" });
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  const pageBundle = {
    ...bundle,
    objects: [
      { local_id: "Object_1", label: "washer", color: "#ff0000" },
      { local_id: "Object_2", label: "nut", color: "#00ff00" },
    ],
  } as unknown as Bundle;

  const target = {
    target_id: "t1",
    object_local_id: "Object_1",
    instance_id: null,
    label: "washer",
    time_start: 2,
    time_end: 4,
    source: "automatic",
    state: "unresolved",
    reasons: ["tracking_gap"],
    details: [],
    seeds: [],
    difficulty: "easy",
    recoverable: false,
    blocker: "",
  };

  const planned = {
    id: "experiment-1",
    run_id: "run-1",
    status: "planned",
    mode: "dry_run",
    generation: 0,
    targets: [target],
    limits: {},
    error: "",
    remote_submitted: false,
    recorded_cost_usd: null,
    applied_revision_id: null,
    cancel_requested: false,
    requests: [],
    proposals: [],
    plan: {
      plan_hash: "hash-1",
      requests: [],
      blocked: [],
      notes: [],
      violations: [],
      totals: { requests: 1, remote_requests: 1, cached_requests: 0, frames: 30, billed_frames: 50, estimated_cost_usd: null, cost_note: "" },
    },
  };

  it("shows a skeleton until the analysis arrives", () => {
    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <BundleContext.Provider value={null}>
          <RecoveryPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );
    expect(screen.getByText("Loading recovery")).toHaveClass("sr-only");
    expect(container.querySelector("[aria-busy='true'] .stage-frame .bone-fill")).toBeInTheDocument();
    expect(screen.queryByTestId("recovery-page")).not.toBeInTheDocument();
  });

  function show() {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const view = render(
      <QueryClientProvider client={client}>
        <BundleContext.Provider value={pageBundle}>
          <RecoveryPage />
        </BundleContext.Provider>
      </QueryClientProvider>,
    );
    const overlay = screen.getByTestId("recovery-overlay");
    overlay.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}) });
    return { ...view, overlay };
  }

  function press(overlay: HTMLElement, x1: number, y1: number, x2 = x1, y2 = y1) {
    fireEvent.pointerDown(overlay, { clientX: x1, clientY: y1, pointerId: 1 });
    fireEvent.pointerMove(overlay, { clientX: x2, clientY: y2, pointerId: 1 });
    fireEvent.pointerUp(overlay, { clientX: x2, clientY: y2, pointerId: 1 });
  }

  it("shows a still frame and steps with the arrows", () => {
    useInspector.getState().setTime(0);
    const { container } = show();
    expect(container.querySelector("video")?.hasAttribute("controls")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Next frame" }));
    expect(useInspector.getState().time).toBe(0.5);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(useInspector.getState().time).toBe(1);
  });

  it("zooms the frame and boxes together, and Shift-drag pans instead of drawing", () => {
    const { container, overlay } = show();
    const frame = overlay.parentElement as HTMLElement;
    frame.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100, x: 0, y: 0, toJSON: () => ({}) });
    const video = container.querySelector("video") as HTMLVideoElement;
    expect(screen.getByRole("button", { name: "Zoom out" })).toHaveProperty("disabled", true);

    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(video.style.transform).toContain("scale(1.5)");
    expect(overlay.style.width).toBe("1.5px");
    expect(screen.getByRole("button", { name: "Zoom out" })).toHaveProperty("disabled", false);

    fireEvent.pointerDown(overlay, { clientX: 100, clientY: 50, pointerId: 1, shiftKey: true });
    fireEvent.pointerMove(frame, { clientX: 120, clientY: 50, pointerId: 1 });
    fireEvent.pointerUp(frame, { clientX: 120, clientY: 50, pointerId: 1 });
    expect(video.style.transform).toContain("translate(20px, 0px)");
    expect(screen.queryByTestId("drawn-seed")).toBeNull();
  });

  it("makes a seed from one click on a box", () => {
    const { overlay } = show();
    press(overlay, 15, 15);
    expect(within(screen.getByTestId("seed-list")).getByText("Object 1")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Seeds (1)" })).toBeInTheDocument();
  });

  it("asks which object before drawing, then keeps seeds from many objects and frames", () => {
    const { overlay } = show();
    press(overlay, 50, 50, 70, 70);
    expect(screen.getByRole("status")).toHaveTextContent("Choose which object");
    fireEvent.click(within(screen.getByRole("group", { name: "Object to seed" })).getByRole("button", { name: "Object 1" }));
    press(overlay, 50, 50, 70, 70);
    fireEvent.click(within(screen.getByRole("group", { name: "Object to seed" })).getByRole("button", { name: "Object 2" }));
    press(overlay, 60, 60, 80, 80);
    act(() => useInspector.getState().setTime(3));
    press(overlay, 60, 60, 80, 80);
    expect(screen.getByRole("heading", { name: "Seeds (3)" })).toBeInTheDocument();
  });

  it("sends all seeds together, and a live run needs the checked plan", async () => {
    const create = vi.spyOn(api, "createRecovery").mockResolvedValue(planned as never);
    const dry = vi.spyOn(api, "recoveryDryRun").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    const live = vi.spyOn(api, "recoveryLive").mockResolvedValue(planned as never);
    const { overlay } = show();
    press(overlay, 15, 15);
    fireEvent.click(within(screen.getByRole("group", { name: "Object to seed" })).getByRole("button", { name: "Object 2" }));
    press(overlay, 60, 60, 80, 80);
    expect(screen.queryByRole("button", { name: /^Run SAM/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Check 2 seeds/ }));
    await waitFor(() => expect(dry).toHaveBeenCalledWith("experiment-1"));
    const payload = create.mock.calls[0][1] as { manual_targets: unknown[] };
    expect(payload.manual_targets).toHaveLength(2);
    expect(live).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: /^Run SAM: 1 request, about 50 frames/ }));
    await waitFor(() => expect(live).toHaveBeenCalledWith("experiment-1", { confirm: true, plan_hash: "hash-1" }));
  });

  it("changing a seed after checking requires checking again", async () => {
    vi.spyOn(api, "createRecovery").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryDryRun").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    const { overlay } = show();
    press(overlay, 15, 15);
    fireEvent.click(screen.getByRole("button", { name: /Check 1 seed/ }));
    await screen.findByRole("button", { name: /^Run SAM/ });
    fireEvent.click(screen.getByRole("button", { name: /Remove seed/ }));
    expect(screen.queryByRole("button", { name: /^Run SAM/ })).toBeNull();
  });

  it("says whether a failure submitted anything", async () => {
    const failed = { ...planned, status: "failed", error: "Upload failed.", remote_submitted: false };
    vi.spyOn(api, "createRecovery").mockResolvedValue(failed as never);
    vi.spyOn(api, "recoveryDryRun").mockResolvedValue(failed as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(failed as never);
    const { overlay } = show();
    press(overlay, 15, 15);
    fireEvent.click(screen.getByRole("button", { name: /Check 1 seed/ }));
    expect(await screen.findByText(/No remote call was submitted/)).toBeInTheDocument();
  });

  it("accepts every pending result at once and marks a synthetic response", async () => {
    const metrics = {
      counts: { supported: 1, provisional: 2, conflicting: 0, rejected: 0, unobserved_frames: 2 },
      previously_missing_frames: 1,
      new_supported_frames: 1,
      original_rows_in_interval: 0,
      remaining_gaps: [{ start: 3, end: 4, frames: 2 }],
      identity: [],
      usage: { requests: 1, billed_frames: 50, latency_ms: 0, input_tokens: 0, output_tokens: 0 },
      synthetic_response: true,
      human_box_overlap: null,
      accuracy_note: "No human-labeled box covers this interval, so accuracy was not measured.",
    };
    const proposal = (id: string) => ({ id, generation: 1, current: true, target_id: id, object_local_id: "Object_1", time_start: 2, time_end: 4, decision: "pending", rows: [], metrics, applied_revision_id: null });
    const done = { ...planned, status: "completed", proposals: [proposal("p1"), proposal("p2")] };
    vi.spyOn(api, "createRecovery").mockResolvedValue(done as never);
    vi.spyOn(api, "recoveryDryRun").mockResolvedValue(done as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(done as never);
    const decide = vi.spyOn(api, "decideRecovery").mockResolvedValue(done as never);
    const { overlay } = show();
    press(overlay, 15, 15);
    fireEvent.click(screen.getByRole("button", { name: /Check 1 seed/ }));
    expect(await screen.findByText(/synthetic offline response/)).toBeInTheDocument();
    expect(screen.getByText(/accuracy was not measured/)).toBeInTheDocument();
    expect(screen.getAllByText(/not evidence of absence/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Accept as a new revision" }));
    await waitFor(() => expect(decide).toHaveBeenCalledWith("experiment-1", { decision: "accepted", proposal_ids: ["p1", "p2"], include_provisional: false }));
  });

  it("dismisses a problem from the list and the lane, and restores it", async () => {
    const candidates = (flag: boolean) => ({ run_id: "run-1", revision_id: "r", targets: [{ ...target, dismissed: flag }], seeds: [], seed_counts: {}, objects: [], price_per_frame_usd: null });
    const dismiss = vi.spyOn(api, "dismissRecoveryProblem").mockResolvedValue(candidates(true) as never);
    const restore = vi.spyOn(api, "restoreRecoveryProblem").mockResolvedValue(candidates(false) as never);
    show();
    const lane = screen.getByTestId("lane-Object_1");
    await waitFor(() => expect(lane.querySelector('[data-kind="problem"]')).not.toBeNull());
    fireEvent.click(screen.getByRole("button", { name: "Dismiss problem for Object 1 at 2.00s" }));
    await waitFor(() => expect(dismiss).toHaveBeenCalledWith("run-1", "t1"));
    await waitFor(() => expect(lane.querySelector('[data-kind="problem"]')).toBeNull());
    expect(screen.getByText("Suspected problems (0)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show dismissed (1)" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore problem for Object 1 at 2.00s" }));
    await waitFor(() => expect(restore).toHaveBeenCalledWith("run-1", "t1"));
    await waitFor(() => expect(lane.querySelector('[data-kind="problem"]')).not.toBeNull());
    expect(screen.getByText("Suspected problems (1)")).toBeInTheDocument();
  });

  it("opens an earlier run from a readable row", async () => {
    const earlier = { ...planned, created_at: "2026-11-01T19:42:00", pending_proposals: 2, status: "completed", targets: [{ ...target, object_local_id: "Object_2", time_start: 11, time_end: 13 }, { ...target, time_start: 12, time_end: 14.5 }] };
    vi.spyOn(api, "recoveryExperiments").mockResolvedValue([earlier] as never);
    const read = vi.spyOn(api, "recoveryExperiment").mockResolvedValue(earlier as never);
    show();
    const row = await screen.findByRole("button", { name: /Object 1, Object 2 · 11.0–14.5s/ });
    expect(row).toHaveTextContent("2 pending review");
    fireEvent.click(row);
    await waitFor(() => expect(read).toHaveBeenCalledWith("experiment-1"));
    await waitFor(() => expect(row).toHaveAttribute("aria-current", "true"));
  });

  it("shows what a request returned and where its boxes went", async () => {
    const done = { ...planned, created_at: "2026-11-01T19:42:00", status: "completed", mode: "live", generation: 1 };
    vi.spyOn(api, "recoveryExperiments").mockResolvedValue([done] as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(done as never);
    vi.spyOn(api, "recoveryDiagnostics").mockResolvedValue({
      experiment_id: "experiment-1",
      generation: 1,
      note: "A returned box shows SAM produced output.",
      requests: [{
        key: "k", verdict: "no_assignment", message: "SAM returned usable boxes and none belong to a prompted object.", boxes_returned: true, synthetic: false,
        request: { status: "completed", submission: "completed", source: "live", provider_request_id: "resp_1", latency_ms: 2000, error: "" },
        prompt: "<0f>1<|box|>", direction: "forward", planned_frames: 10, reachable_frames: 10, frames_with_boxes: 4,
        parsed: { text_chars: 9, frame_lines: 4, empty_frame_lines: 0, box_tokens: 4, dropped_tokens: 0, unparsed_lines: 0, unparsed_samples: [] },
        funnel: { box_tokens: 4, parsed: 4, usable: 4, unassigned: 4, assigned: 0, supported: 0, provisional: 0, conflicting: 0, rejected: 0 },
        tracks: [{ raw_track_id: "7", prompted: false, object_local_id: "", association: "unassigned", association_note: "", boxes: 4, first_time: 1, last_time: 2, first_confidence: 0.5, mean_confidence: 0.4, last_confidence: 0.3, broke: null }],
        missing_frames: [], failed_checks: {}, findings: [{ code: "unprompted_track", severity: "warning", message: "SAM track 7 was not prompted." }], boxes: [],
      }],
    } as never);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /Object 1 · 2.0–4.0s/ }));
    const panel = await screen.findByTestId("sam-diagnostics");
    expect(within(panel).getByText(/none belong to a prompted object/)).toBeInTheDocument();
    expect(within(panel).getByText(/SAM track 7 was not prompted/)).toBeInTheDocument();
    expect(within(panel).getByText(/no object/)).toHaveTextContent("4 boxes");
    expect(screen.getByRole("button", { name: "All SAM output" })).toHaveAttribute("aria-pressed", "false");
  });

  const failure = {
    failure_id: "f1", experiment_id: "e1", time: 3, last_good_time: 2.9, last_supported_time: 2.5, direction: "forward",
    reason: "large_spatial_jump", detail: "A large jump broke the track.", state: "occluded", state_basis: "The box shrank.",
  };
  const segment = {
    segment_id: "reca:1:x/f0", physical_instance_id: "track_1", object_local_id: "Object_1", experiment_id: "e1", session_id: "reca", raw_track_id: "1",
    direction: "forward", seed: { seed_id: "x", trust: "human_verified", source: "human", timestamp: 1 }, init_event: "trusted_seed", start: 1, end: 2.9,
    last_supported_time: 2.5, status: "lost", identity: { status: "linked", basis: [], detail: "A person confirmed the seed." },
    counts: { supported: 5, provisional: 1, conflicting: 0, rejected: 0 }, frames: 6,
  };
  const planItem = (extra: Record<string, unknown> = {}) => ({
    failure, status: "open", uncertain_interval: { start: 3, end: 4 }, last_supported_time: 2.5, reappearance_frames: [3.5], recommended: "candidate_detection",
    strategies: [
      { strategy: "candidate_detection", available: true, requires_remote: true, note: "Stored boxes may show the object.", evidence: [
        { source: "vlm", trust: "candidate", seed_id: "s9", timestamp: 3.5, bbox: box, explanation: "", needs_confirmation: true, may_be: ["Object_2"] },
      ] },
      { strategy: "concept_redetection", available: true, requires_remote: true, note: "Finds locations.", evidence: [], concept: "washer" },
    ],
    blocked_reason: "No trustworthy evidence.", needs_stronger_evidence: false, attempts: [],
    explanation: "SAM kept returning boxes (9 after the failure), and tracking became unreliable at 3.00s.",
    ...extra,
  });
  const historyWith = (plan: unknown) => ({
    run_id: "run-1", note: "Plans only.",
    objects: [{
      object_local_id: "Object_1", physical_instance_id: "track_1", label: "washer", segments: [segment], failures: [failure],
      unresolved: [{ start: 3, end: 4, frames: 9, no_detection_frames: 2, unusable_frames: 7 }], plans: [plan],
      stats: { raw_frames_returned: 15, track_breaks: 1, no_detection_frames: 2, unusable_frames: 7 }, evidence_limits: [],
    }],
  });

  it("shows a segment, a failure that seeks, and what cannot be claimed", async () => {
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem()) as never);
    show();
    const panel = await screen.findByTestId("track-history");
    expect(within(panel).getByText(/tracking became unreliable at 3.00s/)).toBeInTheDocument();
    expect(within(panel).getByText(/Possibly hidden/)).toBeInTheDocument();
    expect(panel.querySelector('[data-kind="unresolved"]')?.getAttribute("title")).toContain("2 with no SAM box");
    fireEvent.click(within(panel).getByRole("button", { name: "Go to the failure at 3.00 seconds" }));
    expect(useInspector.getState().time).toBe(3);
    fireEvent.click(within(panel).getByRole("button", { name: "Last supported 2.50s" }));
    expect(useInspector.getState().time).toBe(2.5);
  });

  it("makes a person confirm a candidate, and the seed records which failure it answers", async () => {
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem()) as never);
    const created = vi.spyOn(api, "createRecovery").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryDryRun").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    show();
    fireEvent.click(await screen.findByRole("button", { name: "This is the object" }));
    expect(screen.getByTestId("seed-list")).toHaveTextContent("reacquires the track lost at 3.00s");
    fireEvent.click(screen.getByRole("button", { name: /Check 1 seed/ }));
    await waitFor(() => expect(created).toHaveBeenCalled());
    const sent = (created.mock.calls[0][1] as { manual_targets: Record<string, unknown>[] }).manual_targets[0];
    expect(sent).toMatchObject({ seed_id: "s9", confirm_seed: true, direction: "both", time_start: 3, reacquires: { failure_id: "f1", strategy: "candidate_detection" } });
    expect(sent.seed).toBeUndefined();
  });

  it("links a drawn seed to the failure and only plans a concept search", async () => {
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem()) as never);
    const plannedStep = vi.spyOn(api, "planReacquisition").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    const live = vi.spyOn(api, "recoveryLive");
    const { overlay } = show();
    fireEvent.click(await screen.findByRole("button", { name: "Draw a new seed" }));
    expect(useInspector.getState().time).toBe(3.5);
    press(overlay, 40, 40, 60, 60);
    expect(screen.getByTestId("seed-list")).toHaveTextContent("reacquires the track lost at 3.00s");
    fireEvent.click(screen.getByRole("button", { name: /Plan a concept search/ }));
    await waitFor(() => expect(plannedStep).toHaveBeenCalled());
    expect(plannedStep.mock.calls[0][1]).toMatchObject({ failure_id: "f1", action: "concept", concept: "washer" });
    expect(live).not.toHaveBeenCalled();
  });

  const search = (candidates: unknown[]) => ({
    window: { start: 3, end: 8, limited: false, note: "" }, frames_examined: 12,
    evidence_inspected: { stored_observations: 1, stored_sam_requests: [], stored_sam_boxes: 4, pipeline_sam_points: 0 },
    candidates, result: "ambiguous_candidate",
  });
  const diagnosis = {
    categories: [{ code: "B_new_track_id", detail: "Later detections exist under another id: SAM track 7." }],
    last_raw_time: 2.9, last_supported_time: 2.5, first_no_detection_time: 3, raw_boxes_after_failure: 0, later_track_ids: ["7"], rejected_by_validation: 0,
    detected_not_identified: 1, analyzed_through: 4, video_end: 4, clip_ended_early: false, request_planned: false, request_executed: false,
  };
  const workflow = [
    { stage: "track_lost", status: "done", detail: "" },
    { stage: "identity", status: "waiting_for_person", detail: "" },
    { stage: "reinitialization", status: "blocked", detail: "" },
  ];
  const candidate = (extra: Record<string, unknown>) => ({
    candidate_id: "c1", source: "cached_concept", timestamp: 3.6, bbox: box, raw_track_id: "7", track_frames: 9, outcome: "ambiguous_candidate",
    reason: "More than one part could be this detection.", needs_confirmation: true, checks: [{ check: "look_alikes", passed: false, detail: "Object_2 shares this label." }], ...extra,
  });

  it("shows why a track stopped and seeks from its breakdown", async () => {
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem({ search: search([]), diagnosis, workflow })) as never);
    show();
    const panel = await screen.findByTestId("recovery-breakdown");
    expect(within(panel).getByTestId("failure-category")).toHaveTextContent("SAM track 7");
    expect(within(panel).getByTestId("reacquisition-workflow")).toHaveTextContent("Waiting for person");
    fireEvent.click(within(panel).getByRole("button", { name: "2.50s" }));
    expect(useInspector.getState().time).toBe(2.5);
    expect(within(panel).getByText(/Nothing stored shows this object/)).toBeInTheDocument();
  });

  it("marks candidates on the lane and sends a confirmation only when a person chooses one", async () => {
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem({ search: search([candidate({})]), diagnosis, workflow })) as never);
    const plannedStep = vi.spyOn(api, "planReacquisition").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    const live = vi.spyOn(api, "recoveryLive");
    show();
    const row = await screen.findByTestId("reappearance-candidate");
    expect(within(row).getByText("Ambiguous candidate")).toBeInTheDocument();
    expect(plannedStep).not.toHaveBeenCalled();
    expect(screen.getByTestId("history-lane-Object_1").querySelector('[data-kind="ambiguous_candidate"]')).not.toBeNull();
    fireEvent.click(within(row).getByRole("button", { name: /This is the object: plan new track/ }));
    await waitFor(() => expect(plannedStep).toHaveBeenCalled());
    expect(plannedStep.mock.calls[0][1]).toMatchObject({ failure_id: "f1", candidate_id: "c1", confirm: true });
    expect(live).not.toHaveBeenCalled();
  });

  it("offers a verified candidate without a confirmation, and no way to use a rejected one", async () => {
    const list = [candidate({ outcome: "verified_reappearance", needs_confirmation: false }), candidate({ candidate_id: "c2", outcome: "rejected_candidate", needs_confirmation: false })];
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(planItem({ search: search(list), diagnosis, workflow })) as never);
    const plannedStep = vi.spyOn(api, "planReacquisition").mockResolvedValue(planned as never);
    vi.spyOn(api, "recoveryExperiment").mockResolvedValue(planned as never);
    show();
    const rows = await screen.findAllByTestId("reappearance-candidate");
    expect(within(rows[1]).queryByRole("button", { name: /plan new track/i })).toBeNull();
    fireEvent.click(within(rows[0]).getByRole("button", { name: "Plan new track (free)" }));
    await waitFor(() => expect(plannedStep.mock.calls[0][1]).toMatchObject({ candidate_id: "c1", confirm: false }));
  });

  it("says when SAM was never asked about the rest of the video", async () => {
    const history = historyWith(planItem({ search: search([]), diagnosis, workflow }));
    (history.objects[0] as Record<string, unknown>).coverage = { analyzed_through: 19, video_end: 21.9 };
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(history as never);
    show();
    expect(await screen.findByTestId("coverage-note")).toHaveTextContent("through 19.00s");
  });

  it("offers no reacquisition controls once a failure is resolved, and keeps the gap", async () => {
    const plan = planItem({ status: "resolved", explanation: "A new seed at 4.00s started a separate segment with 34 supported boxes." });
    vi.spyOn(api, "recoveryReacquisition").mockResolvedValue(historyWith(plan) as never);
    show();
    const panel = await screen.findByTestId("track-history");
    expect(within(panel).getByText(/34 supported boxes/)).toBeInTheDocument();
    expect(within(panel).queryByRole("button", { name: "Draw a new seed" })).toBeNull();
    expect(panel.querySelector('[data-kind="unresolved"]')).not.toBeNull();
  });

  it("draws reacquired boxes on their own layer", () => {
    const again = { ...proposal, rows: proposal.rows.map((row) => ({ ...row, reacquired: true })) } as RecoveryProposal;
    expect(layerBoxes(bundle, [again], 1, { ...none, recovered: true })).toEqual([]);
    expect(layerBoxes(bundle, [again], 1, { ...none, reacquired: true })[0].layer).toBe("reacquired");
  });

  it("keeps limits in the seeds panel, apart from earlier runs", () => {
    show();
    expect(within(screen.getByTestId("seed-list")).getByLabelText("Provider requests")).toBeInTheDocument();
    expect(screen.queryByTestId("earlier-runs")).toBeNull();
  });

  it("draws problems and seeds on the object's lane", async () => {
    const { overlay } = show();
    press(overlay, 15, 15);
    const lane = screen.getByTestId("lane-Object_1");
    await waitFor(() => expect(lane.querySelector('[data-kind="problem"]')).not.toBeNull());
    expect(lane.querySelector('[data-kind="seed"]')).not.toBeNull();
    expect(screen.getByTestId("lane-Object_2").querySelector('[data-kind="problem"]')).toBeNull();
  });
});
