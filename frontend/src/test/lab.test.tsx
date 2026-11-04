import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { modalVerdicts, percent, runLabel, type LabRun } from "../lib/reliability";
import { LabPage } from "../pages/Lab";
import { LabBatchPage } from "../pages/LabBatch";

type Handler = (url: string, init?: RequestInit) => { status?: number; body: unknown } | undefined;

function mockApi(handler: Handler) {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const hit = handler(url, init);
    if (!hit) throw new Error(`unexpected request ${url}`);
    return new Response(JSON.stringify(hit.body), { status: hit.status ?? 200, headers: { "Content-Type": "application/json" } });
  });
  return calls;
}

function show(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/lab" element={<LabPage />} />
          <Route path="/lab/:batchId" element={<LabBatchPage />} />
          <Route path="/lab/:batchId/runs/:runId" element={<p>drill-down</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const rate = (numerator: number, denominator: number) => ({ value: denominator ? numerator / denominator : null, numerator, denominator });
const step = (step_id: string, verdict: string) => ({ step_id, name: step_id, required: true, verdict, raw_verdict: verdict, basis: "model_evidence", error_category: "", evidence_count: 1, evidence_timestamps: [1], missing_evidence: false, observed_objects: [], warnings: [], reason: "", observed: "", expected: "", reinspected: false });
const run = (index: number, status: string, verdict: string | null, steps: ReturnType<typeof step>[], extra = {}) =>
  ({ id: `run${index}`, index, status, analysis_run_id: `a${index}`, verification_id: `v${index}`, error: "", error_kind: "", cancel_requested: false, duration_seconds: 40 + index, started_at: null, finished_at: null, progress: null, result: { outcome: verdict ? "verdict" : undefined, verdict, steps, evidence_count: steps.length }, ...extra }) as unknown as LabRun;

describe("Reliability helpers", () => {
  it("shows N/A for an empty denominator and marks the majority per step", () => {
    expect(percent(rate(8, 10))).toBe("80%");
    expect(percent(rate(0, 0))).toBe("N/A");
    const runs = [run(1, "completed", "PASS", [step("s1", "PASS"), step("s2", "PASS")]), run(2, "completed", "PASS", [step("s1", "PASS"), step("s2", "CANNOT_VERIFY")]), run(3, "completed", "PASS", [step("s1", "PASS"), step("s2", "PASS")])];
    expect(modalVerdicts(runs, ["s1", "s2"])).toEqual({ s1: "PASS", s2: "PASS" });
    expect(modalVerdicts(runs.slice(0, 2), ["s2"])).toEqual({ s2: null });
  });

  it("never shows a failed run as a verdict", () => {
    expect(runLabel(run(1, "failed", null, [], { error_kind: "rate_limited" }))).toBe("ERROR");
    expect(runLabel(run(1, "failed", null, [], { error_kind: "invalid_result" }))).toBe("NO VERDICT");
    expect(runLabel(run(1, "canceled", null, []))).toBe("CANCELED");
    expect(runLabel(run(1, "completed", "INCONCLUSIVE", []))).toBe("INCONCLUSIVE");
  });
});

const counts = { total: 4, queued: 0, running: 1, completed: 2, failed: 1, canceled: 0, finished: 3, valid: 2, execution_errors: 1, invalid_results: 0, error_kinds: { rate_limited: 1 } };
const batch = {
  id: "b1", name: "Baseline", mode: "verification", status: "running", project_id: "p1", project_exists: true, procedure_name: "Cubes", template_id: "t1", template_version: 2,
  video: { id: "v2", filename: "attempt.mp4", duration_sec: 12, checksum_sha256: "abc" }, repetitions: 4, max_concurrency: 2, independent_inference: true,
  ground_truth: { expected_verdict: "PASS", step_labels: {} }, fingerprint: "f00", git_revision: "b608ce5", model_id: "muse-wonder-1.0", segmentation_model_id: "", created_at: "2026-11-04T10:00:00", started_at: null, finished_at: null, wall_clock_seconds: 60,
  counts, steps: [{ step_id: "s1", name: "Place red", required: true }, { step_id: "s2", name: "Place blue", required: true }],
  config: { settings: {}, inference_policy: "fresh", shared_deterministic: [] },
  runs: [
    run(1, "completed", "PASS", [step("s1", "PASS"), step("s2", "PASS")]),
    run(2, "completed", "INCONCLUSIVE", [step("s1", "PASS"), step("s2", "CANNOT_VERIFY")]),
    run(3, "failed", null, [], { error_kind: "rate_limited", error: "429" }),
    run(4, "running", null, [], { progress: { phase: "analysis", stage: "scene_analysis", stages_done: 3, stages_total: 8 } }),
  ],
  metrics: {
    counts,
    consistency: { valid_runs: 2, distribution: { PASS: 1, FAIL: 0, INCONCLUSIVE: 1 }, modal_verdicts: ["PASS", "INCONCLUSIVE"], modal_agreement: rate(1, 2), distinct_verdicts: 2 },
    accuracy: { labeled: true, expected: "PASS", exact_match: rate(1, 2), decisive_accuracy: rate(1, 1), decisive_coverage: rate(1, 2), inconclusive_rate: rate(1, 2), false_pass: { ...rate(0, 0), applicable: false }, false_fail: { ...rate(0, 2), applicable: true } },
    steps: [
      { step_id: "s1", name: "Place red", required: true, runs_with_step: 2, runs_missing_step: 0, distribution: { PASS: 2, FAIL: 0, CANNOT_VERIFY: 0, NOT_APPLICABLE: 0 }, agreement: rate(2, 2), unstable: false, missing_evidence: rate(0, 2), with_warnings: rate(0, 2), reinspected: 0, mean_evidence: 1, error_categories: {}, label: null, accuracy: null },
      { step_id: "s2", name: "Place blue", required: true, runs_with_step: 2, runs_missing_step: 0, distribution: { PASS: 1, FAIL: 0, CANNOT_VERIFY: 1, NOT_APPLICABLE: 0 }, agreement: rate(1, 2), unstable: true, missing_evidence: rate(0, 2), with_warnings: rate(0, 2), reinspected: 0, mean_evidence: 1, error_categories: {}, label: null, accuracy: null },
    ],
    detection: { valid_runs: 2, roles: [], object_views: { runs_with_views: 0, runs_with_view_errors: 0, runs_with_appearance_check: 0, mean_recovered_sightings: null, objects_without_boxes: [], relocated_after_loss: [], interrupted_tracks: [] }, warnings: {} },
    performance: { runtime_seconds: { count: 2, mean: 41.5, median: 41.5, min: 41, max: 42 }, total_run_seconds: 83, batch_wall_clock_seconds: 60, execution_error_rate: rate(1, 3), requests: 3, failed_requests: 1, input_tokens: 10, output_tokens: 5, cost_usd: null },
    independence: { runs_with_results: 3, fully_fresh_runs: 2, replayed_answers: 0 },
  },
};

describe("Experiment dashboard", () => {
  it("shows the verdict mix, keeps errors apart, and flags the unstable step", async () => {
    mockApi((url) => (url === "/api/reliability/batches/b1" ? { body: batch } : undefined));
    show("/lab/b1");
    expect(await screen.findByRole("heading", { name: "Baseline" })).toBeInTheDocument();
    expect(screen.getByText(/3 \/ 4/)).toBeInTheDocument();
    expect(screen.getAllByRole("table").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("ERROR").length).toBeGreaterThan(0);
    expect(screen.getByText("rate limited")).toBeInTheDocument();
    expect(screen.getByText(/analysis: scene analysis/)).toBeInTheDocument();
    // Run 2's S2 differs from nothing (tie) and is in an unstable column; the per-step table says so.
    expect(screen.getByText("unstable")).toBeInTheDocument();
    expect(screen.getByText("Exact match").parentElement).toHaveTextContent("50%");
    expect(screen.getByText(/1 errors, 0 without a verdict/)).toBeInTheDocument();
  });

  it("opens the run at the chosen step from a matrix cell", async () => {
    mockApi((url) => (url === "/api/reliability/batches/b1" ? { body: batch } : undefined));
    show("/lab/b1");
    const cells = await screen.findAllByRole("button", { name: "?" });
    fireEvent.click(cells[0]);
    expect(await screen.findByText("drill-down")).toBeInTheDocument();
  });
});

describe("New experiment", () => {
  const options = {
    procedures: [{ id: "p1", name: "Cubes", recordings: [{ video_id: "v2", filename: "attempt.mp4", role: "execution", duration_sec: 12, ready: true }], templates: [{ id: "t1", version: 2, name: "Cubes", family_id: "t1", acceptance_status: "accepted", created_at: "", steps: [{ step_id: "s1", name: "Place red", required: true }] }] }],
    models: { default_inference: "muse-wonder-1.0", default_segmentation: "sam-3.1" },
    live_enabled: true,
    max_concurrency: 4,
    max_repetitions: 50,
  };
  const plan = { repetitions: 5, model_id: "muse-wonder-1.0", max_requests: 1, object_view_requests: 3, object_view_locate_requests: 2, object_view_frames: 145, estimated_cost_usd: null, live_enabled: true, independent_inference: true };

  it("asks for approval before any live run and sends nothing live until approved", async () => {
    const calls = mockApi((url, init) => {
      if (url === "/api/reliability/options") return { body: options };
      if (url === "/api/reliability/batches" && init?.method === "POST") {
        const body = JSON.parse(String(init.body));
        return body.allow_live ? { body: { ...batch, id: "new" } } : { status: 409, body: { detail: { code: "live_call_required", plan } } };
      }
      if (url === "/api/reliability/batches") return { body: [] };
      if (url === "/api/settings") return { body: { video_storage: "copy" } };
      return undefined;
    });
    show("/lab");
    fireEvent.click(await screen.findByRole("button", { name: "Review and start" }));
    const dialog = await screen.findByRole("dialog", { name: "Approve live calls" });
    expect(within(dialog).getByText(/Approve 5 independent runs/)).toBeInTheDocument();
    const posts = () => calls.filter((call) => call.init?.method === "POST").map((call) => JSON.parse(String(call.init!.body)));
    expect(posts()).toHaveLength(1);
    expect(posts()[0].allow_live).toBe(false);
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(posts().every((body) => body.allow_live === false)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Review and start" }));
    fireEvent.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Approve and start" }));
    await waitFor(() => expect(posts().at(-1)?.allow_live).toBe(true));
    expect(posts().at(-1)).toMatchObject({ mode: "verification", project_id: "p1", template_id: "t1", video_id: "v2", repetitions: 5, max_concurrency: 2, independent_inference: true });
  });
});
