import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProcedurePage } from "../pages/Procedure";
import { ProjectsPage } from "../pages/Projects";
import { TeachPage } from "../pages/Teach";
import { VerifyPage } from "../pages/Verify";

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
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/procedures/:id/teach" element={<TeachPage />} />
          <Route path="/procedures/:id/procedure" element={<ProcedurePage />} />
          <Route path="/procedures/:id/verify" element={<VerifyPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const video = { id: "v1", filename: "demo.mp4", duration_sec: 10, playback_url: "/p/v1", thumbnail_url: "", ingest_status: "ready" };
const attempt = { ...video, id: "v2", filename: "attempt.mp4", playback_url: "/p/v2" };
const run = { id: "r2", video_id: "v2", status: "completed", error: "", model_id: "muse-wonder-1.0", modality: "native_video", profile: "hackathon_guided", source: "cached", cached_at: "", provider_requests: 0, reference_template_id: "t1", stages: {} };
const roles = [
  { role_id: "role:green", label: "green cube", original_label: "", visual_characteristics: "", uncertain: false },
  { role_id: "role:base", label: "gray base", original_label: "", visual_characteristics: "", uncertain: false },
];
const templateStep = {
  step_id: "s1", name: "Place green cube", required: true, requirement_status: "required", object_roles: ["role:green", "role:base"],
  expected_action: "Place it", completion_text: "On the base", uncertainty: 0.2, uncertainty_notes: "", demo_time_start: 4, demo_time_end: 5, removed: false, human_edited: [],
  completion_conditions: [{ id: "rel:0", type: "physical_relationship", subject_role: "role:green", target_role: "role:base", relation: "attached", state_label: "", evidence_requirement: "required", source_evidence_class: "DIRECTLY_OBSERVED", origin: "model" }],
};
const template = (state: string, step = templateStep) => ({
  id: "t1", version: 1, name: "Cubes", acceptance_status: state === "accepted" ? "accepted" : "draft", review_state: state, source_video_id: "v1",
  document: { summary: "Summary", roles, steps: [step], review_log: [] },
});
const summary = { id: "p1", name: "Cubes", thumbnail_url: "", step_count: 1, review_state: "accepted", execution_count: 1, latest_outcome: "FAIL" };
const procedure = (over = {}) => ({
  ...summary, template: template("accepted"), templates: [template("accepted")],
  recordings: [{ role: "demonstration", video, run: null }, { role: "execution", video: attempt, run }],
  verifications: [{ id: "c1", template_id: "t1", template_version: 1, execution_video_id: "v2", execution_run_id: "r2", overall: "FAIL", status: "completed" }],
  ...over,
});
const stepResult = {
  step_id: "s1", name: "Place green cube", required: true, verdict: "FAIL", machine_verdict: "FAIL", basis: "model_evidence", error_category: "MISSING_COMPONENT",
  expected: "green cube attached gray base", observed: "The expected role was not observed.", explanation: "The anchor is visible.", uncertainty_notes: "",
  evidence: [{ timestamp: 9.5, observation: "Anchor is visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "DETERMINISTIC" }], demo_time_start: 4, execution_time: 9.5, history: [],
};
const report = (step = stepResult, overall = "FAIL", assisted = false) => ({
  id: "c1", template_id: "t1", template_version: 1, execution_video_id: "v2", execution_run_id: "r2", execution_revision_id: "rev", status: "completed", overall, error: "",
  config: { provider_id: "meta", model_id: "muse-wonder-1.0", modality: "native_video" },
  human_observations: [],
  report: { overall, steps: [step], warnings: [], human_assisted: assisted, removed_steps: [], demonstration_video_id: "v1", score: { verified: 0, failed: 1, unresolved: 0 }, roles: [] },
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Projects", () => {
  it("shows the promise, each procedure's state, and creates a procedure", async () => {
    const calls = mockApi((url, init) => {
      if (url === "/api/learnloop/procedures" && init?.method === "POST") return { body: { ...summary, id: "new" } };
      if (url === "/api/learnloop/procedures") return { body: [summary] };
      if (url.includes("/procedures/new")) return { body: procedure({ recordings: [], templates: [], template: null }) };
      if (url.includes("/status")) return { body: { model_id: "muse-wonder-1.0", max_clip_seconds: 60 } };
      return undefined;
    });
    show("/");
    expect(screen.getByRole("heading", { name: "Learn a task. Verify the next attempt." })).toBeTruthy();
    expect(await screen.findByText("Cubes")).toBeTruthy();
    expect(screen.getByText("Accepted")).toBeTruthy();
    expect(screen.getByText("FAIL")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Create procedure" }));
    fireEvent.change(screen.getByLabelText("Procedure name"), { target: { value: "New" } });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(calls.some((call) => call.init?.method === "POST")).toBe(true));
    expect(await screen.findByRole("heading", { name: "Teach" })).toBeTruthy();
  });

  it("reports a failed load instead of spinning", async () => {
    mockApi(() => ({ status: 500, body: { detail: "The database is unavailable." } }));
    show("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("The database is unavailable.");
  });
});

describe("Teach", () => {
  let storageMode = "copy";
  const data = () => procedure({ recordings: [{ role: "demonstration", video, run: null }], templates: [], template: null, verifications: [] });
  const base = (url: string) => {
    if (url.endsWith("/api/settings")) return { body: { video_storage: storageMode } };
    if (url.includes("/status")) return { body: { model_id: "muse-wonder-1.0", max_clip_seconds: 60 } };
    if (url.endsWith("/procedures/p1")) return { body: data() };
    if (url.endsWith("/recordings/v1")) return { body: { role: "demonstration", video, run: null } };
    return undefined;
  };

  it("uploads a demonstration from the drop zone and shows progress", async () => {
    mockApi(base);
    const sent: FormData[] = [];
    let request: { upload: { onprogress?: (event: unknown) => void } };
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:clip", revokeObjectURL: () => undefined }));
    vi.stubGlobal("XMLHttpRequest", class {
      upload = {} as { onprogress?: (event: unknown) => void };
      constructor() { request = this; }
      open() {}
      send(body: FormData) { sent.push(body); }
    });
    const { container } = show("/procedures/p1/teach");
    expect(await screen.findByText("Supported: MP4, MOV, WebM")).toBeTruthy();
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [new File(["x"], "demo.mp4", { type: "video/mp4" })] } });
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].get("role")).toBe("demonstration");
    act(() => request.upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 2 }));
    expect(await screen.findByRole("progressbar")).toHaveTextContent("50%");
  });

  it("references the original path without uploading when that is the setting", async () => {
    storageMode = "reference";
    const calls = mockApi((url) => {
      if (url === "/api/videos/locate") return { body: { path: "/clips/demo.mp4" } };
      if (url.endsWith("/recordings/reference")) return { body: { role: "demonstration", video, run: null } };
      return base(url);
    });
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:clip", revokeObjectURL: () => undefined }));
    const { container } = show("/procedures/p1/teach");
    const choice = await screen.findByRole("radio", { name: /Reference the original path/ });
    await waitFor(() => expect(choice).toBeChecked());
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [new File(["x"], "demo.mp4", { type: "video/mp4" })] } });
    await waitFor(() => expect(calls.some((call) => call.url.endsWith("/recordings/reference"))).toBe(true));
    const sent = calls.find((call) => call.url.endsWith("/recordings/reference"));
    expect(JSON.parse(String(sent?.init?.body))).toEqual({ path: "/clips/demo.mp4", role: "demonstration" });
    storageMode = "copy";
  });

  it("removes a recording with its x button", async () => {
    const calls = mockApi((url, init) => (init?.method === "DELETE" ? { body: { deleted: "v1" } } : base(url)));
    show("/procedures/p1/teach");
    fireEvent.click(await screen.findByRole("button", { name: "Remove demo.mp4" }));
    await waitFor(() => expect(calls.some((call) => call.init?.method === "DELETE" && call.url.endsWith("/recordings/v1"))).toBe(true));
  });

  it("asks before a billable call and never sends one without a yes", async () => {
    const calls = mockApi((url) => {
      if (url.endsWith("/analyze")) {
        return { status: 409, body: { detail: { code: "live_call_required", plan: { model_id: "muse-wonder-1.0", modality: "native_video", video_seconds: 10, max_requests: 1, automatic_retries: 0, max_output_tokens: 24000, estimated_input_tokens: 1, estimated_cost_usd: null, live_enabled: true } } } };
      }
      return base(url);
    });
    show("/procedures/p1/teach");
    fireEvent.click(await screen.findByRole("button", { name: "Learn the procedure" }));
    expect(await screen.findByRole("dialog")).toHaveTextContent("At most 1, with no automatic retry");
    expect(screen.getByRole("dialog")).toHaveTextContent("Unknown");
    const analyzes = calls.filter((call) => call.url.endsWith("/analyze"));
    expect(analyzes).toHaveLength(1);
    expect(JSON.parse(String(analyzes[0].init?.body)).allow_live).toBe(false);
  });

  it("shows a failed analysis and the provider's reason", async () => {
    mockApi((url) => {
      if (url.endsWith("/analyze")) return { body: { ...run, id: "rf", status: "pending" } };
      if (url.endsWith("/runs/rf")) return { body: { ...run, id: "rf", status: "failed", error: "the answer stopped before it finished (max_output_tokens)." } };
      return base(url);
    });
    show("/procedures/p1/teach");
    fireEvent.click(await screen.findByRole("button", { name: "Learn the procedure" }));
    expect(await screen.findByText(/stopped before it finished/)).toBeTruthy();
    expect(screen.getByText(/No result was generated/)).toBeTruthy();
  });
});

describe("Procedure review", () => {
  it("shows the draft state, condition, and origin, and sends edits as one revision", async () => {
    const calls = mockApi((url, init) => {
      if (url.includes("/workflow-templates/") && init?.method === "PATCH") return { body: template("accepted") };
      if (url.endsWith("/procedures/p1")) return { body: procedure({ template: template("ai_draft"), templates: [template("ai_draft")] }) };
      if (url.endsWith("/verify")) return { body: {} };
      return undefined;
    });
    show("/procedures/p1/procedure");
    expect(await screen.findByText("AI-generated draft")).toBeTruthy();
    expect(screen.getByText(/green cube is attached to gray base/)).toBeTruthy();
    expect(screen.getByText("From Meta, directly observed")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Step 1 name"), { target: { value: "Seat the cube" } });
    fireEvent.click(screen.getByRole("button", { name: "Save edits" }));
    await waitFor(() => expect(calls.some((call) => call.init?.method === "PATCH")).toBe(true));
    const patch = JSON.parse(String(calls.find((call) => call.init?.method === "PATCH")?.init?.body));
    expect(patch.steps).toEqual([{ step_id: "s1", name: "Seat the cube" }]);
  });

  it("says when a step has nothing the verifier can check", async () => {
    const bare = { ...templateStep, completion_conditions: [] };
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: procedure({ template: template("ai_draft", bare), templates: [template("ai_draft", bare)] }) } : undefined));
    show("/procedures/p1/procedure");
    expect(await screen.findByText(/No checkable condition/)).toBeTruthy();
  });
});

describe("Verify", () => {
  const handler = (step = stepResult, overall = "FAIL", assisted = false): Handler => (url) => {
    if (url.endsWith("/api/settings")) return { body: { video_storage: "copy" } };
    if (url.endsWith("/procedures/p1")) return { body: procedure() };
    if (url.endsWith("/verifications/c1")) return { body: report(step, overall, assisted) };
    return undefined;
  };

  it("explains a failure with its evidence, source, and timestamp", async () => {
    mockApi(handler());
    show("/procedures/p1/verify");
    expect(await screen.findByText("Missing component")).toBeTruthy();
    expect(screen.getByText("green cube attached gray base")).toBeTruthy();
    expect(screen.getByText("Evidence-backed from Meta analysis")).toBeTruthy();
    expect(screen.getByAltText("Frame at 0:09.5")).toHaveAttribute("src", "/api/learnloop/videos/v2/frame?t=9.50");
    expect(screen.getByText(/Cached Meta response\./)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Export report" })).toHaveAttribute("href", "/api/verifications/c1/export");
  });

  it("seeks both videos when a step is selected", async () => {
    mockApi(handler());
    const { container } = show("/procedures/p1/verify");
    await screen.findByText("Missing component");
    const [demo, execution] = Array.from(container.querySelectorAll("video"));
    Object.defineProperty(demo, "readyState", { value: 4 });
    Object.defineProperty(execution, "readyState", { value: 4 });
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
    fireEvent.click(screen.getByRole("button", { name: /Place green cube/ }));
    await waitFor(() => expect(demo.currentTime).toBe(4));
    expect(execution.currentTime).toBe(9.5);
  });

  it("asks for a named review on CANNOT VERIFY and sends an observation, not a verdict", async () => {
    const unknown = { ...stepResult, verdict: "CANNOT_VERIFY", machine_verdict: "CANNOT_VERIFY", basis: "none", error_category: "UNVERIFIABLE_OPERATION", evidence: [] };
    const calls = mockApi((url, init) => {
      if (url.endsWith("/observations") && init?.method === "POST") return { body: report({ ...stepResult, verdict: "PASS", basis: "human_confirmed", machine_verdict: "CANNOT_VERIFY", error_category: "" }, "PASS", true) };
      return handler(unknown, "INCONCLUSIVE")(url, init);
    });
    show("/procedures/p1/verify");
    const installed = await screen.findByRole("button", { name: "Installed" });
    expect(installed).toBeDisabled();
    expect(screen.getByText(/No frame supports this step/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Reviewer name"), { target: { value: "A. Reviewer" } });
    fireEvent.click(installed);
    await waitFor(() => expect(calls.some((call) => call.url.endsWith("/observations"))).toBe(true));
    const body = JSON.parse(String(calls.find((call) => call.url.endsWith("/observations"))?.init?.body));
    expect(body).toMatchObject({ answer: "installed", reviewer: "A. Reviewer", timestamp: 9.8 });
    expect(body.verdict).toBeUndefined();
  });

  it("marks a human-confirmed result as assisted", async () => {
    mockApi(handler({ ...stepResult, verdict: "PASS", basis: "human_confirmed", machine_verdict: "CANNOT_VERIFY", error_category: "" }, "PASS", true));
    show("/procedures/p1/verify");
    expect(await screen.findByText(/Includes human-confirmed observations/)).toBeTruthy();
    expect(screen.getByText(/Human-confirmed\. Automatic reading: CANNOT VERIFY/)).toBeTruthy();
  });

  it("requires an accepted procedure first", async () => {
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: procedure({ template: template("ai_draft"), templates: [template("ai_draft")], verifications: [] }) } : undefined));
    show("/procedures/p1/verify");
    expect(await screen.findByRole("link", { name: "Review the procedure" })).toBeTruthy();
  });
});
