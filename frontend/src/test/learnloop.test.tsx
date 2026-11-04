import { appearanceLine, boxAt, boxBetween, humanizeModel, smoothTrack, learnedProcedures, type Condition, type TemplateDependency } from "../lib/learnloop";
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
  { role_id: "role:green", label: "green cube", source_object_ids: [], original_label: "", visual_characteristics: "", uncertain: false },
  { role_id: "role:base", label: "gray base", source_object_ids: [], original_label: "", visual_characteristics: "", uncertain: false },
];
const templateStep = {
  step_id: "s1", name: "Place green cube", required: true, requirement_status: "required", object_roles: ["role:green", "role:base"],
  expected_action: "Place it", completion_text: "On the base", uncertainty: 0.2, uncertainty_notes: "", demo_time_start: 4, demo_time_end: 5, removed: false, human_edited: [], dependencies: [] as TemplateDependency[],
  completion_conditions: [{ id: "rel:0", type: "physical_relationship", subject_role: "role:green", target_role: "role:base", relation: "attached", state_label: "", evidence_requirement: "required", source_evidence_class: "DIRECTLY_OBSERVED", origin: "model" }],
};
const template = (state: string, step = templateStep) => ({
  id: "t1", version: 1, name: "Cubes", acceptance_status: state === "accepted" ? "accepted" : "draft", review_state: state, family_id: "t1", created_at: "2026-11-01T10:00:00", source_run_id: "r1", source_video_id: "v1", source_filename: "demo.mov",
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
    expect(humanizeModel("sam-3.1")).toBe("SAM 3.1");
    expect(humanizeModel("muse-wonder-1.0")).toBe("Muse Wonder 1.0");
  });

  it("renames a procedure in a dialog and deletes it from the card menu", async () => {
    let name = "Cubes";
    let deleted = false;
    const calls = mockApi((url, init) => {
      if (url === "/api/learnloop/procedures/p1" && init?.method === "PATCH") {
        name = JSON.parse(String(init.body)).name;
        return { body: { ...summary, name } };
      }
      if (url === "/api/learnloop/procedures/p1" && init?.method === "DELETE") {
        deleted = true;
        return { body: { deleted: "p1" } };
      }
      if (url === "/api/learnloop/procedures") return { body: deleted ? [] : [{ ...summary, name }] };
      return undefined;
    });
    vi.stubGlobal("confirm", () => true);
    show("/");
    fireEvent.click(await screen.findByRole("button", { name: "Options for Cubes" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Rename" }));
    const dialog = screen.getByRole("dialog", { name: "Rename procedure" });
    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.change(screen.getByLabelText("Procedure name"), { target: { value: "Cube tower" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Cube tower")).toBeTruthy();
    await waitFor(() => expect(dialog.isConnected).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Options for Cube tower" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete" }));
    expect(await screen.findByText(/No procedures yet/)).toBeTruthy();
    expect(calls.filter((call) => call.init?.method).map((call) => call.init?.method)).toEqual(["PATCH", "DELETE"]);
  });

  it("reports a failed load instead of spinning", async () => {
    mockApi(() => ({ status: 500, body: { detail: "The database is unavailable." } }));
    show("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("The database is unavailable.");
  });
});

describe("Deleting a learned procedure", () => {
  const api = (calls: { deleted: boolean }) => (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/learnloop/templates/t1") && init?.method === "DELETE") {
      calls.deleted = true;
      return { body: { deleted: "t1" } };
    }
    if (url.endsWith("/procedures/p1")) return { body: calls.deleted ? procedure({ template: null, templates: [], recordings: [], verifications: [] }) : procedure() };
    return undefined;
  };

  it("deletes after confirming and leaves nothing learned", async () => {
    vi.stubGlobal("confirm", () => true);
    mockApi(api({ deleted: false }));
    show("/procedures/p1/procedure");
    fireEvent.click(await screen.findByRole("button", { name: "Delete procedure" }));
    expect(await screen.findByText("Nothing has been learned yet.")).toBeTruthy();
  });

  it("keeps the procedure when the person cancels", async () => {
    vi.stubGlobal("confirm", () => false);
    const calls = mockApi(api({ deleted: false }));
    show("/procedures/p1/procedure");
    fireEvent.click(await screen.findByRole("button", { name: "Delete procedure" }));
    expect(calls.some((call) => call.init?.method === "DELETE")).toBe(false);
    expect(screen.getByRole("button", { name: "Delete procedure" })).toBeTruthy();
  });
});

describe("Loading skeletons", () => {
  const loaded = (url: string) => {
    if (url === "/api/learnloop/procedures") return { body: [summary] };
    if (url.endsWith("/api/settings")) return { body: { video_storage: "copy" } };
    if (url.includes("/status")) return { body: { model_id: "muse-wonder-1.0", max_clip_seconds: 60 } };
    if (url.endsWith("/procedures/p1")) return { body: procedure({ template: template("ai_draft"), templates: [template("ai_draft")] }) };
    if (url.endsWith("/verifications/c1")) return { body: report() };
    return undefined;
  };

  it("replaces the procedure cards' placeholders with the procedures", async () => {
    mockApi(loaded);
    const { container } = show("/");
    expect(screen.getByText("Loading procedures")).toHaveClass("sr-only");
    expect(container.querySelector(".cards[aria-busy='true'] .card .thumb-empty .bone-fill")).toBeInTheDocument();
    expect(await screen.findByText("Cubes")).toBeTruthy();
    expect(container.querySelector(".bone")).toBeNull();
  });

  it("draws the procedure's own layout and copy before it arrives", async () => {
    mockApi(loaded);
    const { container } = show("/procedures/p1/procedure");
    expect(screen.getByText("Loading the procedure")).toHaveClass("sr-only");
    expect(screen.getByRole("heading", { name: "Learned procedure" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Objects" })).toBeTruthy();
    expect(screen.getAllByText("What the verifier checks").length).toBeGreaterThan(0);
    expect(container.querySelector(".step .field.bone")).toBeInTheDocument();
    expect(await screen.findByLabelText("Step 1 name")).toHaveValue("Place green cube");
    expect(container.querySelector(".bone")).toBeNull();
  });

  it("does not accept a recording until Teach knows the procedure", async () => {
    mockApi(loaded);
    const { container } = show("/procedures/p1/teach");
    expect(container.querySelector(".drop-zone .bone")).toBeInTheDocument();
    expect(container.querySelector("input[type='file']")).toBeNull();
    await waitFor(() => expect(container.querySelector("input[type='file']")).toBeInTheDocument());
    expect(container.querySelector(".drop-zone .bone")).toBeNull();
  });

  it("draws the report's layout until Verify has it", async () => {
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: procedure() } : loaded(url)));
    const { container } = show("/procedures/p1/verify");
    expect(screen.getByText("Loading the report")).toHaveClass("sr-only");
    expect(container.querySelector(".overall .verdict .bone")).toBeInTheDocument();
    expect(screen.getByText("Expected")).toBeTruthy();
    expect(await screen.findByText("Missing component")).toBeTruthy();
    expect(screen.queryByText("Loading the report")).toBeNull();
    container.querySelectorAll(".frame-image img").forEach((frame) => fireEvent.load(frame));
    expect(container.querySelector(".bone")).toBeNull();
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

  it("keeps the learn button visible but disabled until a demonstration is ready", async () => {
    let demo: unknown = null;
    mockApi((url) => {
      if (url.endsWith("/procedures/p1")) return { body: procedure({ recordings: demo ? [demo] : [], templates: [], template: null, verifications: [] }) };
      if (url.endsWith("/recordings/v1")) return { body: demo };
      return base(url);
    });
    const view = show("/procedures/p1/teach");
    expect((await screen.findByRole("button", { name: "Learn the procedure" })).hasAttribute("disabled")).toBe(true);
    view.unmount();
    demo = { role: "demonstration", video: { ...video, ingest_status: "processing", ingest_detail: "Preparing playback" }, run: null };
    show("/procedures/p1/teach");
    expect(await screen.findByRole("status")).toHaveTextContent("Preparing the recording. Preparing playback");
    expect(screen.getByRole("button", { name: "Learn the procedure" }).hasAttribute("disabled")).toBe(true);
  });

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

  it("keeps a second upload's row when the first upload finishes", async () => {
    mockApi(base);
    const requests: { status: number; response: unknown; onload?: () => void; upload: { onprogress?: (event: unknown) => void } }[] = [];
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:clip", revokeObjectURL: () => undefined }));
    vi.stubGlobal("XMLHttpRequest", class {
      status = 0;
      response: unknown = null;
      onload?: () => void;
      upload = {} as { onprogress?: (event: unknown) => void };
      constructor() { requests.push(this); }
      open() {}
      send() {}
    });
    const { container } = show("/procedures/p1/teach");
    expect(await screen.findByText("Supported: MP4, MOV, WebM")).toBeTruthy();
    const input = container.querySelector('input[type="file"]')!;
    fireEvent.change(input, { target: { files: [new File(["x"], "first.mp4", { type: "video/mp4" })] } });
    fireEvent.change(input, { target: { files: [new File(["x"], "second.mp4", { type: "video/mp4" })] } });
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(screen.getByText("first.mp4")).toBeTruthy();
    expect(screen.getByText("second.mp4")).toBeTruthy();
    act(() => requests[1].upload.onprogress?.({ lengthComputable: true, loaded: 1, total: 2 }));
    await act(async () => {
      requests[0].status = 200;
      requests[0].response = { role: "demonstration", video, run: null };
      requests[0].onload?.();
    });
    await waitFor(() => expect(screen.queryByText("first.mp4")).toBeNull());
    expect(screen.getByText("second.mp4")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "Uploading" })).toHaveTextContent("50%");
  });

  it("rejects a file that is not a supported video without uploading it", async () => {
    mockApi(base);
    const sent: unknown[] = [];
    vi.stubGlobal("XMLHttpRequest", class {
      upload = {};
      open() {}
      send(body: unknown) { sent.push(body); }
    });
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: () => "blob:clip", revokeObjectURL: () => undefined }));
    const { container } = show("/procedures/p1/teach");
    expect(await screen.findByText("Supported: MP4, MOV, WebM")).toBeTruthy();
    const input = container.querySelector('input[type="file"]')!;
    fireEvent.change(input, { target: { files: [new File(["x"], "photo.png", { type: "image/png" })] } });
    expect(await screen.findByRole("alert")).toHaveTextContent("photo.png can't be used. Supported: MP4, MOV, WebM");
    expect(sent).toHaveLength(0);
    expect(screen.queryByText("photo.png")).toBeNull();
    fireEvent.change(input, { target: { files: [new File(["x"], "clip.mov", { type: "" })] } });
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(screen.queryByRole("alert")).toBeNull();
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

  it("counts every request a demonstration can send", async () => {
    mockApi((url) => {
      if (url.endsWith("/analyze")) {
        return { status: 409, body: { detail: { code: "live_call_required", plan: { model_id: "muse-wonder-1.0", modality: "native_video", video_seconds: 10, max_requests: 1, automatic_retries: 0, max_output_tokens: 24000, estimated_input_tokens: 1, estimated_cost_usd: null, live_enabled: true, object_view_requests: 2, object_view_locate_requests: 2, object_view_frames: 81, segmentation_model_id: "sam-3.1" } } } };
      }
      return base(url);
    });
    show("/procedures/p1/teach");
    fireEvent.click(await screen.findByRole("button", { name: "Learn the procedure" }));
    expect(await screen.findByRole("dialog")).toHaveTextContent("At most 5: 1 to analyze the video, up to 2 to find the objects, and up to 2 to follow them. No automatic retry.");
    expect(screen.getByRole("dialog")).toHaveTextContent("24,000 tokens for the analysis");
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
    // Never analyzed, so there is nothing stored to skip.
    expect(screen.queryByRole("checkbox", { name: /Analyze from scratch/ })).toBeNull();
    expect(await screen.findByRole("dialog")).toHaveTextContent("At most 1, with no automatic retry");
    expect(screen.getByRole("dialog")).toHaveTextContent("Unknown");
    const analyzes = calls.filter((call) => call.url.endsWith("/analyze"));
    expect(analyzes).toHaveLength(1);
    expect(JSON.parse(String(analyzes[0].init?.body)).allow_live).toBe(false);
  });

  it("sends the picked models, and segmentation can be turned off", async () => {
    const calls = mockApi((url) => {
      if (url.endsWith("/models")) {
        return { body: { inference: [{ model_id: "muse-wonder-1.0" }, { model_id: "muse-spark-1.3" }], segmentation: [{ model_id: "sam-3.1" }], default_inference: "muse-wonder-1.0", default_segmentation: "sam-3.1" } };
      }
      if (url.endsWith("/analyze")) return { status: 500, body: { detail: "stop here" } };
      return base(url);
    });
    show("/procedures/p1/teach");
    const inference = await screen.findByRole("combobox", { name: "Inference model" });
    const segmentation = screen.getByRole("combobox", { name: "Segmentation model" });
    expect(inference).toHaveValue("muse-wonder-1.0");
    expect(segmentation).toHaveValue("sam-3.1");
    expect(screen.getByRole("option", { name: "SAM 3.1" })).toBeTruthy();
    fireEvent.change(inference, { target: { value: "muse-spark-1.3" } });
    fireEvent.change(segmentation, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Learn the procedure" }));
    await screen.findByRole("alert");
    const body = JSON.parse(String(calls.find((call) => call.url.endsWith("/analyze"))?.init?.body));
    expect(body).toMatchObject({ model_id: "muse-spark-1.3", segmentation_model_id: "" });
  });

  it("analyzes from scratch only after the person approves new requests", async () => {
    const calls = mockApi((url, init) => {
      if (url.endsWith("/analyze")) {
        const body = JSON.parse(String(init?.body));
        if (!body.allow_live) return { status: 409, body: { detail: { code: "live_call_required", plan: { model_id: "muse-wonder-1.0", modality: "native_video", video_seconds: 10, max_requests: 1, automatic_retries: 0, max_output_tokens: 24000, estimated_input_tokens: 1, estimated_cost_usd: null, live_enabled: true, fresh: body.fresh } } } };
        return { status: 500, body: { detail: "stop here" } };
      }
      const seen = { role: "demonstration", video, run: null, analyzed_before: true };
      if (url.endsWith("/procedures/p1")) return { body: procedure({ recordings: [seen], templates: [], template: null, verifications: [] }) };
      if (url.endsWith("/recordings/v1")) return { body: seen };
      return base(url);
    });
    show("/procedures/p1/teach");
    fireEvent.click(await screen.findByRole("checkbox", { name: /Analyze from scratch/ }));
    fireEvent.click(screen.getByRole("button", { name: "Learn the procedure" }));
    expect(await screen.findByRole("dialog")).toHaveTextContent("Stored responses for this video are skipped.");
    fireEvent.click(screen.getByRole("button", { name: "Send one request" }));
    await screen.findByRole("alert");
    const bodies = calls.filter((call) => call.url.endsWith("/analyze")).map((call) => JSON.parse(String(call.init?.body)));
    expect(bodies.map((body) => [body.allow_live, body.fresh])).toEqual([[false, true], [true, true]]);
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
  it("makes object views again only after the person approves the requests", async () => {
    const state = { running: false };
    const plan = { model_id: "muse-wonder-1.0", segmentation_model_id: "sam-3.1", video_seconds: 10, object_view_locate_requests: 2, object_view_requests: 2, object_view_frames: 81, live_enabled: true };
    const calls = mockApi((url, init) => {
      if (url.endsWith("/procedures/p1")) return { body: procedure() };
      if (url.endsWith("/objects/run")) {
        const approved = JSON.parse(String(init?.body)).allow_live;
        if (!approved) return { status: 409, body: { detail: { code: "live_call_required", plan } } };
        state.running = true;
        return { body: { objects: {}, error: "", running: true, progress: 0, can_remake: true } };
      }
      if (url.endsWith("/objects")) return { body: { objects: {}, error: "No stored SAM answer matches, and a live call was not approved.", running: state.running, progress: state.running ? 2 : 0, can_remake: true } };
      return undefined;
    });
    show("/procedures/p1/procedure");
    expect(await screen.findByText(/Object views were not made: No stored SAM answer matches/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Run object views" }));
    const dialog = await screen.findByRole("dialog", { name: "Approve object view requests" });
    expect(dialog).toHaveTextContent("At most 4: up to 2");
    expect(screen.queryByRole("button", { name: "Run object views" })).toBeNull();
    expect(calls.filter((call) => call.url.endsWith("/objects/run")).map((call) => JSON.parse(String(call.init?.body)).allow_live)).toEqual([false]);
    fireEvent.click(screen.getByRole("button", { name: "Send requests" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Making object views");
    const progress = await screen.findByRole("list", { name: "Object view progress" });
    await waitFor(() => expect(progress.querySelector("li.running")).toHaveTextContent("Follow each object"));
    expect(progress.querySelectorAll("li.completed")).toHaveLength(2);
    expect(calls.filter((call) => call.url.endsWith("/objects/run")).map((call) => JSON.parse(String(call.init?.body)).allow_live)).toEqual([false, true]);
  });

  it("shows the draft state, condition, and origin, and sends edits as one revision", async () => {
    const calls = mockApi((url, init) => {
      if (url.includes("/workflow-templates/") && init?.method === "PATCH") return { body: template("accepted") };
      if (url.endsWith("/procedures/p1")) return { body: procedure({ template: template("ai_draft"), templates: [template("ai_draft")] }) };
      if (url.endsWith("/verify")) return { body: {} };
      return undefined;
    });
    show("/procedures/p1/procedure");
    expect(await screen.findByText("AI-generated draft")).toBeTruthy();
    expect(screen.getByLabelText("Check 1 object")).toHaveValue("role:green");
    expect(screen.getByLabelText("Check 1 relation")).toHaveValue("attached");
    expect(screen.getByLabelText("Check 1 target")).toHaveValue("role:base");
    expect(screen.getByText("From Meta, directly observed")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Step 1 name"), { target: { value: "Seat the cube" } });
    fireEvent.click(screen.getByRole("button", { name: "Save edits" }));
    await waitFor(() => expect(calls.some((call) => call.init?.method === "PATCH")).toBe(true));
    const patch = JSON.parse(String(calls.find((call) => call.init?.method === "PATCH")?.init?.body));
    expect(patch.steps).toEqual([{ step_id: "s1", name: "Seat the cube" }]);
  });

  it("edits each check on its own and keeps the rest", async () => {
    const second = { ...templateStep.completion_conditions[0], id: "rel:1", subject_role: "role:base", target_role: "role:green", relation: "inside" };
    const step = { ...templateStep, completion_conditions: [...templateStep.completion_conditions, second] };
    const calls = mockApi((url, init) => {
      if (url.includes("/workflow-templates/") && init?.method === "PATCH") return { body: template("accepted") };
      if (url.endsWith("/procedures/p1")) return { body: procedure({ template: template("ai_draft", step), templates: [template("ai_draft", step)] }) };
      return undefined;
    });
    show("/procedures/p1/procedure");
    fireEvent.change(await screen.findByLabelText("Check 2 relation"), { target: { value: "touching" } });
    expect(screen.getByLabelText("Check 1 relation")).toHaveValue("attached");
    expect(screen.getByText("Human-specified")).toBeTruthy();
    expect(screen.getByText("From Meta, directly observed")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Add check" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove check 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Save edits" }));
    await waitFor(() => expect(calls.some((call) => call.init?.method === "PATCH")).toBe(true));
    const patch = JSON.parse(String(calls.find((call) => call.init?.method === "PATCH")?.init?.body));
    expect(patch.steps[0].completion_conditions.map((item: Condition) => [item.id, item.relation])).toEqual([["rel:1", "touching"], [undefined, "attached"]]);
  });

  const untaught = (status: string, error = "") =>
    procedure({ templates: [], template: null, verifications: [], recordings: [{ role: "demonstration", video, run: { ...run, id: "r1", video_id: "v1", status, error, stages: { scene_analysis: "running" } } }] });

  it("shows teaching in progress", async () => {
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: untaught("running") } : undefined));
    show("/procedures/p1/procedure");
    expect(await screen.findByText("Learning from demo.mp4")).toBeTruthy();
    expect(screen.getByText("Meta analysis")).toHaveClass("running");
    expect(screen.queryByText("Nothing has been learned yet.")).toBeNull();
  });

  it("shows a placeholder and selects the demonstration being learned", async () => {
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: untaught("running") } : undefined));
    const { container } = show("/procedures/p1/procedure");
    const item = await screen.findByRole("button", { name: /demo\.mp4 · Learning/ });
    expect(item).toHaveAttribute("aria-current", "true");
    expect(item.closest(".panel")?.querySelector(".bone")).toBeNull();
    expect(container.querySelector(".step .field.bone")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Learned procedures" })).toBeTruthy();
  });

  it("still lets the person open an earlier procedure while one is being learned", async () => {
    const earlier = template("accepted");
    mockApi((url) => {
      if (url.endsWith("/procedures/p1")) return { body: { ...untaught("running"), template: earlier, templates: [earlier] } };
      if (url.endsWith("/objects")) return { body: { objects: {} } };
      return undefined;
    });
    show("/procedures/p1/procedure");
    await screen.findByRole("button", { name: /demo\.mp4 · Learning/ });
    expect(screen.queryByLabelText("Step 1 name")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /demo\.mov · Latest/ }));
    expect(await screen.findByLabelText("Step 1 name")).toHaveValue("Place green cube");
    expect(screen.getByText("The procedure below is from an earlier demonstration.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /demo\.mp4 · Learning/ }));
    await waitFor(() => expect(screen.queryByLabelText("Step 1 name")).toBeNull());
  });

  it("expands an object's timeline frame and steps past the moments it was not found", async () => {
    const tracked = template("accepted");
    tracked.document.roles = [{ ...roles[0], source_object_ids: ["o1"] as never[] }];
    mockApi((url) => {
      if (url.endsWith("/procedures/p1")) return { body: { ...procedure(), template: tracked, templates: [tracked] } };
      if (url.endsWith("/objects")) {
        return { body: { objects: { o1: { thumbnail_url: null, seen_cells: 2, cells: [{ time: 0, url: "/f0" }, { time: 1, url: null }, { time: 2, url: "/f2", confidence: 0.874 }] } } } };
      }
      return undefined;
    });
    show("/procedures/p1/procedure");
    fireEvent.click(await screen.findByRole("button", { name: "Expand frame at 0:02.0" }));
    expect(screen.getByTitle("Seen in 2 timeline frames")).toHaveTextContent("2");
    expect(screen.getByRole("dialog", { name: "green cube · 0:02.0" }).querySelector("img")).toHaveAttribute("src", "/f2");
    expect(screen.getByRole("dialog").querySelector(".shot-score")).toHaveTextContent("87%");
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("dialog", { name: "green cube · 0:00.0" }).querySelector("img")).toHaveAttribute("src", "/f0");
    expect(screen.getByRole("dialog").querySelector(".shot-score")).toBeNull();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("shows an object's full frame with only its own box", async () => {
    const tracked = template("accepted");
    tracked.document.roles = [{ ...roles[0], source_object_ids: ["o1"] as never[] }, { ...roles[1], source_object_ids: ["o2"] as never[] }];
    const cells = [{ time: 1, frame_time: 1.375, url: "/f1" }, { time: 2, frame_time: 1.875, url: "/f2" }];
    mockApi((url) => {
      if (url.endsWith("/procedures/p1")) return { body: { ...procedure(), template: tracked, templates: [tracked] } };
      if (url.endsWith("/objects")) {
        return { body: { objects: {
          o1: { thumbnail_url: null, seen_cells: 2, cells, track: [{ time: 1.375, x: 0.6, y: 0.6, w: 0.1, h: 0.1 }, { time: 1.875, x: 0.1, y: 0.2, w: 0.3, h: 0.4 }] },
          o2: { thumbnail_url: null, seen_cells: 1, cells: [{ time: 2, url: "/g2" }], track: [{ time: 2, x: 0.5, y: 0.5, w: 0.1, h: 0.1 }] },
        } } };
      }
      return undefined;
    });
    show("/procedures/p1/procedure");
    // Each cell is labeled with the time its frame came from, not the second it stands for.
    expect(await screen.findByRole("button", { name: "Expand frame at 0:01.4" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Expand frame at 0:01.9" }));
    const dialog = screen.getByRole("dialog");
    expect(dialog.querySelector(".asset-preview-box")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Full image" }));
    const image = dialog.querySelector("img") as HTMLImageElement;
    expect(image).toHaveAttribute("src", "/api/learnloop/videos/v1/frame?t=1.88");
    Object.defineProperty(image, "naturalWidth", { value: 1920 });
    Object.defineProperty(image, "naturalHeight", { value: 1080 });
    fireEvent.load(image);
    const boxes = dialog.querySelectorAll<HTMLElement>(".asset-preview-box");
    expect(boxes).toHaveLength(1);
    expect(boxes[0].style.left).toBe("10%");
    expect(boxes[0].style.height).toBe("40%");
    fireEvent.click(screen.getByRole("button", { name: "Object only" }));
    expect(dialog.querySelector("img")).toHaveAttribute("src", "/f2");
    expect(dialog.querySelector(".asset-preview-box")).toBeNull();
  });

  it("learns once when the analysis finished after leaving Teach", async () => {
    const calls = mockApi((url) => {
      if (url.endsWith("/procedures/p1")) return { body: untaught("completed") };
      if (url.endsWith("/learn")) return { body: template("ai_draft") };
      return undefined;
    });
    show("/procedures/p1/procedure");
    await waitFor(() => expect(calls.some((call) => call.url.endsWith("/learn"))).toBe(true));
    expect(JSON.parse(String(calls.find((call) => call.url.endsWith("/learn"))?.init?.body))).toEqual({ run_id: "r1" });
  });

  it("shows why teaching failed", async () => {
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: untaught("failed", "the answer stopped before it finished.") } : undefined));
    show("/procedures/p1/procedure");
    expect(await screen.findByRole("alert")).toHaveTextContent("the answer stopped before it finished.");
  });

  it("says when a step has nothing the verifier can check", async () => {
    const bare = { ...templateStep, completion_conditions: [] };
    mockApi((url) => (url.endsWith("/procedures/p1") ? { body: procedure({ template: template("ai_draft", bare), templates: [template("ai_draft", bare)] }) } : undefined));
    show("/procedures/p1/procedure");
    expect(await screen.findByText(/No checkable condition/)).toBeTruthy();
  });

  it("saves which steps a step must follow", async () => {
    const second = { ...templateStep, step_id: "s2", name: "Lock cube" };
    const draft = { ...template("ai_draft"), document: { ...template("ai_draft").document, steps: [templateStep, second] } };
    const sent: unknown[] = [];
    mockApi((url, init) => {
      if (url.endsWith("/procedures/p1")) return { body: procedure({ template: draft, templates: [draft] }) };
      if (init?.method === "PATCH") {
        sent.push(JSON.parse(String(init.body)));
        return { body: draft };
      }
      return undefined;
    });
    show("/procedures/p1/procedure");
    expect((await screen.findAllByText("Any order")).length).toBe(2);
    expect(screen.queryByRole("checkbox", { name: "Step 1: Place green cube" })).toBeNull();
    const row = screen.getAllByRole("button", { name: /Step order/ })[1];
    expect(row.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(row);
    expect(row.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText(/must happen before this step \(Step 2\)/)).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox", { name: "Step 1: Place green cube" }));
    expect(screen.getByText("After step 1")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Save edits" }));
    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0]).toMatchObject({ steps: [{ step_id: "s2", order_after: ["s1"] }] });
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

  it("shows a running second look as stages, one per unclear step", async () => {
    const second = { ...stepResult, step_id: "s2", name: "Place red cube" };
    mockApi((url, init) => {
      if (url.endsWith("/verifications/c1")) {
        const body = report(stepResult, "INCONCLUSIVE");
        return { body: { ...body, report: { ...body.report, steps: [stepResult, second], reinspection: { status: "pending", steps: ["s1", "s2"], done: ["s1"] } } } };
      }
      return handler()(url, init);
    });
    show("/procedures/p1/verify");
    const stages = await screen.findByRole("list", { name: "Reinspection progress" });
    const items = Array.from(stages.querySelectorAll("li"));
    expect(items.map((item) => [item.textContent, item.className])).toEqual([
      ["Look again at “Place green cube” with a few frames", "completed"],
      ["Look again at “Place red cube” with a few frames", "running"],
      ["Make the verdicts again", "pending"],
    ]);
    expect(screen.queryByText(/Reinspecting/)).toBeNull();
  });

  it("holds a frame's place until its picture loads", async () => {
    mockApi(handler());
    const { container } = show("/procedures/p1/verify");
    expect(container.querySelector(".detail .frame-image .bone-fill")).toBeInTheDocument();
    expect(container.querySelector(".detail figcaption .bone")).toBeInTheDocument();
    const frame = await screen.findByAltText("Frame at 0:09.5");
    expect(frame.parentElement?.querySelector(".bone-fill")).toBeInTheDocument();
    fireEvent.load(frame);
    expect(frame.parentElement?.querySelector(".bone-fill")).toBeNull();
  });

  it("deselects an attempt when its selected tab is clicked again", async () => {
    mockApi(handler());
    show("/procedures/p1/verify");
    await screen.findByText("Missing component");
    const tab = screen.getByRole("tab");
    expect(tab).toHaveAttribute("aria-selected", "true");
    fireEvent.click(tab);
    expect(tab).toHaveAttribute("aria-selected", "false");
    expect(screen.queryByText("Missing component")).toBeNull();
    fireEvent.click(tab);
    expect(tab).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText("Missing component")).toBeTruthy();
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

  it("asks for a review on CANNOT VERIFY and sends an observation, not a verdict", async () => {
    const unknown = { ...stepResult, verdict: "CANNOT_VERIFY", machine_verdict: "CANNOT_VERIFY", basis: "none", error_category: "UNVERIFIABLE_OPERATION", evidence: [] };
    const calls = mockApi((url, init) => {
      if (url.endsWith("/observations") && init?.method === "POST") return { body: report({ ...stepResult, verdict: "PASS", basis: "human_confirmed", machine_verdict: "CANNOT_VERIFY", error_category: "" }, "PASS", true) };
      return handler(unknown, "INCONCLUSIVE")(url, init);
    });
    show("/procedures/p1/verify");
    const installed = await screen.findByRole("button", { name: "Installed" });
    expect(screen.getByText(/No frame supports this step/)).toBeTruthy();
    expect(screen.queryByLabelText("Reviewer name")).toBeNull();
    fireEvent.click(installed);
    await waitFor(() => expect(calls.some((call) => call.url.endsWith("/observations"))).toBe(true));
    const body = JSON.parse(String(calls.find((call) => call.url.endsWith("/observations"))?.init?.body));
    expect(body).toMatchObject({ answer: "installed", timestamp: 9.8 });
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

describe("object tracks and earlier procedures", () => {
  it("finds the box at a moment and none where the object was not found", () => {
    const track = [{ time: 1, x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, { time: 1.125, x: 0.2, y: 0.1, w: 0.2, h: 0.2 }];
    expect(boxAt(track, 1.1)?.x).toBe(0.2);
    expect(boxAt(track, 3)).toBeNull();
  });

  it("glides a playing video's boxes between samples and evens out their wobble", () => {
    const track = [{ time: 1, x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, { time: 1.125, x: 0.2, y: 0.1, w: 0.2, h: 0.2 }];
    expect(boxBetween(track, 1.0625)?.x).toBeCloseTo(0.15);
    expect(boxBetween(track, 3)).toBeNull();
    const wobble = [0.1, 0.14, 0.1].map((x, index) => ({ time: index / 8, x, y: 0.1, w: 0.2, h: 0.2 }));
    expect(smoothTrack(wobble)[1].x).toBeCloseTo(0.12);
    expect(smoothTrack([wobble[0], { ...wobble[2], time: 2 }])[0].x).toBe(0.1);
  });

  it("lists each learned procedure once, by its newest version", () => {
    const base = template("ai_draft");
    const list = learnedProcedures([
      { ...base, id: "a1", family_id: "a", version: 1 },
      { ...base, id: "a2", family_id: "a", version: 2 },
      { ...base, id: "b1", family_id: "b", version: 1 },
    ] as never);
    expect(list.map((item) => item.id)).toEqual(["b1", "a2"]);
  });
});


describe("appearance status", () => {
  it("says how many objects were learned and which were not, and says nothing when the feature is off", () => {
    expect(appearanceLine({ status: "partial", objects: 6, objects_learned: 5 })).toBe("Object references ready: 5 of 6 objects learned. References unavailable for 1 object.");
    expect(appearanceLine({ status: "ready", objects: 3, objects_learned: 3 })).toBe("Object references ready: 3 of 3 objects learned.");
    expect(appearanceLine({ status: "building" })).toContain("Building object references");
    expect(appearanceLine({ status: "unavailable", reason: "No crop passed." })).toBe("Object references unavailable: No crop passed.");
    expect(appearanceLine({ status: "disabled" })).toBe("");
  });
});
