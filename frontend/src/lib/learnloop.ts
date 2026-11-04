/** Product API. Every value on screen comes from these endpoints. */

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly detail: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: init?.body instanceof FormData ? init?.headers : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail: unknown = response.statusText;
    try {
      detail = ((await response.json()) as { detail?: unknown }).detail ?? detail;
    } catch {
      /* The status text is the best description available. */
    }
    throw new ApiError(typeof detail === "string" ? detail : `The request failed (${response.status}).`, response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T,>(path: string, body: unknown = {}) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export type VideoStorage = "copy" | "reference";
export type ReviewState = "ai_draft" | "human_reviewed" | "accepted" | "";
export type Verdict = "PASS" | "FAIL" | "CANNOT_VERIFY" | "NOT_APPLICABLE";
export type Answer = "installed" | "not_installed" | "cannot_determine";

export interface ProcedureSummary {
  id: string;
  name: string;
  thumbnail_url: string;
  step_count: number;
  review_state: ReviewState;
  execution_count: number;
  latest_outcome: string;
}

export interface Run {
  id: string;
  video_id: string;
  status: "pending" | "running" | "completed" | "failed" | "canceled";
  error: string;
  model_id: string;
  modality: string;
  profile: string;
  source: "" | "pending" | "live" | "cached";
  cached_at: string;
  provider_requests: number;
  reference_template_id: string;
  stages: Record<string, string>;
}

export interface VideoInfo {
  id: string;
  filename: string;
  duration_sec: number;
  width?: number;
  height?: number;
  playback_url: string;
  thumbnail_url: string;
  ingest_status: string;
  ingest_progress?: number;
  ingest_detail?: string;
  ingest_error?: string;
}

export interface Recording {
  role: "demonstration" | "execution";
  video: VideoInfo;
  run: Run | null;
  analyzed_before?: boolean;
}

export interface Condition {
  id: string;
  type: string;
  subject_role: string;
  target_role: string;
  relation: string;
  state_label: string;
  evidence_requirement: "required" | "optional";
  source_evidence_class: string;
  origin: "model" | "human";
}

export interface TemplateDependency {
  step_id: string;
  category: string;
  constrains_order: boolean;
  rationale: string;
  origin: "model" | "human";
}

export interface TemplateStep {
  step_id: string;
  name: string;
  required: boolean;
  requirement_status: string;
  object_roles: string[];
  expected_action: string;
  completion_text: string;
  completion_conditions: Condition[];
  dependencies: TemplateDependency[];
  uncertainty: number;
  uncertainty_notes: string;
  demo_time_start: number | null;
  demo_time_end: number | null;
  removed: boolean;
  human_edited: string[];
}

export interface Role {
  role_id: string;
  label: string;
  source_object_ids: string[];
  original_label: string;
  visual_characteristics: string;
  uncertain: boolean;
}

export interface Template {
  id: string;
  version: number;
  name: string;
  acceptance_status: string;
  review_state: ReviewState;
  family_id: string;
  created_at: string;
  source_run_id: string;
  source_video_id: string;
  source_filename: string;
  document: { summary: string; roles: Role[]; steps: TemplateStep[]; review_log: unknown[] };
  /** Appearance references learned automatically from the demonstration. Never vectors. */
  appearance?: AppearanceStatus;
}

export interface AppearanceStatus {
  status: "ready" | "partial" | "unavailable" | "building" | "failed" | "disabled" | "missing";
  objects?: number;
  objects_learned?: number;
  references?: number;
  unavailable?: string[];
  reason?: string;
}

/** One line for a learned procedure's appearance references. */
export function appearanceLine(item?: AppearanceStatus): string {
  if (!item) return "";
  switch (item.status) {
    case "building":
      return "Building object references from the demonstration…";
    case "ready":
    case "partial": {
      const missing = (item.objects ?? 0) - (item.objects_learned ?? 0);
      const learned = `Object references ready: ${item.objects_learned} of ${item.objects} objects learned`;
      return missing > 0 ? `${learned}. References unavailable for ${missing} object${missing === 1 ? "" : "s"}.` : `${learned}.`;
    }
    case "disabled":
      return "";
    case "missing":
      return "No object references yet.";
    default:
      return `Object references unavailable${item.reason ? `: ${item.reason}` : "."}`;
  }
}

export interface Evidence {
  timestamp: number | null;
  observation: string;
  evidence_class: string;
  provenance: string;
}

export interface StepResult {
  step_id: string;
  name: string;
  required: boolean;
  verdict: Verdict;
  machine_verdict: string;
  basis: "model_evidence" | "model_reinspection" | "not_observed" | "human_confirmed" | "human_override" | "none";
  error_category: string;
  expected: string;
  observed: string;
  explanation: string;
  uncertainty_notes: string;
  evidence: Evidence[];
  demo_time_start: number | null;
  execution_time: number | null;
  history: { verdict: string; note: string }[];
}

export interface HumanObservation {
  id: string;
  step_id: string;
  answer: Answer;
  timestamp: number;
  reviewer: string;
  note: string;
  created_at: string;
}

export interface VerificationBrief {
  id: string;
  template_id: string;
  template_version: number;
  execution_video_id: string;
  execution_run_id: string;
  overall: string;
  status: string;
}

export interface Verification extends VerificationBrief {
  execution_revision_id: string;
  error: string;
  config: { provider_id: string; model_id: string; modality: string };
  human_observations: HumanObservation[];
  report: {
    overall: string;
    steps: StepResult[];
    warnings: string[];
    human_assisted: boolean;
    removed_steps: string[];
    demonstration_video_id: string;
    score: { verified: number; failed: number; unresolved: number };
    roles: { role_id: string; label: string; status: string; note: string }[];
    appearance?: {
      used: boolean;
      error: string;
      gallery: { status?: string; objects_learned?: number; objects?: number };
      recovery: { attempted?: boolean; recovered_sightings?: number; unresolved?: number };
      objects: Record<string, { validated_share: number; rejected_share: number; ambiguous_share: number }>;
    };
    reinspection?: {
      status: "pending" | "completed" | "failed" | "unavailable" | "not_needed" | "off";
      reason?: string;
      steps?: string[];
      done?: string[];
      changed?: { step_id: string; before: string; after: string }[];
    };
  };
}

export interface Procedure extends ProcedureSummary {
  recordings: Recording[];
  templates: Template[];
  template: Template | null;
  verifications: VerificationBrief[];
}

export interface CallPlan {
  model_id: string;
  modality: string;
  video_seconds: number;
  max_requests: number;
  automatic_retries: number;
  max_output_tokens: number;
  estimated_input_tokens: number;
  estimated_cost_usd: number | null;
  live_enabled: boolean;
  segmentation_model_id: string;
  object_view_requests: number;
  object_view_locate_requests: number;
  object_view_frames: number;
  fresh?: boolean;
}

export interface ModelOption {
  model_id: string;
  label?: string;
}

export interface ModelChoices {
  inference: ModelOption[];
  segmentation: ModelOption[];
  default_inference: string;
  default_segmentation: string;
}

export interface ObjectViews {
  objects: Record<string, ObjectView>;
  error?: string;
  /** A job is making them again. */
  running?: boolean;
  /** While running: 0 first-frame boxes, 1 later boxes, 2 SAM, 3 the pictures. */
  progress?: number;
  /** The analysis finished and makes object views, so they can be made again. */
  can_remake?: boolean;
}

/** What making object views again could send. Stored answers are replayed. */
export interface ObjectViewsPlan {
  model_id: string;
  segmentation_model_id: string;
  video_seconds: number;
  object_view_locate_requests: number;
  object_view_requests: number;
  object_view_frames: number;
  live_enabled: boolean;
}

export interface ObjectView {
  /** What the analysis called the object. */
  label: string;
  thumbnail_url: string | null;
  cells: { time: number; frame_time?: number | null; url: string | null; confidence?: number | null }[];
  seen_cells: number;
  track?: TrackBox[];
}

export interface TrackBox {
  time: number;
  /** validated or recovered: appearance agrees. provisional or ambiguous: drawn as unconfirmed. */
  status?: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FilmTrack {
  label: string;
  track: TrackBox[];
}

/** The box nearest this moment. Past two samples at 8 per second, the object was not found. */
export function boxAt(track: TrackBox[], time: number): TrackBox | null {
  let best: TrackBox | null = null;
  for (const item of track) {
    if (!best || Math.abs(item.time - time) < Math.abs(best.time - time)) best = item;
  }
  return best && Math.abs(best.time - time) <= 0.25 ? best : null;
}

const SIDES = ["x", "y", "w", "h"] as const;

/** Each box averaged with its neighbors. SAM's boxes wobble from one sample to the next by about as much as the object moves. */
export function smoothTrack(track: TrackBox[]): TrackBox[] {
  const sorted = [...track].sort((a, b) => a.time - b.time);
  return sorted.map((item, index) => {
    const near = [sorted[index - 1], sorted[index + 1]].filter((other) => other && Math.abs(other.time - item.time) <= 0.25);
    const weight = 2 + near.length;
    const out = { ...item };
    for (const side of SIDES) out[side] = (2 * item[side] + near.reduce((sum, other) => sum + other[side], 0)) / weight;
    return out;
  });
}

/** The box at this moment, gliding between samples, so a playing video's boxes move instead of jumping eight times a second. */
export function boxBetween(track: TrackBox[], time: number): TrackBox | null {
  const after = track.findIndex((item) => item.time > time);
  const before = after === -1 ? track[track.length - 1] : track[after - 1];
  const next = after === -1 ? undefined : track[after];
  if (before && next && next.time - before.time <= 0.25) {
    const share = (time - before.time) / (next.time - before.time);
    const out = { ...before, time };
    for (const side of SIDES) out[side] = before[side] + (next[side] - before[side]) * share;
    return out;
  }
  return boxAt(track, time);
}

/** The newest version of each learned procedure, newest first. Versions of one are edits, not separate procedures. */
export function learnedProcedures(templates: Template[]): Template[] {
  const newest = new Map<string, Template>();
  for (const item of templates) newest.set(item.family_id, item);
  return [...newest.values()].reverse();
}

export interface TemplatePatch {
  name?: string;
  acceptance_status?: "accepted" | "draft";
  steps?: (Partial<Omit<TemplateStep, "completion_conditions" | "dependencies">> & { completion_conditions?: Partial<Condition>[]; order_after?: string[] })[];
  roles?: { role_id: string; label: string }[];
}

export const api = {
  models: () => request<ModelChoices>("/api/learnloop/models"),
  status: () => request<{ model_id: string; modality: string; max_clip_seconds: number; live_enabled: boolean; meta_key_configured: boolean }>("/api/learnloop/status"),
  procedures: () => request<ProcedureSummary[]>("/api/learnloop/procedures"),
  createProcedure: (name: string) => post<ProcedureSummary>("/api/learnloop/procedures", { name }),
  procedure: (id: string) => request<Procedure>(`/api/learnloop/procedures/${id}`),
  upload: (id: string, role: Recording["role"], file: File, onProgress?: (ratio: number) => void, onStart?: (abort: () => void) => void) =>
    new Promise<Recording>((resolve, reject) => {
      const form = new FormData();
      form.append("role", role);
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/learnloop/procedures/${id}/recordings`);
      onStart?.(() => xhr.abort());
      xhr.responseType = "json";
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total);
      };
      xhr.onload = () => {
        const payload = xhr.response as (Recording & { detail?: unknown }) | null;
        if (xhr.status >= 200 && xhr.status < 300 && payload) resolve(payload);
        else reject(new Error(typeof payload?.detail === "string" ? payload.detail : `The upload failed (${xhr.status}).`));
      };
      xhr.onerror = () => reject(new Error("The upload failed. Check that the server is running."));
      xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
      xhr.send(form);
    }),
  storage: () => request<{ video_storage: VideoStorage }>("/api/settings"),
  saveStorage: (mode: VideoStorage) => request<{ video_storage: VideoStorage }>("/api/settings", { method: "PUT", body: JSON.stringify({ video_storage: mode }) }),
  locate: (file: File) => post<{ path: string }>("/api/videos/locate", { name: file.name, size: file.size, mtime_ms: Math.round(file.lastModified) }),
  renameProcedure: (id: string, name: string) => request<ProcedureSummary>(`/api/learnloop/procedures/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  removeProcedure: (id: string) => request<{ deleted: string }>(`/api/learnloop/procedures/${id}`, { method: "DELETE" }),
  reference: (id: string, role: Recording["role"], path: string) => post<Recording>(`/api/learnloop/procedures/${id}/recordings/reference`, { path, role }),
  removeRecording: (videoId: string) => request<{ deleted: string }>(`/api/learnloop/recordings/${videoId}`, { method: "DELETE" }),
  analyze: (videoId: string, allowLive: boolean, models?: { model_id: string; segmentation_model_id: string | null; fresh?: boolean }) =>
    post<Run>(`/api/learnloop/recordings/${videoId}/analyze`, { allow_live: allowLive, ...models }),
  procedureRecording: (videoId: string) => request<Recording>(`/api/learnloop/recordings/${videoId}`),
  run: (runId: string) => request<Run>(`/api/learnloop/runs/${runId}`),
  objectViews: (runId: string) => request<ObjectViews>(`/api/learnloop/runs/${runId}/objects`),
  remakeObjectViews: (runId: string, allowLive: boolean) => post<ObjectViews>(`/api/learnloop/runs/${runId}/objects/run`, { allow_live: allowLive }),
  learn: (id: string, runId: string) => post<Template>(`/api/learnloop/procedures/${id}/learn`, { run_id: runId }),
  revise: (templateId: string, patch: TemplatePatch) => request<Template>(`/api/workflow-templates/${templateId}`, { method: "PATCH", body: JSON.stringify(patch) }),
  removeTemplate: (templateId: string) => request<{ deleted: string }>(`/api/learnloop/templates/${templateId}`, { method: "DELETE" }),
  verify: (templateId: string, runId: string) => post<Verification>(`/api/learnloop/templates/${templateId}/verify`, { run_id: runId }),
  verification: (id: string) => request<Verification>(`/api/verifications/${id}`),
  observe: (id: string, stepId: string, body: { answer: Answer; timestamp: number; reviewer?: string; note: string }) =>
    post<Verification>(`/api/verifications/${id}/steps/${stepId}/observations`, body),
};

export const frameUrl = (videoId: string, time: number) => `/api/learnloop/videos/${videoId}/frame?t=${time.toFixed(2)}`;

const MODEL_ACRONYMS = new Set(["sam"]);

/** "muse-wonder-1.0" -> "Muse Wonder 1.0", "sam-3.1" -> "SAM 3.1". */
export function humanizeModel(modelId: string): string {
  return modelId
    .split(/[-_]/)
    .map((word) => (MODEL_ACRONYMS.has(word.toLowerCase()) ? word.toUpperCase() : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

export function clock(value: number | null | undefined): string {
  if (value == null) return "—";
  const minutes = Math.floor(value / 60);
  return `${minutes}:${(value - minutes * 60).toFixed(1).padStart(4, "0")}`;
}

export const reviewLabels: Record<string, string> = {
  ai_draft: "AI-generated draft",
  human_reviewed: "Human-reviewed",
  accepted: "Accepted",
};

export const verdictLabels: Record<string, string> = {
  PASS: "PASS",
  FAIL: "FAIL",
  CANNOT_VERIFY: "CANNOT VERIFY",
  NOT_APPLICABLE: "NOT APPLICABLE",
  INCONCLUSIVE: "INCONCLUSIVE",
  INCOMPLETE: "INCOMPLETE",
  NOT_EVALUATED: "NOT EVALUATED",
};

export const basisLabels: Record<StepResult["basis"], string> = {
  model_evidence: "Evidence-backed from Meta analysis",
  model_reinspection: "Decided by a second Meta look at a few frames",
  not_observed: "Not seen anywhere in this recording",
  human_confirmed: "Human-confirmed",
  human_override: "Human override",
  none: "No deciding evidence",
};

export const errorLabels: Record<string, string> = {
  MISSING_COMPONENT: "Missing component",
  MISSING_STEP: "Missing step",
  INCORRECT_COMPONENT: "Wrong component",
  INCORRECT_ORDER: "Wrong order",
  INCORRECT_PLACEMENT: "Wrong placement",
  INCOMPLETE_OPERATION: "Incomplete operation",
  INCORRECT_PHYSICAL_STATE: "Wrong physical state",
  UNVERIFIABLE_OPERATION: "Not enough evidence",
  TEMPORARY_ERROR_CORRECTED: "Corrected during the attempt",
};

export const relationLabels: Record<string, string> = {
  attached: "attached to",
  inside: "inside",
  supported_by: "resting on",
  above: "above",
  below: "below",
  left_of: "left of",
  right_of: "right of",
  around: "around",
  touching: "touching",
};

export function verdictTone(verdict: string): "pass" | "fail" | "unknown" | "muted" {
  if (verdict === "PASS") return "pass";
  if (verdict === "FAIL") return "fail";
  if (verdict === "CANNOT_VERIFY" || verdict === "INCONCLUSIVE" || verdict === "INCOMPLETE") return "unknown";
  return "muted";
}

export function conditionText(condition: Partial<Condition>, roles: Role[]): string {
  const label = (id?: string) => roles.find((role) => role.role_id === id)?.label ?? id ?? "";
  if (condition.relation) {
    return `${label(condition.subject_role)} is ${relationLabels[condition.relation] ?? condition.relation.replaceAll("_", " ")} ${label(condition.target_role)}`;
  }
  return `${label(condition.subject_role)} is ${condition.state_label ?? ""}`;
}
