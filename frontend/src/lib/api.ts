import type { AudioModelStatus, Bundle, ClaudeCodeSettings, ClaudeCodeStatus, CodexSettings, CodexStatus, CursorCliSettings, CursorCliStatus, Dashboard, ProviderInfo, RunSummary, SamMaskOverlay, VideoInfo } from "./types";
import type { ReacquisitionData, RecoveryCandidates, RecoveryDiagnostics, RecoveryExperiment } from "./recovery";
import type { VerificationView, WorkflowTemplate } from "./verification";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) return (await response.json()) as T;
  return (await response.blob()) as T;
}

export type VideoStorage = "copy" | "reference";

export interface LabSettings {
  video_storage: VideoStorage;
}

export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  created_at: string;
  is_task: boolean;
}

export interface ProviderCall {
  id: string;
  run_id: string;
  run_name: string;
  video_id: string;
  video_filename: string;
  stage: string;
  provider_id: string;
  model_id: string;
  status: string;
  latency_ms: number;
  created_at: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  price_known: boolean;
  api_key: boolean;
}

export interface ProjectCosts {
  project_id: string;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
  unpriced_calls: number;
  unitemized_usd: number;
  calls: ProviderCall[];
}

export const api = {
  dashboard: () => request<Dashboard>("/api/dashboard"),
  projects: () => request<ProjectSummary[]>("/api/projects"),
  createProject: (name: string, description = "", spansDays = false) =>
    request<ProjectSummary>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, description, is_task: spansDays }),
    }),
  deleteProject: (id: string) => request<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),
  projectCosts: (id: string) => request<ProjectCosts>(`/api/projects/${id}/costs`),
  videos: (projectId?: string) => request<VideoInfo[]>(`/api/videos${projectId ? `?project_id=${projectId}` : ""}`),
  unassignedVideos: () => request<VideoInfo[]>("/api/videos?unassigned=true"),
  attachVideos: (projectId: string, videoIds: string[]) =>
    request<VideoInfo[]>(`/api/projects/${projectId}/videos/attach`, {
      method: "POST",
      body: JSON.stringify({ video_ids: videoIds }),
    }),
  video: (id: string) => request<VideoInfo>(`/api/videos/${id}`),
  upload: (
    projectId: string | null,
    file: File,
    onProgress?: (ratio: number) => void,
    control?: { onAbort: (abort: () => void) => void },
  ) =>
    new Promise<VideoInfo>((resolve, reject) => {
      const body = new FormData();
      if (projectId) body.set("project_id", projectId);
      body.set("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/videos");
      xhr.responseType = "json";
      control?.onAbort(() => xhr.abort());
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total);
      };
      xhr.onload = () => {
        const payload = xhr.response as { detail?: string } | VideoInfo | null;
        const detail = payload && typeof payload === "object" && "detail" in payload ? payload.detail : "";
        if (xhr.status >= 200 && xhr.status < 300 && payload && !detail) {
          resolve(payload as VideoInfo);
          return;
        }
        reject(new Error(detail || xhr.statusText || `upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
      xhr.send(body);
    }),
  deleteVideo: (id: string) => request(`/api/videos/${id}`, { method: "DELETE" }),
  settings: () => request<LabSettings>("/api/settings"),
  saveSettings: (payload: LabSettings) =>
    request<LabSettings>("/api/settings", { method: "PUT", body: JSON.stringify(payload) }),
  referenceVideo: (projectId: string | null, path: string) =>
    request<VideoInfo>("/api/videos/reference", {
      method: "POST",
      body: JSON.stringify({ ...(projectId ? { project_id: projectId } : {}), path }),
    }),
  locateVideo: (payload: { name: string; size: number; mtime_ms: number }) =>
    request<{ path: string }>("/api/videos/locate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  orderVideos: (projectId: string, videoIds: string[]) =>
    request<VideoInfo[]>(`/api/projects/${projectId}/videos/order`, {
      method: "POST",
      body: JSON.stringify({ video_ids: videoIds }),
    }),
  runs: (params: { status?: string; provider_id?: string; project_id?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.status) query.set("status", params.status);
    if (params.provider_id) query.set("provider_id", params.provider_id);
    if (params.project_id) query.set("project_id", params.project_id);
    const suffix = query.toString() ? `?${query}` : "";
    return request<RunSummary[]>(`/api/runs${suffix}`);
  },
  run: (id: string) => request<Bundle>(`/api/runs/${id}`),
  recoveryCandidates: (runId: string) => request<RecoveryCandidates>(`/api/runs/${runId}/recovery/candidates`),
  dismissRecoveryProblem: (runId: string, targetId: string) =>
    request<RecoveryCandidates>(`/api/runs/${runId}/recovery/dismissals/${targetId}`, { method: "POST" }),
  restoreRecoveryProblem: (runId: string, targetId: string) =>
    request<RecoveryCandidates>(`/api/runs/${runId}/recovery/dismissals/${targetId}`, { method: "DELETE" }),
  recoveryReacquisition: (runId: string) => request<ReacquisitionData>(`/api/runs/${runId}/recovery/reacquisition`),
  planReacquisition: (runId: string, payload: Record<string, unknown>) =>
    request<RecoveryExperiment>(`/api/runs/${runId}/recovery/reacquisition/plan`, { method: "POST", body: JSON.stringify(payload) }),
  recoveryExperiments: (runId: string) => request<RecoveryExperiment[]>(`/api/runs/${runId}/recovery/experiments`),
  recoveryExperiment: (id: string) => request<RecoveryExperiment>(`/api/recovery/experiments/${id}`),
  recoveryDiagnostics: (id: string) => request<RecoveryDiagnostics>(`/api/recovery/experiments/${id}/diagnostics`),
  createRecovery: (runId: string, payload: Record<string, unknown>) =>
    request<RecoveryExperiment>(`/api/runs/${runId}/recovery/experiments`, { method: "POST", body: JSON.stringify(payload) }),
  updateRecoverySeeds: (id: string, payload: Record<string, unknown>) =>
    request<RecoveryExperiment>(`/api/recovery/experiments/${id}/seeds`, { method: "PATCH", body: JSON.stringify(payload) }),
  recoveryDryRun: (id: string) => request<RecoveryExperiment>(`/api/recovery/experiments/${id}/dry-run`, { method: "POST" }),
  recoveryReplay: (id: string) => request<RecoveryExperiment>(`/api/recovery/experiments/${id}/replay`, { method: "POST" }),
  recoveryLive: (id: string, payload: { confirm: true; plan_hash: string; resubmit_keys?: string[] }) =>
    request<RecoveryExperiment>(`/api/recovery/experiments/${id}/live`, { method: "POST", body: JSON.stringify(payload) }),
  recoveryCancel: (id: string) => request<RecoveryExperiment>(`/api/recovery/experiments/${id}/cancel`, { method: "POST" }),
  decideRecovery: (id: string, payload: { decision: "accepted" | "rejected"; proposal_ids: string[]; include_provisional?: boolean }) =>
    request<RecoveryExperiment>(`/api/recovery/experiments/${id}/proposals/decide`, { method: "POST", body: JSON.stringify(payload) }),
  samMasks: (id: string, time: number) =>
    request<SamMaskOverlay[]>(`/api/runs/${id}/sam-masks?t=${encodeURIComponent(time.toFixed(3))}`),
  createRun: (payload: Record<string, unknown>) =>
    request<RunSummary>("/api/runs", { method: "POST", body: JSON.stringify(payload) }),
  setRunStarred: (id: string, starred: boolean) =>
    request<RunSummary>(`/api/runs/${id}/starred`, { method: "PATCH", body: JSON.stringify({ starred }) }),
  duplicateRun: (id: string) => request<RunSummary>(`/api/runs/${id}/duplicate`, { method: "POST", body: "{}" }),
  cancelRun: (id: string) => request<RunSummary>(`/api/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: string) => request(`/api/runs/${id}`, { method: "DELETE" }),
  retryStage: (id: string, stage: string) => request(`/api/runs/${id}/retry/${stage}`, { method: "POST" }),
  requests: (id: string) => request<Record<string, unknown>[]>(`/api/runs/${id}/requests`),
  scene: (id: string, t: number) => request<Record<string, unknown>>(`/api/runs/${id}/scene?t=${t}`),
  patchObject: (id: string, localId: string, payload: Record<string, unknown>) =>
    request<Bundle>(`/api/runs/${id}/objects/${localId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  addObject: (id: string, label: string, description: string) =>
    request<Bundle>(`/api/runs/${id}/objects`, { method: "POST", body: JSON.stringify({ label, description }) }),
  seedTrack: (id: string, box: { timestamp: number; x: number; y: number; w: number; h: number }) =>
    request<Bundle>(`/api/runs/${id}/perception/seed`, { method: "POST", body: JSON.stringify(box) }),
  decideHypothesis: (id: string, hypothesisId: string, decision: "confirmed" | "rejected") =>
    request<Bundle>(`/api/runs/${id}/perception/hypotheses/${hypothesisId}`, {
      method: "POST",
      body: JSON.stringify({ decision }),
    }),
  mergeInstances: (id: string, keepId: string, dropId: string) =>
    request<Bundle>(`/api/runs/${id}/perception/merge`, {
      method: "POST",
      body: JSON.stringify({ keep_id: keepId, drop_id: dropId }),
    }),
  splitInstance: (id: string, instanceId: string, timestamps: number[]) =>
    request<Bundle>(`/api/runs/${id}/perception/split`, {
      method: "POST",
      body: JSON.stringify({ instance_id: instanceId, timestamps }),
    }),
  mergeObjects: (id: string, keepId: string, dropId: string) =>
    request<Bundle>(`/api/runs/${id}/objects/merge`, {
      method: "POST",
      body: JSON.stringify({ keep_id: keepId, drop_id: dropId }),
    }),
  splitObject: (id: string, localId: string, timestamps: number[], label: string) =>
    request<Bundle>(`/api/runs/${id}/objects/${localId}/split`, {
      method: "POST",
      body: JSON.stringify({ timestamps, label }),
    }),
  patchAction: (id: string, localId: string, payload: Record<string, unknown>) =>
    request<Bundle>(`/api/runs/${id}/actions/${localId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  patchStep: (id: string, localId: string, payload: Record<string, unknown>) =>
    request<Bundle>(`/api/runs/${id}/steps/${localId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  addDependency: (id: string, payload: Record<string, unknown>) =>
    request<Bundle>(`/api/runs/${id}/dependencies`, { method: "POST", body: JSON.stringify(payload) }),
  removeDependency: (id: string, fromStep: string, toStep: string) =>
    request<Bundle>(`/api/runs/${id}/dependencies?from_step=${fromStep}&to_step=${toStep}`, { method: "DELETE" }),
  providerIds: () => request<{ id: string; local: boolean }[]>("/api/providers/ids"),
  providers: () => request<ProviderInfo[]>("/api/providers"),
  saveKey: (providerId: string, apiKey: string, baseUrl = "") =>
    request(`/api/providers/${providerId}/credentials`, {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey, base_url: baseUrl }),
    }),
  validateProvider: (providerId: string) =>
    request<{ ok: boolean }>(`/api/providers/${providerId}/validate`, { method: "POST" }),
  discoverModels: (providerId: string) =>
    request<{ catalog: ProviderInfo["models"]; discovered: string[]; error: string }>(
      `/api/providers/${providerId}/models?refresh=true`,
    ),
  addModel: (providerId: string, modelId: string) =>
    request(`/api/providers/${providerId}/models`, { method: "POST", body: JSON.stringify({ model_id: modelId }) }),
  codex: () => request<CodexStatus>("/api/providers/openai/codex"),
  saveCodex: (binary: string, transport: CodexSettings["transport"]) =>
    request<CodexSettings>("/api/providers/openai/codex", {
      method: "PUT",
      body: JSON.stringify({ binary, transport }),
    }),
  linkCodex: (force = false) =>
    request<CodexStatus>("/api/providers/openai/codex/link", {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
  claudeCode: () => request<ClaudeCodeStatus>("/api/providers/anthropic/claude-code"),
  saveClaudeCode: (binary: string, transport: ClaudeCodeSettings["transport"]) =>
    request<ClaudeCodeSettings>("/api/providers/anthropic/claude-code", {
      method: "PUT",
      body: JSON.stringify({ binary, transport }),
    }),
  linkClaudeCode: (force = false) =>
    request<ClaudeCodeStatus>("/api/providers/anthropic/claude-code/link", {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
  submitClaudeCode: (code: string) =>
    request<ClaudeCodeStatus>("/api/providers/anthropic/claude-code/link/code", {
      method: "POST",
      body: JSON.stringify({ code }),
    }),
  cursorCli: () => request<CursorCliStatus>("/api/providers/cursor/cli"),
  saveCursorCli: (binary: string, transport: CursorCliSettings["transport"]) =>
    request<CursorCliSettings>("/api/providers/cursor/cli", {
      method: "PUT",
      body: JSON.stringify({ binary, transport }),
    }),
  linkCursorCli: (force = false) =>
    request<CursorCliStatus>("/api/providers/cursor/cli/link", {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
  prompts: () =>
    request<{ id: string; key: string; name: string; version: number; body: string }[]>("/api/prompts"),
  savePrompt: (key: string, name: string, body: string) =>
    request("/api/prompts", { method: "POST", body: JSON.stringify({ key, name, body }) }),
  promoteTruth: (runId: string, name: string) =>
    request<{ id: string }>(`/api/runs/${runId}/ground-truth`, { method: "POST", body: JSON.stringify({ name }) }),
  groundTruth: (runId: string) => request<{ id: string; name: string }[]>(`/api/runs/${runId}/ground-truth`),
  evaluate: (runId: string, groundTruthId: string, tolerances: Record<string, number>) =>
    request<{ id: string; metrics: Record<string, unknown> }>(`/api/runs/${runId}/evaluations`, {
      method: "POST",
      body: JSON.stringify({ ground_truth_id: groundTruthId, tolerances }),
    }),
  compare: (left: string, right: string) =>
    request<{ id: string; report: Record<string, unknown> }>("/api/comparisons", {
      method: "POST",
      body: JSON.stringify({ left_run_id: left, right_run_id: right }),
    }),
  setMatches: (comparisonId: string, objectMatches: { left: string; right: string }[]) =>
    request<{ report: Record<string, unknown> }>(`/api/comparisons/${comparisonId}/matches`, {
      method: "POST",
      body: JSON.stringify({ object_matches: objectMatches }),
    }),
  tasks: () => request<{ id: string; name: string; description: string; created_at: string }[]>("/api/tasks"),
  task: (id: string) => request<TaskBundle>(`/api/tasks/${id}`),
  createTask: (name: string, description: string) =>
    request<TaskBundle>("/api/tasks", { method: "POST", body: JSON.stringify({ name, description }) }),
  uploadClips: (projectId: string, files: File[], researchMode: string) => {
    const body = new FormData();
    body.set("research_mode", researchMode);
    for (const file of files) body.append("files", file);
    return request<TaskBundle>(`/api/tasks/${projectId}/clips`, { method: "POST", body });
  },
  referenceClips: (projectId: string, paths: string[], researchMode: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/clips/reference`, {
      method: "POST",
      body: JSON.stringify({ paths, research_mode: researchMode }),
    }),
  createSession: (projectId: string, name: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/sessions`, { method: "POST", body: JSON.stringify({ name }) }),
  patchSession: (projectId: string, sessionId: string, payload: Record<string, unknown>) =>
    request<TaskBundle>(`/api/tasks/${projectId}/sessions/${sessionId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  splitSession: (projectId: string, sessionId: string, videoIds: string[], name: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/sessions/${sessionId}/split`, {
      method: "POST",
      body: JSON.stringify({ video_ids: videoIds, name }),
    }),
  orderClips: (projectId: string, videoIds: string[]) =>
    request<TaskBundle>(`/api/tasks/${projectId}/order`, { method: "POST", body: JSON.stringify({ video_ids: videoIds }) }),
  patchClip: (projectId: string, videoId: string, payload: Record<string, unknown>) =>
    request<TaskBundle>(`/api/tasks/${projectId}/clips/${videoId}`, { method: "PATCH", body: JSON.stringify(payload) }),
  reconcileTask: (projectId: string) => request<TaskBundle>(`/api/tasks/${projectId}/reconcile`, { method: "POST" }),
  confirmLink: (projectId: string, linkId: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/identity/confirm`, { method: "POST", body: JSON.stringify({ link_id: linkId }) }),
  rejectLink: (projectId: string, linkId: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/identity/reject`, { method: "POST", body: JSON.stringify({ link_id: linkId }) }),
  mergeIdentity: (projectId: string, keepId: string, dropId: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/identity/merge`, {
      method: "POST",
      body: JSON.stringify({ keep_id: keepId, drop_id: dropId }),
    }),
  splitIdentity: (projectId: string, linkId: string) =>
    request<TaskBundle>(`/api/tasks/${projectId}/identity/split`, { method: "POST", body: JSON.stringify({ link_id: linkId }) }),
  patchTransition: (projectId: string, transitionId: string, payload: Record<string, unknown>) =>
    request<TaskBundle>(`/api/tasks/${projectId}/transitions/${transitionId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  audioModels: () => request<AudioModelStatus>("/api/audio/models"),
  patchTranscript: (runId: string, segmentId: string, text: string) =>
    request<Bundle>(`/api/runs/${runId}/transcript/${segmentId}`, {
      method: "PATCH",
      body: JSON.stringify({ text }),
    }),
  patchAlignment: (runId: string, alignmentId: string, classification: string, explanation: string) =>
    request<Bundle>(`/api/runs/${runId}/alignments/${alignmentId}`, {
      method: "PATCH",
      body: JSON.stringify({ classification, explanation }),
    }),
  visualOnlyRun: (runId: string) =>
    request<{ id: string; name: string; status: string }>(`/api/runs/${runId}/visual-only`, { method: "POST", body: "{}" }),
  audioComparison: (runId: string, visualRunId: string, assistedRunId: string) =>
    request<Record<string, unknown>>(`/api/runs/${runId}/audio-comparison`, {
      method: "POST",
      body: JSON.stringify({ visual_run_id: visualRunId, assisted_run_id: assistedRunId }),
    }),
  workflowTemplates: (projectId: string) => request<WorkflowTemplate[]>(`/api/tasks/${projectId}/workflow-templates`),
  learnWorkflow: (projectId: string, runId: string, promoteSupported: boolean, accept: boolean) =>
    request<WorkflowTemplate>(`/api/tasks/${projectId}/workflow-templates`, {
      method: "POST",
      body: JSON.stringify({ run_id: runId, promote_supported: promoteSupported, accept }),
    }),
  reviseWorkflow: (templateId: string, payload: Record<string, unknown>) =>
    request<WorkflowTemplate>(`/api/workflow-templates/${templateId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  verifications: (projectId: string) => request<VerificationView[]>(`/api/tasks/${projectId}/verifications`),
  startVerification: (templateId: string, executionRunId: string, strategy: string, reinspect: boolean) =>
    request<VerificationView>(`/api/workflow-templates/${templateId}/verifications`, {
      method: "POST",
      body: JSON.stringify({ execution_run_id: executionRunId, strategy, reinspect }),
    }),
  verification: (id: string) => request<VerificationView>(`/api/verifications/${id}`),
  correctVerificationStep: (id: string, stepId: string, verdict: string, note: string) =>
    request<VerificationView>(`/api/verifications/${id}/steps/${stepId}`, {
      method: "PATCH",
      body: JSON.stringify({ verdict, note }),
    }),
  correctVerificationRole: (id: string, roleId: string, objectLocalId: string) =>
    request<VerificationView>(`/api/verifications/${id}/roles/${encodeURIComponent(roleId)}`, {
      method: "PATCH",
      body: JSON.stringify({ object_local_id: objectLocalId }),
    }),
  requestReinspection: (id: string, stepId: string) =>
    request<{ provider_invoked: boolean }>(`/api/verifications/${id}/reinspect`, {
      method: "POST",
      body: JSON.stringify({ step_id: stepId }),
    }),
};

export interface TaskBundle {
  project: { id: string; name: string; description: string; created_at: string };
  sessions: { id: string; name: string; ordinal: number; locked: boolean; notes: string; video_ids: string[] }[];
  clips: TaskClip[];
  order: { method?: string; fact: boolean; ranked: string[][]; unordered: string[]; ambiguous_ids: string[]; notes: string[] };
  context_hypothesis: { fact: boolean; pairs: string[][]; evidence: { support?: string }[]; note?: string };
  canonical_objects: {
    id: string;
    local_id: string;
    label: string;
    description: string;
    color: string;
    uncertain: boolean;
    history: {
      video_id: string;
      object_local_id: string;
      label: string;
      state_before: string;
      state_after: string;
      image_url: string;
      status: string;
      uncertain: boolean;
    }[];
  }[];
  links: {
    id: string;
    canonical_id: string;
    video_id: string;
    object_local_id: string;
    status: string;
    uncertain: boolean;
    note: string;
    score: number;
  }[];
  transitions: {
    id: string;
    from_video_id: string;
    to_video_id: string;
    observed_change: string;
    hypothesis: string;
    evidence_class: string;
    verified: boolean;
    gap_note: string;
  }[];
  workflow: {
    steps: { id: string; kind: string; label: string; video_ids?: string[]; evidence_class?: string; note?: string; observed_change?: string; hypothesis?: string }[];
    procedural_dependencies: { before: string; after: string; evidence_class: string; accepted: boolean; note: string }[];
    continuous_sequence: boolean;
  };
  revision: { id: string; version: number; change_summary: string; document: Record<string, unknown> } | null;
  revisions: { id: string; version: number; change_summary: string; trigger: string; document: Record<string, unknown> }[];
  time_basis: { project_time: string; clip_time: string };
  jobs_created?: number;
}

export interface TaskClip {
  id: string;
  filename: string;
  duration_sec: number;
  session_id: string | null;
  research_mode: string;
  capture_time_raw: string;
  capture_time_utc: string;
  timestamp_source: string;
  timestamp_reliability: string;
  timestamp_note: string;
  file_mtime: string;
  uploaded_at: string;
  order_locked: boolean;
  order_ambiguous: boolean;
  order_method: string;
  playback_url: string;
  thumbnail_url: string;
  run_id: string;
  actions: { local_id: string; action_type: string; description: string; start: number; end: number; evidence_class: string }[];
}

export function formatUsd(amount: number, priceKnown = true, apiKey = false): string {
  if (!priceKnown) return apiKey ? "—" : "unpriced";
  const value = Number.isFinite(amount) ? amount : 0;
  return `$${value.toFixed(4)}`;
}

export function formatTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const sign = seconds < 0 ? "-" : "";
  const abs = Math.abs(seconds);
  const mins = Math.floor(abs / 60);
  const rest = abs - mins * 60;
  return `${sign}${mins}:${rest.toFixed(2).padStart(5, "0")}`;
}

/** Run length as hours, minutes, and seconds, rounded to the nearest second. Zero units are omitted. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const parts = [
    [Math.floor(total / 3600), "h"],
    [Math.floor((total % 3600) / 60), "m"],
    [total % 60, "s"],
  ].filter(([value]) => value !== 0);
  if (!parts.length) return "0s";
  return parts.map(([value, unit]) => `${value}${unit}`).join(" ");
}

export function formatWhen(raw: string | null | undefined): string {
  const value = raw ?? "";
  const stamped = value && !/(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? `${value}Z` : value;
  const date = new Date(stamped);
  if (!value || Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

export function evidenceClass(value: string | undefined): string {
  if (value === "DIRECTLY_OBSERVED") return "observed";
  if (value === "INFERRED") return "inferred";
  return "unknown";
}
