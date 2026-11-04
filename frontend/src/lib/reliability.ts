/** Reliability Lab API. Every figure on the Lab pages comes from /api/reliability. */
import { ApiError } from "./learnloop";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, cache: "no-store", headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
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

export type Mode = "verification" | "learning";
export type RunStatus = "queued" | "running" | "completed" | "failed" | "canceled";
export type BatchStatus = "queued" | "running" | "canceling" | "completed" | "canceled";
export type Overall = "PASS" | "FAIL" | "INCONCLUSIVE";
export type StepVerdict = "PASS" | "FAIL" | "CANNOT_VERIFY" | "NOT_APPLICABLE";

/** A share with what it was computed from. value is null when the denominator is zero (shown as N/A). */
export interface Rate {
  value: number | null;
  numerator: number;
  denominator: number;
  applicable?: boolean;
}

export interface Counts {
  total: number;
  queued: number;
  running: number;
  completed: number;
  failed: number;
  canceled: number;
  finished: number;
  valid: number;
  execution_errors: number;
  invalid_results: number;
  error_kinds: Record<string, number>;
}

export interface StepOutcome {
  step_id: string;
  name: string;
  required: boolean;
  verdict: StepVerdict | null;
  raw_verdict: string;
  basis: string;
  error_category: string;
  evidence_count: number;
  evidence_timestamps: number[];
  missing_evidence: boolean;
  observed_objects: string[];
  warnings: string[];
  reason: string;
  observed: string;
  expected: string;
  reinspected: boolean;
}

export interface Inference {
  cache_policy: string;
  guided_call: string;
  fully_fresh: boolean;
  replayed_answers: number;
  object_views_live_requests: number;
  object_views_replayed_requests: number;
  locate_live_requests: number;
  locate_replayed_requests: number;
  reinspection: { step_id: string; status: string; source: string }[];
}

export interface ObjectDiagnostics {
  available: boolean;
  error?: string;
  note?: string;
  objects: Record<string, { label: string; boxes: number; seen_cells: number; cells: number; first_seen: number | null; appearance: Record<string, number>; track_segments: number | null }>;
  objects_without_boxes?: string[];
  relocated?: string[];
  located_later?: string[];
  appearance_used?: boolean;
  appearance_error?: string;
  recovered_sightings?: number;
}

export interface RunResult {
  outcome?: "verdict" | "execution_error" | "canceled" | "invalid_result";
  verdict?: Overall | null;
  raw_overall?: string;
  invalid_reason?: string;
  steps?: StepOutcome[];
  evidence_count?: number;
  // learning
  step_count?: number;
  required_steps?: number;
  checkable_steps?: number;
  role_count?: number;
  object_count?: number;
  reviewable?: boolean;
  ambiguities?: string[];
  validation?: { errors: number | null; warnings: number | null; status: string };
  diagnostics?: {
    warnings?: string[];
    roles?: { role_id: string; label: string; status: string; objects: string[] }[];
    unresolved_roles?: string[];
    ambiguous_roles?: string[];
    reinspection?: { status: string; reason: string; changed: { step_id: string; before: string; after: string }[]; incomplete?: boolean };
    objects?: ObjectDiagnostics;
    failed_stages?: Record<string, { status: string; error: string }>;
    error?: string;
  };
  execution_metadata?: {
    analysis_run_id: string;
    verification_id: string;
    model_id: string;
    segmentation_model_id: string;
    git_revision: string;
    inference: Inference;
    usage: { requests: number; failed_requests: number; input_tokens: number; output_tokens: number; cost_usd: number | null };
    stages: Record<string, { status: string; seconds: number | null }>;
  };
}

export interface LearnedStep {
  step_id: string;
  name: string;
  required: boolean;
  removed?: boolean;
  completion_text?: string;
  expected_action?: string;
  demo_time_start: number | null;
  demo_time_end: number | null;
  completion_conditions: { subject_role: string; target_role: string; relation: string; state_label?: string; to_state?: string }[];
}

export interface Learned {
  roles: { role_id: string; label: string; visual_characteristics?: string }[];
  steps: LearnedStep[];
  notes?: string[];
}

export interface LabRun {
  id: string;
  index: number;
  status: RunStatus;
  analysis_run_id: string;
  verification_id: string;
  error: string;
  error_kind: string;
  cancel_requested: boolean;
  duration_seconds: number | null;
  started_at: string | null;
  finished_at: string | null;
  progress: { phase: string; stage: string; stages_done: number; stages_total: number } | null;
  result: RunResult;
  learned?: Learned;
}

export interface StepStats {
  step_id: string;
  name: string;
  required: boolean;
  runs_with_step: number;
  runs_missing_step: number;
  distribution: Record<StepVerdict, number>;
  agreement: Rate;
  unstable: boolean;
  missing_evidence: Rate;
  with_warnings: Rate;
  reinspected: number;
  mean_evidence: number | null;
  error_categories: Record<string, number>;
  label: string | null;
  accuracy: Rate | null;
}

export interface Seconds {
  count: number;
  mean: number | null;
  median: number | null;
  min: number | null;
  max: number | null;
}

export interface Performance {
  runtime_seconds: Seconds;
  total_run_seconds: number | null;
  batch_wall_clock_seconds: number | null;
  execution_error_rate: Rate;
  requests: number;
  failed_requests: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
}

export interface VerificationMetrics {
  counts: Counts;
  consistency: { valid_runs: number; distribution: Record<Overall, number>; modal_verdicts: Overall[]; modal_agreement: Rate; distinct_verdicts: number };
  accuracy: {
    labeled: boolean;
    expected: Overall | null;
    exact_match?: Rate;
    decisive_accuracy?: Rate;
    decisive_coverage?: Rate;
    inconclusive_rate?: Rate;
    false_pass?: Rate;
    false_fail?: Rate;
  };
  steps: StepStats[];
  detection: {
    valid_runs: number;
    roles: { role_id: string; label: string; statuses: Record<string, number>; unresolved: Rate; ambiguous: Rate }[];
    object_views: {
      runs_with_views: number;
      runs_with_view_errors: number;
      runs_with_appearance_check: number;
      mean_recovered_sightings: number | null;
      objects_without_boxes: { object_id: string; label: string; runs: number }[];
      relocated_after_loss: { object_id: string; label: string; runs: number }[];
      interrupted_tracks: { object_id: string; label: string; runs: number }[];
    };
    warnings: Record<string, number>;
  };
  performance: Performance;
  independence: { runs_with_results: number; fully_fresh_runs: number; replayed_answers: number };
}

export interface AlignedStep {
  signature: string;
  runs: number[];
  share: Rate;
  in_every_run: boolean;
  names: string[];
  positions: number[];
  required_in: number;
  one_to_one: boolean;
  start_seconds: { mean: number; min: number; max: number; range: number; stdev: number } | null;
  end_seconds: { mean: number; min: number; max: number; range: number; stdev: number } | null;
}

export interface LearningMetrics {
  counts: Counts;
  valid_runs: number;
  step_count_distribution: Record<string, number>;
  object_count_distribution: Record<string, number>;
  role_count_distribution: Record<string, number>;
  labels: { label: string; runs: number; share: Rate }[];
  labels_in_every_run: string[];
  mean_pairwise_label_overlap: number | null;
  aligned_steps: AlignedStep[];
  steps_in_every_run: number;
  unaligned_steps: { run: number; position: number; name: string; reason: string }[];
  reviewable: Rate;
  validation_errors: Rate;
  runs_with_ambiguities: Rate;
  performance: Performance;
  reference: { reference_signatures: string[]; per_run: { run: number; matched: number; extra: number; missing: string[] }[] } | null;
}

export interface GroundTruth {
  expected_verdict: "PASS" | "FAIL" | null;
  step_labels: Record<string, string>;
  note?: string;
  updated_at?: string;
}

export interface BatchBrief {
  id: string;
  name: string;
  mode: Mode;
  status: BatchStatus;
  project_id: string;
  project_exists: boolean;
  procedure_name: string;
  template_id: string;
  template_version: number | null;
  video: { id: string; filename?: string; duration_sec?: number; checksum_sha256?: string; role?: string };
  repetitions: number;
  max_concurrency: number;
  independent_inference: boolean;
  ground_truth: GroundTruth;
  fingerprint: string;
  git_revision: string;
  model_id: string;
  segmentation_model_id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  wall_clock_seconds: number | null;
  counts: Counts;
  summary?: Record<string, unknown>;
}

export interface Batch extends BatchBrief {
  steps: { step_id: string; name: string; required: boolean }[];
  config: Record<string, unknown> & { settings?: Record<string, unknown>; inference_policy?: string; shared_deterministic?: string[]; plan?: Record<string, unknown> };
  runs: LabRun[];
  metrics: VerificationMetrics | LearningMetrics;
}

export interface LabOptions {
  procedures: {
    id: string;
    name: string;
    recordings: { video_id: string; filename: string; role: "demonstration" | "execution"; duration_sec: number; ready: boolean }[];
    templates: { id: string; version: number; name: string; family_id: string; acceptance_status: string; created_at: string; steps: { step_id: string; name: string; required: boolean }[] }[];
  }[];
  models: { default_inference: string; default_segmentation: string };
  live_enabled: boolean;
  max_concurrency: number;
  max_repetitions: number;
}

export interface NewBatch {
  mode: Mode;
  project_id: string;
  video_id: string;
  template_id?: string;
  reference_template_id?: string;
  repetitions: number;
  max_concurrency: number;
  independent_inference: boolean;
  allow_live: boolean;
  ground_truth?: GroundTruth;
  name?: string;
}

export const lab = {
  options: () => request<LabOptions>("/api/reliability/options"),
  batches: () => request<BatchBrief[]>("/api/reliability/batches"),
  batch: (id: string) => request<Batch>(`/api/reliability/batches/${id}`),
  create: (body: NewBatch) => request<Batch>("/api/reliability/batches", { method: "POST", body: JSON.stringify(body) }),
  cancel: (id: string) => request<Batch>(`/api/reliability/batches/${id}/cancel`, { method: "POST" }),
  cancelRun: (id: string, runId: string) => request<Batch>(`/api/reliability/batches/${id}/runs/${runId}/cancel`, { method: "POST" }),
  groundTruth: (id: string, body: GroundTruth) => request<Batch>(`/api/reliability/batches/${id}/ground-truth`, { method: "PUT", body: JSON.stringify(body) }),
};

export const isActive = (status: BatchStatus) => status === "queued" || status === "running" || status === "canceling";

/** "80%" from a rate, or "N/A" when nothing was there to divide. */
export function percent(item: Rate | null | undefined, digits = 0): string {
  if (!item || item.value == null) return "N/A";
  return `${(item.value * 100).toFixed(digits)}%`;
}

export function fraction(item: Rate | null | undefined): string {
  if (!item) return "";
  return `${item.numerator}/${item.denominator}`;
}

export function seconds(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 90) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value - minutes * 60)}s`;
}

/** One letter per step verdict for the dense matrix. */
export const stepMark: Record<string, string> = { PASS: "P", FAIL: "F", CANNOT_VERIFY: "?", NOT_APPLICABLE: "–" };

/** The run's overall verdict, or why there is none. Execution failures are never shown as a verdict. */
export function runLabel(run: LabRun): string {
  if (run.status === "queued") return "QUEUED";
  if (run.status === "running") return run.cancel_requested ? "CANCELING" : "RUNNING";
  if (run.status === "canceled") return "CANCELED";
  if (run.status === "failed") return run.error_kind === "invalid_result" ? "NO VERDICT" : "ERROR";
  return run.result.verdict ?? "NO VERDICT";
}

export function runTone(run: LabRun): string {
  if (run.status === "completed" && run.result.verdict === "PASS") return "pass";
  if (run.status === "completed" && run.result.verdict === "FAIL") return "fail";
  if (run.status === "completed" && run.result.verdict === "INCONCLUSIVE") return "unknown";
  if (run.status === "failed") return "error";
  return "muted";
}

/** The most common value among a step's verdicts across runs, used to mark the runs that differ from it. */
export function modalVerdicts(runs: LabRun[], stepIds: string[]): Record<string, string | null> {
  const out: Record<string, string | null> = {};
  for (const stepId of stepIds) {
    const tally = new Map<string, number>();
    for (const run of runs) {
      const verdict = run.result.steps?.find((item) => item.step_id === stepId)?.verdict;
      if (run.status === "completed" && verdict) tally.set(verdict, (tally.get(verdict) ?? 0) + 1);
    }
    let best: string | null = null;
    let count = 0;
    let tied = false;
    for (const [verdict, n] of tally) {
      if (n > count) [best, count, tied] = [verdict, n, false];
      else if (n === count) tied = true;
    }
    out[stepId] = tied ? null : best;
  }
  return out;
}

/** Where an experiment lives: under its procedure, beside Teach, Procedure and Verify. */
export function labPath(procedureId: string, batchId?: string, runId?: string): string {
  return `/procedures/${procedureId}/lab${batchId ? `/${batchId}` : ""}${runId ? `/runs/${runId}` : ""}`;
}
