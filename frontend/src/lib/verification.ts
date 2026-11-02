/** Display helpers for execution verification. Verdicts stay distinct from attempt state. */

export interface VerificationEvidence {
  video_id: string;
  timestamp: number | null;
  observation: string;
  evidence_class: string;
  side: string;
  bbox: { x: number; y: number; w: number; h: number } | null;
  object_local_ids: string[];
}

export interface VerificationStep {
  step_id: string;
  name: string;
  required: boolean;
  requirement_status: string;
  verdict: string;
  execution_state: string;
  error_category: string;
  expected: string;
  observed: string;
  explanation: string;
  uncertainty_notes: string;
  review_required: boolean;
  matched_action_ids: string[];
  matched_object_ids: string[];
  evidence: VerificationEvidence[];
  demo_evidence: VerificationEvidence[];
  history: { verdict: string; error_category?: string; note: string }[];
  corrected: boolean;
  demo_time_start: number | null;
  demo_time_end: number | null;
  execution_time: number | null;
}

export interface TemplateStepView {
  step_id: string;
  name: string;
  required: boolean;
  requirement_status: string;
  object_roles: string[];
  expected_action: string;
  uncertainty_notes: string;
  completion_conditions: {
    type: string;
    subject_role: string;
    target_role: string;
    relation: string;
    state_label: string;
    evidence_requirement: string;
  }[];
  dependencies: { step_id: string; category: string; constrains_order: boolean }[];
}

export interface WorkflowTemplate {
  id: string;
  family_id: string;
  version: number;
  name: string;
  acceptance_status: string;
  source_run_id: string;
  source_video_id: string;
  document: {
    notes: string[];
    roles: { role_id: string; label: string; cardinality: string; uncertain: boolean }[];
    steps: TemplateStepView[];
  };
}

export interface VerificationView {
  id: string;
  template_id: string;
  template_version: number;
  execution_video_id: string;
  execution_run_id: string;
  status: string;
  overall: string;
  error: string;
  config: {
    strategy: string;
    provider_id: string;
    model_id: string;
    perception?: { strategy?: string } | null;
    reinspect: boolean;
  };
  report?: {
    overall: string;
    timing_policy: string;
    demonstration_video_id?: string;
    score: {
      verified: number;
      failed: number;
      unresolved: number;
      not_applicable: number;
      value: number | null;
      explanation: string;
    };
    warnings: string[];
    roles: {
      role_id: string;
      label: string;
      status: string;
      object_local_ids: string[];
      hypotheses: { object_local_id: string; score: number }[];
      note: string;
    }[];
    steps: VerificationStep[];
    unexpected_actions: { action_id: string; action_type: string; start: number; failure: boolean; note: string }[];
  };
}

export function verdictClass(verdict: string): string {
  if (verdict === "PASS") return "observed";
  if (verdict === "FAIL") return "unknown";
  if (verdict === "CANNOT_VERIFY" || verdict === "INCONCLUSIVE") return "inferred";
  return "";
}

export function markerKind(step: { verdict: string; execution_state?: string; error_category?: string }): string {
  if (step.execution_state === "corrected" || step.error_category === "TEMPORARY_ERROR_CORRECTED") return "corrected";
  if (step.verdict === "FAIL") return "failure";
  if (step.verdict === "CANNOT_VERIFY") return "uncertain";
  if (step.verdict === "PASS") return "verified";
  return "step";
}

export interface VerificationRecording {
  id: string;
  filename: string;
  run_id: string;
  duration_sec: number;
  playback_url: string;
}

/** Pair each video with its newest completed run. Pass runs newest first; the first match wins. */
export function recordingsForVerification(
  videos: { id: string; filename: string; duration_sec: number; playback_url: string }[],
  runs: { id: string; video_id: string }[],
): VerificationRecording[] {
  const latest = new Map<string, string>();
  for (const run of runs) {
    if (!latest.has(run.video_id)) latest.set(run.video_id, run.id);
  }
  return videos.map((video) => ({
    id: video.id,
    filename: video.filename,
    run_id: latest.get(video.id) ?? "",
    duration_sec: video.duration_sec,
    playback_url: video.playback_url,
  }));
}

export function markerColor(kind: string): string {
  if (kind === "verified") return "#3DDC97";
  if (kind === "failure") return "#F07178";
  if (kind === "uncertain") return "#F0C05A";
  if (kind === "corrected") return "#7EB6FF";
  if (kind === "unexpected") return "#C792EA";
  return "#8B93A7";
}
