import { humanize } from "./humanize";
import type { BBox, Bundle } from "./types";

export type RecoveryOutcome = "supported" | "provisional" | "conflicting" | "rejected";
export type RecoveryLayer = "vlm" | "tracker" | "sam" | "recovered" | "reacquired" | "raw" | "canonical";

export interface RecoverySeed {
  seed_id: string;
  object_local_id: string;
  instance_id: string | null;
  timestamp: number;
  bbox: BBox | null;
  points: SeedPoint[];
  trust: "human_verified" | "independently_supported" | "candidate" | "rejected";
  source: string;
  source_ref: string;
  explanation: string;
  checks: { check: string; passed: boolean; detail: string }[];
  manual: boolean;
}

export interface SeedPoint {
  x: number;
  y: number;
  negative?: boolean;
}

export interface RecoveryTarget {
  target_id: string;
  object_local_id: string;
  instance_id: string | null;
  label: string;
  time_start: number;
  time_end: number;
  source: string;
  state: string;
  reasons: string[];
  details: string[];
  seeds?: RecoverySeed[];
  difficulty?: string;
  recoverable?: boolean;
  blocker?: string;
  dismissed?: boolean;
  direction?: string;
  support?: { observations_in_interval: number; measured_in_interval: number; usable_seeds: number };
}

export interface RecoveryCandidates {
  run_id: string;
  revision_id: string;
  targets: RecoveryTarget[];
  seeds: RecoverySeed[];
  seed_counts: Record<string, number>;
  objects: { local_id: string; label: string; instance_id: string | null }[];
  price_per_frame_usd: number | null;
}

export interface PlannedRequest {
  key: string;
  prompt: string;
  direction: string;
  frame_times: number[];
  source_start: number;
  source_end: number;
  billed_frames: number;
  cached: boolean;
  estimated_cost_usd: number | null;
  width: number;
  height: number;
  objects: { target_id: string; object_local_id: string; sam_object_id: number; seed: RecoverySeed; seed_frame_index: number; prompt_role?: string }[];
}

export interface RecoveryPlan {
  plan_hash: string;
  requests: PlannedRequest[];
  blocked: { target_id: string; object_local_id: string; time_start: number; time_end: number; reason: string }[];
  totals: {
    requests: number;
    remote_requests: number;
    cached_requests: number;
    frames: number;
    billed_frames: number;
    estimated_cost_usd: number | null;
    cost_note: string;
  };
  notes: string[];
  violations: string[];
}

export interface ProposalRow {
  timestamp: number;
  frame_index: number;
  bbox: BBox;
  confidence: number | null;
  outcome: RecoveryOutcome;
  association: string;
  partial: boolean;
  context: boolean;
  direction: string;
  seed_trust: string;
  checks: { check: string; passed: boolean; detail: string }[];
  lifecycle?: string;
  segment_id?: string;
  reacquired?: boolean;
}

export interface RecoveryProposal {
  id: string;
  generation: number;
  current: boolean;
  target_id: string;
  object_local_id: string;
  time_start: number;
  time_end: number;
  decision: "pending" | "accepted" | "rejected";
  rows: ProposalRow[];
  applied_revision_id: string | null;
  metrics: {
    counts: Record<RecoveryOutcome | "unobserved_frames", number> & { no_detection_frames?: number; unusable_frames?: number };
    previously_missing_frames: number;
    new_supported_frames: number;
    original_rows_in_interval: number;
    remaining_gaps: { start: number; end: number; frames: number }[];
    identity: { sam_object_id: number; association: string; note: string; seed_trust: string }[];
    usage: { requests: number; billed_frames: number; latency_ms: number; input_tokens: number; output_tokens: number };
    synthetic_response: boolean;
    human_box_overlap: { frames: number; mean_iou: number } | null;
    accuracy_note: string;
  };
}

export interface RecoveryExperiment {
  id: string;
  run_id: string;
  status: string;
  mode: string;
  generation: number;
  created_at: string;
  pending_proposals: number;
  targets: RecoveryTarget[];
  limits: Record<string, number | boolean>;
  plan: RecoveryPlan | null;
  error: string;
  remote_submitted: boolean;
  recorded_cost_usd: number | null;
  applied_revision_id: string | null;
  cancel_requested: boolean;
  requests: {
    cache_key: string;
    status: string;
    submission: string;
    source: string;
    billed_frames: number;
    latency_ms: number;
    error: string;
  }[];
  proposals: RecoveryProposal[];
}

export interface LayerBox {
  key: string;
  layer: RecoveryLayer;
  objectId: string;
  /** Keeps tracks apart that share an object, so each draws its own box. */
  group?: string;
  bbox: BBox;
  timestamp: number;
  /** Tells apart a directly observed box from an estimate or a conflict. */
  dashed: boolean;
  details: string[];
}

export const LAYERS: { key: RecoveryLayer; label: string; color: string }[] = [
  { key: "vlm", label: "Original VLM", color: "#7EB6FF" },
  { key: "tracker", label: "Tracker", color: "#7dcea0" },
  { key: "sam", label: "Original SAM", color: "#d2a24c" },
  { key: "recovered", label: "Recovered SAM", color: "#e879f9" },
  { key: "reacquired", label: "Reacquired SAM", color: "#34d399" },
  { key: "raw", label: "All SAM output", color: "#22d3ee" },
  { key: "canonical", label: "Final canonical", color: "#f5f5f5" },
];

export const LANE_COLORS: Record<string, string> = {
  stored: "#465066",
  problem: "#b7791f",
  supported: "#e879f9",
  provisional: "#c4a5d8",
  conflicting: "#e07070",
  rejected: "#888888",
};

export const OUTCOME_COLORS: Record<RecoveryOutcome, string> = {
  supported: "#e879f9",
  provisional: "#c4a5d8",
  conflicting: "#e07070",
  rejected: "#888888",
};

/** The same window the Inspect overlay uses, so a box stays up between its stored times. */
const NEAR_SEC = 0.45;

type Row = {
  object_local_id: string;
  timestamp: number;
  bbox: BBox | null;
  source_model?: string;
  observation_role?: string;
  geometry_origin?: string;
  geometry_authority?: string;
  association_status?: string;
  track_ref?: string;
};

function rowLayer(row: Row): RecoveryLayer {
  if (row.source_model === "sam-3.1") return "sam";
  if (row.source_model === "tracker") return "tracker";
  return "vlm";
}

function rowDetails(row: Row): string[] {
  return [
    `source ${row.source_model || row.geometry_authority || "model"}`,
    row.observation_role ? `role ${row.observation_role}` : "",
    row.geometry_origin ? `origin ${row.geometry_origin}` : "",
    row.association_status ? `association ${row.association_status}` : "",
    row.track_ref ? `track ${row.track_ref}` : "",
  ].filter(Boolean);
}

/** The nearest box per object per layer. A row with no nearby time draws nothing rather than holding its box. */
function nearest(boxes: LayerBox[], time: number): LayerBox[] {
  const best = new Map<string, LayerBox>();
  for (const box of boxes) {
    const gap = Math.abs(box.timestamp - time);
    if (gap > NEAR_SEC) continue;
    const key = `${box.layer}:${box.group ?? box.objectId}`;
    const current = best.get(key);
    if (!current || gap < Math.abs(current.timestamp - time)) best.set(key, box);
  }
  return [...best.values()];
}

/** Boxes for the chosen layers near one time. Original rows stay on their own layer, canonical or not. */
export function layerBoxes(
  bundle: Pick<Bundle, "observations" | "raw_observations" | "tracks">,
  proposals: RecoveryProposal[],
  time: number,
  layers: Record<RecoveryLayer, boolean>,
  raw: RawSamBox[] = [],
): LayerBox[] {
  const stored: Row[] = [...bundle.observations, ...(bundle.raw_observations?.rows ?? [])];
  const boxes: LayerBox[] = [];
  stored.forEach((row, index) => {
    if (!row.bbox) return;
    const layer = rowLayer(row);
    const conflict = row.association_status === "conflict";
    if (layers[layer]) {
      boxes.push({
        key: `${layer}-${index}`,
        layer,
        objectId: row.object_local_id,
        bbox: row.bbox,
        timestamp: row.timestamp,
        dashed: layer === "vlm" || conflict,
        details: rowDetails(row),
      });
    }
    if (layers.canonical && row.observation_role === "canonical") {
      boxes.push({
        key: `canonical-${index}`,
        layer: "canonical",
        objectId: row.object_local_id,
        bbox: row.bbox,
        timestamp: row.timestamp,
        dashed: false,
        details: rowDetails(row),
      });
    }
  });
  if (layers.tracker) {
    (bundle.tracks?.points ?? []).forEach((point, index) => {
      if (!point.bbox || point.detector_id === "sam-3.1") return;
      boxes.push({
        key: `track-${index}`,
        layer: "tracker",
        objectId: point.track_ref,
        bbox: point.bbox,
        timestamp: point.timestamp,
        dashed: point.detector_id === "optical_flow",
        details: [`track ${point.track_ref}`, `detector ${point.detector_id || "unknown"}`, point.status ? `status ${point.status}` : ""].filter(Boolean),
      });
    });
  }
  if (layers.recovered || layers.reacquired) {
    for (const proposal of proposals) {
      proposal.rows.forEach((row, index) => {
        const layer: RecoveryLayer = row.reacquired ? "reacquired" : "recovered";
        if (!layers[layer]) return;
        boxes.push({
          key: `${layer}-${proposal.id}-${index}`,
          layer,
          objectId: proposal.object_local_id,
          bbox: row.bbox,
          timestamp: row.timestamp,
          dashed: row.outcome !== "supported",
          details: [
            `outcome ${row.outcome}`,
            `association ${row.association}`,
            `seed ${row.seed_trust}`,
            `direction ${row.direction}`,
            row.lifecycle ? `track state ${humanize(row.lifecycle)}` : "",
            row.segment_id ? `segment ${row.segment_id}` : "not in a segment",
            row.partial ? "partial, visible area only" : "",
            row.confidence != null ? `SAM score ${row.confidence.toFixed(2)}, uncalibrated` : "",
            row.context ? "context frame, outside the interval, never applied" : "",
          ].filter(Boolean),
        });
      });
    }
  }
  if (layers.raw) {
    raw.forEach((box, index) => {
      if (!box.bbox || box.timestamp == null) return;
      boxes.push({
        key: `raw-${index}`,
        layer: "raw",
        objectId: box.object_local_id || `SAM track ${box.raw_track_id}`,
        group: `${box.request_key}:${box.raw_track_id}`,
        bbox: box.bbox,
        timestamp: box.timestamp,
        dashed: box.status !== "associated",
        details: [
          `SAM track ${box.raw_track_id}`,
          `status ${box.status}`,
          box.lifecycle ? `validated as ${box.outcome}, track state ${humanize(box.lifecycle)}` : "not validated",
          box.confidence != null ? `SAM score ${box.confidence.toFixed(2)}, uncalibrated` : "",
        ].filter(Boolean),
      });
    });
  }
  return nearest(boxes, time);
}

/** What an earlier run covered, in words a person can recognize it by. */
export function runCoverage(targets: RecoveryTarget[]): string {
  if (!targets.length) return "No targets";
  const objects = [...new Set(targets.map((item) => humanize(item.object_local_id)))].sort();
  const start = Math.min(...targets.map((item) => item.time_start));
  const end = Math.max(...targets.map((item) => item.time_end));
  return `${objects.join(", ")} · ${start.toFixed(1)}–${end.toFixed(1)}s`;
}

/** One box or set of points queued for the next SAM request, and the stretch it should track. */
export interface SeedDraft {
  id: string;
  objectId: string;
  timestamp: number;
  bbox: BBox | null;
  points: SeedPoint[];
  start: number;
  end: number;
  direction: "forward" | "backward" | "both" | "auto";
  origin: string;
  /** A stored seed chosen by id keeps its own trust level. A person's confirmation is recorded when it is needed. */
  seedId?: string;
  confirmSeed?: boolean;
  reacquires?: ReacquisitionLink;
}

export type TrackDirection = "forward" | "backward" | "both";

export function trackRange(time: number, seconds: number, direction: TrackDirection): { start: number; end: number } {
  return {
    start: direction === "forward" ? time : Math.max(time - seconds, 0),
    end: direction === "backward" ? time : time + seconds,
  };
}

/** A new seed for an object replaces its seed on the same frame. Each frame can hold one per object. */
export function placeSeed(drafts: SeedDraft[], next: Omit<SeedDraft, "id">): SeedDraft[] {
  const kept = drafts.filter((item) => !(item.objectId === next.objectId && Math.abs(item.timestamp - next.timestamp) <= 0.05));
  const draft = { ...next, id: `${next.objectId}@${next.timestamp.toFixed(3)}` };
  return [...kept, draft].sort((a, b) => a.timestamp - b.timestamp);
}

export function manualTargets(drafts: SeedDraft[]): Record<string, unknown>[] {
  return drafts.map((draft) => ({
    object_local_id: draft.objectId,
    time_start: draft.start,
    time_end: draft.end,
    direction: draft.direction,
    ...(draft.reacquires ? { reacquires: draft.reacquires } : {}),
    ...(draft.seedId
      ? { seed_id: draft.seedId, confirm_seed: Boolean(draft.confirmSeed) }
      : { seed: { timestamp: draft.timestamp, bbox: draft.bbox, points: draft.points } }),
  }));
}

/** The smallest box of a known object under a click, so a small part inside a larger box can be picked. */
export function boxAt(boxes: LayerBox[], objectIds: Set<string>, x: number, y: number): LayerBox | null {
  const hits = boxes.filter((box) =>
    objectIds.has(box.objectId) && x >= box.bbox.x && x <= box.bbox.x + box.bbox.w && y >= box.bbox.y && y <= box.bbox.y + box.bbox.h,
  );
  return hits.sort((a, b) => a.bbox.w * a.bbox.h - b.bbox.w * b.bbox.h)[0] ?? null;
}

export interface LaneSpan {
  start: number;
  end: number;
  kind: string;
}

/** Times joined into spans. Times further apart than `gap` stay separate, so holes remain visible. */
export function spansOf(times: number[], kind: string, gap = 1.5): LaneSpan[] {
  const spans: LaneSpan[] = [];
  for (const time of [...times].sort((a, b) => a - b)) {
    const last = spans[spans.length - 1];
    if (last && time - last.end <= gap) last.end = time;
    else spans.push({ start: time, end: time, kind });
  }
  return spans;
}

/** Everything one object's timeline lane draws. */
export function laneSpans(
  bundle: Pick<Bundle, "observations">,
  objectId: string,
  targets: RecoveryTarget[],
  proposals: RecoveryProposal[],
): LaneSpan[] {
  const stored = bundle.observations.filter((row) => row.object_local_id === objectId && row.bbox).map((row) => row.timestamp);
  const results: LaneSpan[] = [];
  for (const proposal of proposals.filter((item) => item.object_local_id === objectId)) {
    for (const outcome of ["supported", "provisional", "conflicting", "rejected"] as const) {
      const times = proposal.rows.filter((row) => !row.context && row.outcome === outcome).map((row) => row.timestamp);
      results.push(...spansOf(times, outcome, 0.2));
    }
  }
  return [
    ...spansOf(stored, "stored"),
    ...targets.filter((item) => item.object_local_id === objectId).map((item) => ({ start: item.time_start, end: item.time_end, kind: "problem" })),
    ...results,
  ];
}

export interface RawSamBox {
  request_key: string;
  raw_track_id: string;
  object_local_id: string;
  frame_index: number;
  timestamp: number | null;
  bbox: BBox | null;
  pixels: number[];
  confidence: number | null;
  has_mask: boolean;
  status: string;
  outcome?: string;
  lifecycle?: string;
  segment_id?: string;
}

export interface TrackSegment {
  segment_id: string;
  physical_instance_id: string;
  object_local_id: string;
  experiment_id: string;
  session_id: string;
  raw_track_id: string;
  direction: string;
  seed: { seed_id: string; trust: string; source: string; timestamp: number };
  init_event: string;
  start: number;
  end: number;
  last_supported_time: number | null;
  status: string;
  identity: { status: string; basis: string[]; detail: string };
  counts: Record<RecoveryOutcome, number>;
  frames: number;
  decision?: string;
}

export interface TrackFailure {
  failure_id: string;
  experiment_id: string;
  time: number;
  last_good_time: number | null;
  last_supported_time: number | null;
  direction: string;
  reason: string;
  detail: string;
  state: string;
  state_basis: string;
}

export interface ReacquisitionEvidence {
  source: string;
  trust: string;
  seed_id: string;
  timestamp: number;
  bbox: BBox | null;
  explanation: string;
  needs_confirmation: boolean;
  may_be: string[];
}

export interface ReacquisitionStrategy {
  strategy: string;
  available: boolean;
  requires_remote: boolean;
  evidence: ReacquisitionEvidence[];
  note: string;
  concept?: string;
  time_start?: number;
  time_end?: number;
}

export interface ReacquisitionLink {
  failure_id: string;
  failure_time: number;
  experiment_id: string;
  strategy: string;
}

export interface ReappearanceCandidate {
  candidate_id: string;
  source: string;
  timestamp: number;
  bbox: BBox;
  raw_track_id: string;
  track_frames: number;
  outcome: "verified_reappearance" | "probable_reappearance" | "ambiguous_candidate" | "rejected_candidate";
  reason: string;
  needs_confirmation: boolean;
  appearance?: { encoder: string; score: number } | null;
  checks: { check: string; passed: boolean | null; detail: string }[];
}

export interface ReappearanceSearch {
  window: { start: number; end: number; limited: boolean; note: string };
  frames_examined: number;
  evidence_inspected: { stored_observations: number; stored_sam_requests: unknown[]; stored_sam_boxes: number; pipeline_sam_points: number };
  candidates: ReappearanceCandidate[];
  result: string;
}

export interface FailureDiagnosis {
  categories: { code: string; detail: string }[];
  last_raw_time: number | null;
  last_supported_time: number | null;
  first_no_detection_time: number | null;
  raw_boxes_after_failure: number;
  later_track_ids: string[];
  rejected_by_validation: number;
  detected_not_identified: number;
  analyzed_through: number;
  video_end: number;
  clip_ended_early: boolean;
  request_planned: boolean;
  request_executed: boolean;
}

export const CANDIDATE_COLORS: Record<string, string> = {
  verified_reappearance: "#34d399",
  probable_reappearance: "#fbbf24",
  ambiguous_candidate: "#fb923c",
  rejected_candidate: "#6b7280",
};

export const WORKFLOW_LABELS: Record<string, string> = {
  track_lost: "Track lost",
  cached_search: "Stored evidence searched",
  identity: "Identity",
  concept_detection: "Concept search",
  reinitialization: "New spatial track",
  acceptance: "Your decision",
};

export const PROMPT_ROLE_LABELS: Record<string, string> = {
  initial_seed: "Initial seed",
  additional_prompt: "Additional same-object prompt",
  reappearance_seed: "Reappearance seed",
  manual_seed: "Manually confirmed seed",
};

export interface ReacquisitionPlanItem {
  search: ReappearanceSearch;
  diagnosis: FailureDiagnosis;
  workflow: { stage: string; status: string; detail: string }[];
  failure: TrackFailure;
  status: "open" | "resolved";
  uncertain_interval: { start: number; end: number };
  last_supported_time: number | null;
  reappearance_frames: number[];
  strategies: ReacquisitionStrategy[];
  recommended: string;
  blocked_reason: string;
  needs_stronger_evidence: boolean;
  attempts: { experiment_id: string; status: string; succeeded: boolean }[];
  explanation: string;
}

export interface ObjectTrackHistory {
  object_local_id: string;
  physical_instance_id: string;
  label: string;
  segments: TrackSegment[];
  failures: TrackFailure[];
  unresolved: { start: number; end: number; frames: number; no_detection_frames: number; unusable_frames: number }[];
  plans: ReacquisitionPlanItem[];
  stats: Record<string, number | null>;
  evidence_limits: string[];
  coverage?: { analyzed_through: number; video_end: number };
}

export interface ReacquisitionData {
  run_id: string;
  objects: ObjectTrackHistory[];
  note: string;
}

export const SEGMENT_COLORS: Record<string, string> = {
  tracked: "#e879f9",
  degraded: "#c4a5d8",
  lost: "#a855a8",
  reacquired: "#34d399",
  reacquisition_candidate: "#7dd3b0",
  ambiguous: "#d2a24c",
};

/** How a stated failure looks. Occluded and out of view are hypotheses from geometry. */
export const GAP_STATE_LABELS: Record<string, string> = {
  lost: "Lost, cause unknown",
  occluded: "Possibly hidden",
  out_of_view: "Possibly out of view",
};

const REASON_LABELS: Record<string, string> = {
  large_spatial_jump: "Large jump",
  absorbed_neighbor: "Grew over another part",
  expanded_beyond_object: "Grew past the part",
  anchor_contradiction: "Measurements disagree",
  seed_mismatch: "Did not start on the seed",
  scene_cut: "Moved across a hard cut",
  no_detection: "SAM stopped returning it",
};

export function failureLabel(reason: string): string {
  return REASON_LABELS[reason] ?? humanize(reason);
}

/** A seed drawn or chosen for this failure. The API records it as a separate attempt. */
export function reacquisitionLink(item: ReacquisitionPlanItem, strategy: string): ReacquisitionLink {
  return {
    failure_id: item.failure.failure_id,
    failure_time: item.failure.time,
    experiment_id: item.failure.experiment_id,
    strategy,
  };
}

/** The stretch a new seed should cover: from the failure, through the seed, plus a little after it. */
export function reacquisitionRange(item: ReacquisitionPlanItem, seedTime: number): { start: number; end: number } {
  return {
    start: Math.min(item.uncertain_interval.start, seedTime),
    end: Math.max(item.uncertain_interval.end, seedTime + 1),
  };
}

export interface SamTrackDiagnostics {
  raw_track_id: string;
  prompted: boolean;
  object_local_id: string;
  association: string;
  association_note: string;
  boxes: number;
  first_time: number | null;
  last_time: number | null;
  first_confidence: number | null;
  mean_confidence: number | null;
  last_confidence: number | null;
  broke: { timestamp: number; detail: string } | null;
}

export interface SamRequestDiagnostics {
  key: string;
  verdict: string;
  message: string;
  boxes_returned: boolean;
  synthetic: boolean;
  request: { status: string; submission: string; source: string; provider_request_id: string; latency_ms: number; error: string };
  prompt: string;
  direction: string;
  planned_frames: number;
  reachable_frames: number;
  frames_with_boxes: number;
  parsed: {
    text_chars: number;
    frame_lines: number;
    empty_frame_lines: number;
    box_tokens: number;
    dropped_tokens: number;
    unparsed_lines: number;
    unparsed_samples: string[];
  };
  funnel: Record<"box_tokens" | "parsed" | "usable" | "unassigned" | "assigned" | "supported" | "provisional" | "conflicting" | "rejected", number>;
  tracks: SamTrackDiagnostics[];
  missing_frames: { start: number; end: number; frames: number }[];
  failed_checks: Record<string, number>;
  findings: { code: string; severity: string; message: string }[];
  boxes: RawSamBox[];
}

export interface RecoveryDiagnostics {
  experiment_id: string;
  generation: number;
  note: string;
  requests: SamRequestDiagnostics[];
}

/** The moments worth stopping on: sampled frames, stored boxes, and recovered boxes inside their interval. */
export function stepTimes(bundle: Pick<Bundle, "frames" | "observations">, proposals: RecoveryProposal[], raw: RawSamBox[] = []): number[] {
  const times = [
    ...bundle.frames.map((frame) => frame.timestamp),
    ...bundle.observations.filter((row) => row.bbox).map((row) => row.timestamp),
    ...proposals.flatMap((proposal) => proposal.rows.filter((row) => !row.context).map((row) => row.timestamp)),
    ...raw.flatMap((box) => (box.bbox && box.timestamp != null ? [box.timestamp] : [])),
  ].sort((a, b) => a - b);
  return times.filter((time, index) => index === 0 || time - times[index - 1] > 0.001);
}

export function nextStep(times: number[], now: number, direction: 1 | -1): number | null {
  if (direction > 0) return times.find((time) => time > now + 0.001) ?? null;
  for (let index = times.length - 1; index >= 0; index -= 1) {
    if (times[index] < now - 0.001) return times[index];
  }
  return null;
}

/** A press and a release on the picture, in normalized coordinates of the displayed frame. */
export function dragBox(startX: number, startY: number, endX: number, endY: number): BBox | null {
  const clamp = (value: number) => Math.min(Math.max(value, 0), 1);
  const x1 = clamp(Math.min(startX, endX));
  const y1 = clamp(Math.min(startY, endY));
  const x2 = clamp(Math.max(startX, endX));
  const y2 = clamp(Math.max(startY, endY));
  if (x2 - x1 < 0.005 || y2 - y1 < 0.005) return null;
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/** The live button is offered only for a plan with something to send and no limit broken. */
export function canApproveLive(plan: RecoveryPlan | null): boolean {
  return Boolean(plan && plan.totals.remote_requests > 0 && plan.violations.length === 0);
}

export function usage(totals: RecoveryPlan["totals"]): string {
  const cost = totals.estimated_cost_usd == null ? "cost unknown" : `about $${totals.estimated_cost_usd.toFixed(4)}`;
  return `${totals.remote_requests} provider ${totals.remote_requests === 1 ? "request" : "requests"}, about ${totals.billed_frames} billed frames, ${cost}`;
}
