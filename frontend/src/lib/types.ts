export type RunStatus = "pending" | "running" | "completed" | "failed" | "canceled";

export interface RunSummary {
  id: string;
  project_id: string | null;
  video_id: string;
  name: string;
  status: RunStatus;
  research_mode: string;
  viewpoint: string;
  task_constraint: string;
  provider_id: string;
  model_id: string;
  modality: string;
  sampling_preset: string;
  analysis_pass: string;
  segmentation_mode: string;
  fixture: boolean;
  fixture_label: string;
  starred: boolean;
  cost_usd: number;
  unpriced_calls: number;
  input_tokens: number;
  output_tokens: number;
  error: string;
  object_count: number;
  action_count: number;
  duration_sec: number | null;
  created_at: string;
  thumbnail_url: string;
  video_filename: string;
}

export interface VideoInfo {
  id: string;
  project_id: string | null;
  filename: string;
  checksum: string;
  duration_sec: number;
  width: number;
  height: number;
  fps: number;
  avg_fps: number;
  variable_frame_rate: boolean;
  codec: string;
  container: string;
  frame_count: number;
  thumbnail_url: string;
  playback_url: string;
  contact_sheet_url: string;
  frames?: FrameInfo[];
  capture_time_raw?: string;
  capture_time_utc?: string;
  timestamp_reliability?: string;
  timestamp_source?: string;
  file_mtime?: string;
  timestamp_note?: string;
  has_audio?: boolean;
  audio_probed?: boolean;
  audio_codec?: string;
  order_index?: number | null;
  ingest_status?: "processing" | "ready" | "failed" | "";
  ingest_progress?: number;
  ingest_detail?: string;
  ingest_error?: string;
}

export interface FrameInfo {
  id: string;
  timestamp: number;
  reason: string;
  url: string;
  motion_score?: number;
  scene_change?: boolean;
}

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface LabObject {
  id: string;
  local_id: string;
  label: string;
  description: string;
  visual_characteristics: string;
  initial_state: string;
  final_state: string;
  uncertain_identity: boolean;
  manipulated: boolean;
  confidence: number | null;
  color: string;
  observation_count: number;
  annotation_status: string;
  primary_image_url: string;
  images: { id: string; kind: string; timestamp: number; url: string; score: number }[];
}

export interface SamMaskOverlay {
  id: string;
  timestamp: number;
  bbox: BBox | null;
  track_ref: string;
  confidence: number | null;
  association_status: string;
  object_local_id: string;
  identity_uncertain: boolean;
  decode: string;
  url: string;
}

export interface TrackPointSample {
  track_ref: string;
  timestamp: number;
  bbox: BBox | null;
  status: string;
  continuity: string;
  association_score: number | null;
  detector_id: string;
  scene_chunk: number;
  note: string;
  object_local_id?: string;
  association_status?: string;
}

export interface PerceptionInstance {
  instance_id: string;
  semantic_label?: string | null;
  status: string;
  first_seen: number;
  last_seen: number;
  observation_count: number;
  trusted_references: number;
  candidate_references: number;
  uncertain: boolean;
  human_initialized: boolean;
  detector_id: string;
}

export interface PerceptionHypothesis {
  hypothesis_id: string;
  timestamp: number;
  subject_instance_id: string;
  candidate_instance_id: string;
  runner_up_instance_id: string | null;
  appearance_cosine: number | null;
  motion_agreement: number | null;
  association_score: number;
  margin: number | null;
  decision: string;
  note: string;
}

export interface PerceptionSummary {
  strategy: string;
  detector_id: string;
  encoder_id?: string | null;
  encoder_version?: string | null;
  encoder_status?: string;
  encoder_detail?: string;
  segmenter_id?: string | null;
  tracking_fps?: number | null;
  frames_tracked?: number;
  world_coordinates_supported?: boolean;
  coordinate_frame?: string;
  used_for_identity?: boolean;
  scores_are_probabilities?: boolean;
  diagnostics?: Record<string, unknown>;
  instances?: PerceptionInstance[];
  hypotheses?: PerceptionHypothesis[];
  appearance_references?: { instance_id: string; timestamp: number; state: string; quality: number; confirmed: boolean }[];
  manual_corrections?: Record<string, unknown>[];
  notes?: string;
}

export interface Observation {
  id: string;
  object_local_id: string;
  timestamp: number;
  bbox: BBox | null;
  visibility: string;
  description: string;
  evidence_class: string;
  provenance: string;
  evidence_note: string;
  continuity: string;
  confidence: number | null;
  coordinate_frame: string;
  camera_compensated: boolean;
  frame_id: string | null;
  track_ref?: string;
  association_status?: string;
  source_model?: string;
  request_id?: string;
  mask_ref?: string;
  observation_role?: string;
  geometry_origin?: string;
  geometry_authority?: string;
  detection_support?: string;
  association_support?: string;
  temporal_support?: string;
  semantic_support?: string;
  evidence_ref?: string;
}

export interface RawObservation {
  object_local_id: string;
  timestamp: number;
  bbox: BBox | null;
  visibility: string;
  source_model: string;
  track_ref: string;
  association_status: string;
  observation_role: string;
  geometry_origin: string;
  geometry_authority: string;
  detection_support: string;
  association_support: string;
  temporal_support: string;
  semantic_support: string;
  evidence_ref: string;
}

export interface LabAction {
  id: string;
  local_id: string;
  action_type: string;
  description: string;
  start: number;
  end: number;
  actor: string;
  object_local_ids: string[];
  initial_state: string;
  final_state: string;
  outcome: string;
  uncertainty: number;
  evidence_class: string;
  provenance: string;
  evidence_note: string;
  evidence_timestamps: number[];
}

export interface Step {
  id: string;
  local_id: string;
  number: number;
  name: string;
  required_objects: string[];
  preconditions: string[];
  expected_action: string;
  completion_condition: string;
  observed_result: string;
  action_local_ids: string[];
  time_start: number | null;
  time_end: number | null;
  uncertainty: number;
  evidence_class: string;
  provenance: string;
  parallel_group: string;
}

export interface Dependency {
  id: string;
  from_step: string;
  to_step: string;
  category: string;
  rationale: string;
  evidence_class: string;
  provenance: string;
}

export interface Ambiguity {
  id: string;
  local_id: string;
  kind: string;
  description: string;
  time_start: number | null;
  time_end: number | null;
  object_local_ids: string[];
  action_local_ids: string[];
  alternative: string;
  evidence_class: string;
  provenance: string;
}

export interface StageInfo {
  stage: string;
  status: string;
  progress_current: number;
  progress_total: number;
  attempt: number;
  error: string;
  checkpoint: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
}

export interface BoxDecision {
  step: string;
  reason_code: string;
  outcome: "dropped" | "reassigned" | "kept";
  object_local_id: string;
  timestamp: number | null;
  bbox: BBox | null;
  confidence: number | null;
  other_object_id: string | null;
  detail: string;
}

export interface Bundle {
  run: RunSummary;
  video: VideoInfo | null;
  revision: { id: string; layer: string; version: number; note: string } | null;
  coverage: {
    modality?: string;
    frames_extracted?: number;
    frames_submitted?: number;
    temporal_coverage_note?: string;
    native_video_submitted?: boolean;
    selection_reasons?: Record<string, number>;
    time_start?: number | null;
    time_end?: number | null;
    time_basis?: string;
    max_gap_sec?: number | null;
    uncovered_gaps?: [number, number][];
    motion_frames?: number;
    candidate_peaks?: number[];
    selected_peaks?: number[];
    budget_exclusions?: number;
    scene_boundaries?: { time: number; kind: string; reason: string }[];
    semantic_selection?: string;
    selection_swaps?: { removed: number; added: number; reason: string }[];
    semantic_mode?: string;
    semantic_calls?: number;
    tracking?: {
      frames?: number;
      max_gap_sec?: number;
      gaps_over_occlusion?: number;
      continuity?: string;
      peak_fps?: number;
      floor_fps?: number;
      processing_chunks?: number;
    };
    segmentation?: {
      chunks?: number;
      frames_submitted?: number;
      skipped_chunks?: { start: number; end: number; reason: string }[];
    };
    media_transforms?: Record<string, unknown>[];
  };
  config: Record<string, unknown>;
  stages: StageInfo[];
  frames: FrameInfo[];
  objects: LabObject[];
  observations: Observation[];
  raw_observations?: { total: number; truncated: boolean; rows: RawObservation[] };
  box_decisions?: {
    total: number;
    truncated: boolean;
    counts: { step: string; reason_code: string; outcome: string; count: number }[];
    rows: BoxDecision[];
  };
  states: {
    id: string;
    object_local_id: string;
    time_start: number;
    time_end: number;
    state_label: string;
    description: string;
    evidence_class: string;
    provenance: string;
  }[];
  transitions: {
    id: string;
    object_local_id: string;
    timestamp: number;
    from_state: string;
    to_state: string;
    description: string;
    evidence_class: string;
    certainty: number | null;
    provenance: string;
  }[];
  relationships: {
    id: string;
    subject_local_id: string;
    other_local_id: string | null;
    kind: string;
    time_start: number;
    time_end: number;
    description: string;
    evidence_class: string;
    provenance: string;
  }[];
  actions: LabAction[];
  steps: Step[];
  dependencies: Dependency[];
  ambiguities: Ambiguity[];
  validation: { severity: string; code: string; message: string; entity_id?: string }[];
  summary: string;
  perception?: PerceptionSummary | null;
  tracks?: { total: number; truncated: boolean; points: TrackPointSample[] };
  camera: {
    viewpoint?: string;
    task_constraint?: string;
    coordinate_frame?: string;
    world_coordinates_supported?: boolean;
    camera_motion_compensated?: boolean;
    compensation_model?: string;
    notes?: string;
  } | null;
  transcripts: TranscriptSegment[];
  alignments: AudioAlignment[];
}

export interface TranscriptSegment {
  id: string;
  video_id: string;
  local_id: string;
  start: number;
  end: number;
  text: string;
  text_model: string;
  language: string;
  language_probability: number | null;
  confidence: number | null;
  words: { start: number; end: number; word: string; probability?: number }[];
  model_name: string;
  device: string;
  corrected: boolean;
  provenance: string;
}

export interface AudioAlignment {
  id: string;
  video_id: string;
  segment_id: string;
  classification: string;
  local_classification: string;
  explanation: string;
  action_local_ids: string[];
  object_local_ids: string[];
  step_local_ids: string[];
  frame_timestamps: number[];
  time_offset_sec: number | null;
  confidence: number | null;
  evidence_class: string;
  provenance: string;
  corrected: boolean;
  speech_is_not_completion: boolean;
}

export interface AudioModelStatus {
  installed: boolean;
  detail: string;
  models: { model: string; downloaded: boolean; repo: string; default: boolean }[];
  acceleration: { device: string; compute_type: string; backend: string; note?: string };
  supported_models: string[];
  default_model: string;
  audio_bytes_leave_machine: boolean;
}

/** Identity fields shared by every local CLI transport. */
export interface CliLinkStatus {
  installed: boolean;
  version: string;
  logged_in: boolean;
  auth_method: string;
  subscription: string;
  account: string;
  error: string;
  login_started: boolean;
  login_url: string;
}

export interface CodexSettings {
  binary: string;
  transport: "api" | "codex";
}

export interface CodexStatus extends CodexSettings, CliLinkStatus {}

export interface ClaudeCodeSettings {
  binary: string;
  transport: "api" | "claude_code";
}

export interface ClaudeCodeStatus extends ClaudeCodeSettings, CliLinkStatus {}

export interface CursorCliSettings {
  binary: string;
  transport: "login" | "api_key";
}

export interface CursorCliStatus extends CursorCliSettings, CliLinkStatus {}

export interface ProviderInfo {
  id: string;
  local: boolean;
  configured: boolean;
  secret_source: string;
  hint: string;
  validated: boolean;
  last_error: string;
  base_url: string;
  models: Capability[];
  sam?: {
    model_id: string;
    provider_id: string;
    image_segmentation: boolean;
    video_tracking: boolean;
    box_prompts: boolean;
    point_prompts: boolean;
    video_tracking_detail: string;
  };
  codex?: CodexSettings;
  claude_code?: ClaudeCodeSettings;
  cursor_cli?: CursorCliSettings;
}

export interface Capability {
  model_id: string;
  provider_id: string;
  text: boolean;
  image: boolean;
  native_video: boolean;
  structured_output: boolean;
  max_images: number;
  supports_temperature: boolean;
  reasoning_efforts: string[];
  price_known: boolean;
  notes: string;
  label?: string;
}

export interface Dashboard {
  videos: number;
  runs_completed: number;
  runs_active: number;
  objects: number;
  actions: number;
  workflows: number;
  ambiguities: number;
  providers_connected: string[];
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  unpriced_calls: number;
  unitemized_usd: number;
  cost_by_provider: ProviderSpend[];
  recent_runs: RunSummary[];
}

export interface ProviderSpend {
  provider_id: string;
  cost_usd: number;
  calls: number;
  unpriced_calls: number;
  input_tokens: number;
  output_tokens: number;
}
