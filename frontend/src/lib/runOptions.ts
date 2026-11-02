import { formatUsd } from "./api";
import { humanize } from "./humanize";

/** Labels match the experiment form, so a finished run names the option that was chosen. */
export const SEGMENTATION_OPTIONS = [
  ["disabled", "Disabled"],
  ["boxes_only", "Boxes and crops only"],
  ["local_grabcut", "Local GrabCut"],
  ["sam", "SAM 3.1 image segmentation"],
  ["sam_video", "SAM 3.1 video tracking with evidence fusion"],
] as const;

export const PROMPT_STRATEGY_OPTIONS = [
  ["concept", "Concept: find each part by its name"],
  ["trusted_spatial", "Trusted spatial: follow verified boxes only"],
  ["hybrid", "Hybrid: verified boxes first, names for the rest"],
] as const;

export const PERCEPTION_OPTIONS = [
  ["A_legacy_motion", "A · Motion baseline"],
  ["B_temporal", "B · Temporal tracking"],
  ["C_appearance", "C · Appearance-assisted"],
  ["D_appearance_segmented", "D · Appearance and GrabCut"],
  ["vlm_only", "VLM only"],
] as const;

export const PIPELINE_OPTIONS = [
  ["single", "Single pass"],
  ["multi", "Multi-stage"],
  ["multi_reinspect", "Multi-stage + reinspection"],
] as const;

export const SEMANTIC_OPTIONS = [
  ["combined", "Combined: interpret in the scene call"],
  ["grounded", "Grounded: discover, measure, then interpret"],
] as const;

const STRATEGY_PRESET: Record<string, string> = {
  legacy_motion: "A_legacy_motion",
  temporal: "B_temporal",
  appearance: "C_appearance",
  appearance_segmented: "D_appearance_segmented",
  vlm_only: "vlm_only",
};

/** Names of the built-in families. A run-specific name, when stored, wins over these. */
const PROMPT_FAMILIES: Record<string, string> = {
  foundational: "Foundational evidence rules",
  scene: "General scene analysis and object identification",
  actions: "Temporal action recognition",
  states: "Object state extraction",
  workflow: "Workflow reconstruction",
  ambiguity: "Ambiguity analysis",
  reinspect: "Targeted interval reinspection",
  comparison: "Cross-run comparison",
};

type OptionList = readonly (readonly [string, string])[];

export function optionLabel(options: OptionList, value: string | null | undefined): string {
  if (!value) return "";
  return options.find(([id]) => id === value)?.[1] ?? humanize(value);
}

function perceptionPreset(config: Record<string, unknown> | null | undefined): string {
  const perception = config?.perception;
  if (!perception || typeof perception !== "object") return "";
  const record = perception as { preset?: unknown; strategy?: unknown };
  if (typeof record.preset === "string" && record.preset) return record.preset;
  if (typeof record.strategy === "string" && record.strategy) {
    return STRATEGY_PRESET[record.strategy] ?? record.strategy;
  }
  return "";
}

function promptFamily(config: Record<string, unknown> | null | undefined): string {
  if (!config) return "";
  const name = config.prompt_name;
  if (typeof name === "string" && name.trim()) return name.trim();
  const key = config.prompt_key;
  if (typeof key !== "string" || !key) return "";
  return PROMPT_FAMILIES[key] ?? humanize(key);
}

/** Model, sampling, and the experiment options recorded on the run. */
type CoverageSummary = {
  time_start?: number | null;
  time_end?: number | null;
  frames_extracted?: number;
  max_gap_sec?: number | null;
  uncovered_gaps?: [number, number][];
  semantic_mode?: string;
  semantic_calls?: number;
  tracking?: { frames?: number; max_gap_sec?: number; gaps_over_occlusion?: number; continuity?: string };
  segmentation?: { skipped_chunks?: unknown[] };
};

/** One line of measured coverage. Empty for older runs that did not record it. */
export function coverageLine(coverage: CoverageSummary | null | undefined): string {
  if (!coverage || coverage.time_end == null) return "";
  const start = Number(coverage.time_start ?? 0);
  const end = Number(coverage.time_end);
  const parts = [`Interval ${start.toFixed(2)}–${end.toFixed(2)} s`];
  if (coverage.frames_extracted) parts.push(`${coverage.frames_extracted} semantic frames`);
  if (coverage.max_gap_sec != null) parts.push(`largest semantic gap ${coverage.max_gap_sec.toFixed(2)} s`);
  const gaps = coverage.uncovered_gaps?.length ?? 0;
  if (gaps) parts.push(`${gaps} uncovered ${gaps === 1 ? "gap" : "gaps"}`);
  const tracking = coverage.tracking;
  if (tracking?.frames) {
    parts.push(`${tracking.frames} perception frames`);
    if (tracking.max_gap_sec != null) parts.push(`largest tracking gap ${tracking.max_gap_sec.toFixed(2)} s`);
    if (tracking.continuity && tracking.continuity !== "supported") parts.push(`tracking continuity ${tracking.continuity}`);
  }
  const skipped = coverage.segmentation?.skipped_chunks?.length ?? 0;
  if (skipped) parts.push(`${skipped} SAM ${skipped === 1 ? "chunk" : "chunks"} skipped`);
  if (coverage.semantic_mode) {
    const count = coverage.semantic_calls ?? 0;
    const calls = count ? `, ${count} semantic ${count === 1 ? "call" : "calls"}` : "";
    parts.push(`${coverage.semantic_mode} mode${calls}`);
  }
  return parts.join(" · ");
}

export function analysisSetupLine(bundle: {
  run: {
    provider_id: string;
    model_id: string;
    modality: string;
    sampling_preset: string;
    segmentation_mode?: string;
    analysis_pass?: string;
    cost_usd: number;
    unpriced_calls?: number;
  };
  revision?: { layer: string; version: number } | null;
  config?: Record<string, unknown> | null;
}): string {
  const parts = [
    `${humanize(bundle.run.provider_id)} / ${humanize(bundle.run.model_id)}`,
    humanize(bundle.run.modality),
    humanize(bundle.run.sampling_preset),
    optionLabel(SEGMENTATION_OPTIONS, bundle.run.segmentation_mode),
    optionLabel(PERCEPTION_OPTIONS, perceptionPreset(bundle.config)),
    promptFamily(bundle.config),
    optionLabel(PIPELINE_OPTIONS, bundle.run.analysis_pass),
    bundle.config?.semantic_mode === "grounded" ? "Grounded interpretation" : "",
  ].filter(Boolean);
  if (bundle.revision) parts.push(`${humanize(bundle.revision.layer)} v${bundle.revision.version}`);
  parts.push(formatUsd(bundle.run.cost_usd));
  if ((bundle.run.unpriced_calls ?? 0) > 0) parts.push(`${bundle.run.unpriced_calls} unpriced`);
  return parts.join(" · ");
}
