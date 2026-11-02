import { describe, expect, it } from "vitest";
import { analysisSetupLine, coverageLine } from "../lib/runOptions";

describe("analysis setup line", () => {
  it("names the segmentation, perception, prompt family, and pipeline options", () => {
    expect(analysisSetupLine({
      run: {
        provider_id: "openai",
        model_id: "gpt-6-astra",
        modality: "sampled_frames",
        sampling_preset: "standard",
        segmentation_mode: "sam_video",
        analysis_pass: "single",
        cost_usd: 0,
      },
      revision: { layer: "model", version: 2 },
      config: {
        perception: { preset: "C_appearance" },
        prompt_name: "General scene analysis and object identification",
      },
    })).toBe(
      "OpenAI / GPT 6 astra · Sampled frames · Standard · SAM 3.1 video tracking with evidence fusion · C · Appearance-assisted · General scene analysis and object identification · Single pass · Model v2 · $0.0000",
    );
  });

  it("uses the stored prompt family when the run has no separate name", () => {
    expect(analysisSetupLine({
      run: {
        provider_id: "openai",
        model_id: "gpt-6-astra",
        modality: "sampled_frames",
        sampling_preset: "standard",
        segmentation_mode: "boxes_only",
        analysis_pass: "multi_reinspect",
        cost_usd: 0,
        unpriced_calls: 2,
      },
      config: {
        perception: { strategy: "temporal" },
        prompt_key: "scene",
      },
    })).toBe(
      "OpenAI / GPT 6 astra · Sampled frames · Standard · Boxes and crops only · B · Temporal tracking · General scene analysis and object identification · Multi-stage + reinspection · $0.0000 · 2 unpriced",
    );
  });

  it("names grounded interpretation when the run asked for it", () => {
    const line = analysisSetupLine({
      run: { provider_id: "openai", model_id: "gpt-6-astra", modality: "sampled_frames", sampling_preset: "standard", segmentation_mode: "sam_video", analysis_pass: "single", cost_usd: 0 },
      config: { semantic_mode: "grounded" },
    });
    expect(line).toContain("Grounded interpretation");
  });
});

describe("coverage line", () => {
  it("is empty for a run that did not record an interval", () => {
    expect(coverageLine({ frames_extracted: 12 })).toBe("");
    expect(coverageLine(null)).toBe("");
  });

  it("reports the interval, gaps, tracking continuity, skipped chunks, and semantic calls", () => {
    expect(coverageLine({
      time_start: 0,
      time_end: 42.5,
      frames_extracted: 24,
      max_gap_sec: 3.1,
      uncovered_gaps: [[10, 14]],
      tracking: { frames: 120, max_gap_sec: 0.5, continuity: "partial" },
      segmentation: { skipped_chunks: [{}] },
      semantic_mode: "grounded",
      semantic_calls: 2,
    })).toBe(
      "Interval 0.00–42.50 s · 24 semantic frames · largest semantic gap 3.10 s · 1 uncovered gap · 120 perception frames · largest tracking gap 0.50 s · tracking continuity partial · 1 SAM chunk skipped · grounded mode, 2 semantic calls",
    );
    expect(coverageLine({ time_start: 0, time_end: 5, semantic_mode: "combined", semantic_calls: 1 })).toBe(
      "Interval 0.00–5.00 s · combined mode, 1 semantic call",
    );
  });
});