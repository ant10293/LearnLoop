import { describe, expect, it } from "vitest";
import { actionCoversTime, coverageAt, diagnosticEntries, frameObjectStills, objectFrameTimeline, observationsAt, frameNear, overlayObservations, rawRowsNear, sightingCount, tracksAt } from "../lib/analysis";
import type { Bundle } from "../lib/types";

const bundle = {
  objects: [
    { local_id: "Object_1", color: "#ff0000", label: "cup" },
  ],
  observations: [
    { id: "a", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, visibility: "visible" },
    { id: "b", object_local_id: "Object_1", timestamp: 4, bbox: { x: 0.4, y: 0.1, w: 0.2, h: 0.2 }, visibility: "visible" },
  ],
  frames: [
    { id: "f1", timestamp: 1.1, reason: "uniform", url: "/a" },
    { id: "f2", timestamp: 3.8, reason: "uniform", url: "/b" },
  ],
} as unknown as Bundle;

describe("analysis helpers", () => {
  it("selects the observation nearest the playhead inside the window", () => {
    const hits = observationsAt(bundle, 1.2);
    expect(hits.map((item) => item.id)).toEqual(["a"]);
  });

  it("does not hold a box across a long gap", () => {
    expect(observationsAt(bundle, 2.5)).toEqual([]);
  });

  it("seeks the nearest extracted frame", () => {
    expect(frameNear(bundle, 3.5)?.id).toBe("f2");
  });

  it("keeps a frame inside an action and drops one outside it", () => {
    const action = { start: 6, end: 18 };
    expect(actionCoversTime(action, 6)).toBe(true);
    expect(actionCoversTime(action, 18)).toBe(true);
    expect(actionCoversTime(action, 5.5)).toBe(false);
    expect(actionCoversTime(action, 18.2)).toBe(false);
  });

  it("lists a crop and label for each object in the frame", () => {
    const scene = {
      objects: [
        {
          local_id: "Object_1",
          label: "rod",
          color: "#7EB6FF",
          primary_image_url: "/rod.png",
          images: [{ id: "img-1", kind: "initial", timestamp: 1, url: "/rod.png", score: 1 }],
        },
        { local_id: "Object_2", label: "nut", color: "#e6b450", primary_image_url: "/nut.png", images: [] },
      ],
      observations: [
        { id: "near", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f1", visibility: "visible" },
        { id: "later", object_local_id: "Object_2", timestamp: 5, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f2", visibility: "visible" },
      ],
      frames: [],
    } as unknown as Bundle;
    expect(frameObjectStills(scene, 1.1)).toEqual([
      { id: "near", url: "/api/media/observations/near/crop", label: "Rod", color: "#7EB6FF", confidence: null },
    ]);
  });

  it("falls back to the object's still when the observation has no frame", () => {
    expect(frameObjectStills(bundle, 1).map((item) => item.url)).toEqual([]);
    const withStill = {
      ...bundle,
      objects: [{ ...bundle.objects[0], primary_image_url: "/cup.png", images: [] }],
    } as unknown as Bundle;
    expect(frameObjectStills(withStill, 1)).toEqual([
      { id: "a", url: "/cup.png", label: "Cup", color: "#ff0000", confidence: null },
    ]);
  });

  it("leaves a blank cell on every frame where the object is missing", () => {
    const cells = objectFrameTimeline(
      [
        { id: "f1", timestamp: 2 },
        { id: "gap", timestamp: 0 },
        { id: "f2", timestamp: 1 },
      ],
      "Object_1",
      [
        { id: "seen", object_local_id: "Object_1", timestamp: 0, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "gap", visibility: "visible", confidence: 0.4 },
        { id: "gone", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f2", visibility: "absent", confidence: 0.9 },
        { id: "other", object_local_id: "Object_2", timestamp: 2, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f1", visibility: "visible", confidence: 1 },
        { id: "back", object_local_id: "Object_1", timestamp: 2, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f1", visibility: "partial", confidence: 0.2 },
      ] as Bundle["observations"],
    );
    expect(cells.map((cell) => cell.frameId)).toEqual(["gap", "f2", "f1"]);
    expect(cells.map((cell) => cell.observationId)).toEqual(["seen", null, "back"]);
    expect(cells[0]?.url).toBe("/api/media/observations/seen/crop");
    expect(cells[1]?.url).toBeNull();
    expect(cells[2]?.url).toBe("/api/media/observations/back/crop");
    expect(sightingCount(
      [
        { id: "f1", timestamp: 2 },
        { id: "gap", timestamp: 0 },
        { id: "f2", timestamp: 1 },
      ],
      "Object_1",
      [
        { id: "seen", object_local_id: "Object_1", timestamp: 0, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "gap", visibility: "visible", confidence: 0.4 },
        { id: "gone", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f2", visibility: "absent", confidence: 0.9 },
        { id: "other", object_local_id: "Object_2", timestamp: 2, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f1", visibility: "visible", confidence: 1 },
        { id: "back", object_local_id: "Object_1", timestamp: 2, bbox: { x: 0, y: 0, w: 0.2, h: 0.2 }, frame_id: "f1", visibility: "partial", confidence: 0.2 },
      ] as Bundle["observations"],
    )).toBe(2);
  });

  it("uses the timestamp when the detection's frame id is not in the strip", () => {
    const cells = objectFrameTimeline(
      [
        { id: "a", timestamp: 0 },
        { id: "b", timestamp: 1 },
        { id: "c", timestamp: 2 },
      ],
      "Object_1",
      [
        { id: "seen", object_local_id: "Object_1", timestamp: 0.001, bbox: { x: 0, y: 0, w: 1, h: 1 }, frame_id: "stored-elsewhere", visibility: "visible", confidence: 0.5 },
        { id: "late", object_local_id: "Object_1", timestamp: 9, bbox: { x: 0, y: 0, w: 1, h: 1 }, frame_id: "way-off", visibility: "visible", confidence: 0.5 },
      ] as Bundle["observations"],
    );
    expect(cells.map((cell) => cell.observationId)).toEqual(["seen", null, null]);
  });

  it("keeps one track sample nearest the playhead", () => {
    const hits = tracksAt([
      { track_ref: "track_1", timestamp: 1, bbox: { x: 0, y: 0, w: 0.1, h: 0.1 }, status: "tracked", continuity: "continuous", association_score: 0.8, detector_id: "proposals", scene_chunk: 0, note: "" },
      { track_ref: "track_1", timestamp: 1.1, bbox: { x: 0.1, y: 0, w: 0.1, h: 0.1 }, status: "tracked", continuity: "continuous", association_score: 0.8, detector_id: "proposals", scene_chunk: 0, note: "" },
      { track_ref: "track_2", timestamp: 4, bbox: { x: 0.4, y: 0, w: 0.1, h: 0.1 }, status: "lost", continuity: "uncertain", association_score: null, detector_id: "proposals", scene_chunk: 0, note: "" },
    ], 1.08);
    expect(hits.map((item) => item.track_ref)).toEqual(["track_1"]);
    expect(hits[0]?.timestamp).toBe(1.1);
  });

  it("keeps model boxes, SAM boxes, and conflicts on separate layers", () => {
    const observations = [
      { id: "vlm", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, visibility: "visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "AI_PREDICTION", evidence_note: "", continuity: "continuous", confidence: 0.4, coordinate_frame: "image", camera_compensated: false, frame_id: null },
      { id: "sam", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.12, y: 0.1, w: 0.2, h: 0.2 }, visibility: "visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "CV_DETECTION", evidence_note: "", continuity: "continuous", confidence: 0.9, coordinate_frame: "image", camera_compensated: false, frame_id: null, source_model: "sam-3.1", association_status: "associated" },
      { id: "clash", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.6, y: 0.6, w: 0.2, h: 0.2 }, visibility: "visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "CV_DETECTION", evidence_note: "", continuity: "uncertain", confidence: 0.2, coordinate_frame: "image", camera_compensated: false, frame_id: null, source_model: "sam-3.1", association_status: "conflict" },
    ] as Bundle["observations"];
    expect(overlayObservations(observations, 1, "vlm").map((item) => item.id)).toEqual(["vlm"]);
    expect(overlayObservations(observations, 1, "sam").map((item) => item.id)).toEqual(["sam"]);
    expect(overlayObservations(observations, 1, "conflict").map((item) => item.id)).toEqual(["clash"]);
    expect(coverageAt(observations, "Object_1", 3)).toBe("unobserved");
    expect(coverageAt(observations, "Object_1", 1)).toBe("conflict");
  });

  it("draws a canonical tracker row on the measured layer and a provisional one dashed", () => {
    const base = { visibility: "visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "AI_PREDICTION", evidence_note: "", continuity: "continuous", confidence: 0.8, coordinate_frame: "image", camera_compensated: false, frame_id: null };
    const observations = [
      { ...base, id: "canon", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, source_model: "tracker", observation_role: "canonical" },
      { ...base, id: "provisional", object_local_id: "Object_2", timestamp: 1, bbox: { x: 0.5, y: 0.1, w: 0.2, h: 0.2 }, source_model: "tracker", observation_role: "raw" },
      { ...base, id: "estimate", object_local_id: "Object_3", timestamp: 1, bbox: { x: 0.5, y: 0.5, w: 0.2, h: 0.2 }, geometry_origin: "vlm_estimated" },
      { ...base, id: "legacy", object_local_id: "Object_4", timestamp: 1, bbox: { x: 0.1, y: 0.6, w: 0.2, h: 0.2 } },
    ] as Bundle["observations"];
    expect(overlayObservations(observations, 1, "sam").map((item) => item.id)).toEqual(["canon"]);
    expect(overlayObservations(observations, 1, "inferred").map((item) => item.id)).toEqual(["provisional"]);
    expect(overlayObservations(observations, 1, "vlm").map((item) => item.id).sort()).toEqual(["estimate", "legacy"]);
  });

  it("draws an unmeasured video-file box dashed and does not call it observed", () => {
    const row = { id: "video", object_local_id: "Object_1", timestamp: 1, bbox: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, visibility: "visible", evidence_class: "DIRECTLY_OBSERVED", provenance: "AI_PREDICTION", evidence_note: "", continuity: "continuous", confidence: 0.8, coordinate_frame: "image", camera_compensated: false, frame_id: null, geometry_authority: "vlm_video_unmeasured" };
    const observations = [row] as Bundle["observations"];
    expect(overlayObservations(observations, 1, "inferred").map((item) => item.id)).toEqual(["video"]);
    expect(overlayObservations(observations, 1, "vlm")).toEqual([]);
    expect(coverageAt(observations, "Object_1", 1)).toBe("unmeasured");
  });

  it("lists raw rows near the playhead and flattens nested diagnostics", () => {
    const row = { bbox: null, visibility: "visible", track_ref: "", association_status: "", observation_role: "raw", geometry_origin: "vlm_estimated", geometry_authority: "vlm", detection_support: "none", association_support: "", temporal_support: "", semantic_support: "labeled", evidence_ref: "" };
    const rows = [
      { ...row, object_local_id: "Object_1", timestamp: 1.2, source_model: "" },
      { ...row, object_local_id: "Object_1", timestamp: 4, source_model: "" },
    ];
    expect(rawRowsNear(rows, 1).map((item) => item.timestamp)).toEqual([1.2]);
    expect(diagnosticEntries({ tracks: 3, resolution: { canonical: 12, conflicts: 0 }, list: [1] })).toEqual([
      ["tracks", "3"],
      ["resolution_canonical", "12"],
      ["resolution_conflicts", "0"],
    ]);
  });
});
