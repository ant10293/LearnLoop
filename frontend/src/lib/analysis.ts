import { humanize } from "./humanize";
import type { BoxDecision, Bundle, LabObject, Observation, RawObservation, TrackPointSample } from "./types";

export type FrameObjectStill = {
  id: string;
  url: string;
  label: string;
  color: string;
  confidence: number | null;
};

/** Crops of the labeled objects visible at the playhead, one still per object. */
export function frameObjectStills(bundle: Bundle, time: number): FrameObjectStill[] {
  return observationsAt(bundle, time).flatMap((observation) => {
    const object = bundle.objects.find((item) => item.local_id === observation.object_local_id);
    const url = stillUrl(object, observation);
    if (!url) return [];
    const name = object?.label?.trim() ? humanize(object.label) : humanize(observation.object_local_id);
    return [{
      id: observation.id,
      url,
      label: name,
      color: object?.color ?? "#7EB6FF",
      confidence: observation.confidence ?? null,
    }];
  });
}

function stillUrl(object: LabObject | undefined, observation: Observation): string {
  if (observation.frame_id) return `/api/media/observations/${observation.id}/crop`;
  const images = object?.images ?? [];
  const stills = images.filter((image) => image.url && image.kind !== "boxed" && !image.kind.endsWith("_boxed"));
  const nearest = stills.reduce<(typeof stills)[number] | undefined>((best, image) => {
    if (!best) return image;
    return Math.abs(image.timestamp - observation.timestamp) < Math.abs(best.timestamp - observation.timestamp) ? image : best;
  }, undefined);
  return nearest?.url || object?.primary_image_url || "";
}

export function observationsAt(bundle: Bundle, time: number, window = 0.45): Observation[] {
  const chosen: Observation[] = [];
  for (const object of bundle.objects) {
    const samples = bundle.observations.filter((item) => item.object_local_id === object.local_id && item.bbox);
    if (!samples.length) continue;
    const nearest = samples.reduce((best, item) =>
      Math.abs(item.timestamp - time) < Math.abs(best.timestamp - time) ? item : best,
    );
    if (Math.abs(nearest.timestamp - time) <= window) chosen.push(nearest);
  }
  return chosen;
}

export type OverlayKind = "vlm" | "sam" | "inferred" | "conflict";

/** A video-file box that no answer on the displayed frames replaced. */
const UNMEASURED_AUTHORITY = "vlm_video_unmeasured";

/** "sam" is the measured layer: canonical rows and SAM masks draw solid. A provisional tracker row stays dashed. */
export function observationKind(observation: Observation): OverlayKind {
  if (observation.association_status === "conflict") return "conflict";
  if (observation.observation_role === "canonical") return "sam";
  if (observation.geometry_authority === UNMEASURED_AUTHORITY) return "inferred";
  if (observation.source_model === "tracker") return "inferred";
  if (observation.evidence_class === "INFERRED") return "inferred";
  if (observation.provenance === "CV_DETECTION" || observation.source_model === "sam-3.1") return "sam";
  return "vlm";
}

/** Nearest box of each object or track. Inferred and conflicting boxes stay distinct from direct ones. */
export function overlayObservations(
  observations: Observation[],
  time: number,
  kind: OverlayKind,
  window = 0.45,
): Observation[] {
  const pool = observations.filter((item) => item.bbox && observationKind(item) === kind);
  const nearest = new Map<string, Observation>();
  for (const item of pool) {
    const gap = Math.abs(item.timestamp - time);
    if (gap > window) continue;
    const key = kind === "sam" || kind === "conflict"
      ? `${item.object_local_id}:${item.track_ref || item.id}`
      : item.object_local_id;
    const current = nearest.get(key);
    if (!current || gap < Math.abs(current.timestamp - time)) nearest.set(key, item);
  }
  return [...nearest.values()];
}

export type CoverageState = "observed" | "unmeasured" | "inferred" | "absent" | "unobserved" | "conflict" | "unassigned";

/** What the playhead can claim. A missing sample is unobserved, not absence. */
export function coverageAt(observations: Observation[], localId: string, time: number, window = 0.45): CoverageState {
  const rows = observations.filter((item) => item.object_local_id === localId && Math.abs(item.timestamp - time) <= window);
  if (!rows.length) return "unobserved";
  if (rows.some((item) => item.association_status === "conflict")) return "conflict";
  if (rows.some((item) => item.association_status === "unassigned")) return "unassigned";
  if (rows.some((item) => item.evidence_class === "DIRECTLY_OBSERVED" && item.visibility === "absent")) return "absent";
  if (rows.some((item) => item.bbox && item.geometry_authority === UNMEASURED_AUTHORITY)) return "unmeasured";
  if (rows.some((item) => item.evidence_class === "DIRECTLY_OBSERVED" && (item.visibility === "visible" || item.visibility === "partial"))) {
    return "observed";
  }
  if (rows.some((item) => item.evidence_class === "INFERRED")) return "inferred";
  return "unobserved";
}

/** Flat key and value pairs. Nested sections such as resolution and grounding read as "resolution canonical 12". */
export function diagnosticEntries(diagnostics: Record<string, unknown>): [string, string][] {
  const entries: [string, string][] = [];
  for (const [key, value] of Object.entries(diagnostics)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [inner, item] of Object.entries(value as Record<string, unknown>)) {
        if (item !== null && typeof item === "object") continue;
        entries.push([`${key}_${inner}`, String(item)]);
      }
    } else if (!Array.isArray(value)) {
      entries.push([key, String(value)]);
    }
  }
  return entries;
}

/** Raw rows near the playhead, for inspection only. They are never drawn as a second overlay. */
/** Removed and moved boxes around the playhead, nearest first. */
export function decisionsNear(rows: BoxDecision[], time: number, window = 0.45): BoxDecision[] {
  return rows
    .filter((row) => row.timestamp != null && Math.abs(row.timestamp - time) <= window)
    .sort((a, b) => Math.abs((a.timestamp ?? 0) - time) - Math.abs((b.timestamp ?? 0) - time) || a.object_local_id.localeCompare(b.object_local_id));
}

export function rawRowsNear(rows: RawObservation[], time: number, window = 0.5, limit = 12): RawObservation[] {
  return rows
    .filter((row) => Math.abs(row.timestamp - time) <= window)
    .sort((a, b) => Math.abs(a.timestamp - time) - Math.abs(b.timestamp - time) || a.object_local_id.localeCompare(b.object_local_id))
    .slice(0, limit);
}

export function tracksAt(points: TrackPointSample[], time: number, window = 0.2): TrackPointSample[] {
  const nearest = new Map<string, TrackPointSample>();
  for (const point of points) {
    if (!point.bbox) continue;
    const gap = Math.abs(point.timestamp - time);
    if (gap > window) continue;
    const current = nearest.get(point.track_ref);
    if (!current || gap < Math.abs(current.timestamp - time)) nearest.set(point.track_ref, point);
  }
  return [...nearest.values()];
}

export type ObjectTimelineCell = {
  frameId: string;
  timestamp: number;
  observationId: string | null;
  url: string | null;
  confidence: number | null;
};

/** Reasoning-model samples. Per-frame SAM rows stay on the track, not in this gallery. */
export function datasetObservations(observations: Observation[]): Observation[] {
  return observations.filter((item) => item.source_model !== "sam-3.1");
}

/** Frames where this object was actually boxed. A missing or absent frame is not a sighting. */
export function sightingCount(
  frames: { id: string; timestamp: number }[] | null | undefined,
  localId: string,
  observations: Observation[],
): number {
  return objectFrameTimeline(frames, localId, observations).filter((cell) => cell.observationId).length;
}

/** One square per sampled frame. A crop when this object is in the frame, otherwise an empty cell. */
export function objectFrameTimeline(
  frames: { id: string; timestamp: number }[] | null | undefined,
  localId: string,
  observations: Observation[],
): ObjectTimelineCell[] {
  const ordered = [...(frames ?? [])].sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id));
  if (!ordered.length) return [];
  const byId = new Map(ordered.map((frame) => [frame.id, frame]));
  const present = datasetObservations(observations).filter(
    (item) => item.object_local_id === localId && item.bbox != null && item.visibility !== "absent",
  );
  const chosen = new Map<string, Observation>();
  const claimed = new Set<string>();

  for (const observation of present) {
    const frame = observation.frame_id ? byId.get(observation.frame_id) : undefined;
    if (!frame) continue;
    claimed.add(observation.id);
    const current = chosen.get(frame.id);
    if (current && !closerObservation(observation, current, frame.timestamp)) continue;
    chosen.set(frame.id, observation);
  }

  const loose = present.filter((item) => !claimed.has(item.id) && item.frame_id);
  const options: { observation: Observation; frameId: string; gap: number }[] = [];
  for (const observation of loose) {
    ordered.forEach((frame, index) => {
      const gap = Math.abs(frame.timestamp - observation.timestamp);
      if (gap > frameSlack(ordered, index)) return;
      options.push({ observation, frameId: frame.id, gap });
    });
  }
  options.sort((a, b) => a.gap - b.gap || a.observation.id.localeCompare(b.observation.id));
  const used = new Set<string>();
  for (const option of options) {
    if (used.has(option.observation.id) || chosen.has(option.frameId)) continue;
    chosen.set(option.frameId, option.observation);
    used.add(option.observation.id);
  }

  return ordered.map((frame) => {
    const observation = chosen.get(frame.id);
    const url = observation?.frame_id ? `/api/media/observations/${observation.id}/crop` : null;
    return {
      frameId: frame.id,
      timestamp: frame.timestamp,
      observationId: url ? observation?.id ?? null : null,
      url,
      confidence: url ? observation?.confidence ?? null : null,
    };
  });
}

/** Half the gap to the nearer neighbor, so a detection lands on one frame. */
function frameSlack(frames: { timestamp: number }[], index: number): number {
  const time = frames[index]?.timestamp ?? 0;
  const gaps: number[] = [];
  if (index > 0) gaps.push(time - frames[index - 1].timestamp);
  if (index < frames.length - 1) gaps.push(frames[index + 1].timestamp - time);
  if (!gaps.length) return 0.5;
  return Math.max(Math.min(...gaps) / 2, 0.001);
}

function closerObservation(next: Observation, current: Observation, frameTime: number): boolean {
  const nextGap = Math.abs(next.timestamp - frameTime);
  const currentGap = Math.abs(current.timestamp - frameTime);
  if (nextGap !== currentGap) return nextGap < currentGap;
  const nextScore = next.confidence ?? -1;
  const currentScore = current.confidence ?? -1;
  if (nextScore !== currentScore) return nextScore > currentScore;
  return next.id.localeCompare(current.id) < 0;
}

/** The playhead still sits in the action. Slack covers timestamp rounding, not the next frame. */
export function actionCoversTime(action: { start: number; end: number }, time: number, slack = 0.01): boolean {
  return time >= action.start - slack && time <= action.end + slack;
}

export function frameNear(bundle: Bundle, time: number) {
  if (!bundle.frames.length) return undefined;
  return bundle.frames.reduce((best, frame) =>
    Math.abs(frame.timestamp - time) < Math.abs(best.timestamp - time) ? frame : best,
  );
}

export function objectColor(bundle: Bundle, localId: string): string {
  return bundle.objects.find((item) => item.local_id === localId)?.color ?? "#7EB6FF";
}
