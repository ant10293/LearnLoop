import type { ReactNode } from "react";
import { formatTime } from "../lib/api";
import { Bone } from "./Skeleton";
import { actionPhrase, humanize, nameObjects, objectLabels } from "../lib/humanize";
import type { Bundle } from "../lib/types";
import { useInspector } from "../stores/ui";

const TRACKS = ["frames", "objects", "tracks", "actions", "states", "ambiguities", "steps"] as const;

export function Timeline({ bundle = null }: { bundle?: Bundle | null }) {
  const duration = Math.max(bundle?.video?.duration_sec ?? 1, 0.1);
  const { time, setTime, selectAction, selectStep, selectObject, selectedStep } = useInspector();

  return (
    <div className="shrink-0">
      <div className="space-y-1">
        <div className="relative">
          {bundle ? (
            <input
              aria-label="Seek"
              className="seek"
              type="range"
              min={0}
              max={duration}
              step={0.01}
              value={Math.min(time, duration)}
              onChange={(event) => setTime(Number(event.target.value))}
            />
          ) : (
            <div className="h-4">
              <Bone className="bone-fill" />
            </div>
          )}
        </div>
        {TRACKS.map((track) =>
          bundle ? (
            <Track key={track} name={track} duration={duration} bundle={bundle} selectedStep={selectedStep} onSeek={setTime} onAction={selectAction} onStep={selectStep} onObject={selectObject} />
          ) : (
            <TrackLane key={track} name={track}>
              <Bone className="bone-fill" />
            </TrackLane>
          ),
        )}
      </div>
      <hr className="-mx-3 mt-3 h-px border-0 bg-[var(--line)]" />
    </div>
  );
}

function TrackLane({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[88px_1fr] items-center gap-2">
      <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">{name}</div>
      <div className="relative h-6 overflow-hidden rounded bg-[#0d1016]">{children}</div>
    </div>
  );
}

function Track({
  name,
  duration,
  bundle,
  selectedStep,
  onSeek,
  onAction,
  onStep,
  onObject,
}: {
  name: (typeof TRACKS)[number];
  duration: number;
  bundle: Bundle;
  selectedStep: string | null;
  onSeek: (time: number) => void;
  onAction: (id: string, span: { start: number; end: number }) => void;
  onStep: (id: string) => void;
  onObject: (id: string) => void;
}) {
  return (
    <TrackLane name={name}>
      {segments(name, bundle).map((item) => (
          <button
            key={item.key}
            title={`${item.label} · ${formatTime(item.start)}–${formatTime(item.end)}`}
            aria-current={item.stepId != null && item.stepId === selectedStep ? "true" : undefined}
            className="absolute top-1 h-4 rounded-sm text-left"
            style={{
              left: `${(item.start / duration) * 100}%`,
              width: `${Math.max(((item.end - item.start) / duration) * 100, 0.8)}%`,
              background: item.color,
              boxShadow: item.stepId != null && item.stepId === selectedStep ? "inset 0 0 0 2px var(--teal)" : undefined,
            }}
            onClick={() => {
              onSeek(item.start);
              if (item.actionId != null && item.actionStart != null && item.actionEnd != null) {
                onAction(item.actionId, { start: item.actionStart, end: item.actionEnd });
              }
              if (item.stepId) onStep(item.stepId);
              if (item.objectId) onObject(item.objectId);
            }}
            onMouseEnter={(event) => {
              const frame = bundle.frames.find((candidate) => Math.abs(candidate.timestamp - item.start) < 0.6);
              if (!frame) return;
              event.currentTarget.title = `${item.label}\n${formatTime(item.start)}`;
            }}
          />
        ))}
    </TrackLane>
  );
}

interface Segment {
  key: string;
  start: number;
  end: number;
  label: string;
  color: string;
  actionId?: string;
  actionStart?: number;
  actionEnd?: number;
  stepId?: string;
  objectId?: string;
}

function segments(name: (typeof TRACKS)[number], bundle: Bundle): Segment[] {
  if (name === "frames") {
    return bundle.frames.map((frame) => ({
      key: frame.id,
      start: frame.timestamp,
      end: frame.timestamp + 0.08,
      label: humanize(frame.reason),
      color: frame.scene_change ? "#e06c75" : "#465066",
    }));
  }
  if (name === "tracks") return trackSegments(bundle);
  if (name === "actions") {
    const labels = objectLabels(bundle.objects);
    return bundle.actions.map((action) => ({
      key: action.id,
      start: action.start,
      end: Math.max(action.end, action.start + 0.05),
      label: `${actionPhrase(action, labels)} · ${humanize(action.outcome)}`,
      color: action.outcome === "completed" ? "#1f6b4a" : action.outcome === "attempted" ? "#8a6a22" : "#5c3a55",
      actionId: action.local_id,
      actionStart: action.start,
      actionEnd: action.end,
    }));
  }
  if (name === "objects") {
    return bundle.objects.flatMap((object) => {
      const times = bundle.observations.filter((item) => item.object_local_id === object.local_id).map((item) => item.timestamp);
      if (!times.length) return [];
      return [{
        key: object.id,
        start: Math.min(...times),
        end: Math.max(...times),
        label: object.label,
        color: object.color,
        objectId: object.local_id,
      }];
    });
  }
  if (name === "states") {
    return bundle.transitions.map((item) => ({
      key: item.id,
      start: item.timestamp,
      end: item.timestamp + 0.12,
      label: `${humanize(item.from_state)} → ${humanize(item.to_state)}`,
      color: "#31597a",
      objectId: item.object_local_id,
    }));
  }
  if (name === "ambiguities") {
    return bundle.ambiguities.map((item) => ({
      key: item.id,
      start: item.time_start ?? 0,
      end: item.time_end ?? (item.time_start ?? 0) + 0.2,
      label: item.description,
      color: "#6a3036",
    }));
  }
  const labels = objectLabels(bundle.objects);
  return bundle.steps.map((step) => ({
    key: step.id,
    start: step.time_start ?? 0,
    end: step.time_end ?? (step.time_start ?? 0) + 0.2,
    label: nameObjects(step.name, labels),
    color: "#2c4a3d",
    stepId: step.local_id,
  }));
}

const TRACK_COLOR: Record<string, string> = {
  tracked: "#3d7a62",
  tentative: "#465066",
  occluded: "#8a6a22",
  lost: "#6a3036",
  archived: "#3a3f4b",
  ambiguous: "#7a4e8a",
  reassociated: "#2f6f8f",
  corrected: "#2f6f8f",
};

function trackSegments(bundle: Bundle): Segment[] {
  const points = [...(bundle.tracks?.points ?? [])].sort((a, b) => a.timestamp - b.timestamp);
  const grouped = new Map<string, typeof points>();
  for (const point of points) {
    const rows = grouped.get(point.track_ref) ?? [];
    rows.push(point);
    grouped.set(point.track_ref, rows);
  }
  const out: Segment[] = [];
  for (const [ref, rows] of grouped) {
    if (!rows.length) continue;
    let start = rows[0];
    let previous = rows[0];
    const flush = (end: typeof previous) => {
      const status = end.continuity === "reassociated" ? "reassociated" : end.status || "tracked";
      out.push({
        key: `${ref}-${start.timestamp}-${status}`,
        start: start.timestamp,
        end: Math.max(end.timestamp, start.timestamp + 0.05),
        label: `${humanize(ref)} · ${humanize(status)}`,
        color: TRACK_COLOR[status] ?? "#465066",
      });
    };
    for (const point of rows.slice(1)) {
      const gap = point.timestamp - previous.timestamp;
      if (gap > 0.45 || point.status !== start.status || point.continuity === "reassociated") {
        flush(previous);
        start = point;
      }
      previous = point;
    }
    flush(previous);
  }
  return out.slice(0, 400);
}
