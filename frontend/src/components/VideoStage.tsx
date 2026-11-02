import { useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { AssetPreview, PreviewNav } from "./AssetPreview";
import { api } from "../lib/api";
import type { BoxDecision, Bundle, Observation, TrackPointSample } from "../lib/types";
import { coverageAt, decisionsNear, frameObjectStills, overlayObservations, tracksAt } from "../lib/analysis";
import { formatConfidence } from "../lib/confidence";
import { humanize, labeledId } from "../lib/humanize";
import { StageZoomControl, useStageZoom } from "./StageZoom";
import { wrapLines } from "../lib/wrapText";
import { useInspector } from "../stores/ui";

const LABEL_SIZE = 14;
const LABEL_LINE = LABEL_SIZE * 1.2;
const LABEL_FONT = `${LABEL_SIZE}px "IBM Plex Sans", sans-serif`;
const CONFIDENCE_INSET = 4;

let labelCanvas: HTMLCanvasElement | null = null;

function labelWidth(text: string): number {
  labelCanvas ??= document.createElement("canvas");
  const context = labelCanvas.getContext("2d");
  if (!context) return text.length * LABEL_SIZE * 0.5;
  context.font = LABEL_FONT;
  return context.measureText(text).width;
}

export function VideoStage({ bundle }: { bundle: Bundle }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const client = useQueryClient();
  const { time, setTime, playing, setPlaying, speed, overlay, layers, seedMode, loop, selectedObject } = useInspector();
  const [seedError, setSeedError] = useState("");
  const [content, setContent] = useState({ x: 0, y: 0, w: 1, h: 1, vw: 640, vh: 360 });
  const { zoom, pan, commitView, stepZoom, frameClass, frameHandlers, frameTransform } = useStageZoom(boxRef, content);
  const [fontReady, setFontReady] = useState(false);
  const [objectsOpen, setObjectsOpen] = useState(true);
  const [preview, setPreview] = useState<number | null>(null);
  const previewRef = useRef<number | null>(null);
  previewRef.current = preview;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const seek = () => {
      const target = useInspector.getState().time;
      if (Number.isFinite(target) && Math.abs(video.currentTime - target) > 0.08) video.currentTime = target;
    };
    seek();
    video.addEventListener("loadedmetadata", seek);
    return () => video.removeEventListener("loadedmetadata", seek);
  }, [time]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = speed;
    if (playing) void video.play().catch(() => setPlaying(false));
    else video.pause();
  }, [playing, speed, setPlaying]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (previewRef.current != null) return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        setPlaying(!useInspector.getState().playing);
      } else if (event.key === "ArrowRight") {
        step(1);
      } else if (event.key === "ArrowLeft") {
        step(-1);
      } else if (event.key === "f") {
        void boxRef.current?.requestFullscreen();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPlaying, bundle.frames]);

  function step(direction: number) {
    const frames = bundle.frames;
    const now = useInspector.getState().time;
    const ordered = [...frames].sort((a, b) => a.timestamp - b.timestamp);
    const next = direction > 0
      ? ordered.find((frame) => frame.timestamp > now + 0.01)
      : [...ordered].reverse().find((frame) => frame.timestamp < now - 0.01);
    setTime(next?.timestamp ?? Math.max(0, now + direction / 10));
  }

  useEffect(() => {
    let cancel = false;
    document.fonts.load(LABEL_FONT).then(() => {
      if (!cancel) setFontReady(true);
    });
    return () => {
      cancel = true;
    };
  }, []);

  const samBoxes = overlay ? overlayObservations(bundle.observations, time, "sam") : [];
  const trackedIds = new Set(samBoxes.map((item) => item.object_local_id));
  const vlmBoxes = overlay ? overlayObservations(bundle.observations, time, "vlm").filter((item) => !trackedIds.has(item.object_local_id)) : [];
  const inferredBoxes = overlay ? overlayObservations(bundle.observations, time, "inferred") : [];
  const conflictBoxes = overlay && layers.conflicts ? overlayObservations(bundle.observations, time, "conflict") : [];
  const dropBoxes = overlay && layers.drops ? decisionsNear(bundle.box_decisions?.rows ?? [], time).filter((row) => row.bbox && row.outcome !== "kept") : [];
  const stills = frameObjectStills(bundle, time);
  const allPoints = bundle.tracks?.points ?? [];
  const trackHits = tracksAt(allPoints, time);
  const showTracks = layers.tracks || layers.trajectory || layers.uncertain;
  const selectedCoverage = selectedObject ? coverageAt(bundle.observations, selectedObject, time) : "";
  const src = bundle.video ? bundle.video.playback_url : "";

  useEffect(() => {
    const box = boxRef.current;
    const video = videoRef.current;
    if (!box) return;
    const apply = () => {
      const frame = boxRef.current;
      const node = videoRef.current;
      if (!frame || !node || !node.videoWidth) return;
      const rect = frame.getBoundingClientRect();
      const scale = Math.min(rect.width / node.videoWidth, rect.height / node.videoHeight);
      const w = node.videoWidth * scale;
      const h = node.videoHeight * scale;
      setContent({ x: (rect.width - w) / 2, y: (rect.height - h) / 2, w, h, vw: node.videoWidth, vh: node.videoHeight });
    };
    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(box);
    video?.addEventListener("loadedmetadata", apply);
    return () => {
      observer.disconnect();
      video?.removeEventListener("loadedmetadata", apply);
    };
  }, [src]);

  return (
    <div className="stage-slot flex min-h-0 w-full flex-col gap-2">
      <div
        ref={boxRef}
        className={`stage-frame rounded-lg bg-black${frameClass}`}
        {...frameHandlers}
      >
        <video
          ref={videoRef}
          src={src}
          style={frameTransform}
          onTimeUpdate={(event) => {
            const current = event.currentTarget.currentTime;
            if (loop && current > loop.end) {
              event.currentTarget.currentTime = loop.start;
              setTime(loop.start);
              return;
            }
            setTime(current);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={(event) => void seedAt(event)}
        />
        {(vlmBoxes.length > 0 || inferredBoxes.length > 0 || samBoxes.length > 0 || conflictBoxes.length > 0 || dropBoxes.length > 0) && (
          <svg
            className="pointer-events-none absolute"
            style={{ left: content.x, top: content.y, width: content.w, height: content.h, ...frameTransform }}
            viewBox={`0 0 ${content.vw} ${content.vh}`}
          >
            {vlmBoxes.map((obs) => (
              <ObservationBox key={obs.id} obs={obs} bundle={bundle} content={content} fontReady={fontReady} showConfidence={layers.confidence} showId />
            ))}
            {inferredBoxes.map((obs) => (
              <ObservationBox key={obs.id} obs={obs} bundle={bundle} content={content} fontReady={fontReady} showConfidence={layers.confidence} showId dashed />
            ))}
            {samBoxes.map((obs) => (
              <ObservationBox key={obs.id} obs={obs} bundle={bundle} content={content} fontReady={fontReady} showConfidence={layers.confidence} showId dashed={false} />
            ))}
            {conflictBoxes.map((obs) => (
              <ObservationBox key={obs.id} obs={obs} bundle={bundle} content={content} fontReady={fontReady} showConfidence={layers.confidence} showId dashed conflict />
            ))}
            {dropBoxes.map((row, index) => (
              <DecisionBox key={`${row.step}-${row.reason_code}-${row.object_local_id}-${row.timestamp}-${index}`} row={row} content={content} />
            ))}
          </svg>
        )}
        {stills.length > 0 && (
          <div
            className="frame-objects"
            onPointerDown={(event) => event.stopPropagation()}
            onWheel={(event) => event.stopPropagation()}
          >
            <div className="frame-objects-head">
              <div className="frame-objects-title">Objects in frame</div>
              <button
                type="button"
                className="frame-objects-toggle"
                aria-expanded={objectsOpen}
                aria-label={objectsOpen ? "Hide objects in frame" : "Show objects in frame"}
                onClick={() => setObjectsOpen((open) => !open)}
              >
                {objectsOpen ? <Eye size={13} strokeWidth={1.75} /> : <EyeOff size={13} strokeWidth={1.75} />}
              </button>
            </div>
            {objectsOpen && (
              <div className="frame-objects-row">
                {stills.map((still, index) => (
                  <button
                    key={still.id}
                    type="button"
                    className="frame-object"
                    title={still.label}
                    aria-label={`View ${still.label}`}
                    onClick={() => setPreview(index)}
                  >
                    <img src={still.url} alt={still.label} style={{ borderColor: still.color }} />
                    <span>{still.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        <StageZoomControl zoom={zoom} onStep={stepZoom} />
        {showTracks && trackHits.length > 0 && (
          <svg
            className="pointer-events-none absolute"
            style={{ left: content.x, top: content.y, width: content.w, height: content.h, ...frameTransform }}
            viewBox={`0 0 ${content.vw} ${content.vh}`}
          >
            {trackHits.map((point) => (
              <TrackMark key={`${point.detector_id}-${point.track_ref}-${point.timestamp}`} point={point} points={allPoints} time={time} content={content} layers={layers} />
            ))}
          </svg>
        )}
        <PreviewNav direction="prev" label="Previous frame" onClick={() => step(-1)} />
        <PreviewNav direction="next" label="Next frame" onClick={() => step(1)} />
      </div>
      <div className="flex flex-col gap-2 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <button className="btn" onClick={() => useInspector.getState().setOverlay(!overlay)}>{overlay ? "Hide boxes" : "Show boxes"}</button>
          <button className="btn" aria-pressed={layers.conflicts} onClick={() => useInspector.getState().setLayer("conflicts", !layers.conflicts)}>
            {layers.conflicts ? "Hide conflicts" : "Show conflicts"}
          </button>
          {(bundle.box_decisions?.total ?? 0) > 0 && (
            <button className="btn" aria-pressed={layers.drops} onClick={() => useInspector.getState().setLayer("drops", !layers.drops)}>
              {layers.drops ? "Hide dropped" : "Show dropped"}
            </button>
          )}
          <button className="btn" aria-pressed={layers.confidence} onClick={() => useInspector.getState().setLayer("confidence", !layers.confidence)}>
            {layers.confidence ? "Hide confidence" : "Show confidence"}
          </button>
          <button className="btn" aria-pressed={layers.tracks} onClick={() => useInspector.getState().setLayer("tracks", !layers.tracks)}>
            {layers.tracks ? "Hide tracks" : "Show tracks"}
          </button>
          {([
            ["trajectory", "trajectory"],
            ["uncertain", "uncertain tracks"],
          ] as const).map(([key, label]) => (
            <button key={key} className="btn" aria-pressed={layers[key]} onClick={() => useInspector.getState().setLayer(key, !layers[key])}>
              {layers[key] ? `Hide ${label}` : `Show ${label}`}
            </button>
          ))}
          <button
            className="btn"
            aria-pressed={seedMode}
            disabled={zoom !== 1 || pan.x !== 0 || pan.y !== 0}
            title="Reset zoom first. The click is mapped through the letterboxed frame."
            onClick={() => useInspector.getState().setSeedMode(!seedMode)}
          >
            {seedMode ? "Click the object" : "Seed a track"}
          </button>
          <button className="btn" onClick={() => commitView({ zoom: 1, panX: 0, panY: 0 })}>Reset zoom</button>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
          <label className="inline-flex items-center gap-2 text-sm text-[var(--muted)]">
            <span>Speed</span>
            <select className="field w-auto text-[var(--text)]" value={speed} onChange={(event) => useInspector.getState().setSpeed(Number(event.target.value))}>
              {[0.25, 0.5, 1, 1.5, 2].map((value) => <option key={value} value={value}>{value}×</option>)}
            </select>
          </label>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="mono text-[var(--muted)]">{time.toFixed(2)}s</span>
          <span className="text-xs text-[var(--muted)]">Space play · arrows step · f fullscreen · scroll over a point to zoom · drag to pan</span>
          {selectedObject && (
            <span className="text-xs text-[var(--muted)]">
              {humanize(selectedObject)} at this time: {coverageLabel(selectedCoverage)}. A drawn trail uses stored points and is not a filled-in location.
            </span>
          )}
        </div>
      </div>
      {preview != null && stills[preview] && (
        <AssetPreview
          key={stills[preview].id}
          images={stills.map((still) => ({ src: still.url, filename: still.label, confidence: still.confidence }))}
          startIndex={preview}
          onClose={() => setPreview(null)}
        />
      )}
      {seedError && <p className="text-xs text-[var(--rose)]">{seedError}</p>}
      {bundle.tracks?.truncated && <p className="text-xs text-[var(--muted)]">Showing the first {bundle.tracks.points.length} of {bundle.tracks.total} track points.</p>}
    </div>
  );

  function seedAt(event: MouseEvent<HTMLVideoElement>) {
    if (!seedMode || zoom !== 1 || pan.x !== 0 || pan.y !== 0) return;
    const rect = boxRef.current?.getBoundingClientRect();
    if (!rect || content.w <= 0 || content.h <= 0) return;
    const x = event.clientX - rect.left - content.x;
    const y = event.clientY - rect.top - content.y;
    if (x < 0 || y < 0 || x > content.w || y > content.h) return;
    const size = 0.12;
    const nx = Math.min(Math.max(0, x / content.w - size / 2), 1 - size);
    const ny = Math.min(Math.max(0, y / content.h - size / 2), 1 - size);
    setSeedError("");
    void api.seedTrack(bundle.run.id, { timestamp: time, x: nx, y: ny, w: size, h: size })
      .then(() => client.invalidateQueries({ queryKey: ["run", bundle.run.id] }))
      .then(() => useInspector.getState().setSeedMode(false))
      .catch((err: unknown) => setSeedError(err instanceof Error ? err.message : "The seed was not saved."));
  }
}

function coverageLabel(state: string) {
  if (state === "observed") return "directly observed";
  if (state === "unmeasured") return "video-file box, not measured on a frame";
  if (state === "inferred") return "inferred between observations";
  if (state === "absent") return "explicitly absent";
  if (state === "conflict") return "model and SAM disagree";
  if (state === "unassigned") return "SAM track was not linked to this object";
  return "unobserved";
}

function DecisionBox({ row, content }: { row: BoxDecision; content: { vw: number; vh: number } }) {
  if (!row.bbox) return null;
  const x = row.bbox.x * content.vw;
  const y = row.bbox.y * content.vh;
  const moved = row.outcome === "reassigned" && row.other_object_id ? ` → ${row.other_object_id}` : "";
  return (
    <g data-testid="dropped-box">
      <rect x={x} y={y} width={row.bbox.w * content.vw} height={row.bbox.h * content.vh} fill="none" stroke="#E6A23C" strokeWidth={2} strokeDasharray="6 4" />
      <text x={x} y={Math.max(LABEL_SIZE, y - 4)} fill="#E6A23C" fontSize={LABEL_SIZE} fontFamily="IBM Plex Sans, sans-serif">
        {`✕ ${row.object_local_id}${moved} · ${humanize(row.reason_code)}`}
      </text>
    </g>
  );
}

function ObservationBox({
  obs,
  bundle,
  content,
  fontReady,
  showConfidence,
  showId,
  dashed = false,
  conflict = false,
}: {
  obs: Observation;
  bundle: Bundle;
  content: { vw: number; vh: number };
  fontReady: boolean;
  showConfidence: boolean;
  showId: boolean;
  dashed?: boolean;
  conflict?: boolean;
}) {
  if (!obs.bbox) return null;
  const object = bundle.objects.find((item) => item.local_id === obs.object_local_id);
  const color = conflict ? "#e07070" : object?.color ?? "#7EB6FF";
  const x = obs.bbox.x * content.vw;
  const y = obs.bbox.y * content.vh;
  const w = obs.bbox.w * content.vw;
  const h = obs.bbox.h * content.vh;
  const label = showId && object ? labeledId(object.local_id, object.label) : "";
  const lines = label ? wrapLines(label, w, labelWidth) : [];
  const labelBaseY = Math.max(LABEL_SIZE, y - 4 - (lines.length - 1) * LABEL_LINE);
  const score = showConfidence && obs.confidence != null ? formatConfidence(obs.confidence) : "";
  const uncertain = obs.association_status === "ambiguous" || obs.continuity === "uncertain";
  return (
    <g key={`${obs.id}-${fontReady ? "font" : "fallback"}`}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        fill={color}
        fillOpacity={dashed ? 0.04 : 0.12}
        stroke={color}
        strokeWidth={conflict ? 2.5 : 2}
        strokeDasharray={dashed || uncertain ? "6 4" : undefined}
      />
      {lines.length > 0 && (
        <text x={x} y={labelBaseY} fill={color} fontSize={LABEL_SIZE} fontFamily="IBM Plex Sans, sans-serif">
          {lines.map((line, index) => (
            <tspan key={index} x={x} dy={index === 0 ? 0 : LABEL_LINE}>{line}</tspan>
          ))}
        </text>
      )}
      {score && (
        <text x={x + CONFIDENCE_INSET} y={y + 14} fill={color} fontSize={11} fontFamily="IBM Plex Sans, sans-serif">{score}</text>
      )}
    </g>
  );
}

function TrackMark({
  point,
  points,
  time,
  content,
  layers,
}: {
  point: TrackPointSample;
  points: TrackPointSample[];
  time: number;
  content: { vw: number; vh: number };
  layers: { tracks: boolean; trajectory: boolean; uncertain: boolean };
}) {
  if (!point.bbox) return null;
  const uncertain = layers.uncertain && (point.continuity === "uncertain" || point.status === "ambiguous" || point.continuity === "reassociated");
  const color = uncertain ? "#d2a24c" : "#7dcea0";
  const x = point.bbox.x * content.vw;
  const y = point.bbox.y * content.vh;
  const w = point.bbox.w * content.vw;
  const h = point.bbox.h * content.vh;
  const trail = layers.trajectory
    ? points.filter((item) => item.track_ref === point.track_ref && item.bbox && item.timestamp <= time && time - item.timestamp < 1.5).slice(-12)
    : [];
  return (
    <g>
      {trail.length > 1 && (
        <polyline
          fill="none"
          stroke={color}
          strokeWidth={2}
          points={trail.map((item) => {
            const box = item.bbox;
            if (!box) return "";
            return `${(box.x + box.w / 2) * content.vw},${(box.y + box.h / 2) * content.vh}`;
          }).join(" ")}
        />
      )}
      {layers.tracks && (
        <>
          <rect x={x} y={y} width={w} height={h} fill={color} fillOpacity={0.08} stroke={color} strokeWidth={2} strokeDasharray={uncertain ? "6 4" : undefined} />
          <text x={x} y={Math.max(14, y - 4)} fill={color} fontSize={14} fontFamily="IBM Plex Sans, sans-serif">{humanize(point.track_ref)}</text>
          <text x={x} y={y + h + 14} fill={color} fontSize={12} fontFamily="IBM Plex Sans, sans-serif">{point.status || point.continuity}</text>
        </>
      )}
    </g>
  );
}
