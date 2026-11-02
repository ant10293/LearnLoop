import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from "react";
import { PreviewNav } from "../components/AssetPreview";
import { Bone, FieldBone } from "../components/Skeleton";
import { StageZoomControl, useStageZoom } from "../components/StageZoom";
import { api, formatTime, formatWhen } from "../lib/api";
import { useOptionalBundle } from "../lib/bundleContext";
import { humanize } from "../lib/humanize";
import {
  boxAt,
  dragBox,
  failureLabel,
  GAP_STATE_LABELS,
  LANE_COLORS,
  laneSpans,
  LAYERS,
  layerBoxes,
  manualTargets,
  nextStep,
  placeSeed,
  reacquisitionLink,
  reacquisitionRange,
  runCoverage,
  SEGMENT_COLORS,
  stepTimes,
  trackRange,
  usage,
  CANDIDATE_COLORS,
  PROMPT_ROLE_LABELS,
  WORKFLOW_LABELS,
  type ObjectTrackHistory,
  type ReappearanceCandidate,
  type RawSamBox,
  type ReacquisitionEvidence,
  type ReacquisitionLink,
  type ReacquisitionPlanItem,
  type RecoveryDiagnostics,
  type RecoveryExperiment,
  type RecoveryLayer,
  type RecoveryProposal,
  type RecoveryTarget,
  type SeedDraft,
  type TrackDirection,
} from "../lib/recovery";
import type { BBox, Bundle } from "../lib/types";
import { useInspector } from "../stores/ui";

type PointTool = "none" | "positive" | "negative";

const START_LAYERS: Record<RecoveryLayer, boolean> = { vlm: true, tracker: false, sam: false, recovered: true, reacquired: true, raw: false, canonical: true };
const RUNNING_POLL_MS = 1500;
const NEAR_SEC = 0.45;
const INSPECT_LABEL_SIZE = 14;
const MIN_LABEL_SIZE = 10;
const SQUARE_BUTTON = { width: 28, height: 28, minWidth: 28, minHeight: 28, display: "inline-flex", alignItems: "center", justifyContent: "center" };
const OBJECT_PLACEHOLDERS = 3;

export function RecoveryPage() {
  const bundle = useOptionalBundle();
  if (!bundle) return <RecoverySkeleton />;
  return <RecoveryWorkspace bundle={bundle} />;
}

function RecoverySkeleton() {
  const objects = Array.from({ length: OBJECT_PLACEHOLDERS }, (_, index) => index);
  return (
    <div className="space-y-3" aria-busy="true">
      <p className="sr-only">Loading recovery</p>
      <div className="grid min-w-0 gap-3 xl:grid-cols-[minmax(0,1fr)_400px]" aria-hidden="true">
        <section className="panel min-w-0 space-y-2 p-3">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {objects.map((index) => <span key={index} className="btn"><Bone width="4rem" /></span>)}
          </div>
          <p className="text-xs"><Bone width="80%" /></p>
          <div className="stage-frame overflow-hidden rounded-lg bg-black">
            <Bone className="bone-fill" />
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="mono"><Bone width="3rem" /></span>
            {LAYERS.map((layer) => <span key={layer.key} className="btn">{layer.label}</span>)}
            <span className="btn">+ point</span>
            <span className="btn">− point</span>
          </div>
          <div className="space-y-1 border-t border-[var(--line)] pt-2">
            {objects.map((index) => (
              <div key={index} className="grid grid-cols-[110px_1fr] items-center gap-2">
                <span className="text-xs"><Bone width="70%" /></span>
                <div className="h-5 overflow-hidden rounded"><Bone className="bone-fill" /></div>
              </div>
            ))}
          </div>
        </section>
        <div className="min-w-0 space-y-3">
          <section className="panel space-y-2 p-4 text-sm">
            <h2 className="text-base">Seeds</h2>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <FieldBone compact />
              <FieldBone compact />
            </div>
            <p className="text-xs"><Bone width="90%" /></p>
            <div className="border-t border-[var(--line)] pt-3">
              <span className="btn primary"><Bone width="8rem" /></span>
            </div>
          </section>
          <div className="panel p-3 text-sm"><Bone width="10rem" /></div>
        </div>
      </div>
    </div>
  );
}

function RecoveryWorkspace({ bundle }: { bundle: Bundle }) {
  const runId = bundle.run.id;
  const client = useQueryClient();
  const time = useInspector((state) => state.time);
  const seek = useInspector((state) => state.setTime);
  const [objectId, setObjectId] = useState("");
  const [drafts, setDrafts] = useState<SeedDraft[]>([]);
  const [seconds, setSeconds] = useState(2);
  const [direction, setDirection] = useState<TrackDirection>("forward");
  const [pointTool, setPointTool] = useState<PointTool>("none");
  const [layers, setLayers] = useState(START_LAYERS);
  const [experimentId, setExperimentId] = useState("");
  const [plannedKey, setPlannedKey] = useState("");
  const [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const [provisional, setProvisional] = useState(false);
  const [drawingFor, setDrawingFor] = useState<{ objectId: string; link: ReacquisitionLink; item: ReacquisitionPlanItem } | null>(null);
  const [maxRequests, setMaxRequests] = useState("4");
  const [maxFrames, setMaxFrames] = useState("300");
  const [maxBilled, setMaxBilled] = useState("600");
  const [budget, setBudget] = useState("1");
  const candidates = useQuery({ queryKey: ["recovery-candidates", runId], queryFn: () => api.recoveryCandidates(runId) });
  const history = useQuery({ queryKey: ["recovery-experiments", runId], queryFn: () => api.recoveryExperiments(runId) });
  const experiment = useQuery({
    queryKey: ["recovery-experiment", experimentId],
    queryFn: () => api.recoveryExperiment(experimentId),
    enabled: Boolean(experimentId),
    refetchInterval: (query) => (query.state.data?.status === "running" ? RUNNING_POLL_MS : false),
  });
  const data = experiment.data ?? null;
  const proposals = useMemo(() => (data?.proposals ?? []).filter((item) => item.current), [data]);
  const allTargets = candidates.data?.targets ?? [];
  const targets = useMemo(() => allTargets.filter((item) => !item.dismissed), [allTargets]);
  const dismissed = useMemo(() => allTargets.filter((item) => item.dismissed), [allTargets]);
  const objectIds = useMemo(() => new Set(bundle.objects.map((item) => item.local_id)), [bundle.objects]);
  const draftsKey = JSON.stringify([drafts, maxRequests, maxFrames, maxBilled, budget]);
  const plan = data?.plan && plannedKey === draftsKey ? data.plan : null;
  const diagnostics = useQuery({
    queryKey: ["recovery-diagnostics", experimentId, data?.generation, data?.status, data?.requests.length],
    queryFn: () => api.recoveryDiagnostics(experimentId),
    enabled: Boolean(data && (data.generation > 0 || data.requests.length > 0) && data.status !== "running"),
  });
  const reacquisition = useQuery({
    queryKey: ["recovery-reacquisition", runId, experimentId, data?.generation, data?.status],
    queryFn: () => api.recoveryReacquisition(runId),
  });
  const rawBoxes = useMemo<RawSamBox[]>(() => diagnostics.data?.requests.flatMap((item) => item.boxes) ?? [], [diagnostics.data]);
  const boxes = layerBoxes(bundle, proposals, time, layers, rawBoxes);
  const steps = useMemo(() => stepTimes(bundle, proposals, layers.raw ? rawBoxes : []), [bundle, proposals, layers.raw, rawBoxes]);

  function step(delta: 1 | -1) {
    const next = nextStep(steps, useInspector.getState().time, delta);
    if (next != null) seek(next);
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.key === "ArrowRight") step(1);
      else if (event.key === "ArrowLeft") step(-1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function publish(result: RecoveryExperiment) {
    client.setQueryData(["recovery-experiment", result.id], result);
    setExperimentId(result.id);
    void client.invalidateQueries({ queryKey: ["recovery-experiments", runId] });
  }

  const act = useMutation({
    mutationFn: (work: () => Promise<RecoveryExperiment>) => work(),
    onMutate: () => setError(""),
    onSuccess: publish,
    onError: (err: Error) => setError(err.message),
  });

  const problemAction = useMutation({
    mutationFn: ({ targetId, restore }: { targetId: string; restore: boolean }) =>
      restore ? api.restoreRecoveryProblem(runId, targetId) : api.dismissRecoveryProblem(runId, targetId),
    onMutate: () => setError(""),
    onSuccess: (result) => client.setQueryData(["recovery-candidates", runId], result),
    onError: (err: Error) => setError(err.message),
  });

  function addSeed(target: string, at: number, bbox: BBox | null, origin: string) {
    const drawing = drawingFor?.objectId === target ? drawingFor : null;
    const range = drawing ? reacquisitionRange(drawing.item, at) : trackRange(at, seconds, direction);
    setDrafts((current) => placeSeed(current, {
      objectId: target,
      timestamp: at,
      bbox,
      points: [],
      ...range,
      direction: drawing ? "both" : direction,
      origin: drawing ? `${origin}, reacquires the track lost at ${drawing.link.failure_time.toFixed(2)}s` : origin,
      ...(drawing ? { reacquires: drawing.link } : {}),
    }));
    setDrawingFor(null);
    setObjectId(target);
    setHint("");
  }

  function addPoint(x: number, y: number) {
    if (!objectId) {
      setHint("Choose an object above first.");
      return;
    }
    const negative = pointTool === "negative";
    const existing = drafts.find((item) => item.objectId === objectId && Math.abs(item.timestamp - time) <= 0.05);
    if (existing) {
      setDrafts((current) => current.map((item) => item.id === existing.id ? { ...item, points: [...item.points, { x, y, negative }] } : item));
    } else if (negative) {
      setHint("Add a box or a positive point first.");
    } else {
      const range = trackRange(time, seconds, direction);
      setDrafts((current) => placeSeed(current, { objectId, timestamp: time, bbox: null, points: [{ x, y }], ...range, direction, origin: "points" }));
    }
  }

  function onPicture(box: BBox | null, x: number, y: number) {
    if (pointTool !== "none") {
      addPoint(x, y);
    } else if (box) {
      if (objectId) addSeed(objectId, time, box, "drawn");
      else setHint("Choose which object you are drawing, then drag again.");
    } else {
      const hit = boxAt(boxes, objectIds, x, y);
      if (hit) {
        seek(hit.timestamp);
        addSeed(hit.objectId, hit.timestamp, hit.bbox, `${LAYERS.find((item) => item.key === hit.layer)?.label ?? "stored"} box`);
      }
    }
  }

  function useSuggested(target: RecoveryTarget) {
    const seed = target.seeds?.[0];
    setObjectId(target.object_local_id);
    seek(seed?.timestamp ?? target.time_start);
    if (!seed?.bbox) return;
    setDrafts((current) => placeSeed(current, {
      objectId: target.object_local_id,
      timestamp: seed.timestamp,
      bbox: seed.bbox,
      points: [],
      start: target.time_start,
      end: target.time_end,
      direction: "auto",
      origin: `suggested, ${humanize(seed.trust)}`,
    }));
  }

  function useEvidence(objectIdToSeed: string, item: ReacquisitionPlanItem, strategy: string, evidence: ReacquisitionEvidence) {
    const range = reacquisitionRange(item, evidence.timestamp);
    setObjectId(objectIdToSeed);
    seek(evidence.timestamp);
    setDrafts((current) => placeSeed(current, {
      objectId: objectIdToSeed,
      timestamp: evidence.timestamp,
      bbox: evidence.bbox,
      points: [],
      ...range,
      direction: "both",
      origin: `${humanize(evidence.source)}, ${humanize(evidence.trust)}, reacquires the track lost at ${item.failure.time.toFixed(2)}s`,
      ...(evidence.seed_id ? { seedId: evidence.seed_id, confirmSeed: evidence.needs_confirmation } : {}),
      reacquires: reacquisitionLink(item, strategy),
    }));
  }

  function planConcept(objectIdToFind: string, item: ReacquisitionPlanItem, concept: string) {
    const key = draftsKey;
    act.mutate(async () => {
      const planned = await api.planReacquisition(runId, {
        object_local_id: objectIdToFind,
        failure_id: item.failure.failure_id,
        action: "concept",
        concept,
        limits: { max_requests: Number(maxRequests), max_frames_per_request: Number(maxFrames), max_total_billed_frames: Number(maxBilled), cost_budget_usd: Number(budget) },
      });
      setPlannedKey(key);
      return planned;
    });
  }

  function planCandidate(objectIdToFind: string, item: ReacquisitionPlanItem, candidate: ReappearanceCandidate) {
    const key = draftsKey;
    act.mutate(async () => {
      const planned = await api.planReacquisition(runId, {
        object_local_id: objectIdToFind,
        failure_id: item.failure.failure_id,
        candidate_id: candidate.candidate_id,
        confirm: candidate.needs_confirmation,
        limits: { max_requests: Number(maxRequests), max_frames_per_request: Number(maxFrames), max_total_billed_frames: Number(maxBilled), cost_budget_usd: Number(budget) },
      });
      setPlannedKey(key);
      return planned;
    });
  }

  function preview() {
    const key = draftsKey;
    act.mutate(async () => {
      const created = await api.createRecovery(runId, {
        manual_targets: manualTargets(drafts),
        limits: {
          max_requests: Number(maxRequests),
          max_frames_per_request: Number(maxFrames),
          max_total_billed_frames: Number(maxBilled),
          cost_budget_usd: Number(budget),
          padding_sec: 0,
        },
      });
      const planned = await api.recoveryDryRun(created.id);
      setPlannedKey(key);
      return planned;
    });
  }

  function decide(decision: "accepted" | "rejected") {
    const pending = proposals.filter((item) => item.decision === "pending").map((item) => item.id);
    act.mutate(async () => {
      const result = await api.decideRecovery(experimentId, { decision, proposal_ids: pending, include_provisional: provisional });
      if (decision === "accepted") {
        setDrafts([]);
        await client.invalidateQueries({ queryKey: ["run", runId] });
        await client.invalidateQueries({ queryKey: ["recovery-candidates", runId] });
      }
      return result;
    });
  }

  return (
    <div className="space-y-3" data-testid="recovery-page">
      <div className="grid min-w-0 gap-3 xl:grid-cols-[minmax(0,1fr)_400px]">
        <section className="panel min-w-0 space-y-2 p-3">
          <div className="flex flex-wrap items-center gap-2 text-sm" role="group" aria-label="Object to seed">
            {bundle.objects.map((object) => (
              <button
                key={object.local_id}
                className="btn"
                aria-pressed={objectId === object.local_id}
                title={object.label}
                style={objectId === object.local_id ? { borderColor: object.color, color: object.color } : undefined}
                onClick={() => setObjectId(objectId === object.local_id ? "" : object.local_id)}
              >
                {humanize(object.local_id)}
              </button>
            ))}
          </div>
          <p className="text-xs text-[var(--muted)]">
            Click a box on the frame to use it as a seed, or choose an object and drag to draw one. Add as many as you need, then send them together. Scroll or use − / + to zoom, and Shift-drag to pan.
          </p>
          {hint && <p className="text-xs text-[var(--amber)]" role="status">{hint}</p>}
          <RecoveryStage bundle={bundle} boxes={boxes} drafts={drafts} objectId={objectId} time={time} pointTool={pointTool} onPicture={onPicture} onStep={step} />
          <div className="flex flex-wrap items-center gap-2 text-sm" role="group" aria-label="Box layers">
            <span className="mono text-[var(--muted)]">{time.toFixed(2)}s</span>
            {LAYERS.map((layer) => (
              <button
                key={layer.key}
                className="btn"
                aria-pressed={layers[layer.key]}
                style={layers[layer.key] ? { borderColor: layer.color, color: layer.color } : undefined}
                onClick={() => setLayers({ ...layers, [layer.key]: !layers[layer.key] })}
              >
                {layer.label}
              </button>
            ))}
            <button className="btn" aria-pressed={pointTool === "positive"} onClick={() => setPointTool(pointTool === "positive" ? "none" : "positive")}>+ point</button>
            <button className="btn" aria-pressed={pointTool === "negative"} onClick={() => setPointTool(pointTool === "negative" ? "none" : "negative")}>− point</button>
          </div>
          <ObjectLanes bundle={bundle} targets={targets} proposals={proposals} drafts={drafts} time={time} objectId={objectId} onSeek={seek} onObject={setObjectId} />
          <details className="border-t border-[var(--line)] pt-2 text-xs">
            <summary className="cursor-pointer text-sm">What is at this frame</summary>
            <div className="mt-2 h-40 space-y-1 overflow-auto" data-testid="box-inspector">
              {boxes.length === 0 ? (
                <p className="text-[var(--muted)]">No selected layer has a box near this frame. A missing box is unobserved, not absence.</p>
              ) : (
                <ul className="space-y-1">
                  {boxes.map((box) => (
                    <li key={box.key}>
                      <span className="font-medium">{LAYERS.find((item) => item.key === box.layer)?.label}</span> · {humanize(box.objectId)} · {formatTime(box.timestamp)} · {box.details.join(" · ")}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </details>
        </section>
        <div className="min-w-0 space-y-3">
          <SeedList
            drafts={drafts}
            seconds={seconds}
            direction={direction}
            onSeconds={setSeconds}
            onDirection={setDirection}
            onSeek={seek}
            onChange={setDrafts}
          >
            <details className="text-xs" data-testid="recovery-limits">
              <summary className="cursor-pointer text-[var(--muted)]">Limits</summary>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Limit label="Provider requests" value={maxRequests} onChange={setMaxRequests} />
                <Limit label="Frames per request" value={maxFrames} onChange={setMaxFrames} />
                <Limit label="Total billed frames" value={maxBilled} onChange={setMaxBilled} />
                <Limit label="Cost budget (USD)" value={budget} onChange={setBudget} />
              </div>
            </details>
            <RunPanel
              drafts={drafts}
              data={data}
              plan={plan}
              pending={act.isPending}
              error={error}
              onPreview={preview}
              onLive={(hash, keys) => act.mutate(() => api.recoveryLive(experimentId, { confirm: true, plan_hash: hash, ...(keys ? { resubmit_keys: keys } : {}) }))}
              onReplay={() => act.mutate(() => api.recoveryReplay(experimentId))}
              onCancel={() => act.mutate(() => api.recoveryCancel(experimentId))}
            />
          </SeedList>
          <ResultsPanel data={data} proposals={proposals} provisional={provisional} onProvisional={setProvisional} pending={act.isPending} onDecide={decide} onSeek={seek} />
          {diagnostics.data && diagnostics.data.requests.length > 0 && <SamDiagnostics data={diagnostics.data} onSeek={seek} />}
          {(reacquisition.data?.objects.length ?? 0) > 0 && (
            <TrackHistory
              objects={reacquisition.data?.objects ?? []}
              note={reacquisition.data?.note ?? ""}
              duration={Math.max(bundle.video?.duration_sec ?? 1, 0.1)}
              time={time}
              drawingFor={drawingFor?.objectId ?? ""}
              busy={act.isPending}
              onSeek={seek}
              onUseEvidence={useEvidence}
              onConcept={planConcept}
              onCandidate={planCandidate}
              onCompare={() => setLayers((current) => ({ ...current, recovered: true, reacquired: true, raw: true }))}
              onDraw={(id, item) => {
                setObjectId(id);
                setPointTool("none");
                setDrawingFor({ objectId: id, link: reacquisitionLink(item, "manual"), item });
                setHint(`Find a frame where ${humanize(id)} is visible again and drag a box around it.`);
                if (item.reappearance_frames[0] != null) seek(item.reappearance_frames[0]);
              }}
            />
          )}
          <ProblemList
            targets={targets}
            dismissed={dismissed}
            pending={candidates.isPending}
            busy={problemAction.isPending}
            error={problemAction.isError ? error : ""}
            onSeek={seek}
            onObject={setObjectId}
            onSuggested={useSuggested}
            onDismiss={(targetId) => problemAction.mutate({ targetId, restore: false })}
            onRestore={(targetId) => problemAction.mutate({ targetId, restore: true })}
          />
          <EarlierRuns
            runs={history.data ?? []}
            selected={experimentId}
            onOpen={(id) => {
              setExperimentId(id);
              setPlannedKey("");
            }}
          />
        </div>
      </div>
    </div>
  );
}

function EarlierRuns({
  runs,
  selected,
  onOpen,
}: {
  runs: RecoveryExperiment[];
  selected: string;
  onOpen: (id: string) => void;
}) {
  if (!runs.length) return null;
  return (
    <details className="panel p-3 text-sm" data-testid="earlier-runs">
      <summary className="cursor-pointer">Earlier runs ({runs.length})</summary>
      <ul className="mt-2 max-h-72 space-y-1 overflow-auto">
        {runs.map((run) => (
          <li key={run.id}>
            <button
              className="w-full rounded border border-[var(--line)] p-2 text-left text-xs"
              aria-current={run.id === selected ? "true" : undefined}
              style={run.id === selected ? { borderColor: "var(--teal)" } : undefined}
              onClick={() => onOpen(run.id)}
            >
              <span className="block font-medium">{formatWhen(run.created_at)} · {runCoverage(run.targets)}</span>
              <span className="block text-[var(--muted)]">
                {humanize(run.status)}{run.pending_proposals > 0 ? ` · ${run.pending_proposals} pending review` : ""}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Limit({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field-label"><span>{label}</span>
      <input className="field stretch" inputMode="decimal" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function RecoveryStage({
  bundle,
  boxes,
  drafts,
  objectId,
  time,
  pointTool,
  onPicture,
  onStep,
}: {
  bundle: Bundle;
  boxes: ReturnType<typeof layerBoxes>;
  drafts: SeedDraft[];
  objectId: string;
  time: number;
  pointTool: PointTool;
  onPicture: (box: BBox | null, x: number, y: number) => void;
  onStep: (delta: 1 | -1) => void;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  // The overlay draws in on-screen pixels, so strokes and labels keep their size on a 4K source.
  const [size, setSize] = useState({ x: 0, y: 0, w: 1, h: 1, vw: 1 });
  const [drag, setDrag] = useState<{ x: number; y: number; ex: number; ey: number } | null>(null);
  const { zoom, pan, panning, stepZoom, frameClass, frameHandlers, frameTransform } = useStageZoom(frameRef, size);
  useEffect(() => {
    const frame = frameRef.current;
    const video = videoRef.current;
    if (!frame || !video) return;
    const apply = () => {
      if (!video.videoWidth) return;
      const rect = frame.getBoundingClientRect();
      const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
      const w = video.videoWidth * scale;
      const h = video.videoHeight * scale;
      setSize({ x: (rect.width - w) / 2, y: (rect.height - h) / 2, w, h, vw: video.videoWidth });
    };
    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(frame);
    video.addEventListener("loadedmetadata", apply);
    return () => {
      observer.disconnect();
      video.removeEventListener("loadedmetadata", apply);
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const show = () => {
      if (Number.isFinite(time) && Math.abs(video.currentTime - time) > 0.001) video.currentTime = time;
    };
    show();
    video.addEventListener("loadedmetadata", show);
    return () => video.removeEventListener("loadedmetadata", show);
  }, [time]);

  function normalized(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height };
  }

  const preview = drag ? dragBox(drag.x, drag.y, drag.ex, drag.ey) : null;
  const color = (layer: RecoveryLayer) => LAYERS.find((item) => item.key === layer)?.color ?? "#fff";
  const shown = drafts.filter((item) => Math.abs(item.timestamp - time) <= NEAR_SEC);
  // Inspect's label size on this video, with a floor so it stays readable on a high-resolution source.
  const labelSize = Math.max(INSPECT_LABEL_SIZE * (size.w / size.vw), MIN_LABEL_SIZE);
  // Sized rather than scaled with the video, so a zoomed frame keeps thin strokes for precise seeds.
  const drawn = { w: size.w * zoom, h: size.h * zoom };
  return (
    <div ref={frameRef} className={`stage-frame rounded-lg bg-black${frameClass}`} {...frameHandlers}>
      <video ref={videoRef} src={bundle.video?.playback_url ?? ""} muted preload="auto" style={frameTransform} />
      <div
        className="absolute"
        data-testid="recovery-overlay"
        style={{
          left: size.x + (size.w - drawn.w) / 2 + pan.x,
          top: size.y + (size.h - drawn.h) / 2 + pan.y,
          width: drawn.w,
          height: drawn.h,
          cursor: panning ? "grabbing" : "crosshair",
          touchAction: "none",
        }}
        onPointerDown={(event) => {
          // Shift-drag falls through to the frame, which pans.
          if (event.shiftKey) return;
          event.stopPropagation();
          const at = normalized(event);
          event.currentTarget.setPointerCapture?.(event.pointerId);
          setDrag({ x: at.x, y: at.y, ex: at.x, ey: at.y });
        }}
        onPointerMove={(event) => {
          if (!drag) return;
          const at = normalized(event);
          setDrag({ ...drag, ex: at.x, ey: at.y });
        }}
        onPointerUp={() => {
          if (drag) onPicture(pointTool === "none" ? preview : null, drag.x, drag.y);
          setDrag(null);
        }}
      >
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${drawn.w} ${drawn.h}`}>
          {boxes.map((box) => (
            <g key={box.key} data-layer={box.layer}>
              <rect
                x={box.bbox.x * drawn.w}
                y={box.bbox.y * drawn.h}
                width={box.bbox.w * drawn.w}
                height={box.bbox.h * drawn.h}
                fill={box.objectId === objectId ? "#ffffff14" : "none"}
                stroke={color(box.layer)}
                strokeWidth={box.objectId === objectId ? 3 : 1.5}
                strokeDasharray={box.dashed ? "6 4" : undefined}
              />
              <text x={box.bbox.x * drawn.w} y={Math.max(13, box.bbox.y * drawn.h - 4)} fill={color(box.layer)} fontSize={labelSize} fontFamily="IBM Plex Sans, sans-serif">
                {humanize(box.objectId)}
              </text>
            </g>
          ))}
          {shown.map((seed) => (
            <g key={seed.id}>
              {seed.bbox && (
                <>
                  <rect data-testid="drawn-seed" x={seed.bbox.x * drawn.w} y={seed.bbox.y * drawn.h} width={seed.bbox.w * drawn.w} height={seed.bbox.h * drawn.h} fill="#ffffff22" stroke="#fff" strokeWidth={3} />
                  <text x={seed.bbox.x * drawn.w} y={(seed.bbox.y + seed.bbox.h) * drawn.h + 15} fill="#fff" fontSize={labelSize} fontFamily="IBM Plex Sans, sans-serif">Seed · {humanize(seed.objectId)}</text>
                </>
              )}
              {seed.points.map((point, index) => (
                <circle key={index} cx={point.x * drawn.w} cy={point.y * drawn.h} r={5} fill={point.negative ? "#e07070" : "#7dcea0"} stroke="#000" />
              ))}
            </g>
          ))}
          {preview && <rect x={preview.x * drawn.w} y={preview.y * drawn.h} width={preview.w * drawn.w} height={preview.h * drawn.h} fill="#ffffff22" stroke="#fff" strokeWidth={2} />}
        </svg>
      </div>
      <StageZoomControl zoom={zoom} onStep={stepZoom} />
      <PreviewNav direction="prev" label="Previous frame" onClick={() => onStep(-1)} />
      <PreviewNav direction="next" label="Next frame" onClick={() => onStep(1)} />
    </div>
  );
}

function ObjectLanes({
  bundle,
  targets,
  proposals,
  drafts,
  time,
  objectId,
  onSeek,
  onObject,
}: {
  bundle: Bundle;
  targets: RecoveryTarget[];
  proposals: RecoveryProposal[];
  drafts: SeedDraft[];
  time: number;
  objectId: string;
  onSeek: (time: number) => void;
  onObject: (id: string) => void;
}) {
  const duration = Math.max(bundle.video?.duration_sec ?? 1, 0.1);
  const percent = (value: number) => `${Math.min(Math.max(value / duration, 0), 1) * 100}%`;
  return (
    <div className="space-y-1 border-t border-[var(--line)] pt-2" data-testid="object-lanes">
      <div className="grid grid-cols-[110px_1fr] items-center gap-2">
        <span aria-hidden="true" />
        <input
          aria-label="Seek"
          className="seek"
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={Math.min(time, duration)}
          onChange={(event) => onSeek(Number(event.target.value))}
        />
      </div>
      {bundle.objects.map((object) => (
        <div key={object.local_id} className="grid grid-cols-[110px_1fr] items-center gap-2">
          <button
            className={`truncate text-left text-xs ${objectId === object.local_id ? "font-medium" : "text-[var(--muted)]"}`}
            style={objectId === object.local_id ? { color: object.color } : undefined}
            title={object.label}
            onClick={() => onObject(object.local_id)}
          >
            {humanize(object.local_id)}
          </button>
          <div
            className="relative h-5 cursor-pointer overflow-hidden rounded bg-[#0d1016]"
            data-testid={`lane-${object.local_id}`}
            onClick={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              onObject(object.local_id);
              onSeek(((event.clientX - rect.left) / rect.width) * duration);
            }}
          >
            {laneSpans(bundle, object.local_id, targets, proposals).map((span, index) => {
              const problem = span.kind === "problem";
              return (
                <div
                  key={index}
                  data-kind={span.kind}
                  className="absolute rounded-sm"
                  title={`${humanize(span.kind)} · ${formatTime(span.start)}–${formatTime(span.end)}`}
                  style={{
                    left: percent(span.start),
                    width: `max(2px, ${(span.end - span.start) / duration * 100}%)`,
                    top: problem ? 0 : span.kind === "stored" ? 7 : 3,
                    bottom: problem ? 0 : span.kind === "stored" ? 7 : 3,
                    background: problem ? "#b7791f55" : LANE_COLORS[span.kind],
                  }}
                />
              );
            })}
            {drafts.filter((item) => item.objectId === object.local_id).map((item) => (
              <div key={item.id} data-kind="seed" className="pointer-events-none absolute inset-y-0 w-[3px] bg-white" style={{ left: percent(item.timestamp) }} />
            ))}
            <div className="pointer-events-none absolute inset-y-0 w-px bg-[var(--teal)]" style={{ left: percent(time) }} />
          </div>
        </div>
      ))}
      <div className="flex flex-wrap gap-3 pt-1 text-[11px] text-[var(--muted)]">
        <Legend color="#465066" label="Stored boxes" />
        <Legend color="#b7791f" label="Problem" />
        <Legend color="#ffffff" label="Your seed" />
        <Legend color={LANE_COLORS.supported} label="Supported" />
        <Legend color={LANE_COLORS.provisional} label="Provisional" />
        <Legend color={LANE_COLORS.conflicting} label="Conflicting" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: color }} />{label}</span>;
}

function SeedList({
  drafts,
  seconds,
  direction,
  onSeconds,
  onDirection,
  onSeek,
  onChange,
  children,
}: {
  drafts: SeedDraft[];
  seconds: number;
  direction: TrackDirection;
  onSeconds: (value: number) => void;
  onDirection: (value: TrackDirection) => void;
  onSeek: (time: number) => void;
  onChange: (drafts: SeedDraft[]) => void;
  children: ReactNode;
}) {
  return (
    <section className="panel space-y-2 p-4 text-sm" data-testid="seed-list">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-base">Seeds ({drafts.length})</h2>
        {drafts.length > 0 && <button className="btn" onClick={() => onChange([])}>Clear all</button>}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
        <span>New seeds track</span>
        <select aria-label="Track direction" className="field w-auto" value={direction} onChange={(event) => onDirection(event.target.value as TrackDirection)}>
          <option value="forward">forward</option>
          <option value="backward">backward</option>
          <option value="both">both ways</option>
        </select>
        <span>for</span>
        <select aria-label="Track seconds" className="field w-auto" value={seconds} onChange={(event) => onSeconds(Number(event.target.value))}>
          {[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}s</option>)}
        </select>
      </div>
      {drafts.length === 0 ? (
        <p className="text-xs text-[var(--muted)]">No seeds yet. Click a correct box on the frame, or choose an object and drag around it.</p>
      ) : (
        <ul className="space-y-1">
          {drafts.map((draft) => (
            <li key={draft.id} className="flex items-center gap-2 rounded border border-[var(--line)] p-2 text-xs">
              <button className="min-w-0 flex-1 text-left" onClick={() => onSeek(draft.timestamp)}>
                <span className="font-medium">{humanize(draft.objectId)}</span> · {draft.timestamp.toFixed(2)}s
                <span className="block text-[var(--muted)]">
                  {draft.origin} · tracks {draft.start.toFixed(2)}–{draft.end.toFixed(2)}s{draft.direction === "auto" ? "" : `, ${draft.direction}`}{draft.points.length ? ` · ${draft.points.length} points` : ""}
                </span>
              </button>
              <button className="btn shrink-0 !p-0" style={SQUARE_BUTTON} aria-label={`Remove seed for ${humanize(draft.objectId)} at ${draft.timestamp.toFixed(2)}s`} onClick={() => onChange(drafts.filter((item) => item.id !== draft.id))}><X size={14} strokeWidth={1.75} /></button>
            </li>
          ))}
        </ul>
      )}
      {children}
    </section>
  );
}

function RunPanel({
  drafts,
  data,
  plan,
  pending,
  error,
  onPreview,
  onLive,
  onReplay,
  onCancel,
}: {
  drafts: SeedDraft[];
  data: RecoveryExperiment | null;
  plan: RecoveryExperiment["plan"];
  pending: boolean;
  error: string;
  onPreview: () => void;
  onLive: (hash: string, keys?: string[]) => void;
  onReplay: () => void;
  onCancel: () => void;
}) {
  const running = data?.status === "running";
  const uncertain = (data?.requests ?? []).filter((item) => item.submission === "submitted" && item.status !== "running").map((item) => item.cache_key);
  const remote = plan?.totals.remote_requests ?? 0;
  const cached = plan?.totals.cached_requests ?? 0;
  const canRun = Boolean(plan && remote > 0 && plan.violations.length === 0);
  return (
    <div className="space-y-2 border-t border-[var(--line)] pt-3" data-testid="recovery-plan">
      {running ? (
        <>
          <p>SAM is running. Cutting the clip and uploading can take a minute.</p>
          <button className="btn" disabled={data?.cancel_requested} onClick={onCancel}>{data?.cancel_requested ? "Canceling…" : "Cancel"}</button>
        </>
      ) : plan ? (
        <>
          <p className="text-xs">{usage(plan.totals)}{cached ? `, ${cached} already stored` : ""}.</p>
          {plan.totals.estimated_cost_usd == null && <p className="text-xs text-[var(--muted)]">No per-frame rate is set, so the cost is unknown. Frame limits still apply.</p>}
          {plan.blocked.map((item) => (
            <p key={`${item.target_id}-${item.reason}`} className="text-xs text-[var(--amber)]">{humanize(item.object_local_id)}: {item.reason}</p>
          ))}
          {plan.violations.map((item) => <p key={item} className="text-xs text-[var(--rose)]">{item}</p>)}
          <div className="flex flex-wrap gap-2">
            {canRun && <button className="btn primary" disabled={pending} onClick={() => onLive(plan.plan_hash)}>Run SAM: {remote} {remote === 1 ? "request" : "requests"}, about {plan.totals.billed_frames} frames</button>}
            {cached > 0 && <button className="btn" disabled={pending} onClick={onReplay}>Use stored responses (free)</button>}
            {canRun && uncertain.length > 0 && <button className="btn" disabled={pending} onClick={() => onLive(plan.plan_hash, uncertain)}>Also resend {uncertain.length} that may have been billed</button>}
          </div>
          <p className="text-xs text-[var(--muted)]">Run SAM may be billed by Meta. It sends only these SAM requests.</p>
          <details className="text-xs">
            <summary className="cursor-pointer text-[var(--muted)]">Request details</summary>
            <ul className="mt-1 space-y-1">
              {plan.requests.map((spec) => (
                <li key={spec.key} className="rounded border border-[var(--line)] p-2">
                  {humanize(spec.direction)} · {spec.source_start.toFixed(2)}–{spec.source_end.toFixed(2)}s · {spec.frame_times.length} frames · {spec.objects.length} seeds
                  {spec.objects.length > 0 && (
                    <ul className="mt-1" data-testid="request-prompts">
                      {spec.objects.map((entry) => (
                        <li key={`${entry.target_id}-${entry.seed.seed_id}`}>
                          {PROMPT_ROLE_LABELS[entry.prompt_role ?? ""] ?? "Seed"} · {humanize(entry.object_local_id)} · SAM id {entry.sam_object_id} · clip frame {entry.seed_frame_index} · {entry.seed.timestamp.toFixed(2)}s
                        </li>
                      ))}
                    </ul>
                  )}
                  <code className="mono mt-1 block break-all text-[var(--muted)]">{spec.prompt}</code>
                </li>
              ))}
            </ul>
          </details>
        </>
      ) : (
        <button className="btn primary" disabled={pending || drafts.length === 0} onClick={onPreview}>
          {pending ? "Checking…" : `Check ${drafts.length || ""} ${drafts.length === 1 ? "seed" : "seeds"} (free)`}
        </button>
      )}
      {!running && !plan && Boolean(data?.plan?.requests.length) && (
        <button className="btn" disabled={pending} onClick={onReplay}>Replay this run's stored responses (free)</button>
      )}
      {data?.error && (
        <p className="text-xs text-[var(--amber)]" role="status">
          {data.error} {data.remote_submitted ? "A remote call was submitted for this run." : "No remote call was submitted."}
        </p>
      )}
      {error && <p className="text-xs text-[var(--rose)]" role="alert">{error}</p>}
    </div>
  );
}

const STAT_LABELS: [string, string][] = [
  ["raw_frames_returned", "SAM boxes returned"],
  ["associated", "Associated"],
  ["supported", "Supported"],
  ["provisional", "Provisional"],
  ["conflicting", "Conflicting"],
  ["rejected", "Rejected"],
  ["no_detection_frames", "No SAM box"],
  ["unusable_frames", "Returned, not trusted"],
  ["track_breaks", "Track breaks"],
  ["reacquisition_attempts", "Reacquisition attempts"],
  ["reacquisitions_succeeded", "Succeeded"],
  ["reacquisitions_failed", "Failed"],
  ["identity_ambiguities", "Identity questions"],
  ["api_requests", "API requests"],
];

function RecoveryBreakdown({ item, onSeek }: { item: ReacquisitionPlanItem; onSeek: (time: number) => void }) {
  const { diagnosis, search, workflow } = item;
  const seekable = (label: string, value: number | null) => (
    <div className="flex justify-between gap-2">
      <dt className="text-[var(--muted)]">{label}</dt>
      <dd className="mono">{value == null ? "none" : <button className="underline" onClick={() => onSeek(value)}>{value.toFixed(2)}s</button>}</dd>
    </div>
  );
  return (
    <div className="space-y-1" data-testid="recovery-breakdown">
      <ol className="flex flex-wrap gap-x-3 gap-y-1" data-testid="reacquisition-workflow">
        {workflow.map((stage) => (
          <li key={stage.stage} title={stage.detail} data-status={stage.status}>
            <span className="font-medium">{WORKFLOW_LABELS[stage.stage] ?? humanize(stage.stage)}</span> <span className="text-[var(--muted)]">{humanize(stage.status)}</span>
          </li>
        ))}
      </ol>
      {diagnosis.categories.map((category) => <p key={category.code} data-testid="failure-category">{category.detail}</p>)}
      <details>
        <summary className="cursor-pointer text-[var(--muted)]">Where the boxes went</summary>
        <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
          {seekable("Last raw SAM box", diagnosis.last_raw_time)}
          {seekable("Last supported box", diagnosis.last_supported_time)}
          {seekable("First frame with no box", diagnosis.first_no_detection_time)}
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Boxes after the failure</dt><dd className="mono">{diagnosis.raw_boxes_after_failure}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Rejected by validation</dt><dd className="mono">{diagnosis.rejected_by_validation}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Later SAM ids</dt><dd className="mono">{diagnosis.later_track_ids.join(", ") || "none"}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Found, not identified</dt><dd className="mono">{diagnosis.detected_not_identified}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Frames searched</dt><dd className="mono">{search.frames_examined}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">New request planned</dt><dd>{diagnosis.request_planned ? "yes" : "no"}</dd></div>
          <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">New request run</dt><dd>{diagnosis.request_executed ? "yes" : "no"}</dd></div>
        </dl>
        {search.candidates.length === 0 && <p className="mt-1">Nothing stored shows this object between {search.window.start.toFixed(2)}s and {search.window.end.toFixed(2)}s.</p>}
      </details>
    </div>
  );
}

function TrackHistory({
  objects,
  note,
  duration,
  time,
  drawingFor,
  busy,
  onSeek,
  onUseEvidence,
  onConcept,
  onCandidate,
  onCompare,
  onDraw,
}: {
  objects: ObjectTrackHistory[];
  note: string;
  duration: number;
  time: number;
  drawingFor: string;
  busy: boolean;
  onSeek: (time: number) => void;
  onUseEvidence: (objectId: string, item: ReacquisitionPlanItem, strategy: string, evidence: ReacquisitionEvidence) => void;
  onConcept: (objectId: string, item: ReacquisitionPlanItem, concept: string) => void;
  onCandidate: (objectId: string, item: ReacquisitionPlanItem, candidate: ReappearanceCandidate) => void;
  onCompare: () => void;
  onDraw: (objectId: string, item: ReacquisitionPlanItem) => void;
}) {
  const percent = (value: number) => `${Math.min(Math.max(value / duration, 0), 1) * 100}%`;
  return (
    <section className="panel space-y-3 p-4 text-sm" data-testid="track-history">
      <div>
        <h2 className="text-base">Track history</h2>
        <p className="text-xs text-[var(--muted)]">{note}</p>
      </div>
      {objects.map((object) => (
        <div key={object.object_local_id} className="space-y-2 border-t border-[var(--line)] pt-2 text-xs" data-testid={`history-${object.object_local_id}`}>
          <p className="font-medium">{humanize(object.object_local_id)} · {object.label}</p>
          <div className="relative h-7 overflow-hidden rounded bg-[#0d1016]" data-testid={`history-lane-${object.object_local_id}`}>
            {object.unresolved.map((span) => (
              <div
                key={`gap-${span.start}`}
                data-kind="unresolved"
                className="absolute inset-y-0 bg-[repeating-linear-gradient(135deg,#3a2f1c_0_4px,#0d1016_4px_8px)]"
                title={`Unresolved · ${formatTime(span.start)}–${formatTime(span.end)} · ${span.no_detection_frames} with no SAM box · ${span.unusable_frames} not trusted`}
                style={{ left: percent(span.start), width: `max(2px, ${(span.end - span.start) / duration * 100}%)` }}
              />
            ))}
            {object.segments.map((segment) => (
              <button
                key={segment.segment_id}
                data-kind={segment.status}
                className="absolute rounded-sm"
                aria-label={`${humanize(segment.status)} segment, ${segment.start.toFixed(2)} to ${segment.end.toFixed(2)} seconds`}
                title={`${humanize(segment.status)} · ${segment.identity.detail}`}
                style={{
                  top: 6,
                  bottom: 6,
                  left: percent(segment.start),
                  width: `max(2px, ${(segment.end - segment.start) / duration * 100}%)`,
                  background: SEGMENT_COLORS[segment.status] ?? "#888",
                }}
                onClick={() => onSeek(segment.start)}
              />
            ))}
            {object.failures.map((failure) => (
              <button
                key={failure.failure_id}
                data-kind="failure"
                className="absolute inset-y-0 w-[3px] bg-[var(--rose)]"
                aria-label={`Go to the failure at ${failure.time.toFixed(2)} seconds`}
                title={`${failureLabel(failure.reason)} at ${formatTime(failure.time)}`}
                style={{ left: percent(failure.time) }}
                onClick={() => onSeek(failure.time)}
              />
            ))}
            {object.plans.flatMap((item) => item.search?.candidates ?? []).map((candidate) => (
              <button
                key={candidate.candidate_id}
                data-kind={candidate.outcome}
                className="absolute bottom-0 h-2 w-2 -translate-x-1/2 rotate-45"
                aria-label={`${humanize(candidate.outcome)} at ${candidate.timestamp.toFixed(2)} seconds`}
                title={`${humanize(candidate.outcome)} · ${candidate.reason}`}
                style={{ left: percent(candidate.timestamp), background: CANDIDATE_COLORS[candidate.outcome] }}
                onClick={() => onSeek(candidate.timestamp)}
              />
            ))}
            <div className="pointer-events-none absolute inset-y-0 w-px bg-[var(--teal)]" style={{ left: percent(time) }} />
          </div>
          {object.coverage && object.coverage.analyzed_through < object.coverage.video_end - 0.05 && (
            <p className="text-[var(--muted)]" data-testid="coverage-note">
              SAM was asked about this object through {object.coverage.analyzed_through.toFixed(2)}s. The video runs to {object.coverage.video_end.toFixed(2)}s, so a missing box after that is not a SAM result.
            </p>
          )}
          {object.evidence_limits.map((item) => <p key={item} className="text-[var(--muted)]">{item}</p>)}
          {object.plans.length === 0 && object.segments.length > 0 && <p className="text-[var(--muted)]">No track break was found in these results.</p>}
          {object.plans.map((item) => {
            const strategy = item.strategies.find((entry) => entry.strategy === item.recommended);
            const concept = item.strategies.find((entry) => entry.strategy === "concept_redetection" && entry.available);
            return (
              <div key={item.failure.failure_id} className="space-y-1 rounded border border-[var(--line)] p-2" data-testid="reacquisition-plan">
                <p className="font-medium" style={{ color: item.status === "resolved" ? "#34d399" : "var(--amber)" }}>
                  {failureLabel(item.failure.reason)} at {item.failure.time.toFixed(2)}s · {item.status === "resolved" ? "reacquired" : "unresolved"}
                </p>
                <p>{item.explanation}</p>
                <p className="text-[var(--muted)]">{item.failure.detail} {GAP_STATE_LABELS[item.failure.state] ?? humanize(item.failure.state)}. {item.failure.state_basis}</p>
                {item.diagnosis && <RecoveryBreakdown item={item} onSeek={onSeek} />}
                <div className="flex flex-wrap gap-1">
                  <button className="btn" onClick={() => onSeek(item.failure.time)}>Go to failure</button>
                  {item.last_supported_time != null && <button className="btn" onClick={() => onSeek(item.last_supported_time as number)}>Last supported {item.last_supported_time.toFixed(2)}s</button>}
                  <button className="btn" onClick={onCompare}>Compare all SAM boxes</button>
                </div>
                {item.status === "open" && (
                  <>
                    {item.search?.candidates.length > 0 && (
                      <ul className="space-y-1" data-testid="reappearance-candidates">
                        <li className="text-[var(--muted)]">Possible reappearances. Identity is checked separately from detection.</li>
                        {item.search.candidates.map((candidate) => (
                          <li key={candidate.candidate_id} className="rounded border border-[var(--line)] p-1" data-testid="reappearance-candidate">
                            <div className="flex flex-wrap items-center gap-2">
                              <button className="mono underline" onClick={() => onSeek(candidate.timestamp)}>{candidate.timestamp.toFixed(2)}s</button>
                              <span className="font-medium" style={{ color: CANDIDATE_COLORS[candidate.outcome] }}>{humanize(candidate.outcome)}</span>
                              <span className="min-w-0 flex-1 text-[var(--muted)]">
                                {humanize(candidate.source)}{candidate.raw_track_id ? ` · SAM track ${candidate.raw_track_id}` : ""}
                              </span>
                              {candidate.outcome !== "rejected_candidate" && (
                                <button className="btn" disabled={busy} onClick={() => onCandidate(object.object_local_id, item, candidate)}>
                                  {candidate.needs_confirmation ? "This is the object: plan new track (free)" : "Plan new track (free)"}
                                </button>
                              )}
                            </div>
                            <p>{candidate.reason}</p>
                            <details>
                              <summary className="cursor-pointer text-[var(--muted)]">Checks</summary>
                              <ul>{candidate.checks.map((check) => <li key={check.check}>{check.passed === null ? "–" : check.passed ? "✓" : "✕"} {humanize(check.check)}: {check.detail}</li>)}</ul>
                            </details>
                          </li>
                        ))}
                      </ul>
                    )}
                    {item.search?.window.note && <p className="text-[var(--amber)]">{item.search.window.note}</p>}
                    {item.reappearance_frames.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1">
                        <span className="text-[var(--muted)]">Frames to check:</span>
                        {item.reappearance_frames.map((frame) => (
                          <button key={frame} className="mono underline" onClick={() => onSeek(frame)}>{frame.toFixed(2)}s</button>
                        ))}
                      </div>
                    )}
                    {strategy && strategy.evidence.length > 0 && (
                      <ul className="space-y-1">
                        <li className="text-[var(--muted)]">{strategy.note}</li>
                        {strategy.evidence.map((evidence, index) => (
                          <li key={`${evidence.seed_id}-${evidence.timestamp}-${index}`} className="flex flex-wrap items-center gap-2">
                            <button className="mono underline" onClick={() => onSeek(evidence.timestamp)}>{evidence.timestamp.toFixed(2)}s</button>
                            <span className="min-w-0 flex-1">
                              {humanize(evidence.source)} · {humanize(evidence.trust)}
                              {evidence.may_be.length > 0 && ` · also overlaps ${evidence.may_be.map(humanize).join(", ")}`}
                            </span>
                            <button className="btn" onClick={() => onUseEvidence(object.object_local_id, item, strategy.strategy, evidence)}>
                              {evidence.needs_confirmation ? "This is the object" : "Use as seed"}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    <div className="flex flex-wrap gap-1">
                      <button className="btn" aria-pressed={drawingFor === object.object_local_id} onClick={() => onDraw(object.object_local_id, item)}>Draw a new seed</button>
                      {concept && (
                        <button className="btn" disabled={busy} title={concept.note} onClick={() => onConcept(object.object_local_id, item, concept.concept ?? object.label)}>
                          Plan a concept search (free to check)
                        </button>
                      )}
                    </div>
                  </>
                )}
              </div>
            );
          })}
          {object.segments.length > 0 && (
            <details>
              <summary className="cursor-pointer text-[var(--muted)]">Segments ({object.segments.length})</summary>
              <ul className="mt-1 space-y-1">
                {object.segments.map((segment) => (
                  <li key={segment.segment_id} className="rounded border border-[var(--line)] p-2">
                    <button className="font-medium" onClick={() => onSeek(segment.start)}>
                      <span style={{ color: SEGMENT_COLORS[segment.status] }}>{humanize(segment.status)}</span> · {segment.start.toFixed(2)}–{segment.end.toFixed(2)}s · {segment.direction}
                    </button>
                    <div>{segment.frames} SAM boxes · {segment.counts.supported} supported · {segment.counts.provisional} provisional · {segment.counts.conflicting} conflicting · {segment.counts.rejected} rejected</div>
                    <div className="text-[var(--muted)]">
                      Started from {humanize(segment.init_event)} · SAM track {segment.raw_track_id} in {segment.session_id} · identity {humanize(segment.identity.status)}. {segment.identity.detail}
                    </div>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <details>
            <summary className="cursor-pointer text-[var(--muted)]">Statistics</summary>
            <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
              {STAT_LABELS.map(([key, label]) => (
                <div key={key} className="flex justify-between gap-2"><dt className="text-[var(--muted)]">{label}</dt><dd className="mono">{object.stats[key] ?? 0}</dd></div>
              ))}
              <div className="flex justify-between gap-2"><dt className="text-[var(--muted)]">Recorded cost</dt><dd className="mono">{object.stats.recorded_cost_usd == null ? "unknown" : `$${object.stats.recorded_cost_usd}`}</dd></div>
            </dl>
          </details>
        </div>
      ))}
    </section>
  );
}

const VERDICT_COLORS: Record<string, string> = {
  boxes_returned: "var(--muted)",
  not_run: "var(--muted)",
};

function SamDiagnostics({ data, onSeek }: { data: RecoveryDiagnostics; onSeek: (time: number) => void }) {
  return (
    <section className="panel space-y-2 p-4 text-sm" data-testid="sam-diagnostics">
      <h2 className="text-base">SAM requests</h2>
      <p className="text-xs text-[var(--muted)]">{data.note} Turn on the All SAM output layer to see every box on the frame.</p>
      {data.requests.map((request) => {
        const funnel = request.funnel;
        const steps: [string, number][] = [
          ["Returned", funnel.box_tokens],
          ["Parsed", funnel.parsed],
          ["Usable", funnel.usable],
          ["Unassigned", funnel.unassigned],
          ["Assigned", funnel.assigned],
          ["Supported", funnel.supported],
          ["Provisional", funnel.provisional],
          ["Conflicting", funnel.conflicting],
          ["Rejected", funnel.rejected],
        ];
        return (
          <div key={request.key} className="space-y-2 rounded border border-[var(--line)] p-2 text-xs" data-testid="sam-request">
            <p className="font-medium" style={{ color: VERDICT_COLORS[request.verdict] ?? "var(--amber)" }}>{request.message}</p>
            {request.synthetic && <p className="text-[var(--amber)]">This response is synthetic.</p>}
            <p className="text-[var(--muted)]">
              {request.frames_with_boxes} of {request.reachable_frames} reachable frames have a box · {request.direction} · {request.planned_frames} planned frames
              {request.request.source ? ` · ${request.request.source}` : ""}
              {request.request.latency_ms ? ` · ${(request.request.latency_ms / 1000).toFixed(1)}s` : ""}
              {request.request.provider_request_id ? ` · ${request.request.provider_request_id}` : ""}
            </p>
            <dl className="flex flex-wrap gap-x-4 gap-y-1">
              {steps.map(([label, value]) => (
                <div key={label}><dt className="text-[var(--muted)]">{label}</dt><dd className="mono">{value}</dd></div>
              ))}
            </dl>
            {request.tracks.length > 0 && (
              <ul className="space-y-1">
                {request.tracks.map((track) => (
                  <li key={track.raw_track_id}>
                    SAM track {track.raw_track_id}
                    {" · "}{track.object_local_id ? humanize(track.object_local_id) : "no object"}
                    {" · "}{humanize(track.association)}
                    {" · "}{track.boxes} boxes
                    {track.first_time != null && track.last_time != null && ` · ${track.first_time.toFixed(2)}–${track.last_time.toFixed(2)}s`}
                    {track.mean_confidence != null && ` · score ${track.first_confidence?.toFixed(2)} first, ${track.mean_confidence.toFixed(2)} mean, ${track.last_confidence?.toFixed(2)} last (uncalibrated)`}
                  </li>
                ))}
              </ul>
            )}
            {request.findings.length > 0 && (
              <ul className="space-y-1">
                {request.findings.map((finding, index) => (
                  <li key={index} style={{ color: finding.severity === "error" ? "var(--rose)" : "var(--amber)" }}>{finding.message}</li>
                ))}
              </ul>
            )}
            {request.tracks.flatMap((track) => (track.broke ? [track.broke] : [])).map((broke) => (
              <button key={broke.timestamp} className="btn" onClick={() => onSeek(broke.timestamp)}>Go to the break at {broke.timestamp.toFixed(2)}s</button>
            ))}
            {request.missing_frames.slice(0, 3).map((gap) => (
              <button key={gap.start} className="block text-[var(--muted)]" onClick={() => onSeek(gap.start)}>{gap.start.toFixed(2)}–{gap.end.toFixed(2)}s has no box from SAM</button>
            ))}
            {Object.keys(request.failed_checks).length > 0 && (
              <p className="text-[var(--muted)]">
                Failed checks: {Object.entries(request.failed_checks).map(([name, count]) => `${humanize(name)} ${count}`).join(" · ")}
              </p>
            )}
            {request.parsed.unparsed_samples.length > 0 && (
              <details>
                <summary className="cursor-pointer text-[var(--muted)]">Lines the parser skipped</summary>
                <pre className="mono mt-1 overflow-x-auto">{request.parsed.unparsed_samples.join("\n")}</pre>
              </details>
            )}
            <details>
              <summary className="cursor-pointer text-[var(--muted)]">Prompt</summary>
              <pre className="mono mt-1 overflow-x-auto">{request.prompt}</pre>
            </details>
          </div>
        );
      })}
    </section>
  );
}

function ResultsPanel({
  data,
  proposals,
  provisional,
  onProvisional,
  pending,
  onDecide,
  onSeek,
}: {
  data: RecoveryExperiment | null;
  proposals: RecoveryProposal[];
  provisional: boolean;
  onProvisional: (value: boolean) => void;
  pending: boolean;
  onDecide: (decision: "accepted" | "rejected") => void;
  onSeek: (time: number) => void;
}) {
  if (!data || !proposals.length) return null;
  const open = proposals.filter((item) => item.decision === "pending");
  return (
    <section className="panel space-y-2 p-4 text-sm" data-testid="recovery-results">
      <h2 className="text-base">Results</h2>
      {proposals.some((item) => item.metrics.synthetic_response) && (
        <p className="text-xs text-[var(--amber)]">Built from a synthetic offline response. It only exercises the plumbing and says nothing about SAM's accuracy.</p>
      )}
      <ul className="space-y-1 text-xs">
        {proposals.map((proposal) => {
          const counts = proposal.metrics.counts;
          return (
            <li key={proposal.id} className="rounded border border-[var(--line)] p-2">
              <button className="font-medium" onClick={() => onSeek(proposal.time_start)}>{humanize(proposal.object_local_id)} · {proposal.time_start.toFixed(2)}–{proposal.time_end.toFixed(2)}s</button>
              {proposal.decision !== "pending" && ` · ${humanize(proposal.decision)}`}
              <div>
                <span style={{ color: LANE_COLORS.supported }}>{counts.supported} supported</span>
                {" · "}<span style={{ color: LANE_COLORS.provisional }}>{counts.provisional} provisional</span>
                {" · "}<span style={{ color: LANE_COLORS.conflicting }}>{counts.conflicting} conflicting</span>
                {" · "}{counts.rejected} rejected
                {counts.no_detection_frames == null ? ` · ${counts.unobserved_frames} with no usable box` : ` · ${counts.no_detection_frames} with no SAM box · ${counts.unusable_frames} returned and not trusted`}
              </div>
              {proposal.metrics.remaining_gaps.slice(0, 3).map((gap) => (
                <button key={gap.start} className="block text-[var(--muted)]" onClick={() => onSeek(gap.start)}>{gap.start.toFixed(2)}–{gap.end.toFixed(2)}s has no box. That is not evidence of absence.</button>
              ))}
            </li>
          );
        })}
      </ul>
      {open.length > 0 && (
        <>
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={provisional} onChange={(event) => onProvisional(event.target.checked)} />
            Include provisional boxes. They stay non-canonical.
          </label>
          <div className="flex flex-wrap gap-2">
            <button className="btn primary" disabled={pending} onClick={() => onDecide("accepted")}>Accept as a new revision</button>
            <button className="btn" disabled={pending} onClick={() => onDecide("rejected")}>Reject all</button>
          </div>
        </>
      )}
      <p className="text-xs text-[var(--muted)]">{proposals[0].metrics.accuracy_note}</p>
    </section>
  );
}

function ProblemList({
  targets,
  dismissed,
  pending,
  busy,
  error,
  onSeek,
  onObject,
  onSuggested,
  onDismiss,
  onRestore,
}: {
  targets: RecoveryTarget[];
  dismissed: RecoveryTarget[];
  pending: boolean;
  busy: boolean;
  error: string;
  onSeek: (time: number) => void;
  onObject: (id: string) => void;
  onSuggested: (target: RecoveryTarget) => void;
  onDismiss: (targetId: string) => void;
  onRestore: (targetId: string) => void;
}) {
  const [showDismissed, setShowDismissed] = useState(false);
  const describe = (target: RecoveryTarget) => `${humanize(target.object_local_id)} at ${target.time_start.toFixed(2)}s`;
  return (
    <details className="panel p-3 text-sm" data-testid="recovery-candidates">
      <summary className="cursor-pointer">Suspected problems ({pending ? <Bone width="2ch" /> : targets.length})</summary>
      <ul className="mt-2 max-h-72 space-y-1 overflow-auto">
        {targets.map((target) => (
          <li key={target.target_id} className="flex items-center gap-2 text-xs">
            <button
              className="min-w-0 flex-1 text-left"
              onClick={() => {
                onObject(target.object_local_id);
                onSeek(target.time_start);
              }}
            >
              <span className="font-medium">{humanize(target.object_local_id)}</span> · {target.time_start.toFixed(2)}–{target.time_end.toFixed(2)}s
              <span className="block text-[var(--muted)]">{target.reasons.map(humanize).join(", ")}</span>
            </button>
            {target.recoverable && (target.seeds?.length ?? 0) > 0 && <button className="btn" onClick={() => onSuggested(target)}>Use suggested seed</button>}
            <button className="btn shrink-0 !p-0" style={SQUARE_BUTTON} disabled={busy} title="Not a problem" aria-label={`Dismiss problem for ${describe(target)}`} onClick={() => onDismiss(target.target_id)}>
              <X size={14} strokeWidth={1.75} />
            </button>
          </li>
        ))}
      </ul>
      {!pending && targets.length === 0 && <p className="mt-2 text-xs text-[var(--muted)]">Nothing is flagged. That does not mean every box is correct.</p>}
      {dismissed.length > 0 && (
        <div className="mt-2 space-y-1 text-xs">
          <button className="text-[var(--muted)] underline" aria-expanded={showDismissed} onClick={() => setShowDismissed(!showDismissed)}>
            {showDismissed ? "Hide" : "Show"} dismissed ({dismissed.length})
          </button>
          {showDismissed && (
            <ul className="space-y-1">
              {dismissed.map((target) => (
                <li key={target.target_id} className="flex items-center gap-2 text-[var(--muted)]">
                  <span className="min-w-0 flex-1">
                    {humanize(target.object_local_id)} · {target.time_start.toFixed(2)}–{target.time_end.toFixed(2)}s · {target.reasons.map(humanize).join(", ")}
                  </span>
                  <button className="btn" disabled={busy} aria-label={`Restore problem for ${describe(target)}`} onClick={() => onRestore(target.target_id)}>Restore</button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {error && <p className="mt-2 text-xs text-[var(--rose)]" role="alert">{error}</p>}
    </details>
  );
}
