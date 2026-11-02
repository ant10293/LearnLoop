import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, type CallPlan, type Recording, type Run } from "../lib/learnloop";
import { ErrorNote } from "./ui";

const STAGES: [string, string][] = [
  ["metadata", "Read the recording"],
  ["sample_frames", "Prepare timestamps"],
  ["scene_analysis", "Meta analysis"],
  ["validation", "Check the structured answer"],
];

export function AnalysisStages({ run }: { run: Run }) {
  return (
    <ol className="stages" aria-label="Analysis progress">
      {STAGES.map(([key, label]) => (
        <li key={key} className={run.stages[key] ?? "pending"}>{label}</li>
      ))}
    </ol>
  );
}

/** Prepare a recording, then run one analysis. A billable call waits for an explicit yes. */
export function RecordingAnalysis({ recording, onDone, action }: { recording: Recording; onDone: (run: Run) => void | Promise<void>; action: string }) {
  const videoId = recording.video.id;
  const [runId, setRunId] = useState(recording.run?.status === "running" || recording.run?.status === "pending" ? recording.run.id : "");
  const [plan, setPlan] = useState<CallPlan | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const finished = useRef("");
  const live = useQuery({
    queryKey: ["recording", videoId],
    queryFn: () => api.procedureRecording(videoId),
    refetchInterval: (query) => (query.state.data?.video.ingest_status === "ready" || query.state.data?.video.ingest_status === "failed" ? false : 1000),
    initialData: recording,
  });
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) => (query.state.data && ["completed", "failed", "canceled"].includes(query.state.data.status) ? false : 800),
  });
  const video = live.data?.video ?? recording.video;

  useEffect(() => {
    const data = run.data;
    if (data?.status === "completed" && finished.current !== data.id) {
      finished.current = data.id;
      Promise.resolve(onDone(data)).catch((exc: Error) => setError(exc.message));
    }
  }, [run.data, onDone]);

  async function start(allowLive: boolean) {
    setError("");
    setBusy(true);
    try {
      const created = await api.analyze(videoId, allowLive);
      setPlan(null);
      setRunId(created.id);
    } catch (exc) {
      const detail = exc instanceof ApiError ? (exc.detail as { code?: string; plan?: CallPlan }) : undefined;
      if (detail?.code === "live_call_required" && detail.plan) setPlan(detail.plan);
      else setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const status = run.data?.status;
  const running = status === "pending" || status === "running";
  if (video.ingest_status === "failed") return <ErrorNote message={video.ingest_error || "This recording could not be read."} />;
  if (video.ingest_status && video.ingest_status !== "ready") {
    return <p className="status-line" role="status">Preparing the recording. {video.ingest_detail}</p>;
  }
  return (
    <div className="analysis">
      {!running && status !== "completed" && !plan && (
        <button className="btn primary" disabled={busy} onClick={() => void start(false)}>
          {status === "failed" ? "Analyze again" : action}
        </button>
      )}
      {plan && <PlanCard plan={plan} busy={busy} onApprove={() => void start(true)} onCancel={() => setPlan(null)} />}
      {run.data && (
        <AnalysisStages run={run.data} />
      )}
      {status === "failed" && (
        <div className="error" role="alert">
          <strong>The analysis failed. No result was generated.</strong>
          <p>{run.data?.error}</p>
          <p className="small">Starting again asks for a new approval when it would send another request.</p>
        </div>
      )}
      {status === "completed" && run.data && (
        <p className="status-line" role="status">
          Analysis complete with {run.data.model_id}. {run.data.source === "cached" ? "Replayed from a stored Meta response for this exact video. No request was sent." : "Live Meta response, stored for replay."}
        </p>
      )}
      <ErrorNote message={error} />
    </div>
  );
}

function PlanCard({ plan, busy, onApprove, onCancel }: { plan: CallPlan; busy: boolean; onApprove: () => void; onCancel: () => void }) {
  return (
    <div className="plan" role="dialog" aria-label="Approve Meta request">
      <h3>This needs one Meta request</h3>
      <p>No stored response matches this exact video, model, and prompt.</p>
      <dl>
        <dt>Model</dt><dd>{plan.model_id}, {plan.modality === "native_video" ? "video file" : "frames"}</dd>
        <dt>Recording</dt><dd>{plan.video_seconds.toFixed(1)} seconds</dd>
        <dt>Requests</dt><dd>At most {plan.max_requests}, with no automatic retry</dd>
        <dt>Output limit</dt><dd>{plan.max_output_tokens.toLocaleString()} tokens</dd>
        <dt>Estimated cost</dt><dd>{plan.estimated_cost_usd == null ? "Unknown. No rate is recorded for this model." : `$${plan.estimated_cost_usd}`}</dd>
      </dl>
      {plan.live_enabled ? (
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={onApprove}>Send one request</button>
          <button className="btn" onClick={onCancel}>Cancel</button>
        </div>
      ) : (
        <>
          <p className="muted">Live calls are off on this server. Set <code>LEARNLOOP_LIVE_INFERENCE=1</code> and restart to allow them. Nothing was sent.</p>
          <button className="btn" onClick={onCancel}>Close</button>
        </>
      )}
    </div>
  );
}
