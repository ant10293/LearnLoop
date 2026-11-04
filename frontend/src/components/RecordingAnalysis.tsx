import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ApiError, api, humanizeModel, type CallPlan, type ModelOption, type ObjectViews, type ObjectViewsPlan, type Recording, type Run } from "../lib/learnloop";
import { FieldBone } from "./Skeleton";
import { ErrorNote } from "./ui";

const STAGES: [string, string][] = [
  ["metadata", "Read the recording"],
  ["sample_frames", "Prepare timestamps"],
  ["scene_analysis", "Meta analysis"],
  ["validation", "Check the structured answer"],
];

export function AnalysisStages({ run }: { run?: Run }) {
  return (
    <ol className="stages" aria-label="Analysis progress">
      {STAGES.map(([key, label]) => (
        <li key={key} className={run?.stages[key] ?? "pending"}>{label}</li>
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
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  // Null keeps the server's default until the person picks.
  const [inference, setInference] = useState<string | null>(null);
  const [segmentation, setSegmentation] = useState<string | null>(null);
  // Skips stored answers for this video. The requests it sends still wait for approval.
  // Attempts are followed with SAM too, so their boxes can be checked against the procedure's references and seen.
  const segments = true;

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
      const created = await api.analyze(videoId, allowLive, { model_id: inference ?? "", segmentation_model_id: segments ? segmentation : "", fresh: Boolean(live.data?.analyzed_before) });
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
  const runLoading = Boolean(runId) && run.isPending;
  const running = runLoading || status === "pending" || status === "running";
  if (video.ingest_status === "failed") return <ErrorNote message={video.ingest_error || "This recording could not be read."} />;
  if (video.ingest_status && video.ingest_status !== "ready") {
    return (
      <div className="analysis">
        <p className="status-line" role="status">Preparing the recording. {video.ingest_detail}</p>
        <button className="btn primary" disabled>{action}</button>
      </div>
    );
  }
  return (
    <div className="analysis">
      {!running && status !== "completed" && !plan && (
        <>
          <div className="model-picker">
            <label className="stack small">
              <span className="muted">Inference model</span>
              {models.isPending ? <FieldBone select /> : (
                <ModelSelect label="Inference model" options={models.data?.inference ?? []} value={inference ?? models.data?.default_inference ?? ""} onChange={setInference} />
              )}
            </label>
            {segments && (models.isPending || Boolean(models.data?.segmentation.length)) && (
              <label className="stack small">
                <span className="muted">Segmentation model (optional)</span>
                {models.isPending ? <FieldBone select /> : (
                  <ModelSelect label="Segmentation model" options={models.data?.segmentation ?? []} value={segmentation ?? models.data?.default_segmentation ?? ""} onChange={setSegmentation} none="None" />
                )}
              </label>
            )}
          </div>
          <button className="btn primary" disabled={busy} onClick={() => void start(false)}>
            {status === "failed" ? "Analyze again" : action}
          </button>
        </>
      )}
      {plan && <PlanCard plan={plan} busy={busy} onApprove={() => void start(true)} onCancel={() => setPlan(null)} />}
      {(run.data || runLoading) && (
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
          Analysis complete with {humanizeModel(run.data.model_id)}. {run.data.source === "cached" ? "Replayed from a stored Meta response for this exact video. No request was sent." : "Live Meta response, stored for replay."}
        </p>
      )}
      <ErrorNote message={error} />
    </div>
  );
}

function ModelSelect({ label, options, value, onChange, none }: { label: string; options: ModelOption[]; value: string; onChange: (value: string) => void; none?: string }) {
  return (
    <select className="field" aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
      {none && <option value="">{none}</option>}
      {options.map((item) => <option key={item.model_id} value={item.model_id}>{item.label || humanizeModel(item.model_id)}</option>)}
    </select>
  );
}

const VIEW_STAGES = ["Find the objects on the first frame", "Find objects that appear later", "Follow each object", "Make thumbnails and timelines"];

/** Object views that were not made, or are being made again. Making them again asks first, like an analysis. */
export function ObjectViewsNotice({ runId, views, onStarted }: { runId: string; views?: ObjectViews; onStarted: () => void }) {
  const [plan, setPlan] = useState<ObjectViewsPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (!views) return null;
  if (views.running) {
    const current = views.progress ?? 0;
    return (
      <div className="panel views-notice" role="status">
        <h2>Making object views</h2>
        <ol className="stages" aria-label="Object view progress">
          {VIEW_STAGES.map((label, index) => (
            <li key={label} className={index < current ? "completed" : index === current ? "running" : "pending"}>{label}</li>
          ))}
        </ol>
      </div>
    );
  }
  if (!views.can_remake || !(views.error || Object.keys(views.objects).length === 0)) return null;

  async function start(allowLive: boolean) {
    setError("");
    setBusy(true);
    try {
      await api.remakeObjectViews(runId, allowLive);
      setPlan(null);
      onStarted();
    } catch (exc) {
      const detail = exc instanceof ApiError ? (exc.detail as { code?: string; plan?: ObjectViewsPlan }) : undefined;
      if (detail?.code === "live_call_required" && detail.plan) setPlan(detail.plan);
      else setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel views-notice">
      <p className="small">Object views were not made{views.error ? `: ${views.error}` : "."}</p>
      {!plan && <button className="btn" disabled={busy} onClick={() => void start(false)}>Run object views</button>}
      <ErrorNote message={error} />
      {plan && (
        <div className="plan" role="dialog" aria-label="Approve object view requests">
          <h3>This may need Meta requests</h3>
          <p>Stored answers are replayed. Only the requests without one are sent. The analysis is not run again.</p>
          <dl>
            <dt>Recording</dt><dd>{plan.video_seconds.toFixed(1)} seconds</dd>
            <dt>Requests</dt><dd>{`At most ${plan.object_view_locate_requests + plan.object_view_requests}: up to ${plan.object_view_locate_requests} ${humanizeModel(plan.model_id)} requests to find the objects, and up to ${plan.object_view_requests} ${humanizeModel(plan.segmentation_model_id)} requests to follow them. No automatic retry.`}</dd>
            <dt>Billed frames</dt><dd>About {plan.object_view_frames.toLocaleString()} for the first {humanizeModel(plan.segmentation_model_id)} request, and fewer for a second, which covers only the rest of the recording after an object found later or found again.</dd>
          </dl>
          {plan.live_enabled ? (
            <div className="row">
              <button className="btn primary" disabled={busy} onClick={() => void start(true)}>Send requests</button>
              <button className="btn" onClick={() => setPlan(null)}>Cancel</button>
            </div>
          ) : (
            <>
              <p className="muted">Live calls are off on this server. Set <code>LEARNLOOP_LIVE_INFERENCE=1</code> and restart to allow them. Nothing was sent.</p>
              <button className="btn" onClick={() => setPlan(null)}>Close</button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function PlanCard({ plan, busy, onApprove, onCancel }: { plan: CallPlan; busy: boolean; onApprove: () => void; onCancel: () => void }) {
  const views = plan.object_view_requests ?? 0;
  const locate = plan.object_view_locate_requests ?? 0;
  return (
    <div className="plan" role="dialog" aria-label="Approve Meta request">
      <h3>{views > 0 ? "This needs Meta requests" : "This needs a Meta request"}</h3>
      <p>{plan.fresh ? "Stored responses for this video are skipped. New ones replace them." : "No stored response matches this exact video, model, and prompt."}</p>
      <dl>
        <dt>Model</dt><dd>{humanizeModel(plan.model_id)}, {plan.modality === "native_video" ? "video file" : "frames"}</dd>
        <dt>Recording</dt><dd>{plan.video_seconds.toFixed(1)} seconds</dd>
        <dt>Requests</dt><dd>{views > 0 ? `At most ${plan.max_requests + locate + views}: ${plan.max_requests} to analyze the video, up to ${locate} to find the objects, and up to ${views} to follow them. No automatic retry.` : `At most ${plan.max_requests}, with no automatic retry`}</dd>
        {plan.object_view_requests > 0 && <><dt>Object views</dt><dd>One more Meta request to find the objects on the first frame, and a second on a few later stills only if some object is not there or is lost from view. Then one {humanizeModel(plan.segmentation_model_id)} request, about {plan.object_view_frames.toLocaleString()} billed frames, and a second over the rest of the recording only for an object found later or found again.</dd></>}
        <dt>Output limit</dt><dd>{plan.max_output_tokens.toLocaleString()} tokens{views > 0 ? " for the analysis" : ""}</dd>
        <dt>Estimated cost</dt><dd>{plan.estimated_cost_usd == null ? "Unknown. No rate is recorded for this model." : `$${plan.estimated_cost_usd}`}</dd>
      </dl>
      {plan.live_enabled ? (
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={onApprove}>{plan.object_view_requests > 0 ? "Send requests" : "Send one request"}</button>
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
