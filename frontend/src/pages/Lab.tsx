import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { RecordingDrop } from "../components/RecordingDrop";
import { ErrorNote, Layout } from "../components/ui";
import { ApiError, humanizeModel } from "../lib/learnloop";
import { isActive, lab, labPath, seconds, type BatchBrief, type GroundTruth, type Mode } from "../lib/reliability";

/** What sending the batch would involve, from the server's 409 answer. Nothing is sent until the person approves. */
interface LivePlan {
  repetitions: number;
  model_id: string;
  max_requests: number;
  object_view_requests: number;
  object_view_locate_requests: number;
  object_view_frames: number;
  estimated_cost_usd: number | null;
  live_enabled: boolean;
  independent_inference: boolean;
}

export function LabPage() {
  const { id = "" } = useParams();
  const options = useQuery({ queryKey: ["lab-options"], queryFn: lab.options });
  const batches = useQuery({
    queryKey: ["lab-batches"],
    queryFn: lab.batches,
    refetchInterval: (query) => (query.state.data?.some((item) => isActive(item.status)) ? 2000 : false),
  });
  const procedure = options.data?.procedures.find((item) => item.id === id);
  const mine = batches.data?.filter((item) => item.project_id === id);
  return (
    <Layout procedureId={id} title={procedure?.name} titleLoading={options.isPending}>
      <div className="heading">
        <div>
          <h1>Reliability Lab</h1>
          <p className="muted">Run one recording of this procedure through verification or learning several times under identical conditions, then compare the results. Developer tool.</p>
        </div>
      </div>
      {options.data && !procedure ? <p className="error" role="alert">This procedure was not found.</p> : <NewExperiment projectId={id} />}
      <section className="panel lab-section">
        <h2>Experiments</h2>
        <ErrorNote message={batches.error ? (batches.error as Error).message : ""} />
        {mine?.length === 0 && <p className="muted small">No experiments on this procedure yet.</p>}
        {mine && mine.length > 0 && <History items={mine} />}
      </section>
    </Layout>
  );
}

/** Experiments used to live at /lab/...; send old links to the experiment's procedure. */
export function LegacyLabRedirect() {
  const { batchId = "", runId } = useParams();
  const batch = useQuery({ queryKey: ["lab-batch", batchId], queryFn: () => lab.batch(batchId) });
  if (batch.data) return <Navigate to={labPath(batch.data.project_id, batchId, runId)} replace />;
  if (batch.error) return <Navigate to="/" replace />;
  return null;
}

function History({ items }: { items: BatchBrief[] }) {
  const revisions = new Set(items.map((item) => item.git_revision));
  const fingerprints = new Set(items.map((item) => item.fingerprint));
  return (
    <div className="table-wrap">
      <table className="lab-table">
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Mode</th>
            <th>Status</th>
            <th>Runs</th>
            <th>Result</th>
            <th>Mean runtime</th>
            <th title="Code revision and a hash of the models, settings, prompt, and procedure the runs used">Revision · config</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id}>
              <td><Link to={labPath(item.project_id, item.id)}>{item.name}</Link>{item.template_version ? <div className="muted small">v{item.template_version}</div> : null}</td>
              <td>{item.mode}{!item.independent_inference && <div className="muted small">replay allowed</div>}</td>
              <td><span className={`lab-status ${item.status}`}>{item.status}</span></td>
              <td className="mono">{item.counts.finished}/{item.repetitions}{item.counts.execution_errors + item.counts.invalid_results > 0 && <span className="lab-error-text"> · {item.counts.execution_errors + item.counts.invalid_results} err</span>}</td>
              <td className="small">{summaryText(item)}</td>
              <td className="mono">{seconds(item.summary?.mean_runtime as number | null)}</td>
              <td className="mono small">{item.git_revision || "—"} · {item.fingerprint}</td>
              <td className="small">{new Date(item.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {(revisions.size > 1 || fingerprints.size > 1) && <p className="muted small">These experiments ran on {revisions.size} code revision{revisions.size === 1 ? "" : "s"} and {fingerprints.size} configuration{fingerprints.size === 1 ? "" : "s"}. Compare like with like.</p>}
    </div>
  );
}

function summaryText(item: BatchBrief): string {
  const summary = item.summary ?? {};
  if (item.mode === "verification") {
    const distribution = (summary.distribution ?? {}) as Record<string, number>;
    const parts = ["PASS", "FAIL", "INCONCLUSIVE"].filter((key) => distribution[key]).map((key) => `${distribution[key]} ${key}`);
    const agreement = summary.modal_agreement as number | null;
    const exact = summary.exact_match as number | null;
    return [parts.join(", ") || "—", agreement != null ? `agree ${(agreement * 100).toFixed(0)}%` : "", exact != null ? `exact ${(exact * 100).toFixed(0)}%` : ""].filter(Boolean).join(" · ");
  }
  const counts = (summary.step_counts ?? {}) as Record<string, number>;
  const shapes = Object.entries(counts).map(([steps, runs]) => `${runs}× ${steps} steps`).join(", ");
  return [shapes || "—", `${summary.steps_in_every_run ?? 0} steps in every run`].join(" · ");
}

function NewExperiment({ projectId }: { projectId: string }) {
  const client = useQueryClient();
  const navigate = useNavigate();
  const options = useQuery({ queryKey: ["lab-options"], queryFn: lab.options });
  const [mode, setMode] = useState<Mode>("verification");
  const [templateId, setTemplateId] = useState("");
  const [referenceId, setReferenceId] = useState("");
  const [videoId, setVideoId] = useState("");
  const [repetitions, setRepetitions] = useState(5);
  const [concurrency, setConcurrency] = useState(2);
  const [independent, setIndependent] = useState(true);
  const [expected, setExpected] = useState<"" | "PASS" | "FAIL">("");
  const [stepLabels, setStepLabels] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [plan, setPlan] = useState<LivePlan | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const procedures = options.data?.procedures ?? [];
  const procedure = procedures.find((item) => item.id === projectId);
  const accepted = (procedure?.templates ?? []).filter((item) => item.acceptance_status === "accepted");
  const template = accepted.find((item) => item.id === templateId);
  const recordings = (procedure?.recordings ?? []).filter((item) => (mode === "verification" ? item.role === "execution" : item.role === "demonstration"));

  const firstAccepted = accepted[0]?.id ?? "";
  const templateKnown = accepted.some((item) => item.id === templateId);
  useEffect(() => {
    if (!templateKnown && templateId !== firstAccepted) {
      setTemplateId(firstAccepted);
      setStepLabels({});
    }
  }, [templateKnown, templateId, firstAccepted]);
  useEffect(() => {
    if (!recordings.some((item) => item.video_id === videoId)) setVideoId(recordings[recordings.length - 1]?.video_id ?? "");
  }, [recordings, videoId]);
  useEffect(() => setPlan(null), [mode, projectId, templateId, videoId, repetitions, independent]);

  const ready = Boolean(projectId && videoId && (mode === "learning" || templateId));
  const groundTruth: GroundTruth | undefined = useMemo(
    () => (mode === "verification" ? { expected_verdict: expected || null, step_labels: Object.fromEntries(Object.entries(stepLabels).filter(([, value]) => value)) } : undefined),
    [mode, expected, stepLabels],
  );

  async function start(event: FormEvent | null, allowLive: boolean) {
    event?.preventDefault();
    setError("");
    setBusy(true);
    try {
      const created = await lab.create({
        mode,
        project_id: projectId,
        video_id: videoId,
        template_id: mode === "verification" ? templateId : "",
        reference_template_id: mode === "learning" ? referenceId : "",
        repetitions,
        max_concurrency: concurrency,
        independent_inference: independent,
        allow_live: allowLive,
        ground_truth: groundTruth,
        name,
      });
      await client.invalidateQueries({ queryKey: ["lab-batches"] });
      navigate(labPath(projectId, created.id));
    } catch (exc) {
      const detail = exc instanceof ApiError ? (exc.detail as { code?: string; plan?: LivePlan }) : null;
      if (detail && typeof detail === "object" && detail.code === "live_call_required" && detail.plan) setPlan(detail.plan);
      else setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (options.isPending) return <section className="panel lab-section"><p className="muted small">Loading procedures…</p></section>;
  return (
    <section className="panel lab-section">
      <h2>New experiment</h2>
      <ErrorNote message={options.error ? (options.error as Error).message : ""} />
      <form className="lab-form" onSubmit={(event) => void start(event, false)}>
        <div className="tabs" role="tablist" aria-label="Experiment mode">
          <button type="button" role="tab" aria-selected={mode === "verification"} className={`tab ${mode === "verification" ? "active" : ""}`} onClick={() => setMode("verification")}>Verification reliability</button>
          <button type="button" role="tab" aria-selected={mode === "learning"} className={`tab ${mode === "learning" ? "active" : ""}`} onClick={() => setMode("learning")}>Learning reliability</button>
        </div>
        <div className="lab-grid">
          {mode === "verification" ? (
            <label className="stack">
              <span className="small muted">Accepted procedure version</span>
              <select className="field" value={templateId} onChange={(event) => { setTemplateId(event.target.value); setStepLabels({}); }}>
                {accepted.length === 0 && <option value="">No accepted version</option>}
                {accepted.map((item) => <option key={item.id} value={item.id}>v{item.version} · {item.name} · {item.steps.length} steps</option>)}
              </select>
            </label>
          ) : (
            <label className="stack">
              <span className="small muted">Reference procedure (optional)</span>
              <select className="field" value={referenceId} onChange={(event) => setReferenceId(event.target.value)}>
                <option value="">None</option>
                {(procedure?.templates ?? []).map((item) => <option key={item.id} value={item.id}>v{item.version} · {item.acceptance_status}</option>)}
              </select>
            </label>
          )}
          <label className="stack">
            <span className="small muted">{mode === "verification" ? "Attempt recording" : "Demonstration recording"}</span>
            <select className="field" value={videoId} onChange={(event) => setVideoId(event.target.value)}>
              {recordings.length === 0 && <option value="">No {mode === "verification" ? "attempts" : "demonstrations"} yet</option>}
              {recordings.map((item) => <option key={item.video_id} value={item.video_id} disabled={!item.ready}>{item.filename} · {item.duration_sec.toFixed(1)}s{item.ready ? "" : " (preparing)"}</option>)}
            </select>
          </label>
          <label className="stack">
            <span className="small muted">Repetitions</span>
            <CountField value={repetitions} max={options.data?.max_repetitions ?? 50} onChange={setRepetitions} />
          </label>
          <label className="stack">
            <span className="small muted">Concurrent runs (max {options.data?.max_concurrency ?? 4})</span>
            <CountField value={concurrency} max={options.data?.max_concurrency ?? 4} onChange={setConcurrency} />
          </label>
          <label className="stack">
            <span className="small muted">Name (optional)</span>
            <input className="field" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. baseline before tracker change" />
          </label>
        </div>
        {procedure && mode === "verification" && (
          <details className="lab-upload">
            <summary className="small">Upload another attempt to this procedure</summary>
            <RecordingDrop procedureId={procedure.id} role="execution" prompt="Drag and drop an attempt here or" recordings={[]} onChange={async (created) => {
              await client.invalidateQueries({ queryKey: ["lab-options"] });
              if (created) setVideoId(created.video.id);
            }} />
          </details>
        )}
        <label className="check">
          <input type="checkbox" checked={independent} onChange={(event) => setIndependent(event.target.checked)} />
          Independent inference: every repetition sends its own Meta requests and replays no stored answer
        </label>
        {!independent && <p className="warning">With replay allowed, repetitions can reuse one stored answer. That measures the deterministic verifier, not model variation.</p>}
        {mode === "verification" && (
          <fieldset className="lab-truth">
            <legend className="small">Ground truth (optional)</legend>
            <div className="row">
              <span className="small">This attempt is</span>
              <select className="field" value={expected} onChange={(event) => setExpected(event.target.value as "" | "PASS" | "FAIL")} aria-label="Expected verdict">
                <option value="">Unlabeled</option>
                <option value="PASS">Correct execution — expected PASS</option>
                <option value="FAIL">Incorrect execution — expected FAIL</option>
              </select>
            </div>
            {template && (
              <details>
                <summary className="small">Label individual steps</summary>
                <StepLabels steps={template.steps} labels={stepLabels} onChange={setStepLabels} />
              </details>
            )}
          </fieldset>
        )}
        <ErrorNote message={error} />
        {plan ? (
          <div className="plan" role="dialog" aria-label="Approve live calls">
            <h3>Approve {plan.repetitions} independent {plan.repetitions === 1 ? "run" : "runs"}?</h3>
            <dl>
              <dt>Analysis model</dt><dd>{humanizeModel(plan.model_id)}, {plan.max_requests} request per run</dd>
              <dt>Object views</dt><dd>{plan.object_view_locate_requests ? `up to ${plan.object_view_locate_requests} locate and ${plan.object_view_requests} SAM requests per run, up to ${plan.object_view_frames} frames each` : "none"}</dd>
              <dt>Second look</dt><dd>up to 3 requests per run when a required step is unresolved</dd>
              <dt>Cost</dt><dd>{plan.estimated_cost_usd == null ? "Unknown: no Meta rate is configured" : `$${(plan.estimated_cost_usd * plan.repetitions).toFixed(2)}`}</dd>
              <dt>Retries</dt><dd>None. A failed request fails its run only.</dd>
            </dl>
            {!plan.live_enabled && <p className="warning">Live Meta calls are off on this server (LEARNLOOP_LIVE_INFERENCE). Nothing can be sent.</p>}
            <div className="row">
              <button type="button" className="btn primary" disabled={busy || !plan.live_enabled} onClick={() => void start(null, true)}>Approve and start</button>
              <button type="button" className="btn" onClick={() => setPlan(null)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button className="btn primary" disabled={!ready || busy}>Review and start</button>
        )}
      </form>
    </section>
  );
}

export function StepLabels({ steps, labels, onChange }: { steps: { step_id: string; name: string }[]; labels: Record<string, string>; onChange: (labels: Record<string, string>) => void }) {
  return (
    <div className="lab-step-labels">
      {steps.map((step) => (
        <label key={step.step_id} className="row">
          <span className="grow small">{step.name}</span>
          <select className="field" aria-label={`Expected verdict for ${step.name}`} value={labels[step.step_id] ?? ""} onChange={(event) => onChange({ ...labels, [step.step_id]: event.target.value })}>
            <option value="">Unlabeled</option>
            <option value="PASS">PASS</option>
            <option value="FAIL">FAIL</option>
            <option value="NOT_APPLICABLE">NOT APPLICABLE</option>
          </select>
        </label>
      ))}
    </div>
  );
}

/** A whole-number field that can be cleared and retyped. The parent gets each valid value as it is typed; leaving the field puts back the clamped value. */
function CountField({ value, max, onChange }: { value: number; max: number; onChange: (value: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const clamp = (raw: number) => Math.min(max, Math.max(1, Math.round(raw)));
  return (
    <input
      className="field"
      type="number"
      min={1}
      max={max}
      step={1}
      value={text}
      onChange={(event) => {
        setText(event.target.value);
        const parsed = Number(event.target.value);
        if (event.target.value.trim() !== "" && Number.isFinite(parsed) && parsed >= 1 && parsed <= max) onChange(Math.round(parsed));
      }}
      onBlur={() => {
        const parsed = Number(text);
        const next = text.trim() === "" || !Number.isFinite(parsed) ? value : clamp(parsed);
        setText(String(next));
        if (next !== value) onChange(next);
      }}
    />
  );
}
