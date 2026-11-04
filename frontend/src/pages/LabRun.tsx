import { useQuery } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ErrorNote, Layout } from "../components/ui";
import { api, clock, frameUrl, type Recording } from "../lib/learnloop";
import { isActive, lab, runLabel, runTone, seconds, type Batch, type LabRun, type Learned } from "../lib/reliability";
import { Report } from "./Verify";

export function LabRunPage() {
  const { batchId = "", runId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const stepId = params.get("step") ?? "";
  const query = useQuery({
    queryKey: ["lab-batch", batchId],
    queryFn: () => lab.batch(batchId),
    refetchInterval: (state) => (state.state.data && isActive(state.state.data.status) ? 1500 : false),
  });
  const batch = query.data;
  const run = batch?.runs.find((item) => item.id === runId);
  const ordered = [...(batch?.runs ?? [])].sort((a, b) => a.index - b.index);
  const position = ordered.findIndex((item) => item.id === runId);
  const neighbour = (offset: number) => ordered[position + offset];
  const link = (item: LabRun) => `/lab/${batchId}/runs/${item.id}${stepId ? `?step=${encodeURIComponent(stepId)}` : ""}`;
  return (
    <Layout title="Reliability Lab">
      <p className="small row">
        <Link to={`/lab/${batchId}`}>← {batch?.name ?? "Experiment"}</Link>
        <span className="grow" />
        {neighbour(-1) && <Link className="btn small" to={link(neighbour(-1))}>← Run {neighbour(-1).index}</Link>}
        {neighbour(1) && <Link className="btn small" to={link(neighbour(1))}>Run {neighbour(1).index} →</Link>}
      </p>
      <ErrorNote message={query.error ? (query.error as Error).message : ""} />
      {batch && !run && <p className="error">This run is not part of the experiment.</p>}
      {batch && run && (
        <>
          <RunHeader batch={batch} run={run} />
          {batch.mode === "verification" && stepId && <StepAcrossRuns batch={batch} run={run} stepId={stepId} />}
          {batch.mode === "verification" && run.verification_id && run.status === "completed" && batch.project_exists && (
            <EmbeddedReport batch={batch} run={run} stepId={stepId} onStep={(id) => setParams({ step: id }, { replace: true })} />
          )}
          {batch.mode === "learning" && run.learned && run.status === "completed" && <LearnedProcedure learned={run.learned} />}
          <RunDiagnostics run={run} />
        </>
      )}
    </Layout>
  );
}

function RunHeader({ batch, run }: { batch: Batch; run: LabRun }) {
  const inference = run.result.execution_metadata?.inference;
  return (
    <section className="panel lab-section">
      <div className="heading">
        <div>
          <h1 className="lab-title">Run {String(run.index).padStart(2, "0")}</h1>
          <p className="muted small">{batch.mode === "verification" ? "Verification" : "Learning"} repetition {run.index} of {batch.repetitions} · {batch.video.filename}</p>
        </div>
        <span className={`verdict large ${runTone(run)}`}>{runLabel(run)}</span>
      </div>
      {run.status === "failed" && <p className="error"><strong>{run.error_kind.replaceAll("_", " ")}:</strong> {run.error}</p>}
      {run.status === "canceled" && <p className="warning">{run.error}</p>}
      {run.result.diagnostics?.reinspection?.incomplete && <p className="warning">The second look did not finish, so these are the verdicts from before it.</p>}
      <dl className="facts">
        <dt>Runtime</dt><dd>{seconds(run.duration_seconds)}{run.started_at ? ` · started ${new Date(run.started_at).toLocaleTimeString()}` : ""}{run.finished_at ? `, finished ${new Date(run.finished_at).toLocaleTimeString()}` : ""}</dd>
        <dt>Inference</dt>
        <dd>
          {inference ? (
            <>
              {inference.fully_fresh ? "Fresh: every answer in this run came from its own request." : `${inference.replayed_answers} stored answer${inference.replayed_answers === 1 ? "" : "s"} replayed.`}{" "}
              Analysis call {inference.guided_call || "—"}; first-frame boxes {inference.locate_live_requests} live / {inference.locate_replayed_requests} replayed; SAM {inference.object_views_live_requests} live / {inference.object_views_replayed_requests} replayed
              {inference.reinspection.length > 0 && `; second look ${inference.reinspection.map((item) => `${item.step_id} ${item.status}${item.source ? ` (${item.source})` : ""}`).join(", ")}`}. Cache policy {inference.cache_policy}.
            </>
          ) : "—"}
        </dd>
        <dt>Analysis run</dt><dd className="mono small">{run.analysis_run_id || "not started"}</dd>
        {run.verification_id && <><dt>Report</dt><dd className="mono small"><a href={`/api/verifications/${run.verification_id}/export`} download={`report-${run.verification_id}.json`}>{run.verification_id}</a></dd></>}
      </dl>
    </section>
  );
}

/** The selected step in every run, side by side, so a disagreement can be read from its evidence. No cause is assigned. */
function StepAcrossRuns({ batch, run, stepId }: { batch: Batch; run: LabRun; stepId: string }) {
  const name = batch.steps.find((item) => item.step_id === stepId)?.name ?? stepId;
  const rows = [...batch.runs].sort((a, b) => a.index - b.index);
  return (
    <section className="panel lab-section">
      <h2>“{name}” across runs</h2>
      <p className="muted small">What each run recorded for this step. Compare verdicts, evidence moments, matched objects, and reasons; differences are shown, not explained.</p>
      <div className="table-wrap">
        <table className="lab-table">
          <thead><tr><th>Run</th><th>Verdict</th><th>Basis</th><th>Evidence</th><th>Objects</th><th>Warnings</th><th>Reason</th></tr></thead>
          <tbody>
            {rows.map((item) => {
              const step = item.result.steps?.find((entry) => entry.step_id === stepId);
              return (
                <tr key={item.id} className={item.id === run.id ? "current-row" : ""}>
                  <td className="mono"><Link to={`/lab/${batch.id}/runs/${item.id}?step=${encodeURIComponent(stepId)}`}>{String(item.index).padStart(2, "0")}</Link></td>
                  <td>{step ? <span className={`verdict ${step.verdict === "PASS" ? "pass" : step.verdict === "FAIL" ? "fail" : step.verdict === "CANNOT_VERIFY" ? "unknown" : "muted"}`}>{(step.verdict ?? step.raw_verdict).replaceAll("_", " ")}</span> : <span className="muted small">{runLabel(item)}</span>}</td>
                  <td className="small">{step ? `${step.basis.replaceAll("_", " ")}${step.error_category ? ` · ${step.error_category.toLowerCase().replaceAll("_", " ")}` : ""}${step.reinspected ? " · second look" : ""}` : ""}</td>
                  <td className="small">
                    {step ? (
                      <div className="lab-thumbs">
                        {step.evidence_timestamps.length === 0 && <span className="muted">none</span>}
                        {step.evidence_timestamps.slice(0, 3).map((time) => (
                          <a key={time} href={frameUrl(batch.video.id, time)} target="_blank" rel="noreferrer" title={`Frame at ${clock(time)}`}>
                            <img src={frameUrl(batch.video.id, time)} alt={`Run ${item.index} evidence at ${clock(time)}`} loading="lazy" />
                            <span className="mono">{clock(time)}</span>
                          </a>
                        ))}
                      </div>
                    ) : null}
                  </td>
                  <td className="mono small">{step?.observed_objects.join(", ") || "—"}</td>
                  <td className="small">{step?.warnings.map((warning) => warning.replaceAll("_", " ")).join(", ") || "—"}</td>
                  <td className="small">{step?.reason || step?.observed || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EmbeddedReport({ batch, run, stepId, onStep }: { batch: Batch; run: LabRun; stepId: string; onStep: (id: string) => void }) {
  const procedure = useQuery({ queryKey: ["procedure", batch.project_id], queryFn: () => api.procedure(batch.project_id) });
  const analysis = useQuery({ queryKey: ["run", run.analysis_run_id], queryFn: () => api.run(run.analysis_run_id), enabled: Boolean(run.analysis_run_id) });
  if (procedure.error || analysis.error) return <ErrorNote message={((procedure.error ?? analysis.error) as Error).message} />;
  if (!procedure.data || !analysis.data) return <p className="muted small">Loading the report…</p>;
  const source = procedure.data.recordings.find((item) => item.video.id === batch.video.id);
  if (!source) return <p className="warning">The attempt recording is no longer in this procedure.</p>;
  // The report is drawn against this repetition's own analysis, not the attempt's latest ordinary one.
  const recording: Recording = { ...source, run: analysis.data };
  return (
    <section className="lab-report" aria-label="This run's report">
      <h2>Report for this run</h2>
      <p className="muted small">The ordinary report viewer, read-only. Boxes are this run’s own SAM tracks.</p>
      <Report verificationId={run.verification_id} procedure={procedure.data} recording={recording} readOnly initialStepId={stepId} onStepChange={onStep} />
    </section>
  );
}

function LearnedProcedure({ learned }: { learned: Learned }) {
  const labels = new Map(learned.roles.map((role) => [role.role_id, role.label]));
  const describe = (condition: Learned["steps"][number]["completion_conditions"][number]) =>
    condition.relation ? `${labels.get(condition.subject_role) ?? condition.subject_role} ${condition.relation.replaceAll("_", " ")} ${labels.get(condition.target_role) ?? condition.target_role}` : `${labels.get(condition.subject_role) ?? condition.subject_role} is ${condition.to_state || condition.state_label || ""}`;
  return (
    <section className="panel lab-section">
      <h2>Learned procedure</h2>
      <p className="muted small">Kept with this run only. It was not added to the procedure.</p>
      <h3>Objects</h3>
      <ul className="lab-list small">{learned.roles.map((role) => <li key={role.role_id}><strong>{role.label}</strong>{role.visual_characteristics ? ` — ${role.visual_characteristics}` : ""}</li>)}</ul>
      <h3>Steps</h3>
      <ol className="lab-list small">
        {learned.steps.filter((step) => !step.removed).map((step) => (
          <li key={step.step_id}>
            <strong>{step.name}</strong>{step.required ? "" : " (optional)"} <span className="muted mono">{clock(step.demo_time_start)}–{clock(step.demo_time_end)}</span>
            <div>{step.completion_conditions.map(describe).join("; ") || <span className="muted">no checkable condition</span>}</div>
          </li>
        ))}
      </ol>
      {learned.notes && learned.notes.length > 0 && <><h3>Notes</h3><ul className="lab-list small">{learned.notes.map((note) => <li key={note}>{note}</li>)}</ul></>}
    </section>
  );
}

function RunDiagnostics({ run }: { run: LabRun }) {
  const diagnostics = run.result.diagnostics ?? {};
  const meta = run.result.execution_metadata;
  const objects = diagnostics.objects;
  return (
    <details className="panel lab-section diagnostics" open={run.status === "failed"}>
      <summary>Run diagnostics</summary>
      {diagnostics.failed_stages && Object.keys(diagnostics.failed_stages).length > 0 && (
        <>
          <h3>Stages that did not finish</h3>
          <ul className="lab-list small">{Object.entries(diagnostics.failed_stages).map(([stage, item]) => <li key={stage}><strong>{stage}</strong> {item.status}{item.error ? `: ${item.error}` : ""}</li>)}</ul>
        </>
      )}
      {diagnostics.roles && diagnostics.roles.length > 0 && (
        <>
          <h3>Roles</h3>
          <ul className="lab-list small">{diagnostics.roles.map((role) => <li key={role.role_id}>{role.label}: {role.status}{role.objects.length ? ` → ${role.objects.join(", ")}` : ""}</li>)}</ul>
        </>
      )}
      {objects && (
        <>
          <h3>Objects (SAM)</h3>
          {!objects.available ? <p className="muted small">{objects.note}</p> : (
            <>
              {objects.error && <p className="warning">{objects.error}</p>}
              <table className="lab-table small">
                <thead><tr><th>Object</th><th>Boxes</th><th>Seen cells</th><th>First seen</th><th>Track pieces</th><th>Appearance</th></tr></thead>
                <tbody>
                  {Object.entries(objects.objects).map(([id, item]) => (
                    <tr key={id}><td>{item.label} <span className="muted mono">{id}</span></td><td className="mono">{item.boxes}</td><td className="mono">{item.seen_cells}/{item.cells}</td><td className="mono">{clock(item.first_seen)}</td><td className="mono">{item.track_segments ?? "—"}</td><td className="small">{Object.entries(item.appearance).map(([key, count]) => `${key} ${count}`).join(", ") || "—"}</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
      {diagnostics.warnings && diagnostics.warnings.length > 0 && <><h3>Report warnings</h3><ul className="lab-list small">{diagnostics.warnings.map((item) => <li key={item}>{item}</li>)}</ul></>}
      {meta && (
        <>
          <h3>Stages</h3>
          <ul className="lab-list small">{Object.entries(meta.stages).map(([stage, item]) => <li key={stage}>{stage}: {item.status}{item.seconds != null ? ` (${item.seconds}s)` : ""}</li>)}</ul>
          <h3>Usage</h3>
          <p className="small">{meta.usage.requests} requests{meta.usage.failed_requests ? `, ${meta.usage.failed_requests} failed` : ""} · {meta.usage.input_tokens.toLocaleString()} in / {meta.usage.output_tokens.toLocaleString()} out tokens · cost {meta.usage.cost_usd == null ? "unknown" : `$${meta.usage.cost_usd.toFixed(4)}`} · {meta.model_id}{meta.segmentation_model_id ? ` + ${meta.segmentation_model_id}` : ""} · rev {meta.git_revision || "—"}</p>
        </>
      )}
    </details>
  );
}
