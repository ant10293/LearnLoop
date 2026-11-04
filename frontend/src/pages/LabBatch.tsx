import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { ErrorNote, Layout } from "../components/ui";
import { humanizeModel } from "../lib/learnloop";
import {
  fraction,
  isActive,
  labPath,
  lab,
  modalVerdicts,
  percent,
  runLabel,
  runTone,
  seconds,
  stepMark,
  type Batch,
  type GroundTruth,
  type LabRun,
  type LearningMetrics,
  type Rate,
  type VerificationMetrics,
} from "../lib/reliability";
import { StepLabels } from "./Lab";

export function LabBatchPage() {
  const { id = "", batchId = "" } = useParams();
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["lab-batch", batchId],
    queryFn: () => lab.batch(batchId),
    refetchInterval: (state) => (state.state.data && isActive(state.state.data.status) ? 1500 : false),
  });
  const [error, setError] = useState("");
  const batch = query.data;
  const set = (next: Batch) => client.setQueryData(["lab-batch", batchId], next);

  async function cancel() {
    setError("");
    try {
      set(await lab.cancel(batchId));
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <Layout procedureId={id} title={batch?.procedure_name} titleLoading={query.isPending}>
      {batch && batch.project_id !== id && <Navigate to={labPath(batch.project_id, batch.id)} replace />}
      <p className="small"><Link to={labPath(id)}>← All experiments</Link></p>
      <ErrorNote message={query.error ? (query.error as Error).message : error} />
      {!batch && !query.error && <p className="muted">Loading the experiment…</p>}
      {batch && (
        <>
          <Overview batch={batch} onCancel={() => void cancel()} />
          {batch.mode === "verification" ? <VerificationView batch={batch} onChange={set} /> : <LearningView batch={batch} />}
          <Reproducibility batch={batch} />
        </>
      )}
    </Layout>
  );
}

function Overview({ batch, onCancel }: { batch: Batch; onCancel: () => void }) {
  const { counts } = batch;
  const share = counts.total ? (counts.finished / counts.total) * 100 : 0;
  return (
    <section className="panel lab-section">
      <div className="heading">
        <div>
          <h1 className="lab-title">{batch.name}</h1>
          <p className="muted small">
            {batch.mode === "verification" ? "Verification experiment" : "Learning experiment"} · {batch.procedure_name}
            {batch.template_version ? ` · procedure v${batch.template_version}` : ""} · {batch.video.filename} ({batch.video.duration_sec?.toFixed(1)}s)
          </p>
        </div>
        <div className="row">
          <span className={`lab-status ${batch.status}`}>{batch.status}</span>
          {isActive(batch.status) && batch.status !== "canceling" && <button className="btn danger small" onClick={onCancel}>Cancel experiment</button>}
          <a className="btn small" href={`/api/reliability/batches/${batch.id}/export?format=json`} download={`reliability-${batch.id.slice(0, 8)}.json`}>JSON</a>
          <a className="btn small" href={`/api/reliability/batches/${batch.id}/export?format=csv`} download={`reliability-${batch.id.slice(0, 8)}.csv`}>CSV</a>
        </div>
      </div>
      <div className="lab-progress" role="progressbar" aria-valuenow={counts.finished} aria-valuemin={0} aria-valuemax={counts.total} aria-label="Runs finished">
        <div style={{ width: `${share}%` }} />
      </div>
      <p className="small lab-counts">
        <strong>{counts.finished} / {counts.total}</strong> finished · {counts.completed} completed · {counts.running} running · {counts.queued} queued · {counts.failed} failed · {counts.canceled} canceled
        <span className="muted"> · up to {batch.max_concurrency} at once · {batch.independent_inference ? "independent inference" : "replay allowed"} · {humanizeModel(batch.model_id)}{batch.segmentation_model_id ? ` + ${humanizeModel(batch.segmentation_model_id)}` : ""}</span>
      </p>
      {!batch.independent_inference && <p className="warning">Replay allowed: repetitions may reuse the same stored answers, so agreement here measures the deterministic verifier, not model variation.</p>}
      {batch.mode === "verification" && <p className="small">Expected:{batch.ground_truth.expected_verdict ?? <span className="muted">unlabeled</span>}</p>}
      {!batch.project_exists && <p className="warning">The procedure was deleted. Results recorded here remain; links into its reports may not work.</p>}
    </section>
  );
}

function Tile({ label, value, detail, tone }: { label: string; value: ReactNode; detail?: ReactNode; tone?: string }) {
  return (
    <div className={`lab-tile ${tone ?? ""}`}>
      <span className="small muted">{label}</span>
      <strong>{value}</strong>
      {detail && <span className="small muted">{detail}</span>}
    </div>
  );
}

const rateDetail = (item: Rate | null | undefined) => (item && item.denominator ? fraction(item) : "no runs");

// ---------------------------------------------------------------- verification

function VerificationView({ batch, onChange }: { batch: Batch; onChange: (batch: Batch) => void }) {
  const metrics = batch.metrics as VerificationMetrics;
  const { consistency, accuracy, performance, counts } = metrics;
  return (
    <>
      <section className="lab-tiles" aria-label="Summary">
        <Tile label="Consistency (modal agreement)" value={percent(consistency.modal_agreement)} detail={`${rateDetail(consistency.modal_agreement)} runs agree on ${consistency.modal_verdicts.join(" / ") || "—"}`} />
        <Tile label="Verdicts" value={<span className="lab-dist">{(["PASS", "FAIL", "INCONCLUSIVE"] as const).map((key) => <span key={key} className={`verdict ${key === "PASS" ? "pass" : key === "FAIL" ? "fail" : "unknown"}`}>{consistency.distribution[key]} {key === "INCONCLUSIVE" ? "INCONCL." : key}</span>)}</span>} detail={`${consistency.valid_runs} runs with a verdict`} />
        <Tile label="Exact match" value={accuracy.labeled ? percent(accuracy.exact_match) : "N/A"} detail={accuracy.labeled ? `${rateDetail(accuracy.exact_match)} vs expected ${accuracy.expected}` : "unlabeled: no accuracy"} />
        <Tile label="Decisive accuracy" value={accuracy.labeled ? percent(accuracy.decisive_accuracy) : "N/A"} detail={accuracy.labeled ? `coverage ${percent(accuracy.decisive_coverage)} (${rateDetail(accuracy.decisive_coverage)})` : ""} />
        <Tile label={accuracy.expected === "FAIL" ? "False pass" : "False fail"} value={accuracy.labeled ? percent(accuracy.expected === "FAIL" ? accuracy.false_pass : accuracy.false_fail) : "N/A"} detail={accuracy.labeled ? rateDetail(accuracy.expected === "FAIL" ? accuracy.false_pass : accuracy.false_fail) : ""} tone={accuracy.labeled && ((accuracy.expected === "FAIL" ? accuracy.false_pass : accuracy.false_fail)?.numerator ?? 0) > 0 ? "bad" : ""} />
        <Tile label="Inconclusive" value={percent(rateOf(consistency.distribution.INCONCLUSIVE, consistency.valid_runs))} detail={`${consistency.distribution.INCONCLUSIVE}/${consistency.valid_runs}`} />
        <Tile label="Execution errors" value={percent(performance.execution_error_rate)} detail={`${counts.execution_errors} errors, ${counts.invalid_results} without a verdict, ${counts.canceled} canceled — not verdicts`} tone={counts.execution_errors + counts.invalid_results > 0 ? "bad" : ""} />
        <Tile label="Runtime per run" value={seconds(performance.runtime_seconds.mean)} detail={`median ${seconds(performance.runtime_seconds.median)} · ${seconds(performance.runtime_seconds.min)}–${seconds(performance.runtime_seconds.max)} · batch ${seconds(performance.batch_wall_clock_seconds)}`} />
        <Tile label="Fresh inference" value={`${metrics.independence.fully_fresh_runs}/${metrics.independence.runs_with_results}`} detail={`${metrics.independence.replayed_answers} stored answers replayed`} tone={batch.independent_inference && metrics.independence.replayed_answers > 0 ? "bad" : ""} />
      </section>
      <RunMatrix batch={batch} />
      <StepTable batch={batch} metrics={metrics} />
      <GroundTruthEditor batch={batch} onChange={onChange} />
      <Detection metrics={metrics} />
      <Usage metrics={metrics} />
    </>
  );
}

const rateOf = (numerator: number, denominator: number): Rate => ({ value: denominator ? numerator / denominator : null, numerator, denominator });

type SortKey = "run" | "verdict" | "runtime" | "evidence";

function RunMatrix({ batch }: { batch: Batch }) {
  const navigate = useNavigate();
  const [sort, setSort] = useState<SortKey>("run");
  const [filter, setFilter] = useState("all");
  const stepIds = batch.steps.map((step) => step.step_id);
  const modal = useMemo(() => modalVerdicts(batch.runs, stepIds), [batch.runs, stepIds]);
  const unstable = new Set(((batch.metrics as VerificationMetrics).steps ?? []).filter((item) => item.unstable).map((item) => item.step_id));
  const overallModal = (batch.metrics as VerificationMetrics).consistency.modal_verdicts;
  const rows = batch.runs
    .filter((run) => filter === "all" || runLabel(run) === filter || (filter === "disagree" && run.status === "completed" && overallModal.length === 1 && run.result.verdict !== overallModal[0]))
    .sort((a, b) => {
      if (sort === "verdict") return runLabel(a).localeCompare(runLabel(b)) || a.index - b.index;
      if (sort === "runtime") return (a.duration_seconds ?? Infinity) - (b.duration_seconds ?? Infinity);
      if (sort === "evidence") return (b.result.evidence_count ?? -1) - (a.result.evidence_count ?? -1);
      return a.index - b.index;
    });
  const labels = Array.from(new Set(batch.runs.map(runLabel)));
  return (
    <section className="panel lab-section">
      <div className="heading">
        <div>
          <h2>Runs × steps</h2>
          <p className="muted small">P pass · F fail · ? cannot verify · – not applicable. Outlined cells differ from that step’s most common verdict; shaded columns changed between runs. Select a cell to inspect that run at that step.</p>
        </div>
        <div className="row">
          <select className="field" aria-label="Filter runs" value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">All runs</option>
            <option value="disagree">Disagrees with the majority</option>
            {labels.map((label) => <option key={label} value={label}>{label}</option>)}
          </select>
          <select className="field" aria-label="Sort runs" value={sort} onChange={(event) => setSort(event.target.value as SortKey)}>
            <option value="run">By run</option>
            <option value="verdict">By verdict</option>
            <option value="runtime">By runtime</option>
            <option value="evidence">By evidence</option>
          </select>
        </div>
      </div>
      <div className="table-wrap">
        <table className="lab-table lab-matrix">
          <thead>
            <tr>
              <th>Run</th>
              <th>Verdict</th>
              {batch.steps.map((step, index) => (
                <th key={step.step_id} className={unstable.has(step.step_id) ? "unstable" : ""} title={`${step.name}${step.required ? "" : " (not required)"}`}>
                  S{index + 1}{!step.required && <span className="muted">*</span>}
                </th>
              ))}
              <th>Evidence</th>
              <th>Runtime</th>
              <th>Inference</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((run) => (
              <tr key={run.id} className={run.status === "failed" ? "errored" : ""}>
                <td className="mono"><Link to={labPath(batch.project_id, batch.id, run.id)}>{String(run.index).padStart(2, "0")}</Link></td>
                <td>
                  <span className={`verdict ${runTone(run)}`}>{runLabel(run)}</span>
                  {run.status === "running" && run.progress && <div className="muted small">{run.progress.phase}{run.progress.stage ? `: ${run.progress.stage.replaceAll("_", " ")}` : ""} ({run.progress.stages_done}/{run.progress.stages_total})</div>}
                  {run.status === "failed" && <div className="lab-error-text small" title={run.error}>{run.error_kind.replaceAll("_", " ")}</div>}
                </td>
                {stepIds.map((stepId) => {
                  const step = run.result.steps?.find((item) => item.step_id === stepId);
                  const differs = Boolean(step?.verdict && modal[stepId] && step.verdict !== modal[stepId]);
                  return (
                    <td key={stepId} className={unstable.has(stepId) ? "unstable" : ""}>
                      {step ? (
                        <button type="button" className={`lab-cell ${step.verdict ?? "none"}${differs ? " differs" : ""}`} title={`${step.name}: ${step.verdict ?? step.raw_verdict} · ${step.evidence_count} evidence${step.reason ? ` · ${step.reason}` : ""}`} onClick={() => navigate(`${labPath(batch.project_id, batch.id, run.id)}?step=${encodeURIComponent(stepId)}`)}>
                          {stepMark[step.verdict ?? ""] ?? "·"}
                        </button>
                      ) : <span className="muted">·</span>}
                    </td>
                  );
                })}
                <td className="mono">{run.result.evidence_count ?? "—"}</td>
                <td className="mono">{seconds(run.duration_seconds)}</td>
                <td className="small">{inferenceLabel(run)}</td>
                <td>{run.status === "running" || run.status === "queued" ? <CancelRun batch={batch} run={run} /> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ol className="lab-step-key small muted">
        {batch.steps.map((step, index) => <li key={step.step_id}>S{index + 1} {step.name}{step.required ? "" : " (not required)"}</li>)}
      </ol>
    </section>
  );
}

function inferenceLabel(run: LabRun): string {
  const inference = run.result.execution_metadata?.inference;
  if (!inference) return "—";
  if (inference.fully_fresh) return "fresh";
  return inference.replayed_answers ? `${inference.replayed_answers} replayed` : inference.guided_call || "—";
}

function CancelRun({ batch, run }: { batch: Batch; run: LabRun }) {
  const client = useQueryClient();
  if (run.cancel_requested) return <span className="muted small">stopping…</span>;
  return (
    <button className="btn small" onClick={async () => client.setQueryData(["lab-batch", batch.id], await lab.cancelRun(batch.id, run.id))}>Cancel</button>
  );
}

function StepTable({ batch, metrics }: { batch: Batch; metrics: VerificationMetrics }) {
  const labeled = metrics.steps.some((item) => item.label);
  return (
    <section className="panel lab-section">
      <h2>Per-step reliability</h2>
      <p className="muted small">Counts are over runs that produced a report. Agreement is the share giving that step’s most common verdict. Accuracy needs a step label.</p>
      <div className="table-wrap">
        <table className="lab-table">
          <thead>
            <tr>
              <th>Step</th><th>PASS</th><th>FAIL</th><th>CANNOT VERIFY</th><th>N/A</th><th>Agreement</th>
              {labeled && <><th>Label</th><th>Accuracy</th></>}
              <th>No evidence</th><th>Warnings</th><th>Mean evidence</th><th>Second look</th><th>Error categories</th>
            </tr>
          </thead>
          <tbody>
            {metrics.steps.map((item, index) => (
              <tr key={item.step_id} className={item.unstable ? "unstable-row" : ""}>
                <td>S{index + 1} {item.name}{item.unstable && <span className="lab-flag">unstable</span>}{item.runs_missing_step > 0 && <div className="muted small">missing from {item.runs_missing_step} reports</div>}</td>
                <td className="mono">{item.distribution.PASS}</td>
                <td className="mono">{item.distribution.FAIL}</td>
                <td className="mono">{item.distribution.CANNOT_VERIFY}</td>
                <td className="mono">{item.distribution.NOT_APPLICABLE}</td>
                <td className="mono">{percent(item.agreement)} <span className="muted">{fraction(item.agreement)}</span></td>
                {labeled && <><td>{item.label ?? <span className="muted">—</span>}</td><td className="mono">{item.accuracy ? `${percent(item.accuracy)} ${fraction(item.accuracy)}` : "N/A"}</td></>}
                <td className="mono">{percent(item.missing_evidence)}</td>
                <td className="mono">{percent(item.with_warnings)}</td>
                <td className="mono">{item.mean_evidence ?? "—"}</td>
                <td className="mono">{item.reinspected}</td>
                <td className="small">{Object.entries(item.error_categories).map(([key, count]) => `${key.toLowerCase().replaceAll("_", " ")} ×${count}`).join(", ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {batch.steps.length === 0 && <p className="muted small">This procedure version has no steps.</p>}
    </section>
  );
}

function GroundTruthEditor({ batch, onChange }: { batch: Batch; onChange: (batch: Batch) => void }) {
  const [expected, setExpected] = useState(batch.ground_truth.expected_verdict ?? "");
  const [labels, setLabels] = useState<Record<string, string>>(batch.ground_truth.step_labels ?? {});
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  useEffect(() => setSaved(false), [expected, labels]);
  async function save() {
    setError("");
    try {
      const body: GroundTruth = { expected_verdict: (expected || null) as GroundTruth["expected_verdict"], step_labels: Object.fromEntries(Object.entries(labels).filter(([, value]) => value)) };
      onChange(await lab.groundTruth(batch.id, body));
      setSaved(true);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }
  return (
    <details className="panel lab-section">
      <summary><h2 className="lab-inline">Ground truth</h2> <span className="muted small">{batch.ground_truth.expected_verdict ? `expected ${batch.ground_truth.expected_verdict}` : "unlabeled"}{Object.keys(batch.ground_truth.step_labels ?? {}).length ? `, ${Object.keys(batch.ground_truth.step_labels).length} step labels` : ""}</span></summary>
      <p className="muted small">Labels are stored apart from the results. Saving recalculates accuracy from the recorded runs; nothing runs again.</p>
      <div className="row">
        <span className="small">This attempt is</span>
        <select className="field" aria-label="Expected verdict" value={expected} onChange={(event) => setExpected(event.target.value)}>
          <option value="">Unlabeled</option>
          <option value="PASS">Correct execution — expected PASS</option>
          <option value="FAIL">Incorrect execution — expected FAIL</option>
        </select>
      </div>
      <StepLabels steps={batch.steps} labels={labels} onChange={setLabels} />
      <div className="row">
        <button className="btn primary" onClick={() => void save()}>Save labels</button>
        {saved && <span className="small muted">Saved. Metrics recalculated.</span>}
      </div>
      <ErrorNote message={error} />
    </details>
  );
}

function Detection({ metrics }: { metrics: VerificationMetrics }) {
  const { detection } = metrics;
  const views = detection.object_views;
  const warnings = Object.entries(detection.warnings);
  return (
    <section className="panel lab-section">
      <h2>Detection and evidence</h2>
      <p className="muted small">Only what the pipeline recorded. Object ids come from each attempt’s own analysis and can differ between runs, so labels are shown beside them. Identity switches are not recorded by the pipeline and are not reported.</p>
      <div className="lab-columns">
        <div>
          <h3>Role matching across runs</h3>
          {detection.roles.length === 0 && <p className="muted small">No roles recorded.</p>}
          <ul className="lab-list small">
            {detection.roles.map((role) => (
              <li key={role.role_id}><strong>{role.label}</strong>: unresolved {percent(role.unresolved)} ({fraction(role.unresolved)}), ambiguous {percent(role.ambiguous)}</li>
            ))}
          </ul>
          {warnings.length > 0 && (
            <>
              <h3>Report warnings</h3>
              <ul className="lab-list small">{warnings.map(([text, count]) => <li key={text}>{count}× {text}</li>)}</ul>
            </>
          )}
        </div>
        <div>
          <h3>Object views (SAM)</h3>
          <p className="small">{views.runs_with_views} of {detection.valid_runs} runs made object views{views.runs_with_view_errors ? `, ${views.runs_with_view_errors} with an error` : ""}; appearance check used in {views.runs_with_appearance_check}.{views.mean_recovered_sightings != null ? ` Mean recovered sightings ${views.mean_recovered_sightings}.` : ""}</p>
          <ObjectList title="Never boxed" items={views.objects_without_boxes} />
          <ObjectList title="Lost and boxed again" items={views.relocated_after_loss} />
          <ObjectList title="Track split into pieces" items={views.interrupted_tracks} />
        </div>
      </div>
    </section>
  );
}

function ObjectList({ title, items }: { title: string; items: { object_id: string; label: string; runs: number }[] }) {
  if (!items.length) return null;
  return (
    <>
      <h3>{title}</h3>
      <ul className="lab-list small">{items.map((item) => <li key={item.object_id}>{item.label} <span className="muted mono">{item.object_id}</span> in {item.runs} run{item.runs === 1 ? "" : "s"}</li>)}</ul>
    </>
  );
}

function Usage({ metrics }: { metrics: VerificationMetrics | LearningMetrics }) {
  const { performance } = metrics;
  return (
    <section className="panel lab-section">
      <h2>Execution</h2>
      <dl className="facts">
        <dt>Runtime per run</dt><dd>mean {seconds(performance.runtime_seconds.mean)}, median {seconds(performance.runtime_seconds.median)}, min {seconds(performance.runtime_seconds.min)}, max {seconds(performance.runtime_seconds.max)} over {performance.runtime_seconds.count} completed runs</dd>
        <dt>Total run time</dt><dd>{seconds(performance.total_run_seconds)} summed across runs</dd>
        <dt>Batch elapsed</dt><dd>{seconds(performance.batch_wall_clock_seconds)} wall clock</dd>
        <dt>Error rate</dt><dd>{percent(performance.execution_error_rate)} ({fraction(performance.execution_error_rate)} finished runs, canceled excluded)</dd>
        <dt>Requests</dt><dd>{performance.requests} sent{performance.failed_requests ? `, ${performance.failed_requests} failed` : ""}</dd>
        <dt>Tokens</dt><dd>{performance.input_tokens.toLocaleString()} in, {performance.output_tokens.toLocaleString()} out</dd>
        <dt>Cost</dt><dd>{performance.cost_usd == null ? "Unknown (no known price for every response)" : `$${performance.cost_usd.toFixed(4)}`}</dd>
      </dl>
    </section>
  );
}

// ---------------------------------------------------------------- learning

function LearningView({ batch }: { batch: Batch }) {
  const metrics = batch.metrics as LearningMetrics;
  const distribution = (values: Record<string, number>) => Object.entries(values).map(([key, count]) => `${count}× ${key}`).join(", ") || "—";
  return (
    <>
      <section className="lab-tiles" aria-label="Summary">
        <Tile label="Step counts" value={distribution(metrics.step_count_distribution)} detail={`${metrics.valid_runs} learned procedures`} />
        <Tile label="Steps in every run" value={metrics.steps_in_every_run} detail="same conditions, exact normalized text" />
        <Tile label="Object counts" value={distribution(metrics.object_count_distribution)} />
        <Tile label="Label overlap" value={metrics.mean_pairwise_label_overlap == null ? "N/A" : `${(metrics.mean_pairwise_label_overlap * 100).toFixed(0)}%`} detail="mean over pairs of runs" />
        <Tile label="Reviewable" value={percent(metrics.reviewable)} detail={`${fraction(metrics.reviewable)} pass validation with a checkable step`} />
        <Tile label="Execution errors" value={percent(metrics.performance.execution_error_rate)} detail={rateDetail(metrics.performance.execution_error_rate)} tone={metrics.counts.execution_errors ? "bad" : ""} />
        <Tile label="Runtime per run" value={seconds(metrics.performance.runtime_seconds.mean)} detail={`batch ${seconds(metrics.performance.batch_wall_clock_seconds)}`} />
      </section>
      <section className="panel lab-section">
        <h2>Runs</h2>
        <div className="table-wrap">
          <table className="lab-table">
            <thead><tr><th>Run</th><th>Status</th><th>Steps</th><th>Required</th><th>Checkable</th><th>Roles</th><th>Objects</th><th>Validation</th><th>Reviewable</th><th>Runtime</th><th>Inference</th><th /></tr></thead>
            <tbody>
              {batch.runs.map((run) => (
                <tr key={run.id} className={run.status === "failed" ? "errored" : ""}>
                  <td className="mono"><Link to={labPath(batch.project_id, batch.id, run.id)}>{String(run.index).padStart(2, "0")}</Link></td>
                  <td><span className={`verdict ${run.status === "completed" ? "muted" : runTone(run)}`}>{run.status.toUpperCase()}</span>{run.status === "failed" && <div className="lab-error-text small" title={run.error}>{run.error_kind.replaceAll("_", " ")}</div>}{run.progress && <div className="muted small">{run.progress.phase} {run.progress.stages_done}/{run.progress.stages_total}</div>}</td>
                  <td className="mono">{run.result.step_count ?? "—"}</td>
                  <td className="mono">{run.result.required_steps ?? "—"}</td>
                  <td className="mono">{run.result.checkable_steps ?? "—"}</td>
                  <td className="mono">{run.result.role_count ?? "—"}</td>
                  <td className="mono">{run.result.object_count ?? "—"}</td>
                  <td className="small">{run.result.validation ? `${run.result.validation.errors ?? "?"} errors, ${run.result.validation.warnings ?? "?"} warnings` : "—"}</td>
                  <td>{run.result.reviewable == null ? "—" : run.result.reviewable ? "yes" : "no"}</td>
                  <td className="mono">{seconds(run.duration_seconds)}</td>
                  <td className="small">{inferenceLabel(run)}</td>
                  <td>{run.status === "running" || run.status === "queued" ? <CancelRun batch={batch} run={run} /> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel lab-section">
        <h2>Steps aligned by their conditions</h2>
        <p className="muted small">A step lines up with steps of other runs only when its checkable conditions, in normalized object labels, are exactly the same. Names and positions are shown for reading; they are not used to align. Times are compared only when the conditions occur once per run.</p>
        <div className="table-wrap">
          <table className="lab-table">
            <thead><tr><th>Conditions</th><th>Runs</th><th>Names used</th><th>Positions</th><th>Required in</th><th>Start (s)</th><th>End (s)</th></tr></thead>
            <tbody>
              {metrics.aligned_steps.map((item) => (
                <tr key={item.signature} className={item.in_every_run ? "" : "unstable-row"}>
                  <td className="mono small">{item.signature}</td>
                  <td className="mono">{fraction(item.share)} <span className="muted">#{item.runs.join(", #")}</span></td>
                  <td className="small">{item.names.filter(Boolean).join(" · ") || "—"}</td>
                  <td className="mono">{item.positions.join(", ")}</td>
                  <td className="mono">{item.required_in}</td>
                  <td className="mono small">{item.start_seconds ? `${item.start_seconds.mean} (±${item.start_seconds.stdev}, range ${item.start_seconds.range})` : "—"}</td>
                  <td className="mono small">{item.end_seconds ? `${item.end_seconds.mean} (±${item.end_seconds.stdev}, range ${item.end_seconds.range})` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {metrics.unaligned_steps.length > 0 && (
          <>
            <h3>Steps with nothing to align on (for review)</h3>
            <ul className="lab-list small">{metrics.unaligned_steps.map((item) => <li key={`${item.run}-${item.position}`}>Run #{item.run}, step {item.position}: {item.name || "unnamed"} — {item.reason}</li>)}</ul>
          </>
        )}
      </section>
      <section className="panel lab-section lab-columns">
        <div>
          <h2>Object labels</h2>
          <p className="muted small">Normalized: lowercase, no articles or punctuation, plural s dropped.</p>
          <ul className="lab-list small">{metrics.labels.map((item) => <li key={item.label}><span className="mono">{item.label}</span> in {fraction(item.share)} runs</li>)}</ul>
        </div>
        <div>
          <h2>Validation and warnings</h2>
          <p className="small">Validation errors in {fraction(metrics.validation_errors)} runs. Ambiguities reported in {fraction(metrics.runs_with_ambiguities)} runs.</p>
          {metrics.reference && (
            <>
              <h3>Against the reference procedure</h3>
              <p className="muted small">Exact condition matches only. Structural agreement, not correctness.</p>
              <ul className="lab-list small">{metrics.reference.per_run.map((item) => <li key={item.run}>Run #{item.run}: {item.matched}/{metrics.reference!.reference_signatures.length} matched, {item.extra} extra{item.missing.length ? `; missing ${item.missing.join("; ")}` : ""}</li>)}</ul>
            </>
          )}
        </div>
      </section>
      <Usage metrics={metrics} />
    </>
  );
}

// ---------------------------------------------------------------- reproducibility

function Reproducibility({ batch }: { batch: Batch }) {
  const config = batch.config;
  return (
    <details className="panel lab-section diagnostics">
      <summary>Reproducibility</summary>
      <dl className="facts">
        <dt>Experiment</dt><dd className="mono">{batch.id}</dd>
        <dt>Created</dt><dd>{new Date(batch.created_at).toLocaleString()}{batch.started_at ? `, started ${new Date(batch.started_at).toLocaleString()}` : ""}{batch.finished_at ? `, finished ${new Date(batch.finished_at).toLocaleString()}` : ""}</dd>
        <dt>Code</dt><dd className="mono">{batch.git_revision || "no git revision available"} · app {String(config.app_version ?? "")}</dd>
        <dt>Configuration</dt><dd className="mono">{batch.fingerprint}</dd>
        <dt>Recording</dt><dd>{batch.video.filename} · <span className="mono small">sha256 {batch.video.checksum_sha256}</span></dd>
        {batch.template_id && <><dt>Procedure</dt><dd>version {batch.template_version} · <span className="mono small">{batch.template_id}</span> (frozen copy stored with the experiment)</dd></>}
        <dt>Models</dt><dd>{batch.model_id}{batch.segmentation_model_id ? `, ${batch.segmentation_model_id}` : ", no segmentation"}</dd>
        <dt>Inference policy</dt><dd>{String(config.inference_policy ?? "")}</dd>
        {(config.shared_deterministic ?? []).length > 0 && <><dt>Shared, deterministic</dt><dd><ul className="lab-list">{config.shared_deterministic!.map((item) => <li key={item}>{item}</li>)}</ul></dd></>}
        <dt>Prompt key</dt><dd className="mono small">{String(config.prompt_cache_key ?? "")}</dd>
      </dl>
      <h3>Settings at creation</h3>
      <pre className="lab-pre">{JSON.stringify(config.settings ?? {}, null, 2)}</pre>
    </details>
  );
}
