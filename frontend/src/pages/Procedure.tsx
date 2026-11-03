import { X } from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AssetPreview } from "../components/AssetPreview";
import { AnalysisStages, ObjectViewsNotice } from "../components/RecordingAnalysis";
import { Bone, FieldBone, TagBone } from "../components/Skeleton";
import { ErrorNote, Film, Layout, ReviewTag } from "../components/ui";
import { api, boxAt, clock, conditionText, frameUrl, relationLabels, learnedProcedures, type Condition, type ObjectView, type Procedure, type Recording, type Role, type Template, type TemplateDependency, type TemplatePatch, type TemplateStep } from "../lib/learnloop";

type StepEdit = Partial<Pick<TemplateStep, "name" | "expected_action" | "completion_text" | "removed" | "requirement_status">> & { completion_conditions?: Partial<Condition>[]; order_after?: string[] };
type OtherStep = { step_id: string; number: number; name: string };

const ROLE_PLACEHOLDERS = 2;
const STEP_PLACEHOLDERS = 3;

export function ProcedurePage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const procedure = useQuery({
    queryKey: ["procedure", id],
    queryFn: () => api.procedure(id),
    refetchInterval: (query) => {
      const latest = latestDemonstration(query.state.data);
      return latest && (preparing(latest) || analyzing(latest)) ? 1000 : false;
    },
  });
  // The newest procedure unless the person picked an earlier one. Saving an edit adds a version to the same one.
  const [picked, setPicked] = useState("");
  const learned = learnedProcedures(procedure.data?.templates ?? []);
  const latest = latestDemonstration(procedure.data);
  const learnFrom = latest?.run?.status === "completed" && !procedure.data?.templates.some((item) => item.source_run_id === latest.run?.id) ? latest.run.id : "";
  // A demonstration still being learned is selected until the person picks a finished one.
  const teaching = latest && (preparing(latest) || analyzing(latest) || learnFrom) ? latest : undefined;
  const template = learned.find((item) => item.family_id === picked) ?? (teaching ? undefined : procedure.data?.template);
  const learning = useRef("");
  const [steps, setSteps] = useState<Record<string, StepEdit>>({});
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [title, setTitle] = useState("");
  const [seek, setSeek] = useState({ time: null as number | null, tick: 0 });
  const [error, setError] = useState("");
  const [removeError, setRemoveError] = useState("");
  const [saving, setSaving] = useState(false);
  const views = useQuery({
    queryKey: ["object-views", template?.source_run_id],
    queryFn: () => api.objectViews(template?.source_run_id ?? ""),
    enabled: Boolean(template?.source_run_id),
    refetchInterval: (query) => (query.state.data?.running ? 1000 : false),
  });
  useEffect(() => {
    setSteps({});
    setLabels({});
    setTitle("");
  }, [template?.id]);

  // The analysis can finish after the person has left Teach. The procedure is still learned.
  useEffect(() => {
    if (!learnFrom || learning.current === learnFrom) return;
    learning.current = learnFrom;
    api
      .learn(id, learnFrom)
      .then(() => client.invalidateQueries({ queryKey: ["procedure", id] }))
      .catch((exc: Error) => setError(exc.message));
  }, [client, id, learnFrom]);

  if (procedure.isPending) return <Layout procedureId={id} titleLoading><ProcedureSkeleton /></Layout>;
  const history = (learned.length > 1 || teaching) && (
    <LearnedList learned={learned} recordings={procedure.data?.recordings ?? []} current={template?.id} teaching={teaching} onPick={setPicked} />
  );
  if (teaching && !template) {
    return (
      <Layout procedureId={id} title={procedure.data?.name}>
        <ProcedureSkeleton
          notices={
            <>
              <ErrorNote message={error} />
              <div className="panel teaching-banner"><TeachingStatus recording={teaching} procedureId={id} /></div>
            </>
          }
          history={history}
        />
      </Layout>
    );
  }
  if (!template) {
    return (
      <Layout procedureId={id} title={procedure.data?.name}>
        <h1>Procedure</h1>
        {latest?.run?.status === "failed" ? (
          <TeachingStatus recording={latest} procedureId={id} />
        ) : (
          <>
            <p className="lead">Nothing has been learned yet.</p>
            <Link className="btn primary" to={`/procedures/${id}/teach`}>Teach this procedure</Link>
          </>
        )}
        <ErrorNote message={error} />
        <ErrorNote message={procedure.error ? (procedure.error as Error).message : ""} />
      </Layout>
    );
  }

  const demo = procedure.data?.recordings.find((item) => item.video.id === template.source_video_id);
  const roles = template.document.roles;
  const dirty = Object.keys(steps).length > 0 || Object.keys(labels).length > 0 || Boolean(title);
  const change = (stepId: string, edit: StepEdit) => setSteps((current) => ({ ...current, [stepId]: { ...current[stepId], ...edit } }));

  function patch(): TemplatePatch {
    return {
      ...(title ? { name: title } : {}),
      steps: Object.entries(steps).map(([step_id, edit]) => ({ step_id, ...edit })),
      roles: Object.entries(labels).map(([role_id, label]) => ({ role_id, label })),
    };
  }

  async function save(accept: boolean) {
    setSaving(true);
    setError("");
    try {
      await api.revise(template!.id, { ...patch(), ...(accept ? { acceptance_status: "accepted" as const } : {}) });
      await client.invalidateQueries({ queryKey: ["procedure", id] });
      if (accept) navigate(`/procedures/${id}/verify`);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!window.confirm(`Delete “${template!.name}”? All its versions, its demonstration, and its verification reports will be removed. This cannot be undone.`)) return;
    setSaving(true);
    setRemoveError("");
    try {
      await api.removeTemplate(template!.id);
      setPicked("");
      await Promise.all([client.invalidateQueries({ queryKey: ["procedure", id] }), client.invalidateQueries({ queryKey: ["procedures"] })]);
    } catch (exc) {
      setRemoveError((exc as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const required = template.document.steps.filter((step) => !step.removed && (steps[step.step_id]?.requirement_status ?? step.requirement_status) === "required");
  return (
    <Layout procedureId={id} title={procedure.data?.name}>
      <ProcedureView
        version={<>Version {template.version} · Proposed by Meta from the demonstration. Each edit you save becomes a new version; the original is kept.</>}
        actions={
          <>
            <ReviewTag state={template.review_state} />
            {dirty && <button className="btn" disabled={saving} onClick={() => void save(false)}>Save edits</button>}
            {(dirty || template.review_state !== "accepted") && (
              <button className="btn primary" disabled={saving || required.length === 0} onClick={() => void save(true)}>Accept procedure</button>
            )}
          </>
        }
        notices={
          <>
            <ErrorNote message={error} />
            {latest && (teaching || latest.run?.status === "failed") && (
              <div className="panel teaching-banner">
                <p className="small muted">The procedure below is from an earlier demonstration.</p>
                <TeachingStatus recording={latest} procedureId={id} />
              </div>
            )}
            {template.source_run_id && (
              <ObjectViewsNotice runId={template.source_run_id} views={views.data} onStarted={() => void client.invalidateQueries({ queryKey: ["object-views", template.source_run_id] })} />
            )}
            {required.length === 0 && <p className="warning">Mark at least one step as required before accepting.</p>}
          </>
        }
        film={<Film src={demo?.video.playback_url ?? ""} time={seek.time} seek={seek.tick} label="Demonstration" caption={template.document.summary} tracks={roles.flatMap((role) => {
          const track = views.data?.running ? undefined : views.data?.objects[role.source_object_ids[0]]?.track;
          return track?.length ? [{ label: labels[role.role_id] ?? role.label, track }] : [];
        })} />}
        history={history}
        objects={roles.map((role) => {
          const remaking = Boolean(views.data?.running);
          const view = remaking ? undefined : views.data?.objects[role.source_object_ids[0]];
          return (
            <div key={role.role_id} className="object">
              <div className="object-head">
                <span className="object-thumb-wrap">
                  {remaking ? <Bone className="object-thumb" /> : view?.thumbnail_url ? <img className="object-thumb" src={view.thumbnail_url} alt={role.label} /> : <span className="object-thumb empty" aria-hidden="true" />}
                  {view && view.seen_cells > 0 && (
                    <span className="object-count" title={`Seen in ${view.seen_cells} timeline ${view.seen_cells === 1 ? "frame" : "frames"}`}>
                      {view.seen_cells}
                    </span>
                  )}
                </span>
                <input className="field" aria-label={`Label for ${role.label}`} value={labels[role.role_id] ?? role.label} onChange={(event) => setLabels((current) => ({ ...current, [role.role_id]: event.target.value }))} />
              </div>
              {remaking && <Bone width="100%" />}
              {view && <ObjectTimeline view={view} label={role.label} videoId={template.source_video_id} onSeek={(time) => setSeek((current) => ({ time, tick: current.tick + 1 }))} />}
              {role.original_label && <span className="edited">Human-corrected from “{role.original_label}”</span>}
            </div>
          );
        })}
        steps={template.document.steps.map((step, index) => (
          <StepCard key={step.step_id} index={index} step={step} edit={steps[step.step_id] ?? {}} roles={roles} template={template} others={template.document.steps.flatMap((other, position) => (other.step_id === step.step_id || (steps[other.step_id]?.removed ?? other.removed) ? [] : [{ step_id: other.step_id, number: position + 1, name: steps[other.step_id]?.name ?? other.name }]))} onChange={(edit) => change(step.step_id, edit)} onSeek={(time) => setSeek((current) => ({ time, tick: current.tick + 1 }))} />
        ))}
      />
      <ErrorNote message={removeError} />
      <div className="procedure-footer">
        <button className="btn danger" disabled={saving} onClick={() => void remove()}>Delete procedure</button>
      </div>
    </Layout>
  );
}

/** The learned procedure's layout and fixed copy, for the loaded page and its skeleton. */
function ProcedureView({ version, actions, notices, film, history, objects, steps }: { version: ReactNode; actions: ReactNode; notices?: ReactNode; film: ReactNode; history?: ReactNode; objects: ReactNode; steps: ReactNode }) {
  return (
    <>
      <div className="heading">
        <div>
          <h1>Learned procedure</h1>
          <p className="muted small">{version}</p>
        </div>
        <div className="row">{actions}</div>
      </div>
      {notices}
      <div className="split procedure">
        <div className="sticky">
          {film}
          <section className="panel">
            <h2>Objects</h2>
            <p className="muted small">Correct a label if the model named an object wrongly.</p>
            {objects}
          </section>
          {history}
        </div>
        <ol className="steps">{steps}</ol>
      </div>
    </>
  );
}

/** Where SAM found the object, moment by moment. A blank cell means it was not found, not that it was gone. */
function ObjectTimeline({ view, label, videoId, onSeek }: { view: ObjectView; label: string; videoId: string; onSeek: (time: number) => void }) {
  const [preview, setPreview] = useState<number | null>(null);
  if (view.seen_cells === 0) return <p className="muted small">SAM did not find this object.</p>;
  // The crop came from the most confident sighting in its second, so the cell is labeled, seeks, and shows its full frame at that sighting's own time.
  const shown = (cell: ObjectView["cells"][number]) => cell.frame_time ?? cell.time;
  const frames = view.cells.flatMap((cell) => {
    if (!cell.url) return [];
    const box = boxAt(view.track ?? [], shown(cell));
    return [{ time: shown(cell), src: cell.url, filename: `${label} · ${clock(shown(cell))}`, confidence: cell.confidence, full: box ? { src: frameUrl(videoId, box.time), box } : undefined }];
  });
  return (
    <div className="object-timeline" role="list" aria-label={`Timeline for ${label}`}>
      {view.cells.map((cell) => {
        const stamp = clock(shown(cell));
        return (
          <div key={cell.time} role="listitem" className="object-cell" aria-label={cell.url ? `Frame at ${stamp}` : `Not found at ${stamp}`}>
            {cell.url ? (
              <button
                type="button"
                className="object-frame"
                title={stamp}
                aria-label={`Expand frame at ${stamp}`}
                onClick={() => {
                  onSeek(shown(cell));
                  setPreview(frames.findIndex((frame) => frame.src === cell.url));
                }}
              >
                <img src={cell.url} alt="" loading="lazy" />
              </button>
            ) : (
              <div className="object-frame missing" title={`Not found · ${stamp}`} />
            )}
            <div className="object-time mono">{stamp}</div>
          </div>
        );
      })}
      {preview != null && <AssetPreview images={frames} startIndex={preview} onClose={() => setPreview(null)} />}
    </div>
  );
}

function LearnedList({ learned, recordings, current, teaching, onPick }: { learned: Template[]; recordings: Recording[]; current?: string; teaching?: Recording; onPick: (familyId: string) => void }) {
  return (
    <section className="panel">
      <h2>Learned procedures</h2>
      <p className="muted small">Each comes from one analyzed demonstration. Pick one to see it.</p>
      <div className="learned-list">
        {teaching && (
          <button type="button" className="learned-item" aria-current={!current} onClick={() => onPick("")}>
            {teaching.video.thumbnail_url ? <img src={teaching.video.thumbnail_url} alt="" /> : <span className="learned-thumb" />}
            <span className="learned-text">
              <strong className="small">{teaching.video.filename || "Demonstration"} · Learning</strong>
              <span className="muted small">{preparing(teaching) ? "Preparing the recording" : analyzing(teaching) ? "Analyzing the demonstration" : "Building the procedure"}</span>
            </span>
          </button>
        )}
        {learned.map((item, index) => {
          const still = recordings.find((row) => row.video.id === item.source_video_id)?.video.thumbnail_url;
          return (
            <button key={item.family_id} type="button" className="learned-item" aria-current={item.id === current} onClick={() => onPick(item.family_id)}>
              {still ? <img src={still} alt="" /> : <span className="learned-thumb" />}
              <span className="learned-text">
                <strong className="small">{item.source_filename || "Demonstration"}{index === 0 ? " · Latest" : ""}</strong>
                <span className="muted small">Learned {new Date(`${item.created_at}Z`).toLocaleString()} · {item.document.steps.length} steps · {item.acceptance_status === "accepted" ? "Accepted" : "Draft"}</span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

/** Stands in for the procedure while it loads or is being learned. `history` is real, so it stays usable. */
function ProcedureSkeleton({ notices, history }: { notices?: ReactNode; history?: ReactNode }) {
  return (
    <div aria-busy={history ? undefined : "true"}>
      {!history && <p className="sr-only">Loading the procedure</p>}
      <ProcedureView
        version={<Bone width="24rem" />}
        actions={<TagBone />}
        notices={notices}
        history={history}
        film={<Film label="Demonstration" caption={<Bone width="70%" />} />}
        objects={Array.from({ length: ROLE_PLACEHOLDERS }, (_, index) => (
          <span key={index} className="object"><FieldBone /></span>
        ))}
        steps={Array.from({ length: STEP_PLACEHOLDERS }, (_, index) => (
          <StepFrame
            key={index}
            index={index}
            title={<FieldBone className="title grow" />}
            time={<Bone width="15ch" />}
            footer={<><Bone width="5rem" /><Bone width="6rem" /></>}
          >
            <StepFields
              objects={<Bone width="40%" />}
              action={<FieldBone />}
              completion={<FieldBone />}
              checks={<div className="condition condition-edit"><FieldBone select /><FieldBone select /><FieldBone select /></div>}
              uncertainty={<Bone width="30%" />}
            />
          </StepFrame>
        ))}
      />
    </div>
  );
}

function latestDemonstration(procedure?: Procedure): Recording | undefined {
  return [...(procedure?.recordings ?? [])].reverse().find((item) => item.role === "demonstration");
}

function preparing(recording: Recording): boolean {
  return !["ready", "failed"].includes(recording.video.ingest_status);
}

function analyzing(recording: Recording): boolean {
  return recording.run?.status === "pending" || recording.run?.status === "running";
}

function TeachingStatus({ recording, procedureId }: { recording: Recording; procedureId: string }) {
  const { video, run } = recording;
  if (run?.status === "failed") {
    return (
      <div className="error" role="alert">
        <strong>Learning from {video.filename} failed. No procedure was generated.</strong>
        <p>{run.error}</p>
        <Link className="btn small" to={`/procedures/${procedureId}/teach`}>Go to Teach</Link>
      </div>
    );
  }
  return (
    <div role="status">
      <h2>Learning from {video.filename}</h2>
      {preparing(recording) ? <p className="status-line">Preparing the recording. {video.ingest_detail}</p> : run && <AnalysisStages run={run} />}
      {run?.status === "completed" && <p className="status-line">Building the procedure.</p>}
    </div>
  );
}

function StepFrame({ index, removed, title, time, footer, children }: { index: number; removed?: boolean; title: ReactNode; time?: ReactNode; footer: ReactNode; children: ReactNode }) {
  return (
    <li className={`step ${removed ? "removed" : ""}`}>
      <header>
        <span className="index">{index + 1}</span>
        {title}
        {time}
      </header>
      {children}
      <footer className="row">{footer}</footer>
    </li>
  );
}

function StepFields({ objects, action, completion, checks, order, uncertainty, edited }: { objects: ReactNode; action: ReactNode; completion: ReactNode; checks: ReactNode; order?: ReactNode; uncertainty: ReactNode; edited?: ReactNode }) {
  return (
    <>
      <p className="small"><span className="muted">Objects</span> {objects}</p>
      <label className="stack small"><span className="muted">Expected action</span>
        {action}
      </label>
      <label className="stack small"><span className="muted">Observable completion</span>
        {completion}
      </label>
      <div className="stack small">
        <span className="muted">What the verifier checks</span>
        {checks}
      </div>
      {order}
      <p className="muted small">{uncertainty}</p>
      {edited}
    </>
  );
}

function orderSummary(numbers: number[]) {
  if (numbers.length === 0) return "Any order";
  if (numbers.length === 1) return `After step ${numbers[0]}`;
  return `After steps ${numbers.slice(0, -1).join(", ")} and ${numbers[numbers.length - 1]}`;
}

function StepOrder({ number, others, selected, dependencies, onChange }: { number: number; others: OtherStep[]; selected: string[]; dependencies: TemplateDependency[]; onChange: (order_after: string[]) => void }) {
  const [editing, setEditing] = useState(false);
  return (
    <div className="stack small">
      <button type="button" className="order-row" aria-expanded={editing} onClick={() => setEditing(!editing)}>
        <span>Step order</span>
        <span className="order-value muted">{orderSummary(others.filter((other) => selected.includes(other.step_id)).map((other) => other.number))}</span>
        <svg className="order-chevron" viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" /></svg>
      </button>
      {editing && (
        <fieldset className="order-edit">
          <legend className="muted">Select the steps that must happen before this step (Step {number}). The rest can come in any order.</legend>
          <div className="order-cards">
            {others.map((other) => {
              const checked = selected.includes(other.step_id);
              const dep = dependencies.find((item) => item.step_id === other.step_id);
              return (
                <label key={other.step_id} className={`order-card ${checked ? "selected" : ""}`}>
                  <input type="checkbox" aria-label={`Step ${other.number}: ${other.name}`} checked={checked} onChange={(event) => onChange(event.target.checked ? [...selected, other.step_id] : selected.filter((item) => item !== other.step_id))} />
                  <span className="order-number mono">{other.number}</span>
                  <span>
                    {other.name}
                    {dep?.origin === "model" && dep.category === "necessary_physical" && <span className="origin order-origin" title={dep.rationale}>Meta: physically necessary</span>}
                  </span>
                  {checked && <span className="order-tick" aria-hidden="true">✓</span>}
                </label>
              );
            })}
          </div>
        </fieldset>
      )}
    </div>
  );
}

function StepCard({ index, step, edit, roles, others, onChange, onSeek }: { index: number; step: TemplateStep; edit: StepEdit; roles: Role[]; template: Template; others: OtherStep[]; onChange: (edit: StepEdit) => void; onSeek: (time: number) => void }) {
  const orderAfter = edit.order_after ?? step.dependencies.filter((dep) => dep.constrains_order).map((dep) => dep.step_id);
  const status = edit.requirement_status ?? step.requirement_status;
  const removed = edit.removed ?? step.removed;
  const conditions = (edit.completion_conditions ?? step.completion_conditions) as Partial<Condition>[];
  const required = conditions.filter((item) => item.evidence_requirement !== "optional");
  const objects = step.object_roles.map((role) => roles.find((item) => item.role_id === role)?.label ?? role);
  const setConditions = (next: Partial<Condition>[]) => onChange({ completion_conditions: next });
  const setCondition = (position: number, next: Partial<Condition>) => setConditions(required.map((item, index) => (index === position ? { ...item, ...next, origin: "human" } : item)));
  const addCondition = () => setConditions([...required, { subject_role: step.object_roles[0] ?? roles[0]?.role_id, target_role: step.object_roles[1] ?? roles[1]?.role_id, relation: "attached", origin: "human" }]);
  const roleOptions = roles.map((role) => <option key={role.role_id} value={role.role_id}>{role.label}</option>);
  return (
    <StepFrame
      index={index}
      removed={removed}
      title={<input className="field title grow" aria-label={`Step ${index + 1} name`} value={edit.name ?? step.name} onChange={(event) => onChange({ name: event.target.value })} disabled={removed} />}
      time={step.demo_time_start != null && <button className="btn small mono step-time" onClick={() => onSeek(step.demo_time_start as number)}>{clock(step.demo_time_start)}{step.demo_time_end != null && ` - ${clock(step.demo_time_end)}`}</button>}
      footer={
        <>
          {!removed && (
            <label className="check"><input type="checkbox" checked={status === "required"} onChange={(event) => onChange({ requirement_status: event.target.checked ? "required" : "optional" })} /> Required</label>
          )}
          <button className={removed ? "btn small" : "btn small danger"} onClick={() => onChange({ removed: !removed })}>{removed ? "Restore step" : "Remove step"}</button>
        </>
      }
    >
      {removed ? (
        <p className="muted small">Removed from the accepted procedure. It stays in the version history.</p>
      ) : (
        <StepFields
          objects={objects.join(", ") || "None identified"}
          action={<input className="field" value={edit.expected_action ?? step.expected_action} onChange={(event) => onChange({ expected_action: event.target.value })} />}
          completion={<input className="field" value={edit.completion_text ?? step.completion_text} onChange={(event) => onChange({ completion_text: event.target.value })} />}
          checks={
            <>
              {required.length > 0 ? required.map((item, position) => (
                <div key={`${item.id ?? "new"}-${position}`} className="condition check-row">
                  <div className="condition-meta">
                    <span className="origin">{item.origin === "human" ? "Human-specified" : item.source_evidence_class === "DIRECTLY_OBSERVED" ? "From Meta, directly observed" : item.source_evidence_class === "INFERRED" ? "From Meta, inferred. Please confirm" : "From Meta"}</span>
                    <button type="button" className="condition-remove" aria-label={`Remove check ${position + 1}`} title="Remove check" onClick={() => setConditions(required.filter((_, index) => index !== position))}>
                      <X size={14} strokeWidth={2} />
                    </button>
                  </div>
                  {item.relation ? (
                    <div className="row condition-edit">
                      <select className="field" aria-label={`Check ${position + 1} object`} value={item.subject_role ?? ""} onChange={(event) => setCondition(position, { subject_role: event.target.value })}>
                        <option value="" disabled>Object</option>
                        {roleOptions}
                      </select>
                      <select className="field" aria-label={`Check ${position + 1} relation`} value={item.relation} onChange={(event) => setCondition(position, { relation: event.target.value })}>
                        {Object.entries(relationLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                      </select>
                      <select className="field" aria-label={`Check ${position + 1} target`} value={item.target_role ?? ""} onChange={(event) => setCondition(position, { target_role: event.target.value })}>
                        <option value="" disabled>Target</option>
                        {roleOptions}
                      </select>
                    </div>
                  ) : (
                    <span>{conditionText(item, roles)}</span>
                  )}
                </div>
              )) : <p className="warning">No checkable condition. This step reports CANNOT VERIFY until you set one.</p>}
              <button type="button" className="btn small add-check" onClick={addCondition}>Add check</button>
            </>
          }
          order={others.length > 0 && <StepOrder number={index + 1} others={others} selected={orderAfter} dependencies={step.dependencies} onChange={(order_after) => onChange({ order_after })} />}
          uncertainty={<>Model uncertainty {Math.round(step.uncertainty * 100)}%{step.uncertainty_notes ? `. ${step.uncertainty_notes}` : ""}</>}
          edited={step.human_edited.length > 0 && <p className="edited">Human-edited: {step.human_edited.map((item) => item.replaceAll("_", " ")).join(", ")}</p>}
        />
      )}
    </StepFrame>
  );
}
