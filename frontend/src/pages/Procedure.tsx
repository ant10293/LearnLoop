import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ErrorNote, Film, Layout, ReviewTag } from "../components/ui";
import { api, clock, conditionText, relationLabels, type Condition, type Role, type Template, type TemplatePatch, type TemplateStep } from "../lib/learnloop";

type StepEdit = Partial<Pick<TemplateStep, "name" | "expected_action" | "completion_text" | "removed" | "requirement_status">> & { completion_conditions?: Partial<Condition>[] };

export function ProcedurePage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const procedure = useQuery({ queryKey: ["procedure", id], queryFn: () => api.procedure(id) });
  const template = procedure.data?.template;
  const [steps, setSteps] = useState<Record<string, StepEdit>>({});
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [title, setTitle] = useState("");
  const [seek, setSeek] = useState({ time: null as number | null, tick: 0 });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    setSteps({});
    setLabels({});
    setTitle("");
  }, [template?.id]);

  if (procedure.isPending) return <Layout procedureId={id}><p className="muted" role="status">Loading the procedure</p></Layout>;
  if (!template) {
    return (
      <Layout procedureId={id} title={procedure.data?.name}>
        <h1>Procedure</h1>
        <p className="lead">Nothing has been learned yet.</p>
        <Link className="btn primary" to={`/procedures/${id}/teach`}>Teach this procedure</Link>
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

  const required = template.document.steps.filter((step) => !step.removed && (steps[step.step_id]?.requirement_status ?? step.requirement_status) === "required");
  return (
    <Layout procedureId={id} title={procedure.data?.name}>
      <div className="heading">
        <div>
          <h1>Learned procedure</h1>
          <p className="muted small">Version {template.version} · Proposed by Meta from the demonstration. Each edit you save becomes a new version; the original is kept.</p>
        </div>
        <div className="row">
          <ReviewTag state={template.review_state} />
          {dirty && <button className="btn" disabled={saving} onClick={() => void save(false)}>Save edits</button>}
          {(dirty || template.review_state !== "accepted") && (
            <button className="btn primary" disabled={saving || required.length === 0} onClick={() => void save(true)}>Accept procedure</button>
          )}
        </div>
      </div>
      <ErrorNote message={error} />
      {required.length === 0 && <p className="warning">Mark at least one step as required before accepting.</p>}
      <div className="split procedure">
        <div className="sticky">
          <Film src={demo?.video.playback_url ?? ""} time={seek.time} seek={seek.tick} label="Demonstration" caption={template.document.summary} />
          <section className="panel">
            <h2>Objects</h2>
            <p className="muted small">Correct a label if the model named an object wrongly.</p>
            {roles.map((role) => (
              <label key={role.role_id} className="object">
                <input className="field" aria-label={`Label for ${role.label}`} value={labels[role.role_id] ?? role.label} onChange={(event) => setLabels((current) => ({ ...current, [role.role_id]: event.target.value }))} />
                {role.original_label && <span className="edited">Human-corrected from “{role.original_label}”</span>}
              </label>
            ))}
          </section>
        </div>
        <ol className="steps">
          {template.document.steps.map((step, index) => (
            <StepCard key={step.step_id} index={index} step={step} edit={steps[step.step_id] ?? {}} roles={roles} template={template} onChange={(edit) => change(step.step_id, edit)} onSeek={(time) => setSeek((current) => ({ time, tick: current.tick + 1 }))} />
          ))}
        </ol>
      </div>
    </Layout>
  );
}

function StepCard({ index, step, edit, roles, onChange, onSeek }: { index: number; step: TemplateStep; edit: StepEdit; roles: Role[]; template: Template; onChange: (edit: StepEdit) => void; onSeek: (time: number) => void }) {
  const status = edit.requirement_status ?? step.requirement_status;
  const removed = edit.removed ?? step.removed;
  const conditions = (edit.completion_conditions ?? step.completion_conditions) as Partial<Condition>[];
  const required = conditions.filter((item) => item.evidence_requirement !== "optional");
  const relation = required.find((item) => item.relation);
  const objects = step.object_roles.map((role) => roles.find((item) => item.role_id === role)?.label ?? role);
  const setCondition = (next: Partial<Condition>) => onChange({ completion_conditions: [{ subject_role: relation?.subject_role ?? roles[0]?.role_id, target_role: relation?.target_role ?? roles[1]?.role_id, relation: relation?.relation ?? "attached", ...next }] });
  return (
    <li className={`step ${removed ? "removed" : ""}`}>
      <header>
        <span className="index">{index + 1}</span>
        <input className="field title grow" aria-label={`Step ${index + 1} name`} value={edit.name ?? step.name} onChange={(event) => onChange({ name: event.target.value })} disabled={removed} />
        {step.demo_time_start != null && <button className="btn small mono" onClick={() => onSeek(step.demo_time_start as number)}>{clock(step.demo_time_start)}</button>}
      </header>
      {removed ? (
        <p className="muted small">Removed from the accepted procedure. It stays in the version history.</p>
      ) : (
        <>
          <p className="small"><span className="muted">Objects</span> {objects.join(", ") || "None identified"}</p>
          <label className="stack small"><span className="muted">Expected action</span>
            <input className="field" value={edit.expected_action ?? step.expected_action} onChange={(event) => onChange({ expected_action: event.target.value })} />
          </label>
          <label className="stack small"><span className="muted">Observable completion</span>
            <input className="field" value={edit.completion_text ?? step.completion_text} onChange={(event) => onChange({ completion_text: event.target.value })} />
          </label>
          <div className="stack small">
            <span className="muted">What the verifier checks</span>
            {required.length > 0 ? required.map((item, position) => (
              <p key={`${item.id ?? "new"}-${position}`} className="condition">
                {conditionText(item, roles)}
                <span className="origin">{item.origin === "human" || edit.completion_conditions ? "Human-specified" : item.source_evidence_class === "DIRECTLY_OBSERVED" ? "From Meta, directly observed" : "From Meta"}</span>
              </p>
            )) : <p className="warning">No checkable condition. This step reports CANNOT VERIFY until you set one.</p>}
            <div className="row condition-edit">
              <select className="field" aria-label="Object" value={relation?.subject_role ?? ""} onChange={(event) => setCondition({ subject_role: event.target.value })}>
                <option value="" disabled>Object</option>
                {roles.map((role) => <option key={role.role_id} value={role.role_id}>{role.label}</option>)}
              </select>
              <select className="field" aria-label="Relation" value={relation?.relation ?? ""} onChange={(event) => setCondition({ relation: event.target.value })}>
                <option value="" disabled>Relation</option>
                {Object.entries(relationLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
              <select className="field" aria-label="Target" value={relation?.target_role ?? ""} onChange={(event) => setCondition({ target_role: event.target.value })}>
                <option value="" disabled>Target</option>
                {roles.map((role) => <option key={role.role_id} value={role.role_id}>{role.label}</option>)}
              </select>
            </div>
          </div>
          <p className="muted small">Model uncertainty {Math.round(step.uncertainty * 100)}%{step.uncertainty_notes ? `. ${step.uncertainty_notes}` : ""}</p>
          {step.human_edited.length > 0 && <p className="edited">Human-edited: {step.human_edited.map((item) => item.replaceAll("_", " ")).join(", ")}</p>}
        </>
      )}
      <footer className="row">
        {!removed && (
          <label className="check"><input type="checkbox" checked={status === "required"} onChange={(event) => onChange({ requirement_status: event.target.checked ? "required" : "optional" })} /> Required</label>
        )}
        <button className="btn small" onClick={() => onChange({ removed: !removed })}>{removed ? "Restore step" : "Remove step"}</button>
      </footer>
    </li>
  );
}
