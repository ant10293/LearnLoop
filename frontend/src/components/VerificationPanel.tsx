import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bone, FieldBone } from "./Skeleton";
import { api, formatTime } from "../lib/api";
import { humanize } from "../lib/humanize";
import {
  markerColor,
  markerKind,
  recordingsForVerification,
  verdictClass,
  type VerificationRecording,
  type VerificationStep,
  type WorkflowTemplate,
} from "../lib/verification";

/** Learned procedures and evidence-backed execution checks for any project. */
export function VerificationPanel({ projectId }: { projectId: string }) {
  const client = useQueryClient();
  const videos = useQuery({
    queryKey: ["videos", projectId],
    queryFn: () => api.videos(projectId),
    enabled: Boolean(projectId),
  });
  const runs = useQuery({
    queryKey: ["runs", projectId, "completed"],
    queryFn: () => api.runs({ project_id: projectId, status: "completed" }),
    enabled: Boolean(projectId),
  });
  const clips = useMemo(
    () => recordingsForVerification(videos.data ?? [], runs.data ?? []),
    [videos.data, runs.data],
  );
  const templates = useQuery({
    queryKey: ["workflow-templates", projectId],
    queryFn: () => api.workflowTemplates(projectId),
  });
  const verifications = useQuery({
    queryKey: ["verifications", projectId],
    queryFn: () => api.verifications(projectId),
  });
  const clipsPending = (videos.data === undefined && !videos.isError) || (runs.data === undefined && !runs.isError);
  const templatesPending = templates.data === undefined && !templates.isError;
  const checksPending = verifications.data === undefined && !verifications.isError;
  const analyzed = clips.filter((clip) => clip.run_id);
  const [demoRun, setDemoRun] = useState("");
  const [templateId, setTemplateId] = useState("");
  const [executionRun, setExecutionRun] = useState("");
  const [strategy, setStrategy] = useState("hybrid");
  const [reinspect, setReinspect] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [error, setError] = useState("");
  const detail = useQuery({
    queryKey: ["verification", selectedId],
    queryFn: () => api.verification(selectedId),
    enabled: Boolean(selectedId),
    refetchInterval: (query) => (query.state.data?.status === "waiting" ? 2000 : false),
  });
  const template = templates.data?.find((item) => item.id === templateId) ?? templates.data?.[0];

  async function refresh() {
    await client.invalidateQueries({ queryKey: ["workflow-templates", projectId] });
    await client.invalidateQueries({ queryKey: ["verifications", projectId] });
    if (selectedId) await client.invalidateQueries({ queryKey: ["verification", selectedId] });
  }

  async function learn() {
    setError("");
    try {
      const created = await api.learnWorkflow(projectId, demoRun, true, false);
      setTemplateId(created.id);
      await refresh();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function verify() {
    if (!template) return;
    setError("");
    try {
      const created = await api.startVerification(template.id, executionRun, strategy, reinspect);
      setSelectedId(created.id);
      await refresh();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <section className="space-y-3">
      <h2 className="text-lg">Verification</h2>
      <p className="text-xs text-[var(--muted)]">
        Learn a procedure from an analyzed demonstration, then check another recording against that template version.
        Provider and perception settings are the ones already stored on the execution run.
      </p>
      {error && <p className="text-sm text-[var(--rose)]">{error}</p>}
      <div className="panel flex flex-wrap items-end gap-2 p-3">
        <label className="text-sm">
          Demonstration
          {clipsPending ? <FieldBone compact className="mt-1" /> : (
            <select className="field mt-1 block" value={demoRun} onChange={(event) => setDemoRun(event.target.value)}>
              <option value="">Analyzed clip</option>
              {analyzed.map((clip) => (
                <option key={clip.run_id} value={clip.run_id}>{clip.filename}</option>
              ))}
            </select>
          )}
        </label>
        <button className="btn" disabled={!demoRun} onClick={() => void learn()}>Learn workflow</button>
      </div>
      {template && (
        <TemplateCard
          template={template}
          onChange={async (next) => {
            setTemplateId(next);
            await refresh();
          }}
        />
      )}
      <div className="panel flex flex-wrap items-end gap-2 p-3">
        <label className="text-sm">
          Template
          {templatesPending ? <FieldBone compact className="mt-1" /> : (
            <select className="field mt-1 block" value={template?.id ?? ""} onChange={(event) => setTemplateId(event.target.value)}>
              {(templates.data ?? []).map((item) => (
                <option key={item.id} value={item.id}>{item.name} · v{item.version}</option>
              ))}
            </select>
          )}
        </label>
        <label className="text-sm">
          Execution
          {clipsPending ? <FieldBone compact className="mt-1" /> : (
            <select className="field mt-1 block" value={executionRun} onChange={(event) => setExecutionRun(event.target.value)}>
              <option value="">Analyzed clip</option>
              {analyzed.map((clip) => (
                <option key={clip.run_id} value={clip.run_id}>{clip.filename}</option>
              ))}
            </select>
          )}
        </label>
        <label className="text-sm">
          Strategy
          <select className="field mt-1 block" value={strategy} onChange={(event) => setStrategy(event.target.value)}>
            <option value="hybrid">Hybrid</option>
            <option value="tracked">Tracked evidence</option>
            <option value="vlm_align">Model plus alignment</option>
            <option value="vlm_only">Model proposal only</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={reinspect} onChange={(event) => setReinspect(event.target.checked)} />
          Plan reinspection
        </label>
        <button className="btn primary" disabled={!template || !executionRun} onClick={() => void verify()}>
          Verify new execution
        </button>
      </div>
      <div className="flex flex-wrap gap-2" aria-busy={checksPending || undefined}>
        {checksPending && <p className="sr-only">Loading verifications</p>}
        {checksPending ? Array.from({ length: 2 }, (_, index) => (
          <span key={index} className="btn" aria-hidden="true"><Bone width="7rem" /></span>
        )) : (verifications.data ?? []).map((item) => (
          <button key={item.id} className={`btn ${item.id === selectedId ? "primary" : ""}`} onClick={() => setSelectedId(item.id)}>
            {humanize(item.overall)} · v{item.template_version}
          </button>
        ))}
      </div>
      {detail.data && (
        <Report
          clips={clips}
          template={templates.data?.find((item) => item.id === detail.data?.template_id)}
          verification={detail.data}
          onChange={refresh}
        />
      )}
    </section>
  );
}

function TemplateCard({ template, onChange }: { template: WorkflowTemplate; onChange: (id: string) => Promise<void> }) {
  const roles = new Map(template.document.roles.map((role) => [role.role_id, role.label]));
  return (
    <article className="panel space-y-2 p-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <strong>{template.name}</strong>
        <span className="text-xs text-[var(--muted)]">Version {template.version} · {humanize(template.acceptance_status)}</span>
      </div>
      {template.document.notes?.slice(0, 2).map((note) => (
        <p key={note} className="text-xs text-[var(--muted)]">{note}</p>
      ))}
      {template.document.steps.map((step) => (
        <div key={step.step_id} className="border-t border-[var(--line)] pt-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>{step.name}</span>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={step.required}
                onChange={async (event) => {
                  const revised = await api.reviseWorkflow(template.id, {
                    steps: [{ step_id: step.step_id, requirement_status: event.target.checked ? "required" : "optional" }],
                  });
                  await onChange(revised.id);
                }}
              />
              Required
            </label>
          </div>
          <p className="text-xs text-[var(--muted)]">
            Roles: {step.object_roles.map((role) => roles.get(role) || humanize(role)).join(", ") || "none"}
            {step.expected_action ? ` · ${step.expected_action}` : ""}
          </p>
          {step.completion_conditions.filter((item) => item.evidence_requirement === "required").map((item) => (
            <p key={`${step.step_id}-${item.type}-${item.relation}-${item.state_label}`} className="text-xs">
              Completion: {roles.get(item.subject_role) || item.subject_role} {item.relation || item.state_label} {roles.get(item.target_role) || ""}
            </p>
          ))}
          {step.dependencies.filter((item) => item.constrains_order).map((item) => (
            <p key={item.step_id} className="text-xs">Depends on {item.step_id} ({humanize(item.category)})</p>
          ))}
          {step.uncertainty_notes && <p className="text-xs text-[var(--amber)]">{step.uncertainty_notes}</p>}
        </div>
      ))}
      {template.acceptance_status !== "accepted" && (
        <button
          className="btn"
          onClick={async () => {
            const revised = await api.reviseWorkflow(template.id, { acceptance_status: "accepted" });
            await onChange(revised.id);
          }}
        >
          Accept template
        </button>
      )}
    </article>
  );
}

function Report({
  clips,
  template,
  verification,
  onChange,
}: {
  clips: VerificationRecording[];
  template?: WorkflowTemplate;
  verification: NonNullable<Awaited<ReturnType<typeof api.verification>>>;
  onChange: () => Promise<void>;
}) {
  const report = verification.report;
  const [stepId, setStepId] = useState(report?.steps?.[0]?.step_id ?? "");
  const [roleObject, setRoleObject] = useState("");
  const step = report?.steps.find((item) => item.step_id === stepId) ?? report?.steps[0];
  const demo = clips.find((clip) => clip.id === (report?.demonstration_video_id || template?.source_video_id));
  const execution = clips.find((clip) => clip.id === verification.execution_video_id);
  const duration = Math.max(execution?.duration_sec || 1, 0.1);
  useEffect(() => {
    if (report?.steps?.length && !report.steps.some((item) => item.step_id === stepId)) {
      setStepId(report.steps[0].step_id);
    }
  }, [report, stepId]);

  return (
    <article className="space-y-3">
      <div className="panel p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`tag ${verdictClass(verification.overall)}`}>{humanize(verification.overall)}</span>
          <span className="text-sm">{humanize(verification.status)}</span>
          <span className="text-xs text-[var(--muted)]">
            Template v{verification.template_version} · {verification.config.provider_id || "cached"} / {verification.config.model_id || "cached"}
            {verification.config.perception?.strategy ? ` · ${humanize(verification.config.perception.strategy)}` : ""}
          </span>
        </div>
        {verification.error && <p className="mt-2 text-sm text-[var(--rose)]">{verification.error}</p>}
        {report?.score && (
          <p className="mt-2 text-xs text-[var(--muted)]">
            Verified {report.score.verified}, failed {report.score.failed}, unresolved {report.score.unresolved}.
            {report.score.value == null ? " No score." : ` Score ${report.score.value}.`} {report.score.explanation}
          </p>
        )}
        {report?.warnings?.map((warning) => <p key={warning} className="mt-1 text-xs text-[var(--amber)]">{warning}</p>)}
        <p className="mt-1 text-xs text-[var(--muted)]">{report?.timing_policy}</p>
      </div>
      <ol className="space-y-2">
        {report?.steps.map((item) => (
          <li key={item.step_id}>
            <button className={`panel w-full p-3 text-left text-sm ${step?.step_id === item.step_id ? "border-[var(--teal)]" : ""}`} onClick={() => setStepId(item.step_id)}>
              <span className={`tag ${verdictClass(item.verdict)}`}>{humanize(item.verdict)}</span> {item.name}
              {item.error_category ? ` · ${humanize(item.error_category)}` : ""}
              <span className="mt-1 block text-xs text-[var(--muted)]">{humanize(item.execution_state)}</span>
            </button>
          </li>
        ))}
      </ol>
      {step && (
        <StepDetail
          step={step}
          demoSrc={demo?.playback_url ?? ""}
          executionSrc={execution?.playback_url ?? ""}
          duration={duration}
          steps={report?.steps ?? []}
          roles={report?.roles ?? []}
          roleObject={roleObject}
          onRoleObject={setRoleObject}
          onSelect={setStepId}
          unexpected={report?.unexpected_actions ?? []}
          onCorrect={async (verdict, note) => {
            await api.correctVerificationStep(verification.id, step.step_id, verdict, note);
            await onChange();
          }}
          onRole={async (roleId) => {
            if (!roleObject.trim()) return;
            await api.correctVerificationRole(verification.id, roleId, roleObject.trim());
            await onChange();
          }}
          onInspect={async () => {
            await api.requestReinspection(verification.id, step.step_id);
            await onChange();
          }}
        />
      )}
    </article>
  );
}

function StepDetail({
  step,
  demoSrc,
  executionSrc,
  duration,
  steps,
  roles,
  roleObject,
  onRoleObject,
  onSelect,
  unexpected,
  onCorrect,
  onRole,
  onInspect,
}: {
  step: VerificationStep;
  demoSrc: string;
  executionSrc: string;
  duration: number;
  steps: VerificationStep[];
  roles: { role_id: string; label: string; status: string; object_local_ids: string[]; note: string }[];
  roleObject: string;
  onRoleObject: (value: string) => void;
  onSelect: (stepId: string) => void;
  unexpected: { action_id: string; action_type: string; start: number; note: string }[];
  onCorrect: (verdict: string, note: string) => Promise<void>;
  onRole: (roleId: string) => Promise<void>;
  onInspect: () => Promise<void>;
}) {
  const [roleId, setRoleId] = useState(roles[0]?.role_id ?? "");
  const boxes = step.evidence.map((item) => item.bbox).filter((box): box is NonNullable<typeof box> => Boolean(box));
  const markers = useMemo(
    () => steps.flatMap((item) => (item.execution_time == null ? [] : [{ id: item.step_id, time: item.execution_time, kind: markerKind(item), label: item.name }])),
    [steps],
  );
  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2">
        <Film src={demoSrc} time={step.demo_time_start} caption={`Demonstration · ${formatTime(step.demo_time_start)}`} boxes={[]} />
        <Film src={executionSrc} time={step.execution_time} caption={`Execution · ${formatTime(step.execution_time)}`} boxes={boxes} />
      </div>
      <div className="panel space-y-1 p-3 text-sm">
        <p>Expected: {step.expected || "No completion condition"}</p>
        <p>Observed: {step.observed}</p>
        <p className="text-[var(--muted)]">{step.explanation}</p>
        {step.uncertainty_notes && <p className="text-[var(--amber)]">{step.uncertainty_notes}</p>}
        {step.history?.map((item, index) => (
          <p key={`${item.verdict}-${index}`} className="text-xs text-[var(--muted)]">Earlier verdict {humanize(item.verdict)}. {item.note}</p>
        ))}
      </div>
      <div className="relative h-8 rounded bg-[#0d1016]">
        {markers.map((marker) => (
          <button
            key={marker.id}
            className="absolute top-1 h-6 w-2 rounded"
            style={{ left: `${Math.min(100, ((marker.time ?? 0) / duration) * 100)}%`, background: markerColor(marker.kind) }}
            title={`${marker.label} · ${marker.kind}`}
            onClick={() => onSelect(marker.id)}
          />
        ))}
        {unexpected.map((item) => (
          <button
            key={item.action_id}
            className="absolute top-1 h-6 w-2 rounded"
            style={{ left: `${Math.min(100, (item.start / duration) * 100)}%`, background: markerColor("unexpected") }}
            title={`${item.action_type} · ${item.note}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="btn" onClick={() => void onCorrect(step.verdict, "Confirmed by researcher.")}>Confirm verdict</button>
        <button className="btn" onClick={() => void onCorrect(step.verdict === "PASS" ? "FAIL" : "PASS", "Rejected by researcher.")}>Reject verdict</button>
        <button className="btn" onClick={() => void onCorrect("CANNOT_VERIFY", "Marked unverifiable by researcher.")}>Mark unverifiable</button>
        <button className="btn" onClick={() => void onInspect()}>Request inspection</button>
      </div>
      <div className="panel space-y-2 p-3 text-sm">
        {roles.map((role) => (
          <p key={role.role_id}>
            {role.label || humanize(role.role_id)} · {humanize(role.status)}
            {role.object_local_ids.length ? ` · ${role.object_local_ids.join(", ")}` : ""}
          </p>
        ))}
        <div className="flex flex-wrap gap-2">
          <select className="field" value={roleId} onChange={(event) => setRoleId(event.target.value)} aria-label="Role to correct">
            {roles.map((role) => (
              <option key={role.role_id} value={role.role_id}>{role.label || humanize(role.role_id)}</option>
            ))}
          </select>
          <input className="field" value={roleObject} onChange={(event) => onRoleObject(event.target.value)} placeholder="Execution object id" aria-label="Role correction object" />
          <button className="btn" disabled={!roleId || !roleObject.trim()} onClick={() => void onRole(roleId)}>
            Assign role
          </button>
        </div>
      </div>
    </div>
  );
}

function Film({ src, time, caption, boxes }: { src: string; time: number | null; caption: string; boxes: { x: number; y: number; w: number; h: number }[] }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node || time == null) return;
    const seek = () => {
      node.currentTime = time;
    };
    if (node.readyState >= 1) seek();
    else node.addEventListener("loadedmetadata", seek, { once: true });
  }, [src, time]);
  return (
    <figure className="space-y-2">
      <figcaption className="text-sm">{caption}</figcaption>
      {src ? (
        <div className="relative bg-black">
          <video ref={ref} className="w-full" src={src} controls />
          {boxes.map((box, index) => (
            <div
              key={`${box.x}-${box.y}-${index}`}
              className="pointer-events-none absolute border border-[var(--teal)]"
              style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }}
            />
          ))}
        </div>
      ) : (
        <p className="text-sm text-[var(--muted)]">No playback file is stored for this recording. The timestamp above is the evidence time.</p>
      )}
    </figure>
  );
}
