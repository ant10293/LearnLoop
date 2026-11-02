import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Bone, FieldBone } from "../components/Skeleton";
import { api, evidenceClass, formatTime, type TaskBundle, type TaskClip } from "../lib/api";
import { captureToLocalInput, localInputToCapture } from "../lib/captureTime";
import { humanize, labeledId } from "../lib/humanize";
import { clipPosition } from "../lib/taskOrder";

const MODES = ["", "fixed_camera", "constrained_head_mounted", "unconstrained_head_mounted"];
const SESSION_PLACEHOLDERS = 2;
const STEP_PLACEHOLDERS = 3;
const OBJECT_PLACEHOLDERS = 2;
const NOTE_PLACEHOLDERS = 2;

function TransitionCard({
  row,
  onSave,
}: {
  row: TaskBundle["transitions"][number] | null;
  onSave?: (hypothesis: string) => Promise<void>;
}) {
  return (
    <article className="panel space-y-2 p-3 text-sm" aria-hidden={row ? undefined : true}>
      {row ? (
        <span className={`tag ${evidenceClass(row.evidence_class)}`}>{humanize(row.evidence_class)}</span>
      ) : (
        <span className="tag"><Bone width="5rem" /></span>
      )}
      {row?.verified && <span className="tag">researcher verified</span>}
      <p><strong>Observed change.</strong> {row ? row.observed_change : <Bone width="50%" />}</p>
      <p><strong>Hypothesis.</strong> {row ? row.hypothesis || "None stored." : <Bone width="40%" />}</p>
      <p className="text-[var(--muted)]">{row ? row.gap_note : <Bone width="70%" />}</p>
      {row && onSave && <HypothesisEditor initial={row.hypothesis} onSave={onSave} />}
    </article>
  );
}

function PendingClip() {
  return (
    <div className="mt-2 rounded-md border border-[var(--line)] p-2 text-sm" aria-hidden="true">
      <div className="flex items-center gap-3">
        <div className="relative aspect-video w-40 shrink-0 overflow-hidden rounded bg-black">
          <Bone className="bone-fill absolute inset-0" />
        </div>
        <div className="min-w-0 flex-1 space-y-2">
          <Bone width="40%" />
          <Bone width="80%" />
          <div className="flex flex-wrap gap-2">
            <FieldBone compact />
            <FieldBone compact />
            <FieldBone compact />
          </div>
        </div>
      </div>
    </div>
  );
}

/** Sessions, chronology, and cross-session reading. The project page supplies the shared detail around this. */
export function TaskProjectSections({ projectId }: { projectId: string }) {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["task", projectId], queryFn: () => api.task(projectId), enabled: Boolean(projectId) });
  const [sessionName, setSessionName] = useState("");
  const [openSessions, setOpenSessions] = useState<Record<string, boolean>>({});
  const [compare, setCompare] = useState<string[]>([]);
  const [dragId, setDragId] = useState("");
  const refresh = async () => {
    await client.invalidateQueries({ queryKey: ["task", projectId] });
    await client.invalidateQueries({ queryKey: ["videos"] });
  };
  const task = query.data ?? null;
  const loading = task == null;
  const clipById = useMemo(() => new Map((task?.clips ?? []).map((clip) => [clip.id, clip])), [task]);
  if (query.isError) return <p className="text-sm text-[var(--rose)]">{(query.error as Error).message}</p>;
  const displayIds = task ? [
    ...task.order.ranked.flat(),
    ...task.clips.map((clip) => clip.id).filter((id) => !task.order.ranked.flat().includes(id)),
  ] : [];

  async function dropOn(targetId: string) {
    if (!dragId || dragId === targetId) return;
    const next = displayIds.filter((id) => id !== dragId);
    next.splice(next.indexOf(targetId), 0, dragId);
    await api.orderClips(projectId, next);
    setDragId("");
    await refresh();
  }

  return (
    <div className="space-y-5" aria-busy={loading || undefined}>
      {loading && <p className="sr-only">Loading chronology</p>}
      <section className="space-y-2">
        <h2 className="text-lg">Chronology</h2>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <input
              className="field w-40"
              value={sessionName}
              onChange={(event) => setSessionName(event.target.value)}
              placeholder="Session name"
              aria-label="New session name"
            />
            <button
              className="btn whitespace-nowrap"
              disabled={!sessionName.trim()}
              onClick={async () => {
                const name = sessionName.trim();
                if (!name) return;
                await api.createSession(projectId, name);
                setSessionName("");
                await refresh();
              }}
            >
              New session
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn whitespace-nowrap" onClick={async () => { await api.reconcileTask(projectId); await refresh(); }}>Reconcile</button>
            <a className="btn whitespace-nowrap" href={`/api/tasks/${projectId}/export`}>Export task</a>
          </div>
        </div>
        <p className="text-xs text-[var(--muted)]">
          {task ? (
            <>
              Method {task.order.method ? humanize(task.order.method) : "none"}. This order is {task.order.fact ? "marked as fact" : "not treated as fact"}.
              Drag a clip onto another to lock a manual order. Ambiguous positions stay highlighted.
            </>
          ) : <Bone width="80%" />}
        </p>
        {task?.order.notes?.map((note) => <p key={note} className="text-xs text-[var(--amber)]">{note}</p>)}
        <div className="grid gap-2">
          {loading && Array.from({ length: SESSION_PLACEHOLDERS }, (_, index) => (
            <div key={index} className="panel p-3" aria-hidden="true">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-left font-medium"><Bone width="9rem" /></span>
                <span className="btn">Lock session order</span>
              </div>
              <PendingClip />
            </div>
          ))}
          {task && task.sessions.map((recording) => {
            const open = openSessions[recording.id] !== false;
            const members = displayIds.map((id) => clipById.get(id)).filter((clip): clip is TaskClip => Boolean(clip && clip.session_id === recording.id));
            return (
              <div key={recording.id} className="panel p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <button className="text-left font-medium" onClick={() => setOpenSessions((state) => ({ ...state, [recording.id]: !open }))}>
                    {open ? "▾" : "▸"} {recording.name}
                  </button>
                  <button
                    className="btn"
                    onClick={async () => { await api.patchSession(projectId, recording.id, { locked: !recording.locked }); await refresh(); }}
                  >
                    {recording.locked ? "Unlock session" : "Lock session order"}
                  </button>
                </div>
                {open && members.map((clip) => (
                  <ClipRow
                    key={clip.id}
                    task={task}
                    clip={clip}
                    onDragStart={() => setDragId(clip.id)}
                    onDrop={() => dropOn(clip.id)}
                    onChange={refresh}
                  />
                ))}
              </div>
            );
          })}
        </div>
        {task && task.context_hypothesis?.evidence?.length > 0 && (
          <div className="panel p-3 text-sm">
            <div className="tag inferred">Context hypothesis, not a fact</div>
            {task.context_hypothesis.evidence.map((item, index) => <p key={index} className="mt-2">{item.support}</p>)}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Timeline</h2>
        <p className="text-xs text-[var(--muted)]">
          {task ? <>{task.time_basis.project_time}. {task.time_basis.clip_time}.</> : <Bone width="70%" />}
        </p>
        <ol className="space-y-2">
          {loading && Array.from({ length: STEP_PLACEHOLDERS }, (_, index) => (
            <li key={index} className="panel p-3 text-sm" aria-hidden="true">
              <div className="flex flex-wrap items-center gap-2">
                <span className="tag"><Bone width="4.5rem" /></span>
                <Bone width="40%" />
              </div>
            </li>
          ))}
          {task?.workflow.steps.map((step) => (
            <li key={step.id} className="panel p-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className={`tag ${evidenceClass(step.evidence_class)}`}>{humanize(step.kind)}</span>
                <span>{humanize(step.label)}</span>
              </div>
              {step.note && <p className="mt-1 text-[var(--muted)]">{step.note}</p>}
              {step.observed_change && <p className="mt-1">Observed change: {step.observed_change}</p>}
              {step.hypothesis && <p className="mt-1 text-[var(--amber)]">{step.hypothesis}</p>}
              <div className="mt-2 flex flex-wrap gap-2">
                {(step.video_ids ?? []).map((videoId) => {
                  const clip = clipById.get(videoId);
                  if (!clip) return null;
                  const href = clip.run_id ? `/runs/${clip.run_id}?t=0` : `/videos/${clip.id}?t=0`;
                  return <Link key={videoId} className="btn" to={href}>Open {clip.filename} at clip time 0:00.00</Link>;
                })}
              </div>
            </li>
          ))}
        </ol>
        {task?.clips.flatMap((clip) => clip.actions.map((action) => (
          <Link
            key={`${clip.id}-${action.local_id}`}
            className="panel block p-3 text-sm"
            to={clip.run_id ? `/runs/${clip.run_id}?t=${action.start}` : `/videos/${clip.id}?t=${action.start}`}
          >
            <span className={`tag ${evidenceClass(action.evidence_class)}`}>clip action</span> {humanize(action.action_type)} · clip time {formatTime(action.start)}–{formatTime(action.end)} · {clip.filename}
          </Link>
        )))}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg">Cross-session objects</h2>
        {loading && Array.from({ length: OBJECT_PLACEHOLDERS }, (_, index) => (
          <article key={index} className="panel p-3" aria-hidden="true">
            <div className="flex flex-wrap items-center gap-2">
              <Bone width="0.75rem" />
              <strong><Bone width="8rem" /></strong>
            </div>
            <p className="mt-1 text-sm"><Bone width="70%" /></p>
          </article>
        ))}
        {task?.canonical_objects.map((object) => (
          <article key={object.id} className="panel p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-block h-3 w-3 rounded-full" style={{ background: object.color }} />
              <strong>{labeledId(object.local_id, object.label)}</strong>
              {object.uncertain && <span className="tag unknown">identity uncertain</span>}
            </div>
            <p className="mt-1 text-sm text-[var(--muted)]">{object.description}</p>
            <div className="mt-3 grid gap-2 md:grid-cols-3">
              {object.history.map((entry) => {
                const clip = clipById.get(entry.video_id);
                const key = `${object.id}:${entry.video_id}:${entry.object_local_id}`;
                return (
                  <button
                    key={key}
                    className={`panel p-2 text-left text-sm ${compare.includes(key) ? "border-[var(--teal)]" : ""}`}
                    onClick={() => setCompare((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current.slice(-1), key])}
                  >
                    {entry.image_url && <img src={entry.image_url} alt="" className="mb-2 h-28 w-full object-contain bg-black" />}
                    <div>{clip?.filename ?? entry.video_id}</div>
                    <div className="text-[var(--muted)]">Before: {humanize(entry.state_before) || "undescribed"}</div>
                    <div className="text-[var(--muted)]">After: {humanize(entry.state_after) || "undescribed"}</div>
                    <div className="text-xs">{humanize(entry.status)}{entry.uncertain ? " · uncertain" : ""}</div>
                  </button>
                );
              })}
            </div>
          </article>
        ))}
        {task && compare.length === 2 && <ComparePane task={task} keys={compare} />}
        <div className="grid gap-2">
          {task?.links.filter((link) => link.status !== "rejected").map((link) => (
            <div key={link.id} className="panel flex flex-wrap items-center justify-between gap-2 p-3 text-sm">
              <div>
                {clipById.get(link.video_id)?.filename} / {humanize(link.object_local_id)} · {humanize(link.status)}
                {link.uncertain && " · uncertain"}
                <div className="text-xs text-[var(--muted)]">{link.note}</div>
              </div>
              <div className="flex gap-2">
                <button className="btn" onClick={async () => { await api.confirmLink(projectId, link.id); await refresh(); }}>Confirm</button>
                <button className="btn" onClick={async () => { await api.rejectLink(projectId, link.id); await refresh(); }}>Reject</button>
                <button className="btn" onClick={async () => { await api.splitIdentity(projectId, link.id); await refresh(); }}>Split</button>
              </div>
            </div>
          ))}
        </div>
        {task && task.canonical_objects.length >= 2 && (
          <button
            className="btn"
            onClick={async () => {
              await api.mergeIdentity(projectId, task.canonical_objects[0].id, task.canonical_objects[1].id);
              await refresh();
            }}
          >
            Merge first two canonical objects
          </button>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Inter-session changes</h2>
        {loading && Array.from({ length: NOTE_PLACEHOLDERS }, (_, index) => (
          <TransitionCard key={index} row={null} />
        ))}
        {task?.transitions.map((row) => (
          <TransitionCard
            key={row.id}
            row={row}
            onSave={async (hypothesis) => {
              await api.patchTransition(projectId, row.id, { hypothesis, verified: true });
              await refresh();
            }}
          />
        ))}
        {task?.workflow.procedural_dependencies?.map((edge) => (
          <p key={`${edge.before}-${edge.after}`} className="text-sm text-[var(--amber)]">
            Procedural dependency is inferred and {edge.accepted ? "accepted" : "not accepted"}: {edge.note}
          </p>
        ))}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg">Project readings</h2>
        {loading && Array.from({ length: NOTE_PLACEHOLDERS }, (_, index) => (
          <div key={index} className="panel p-3 text-sm" aria-hidden="true">
            <div><Bone width="40%" /></div>
            <p className="text-[var(--muted)]"><Bone width="70%" /></p>
          </div>
        ))}
        {task?.revisions.map((revision) => (
          <div key={revision.id} className="panel p-3 text-sm">
            <div>Version {revision.version} · {humanize(revision.trigger)}</div>
            <p className="text-[var(--muted)]">{revision.change_summary}</p>
          </div>
        ))}
      </section>
    </div>
  );
}

function CaptureTimeField({
  clip,
  projectId,
  onChange,
}: {
  clip: TaskClip;
  projectId: string;
  onChange: () => Promise<void>;
}) {
  const stored = captureToLocalInput(clip.capture_time_raw, clip.capture_time_utc);
  const [value, setValue] = useState(stored);
  useEffect(() => setValue(stored), [stored]);

  async function commit(next: string) {
    if (!next || next === stored) return;
    await api.patchClip(projectId, clip.id, { capture_time_raw: localInputToCapture(next) });
    await onChange();
  }

  return (
    <input
      type="datetime-local"
      step={1}
      className="field w-auto"
      value={value}
      aria-label={`Capture time for ${clip.filename}`}
      onChange={(event) => setValue(event.target.value)}
      onBlur={(event) => { void commit(event.currentTarget.value); }}
    />
  );
}

function ClipRow({
  task,
  clip,
  onDragStart,
  onDrop,
  onChange,
}: {
  task: TaskBundle;
  clip: TaskClip;
  onDragStart: () => void;
  onDrop: () => void;
  onChange: () => Promise<void>;
}) {
  const position = clipPosition(task.order, clip.id);
  const highlight = position.ambiguous || clip.order_ambiguous;
  return (
    <div
      className={`mt-2 rounded-md border p-2 text-sm ${highlight ? "border-[var(--amber)]" : "border-[var(--line)]"}`}
      draggable
      onDragStart={onDragStart}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        onDrop();
      }}
    >
      <div className="flex items-center gap-3">
        {clip.thumbnail_url ? (
          <img src={clip.thumbnail_url} alt="" draggable={false} className="aspect-video w-40 shrink-0 rounded bg-black object-contain" />
        ) : (
          <div className="aspect-video w-40 shrink-0 rounded bg-black" aria-hidden="true" />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <strong>{clip.filename}</strong>
            {highlight && <span className="tag inferred">{position.tied ? "tied timestamp" : "ambiguous position"}</span>}
            {clip.order_locked && <span className="tag">locked</span>}
            <span className="text-xs text-[var(--muted)]">{humanize(clip.research_mode) || "camera unset"}</span>
          </div>
          <p className="mt-1 text-xs text-[var(--muted)]">
            Capture {clip.capture_time_utc || clip.capture_time_raw || "not in the file"} · source {humanize(clip.timestamp_source)} · reliability {humanize(clip.timestamp_reliability)}
            <br />
            File modified {clip.file_mtime || "unknown"} · uploaded {clip.uploaded_at} · clip duration {formatTime(clip.duration_sec)}
          </p>
          <p className="text-xs text-[var(--muted)]">{clip.timestamp_note}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <select
              className="field w-auto"
              value={clip.session_id ?? ""}
              aria-label={`Session for ${clip.filename}`}
              onChange={async (event) => { await api.patchClip(task.project.id, clip.id, { session_id: event.target.value }); await onChange(); }}
            >
              {task.sessions.map((recording) => <option key={recording.id} value={recording.id}>{recording.name}</option>)}
            </select>
            <select
              className="field w-auto"
              value={clip.research_mode}
              aria-label={`Camera mode for ${clip.filename}`}
              onChange={async (event) => { await api.patchClip(task.project.id, clip.id, { research_mode: event.target.value }); await onChange(); }}
            >
              {MODES.map((item) => <option key={item || "unset"} value={item}>{humanize(item) || "Unset"}</option>)}
            </select>
            <CaptureTimeField clip={clip} projectId={task.project.id} onChange={onChange} />
          </div>
        </div>
      </div>
    </div>
  );
}

function HypothesisEditor({ initial, onSave }: { initial: string; onSave: (value: string) => Promise<void> }) {
  const [value, setValue] = useState(initial);
  return (
    <div className="flex flex-wrap gap-2">
      <input className="field" value={value} onChange={(event) => setValue(event.target.value)} aria-label="Hypothesis" />
      <button className="btn" onClick={() => onSave(value)}>Save hypothesis</button>
    </div>
  );
}

function ComparePane({ task, keys }: { task: TaskBundle; keys: string[] }) {
  const entries = keys.map((key) => {
    for (const object of task.canonical_objects) {
      for (const entry of object.history) {
        if (`${object.id}:${entry.video_id}:${entry.object_local_id}` === key) {
          return { object, entry, clip: task.clips.find((clip) => clip.id === entry.video_id) };
        }
      }
    }
    return null;
  });
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {entries.map((item) => item && (
        <figure key={`${item.object.id}-${item.entry.video_id}`} className="panel p-3">
          {item.entry.image_url ? <img src={item.entry.image_url} alt="" className="max-h-64 w-full object-contain bg-black" /> : <div className="text-sm text-[var(--muted)]">No stored crop for this appearance.</div>}
          <figcaption className="mt-2 text-sm">
            {humanize(item.object.local_id)} · {item.clip?.filename}
            <div>Before: {humanize(item.entry.state_before) || "undescribed"}</div>
            <div>After: {humanize(item.entry.state_after) || "undescribed"}</div>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
