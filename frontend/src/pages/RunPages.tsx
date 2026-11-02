import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Star } from "lucide-react";
import { Link, NavLink, Outlet, useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { AssetPreview } from "../components/AssetPreview";
import { ShotFrame } from "../components/ShotFrame";
import { BeforeAfter } from "../components/BeforeAfter";
import { RunsTable } from "../components/RunsTable";
import { Bone, FieldBone } from "../components/Skeleton";
import { DecisionPanel } from "../components/DecisionPanel";
import { PerceptionPanel } from "../components/PerceptionPanel";
import { Timeline } from "../components/Timeline";
import { VideoStage } from "../components/VideoStage";
import { WorkflowGraph } from "../components/WorkflowGraph";
import { api, evidenceClass, formatTime, formatUsd } from "../lib/api";
import { analysisSetupLine, coverageLine } from "../lib/runOptions";
import { formatConfidence } from "../lib/confidence";
import { formatLogText } from "../lib/formatLog";
import { actionPhrase, humanize, humanizeText, labeledId, nameObjects, objectLabels } from "../lib/humanize";
import { datasetObservations, objectFrameTimeline, sightingCount } from "../lib/analysis";
import { objectScene } from "../lib/sceneSlice";
import { BundleContext, useOptionalBundle } from "../lib/bundleContext";
import type { Bundle, Observation } from "../lib/types";
import { useInspector } from "../stores/ui";
import { AudioPage } from "./Audio";
import { RecoveryPage } from "./Recovery";

function label(tab: string) {
  return tab.charAt(0).toUpperCase() + tab.slice(1);
}

const ANALYSIS_TABS = ["", "objects", "states", "workflow", "steps", "recovery", "pipeline", "audio", "evaluate", "logs"];
const OBJECT_GRID = "grid gap-3 sm:grid-cols-2 xl:grid-cols-4";
const OBJECT_PLACEHOLDERS = 4;
const INSPECT_LAYOUT = "grid min-h-[70vh] min-w-0 gap-3 xl:grid-cols-[280px_minmax(0,1fr)_320px]";
const INSPECT_SIDE = "panel flex max-h-[78vh] min-h-0 flex-col gap-3 overflow-hidden p-3";
const ACTION_LIST = "min-h-0 flex-1 space-y-1 overflow-y-auto";
const INSPECT_STAGE = "panel min-w-0 p-3";
const INSPECT_DETAIL = "panel max-h-[78vh] space-y-3 overflow-auto p-3 text-sm";
const ACTION_PLACEHOLDERS = 5;
const STATE_PLACEHOLDERS = 3;
const LINE_PLACEHOLDERS = 2;
const DEPENDENCY_PLACEHOLDERS = 3;
const STAGE_PLACEHOLDERS = 6;
const STEP_MEDIA_PLACEHOLDERS = 3;
const TRUTH_PLACEHOLDERS = 2;
const REQUEST_PLACEHOLDERS = 3;

/** The still that stands in for an object. Prefer its first appearance over the leading gallery asset. */
function objectThumbnail(object: Bundle["objects"][number] | null | undefined): string {
  if (!object) return "";
  const initial = object.images.find((image) => image.kind === "initial" && image.url);
  return initial?.url || object.primary_image_url;
}

/** Stills left out of the fallback gallery: boxed frames, and the crop that duplicates unobstructed. */
function isGalleryImage(kind: string) {
  if (kind === "boxed" || kind.endsWith("_boxed")) return false;
  if (kind === "crop" || kind === "primary") return false;
  return true;
}

const NAMED_STILLS = ["unobstructed", "initial", "final", "alternate"] as const;

type ObjectShot = { id: string; url: string; caption: string; confidence: number | null };

/** Every boxed observation, as a zoomed crop. Named stills keep their labels when the time matches. */
function objectShots(object: Bundle["objects"][number], observations: Observation[]): ObjectShot[] {
  const owned = datasetObservations(observations).filter((item) => item.object_local_id === object.local_id);
  const boxed = owned
    .filter((item) => item.bbox && item.frame_id && item.visibility !== "absent")
    .sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id));
  if (boxed.length) {
    return boxed.map((item) => ({
      id: item.id,
      url: `/api/media/observations/${item.id}/crop`,
      caption: observationCaption(item, object.images),
      confidence: item.confidence,
    }));
  }
  return object.images
    .filter((image) => image.url && isGalleryImage(image.kind))
    .map((image) => ({
      id: image.id,
      url: image.url,
      caption: `${humanize(image.kind)} · ${image.timestamp.toFixed(2)}s`,
      confidence: owned.find((item) => Math.abs(item.timestamp - image.timestamp) < 0.001)?.confidence ?? null,
    }));
}

function shotLabel(shot: ObjectShot): string {
  const view = `View ${shot.caption}`;
  if (shot.confidence == null) return view;
  return `${view}, ${formatConfidence(shot.confidence)} confidence`;
}

function observationCaption(observation: Observation, images: Bundle["objects"][number]["images"]): string {
  const time = `${observation.timestamp.toFixed(2)}s`;
  const roles = NAMED_STILLS.filter((kind) =>
    images.some((image) => image.kind === kind && Math.abs(image.timestamp - observation.timestamp) < 0.001),
  ).map((kind) => humanize(kind));
  if (roles.length) return `${roles.join(", ")} · ${time}`;
  if (observation.visibility && observation.visibility !== "visible") return `${humanize(observation.visibility)} · ${time}`;
  return time;
}

function analysisTab(pathname: string, runId: string): string {
  const base = `/runs/${runId}`;
  if (!pathname.startsWith(base)) return "";
  const head = pathname.slice(base.length).replace(/^\//, "").split("/")[0] ?? "";
  return (ANALYSIS_TABS as readonly string[]).includes(head) ? head : "";
}

const OpenInspectContext = createContext<() => void>(() => {});

function useOpenInspect() {
  return useContext(OpenInspectContext);
}

export function RunLayout() {
  const { runId = "" } = useParams();
  return <RunAnalysis runId={runId} />;
}

export function RunAnalysis({ runId, embedded = false }: { runId: string; embedded?: boolean }) {
  const [params] = useSearchParams();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [tab, setTab] = useState("objects");
  const openInspect = () => {
    if (embedded) setTab("");
    else navigate(`/runs/${runId}`);
  };
  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    refetchInterval: (q) => {
      const status = q.state.data?.run.status;
      return status === "pending" || status === "running" ? 1200 : false;
    },
  });
  useEffect(() => {
    const time = Number(params.get("t"));
    if (Number.isFinite(time) && params.get("t")) useInspector.getState().setTime(time);
  }, [params, runId]);
  const activeTab = embedded ? tab : analysisTab(pathname, runId);
  if (query.isError) return <p className="text-[var(--rose)]">{(query.error as Error).message}</p>;
  if (!query.data) return <AnalysisPending embedded={embedded} runId={runId} tab={activeTab} onTab={setTab} />;
  const bundle = query.data;
  return (
    <OpenInspectContext.Provider value={openInspect}>
      <BundleContext.Provider value={bundle}>
        <AnalysisFrame embedded={embedded} runId={runId} tab={activeTab} onTab={setTab} bundle={bundle}>
          {embedded ? <AnalysisTab tab={tab} /> : <Outlet />}
        </AnalysisFrame>
      </BundleContext.Provider>
    </OpenInspectContext.Provider>
  );
}

export function AnalysisPending({
  embedded,
  runId = "",
  tab,
  onTab,
}: {
  embedded: boolean;
  runId?: string;
  tab: string;
  onTab?: (tab: string) => void;
}) {
  return (
    <AnalysisFrame embedded={embedded} runId={runId} tab={tab} onTab={onTab} bundle={null} pending>
      <PendingPane tab={tab} />
    </AnalysisFrame>
  );
}

function AnalysisFrame({
  embedded,
  runId,
  tab,
  onTab,
  bundle,
  pending = false,
  children,
}: {
  embedded: boolean;
  runId: string;
  tab: string;
  onTab?: (tab: string) => void;
  bundle: Bundle | null;
  pending?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      <div>
        {!embedded && <Link className="text-sm text-[var(--muted)]" to="/runs">← Back to runs</Link>}
        <AnalysisHeader bundle={bundle} embedded={embedded} />
      </div>
      <AnalysisNav embedded={embedded} runId={runId} tab={tab} pending={pending} onTab={onTab} />
      {children}
      {pending && <span className="sr-only">Loading analysis</span>}
    </div>
  );
}

function AnalysisHeader({ bundle, embedded }: { bundle: Bundle | null; embedded: boolean }) {
  const Title = embedded ? "h2" : "h1";
  return (
    <header className={`${embedded ? "" : "mt-2 "}flex flex-wrap items-start justify-between gap-3`}>
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <Title className="text-xl font-medium">
            {bundle ? humanizeText(bundle.run.name) : <Bone width="14rem" />}
          </Title>
          {bundle ? (
            <>
              <span className="tag">{humanize(bundle.run.status)}</span>
              {bundle.run.fixture && <span className="tag fixture">Fixture — authored, not model output</span>}
              <span className="tag">{humanize(bundle.run.research_mode)}</span>
            </>
          ) : (
            <>
              <span className="tag"><Bone width="5.5rem" /></span>
              <span className="tag"><Bone width="6.5rem" /></span>
            </>
          )}
        </div>
        <p className="mt-1 text-xs text-[var(--muted)]">
          {bundle ? (
            analysisSetupLine(bundle)
          ) : (
            <Bone width="28rem" />
          )}
        </p>
        {bundle && coverageLine(bundle.coverage) && (
          <p className="mt-1 max-w-3xl text-xs text-[var(--muted)]" data-testid="coverage-line">
            {coverageLine(bundle.coverage)}
          </p>
        )}
        {bundle?.coverage?.temporal_coverage_note && (
          <p className="mt-1 max-w-3xl text-xs text-[var(--amber)]">{bundle.coverage.temporal_coverage_note}</p>
        )}
      </div>
      <RunActions bundle={bundle} embedded={embedded} />
    </header>
  );
}

function AnalysisNav({
  embedded,
  runId,
  tab,
  pending,
  onTab,
}: {
  embedded: boolean;
  runId: string;
  tab: string;
  pending: boolean;
  onTab?: (tab: string) => void;
}) {
  return (
    <nav className="flex gap-2 overflow-auto text-sm">
      {ANALYSIS_TABS.map((item) => {
        const name = label(item || "inspect");
        const key = item || "inspect";
        if (embedded) {
          const selected = tab === item;
          return (
            <button
              key={key}
              type="button"
              disabled={pending}
              className={`rounded-md px-2 py-1 ${selected ? "bg-[#1a221d] text-[var(--teal)]" : "text-[var(--muted)]"}`}
              onClick={() => onTab?.(item)}
            >
              {name}
            </button>
          );
        }
        return (
          <NavLink
            key={key}
            end={item === ""}
            to={item ? `/runs/${runId}/${item}` : `/runs/${runId}`}
            className={({ isActive }) => `rounded-md px-2 py-1 ${isActive ? "bg-[#1a221d] text-[var(--teal)]" : "text-[var(--muted)]"}`}
          >
            {name}
          </NavLink>
        );
      })}
    </nav>
  );
}

function PendingPane({ tab }: { tab: string }) {
  return <AnalysisTab tab={tab} />;
}

function AnalysisTab({ tab }: { tab: string }) {
  switch (tab) {
    case "objects":
      return <ObjectsPage />;
    case "states":
      return <StatesPage />;
    case "workflow":
      return <WorkflowPage />;
    case "steps":
      return <StepsPage />;
    case "recovery":
      return <RecoveryPage />;
    case "pipeline":
      return <PipelinePage />;
    case "audio":
      return <AudioPage />;
    case "evaluate":
      return <EvaluatePage />;
    case "logs":
      return <LogsPage />;
    default:
      return <InspectPage />;
  }
}

function RunActions({ bundle, embedded }: { bundle: Bundle | null; embedded: boolean }) {
  const client = useQueryClient();
  const navigate = useNavigate();
  if (!bundle) {
    return (
      <div className="flex flex-wrap gap-2" aria-hidden="true">
        <span className="btn flex items-center"><Star size={16} /></span>
        <span className="btn">Duplicate</span>
        <span className="btn">Cancel</span>
        <span className="btn">Delete</span>
        <span className="btn">Export archive</span>
        <span className="btn">Analysis JSON</span>
      </div>
    );
  }
  const refresh = () => client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
  const remove = async () => {
    await api.deleteRun(bundle.run.id);
    await client.invalidateQueries({ queryKey: ["runs"] });
    await client.invalidateQueries({ queryKey: ["dashboard"] });
    navigate(embedded ? "/" : "/runs");
  };
  const toggleStar = async () => {
    await api.setRunStarred(bundle.run.id, !bundle.run.starred);
    await refresh();
    await client.invalidateQueries({ queryKey: ["runs"] });
    await client.invalidateQueries({ queryKey: ["dashboard"] });
  };
  return (
    <div className="flex flex-wrap gap-2">
      <button
        className="btn flex items-center"
        aria-label="Star run"
        aria-pressed={bundle.run.starred}
        title={bundle.run.starred ? "Remove star" : "Star run"}
        onClick={toggleStar}
      >
        <Star className={bundle.run.starred ? "text-[var(--amber)]" : undefined} size={16} fill={bundle.run.starred ? "currentColor" : "none"} />
      </button>
      <button className="btn" onClick={async () => { const copy = await api.duplicateRun(bundle.run.id); navigate(`/runs/${copy.id}/pipeline`); }}>Duplicate</button>
      <button className="btn" onClick={async () => { await api.cancelRun(bundle.run.id); await refresh(); }}>Cancel</button>
      <button className="btn" onClick={remove}>Delete</button>
      <a className="btn" href={`/api/runs/${bundle.run.id}/export`}>Export archive</a>
      <a className="btn" href={`/api/runs/${bundle.run.id}/analysis.json`}>Analysis JSON</a>
    </div>
  );
}

type LabAction = Bundle["actions"][number];

export function InspectPage() {
  const bundle = useOptionalBundle();
  const labels = objectLabels(bundle?.objects ?? []);
  const actionId = useInspector((state) => state.selectedAction);
  const action = bundle?.actions.find((item) => item.local_id === actionId) ?? null;
  useEffect(() => {
    if (!bundle || !actionId || action) return;
    useInspector.getState().selectAction(null);
  }, [bundle, actionId, action]);
  const actions: (LabAction | null)[] = bundle ? bundle.actions : Array.from({ length: ACTION_PLACEHOLDERS }, () => null);
  return (
    <div className={INSPECT_LAYOUT}>
      <section className={INSPECT_SIDE}>
        <Timeline bundle={bundle} />
        <div className={ACTION_LIST} aria-hidden={bundle ? undefined : true}>
          {actions.map((item, index) => (
            <ActionRow key={item?.id ?? `pending-${index}`} item={item} index={index} labels={labels} selected={action?.id === item?.id} />
          ))}
        </div>
      </section>
      <section className={INSPECT_STAGE}>
        {bundle ? <VideoStage bundle={bundle} /> : (
          <div className="stage-slot flex min-h-0 w-full flex-col gap-2" aria-hidden="true">
            <div className="stage-frame overflow-hidden rounded-lg bg-black">
              <Bone className="bone-fill" />
            </div>
          </div>
        )}
      </section>
      <section className={INSPECT_DETAIL}>
        {bundle && bundle.actions.length === 0 ? <p className="text-[var(--muted)]">No actions yet.</p> : <ActionDetail bundle={bundle} action={action} />}
      </section>
    </div>
  );
}

function ActionRow({
  item,
  index,
  labels,
  selected,
}: {
  item: LabAction | null;
  index: number;
  labels: ReturnType<typeof objectLabels>;
  selected: boolean;
}) {
  if (!item) {
    return (
      <div className="block w-full rounded-md px-2 py-1 text-left text-sm">
        <Bone width="78%" />
        <div className="mono text-[10px]"><Bone width="6rem" /></div>
      </div>
    );
  }
  return (
    <button
      className={`block w-full rounded-md px-2 py-1 text-left text-sm ${selected ? "bg-[#1a221d]" : ""}`}
      onClick={() => {
        const inspector = useInspector.getState();
        inspector.setTime(item.start);
        inspector.selectAction(item.local_id, { start: item.start, end: item.end });
        inspector.setLoop({ start: item.start, end: item.end });
      }}
    >
      <div>{index + 1}. {actionPhrase(item, labels)} <span className="text-[var(--muted)]">{humanize(item.outcome)}</span></div>
      <div className="mono text-[10px] text-[var(--muted)]">{formatTime(item.start)}–{formatTime(item.end)}</div>
    </button>
  );
}

function ActionDetail({ bundle, action }: { bundle: Bundle | null; action: LabAction | null }) {
  const [label, setLabel] = useState("");
  const client = useQueryClient();
  if (bundle && !action) return <UnselectedAction />;
  const pending = action == null;
  return (
    <div className="space-y-2" aria-hidden={pending || undefined}>
      <div className="flex flex-wrap gap-2">
        {action ? (
          <>
            <span className={`tag ${evidenceClass(action.evidence_class)}`}>{humanize(action.evidence_class)}</span>
            <span className="tag">{humanize(action.provenance)}</span>
            <span className="tag">{humanize(action.outcome)}</span>
          </>
        ) : (
          <>
            <span className="tag"><Bone width="5rem" /></span>
            <span className="tag"><Bone width="5rem" /></span>
            <span className="tag"><Bone width="5rem" /></span>
          </>
        )}
      </div>
      <h2 className="text-base">{action ? humanize(action.action_type) : <Bone width="40%" />}</h2>
      <p>{action ? action.description : <Bone width="92%" />}</p>
      <p className="text-xs text-[var(--muted)]">
        {action ? <>{formatTime(action.start)}–{formatTime(action.end)} · uncertainty {action.uncertainty}</> : <Bone width="48%" />}
      </p>
      <p><span className="text-[var(--muted)]">Before: </span>{action ? humanize(action.initial_state) || "not stated" : <Bone width="8rem" />}</p>
      <p><span className="text-[var(--muted)]">After: </span>{action ? humanize(action.final_state) || "not stated" : <Bone width="8rem" />}</p>
      <p className="text-xs">{action ? action.evidence_note : <Bone width="70%" />}</p>
      <div className="flex flex-wrap gap-2">
        {action && bundle ? action.object_local_ids.map((id) => {
          const object = bundle.objects.find((item) => item.local_id === id);
          const thumb = objectThumbnail(object);
          return thumb ? <img key={id} src={thumb} alt={humanize(object?.label ?? id)} title={humanize(object?.label ?? id)} className="h-16 w-16 rounded bg-black object-contain" /> : <span key={id} className="tag">{humanize(id)}</span>;
        }) : (
          <div className="relative h-16 w-16 overflow-hidden rounded bg-black"><Bone className="bone-fill absolute inset-0" /></div>
        )}
      </div>
      <BeforeAfter bundle={bundle} action={action} />
      <label className="block text-xs text-[var(--muted)]">
        Correct description
        {action ? (
          <textarea className="field mt-1" value={label} placeholder={action.description} onChange={(event) => setLabel(event.target.value)} />
        ) : (
          <FieldBone textarea className="mt-1" />
        )}
      </label>
      {action && bundle ? (
        <button
          className="btn"
          onClick={async () => {
            await api.patchAction(bundle.run.id, action.local_id, { description: label || action.description });
            setLabel("");
            await client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
          }}
        >
          Save correction
        </button>
      ) : (
        <span className="btn">Save correction</span>
      )}
    </div>
  );
}

function UnselectedAction() {
  const time = useInspector((state) => state.time);
  return (
    <div className="space-y-2">
      <h2 className="text-base">No action selected</h2>
      <p className="text-xs text-[var(--muted)]">{formatTime(time)}</p>
      <p className="text-[var(--muted)]">Choose an action from the list. A frame outside its range clears the selection.</p>
    </div>
  );
}

function ObjectTile({
  object,
  sightings,
  selected = false,
  onOpen,
}: {
  object: Bundle["objects"][number] | null;
  sightings?: number;
  selected?: boolean;
  onOpen?: () => void;
}) {
  const thumb = objectThumbnail(object);
  const media = thumb ? (
    <img src={thumb} alt="" className="max-h-48 w-auto max-w-full object-contain" />
  ) : object ? (
    <div className="grid h-48 place-items-center text-xs text-[var(--muted)]">No crop yet</div>
  ) : (
    <div className="h-48 w-full"><Bone className="bone-fill" /></div>
  );
  const body = (
    <>
      <div className="flex flex-1 items-center justify-center p-3">{media}</div>
      <div className="mt-auto space-y-1 px-3 pb-3 text-sm">
        {object ? (
          <>
            <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: object.color }} />{humanize(object.local_id)}</div>
            <div className="font-medium">{humanize(object.label)}</div>
            <div className="text-xs text-[var(--muted)]">{sightings ?? object.observation_count} observations · {object.uncertain_identity ? "identity uncertain" : "identity stated"}</div>
          </>
        ) : (
          <>
            <div><Bone width="38%" /></div>
            <div className="font-medium"><Bone width="64%" /></div>
            <div className="text-xs"><Bone width="80%" /></div>
          </>
        )}
      </div>
    </>
  );
  if (!object) return <div className="panel flex h-full flex-col overflow-hidden text-left" aria-hidden="true">{body}</div>;
  return (
    <button
      type="button"
      aria-pressed={selected}
      className={`panel flex h-full flex-col overflow-hidden text-left${selected ? " outline outline-2 -outline-offset-2 outline-[var(--teal)]" : ""}`}
      onClick={onOpen}
    >
      {body}
    </button>
  );
}

export function ObjectsPage() {
  const bundle = useOptionalBundle();
  const pending = bundle == null;
  const [open, setOpen] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [mergeDrop, setMergeDrop] = useState("");
  const [preview, setPreview] = useState<number | null>(null);
  const client = useQueryClient();
  const objects = bundle?.objects ?? Array.from({ length: OBJECT_PLACEHOLDERS }, () => null);
  const selected = bundle?.objects.find((item) => item.local_id === open);
  const selectedThumb = objectThumbnail(selected);
  const shots = selected && bundle ? objectShots(selected, bundle.observations) : [];
  const frameCells = selected && bundle ? objectFrameTimeline(bundle.frames, selected.local_id, bundle.observations) : [];
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading objects</p>}
      <div className="flex gap-2">
        <input className="field max-w-xs" placeholder="Add a missed object" value={label} disabled={pending} onChange={(event) => setLabel(event.target.value)} />
        <button className="btn" disabled={pending || !label || !bundle} onClick={async () => {
          if (!bundle) return;
          await api.addObject(bundle.run.id, label, "Added by the researcher.");
          setLabel("");
          await client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
        }}>Add</button>
      </div>
      <div className={OBJECT_GRID}>
        {objects.map((object, index) => (
          <ObjectTile
            key={object?.id ?? `pending-${index}`}
            object={object}
            sightings={object && bundle ? sightingCount(bundle.frames, object.local_id, bundle.observations) : undefined}
            selected={object?.local_id === open}
            onOpen={object ? () => {
              setOpen((current) => (current === object.local_id ? null : object.local_id));
              setPreview(null);
            } : undefined}
          />
        ))}
      </div>
      {bundle && selected && (
        <div className="panel grid gap-4 p-4 lg:grid-cols-[280px_1fr]">
          {selectedThumb && <img src={selectedThumb} alt="" className="w-full rounded" />}
          <div className="space-y-2 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="shrink-0 font-medium">{humanize(selected.local_id)}:</h2>
              <input
                key={selected.local_id}
                className="field max-w-full"
                style={{ width: "calc((48rem - 0.75rem) / 2)" }}
                defaultValue={selected.label}
                id={`label-${selected.local_id}`}
                aria-label={`${humanize(selected.local_id)} name`}
              />
              <button className="btn shrink-0" onClick={async () => {
                const input = document.getElementById(`label-${selected.local_id}`) as HTMLInputElement;
                await api.patchObject(bundle.run.id, selected.local_id, { label: input.value });
                await client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
              }}>Save label</button>
            </div>
            <p>{selected.description}</p>
            <div className="text-[var(--muted)]">
              <div>Initial: {humanize(selected.initial_state) || "—"}</div>
              <div>Final: {humanize(selected.final_state) || "—"}</div>
            </div>
            <div className="flex gap-2">
              <select className="field" value={mergeDrop} onChange={(event) => setMergeDrop(event.target.value)}>
                <option value="">Merge another id into this one</option>
                {bundle.objects.filter((item) => item.local_id !== selected.local_id).map((item) => <option key={item.local_id} value={item.local_id}>{humanize(item.local_id)}</option>)}
              </select>
              <button className="btn" disabled={!mergeDrop} onClick={async () => { await api.mergeObjects(bundle.run.id, selected.local_id, mergeDrop); await client.invalidateQueries({ queryKey: ["run", bundle.run.id] }); }}>Merge</button>
            </div>
          </div>
          {frameCells.length > 0 && (
            <div className="min-w-0 lg:col-span-2">
              <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">Timeline</div>
              <div className="flex overflow-x-auto pb-1" role="list" aria-label={`${humanize(selected.label)} timeline`}>
                {frameCells.map((cell) => {
                  const stamp = formatTime(cell.timestamp);
                  const present = Boolean(cell.url && cell.observationId);
                  const shotIndex = present ? shots.findIndex((shot) => shot.id === cell.observationId) : -1;
                  return (
                    <div key={cell.frameId} role="listitem" className="w-14 shrink-0" aria-label={present ? `Frame at ${stamp}` : `Missing at ${stamp}`}>
                      {present ? (
                        <button
                          type="button"
                          title={stamp}
                          className="block h-14 w-14 overflow-hidden border-0 bg-black p-0"
                          onClick={() => {
                            useInspector.getState().setTime(cell.timestamp);
                            if (shotIndex >= 0) setPreview(shotIndex);
                          }}
                        >
                          <img src={cell.url ?? ""} alt="" className="block h-full w-full object-cover object-center" />
                        </button>
                      ) : (
                        <div title={`Missing · ${stamp}`} className="h-14 w-14 border border-dashed border-[var(--line)] bg-[#0d1016]" />
                      )}
                      <div className="mono mt-1 text-center text-[10px] leading-none text-[var(--muted)]">{stamp}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div className="min-w-0 lg:col-span-2">
            <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">Dataset</div>
            <div className="flex flex-wrap items-start gap-3">
            {shots.map((image, index) => (
              <button
                key={image.id}
                type="button"
                className="block w-fit max-w-full border-0 bg-transparent p-0 text-left"
                aria-label={shotLabel(image)}
                onClick={() => setPreview(index)}
              >
                <ShotFrame
                  src={image.url}
                  confidence={image.confidence}
                  imgClassName="h-64 w-auto max-w-full rounded bg-black object-contain"
                />
                <div className="mt-1 text-[10px] text-[var(--muted)]">{image.caption}</div>
              </button>
            ))}
            </div>
          </div>
          {preview != null && (
            <AssetPreview
              key={`${selected.local_id}:${preview}`}
              images={shots.map((image) => ({
                src: image.url,
                filename: image.caption,
                confidence: image.confidence,
              }))}
              startIndex={preview}
              onClose={() => setPreview(null)}
            />
          )}
        </div>
      )}
      {bundle && <DecisionPanel bundle={bundle} />}
      {bundle && <PerceptionPanel bundle={bundle} />}
    </div>
  );
}

export function StatesPage() {
  const bundle = useOptionalBundle();
  const pending = bundle == null;
  const [localId, setLocalId] = useState("");
  const time = useInspector((state) => state.time);
  useEffect(() => {
    if (!localId && bundle?.objects[0]) setLocalId(bundle.objects[0].local_id);
  }, [bundle, localId]);
  const scene = useQuery({
    queryKey: ["scene", bundle?.run.id ?? "pending", time.toFixed(2)],
    queryFn: () => api.scene(bundle?.run.id ?? "", Number(time.toFixed(2))),
    enabled: Boolean(bundle),
  });
  const states = bundle?.states.filter((item) => item.object_local_id === localId) ?? [];
  const transitions = bundle?.transitions.filter((item) => item.object_local_id === localId) ?? [];
  const relations = bundle?.relationships.filter((item) => item.subject_local_id === localId || item.other_local_id === localId) ?? [];
  const selected = bundle?.objects.find((item) => item.local_id === localId);
  const record = bundle ? objectScene(scene.data, bundle, localId, time) : null;
  const objects = bundle?.objects ?? Array.from({ length: OBJECT_PLACEHOLDERS }, () => null);
  const stateRows = pending ? Array.from({ length: STATE_PLACEHOLDERS }, () => null) : states;
  const transitionRows = pending ? Array.from({ length: LINE_PLACEHOLDERS }, () => null) : transitions;
  const relationRows = pending ? Array.from({ length: LINE_PLACEHOLDERS }, () => null) : relations;
  return (
    <div className="grid gap-4 lg:grid-cols-[240px_1fr]" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading states</p>}
      <div className="space-y-1">
        {objects.map((object, index) => (
          object ? (
            <button key={object.id} className={`btn block w-full text-left ${localId === object.local_id ? "primary" : ""}`} onClick={() => setLocalId(object.local_id)}>{labeledId(object.local_id, object.label)}</button>
          ) : (
            <div key={`pending-${index}`} className="btn block w-full text-left" aria-hidden="true"><Bone width="70%" /></div>
          )
        ))}
      </div>
      <div className="space-y-3">
        <h2 className="text-lg">State history</h2>
        {stateRows.map((state, index) => (
          state ? (
            <button key={state.id} className="panel block w-full p-3 text-left text-sm" onClick={() => useInspector.getState().setTime(state.time_start)}>
              <div className="mono text-xs text-[var(--muted)]">{formatTime(state.time_start)}–{formatTime(state.time_end)}</div>
              <div>{humanize(state.state_label)}</div>
              <div className="text-[var(--muted)]">{state.description}</div>
              <span className={`tag ${evidenceClass(state.evidence_class)}`}>{humanize(state.evidence_class)}</span>
            </button>
          ) : (
            <div key={`pending-${index}`} className="panel block w-full p-3 text-left text-sm" aria-hidden="true">
              <div className="mono text-xs"><Bone width="8rem" /></div>
              <div><Bone width="36%" /></div>
              <div><Bone width="70%" /></div>
              <span className="tag"><Bone width="5rem" /></span>
            </div>
          )
        ))}
        <h3 className="text-sm">Transitions</h3>
        {transitionRows.map((item, index) => (
          <div key={item?.id ?? `pending-${index}`} className="text-sm">
            {item ? <>{formatTime(item.timestamp)} · {humanize(item.from_state)} → {humanize(item.to_state)} · <span className={`tag ${evidenceClass(item.evidence_class)}`}>{humanize(item.evidence_class)}</span></> : <Bone width="60%" />}
          </div>
        ))}
        <h3 className="text-sm">Relationships</h3>
        {relationRows.map((item, index) => (
          <div key={item?.id ?? `pending-${index}`} className="text-sm">
            {item ? <>{humanize(item.subject_local_id)} {humanize(item.kind)} {humanize(item.other_local_id ?? "")} · {item.description}</> : <Bone width="70%" />}
          </div>
        ))}
        <div className="panel p-3 text-sm">
          <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">
            {selected ? labeledId(selected.local_id, selected.label) : pending ? <Bone width="8rem" /> : "Object"} · scene at {pending ? <Bone width="4ch" /> : `${time.toFixed(2)}s`}
          </div>
          <pre className="mono max-h-64 overflow-auto text-xs">{pending ? <Bone width="70%" /> : JSON.stringify(record, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}

export function WorkflowPage() {
  const bundle = useOptionalBundle();
  const pending = bundle == null;
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [category, setCategory] = useState("observed_temporal");
  const client = useQueryClient();
  useEffect(() => {
    if (!bundle) return;
    setFrom((current) => current || bundle.steps[0]?.local_id || "");
    setTo((current) => current || bundle.steps[1]?.local_id || "");
  }, [bundle]);
  const stepId = useInspector((state) => state.selectedStep);
  const step = bundle?.steps.find((item) => item.local_id === stepId);
  const labels = objectLabels(bundle?.objects ?? []);
  const dependencies = pending ? Array.from({ length: DEPENDENCY_PLACEHOLDERS }, () => null) : bundle.dependencies;
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading workflow</p>}
      <WorkflowGraph bundle={bundle} />
      <div className="flex flex-wrap gap-2 text-sm">
        {pending ? <FieldBone compact /> : (
          <select className="field w-auto" value={from} onChange={(event) => setFrom(event.target.value)}>{bundle.steps.map((item) => <option key={item.local_id} value={item.local_id}>{humanize(item.local_id)}</option>)}</select>
        )}
        {pending ? <FieldBone compact /> : (
          <select className="field w-auto" value={to} onChange={(event) => setTo(event.target.value)}>{bundle.steps.map((item) => <option key={item.local_id} value={item.local_id}>{humanize(item.local_id)}</option>)}</select>
        )}
        <select className="field w-auto" value={category} disabled={pending} onChange={(event) => setCategory(event.target.value)}>
          {["necessary_physical", "observed_temporal", "inferred", "unknown"].map((item) => <option key={item} value={item}>{humanize(item)}</option>)}
        </select>
        <button className="btn" disabled={pending || !bundle} onClick={async () => {
          if (!bundle) return;
          await api.addDependency(bundle.run.id, { from_step: from, to_step: to, category, rationale: "Added by the researcher." });
          await client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
        }}>Add dependency</button>
      </div>
      {step && (
        <div className="panel p-3 text-sm">
          <div className="font-medium">{nameObjects(step.name, labels)}</div>
          <p>Completion: {nameObjects(step.completion_condition, labels)}</p>
          <p>Observed: {nameObjects(step.observed_result, labels)}</p>
          <p className="text-[var(--muted)]">Objects: {step.required_objects.map(humanize).join(", ") || "none"}</p>
        </div>
      )}
      <ul className="space-y-1 text-sm">
        {dependencies.map((dep, index) => (
          <li key={dep?.id ?? `pending-${index}`} className="flex items-center justify-between gap-2">
            {dep && bundle ? (
              <>
                <span>{humanize(dep.from_step)} → {humanize(dep.to_step)} · {humanize(dep.category)} · {humanizeText(nameObjects(dep.rationale, labels))}</span>
                <button className="btn" onClick={async () => { await api.removeDependency(bundle.run.id, dep.from_step, dep.to_step); await client.invalidateQueries({ queryKey: ["run", bundle.run.id] }); }}>Remove</button>
              </>
            ) : (
              <>
                <Bone width="70%" />
                <span className="btn" aria-hidden="true">Remove</span>
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function StepsPage() {
  const bundle = useOptionalBundle();
  const openInspect = useOpenInspect();
  const [index, setIndex] = useState(0);
  const step = bundle?.steps[index];
  const labels = objectLabels(bundle?.objects ?? []);
  if (bundle && !step) return <p className="text-[var(--muted)]">No workflow steps.</p>;
  const start = step?.time_start ?? 0;
  const end = step?.time_end ?? start;
  const pending = !bundle;
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading steps</p>}
      <div className="flex items-center gap-2">
        <span className="mr-1 text-sm tabular-nums text-[var(--muted)]">
          {bundle ? `${index + 1} / ${bundle.steps.length}` : <Bone width="4ch" />}
        </span>
        <button className="btn" disabled={pending} onClick={() => setIndex((value) => Math.max(0, value - 1))}>Previous</button>
        <button className="btn" disabled={pending || !bundle} onClick={() => {
          if (!bundle) return;
          const count = bundle.steps.length;
          setIndex((value) => Math.min(count - 1, value + 1));
        }}>Next</button>
      </div>
      <h2 className="text-xl">{step ? <>{step.number}. {nameObjects(step.name, labels)}</> : <Bone width="14rem" />}</h2>
      <p>{step ? nameObjects(step.expected_action, labels) : <Bone width="80%" />}</p>
      <p className="text-sm text-[var(--muted)]">Expected: {step ? nameObjects(step.completion_condition, labels) : <Bone width="60%" />}</p>
      <p className="text-sm">Observed: {step ? nameObjects(step.observed_result, labels) : <Bone width="60%" />}</p>
      <div className="grid gap-2 md:grid-cols-3">
        {bundle && step ? [start, (start + end) / 2, end].map((stamp) => {
          const frame = bundle.frames.reduce((best, item) => Math.abs(item.timestamp - stamp) < Math.abs(best.timestamp - stamp) ? item : best, bundle.frames[0]);
          return frame ? <img key={stamp} src={frame.url} alt="" className="panel aspect-video w-full bg-black object-contain" /> : null;
        }) : Array.from({ length: STEP_MEDIA_PLACEHOLDERS }, (_, mediaIndex) => (
          <div key={mediaIndex} className="panel relative aspect-video w-full overflow-hidden bg-black" aria-hidden="true">
            <Bone className="bone-fill absolute inset-0" />
          </div>
        ))}
      </div>
      <div className="flex flex-wrap items-start gap-4">
        {bundle && step ? step.required_objects.map((id) => {
          const object = bundle.objects.find((item) => item.local_id === id);
          const thumb = objectThumbnail(object);
          return (
            <div key={id} className="w-28 shrink-0 text-xs">
              {thumb && <img src={thumb} alt="" className="mb-2 h-16 w-16 rounded bg-black object-contain" />}
              <div className="leading-snug">{humanize(object?.label ?? id)}</div>
            </div>
          );
        }) : Array.from({ length: STEP_MEDIA_PLACEHOLDERS }, (_, mediaIndex) => (
          <div key={mediaIndex} className="w-28 shrink-0 text-xs" aria-hidden="true">
            <div className="relative mb-2 h-16 w-16 overflow-hidden rounded bg-black">
              <Bone className="bone-fill absolute inset-0" />
            </div>
            <Bone width="80%" />
          </div>
        ))}
      </div>
      <button className="btn" type="button" disabled={pending} onClick={() => {
        if (!bundle || !step) return;
        const inspector = useInspector.getState();
        const linked = bundle.actions.find((item) => step.action_local_ids.includes(item.local_id));
        const covering = bundle.actions.find((item) => item.start <= end && item.end >= start);
        const chosen = linked ?? covering ?? null;
        inspector.setPlaying(false);
        inspector.setTime(start);
        inspector.selectStep(step.local_id);
        inspector.selectAction(chosen?.local_id ?? null, chosen ? { start: chosen.start, end: chosen.end } : undefined);
        inspector.setLoop(end > start ? { start, end } : null);
        openInspect();
      }}>Show on timeline</button>
    </div>
  );
}

export function PipelinePage() {
  const bundle = useOptionalBundle();
  const client = useQueryClient();
  const pending = bundle == null;
  const stages = bundle?.stages ?? [];
  const done = stages.filter((stage) => stage.status === "completed" || stage.status === "skipped").length;
  const fraction = stages.length ? done / stages.length : 0;
  const rows = pending ? Array.from({ length: STAGE_PLACEHOLDERS }, () => null) : stages;
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading pipeline</p>}
      <div className="h-2 overflow-hidden rounded bg-[#0d1016]">
        <div className="h-full bg-[var(--teal)]" style={{ width: `${fraction * 100}%` }} />
      </div>
      <p className="text-xs text-[var(--muted)]">
        {pending ? <Bone width="24rem" /> : `${done} of ${stages.length} stages finished or skipped. The bar counts stages, not a guessed percentage.`}
      </p>
      {rows.map((stage, index) => (
        <div key={stage?.stage ?? `pending-${index}`} className="panel grid gap-2 p-3 text-sm md:grid-cols-[180px_120px_1fr_auto]" aria-hidden={stage ? undefined : true}>
          <div>{stage ? humanize(stage.stage) : <Bone width="8rem" />}</div>
          <div>{stage ? humanize(stage.status) : <Bone width="5rem" />}</div>
          <div className="text-[var(--muted)]">
            {stage ? (
              <>
                {stage.progress_current}/{stage.progress_total || 0}
                {stage.error ? ` · ${humanize(stage.error)}` : ""}
                {stage.checkpoint?.reason ? ` · ${humanize(String(stage.checkpoint.reason))}` : ""}
              </>
            ) : <Bone width="6rem" />}
          </div>
          {stage?.status === "failed" && bundle && (
            <button className="btn h-fit self-center" onClick={async () => { await api.retryStage(bundle.run.id, stage.stage); await client.invalidateQueries({ queryKey: ["run", bundle.run.id] }); }}>Retry</button>
          )}
        </div>
      ))}
    </div>
  );
}

export function EvaluatePage() {
  const bundle = useOptionalBundle();
  const client = useQueryClient();
  const truths = useQuery({
    queryKey: ["truth", bundle?.run.id ?? "pending"],
    queryFn: () => api.groundTruth(bundle?.run.id ?? ""),
    enabled: Boolean(bundle),
  });
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [iou, setIou] = useState(0.3);
  const pending = bundle == null || (truths.data === undefined && !truths.isError);
  return (
    <div className="space-y-3" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading evaluation</p>}
      <p className="text-sm text-[var(--muted)]">Promote the current revision, then score another run against it. Matching uses interval overlap. Model confidence is not accuracy.</p>
      <button className="btn" disabled={!bundle} onClick={async () => {
        if (!bundle) return;
        await api.promoteTruth(bundle.run.id, `Reference ${bundle.run.name}`);
        await client.invalidateQueries({ queryKey: ["truth", bundle.run.id] });
      }}>Save current revision as ground truth</button>
      <label className="flex w-fit items-center gap-4 text-sm">
        <span className="whitespace-nowrap">Action IoU threshold</span>
        <input className="field max-w-xs" type="number" step={0.05} value={iou} onChange={(event) => setIou(Number(event.target.value))} />
      </label>
      {truths.isError && <p className="text-sm text-[var(--rose)]">{(truths.error as Error).message}</p>}
      <ul className="space-y-2 text-sm">
        {pending ? Array.from({ length: TRUTH_PLACEHOLDERS }, (_, index) => (
          <li key={index} aria-hidden="true"><span className="btn"><Bone width="10rem" /></span></li>
        )) : truths.data?.map((item) => (
          <li key={item.id}>
            <button className="btn" onClick={async () => {
              if (!bundle) return;
              const report = await api.evaluate(bundle.run.id, item.id, { action_iou: iou });
              setResult(report.metrics);
            }}>{humanizeText(item.name)}</button>
          </li>
        ))}
      </ul>
      {bundle && result && <IdentityReport report={result} runId={bundle.run.id} />}
      {result && <pre className="panel mono overflow-auto p-3 text-xs">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function IdentityReport({ report, runId }: { report: Record<string, unknown>; runId: string }) {
  const identity = report.identity as Record<string, unknown> | undefined;
  const navigate = useNavigate();
  const setTime = useInspector((state) => state.setTime);
  if (!identity) return null;
  const failures = Array.isArray(identity.failures) ? identity.failures as { kind?: string; timestamp?: number | null; detail?: string }[] : [];
  return (
    <section className="panel space-y-2 p-3 text-sm">
      <h2 className="text-base">Identity</h2>
      {identity.evaluated === false ? (
        <p className="text-[var(--muted)]">{String(identity.reason ?? "Identity accuracy was withheld.")}</p>
      ) : (
        <p>
          IDF1 {numberText(identity.idf1)} · switches {String(identity.identity_switches ?? 0)} · false merges {String(identity.false_merges ?? 0)} · coverage {numberText(identity.tracking_coverage)} · re-id {numberText(identity.reid_accuracy)} · abstention {numberText(identity.abstention_rate)}
        </p>
      )}
      <ul className="space-y-1">
        {failures.map((item, index) => (
          <li key={`${item.kind}-${item.timestamp}-${index}`}>
            <button
              className="btn"
              disabled={item.timestamp == null}
              onClick={() => {
                if (item.timestamp == null) return;
                setTime(item.timestamp);
                navigate(`/runs/${runId}`);
              }}
            >
              {humanize(item.kind ?? "failure")}{item.timestamp == null ? "" : ` · ${item.timestamp.toFixed(2)}s`}
            </button>
            {item.detail ? <span className="ml-2 text-xs text-[var(--muted)]">{item.detail}</span> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function numberText(value: unknown) {
  return typeof value === "number" ? value.toFixed(2) : "—";
}

const LOG_COLUMNS: { label: string; className?: string }[] = [
  { label: "Stage" },
  { label: "Model" },
  { label: "Status" },
  { label: "Latency", className: "num" },
  { label: "Tokens in", className: "num" },
  { label: "Tokens out", className: "num" },
  { label: "Cost", className: "num" },
];

const LOG_BONES = ["7rem", "8rem", "5rem", "4rem", "4rem", "4rem", "5rem"];

type LogUsage = {
  cost_usd?: number;
  price_known?: boolean;
  api_key?: boolean;
  input_tokens?: number;
  output_tokens?: number;
};

function logUsage(row: Record<string, unknown>): LogUsage {
  return (row.usage ?? {}) as LogUsage;
}

export function LogsPage() {
  const bundle = useOptionalBundle();
  const [open, setOpen] = useState<string | null>(null);
  const rows = useQuery({
    queryKey: ["requests", bundle?.run.id ?? "pending"],
    queryFn: () => api.requests(bundle?.run.id ?? ""),
    enabled: Boolean(bundle),
  });
  const requestsPending = rows.data === undefined && !rows.isError;
  const requests = rows.data ?? [];
  const showTable = requestsPending || requests.length > 0;
  return (
    <div className="space-y-3" aria-busy={!bundle || requestsPending || undefined}>
      {(!bundle || requestsPending) && <p className="sr-only">Loading logs</p>}
      <h2 className="text-lg">Requests and raw responses</h2>
      {!bundle ? (
        <p className="text-sm" aria-hidden="true"><Bone width="16rem" /></p>
      ) : bundle.validation?.length ? (
        <ul className="space-y-1 text-sm">
          {bundle.validation.map((issue, index) => <li key={index} className={issue.severity === "error" ? "text-[var(--rose)]" : "text-[var(--amber)]"}>{humanize(issue.severity)} · {humanize(issue.code)} · {issue.message}</li>)}
        </ul>
      ) : <p className="text-sm text-[var(--muted)]">No validation issues on the active document.</p>}
      {rows.isError && <p className="text-sm text-[var(--rose)]">{(rows.error as Error).message}</p>}
      {!requestsPending && !rows.isError && !requests.length && (
        <p className="text-sm text-[var(--muted)]">This run has not recorded a model request. Fixture and deterministic analyses are stored as documents, not provider calls.</p>
      )}
      {showTable && (
        <div className="panel overflow-x-auto" aria-busy={requestsPending || undefined}>
          <table className="runs-table">
            <thead>
              <tr>
                {LOG_COLUMNS.map((column) => (
                  <th key={column.label} className={column.className} scope="col">{column.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {requestsPending ? Array.from({ length: REQUEST_PLACEHOLDERS }, (_, index) => (
                <tr key={index}>
                  {LOG_BONES.map((width, cell) => (
                    <td key={cell} className={LOG_COLUMNS[cell]?.className} aria-hidden="true"><Bone width={width} /></td>
                  ))}
                </tr>
              )) : requests.map((row) => {
                const id = String(row.id);
                const expanded = open === id;
                const usage = logUsage(row);
                const completed = row.status === "completed";
                return (
                  <LogRequest
                    key={id}
                    row={row}
                    expanded={expanded}
                    completed={completed}
                    usage={usage}
                    onToggle={() => setOpen((current) => (current === id ? null : id))}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function LogRequest({
  row,
  expanded,
  completed,
  usage,
  onToggle,
}: {
  row: Record<string, unknown>;
  expanded: boolean;
  completed: boolean;
  usage: LogUsage;
  onToggle: () => void;
}) {
  const id = String(row.id);
  const bodyId = `log-body-${id}`;
  return (
    <>
      <tr>
        <th scope="row">
          <button type="button" className="run-link" aria-expanded={expanded} aria-controls={bodyId} onClick={onToggle}>
            {humanize(String(row.stage))}
          </button>
        </th>
        <td>{humanize(String(row.model_id))}</td>
        <td><span className="tag">{humanize(String(row.status))}</span></td>
        <td className="num mono">{String(row.latency_ms)} ms</td>
        <td className="num mono">{completed ? usage.input_tokens ?? 0 : "—"}</td>
        <td className="num mono">{completed ? usage.output_tokens ?? 0 : "—"}</td>
        <td className="num mono">{completed ? formatUsd(Number(usage.cost_usd ?? 0), Boolean(usage.price_known), Boolean(usage.api_key)) : "—"}</td>
      </tr>
      {expanded && (
        <tr>
          <td id={bodyId} className="log-cell" colSpan={LOG_COLUMNS.length}>
            <LogBody row={row} />
          </td>
        </tr>
      )}
    </>
  );
}

function LogBody({ row }: { row: Record<string, unknown> }) {
  const lines = formatLogText(String(row.response_text || row.error || "")).split("\n");
  if (lines.length === 1 && lines[0] === "") {
    return <p className="text-xs text-[var(--muted)]">No response body.</p>;
  }
  return (
    <div className="log-body mono" role="log">
      {lines.map((line, index) => (
        <div key={index}>{line || "\u00a0"}</div>
      ))}
    </div>
  );
}

const RUN_PLACEHOLDERS = 4;

export function RunsPage() {
  const [status, setStatus] = useState("");
  const runs = useQuery({ queryKey: ["runs", status], queryFn: () => api.runs(status ? { status } : {}) });
  const rows = runs.data;
  const loading = rows == null && !runs.isError;
  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-medium">Runs</h1>
      <select className="field max-w-xs" aria-label="Filter by status" value={status} onChange={(event) => setStatus(event.target.value)}>
        <option value="">All statuses</option>
        {["pending", "running", "completed", "failed", "canceled"].map((item) => <option key={item} value={item}>{humanize(item)}</option>)}
      </select>
      {runs.isError && <p className="text-sm text-[var(--rose)]">{(runs.error as Error).message}</p>}
      {rows && !rows.length && (
        <p className="text-sm text-[var(--muted)]">{status ? "No runs with this status." : "No runs yet."}</p>
      )}
      {(loading || (rows && rows.length > 0)) && (
        <div className="panel overflow-x-auto" aria-busy={loading || undefined}>
          {loading && <p className="sr-only">Loading runs</p>}
          <RunsTable runs={rows ?? Array.from({ length: RUN_PLACEHOLDERS }, () => null)} />
        </div>
      )}
    </div>
  );
}

