import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { AssetDropZone } from "../components/AssetDropZone";
import { VerificationPanel } from "../components/VerificationPanel";
import { LibraryPicker } from "../components/LibraryPicker";
import { ProjectCosts } from "../components/ProjectCosts";
import { Bone, MediaPlaceholder } from "../components/Skeleton";
import { StorageSettings } from "../components/StorageSettings";
import { VideoCard, VideoFace } from "../components/VideoCard";
import { api, formatWhen, type ProjectSummary } from "../lib/api";
import type { VideoInfo } from "../lib/types";
import { videoReady } from "../lib/uploadProgress";
import { useUploads } from "../stores/uploads";
import { TaskProjectSections } from "./Tasks";

function projectDetailHelp(reference: boolean, spansDays: boolean): string {
  const lead = "Drop videos to add them.";
  const running = reference
    ? "Analysis keeps running if you switch pages."
    : "Upload and analysis keep running if you switch pages.";
  return spansDays ? `${lead} ${running}` : `${lead} ${running} Drag a card to reorder it.`;
}

function compareDisplayOrder(left: VideoInfo, right: VideoInfo): number {
  const leftRank = left.order_index;
  const rightRank = right.order_index;
  if (leftRank == null && rightRank == null) return 0;
  if (leftRank == null) return 1;
  if (rightRank == null) return -1;
  return leftRank - rightRank;
}

function inDisplayOrder(videos: VideoInfo[]): VideoInfo[] {
  if (!videos.some((video) => video.order_index != null)) return videos;
  return [...videos].sort(compareDisplayOrder);
}

/** Columns for the project mosaic. One row for 1–3 thumbnails; two rows after that, so four are 2×2. */
function mosaicColumns(count: number): number {
  if (count <= 3) return Math.max(1, count);
  return Math.min(4, Math.ceil(count / 2));
}

const PROJECT_PLACEHOLDERS = 2;

function ProjectCard({
  project,
  videos = [],
  videosPending = false,
  onOpen,
}: {
  project?: ProjectSummary;
  videos?: VideoInfo[];
  videosPending?: boolean;
  onOpen?: () => void;
}) {
  const pending = project == null;
  const listed = videos.filter((video) => video.ingest_status !== "failed");
  const frames = listed.filter((video) => video.thumbnail_url && videoReady(video));
  const columns = mosaicColumns(frames.length);
  const preview = frames.slice(0, columns * 2);
  const processing = listed.some((video) => video.ingest_status === "processing");
  const media = pending || videosPending ? (
    <MediaPlaceholder />
  ) : preview.length ? (
    <div className={preview.length === 1 ? "thumb-grid is-single" : "thumb-grid"} style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {preview.map((video) => (
        <img key={video.id} src={video.thumbnail_url} alt="" />
      ))}
    </div>
  ) : (
    <div className="thumb-empty">
      {processing ? "Processing…" : "No videos"}
    </div>
  );
  const body = (
    <>
      {media}
      <div className="project-card-meta space-y-1 p-3 text-sm">
        <div className="truncate">{pending ? <Bone width="58%" /> : project.name}</div>
        <div className="mono text-xs text-[var(--muted)]">
          {pending || videosPending ? <Bone width="9ch" /> : `${listed.length} ${listed.length === 1 ? "video" : "videos"}`}
          {!pending && project.is_task ? " · Across several days" : ""}
          {!pending && formatWhen(project.created_at) ? ` · ${formatWhen(project.created_at)}` : ""}
        </div>
      </div>
    </>
  );
  if (pending || !onOpen) return <div className="project-card panel overflow-hidden text-left" aria-hidden="true">{body}</div>;
  return (
    <button type="button" className="project-card panel overflow-hidden text-left" onClick={onOpen}>
      {body}
    </button>
  );
}

type SortCell = { key: string; video: VideoInfo | null; order: number; sortIndex: number };

function cellsWithSlot(videos: VideoInfo[], dragId: string, insertAt: number): SortCell[] {
  const resting = videos.filter((video) => video.id !== dragId);
  const slot = Math.max(0, Math.min(insertAt, resting.length));
  const cells: SortCell[] = [];
  let visual = 0;
  for (let index = 0; index <= resting.length; index += 1) {
    if (index === slot) {
      cells.push({ key: "slot", video: null, order: visual + 1, sortIndex: visual });
      visual += 1;
    }
    if (index < resting.length) {
      cells.push({ key: resting[index].id, video: resting[index], order: visual + 1, sortIndex: visual });
      visual += 1;
    }
  }
  return cells;
}

function slotIndexAtPoint(grid: HTMLElement, x: number, y: number): number | null {
  const cells = [...grid.querySelectorAll<HTMLElement>("[data-sort-index]")];
  if (!cells.length) return null;
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const cell of cells) {
    const rect = cell.getBoundingClientRect();
    const dx = x - (rect.left + rect.width / 2);
    const dy = y - (rect.top + rect.height / 2);
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = Number(cell.dataset.sortIndex);
    }
  }
  return best;
}

const VIDEO_PLACEHOLDERS = 3;

function ProjectDetailSkeleton({ onBack }: { onBack: () => void }) {
  return (
    <div className="space-y-4" aria-busy="true">
      <p className="sr-only">Loading project</p>
      <section className="space-y-3">
        <div>
          <button className="text-sm text-[var(--muted)]" onClick={onBack}>← Back to projects</button>
          <div className="mt-2 flex items-end justify-between gap-4">
            <h2 className="min-w-0 text-lg"><Bone width="12rem" /></h2>
            <Bone width="6rem" />
          </div>
          <p className="text-sm"><Bone width="80%" /></p>
        </div>
        <div className="drop-zone" aria-hidden="true">
          <Bone width="16rem" />
          <Bone width="10rem" />
        </div>
      </section>
      <div className="video-grid">
        {Array.from({ length: VIDEO_PLACEHOLDERS }, (_, index) => (
          <div key={index} className="video-card panel overflow-hidden">
            <VideoFace />
          </div>
        ))}
      </div>
    </div>
  );
}

type DragState = {
  id: string;
  index: number;
  origin: number;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
  active: boolean;
};

export function ProjectsPage() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const { projectId: routeProjectId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const labSettings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const view = useUploads((state) => state.view);
  const setView = useUploads((state) => state.setView);
  const items = useUploads((state) => state.items);
  const stageFiles = useUploads((state) => state.stageFiles);
  const enqueue = useUploads((state) => state.enqueue);
  const referenceDropped = useUploads((state) => state.referenceDropped);
  const removeItem = useUploads((state) => state.removeItem);
  const beginCreate = useUploads((state) => state.beginCreate);
  const clearDraft = useUploads((state) => state.clearDraft);
  const forgetProject = useUploads((state) => state.forgetProject);
  const noteTask = useUploads((state) => state.noteTask);
  const adoptServer = useUploads((state) => state.adoptServer);
  const mode = routeProjectId ? "edit" : view.mode === "create" ? "create" : "library";
  const editId = routeProjectId;
  const name = view.name;
  const spansDays = view.spansDays;
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [libraryPicks, setLibraryPicks] = useState<VideoInfo[]>([]);
  const [pendingDetail, setPendingDetail] = useState<{ id: string; spansDays: boolean; updatedAt: number } | null>(null);
  const createdProjectId = useRef("");
  const studies = projects.data ?? [];
  const editing = studies.find((project) => project.id === editId);
  const libraryVideos = useQuery({ queryKey: ["videos", "all"], queryFn: () => api.videos() });
  const projectVideos = useQuery({
    queryKey: ["videos", editId],
    queryFn: () => api.videos(editId),
    enabled: mode === "edit" && Boolean(editId),
  });
  const libraryCatalog = useQuery({
    queryKey: ["videos", "library"],
    queryFn: api.unassignedVideos,
    enabled: pickerOpen,
  });
  const videosByProject = new Map<string, VideoInfo[]>();
  for (const video of libraryVideos.data ?? []) {
    if (!video.project_id) continue;
    const group = videosByProject.get(video.project_id) ?? [];
    group.push(video);
    videosByProject.set(video.project_id, group);
  }
  for (const [projectId, group] of videosByProject) {
    videosByProject.set(projectId, inDisplayOrder(group));
  }
  const remove = useMutation({
    mutationFn: () => {
      forgetProject(editId);
      return api.deleteProject(editId);
    },
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["projects"] });
      await client.invalidateQueries({ queryKey: ["videos"] });
      setView({ mode: "library", editId: "", name: "", spansDays: false });
      navigate("/projects");
    },
    onError: (err: Error) => setError(err.message),
  });
  const deleteVideo = useMutation({
    mutationFn: (id: string) => api.deleteVideo(id),
    onSuccess: async () => {
      setError("");
      await client.invalidateQueries({ queryKey: ["videos"] });
      if (editId) await client.invalidateQueries({ queryKey: ["task", editId] });
    },
    onError: (err: Error) => setError(err.message),
  });
  const reorder = useMutation({
    mutationFn: (ids: string[]) => api.orderVideos(editId, ids),
    onMutate: async (ids) => {
      setError("");
      await client.cancelQueries({ queryKey: ["videos", editId] });
      const previous = client.getQueryData<VideoInfo[]>(["videos", editId]);
      const library = client.getQueryData<VideoInfo[]>(["videos", "all"]);
      if (previous) {
        const byId = new Map(previous.map((video) => [video.id, video]));
        const ordered = ids.flatMap((id, index) => {
          const video = byId.get(id);
          return video ? [{ ...video, order_index: index }] : [];
        });
        const rest = previous.filter((video) => !ids.includes(video.id));
        client.setQueryData(["videos", editId], [...ordered, ...rest]);
      }
      if (library) {
        const rank = new Map(ids.map((id, index) => [id, index]));
        client.setQueryData(
          ["videos", "all"],
          library.map((video) => (rank.has(video.id) ? { ...video, order_index: rank.get(video.id) } : video)),
        );
      }
      return { previous, library };
    },
    onError: (err: Error, _ids, context) => {
      if (context?.previous) client.setQueryData(["videos", editId], context.previous);
      if (context?.library) client.setQueryData(["videos", "all"], context.library);
      setError(err.message);
    },
    onSettled: async () => {
      await client.invalidateQueries({ queryKey: ["videos"] });
    },
  });
  const draftItems = items.filter((item) => item.projectId === "");
  const canSave = Boolean(name.trim()) && (draftItems.some((item) => item.phase === "staged" || item.phase === "resolving") || libraryPicks.length > 0) && !saving;
  const creating = mode === "create";
  const studiesReady = projects.data !== undefined;
  const libraryLoading = mode === "library" && !projects.isError && !studiesReady;
  const projectPending = mode === "edit" && Boolean(editId) && projects.data === undefined && !projects.isError;
  const storagePending = labSettings.data === undefined && !labSettings.isError;
  const videosPending = mode === "edit" && Boolean(editId) && projectVideos.data === undefined && !projectVideos.isError;
  const [drag, setDrag] = useState<DragState | null>(null);
  const dragLive = useRef<DragState | null>(null);
  const suppressClick = useRef(false);
  const orderRef = useRef<string[]>([]);
  const gridRef = useRef<HTMLDivElement>(null);
  const stopGesture = useRef<(() => void) | null>(null);
  const shown = mode === "edit" ? inDisplayOrder(projectVideos.data ?? []) : [];
  useEffect(() => {
    if (reorder.isPending) return;
    orderRef.current = inDisplayOrder(projectVideos.data ?? []).filter(videoReady).map((video) => video.id);
  }, [projectVideos.data, reorder.isPending]);

  useEffect(() => () => stopGesture.current?.(), []);

  useEffect(() => {
    if (!routeProjectId || view.mode === "library") return;
    setView({ mode: "library", editId: "", name: "", spansDays: false });
  }, [routeProjectId, view.mode, setView]);

  useEffect(() => {
    if (routeProjectId || params.get("new") !== "1") return;
    clearDraft();
    setLibraryPicks([]);
    createdProjectId.current = "";
    setView({ mode: "create", editId: "", name: "", spansDays: false });
    setError("");
    setParams({}, { replace: true });
  }, [routeProjectId, params, setParams, setView, clearDraft]);

  useEffect(() => {
    if (mode === "edit" && editId && projectVideos.data) adoptServer(editId, projectVideos.data);
  }, [mode, editId, projectVideos.data, adoptServer]);

  useEffect(() => {
    if (editId && editing?.is_task) noteTask(editId);
  }, [editId, editing?.is_task, noteTask]);

  useEffect(() => {
    if (!pendingDetail || pendingDetail.id !== routeProjectId) return;
    if (editing) {
      setPendingDetail(null);
      return;
    }
    const settledWithoutIt = projects.isSuccess && !projects.isFetching && projects.dataUpdatedAt !== pendingDetail.updatedAt;
    if (settledWithoutIt || projects.isError) setPendingDetail(null);
  }, [pendingDetail, routeProjectId, editing, projects.isSuccess, projects.isFetching, projects.dataUpdatedAt, projects.isError]);

  function commit(index: number, sourceId: string) {
    const current = orderRef.current.length
      ? orderRef.current
      : inDisplayOrder(projectVideos.data ?? []).filter(videoReady).map((video) => video.id);
    if (!sourceId || reorder.isPending) return;
    const next = current.filter((id) => id !== sourceId);
    next.splice(Math.max(0, Math.min(index, next.length)), 0, sourceId);
    if (next.length === current.length && next.every((id, position) => id === current[position])) return;
    orderRef.current = next;
    reorder.mutate(next);
  }

  function grabCard(video: VideoInfo, index: number, event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || reorder.isPending) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const live: DragState = {
      id: video.id,
      index,
      origin: index,
      x: event.clientX,
      y: event.clientY,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      width: rect.width,
      height: rect.height,
      active: false,
    };
    dragLive.current = live;
    const move = (pointer: PointerEvent) => {
      const current = dragLive.current;
      if (!current) return;
      const distance = Math.hypot(pointer.clientX - current.x, pointer.clientY - current.y);
      if (!current.active && distance < 6) return;
      current.x = pointer.clientX;
      current.y = pointer.clientY;
      if (!current.active) {
        current.active = true;
        current.index = current.origin;
        document.body.classList.add("is-sorting");
        setDrag({ ...current });
        return;
      }
      const grid = gridRef.current;
      if (grid) {
        const next = slotIndexAtPoint(grid, pointer.clientX, pointer.clientY);
        if (next !== null) current.index = next;
      }
      setDrag({ ...current });
    };
    const end = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", end);
      stopGesture.current = null;
      const finished = dragLive.current;
      dragLive.current = null;
      setDrag(null);
      document.body.classList.remove("is-sorting");
      if (finished?.active) {
        suppressClick.current = true;
        commit(finished.index, finished.id);
      }
    };
    stopGesture.current = end;
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
  }

  function openCreate() {
    clearDraft();
    setLibraryPicks([]);
    createdProjectId.current = "";
    setView({ mode: "create", editId: "", name: "", spansDays: false });
    setError("");
  }

  function openEdit(id: string) {
    setError("");
    navigate(`/projects/${id}`);
  }

  function backToProjects() {
    setView({ mode: "library", editId: "", name: "", spansDays: false });
    setLibraryPicks([]);
    setError("");
    if (routeProjectId) navigate("/projects");
  }

  function requestBackToProjects() {
    const savedId = createdProjectId.current;
    if (savedId) {
      createdProjectId.current = "";
      setLibraryPicks([]);
      setError("");
      navigate(`/projects/${savedId}`);
      return;
    }
    const dirty = draftItems.length > 0 || libraryPicks.length > 0;
    if (creating && dirty && !window.confirm("Go back without saving? This project will be lost if you leave before saving.")) return;
    if (creating) {
      clearDraft();
      setLibraryPicks([]);
    }
    backToProjects();
  }

  async function saveProject() {
    setSaving(true);
    setError("");
    try {
      const projectId = createdProjectId.current || await beginCreate(name.trim(), { spansDays, allowEmpty: libraryPicks.length > 0 });
      createdProjectId.current = projectId;
      if (libraryPicks.length) await api.attachVideos(projectId, libraryPicks.map((video) => video.id));
      createdProjectId.current = "";
      setLibraryPicks([]);
      setPendingDetail({ id: projectId, spansDays, updatedAt: projects.dataUpdatedAt });
      navigate(`/projects/${projectId}`);
      void client.invalidateQueries({ queryKey: ["projects"] });
      void client.invalidateQueries({ queryKey: ["videos"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the project");
    } finally {
      setSaving(false);
    }
  }

  async function chooseFromLibrary(chosen: VideoInfo[]) {
    setPickerOpen(false);
    if (creating) {
      setLibraryPicks(chosen);
      return;
    }
    if (!editId || !chosen.length) return;
    try {
      await api.attachVideos(editId, chosen.map((video) => video.id));
      if (editing?.is_task) await api.reconcileTask(editId);
      await client.invalidateQueries({ queryKey: ["videos"] });
      await client.invalidateQueries({ queryKey: ["task", editId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not add those videos");
    }
  }

  const pendingHere = pendingDetail?.id === routeProjectId ? pendingDetail : null;
  const detailLoading = Boolean(routeProjectId) && !editing && !projects.isError && (Boolean(pendingHere) || projects.data === undefined || projects.isFetching);
  if (routeProjectId && projects.isError && !editing) {
    return <p className="text-sm text-[var(--rose)]">{(projects.error as Error).message}</p>;
  }
  if (routeProjectId && !editing && !detailLoading) {
    return <p className="text-sm text-[var(--rose)]">Project not found.</p>;
  }
  if (detailLoading) return <ProjectDetailSkeleton onBack={backToProjects} />;
  const spansDaysView = Boolean(editing?.is_task);
  const assetRows = items.filter((item) => (creating ? item.projectId === "" : item.projectId === editId));
  const editingVideos = (shown ?? []).filter(
    (video) => videoReady(video) && (!spansDaysView || !assetRows.some((item) => item.videoId === video.id)),
  );
  const reference = labSettings.data?.video_storage === "reference";
  const sorting = drag?.active ? cellsWithSlot(editingVideos, drag.id, drag.index) : null;
  const draggedVideo = drag ? editingVideos.find((video) => video.id === drag.id) : undefined;

  return (
    <div className="space-y-4">
      {mode === "library" && (
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-medium">Projects</h1>
            <p className="text-sm text-[var(--muted)]">Each project holds the recordings of one procedure. One that spans several days keeps its sessions and chronology.</p>
          </div>
          <button className="btn primary" onClick={openCreate}>New project</button>
        </header>
      )}

      {mode === "library" && projects.isError && (
        <p className="text-sm text-[var(--rose)]">{(projects.error as Error).message}</p>
      )}
      {libraryLoading && (
        <div aria-busy="true">
          <p className="sr-only">Loading projects</p>
          <div className="project-grid">
            {Array.from({ length: PROJECT_PLACEHOLDERS }, (_, index) => <ProjectCard key={index} />)}
          </div>
        </div>
      )}
      {mode === "library" && studiesReady && (
        studies.length ? (
          <div className="project-grid">
            {studies.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                videos={videosByProject.get(project.id) ?? []}
                videosPending={libraryVideos.data === undefined && !libraryVideos.isError}
                onOpen={() => openEdit(project.id)}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-[var(--muted)]">No projects yet. Create one and add a video before it is saved.</p>
        )
      )}

      {(creating || (mode === "edit" && editing) || projectPending) && (
        <section className="space-y-3">
          <div>
            <button className="text-sm text-[var(--muted)] disabled:opacity-45" onClick={requestBackToProjects} disabled={saving}>← Back to projects</button>
            {creating ? (
              <div className="mt-3.5 mb-2 w-fit max-w-full">
                <input
                  className="field stretch block min-w-0"
                  value={name}
                  onChange={(event) => setView({ name: event.target.value })}
                  placeholder="Project name"
                  aria-label="Project name"
                  autoFocus
                />
                <label className="mt-4 flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={spansDays}
                    onChange={(event) => setView({ spansDays: event.target.checked })}
                  />
                  Across several days
                </label>
                <p className="mt-0.5 text-sm text-[var(--muted)]">Add a name and at least one video. You can drop files or choose videos from the library. Processing keeps going after you save.</p>
              </div>
            ) : (
              <>
                <div className="relative mt-2">
                  <h2 className="min-w-0 pr-36 text-lg">{editing ? editing.name : <Bone width="12rem" />}</h2>
                  {mode === "edit" && editId && <ProjectCosts projectId={editId} className="is-end is-overlay" />}
                </div>
                {spansDaysView && editing?.description ? (
                  <p className="pr-36 text-sm text-[var(--muted)]">{editing.description}</p>
                ) : null}
                <p className="pr-36 text-sm text-[var(--muted)]">
                  {storagePending ? <Bone width="80%" /> : projectDetailHelp(reference, spansDaysView)}
                </p>
                {spansDaysView && (
                  <p className="pr-36 text-xs text-[var(--amber)]">
                    Project time is the capture clock. Clip time is seconds from the start of that recording. File modification time and upload time are stored separately.
                  </p>
                )}
              </>
            )}
          </div>
          <StorageSettings />
          {storagePending ? (
            <div className="drop-zone" aria-hidden="true">
              <Bone width="16rem" />
              <Bone width="10rem" />
            </div>
          ) : <AssetDropZone
            accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
            multiple
            prompt="Drag and drop videos here,"
            chooseLabel="Choose files"
            libraryLabel="Choose from library"
            onChooseLibrary={() => setPickerOpen(true)}
            supported="Supported: MP4, MOV, WebM"
            rows={[
              ...assetRows.map((item) => ({
                id: item.key,
                name: item.name,
                url: item.previewUrl,
                subtitle: item.subtitle,
                progress: item.progress,
                failed: item.phase === "error",
              })),
              ...(creating ? libraryPicks.map((video) => ({
                id: video.id,
                name: video.filename,
                url: video.thumbnail_url,
                subtitle: "",
                progress: null,
              })) : []),
            ]}
            onRemove={(id) => {
              if (creating && libraryPicks.some((video) => video.id === id)) {
                setLibraryPicks((current) => current.filter((item) => item.id !== id));
                return;
              }
              removeItem(id);
            }}
            onFiles={(files, paths) => {
              if (!files.length) return;
              if (reference) {
                if (creating) referenceDropped("", files, paths);
                else if (editId) referenceDropped(editId, files, paths);
              } else if (creating) stageFiles(files);
              else if (editId) enqueue(editId, files);
            }}
          />}
        </section>
      )}

      {videosPending && (
        <div aria-busy="true">
          <p className="sr-only">Loading videos</p>
          <div className="video-grid">
            {Array.from({ length: VIDEO_PLACEHOLDERS }, (_, index) => (
              <div key={index} className="video-card panel overflow-hidden">
                <VideoFace />
              </div>
            ))}
          </div>
        </div>
      )}
      {mode === "edit" && !videosPending && (
        <div
          ref={gridRef}
          className="video-grid"
          onClickCapture={(event) => {
            if (!suppressClick.current) return;
            event.preventDefault();
            event.stopPropagation();
            suppressClick.current = false;
          }}
        >
          {(sorting ?? editingVideos.map((video, index) => ({ key: video.id, video, order: index + 1, sortIndex: index }))).map((cell) =>
            cell.video ? (
              <VideoCard
                key={cell.key}
                video={cell.video}
                order={cell.order}
                sortIndex={cell.sortIndex}
                deleting={deleteVideo.isPending && deleteVideo.variables === cell.video.id}
                actionLabel={`Delete ${cell.video.filename}`}
                actionTitle="Delete video"
                onGrab={spansDaysView ? undefined : (event) => grabCard(cell.video as VideoInfo, cell.sortIndex, event)}
                onDelete={() => {
                  const filename = cell.video?.filename ?? "this video";
                  const id = cell.video?.id ?? "";
                  if (window.confirm(`Delete “${filename}”? This removes it from the project and the video library. A referenced original stays on disk.`)) deleteVideo.mutate(id);
                }}
              />
            ) : (
              <div
                key={cell.key}
                className="video-card-slot"
                data-sort-index={cell.sortIndex}
                style={{ height: drag?.height }}
                aria-label={`Drop position ${cell.order}`}
              />
            ),
          )}
        </div>
      )}
      {drag?.active && draggedVideo && createPortal(
        <div
          className="video-card-ghost panel"
          style={{ left: drag.x - drag.offsetX, top: drag.y - drag.offsetY, width: drag.width, height: drag.height }}
        >
          <VideoFace video={draggedVideo} order={drag.index + 1} />
        </div>,
        document.body,
      )}
      {libraryVideos.isError && <p className="text-sm text-[var(--rose)]">{(libraryVideos.error as Error).message}</p>}
      {mode === "edit" && projectVideos.isSuccess && !editingVideos.length && !assetRows.length && (
        <p className="text-sm text-[var(--muted)]" aria-live="polite">
          This project has no videos yet. Drop one above, or choose from the library.
        </p>
      )}
      {(creating || (mode === "edit" && editing)) && (
        <div className="border-t border-[var(--line)] pt-4">
          {error && <p className="mb-3 text-sm text-[var(--rose)]">{error}</p>}
          {creating ? (
            <button className="btn primary" disabled={!canSave} onClick={() => void saveProject()}>
              {saving ? "Saving…" : "Save project"}
            </button>
          ) : (
            <button
              className="btn danger"
              disabled={remove.isPending}
              onClick={() => {
                const label = editing?.name || "this project";
                if (window.confirm(`Delete “${label}”? Copied videos and previews will be removed. Referenced files stay on disk.`)) remove.mutate();
              }}
            >
              {remove.isPending ? "Deleting…" : "Delete project"}
            </button>
          )}
        </div>
      )}
      {spansDaysView && editId && <TaskProjectSections projectId={editId} />}
      {mode === "edit" && editId && <VerificationPanel projectId={editId} />}
      {pickerOpen && (
        <LibraryPicker
          videos={libraryCatalog.data ?? []}
          loading={libraryCatalog.isLoading}
          error={libraryCatalog.isError ? (libraryCatalog.error as Error).message : ""}
          initialIds={libraryPicks.map((video) => video.id)}
          onClose={() => setPickerOpen(false)}
          onOpen={(chosen) => void chooseFromLibrary(chosen)}
        />
      )}
    </div>
  );
}

