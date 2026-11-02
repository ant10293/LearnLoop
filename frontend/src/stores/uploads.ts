import { create } from "zustand";
import { api } from "../lib/api";
import { droppedPath } from "../lib/droppedPath";
import { queryClient } from "../lib/queryClient";
import type { VideoInfo } from "../lib/types";
import { analysisPercent, progressLabel, progressStepCount, RECONCILE_DETAIL, uploadPercent, videoReady } from "../lib/uploadProgress";

export type UploadPhase = "resolving" | "staged" | "uploading" | "analyzing" | "reconciling" | "error";

export type UploadItem = {
  key: string;
  projectId: string;
  name: string;
  previewUrl: string;
  phase: UploadPhase;
  progress: number | null;
  subtitle: string;
  videoId?: string;
};

type ViewState = {
  mode: "library" | "create" | "edit";
  editId: string;
  name: string;
  spansDays: boolean;
};

type UploadStore = {
  view: ViewState;
  items: UploadItem[];
  setView: (patch: Partial<ViewState>) => void;
  stageFiles: (files: File[]) => void;
  stagePaths: (paths: string[]) => void;
  enqueue: (projectId: string, files: File[]) => void;
  enqueuePaths: (projectId: string, paths: string[]) => void;
  enqueueUnassigned: (files: File[]) => void;
  enqueueUnassignedPaths: (paths: string[]) => void;
  referenceDropped: (projectId: string, files: File[], hintedPaths?: string[]) => void;
  removeItem: (key: string) => void;
  clearDraft: () => void;
  forgetProject: (projectId: string) => void;
  beginCreate: (name: string, options?: { spansDays?: boolean; allowEmpty?: boolean }) => Promise<string>;
  adoptServer: (projectId: string, videos: VideoInfo[]) => void;
  noteTask: (projectId: string) => void;
};

export const LIBRARY_QUEUE = "library";
const files = new Map<string, File>();

function destination(projectId: string): string | null | undefined {
  if (projectId === LIBRARY_QUEUE) return null;
  if (!projectId) return undefined;
  return projectId;
}
const paths = new Map<string, string>();
const aborters = new Map<string, () => void>();
const cancelled = new Set<string>();
const draining = new Set<string>();
const polling = new Set<string>();
const taskProjects = new Set<string>();

function stepCount(projectId: string): number {
  return progressStepCount(taskProjects.has(projectId));
}

function revoke(item: UploadItem | undefined) {
  if (item?.previewUrl.startsWith("blob:")) URL.revokeObjectURL(item.previewUrl);
}

function patchItem(key: string, patch: Partial<UploadItem>) {
  useUploads.setState((state) => ({
    items: state.items.map((item) => (item.key === key ? { ...item, ...patch } : item)),
  }));
}

function dropItem(key: string) {
  const item = useUploads.getState().items.find((entry) => entry.key === key);
  revoke(item);
  files.delete(key);
  paths.delete(key);
  useUploads.setState((state) => ({ items: state.items.filter((entry) => entry.key !== key) }));
}

function remember(video: VideoInfo) {
  const { frames: _frames, ...summary } = video;
  queryClient.setQueriesData<VideoInfo[]>({ queryKey: ["videos"] }, (current) => {
    if (!Array.isArray(current)) return current;
    if (current.some((item) => item.id === summary.id)) {
      return current.map((item) => (item.id === summary.id ? { ...item, ...summary } : item));
    }
    const sameProject = current.some((item) => item.project_id === summary.project_id);
    if (current.length && !sameProject) return current;
    return [summary, ...current];
  });
  void queryClient.invalidateQueries({ queryKey: ["videos"] });
}

function fileName(path: string): string {
  return path.split(/[/\\]/).filter(Boolean).pop() || path;
}

function addPaths(projectId: string, pathList: string[], subtitle: string) {
  const added: UploadItem[] = pathList.map((path) => {
    const key = crypto.randomUUID();
    paths.set(key, path);
    return {
      key,
      projectId,
      name: fileName(path),
      previewUrl: "",
      phase: "staged",
      progress: null,
      subtitle,
    };
  });
  useUploads.setState((state) => ({ items: [...state.items, ...added] }));
}

function queueReferenced(projectId: string, fileList: File[], hinted: string[]) {
  const added: UploadItem[] = fileList.map((file, index) => {
    const key = crypto.randomUUID();
    const direct = hinted[index] || droppedPath(file);
    if (direct) paths.set(key, direct);
    return {
      key,
      projectId,
      name: file.name,
      previewUrl: URL.createObjectURL(file),
      phase: direct ? "staged" : "resolving",
      progress: null,
      subtitle: direct ? (projectId ? "Queued" : "Ready") : "Finding the original file",
    };
  });
  useUploads.setState((state) => ({ items: [...state.items, ...added] }));
  if (projectId && added.some((item) => item.phase === "staged")) kick(projectId);
  fileList.forEach((file, index) => {
    if (added[index].phase === "resolving") void resolveDropped(added[index].key, file);
  });
}

async function resolveDropped(key: string, file: File) {
  try {
    const found = await api.locateVideo({ name: file.name, size: file.size, mtime_ms: Math.round(file.lastModified) });
    if (cancelled.has(key)) return;
    const item = useUploads.getState().items.find((entry) => entry.key === key);
    if (!item || item.phase !== "resolving") return;
    paths.set(key, found.path);
    const queued = Boolean(item.projectId);
    patchItem(key, { phase: "staged", subtitle: queued ? "Queued" : "Ready" });
    if (queued) kick(item.projectId);
  } catch (error) {
    if (cancelled.has(key) || !useUploads.getState().items.some((entry) => entry.key === key)) return;
    const message = error instanceof Error ? error.message : "Could not find the original file";
    patchItem(key, { phase: "error", progress: null, subtitle: message });
  }
}

function addItems(projectId: string, fileList: File[], subtitle: string) {
  const added: UploadItem[] = fileList.map((file) => {
    const key = crypto.randomUUID();
    files.set(key, file);
    return {
      key,
      projectId,
      name: file.name,
      previewUrl: URL.createObjectURL(file),
      phase: "staged",
      progress: null,
      subtitle,
    };
  });
  useUploads.setState((state) => ({ items: [...state.items, ...added] }));
}

function analysisSubtitle(video: VideoInfo): string {
  if (video.ingest_status === "failed") return video.ingest_error || "Analysis failed";
  if (video.ingest_detail && video.ingest_detail !== "Queued") return video.ingest_detail;
  return "Analyzing";
}

function previewPatch(item: UploadItem, video: VideoInfo): Partial<UploadItem> {
  if (video.thumbnail_url && !item.previewUrl.startsWith("blob:")) return { previewUrl: video.thumbnail_url };
  return {};
}

function analysisPatch(item: UploadItem, video: VideoInfo): Partial<UploadItem> {
  const incoming = analysisPercent(video.ingest_progress ?? 0);
  const held = item.phase === "analyzing" ? item.progress ?? 0 : 0;
  if (incoming < held) return previewPatch(item, video);
  return {
    phase: "analyzing",
    progress: incoming,
    subtitle: progressLabel(2, stepCount(item.projectId), analysisSubtitle(video)),
    ...previewPatch(item, video),
  };
}

async function publishReady(key: string, video: VideoInfo) {
  const item = useUploads.getState().items.find((entry) => entry.key === key);
  if (cancelled.has(key) || !item) {
    await api.deleteVideo(video.id).catch(() => undefined);
    void queryClient.invalidateQueries({ queryKey: ["videos"] });
    return;
  }
  if (taskProjects.has(item.projectId)) {
    if (item.phase === "reconciling") return;
    patchItem(key, {
      phase: "reconciling",
      progress: 0,
      subtitle: progressLabel(3, 3, RECONCILE_DETAIL),
      videoId: video.id,
      ...previewPatch(item, video),
    });
    try {
      const bundle = await api.reconcileTask(item.projectId);
      if (cancelled.has(key) || !useUploads.getState().items.some((entry) => entry.key === key)) return;
      queryClient.setQueryData(["task", item.projectId], bundle);
    } catch (error) {
      if (!useUploads.getState().items.some((entry) => entry.key === key)) return;
      const message = error instanceof Error ? error.message : "Could not reconcile the project reading";
      patchItem(key, { phase: "error", progress: null, subtitle: message, videoId: video.id });
      remember(video);
      return;
    }
  }
  remember(video);
  dropItem(key);
  void queryClient.invalidateQueries({ queryKey: ["task", item.projectId] });
}

async function settle(key: string, video: VideoInfo) {
  const stillThere = useUploads.getState().items.some((entry) => entry.key === key);
  if (cancelled.has(key) || !stillThere) {
    await api.deleteVideo(video.id).catch(() => undefined);
    void queryClient.invalidateQueries({ queryKey: ["videos"] });
    return;
  }
  if (videoReady(video)) {
    await publishReady(key, video);
    return;
  }
  if (video.ingest_status === "failed") {
    patchItem(key, {
      phase: "error",
      progress: null,
      subtitle: video.ingest_error || "Analysis failed",
      videoId: video.id,
    });
    remember(video);
    return;
  }
  const current = useUploads.getState().items.find((entry) => entry.key === key);
  patchItem(key, {
    videoId: video.id,
    ...(current
      ? analysisPatch(current, video)
      : { phase: "analyzing", progress: analysisPercent(video.ingest_progress ?? 0), subtitle: progressLabel(2, stepCount(""), analysisSubtitle(video)) }),
  });
  remember(video);
}

async function runReference(key: string) {
  const item = useUploads.getState().items.find((entry) => entry.key === key);
  const path = paths.get(key);
  const projectId = item ? destination(item.projectId) : undefined;
  if (!item || !path || projectId === undefined) return;
  patchItem(key, { phase: "analyzing", progress: 0, subtitle: progressLabel(2, stepCount(item.projectId), "Reading the file") });
  try {
    const video = await api.referenceVideo(projectId, path);
    paths.delete(key);
    await settle(key, video);
  } catch (error) {
    if (cancelled.has(key)) return;
    if (useUploads.getState().items.some((entry) => entry.key === key)) {
      const message = error instanceof Error ? error.message : "Could not read that path";
      patchItem(key, { phase: "error", progress: null, subtitle: message });
    }
  }
}

async function runUpload(key: string) {
  const item = useUploads.getState().items.find((entry) => entry.key === key);
  const file = files.get(key);
  const projectId = item ? destination(item.projectId) : undefined;
  if (!item || !file || projectId === undefined) return;
  const total = stepCount(item.projectId);
  patchItem(key, { phase: "uploading", progress: 0, subtitle: progressLabel(1, total, "Uploading") });
  try {
    const video = await api.upload(
      projectId,
      file,
      (ratio) => {
        if (cancelled.has(key)) return;
        patchItem(key, { progress: uploadPercent(ratio), subtitle: progressLabel(1, total, "Uploading") });
      },
      { onAbort: (abort) => aborters.set(key, abort) },
    );
    files.delete(key);
    await settle(key, video);
  } catch (error) {
    if (cancelled.has(key) || (error instanceof DOMException && error.name === "AbortError")) {
      void queryClient.invalidateQueries({ queryKey: ["videos"] });
      return;
    }
    if (useUploads.getState().items.some((entry) => entry.key === key)) {
      const message = error instanceof Error ? error.message : "Upload failed";
      patchItem(key, { phase: "error", progress: null, subtitle: message });
    }
  } finally {
    aborters.delete(key);
  }
}

async function drain(projectId: string) {
  for (;;) {
    const item = useUploads.getState().items.find((entry) => entry.projectId === projectId && entry.phase === "staged");
    if (!item) return;
    if (paths.has(item.key)) await runReference(item.key);
    else await runUpload(item.key);
  }
}

function kick(projectId: string) {
  if (!projectId || draining.has(projectId)) return;
  draining.add(projectId);
  void (async () => {
    try {
      await drain(projectId);
    } finally {
      draining.delete(projectId);
      if (useUploads.getState().items.some((entry) => entry.projectId === projectId && entry.phase === "staged")) {
        kick(projectId);
      }
    }
  })();
}

function release(key: string, deleteOnServer: boolean) {
  cancelled.add(key);
  aborters.get(key)?.();
  aborters.delete(key);
  const item = useUploads.getState().items.find((entry) => entry.key === key);
  dropItem(key);
  if (deleteOnServer && item?.videoId) {
    void api.deleteVideo(item.videoId).catch(() => undefined).finally(() => {
      void queryClient.invalidateQueries({ queryKey: ["videos"] });
    });
  }
}

async function pollOnce() {
  const pending = useUploads.getState().items.filter((item) => item.phase === "analyzing" && item.videoId);
  await Promise.all(
    pending.map(async (item) => {
      const videoId = item.videoId;
      if (!videoId || polling.has(videoId)) return;
      polling.add(videoId);
      try {
        const video = await api.video(videoId);
        const current = useUploads.getState().items.find((entry) => entry.key === item.key);
        if (!current) return;
        if (videoReady(video)) {
          await publishReady(item.key, video);
          return;
        }
        if (video.ingest_status === "failed") {
          patchItem(item.key, { phase: "error", progress: null, subtitle: analysisSubtitle(video), videoId });
          remember(video);
          return;
        }
        patchItem(item.key, analysisPatch(current, video));
      } catch (error) {
        const message = error instanceof Error ? error.message : "";
        if (/not found/i.test(message)) dropItem(item.key);
      } finally {
        polling.delete(videoId);
      }
    }),
  );
}

const pollFlag = globalThis as typeof globalThis & { __learnloopUploadPoll?: number };
if (typeof window !== "undefined" && !pollFlag.__learnloopUploadPoll) {
  pollFlag.__learnloopUploadPoll = window.setInterval(() => void pollOnce(), 500);
}

export const useUploads = create<UploadStore>((set, get) => ({
  view: { mode: "library", editId: "", name: "", spansDays: false },
  items: [],
  setView: (patch) => set((state) => ({ view: { ...state.view, ...patch } })),
  stageFiles: (fileList) => addItems("", fileList, "Ready"),
  stagePaths: (pathList) => addPaths("", pathList, "Ready"),
  enqueue: (projectId, fileList) => {
    addItems(projectId, fileList, "Queued");
    kick(projectId);
  },
  enqueuePaths: (projectId, pathList) => {
    addPaths(projectId, pathList, "Queued");
    kick(projectId);
  },
  enqueueUnassigned: (fileList) => {
    addItems(LIBRARY_QUEUE, fileList, "Queued");
    kick(LIBRARY_QUEUE);
  },
  enqueueUnassignedPaths: (pathList) => {
    addPaths(LIBRARY_QUEUE, pathList, "Queued");
    kick(LIBRARY_QUEUE);
  },
  referenceDropped: (projectId, fileList, hintedPaths = []) => queueReferenced(projectId, fileList, hintedPaths),
  removeItem: (key) => release(key, true),
  clearDraft: () => {
    for (const item of get().items.filter((entry) => entry.projectId === "")) release(item.key, false);
  },
  forgetProject: (projectId) => {
    for (const item of get().items.filter((entry) => entry.projectId === projectId)) release(item.key, false);
  },
  beginCreate: async (name, options) => {
    const trimmed = name.trim();
    const pending = get().items.some((item) => item.projectId === "" && (item.phase === "staged" || item.phase === "resolving"));
    if (!trimmed || (!pending && !options?.allowEmpty)) throw new Error("Name the project and add at least one video.");
    const project = await api.createProject(trimmed, "", Boolean(options?.spansDays));
    if (options?.spansDays) taskProjects.add(project.id);
    if (pending) {
      set((state) => ({
        items: state.items.map((item) =>
          item.projectId === "" && (item.phase === "staged" || item.phase === "resolving")
            ? { ...item, projectId: project.id, subtitle: item.phase === "resolving" ? item.subtitle : "Queued" }
            : item,
        ),
      }));
      kick(project.id);
    }
    return project.id;
  },
  adoptServer: (projectId, videos) => {
    const known = new Map(videos.map((video) => [video.id, video]));
    set((state) => {
      const next: UploadItem[] = [];
      const seen = new Set<string>();
      for (const item of state.items) {
        if (item.projectId !== projectId || !item.videoId) {
          next.push(item);
          continue;
        }
        seen.add(item.videoId);
        const video = known.get(item.videoId);
        if (!video || videoReady(video)) {
          revoke(item);
          continue;
        }
        if (item.phase === "uploading" || item.phase === "reconciling") {
          next.push(item);
          continue;
        }
        next.push({
          ...item,
          ...(video.ingest_status === "failed"
            ? { phase: "error" as const, progress: null, subtitle: analysisSubtitle(video) }
            : analysisPatch(item, video)),
        });
      }
      for (const video of videos) {
        if (seen.has(video.id) || videoReady(video)) continue;
        if (video.ingest_status !== "processing" && video.ingest_status !== "failed") continue;
        next.push({
          key: `server-${video.id}`,
          projectId,
          name: video.filename,
          previewUrl: video.thumbnail_url || "",
          ...(video.ingest_status === "failed"
            ? { phase: "error" as const, progress: null, subtitle: analysisSubtitle(video) }
            : {
                phase: "analyzing" as const,
                progress: analysisPercent(video.ingest_progress ?? 0),
                subtitle: progressLabel(2, stepCount(projectId), analysisSubtitle(video)),
              }),
          videoId: video.id,
        });
      }
      return { items: next };
    });
  },
  noteTask: (projectId) => {
    if (projectId) taskProjects.add(projectId);
  },
}));
