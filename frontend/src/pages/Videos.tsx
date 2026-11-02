import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { AssetDropZone } from "../components/AssetDropZone";
import { Bone, MediaPlaceholder } from "../components/Skeleton";
import { StorageSettings } from "../components/StorageSettings";
import { api, formatTime } from "../lib/api";
import type { VideoInfo } from "../lib/types";
import { LIBRARY_QUEUE, useUploads } from "../stores/uploads";

const PLACEHOLDERS = 3;

function LibraryFace({ video }: { video?: VideoInfo }) {
  if (!video) {
    return (
      <div className="video-face" aria-hidden="true">
        <div className="video-face-media">
          <Bone className="bone-fill" />
        </div>
        <div className="video-face-meta space-y-1 p-3">
          <Bone width="72%" />
          <Bone width="92%" />
        </div>
      </div>
    );
  }
  const waiting = !video.thumbnail_url && video.ingest_status !== "failed";
  return (
    <div className="video-face">
      <div className="video-face-media">
        {waiting ? <MediaPlaceholder /> : video.thumbnail_url && <img src={video.thumbnail_url} alt="" />}
      </div>
      <div className="video-face-meta space-y-1 p-3 text-sm">
        <div className="truncate">{video.filename}</div>
        <div className="mono truncate text-xs text-[var(--muted)]">
          {waiting ? <Bone width="92%" /> : (
            <>
              {formatTime(video.duration_sec)} · {video.width}×{video.height} · {video.fps.toFixed(2)} fps
              {video.variable_frame_rate ? " · VFR" : " · CFR"} · {video.codec}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function VideosPage() {
  const client = useQueryClient();
  const videos = useQuery({ queryKey: ["videos", "library"], queryFn: api.unassignedVideos });
  const labSettings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const items = useUploads((state) => state.items);
  const enqueueUnassigned = useUploads((state) => state.enqueueUnassigned);
  const referenceDropped = useUploads((state) => state.referenceDropped);
  const removeItem = useUploads((state) => state.removeItem);
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteVideo(id),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["videos"] });
    },
  });
  const reference = labSettings.data?.video_storage === "reference";
  const storagePending = labSettings.data === undefined && !labSettings.isError;
  const pending = videos.data === undefined && !videos.isError;
  const rows = items.filter((item) => item.projectId === LIBRARY_QUEUE);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-medium">Video library</h1>
        <p className="text-sm text-[var(--muted)]">Videos here are not in a project. Drop one, or add it to a project from that project.</p>
      </header>
      <StorageSettings />
      {videos.isError && <p className="text-sm text-[var(--rose)]">{(videos.error as Error).message}</p>}
      {remove.isError && <p className="text-sm text-[var(--rose)]">{(remove.error as Error).message}</p>}
      {storagePending ? (
        <div className="drop-zone" aria-hidden="true">
          <Bone width="16rem" />
          <Bone width="10rem" />
        </div>
      ) : (
        <AssetDropZone
          accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
          multiple
          prompt="Drag and drop videos here or"
          chooseLabel="Choose files"
          supported="Supported: MP4, MOV, WebM"
          rows={rows.map((item) => ({
            id: item.key,
            name: item.name,
            url: item.previewUrl,
            subtitle: item.subtitle,
            progress: item.progress,
            failed: item.phase === "error",
          }))}
          onRemove={removeItem}
          onFiles={(files, paths) => {
            if (!files.length) return;
            if (reference) referenceDropped(LIBRARY_QUEUE, files, paths);
            else enqueueUnassigned(files);
          }}
        />
      )}
      {pending && (
        <div className="video-grid" aria-busy="true">
          <p className="sr-only">Loading videos</p>
          {Array.from({ length: PLACEHOLDERS }, (_, index) => (
            <div key={index} className="video-card panel overflow-hidden">
              <LibraryFace />
            </div>
          ))}
        </div>
      )}
      {videos.isSuccess && (
        videos.data.length ? (
          <div className="video-grid">
            {videos.data.map((video) => (
              <div key={video.id} className="video-card panel overflow-hidden">
                <button
                  type="button"
                  className="video-card-delete"
                  aria-label={`Delete ${video.filename}`}
                  title="Delete video"
                  disabled={remove.isPending && remove.variables === video.id}
                  onClick={() => {
                    if (window.confirm(`Delete “${video.filename}”? A referenced original stays on disk.`)) remove.mutate(video.id);
                  }}
                >
                  <Trash2 size={15} aria-hidden="true" />
                </button>
                <Link to={`/videos/${video.id}`}>
                  <LibraryFace video={video} />
                </Link>
              </div>
            ))}
          </div>
        ) : (
          !rows.length && <p className="text-sm text-[var(--muted)]">The library is empty.</p>
        )
      )}
    </div>
  );
}

const VIDEO_FACTS: { key: string; label: string; bone: string; mono?: boolean; value: (item: VideoInfo) => string }[] = [
  { key: "checksum", label: "Checksum", bone: "12ch", mono: true, value: (item) => item.checksum },
  { key: "duration", label: "Duration", bone: "6ch", value: (item) => formatTime(item.duration_sec) },
  { key: "frame", label: "", bone: "18ch", value: (item) => `${item.width}×${item.height} · ${item.codec} · ${item.container}` },
  { key: "capture", label: "Capture", bone: "18ch", value: (item) => `${item.capture_time_utc || item.capture_time_raw || "not in the file"} (${item.timestamp_reliability || "absent"})` },
  { key: "mtime", label: "File modified", bone: "10ch", value: (item) => item.file_mtime || "unknown" },
  { key: "source", label: "Source", bone: "8ch", value: (item) => item.timestamp_source || "absent" },
];

function VideoFact({ item, fact }: { item: VideoInfo | null; fact: (typeof VIDEO_FACTS)[number] }) {
  const value: ReactNode = item ? fact.value(item) : <Bone width={fact.bone} />;
  return (
    <div>
      {fact.label ? <>{fact.label} </> : null}
      {fact.mono ? <span className="mono text-xs">{value}</span> : value}
    </div>
  );
}

export function VideoDetailPage() {
  const { videoId = "" } = useParams();
  const [params] = useSearchParams();
  const player = useRef<HTMLVideoElement>(null);
  const video = useQuery({ queryKey: ["video", videoId], queryFn: () => api.video(videoId), enabled: Boolean(videoId) });
  useEffect(() => {
    const node = player.current;
    const time = Number(params.get("t"));
    if (!node || !Number.isFinite(time)) return;
    const seek = () => {
      node.currentTime = time;
    };
    node.addEventListener("loadedmetadata", seek);
    if (node.readyState >= 1) seek();
    return () => node.removeEventListener("loadedmetadata", seek);
  }, [params, video.data?.id]);
  const item = video.data ?? null;
  const pending = item == null && !video.isError;
  if (video.isError && !item) return <p className="text-sm text-[var(--rose)]">{(video.error as Error).message}</p>;
  const backHref = item?.project_id ? `/projects/${item.project_id}` : "/videos";
  const backLabel = item?.project_id ? "← Back to project" : "← Back to library";
  return (
    <div className="space-y-4" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading video</p>}
      <div>
        {item ? (
          <Link className="text-sm text-[var(--muted)]" to={backHref}>{backLabel}</Link>
        ) : (
          <div className="text-sm" aria-hidden="true"><Bone width="11rem" /></div>
        )}
        <h1 className="mt-2 text-2xl font-medium">{item ? item.filename : <Bone width="16rem" />}</h1>
      </div>
      <p className="text-xs text-[var(--amber)]">Clip time starts at the beginning of this file. It is not the project capture clock.</p>
      {item ? (
        <video ref={player} src={item.playback_url} controls className="max-h-[480px] w-full rounded-lg bg-black" />
      ) : (
        <div className="relative aspect-video max-h-[480px] w-full overflow-hidden rounded-lg bg-black" aria-hidden="true">
          <Bone className="bone-fill absolute inset-0" />
        </div>
      )}
      <div className="panel grid gap-2 p-3 text-sm md:grid-cols-3">
        {VIDEO_FACTS.map((fact) => <VideoFact key={fact.key} item={item} fact={fact} />)}
      </div>
      {item?.contact_sheet_url && <img src={item.contact_sheet_url} alt="Contact sheet" className="panel w-full" />}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {item?.frames?.map((frame) => (
          <figure key={frame.id} className="panel overflow-hidden">
            <img src={frame.url} alt="" className="aspect-video w-full bg-black object-contain" />
            <figcaption className="mono px-2 py-1 text-xs text-[var(--muted)]">{frame.timestamp.toFixed(2)}s · {frame.reason}</figcaption>
          </figure>
        ))}
      </div>
      {item ? (
        <Link className="btn primary inline-block" to={`/experiments/new?video=${item.id}`}>Analyze this video</Link>
      ) : (
        <span className="btn primary" aria-hidden="true">Analyze this video</span>
      )}
    </div>
  );
}
