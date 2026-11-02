import { Trash2 } from "lucide-react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Link } from "react-router-dom";
import { formatTime } from "../lib/api";
import type { VideoInfo } from "../lib/types";
import { Bone, MediaPlaceholder } from "./Skeleton";

export function VideoFace({ video, order }: { video?: VideoInfo; order?: number }) {
  if (!video || order == null) {
    return (
      <div className="video-face" aria-hidden="true">
        <div className="video-face-media">
          <Bone className="bone-fill" />
        </div>
        <div className="video-face-meta flex items-end gap-3 p-3 text-sm">
          <div className="min-w-0 flex-1 space-y-1">
            <div className="truncate"><Bone width="72%" /></div>
            <div className="mono truncate text-xs text-[var(--muted)]"><Bone width="92%" /></div>
          </div>
          <div className="mono shrink-0 text-xs text-[var(--muted)]"><Bone width="1.5ch" /></div>
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
      <div className="video-face-meta flex items-end gap-3 p-3 text-sm">
        <div className="min-w-0 flex-1 space-y-1">
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
        <div className="mono shrink-0 text-xs text-[var(--muted)]" aria-label={`Order ${order}`}>{order}</div>
      </div>
    </div>
  );
}

export function VideoCard({
  video,
  order,
  sortIndex,
  deleting,
  actionLabel,
  actionTitle = "Delete video",
  onDelete,
  onGrab,
}: {
  video: VideoInfo;
  order: number;
  sortIndex?: number;
  deleting: boolean;
  actionLabel?: string;
  actionTitle?: string;
  onDelete: () => void;
  onGrab?: (event: ReactPointerEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      className={`video-card panel overflow-hidden${onGrab ? "" : " is-static"}`}
      data-sort-index={onGrab ? sortIndex : undefined}
      onPointerDown={onGrab}
    >
      <button
        type="button"
        className="video-card-delete"
        aria-label={actionLabel ?? `Delete ${video.filename}`}
        title={actionTitle}
        disabled={deleting}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onDelete();
        }}
      >
        <Trash2 size={15} aria-hidden="true" />
      </button>
      <Link to={`/videos/${video.id}`} draggable={false}>
        <VideoFace video={video} order={order} />
      </Link>
    </div>
  );
}
