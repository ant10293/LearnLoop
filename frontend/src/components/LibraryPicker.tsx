import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Bone } from "./Skeleton";
import type { VideoInfo } from "../lib/types";

const PICKER_PLACEHOLDERS = 8;

function PickerCell({
  video,
  selected = false,
  onToggle,
}: {
  video: VideoInfo | null;
  selected?: boolean;
  onToggle?: () => void;
}) {
  if (!video) {
    return (
      <div className="library-picker-cell" aria-hidden="true">
        <span className="library-picker-thumb">
          <Bone className="bone-fill" />
        </span>
        <span className="library-picker-name"><Bone width="80%" /></span>
      </div>
    );
  }
  return (
    <button
      type="button"
      className={`library-picker-cell${selected ? " is-selected" : ""}`}
      aria-pressed={selected}
      aria-label={video.filename}
      onClick={onToggle}
    >
      <span className="library-picker-thumb">
        {video.thumbnail_url ? <img src={video.thumbnail_url} alt="" /> : null}
      </span>
      <span className="library-picker-name" title={video.filename}>{video.filename}</span>
    </button>
  );
}

export function LibraryPicker({
  videos,
  loading = false,
  error = "",
  initialIds = [],
  onClose,
  onOpen,
}: {
  videos: VideoInfo[];
  loading?: boolean;
  error?: string;
  initialIds?: string[];
  onClose: () => void;
  onOpen: (videos: VideoInfo[]) => void;
}) {
  const [selected, setSelected] = useState(() => new Set(initialIds));

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function toggle(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const chosen = videos.filter((video) => selected.has(video.id));

  return createPortal(
    <div className="library-picker-backdrop" onMouseDown={onClose}>
      <div
        className="library-picker"
        role="dialog"
        aria-modal="true"
        aria-label="Video library"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="library-picker-grid" aria-busy={loading || undefined}>
          {loading && <p className="sr-only">Loading videos</p>}
          {loading && Array.from({ length: PICKER_PLACEHOLDERS }, (_, index) => <PickerCell key={index} video={null} />)}
          {!loading && error && <p className="library-picker-empty">{error}</p>}
          {!loading && !error && !videos.length && <p className="library-picker-empty">No videos in the library.</p>}
          {!loading && !error && videos.map((video) => (
            <PickerCell key={video.id} video={video} selected={selected.has(video.id)} onToggle={() => toggle(video.id)} />
          ))}
        </div>
        <div className="library-picker-foot">
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="button" className="btn primary" disabled={!chosen.length} onClick={() => onOpen(chosen)}>
            Open
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
