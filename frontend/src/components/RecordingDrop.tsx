import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, type Recording } from "../lib/learnloop";
import { AssetDropZone, type DropRow } from "./AssetDropZone";
import { StorageSettings } from "./StorageSettings";
import { ErrorNote } from "./ui";

const ACCEPT = "video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm";

type Upload = { id: string; name: string; url: string; subtitle: string; percent: number | null; error: string };

/** The upload section from vision-test-bench. It sends recordings and lists this role's recordings. */
export function RecordingDrop({
  procedureId,
  role,
  prompt,
  recordings,
  onChange,
  loading,
}: {
  procedureId: string;
  role: Recording["role"];
  prompt: string;
  recordings: Recording[];
  onChange: (recording?: Recording) => void | Promise<void>;
  loading?: boolean;
}) {
  const storage = useQuery({ queryKey: ["storage"], queryFn: api.storage });
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [error, setError] = useState("");
  const aborts = useRef(new Map<string, () => void>());
  const nextUpload = useRef(0);
  const uploadsRef = useRef(uploads);
  uploadsRef.current = uploads;
  useEffect(() => () => {
    for (const item of uploadsRef.current) URL.revokeObjectURL(item.url);
  }, []);

  function patchUpload(id: string, changes: Partial<Upload>) {
    setUploads((current) => current.map((item) => (item.id === id ? { ...item, ...changes } : item)));
  }

  function dropUploads(gone: (item: Upload) => boolean) {
    setUploads((current) => current.filter((item) => {
      if (!gone(item)) return true;
      URL.revokeObjectURL(item.url);
      return false;
    }));
  }

  async function send(files: File[], paths?: string[]) {
    const file = files[0];
    if (!file) return;
    setError("");
    const reference = storage.data?.video_storage === "reference";
    const id = `upload-${++nextUpload.current}`;
    dropUploads((item) => Boolean(item.error));
    setUploads((current) => [...current, { id, name: file.name, url: URL.createObjectURL(file), subtitle: reference ? "Finding the original file" : "Uploading", percent: reference ? null : 0, error: "" }]);
    try {
      let created: Recording;
      if (reference) {
        const path = paths?.[0] || (await api.locate(file)).path;
        created = await api.reference(procedureId, role, path);
      } else {
        created = await api.upload(
          procedureId,
          role,
          file,
          (ratio) => patchUpload(id, { percent: Math.round(ratio * 100) }),
          (cancel) => {
            aborts.current.set(id, cancel);
          },
        );
      }
      await onChange(created);
      dropUploads((item) => item.id === id);
    } catch (exc) {
      if ((exc as Error).name === "AbortError") {
        dropUploads((item) => item.id === id);
        return;
      }
      patchUpload(id, { error: (exc as Error).message });
    } finally {
      aborts.current.delete(id);
    }
  }

  async function remove(id: string) {
    if (uploads.some((item) => item.id === id)) {
      const cancel = aborts.current.get(id);
      if (cancel) cancel();
      else dropUploads((item) => item.id === id);
      return;
    }
    const recording = recordings.find((item) => item.video.id === id);
    if (recording?.run && !window.confirm(`Remove ${recording.video.filename} and its analysis?`)) return;
    setError("");
    try {
      await api.removeRecording(id);
      await onChange();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const rows: DropRow[] = recordings.map(({ video }) => {
    const failed = video.ingest_status === "failed";
    const working = !failed && video.ingest_status !== "ready";
    return {
      id: video.id,
      name: video.filename,
      url: video.thumbnail_url,
      subtitle: failed ? video.ingest_error || "This recording could not be read." : working ? video.ingest_detail || "Preparing the recording" : `${video.duration_sec.toFixed(1)} seconds`,
      progress: working ? (video.ingest_progress ?? null) : null,
      failed,
    };
  });
  for (const upload of uploads) {
    rows.push({ id: upload.id, name: upload.name, url: upload.url, subtitle: upload.error || upload.subtitle, progress: upload.error ? null : upload.percent, failed: Boolean(upload.error) });
  }

  return (
    <div className="upload-section">
      <StorageSettings />
      <AssetDropZone
        accept={ACCEPT}
        prompt={prompt}
        chooseLabel="Choose file"
        supported="Supported: MP4, MOV, WebM"
        rows={rows}
        loading={loading}
        onRemove={(id) => void remove(id)}
        onFiles={(files, paths) => void send(files, paths)}
      />
      <ErrorNote message={error} />
    </div>
  );
}
