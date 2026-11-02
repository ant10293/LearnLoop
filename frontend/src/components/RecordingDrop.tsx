import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, type Recording } from "../lib/learnloop";
import { AssetDropZone, type DropRow } from "./AssetDropZone";
import { StorageSettings } from "./StorageSettings";
import { ErrorNote } from "./ui";

const ACCEPT = "video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm";

type Upload = { name: string; url: string; subtitle: string; percent: number | null; error: string };

/** The upload section from vision-test-bench. It sends one recording and lists this role's recordings. */
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
  const [upload, setUpload] = useState<Upload | null>(null);
  const [error, setError] = useState("");
  const abort = useRef<(() => void) | null>(null);
  const previewUrl = upload?.url;
  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  async function send(files: File[], paths?: string[]) {
    const file = files[0];
    if (!file) return;
    setError("");
    const reference = storage.data?.video_storage === "reference";
    setUpload({ name: file.name, url: URL.createObjectURL(file), subtitle: reference ? "Finding the original file" : "Uploading", percent: reference ? null : 0, error: "" });
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
          (ratio) => setUpload((current) => current && { ...current, percent: Math.round(ratio * 100) }),
          (cancel) => {
            abort.current = cancel;
          },
        );
      }
      await onChange(created);
      setUpload(null);
    } catch (exc) {
      if ((exc as Error).name === "AbortError") {
        setUpload(null);
        return;
      }
      const message = (exc as Error).message;
      setUpload((current) => current && { ...current, error: message });
    } finally {
      abort.current = null;
    }
  }

  async function remove(id: string) {
    if (id === "upload") {
      if (abort.current) abort.current();
      else setUpload(null);
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
  if (upload) {
    rows.push({ id: "upload", name: upload.name, url: upload.url, subtitle: upload.error || upload.subtitle, progress: upload.error ? null : upload.percent, failed: Boolean(upload.error) });
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
