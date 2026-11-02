import { useEffect, useState } from "react";
import { api, type Recording, type VideoInfo } from "../lib/learnloop";
import { AssetDropZone, type DropRow } from "./AssetDropZone";

const ACCEPT = "video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm";

type Upload = { name: string; url: string; percent: number; error: string };

/** The upload section from vision-test-bench, sending one recording to a procedure. */
export function RecordingDrop({
  procedureId,
  role,
  prompt,
  preparing,
  onUploaded,
}: {
  procedureId: string;
  role: Recording["role"];
  prompt: string;
  preparing?: VideoInfo;
  onUploaded: (recording: Recording) => void | Promise<void>;
}) {
  const [upload, setUpload] = useState<Upload | null>(null);
  useEffect(() => () => {
    if (upload) URL.revokeObjectURL(upload.url);
  }, [upload?.url]); // eslint-disable-line react-hooks/exhaustive-deps

  async function send(files: File[]) {
    const file = files[0];
    if (!file) return;
    setUpload({ name: file.name, url: URL.createObjectURL(file), percent: 0, error: "" });
    try {
      const created = await api.upload(procedureId, role, file, (ratio) =>
        setUpload((current) => current && { ...current, percent: Math.round(ratio * 100) }),
      );
      await onUploaded(created);
      setUpload(null);
    } catch (exc) {
      const message = (exc as Error).message;
      setUpload((current) => current && { ...current, error: message });
    }
  }

  const rows: DropRow[] = [];
  if (upload) {
    rows.push({
      id: "upload",
      name: upload.name,
      url: upload.url,
      subtitle: upload.error || "Uploading",
      progress: upload.error ? null : upload.percent,
      failed: Boolean(upload.error),
    });
  } else if (preparing && preparing.ingest_status !== "ready" && preparing.ingest_status !== "failed") {
    rows.push({
      id: preparing.id,
      name: preparing.filename,
      url: preparing.thumbnail_url,
      subtitle: preparing.ingest_detail || "Preparing the recording",
      progress: preparing.ingest_progress ?? null,
    });
  }

  return (
    <AssetDropZone
      accept={ACCEPT}
      prompt={prompt}
      chooseLabel="Choose file"
      supported="Supported: MP4, MOV, WebM"
      rows={rows}
      onRemove={upload?.error ? () => setUpload(null) : undefined}
      onFiles={(files) => void send(files)}
    />
  );
}
