import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { RecordingAnalysis } from "../components/RecordingAnalysis";
import { ErrorNote, Layout } from "../components/ui";
import { api, type Run } from "../lib/learnloop";

export function TeachPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const procedure = useQuery({ queryKey: ["procedure", id], queryFn: () => api.procedure(id) });
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const demo = [...(procedure.data?.recordings ?? [])].reverse().find((item) => item.role === "demonstration");

  async function upload(file?: File) {
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      await api.upload(id, "demonstration", file);
      await client.invalidateQueries({ queryKey: ["procedure", id] });
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setUploading(false);
    }
  }

  const learned = useCallback(
    async (run: Run) => {
      if (!procedure.data?.templates.some((item) => item.source_video_id === run.video_id)) {
        await api.learn(id, run.id);
      }
      await client.invalidateQueries({ queryKey: ["procedure", id] });
      navigate(`/procedures/${id}/procedure`);
    },
    [client, id, navigate, procedure.data],
  );

  return (
    <Layout procedureId={id} title={procedure.data?.name}>
      <h1>Teach</h1>
      <p className="lead">Upload a recording of the task done correctly.</p>
      <ErrorNote message={error || (procedure.error ? (procedure.error as Error).message : "")} />
      <div className="split">
        <section className="panel">
          {demo?.video.playback_url && demo.video.ingest_status === "ready" ? (
            <video className="preview" src={demo.video.playback_url} controls muted playsInline aria-label="Demonstration" />
          ) : (
            <div className="film-empty">{demo ? "Preparing the recording" : "No demonstration yet"}</div>
          )}
          <label className={`btn ${demo ? "" : "primary"} upload`}>
            {uploading ? "Uploading" : demo ? "Replace demonstration" : "Choose video"}
            <input type="file" accept="video/mp4,video/quicktime,video/webm,.mov,.mp4,.webm" disabled={uploading} onChange={(event) => void upload(event.target.files?.[0])} />
          </label>
        </section>
        <section className="panel">
          <h2>Recordings that work</h2>
          <ul className="guide">
            <li>Camera fixed in one place</li>
            <li>Up to {status.data?.max_clip_seconds ?? 60} seconds</li>
            <li>Good lighting, whole workspace in view</li>
            <li>A few large, distinct objects</li>
            <li>Hands clear of the finished result at the end</li>
          </ul>
          <p className="muted small">Moving or head-mounted cameras and small parts are not supported in this version.</p>
          <p className="small">Model: <strong>{status.data?.model_id}</strong> (Meta)</p>
          {demo && <RecordingAnalysis key={demo.video.id} recording={demo} onDone={learned} action="Learn the procedure" />}
        </section>
      </div>
    </Layout>
  );
}
