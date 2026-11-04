import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { RecordingAnalysis } from "../components/RecordingAnalysis";
import { RecordingDrop } from "../components/RecordingDrop";
import { Bone } from "../components/Skeleton";
import { ErrorNote, Layout } from "../components/ui";
import { api, type Run } from "../lib/learnloop";

export function TeachPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const procedure = useQuery({
    queryKey: ["procedure", id],
    queryFn: () => api.procedure(id),
    refetchInterval: (query) => (query.state.data?.recordings.some((item) => !["ready", "failed"].includes(item.video.ingest_status)) ? 1000 : false),
  });
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });
  const demo = [...(procedure.data?.recordings ?? [])].reverse().find((item) => item.role === "demonstration");

  const learned = useCallback(
    async (run: Run) => {
      if (!procedure.data?.templates.some((item) => item.source_run_id === run.id)) {
        await api.learn(id, run.id);
      }
      await client.invalidateQueries({ queryKey: ["procedure", id] });
      navigate(`/procedures/${id}/procedure`);
    },
    [client, id, navigate, procedure.data],
  );

  return (
    <Layout procedureId={id} title={procedure.data?.name} titleLoading={procedure.isPending}>
      <h1>Teach</h1>
      <p className="lead">Upload a recording of the task done correctly.</p>
      <ErrorNote message={procedure.error ? (procedure.error as Error).message : ""} />
      <div className="split">
        <section className="panel">
          {demo?.video.ingest_status === "ready" && (
            <video className="preview upload-preview" src={demo.video.playback_url} controls muted playsInline aria-label="Demonstration" />
          )}
          <RecordingDrop
            procedureId={id}
            role="demonstration"
            prompt={demo ? "Drag and drop a new demonstration here or" : "Drag and drop a video here or"}
            recordings={demo ? [demo] : []}
            loading={procedure.isPending}
            onChange={() => client.invalidateQueries({ queryKey: ["procedure", id] })}
          />
        </section>
        <section className="panel">
          <h2>Tips for good recordings</h2>
          <ul className="guide">
            <li>Up to {status.isPending ? <Bone width="2ch" /> : status.data?.max_clip_seconds ?? 60} seconds</li>
            <li>Good lighting, with the workspace in view</li>
            <li>A steady, high resolution camera (1080p - 2160p)</li>
          </ul>
          <p className="muted small">These make learning easier, but aren't required:</p>
          <ul className="guide">
            <li>A few large, distinct objects</li>
            <li>All parts visible in the first frame</li>
            <li>Controlled, deliberate movements</li>
          </ul>
          {demo ? (
            <RecordingAnalysis key={demo.video.id} recording={demo} onDone={learned} action="Learn the procedure" />
          ) : procedure.data && (
            <div className="analysis">
              <button className="btn primary" disabled>Learn the procedure</button>
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
}
