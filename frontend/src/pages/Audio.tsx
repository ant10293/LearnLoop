import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, formatTime } from "../lib/api";
import { humanize } from "../lib/humanize";
import { useOptionalBundle } from "../lib/bundleContext";
import { Bone, FieldBone } from "../components/Skeleton";
import type { AudioAlignment, TranscriptSegment } from "../lib/types";
import { useInspector } from "../stores/ui";

const FILTERS = ["all", "visually_supported", "contextual", "contradictory", "irrelevant", "uncertain"];
const SEGMENT_PLACEHOLDERS = 2;

const TONE: Record<string, string> = {
  visually_supported: "text-[var(--teal)]",
  contextual: "text-[var(--amber)]",
  contradictory: "text-[var(--rose)]",
  irrelevant: "text-[var(--muted)]",
  uncertain: "text-[var(--amber)]",
};

export function AudioPage() {
  const bundle = useOptionalBundle();
  const setTime = useInspector((state) => state.setTime);
  const [filter, setFilter] = useState("all");
  const [compareId, setCompareId] = useState("");
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const client = useQueryClient();
  const navigate = useNavigate();
  const config = bundle?.config || {};
  const segments = bundle?.transcripts ?? [];
  const alignments = bundle?.alignments ?? [];
  const bySegment = new Map(alignments.map((row) => [row.segment_id, row]));
  const visible = segments.filter((segment) => {
    if (filter === "all") return true;
    return bySegment.get(segment.id)?.classification === filter;
  });
  const models = useQuery({ queryKey: ["audio-models"], queryFn: api.audioModels });
  const saveText = useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) => api.patchTranscript(bundle?.run.id ?? "", id, text),
    onSuccess: (next) => { if (bundle) client.setQueryData(["run", bundle.run.id], next); },
  });
  const saveClass = useMutation({
    mutationFn: (row: AudioAlignment) =>
      api.patchAlignment(bundle?.run.id ?? "", row.id, row.classification, row.explanation),
    onSuccess: (next) => { if (bundle) client.setQueryData(["run", bundle.run.id], next); },
  });
  const visualCopy = useMutation({
    mutationFn: () => api.visualOnlyRun(bundle?.run.id ?? ""),
    onSuccess: (copy) => navigate(`/runs/${copy.id}/pipeline`),
  });
  const compare = useMutation({
    mutationFn: () => api.audioComparison(bundle?.run.id ?? "", compareId, bundle?.run.id ?? ""),
    onSuccess: (body) => setReport(body),
  });

  const pending = bundle == null;
  return (
    <div className="space-y-4" aria-busy={pending || undefined}>
      {pending && <p className="sr-only">Loading audio</p>}
      <section className="space-y-2">
        <h2 className="text-sm font-medium">Whisper configuration</h2>
        <p className="text-xs text-[var(--muted)]">
          {bundle ? (
            <>
              Transcription {config.transcribe ? "on" : "off"} · workflow use {config.use_transcript ? "on" : "off"} · model {humanize(String(config.whisper_model || "base"))} · language {String(config.whisper_language || "auto")} · silence filter {config.whisper_vad === false ? "off" : "on"} · window {String(config.align_window_sec || 4)}s
            </>
          ) : <Bone width="80%" />}
        </p>
        {bundle && Boolean(config.transcript_text_may_be_sent) && (
          <p className="text-xs text-[var(--amber)]">
            This run can send transcript text to {humanize(bundle.run.provider_id)}. The audio track stays on this machine.
          </p>
        )}
        {models.data && !models.data.installed && <p className="text-xs text-[var(--amber)]">{models.data.detail}</p>}
        {models.data?.models?.some((item) => !item.downloaded) && (
          <p className="text-xs text-[var(--muted)]">
            Not downloaded yet: {models.data.models.filter((item) => !item.downloaded).map((item) => humanize(item.model)).join(", ")}. The first transcription of a model downloads it once.
          </p>
        )}
        {bundle && bundle.video && (
          <p className="text-xs text-[var(--muted)]">
            {bundle.video.audio_probed
              ? (bundle.video.has_audio ? `Audio track: ${bundle.video.audio_codec || "present"}` : "No audio track. Visual analysis is unchanged.")
              : "Audio has not been checked. Transcription is the only stage that reads the audio track."}
          </p>
        )}
      </section>

      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-sm font-medium">Transcript timeline</h2>
          {FILTERS.map((item) => (
            <button key={item} className={`btn ${filter === item ? "primary" : ""}`} onClick={() => setFilter(item)}>
              {humanize(item)}
            </button>
          ))}
        </div>
        {bundle && visible.length === 0 && (
          <p className="text-sm text-[var(--muted)]">
            {config.transcribe ? "No speech segments are stored for this filter." : "Local transcription was not requested for this run."}
          </p>
        )}
        <ul className="space-y-2">
          {pending ? Array.from({ length: SEGMENT_PLACEHOLDERS }, (_, index) => (
            <SegmentCard key={index} segment={null} />
          )) : visible.map((segment) => (
            <SegmentCard
              key={segment.id}
              segment={segment}
              alignment={bySegment.get(segment.id)}
              onSeek={(time) => setTime(time)}
              onText={(text) => saveText.mutate({ id: segment.id, text })}
              onClass={(row) => saveClass.mutate(row)}
            />
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-sm font-medium">Visual-only comparison</h2>
        <p className="text-xs text-[var(--muted)]">
          A visual-only copy uses the same recording and model settings with both audio options off. Scores appear only when a human ground-truth set exists.
        </p>
        <div className="flex flex-wrap gap-2">
          <button className="btn" disabled={pending} onClick={() => { if (bundle) visualCopy.mutate(); }}>Start visual-only copy</button>
          <input className="field max-w-sm" placeholder="Visual-only run id" value={compareId} disabled={pending} onChange={(event) => setCompareId(event.target.value)} />
          <button className="btn" disabled={pending || !compareId || compare.isPending} onClick={() => { if (bundle) compare.mutate(); }}>Compare</button>
        </div>
        {report && <ComparisonReport report={report} />}
      </section>
    </div>
  );
}

function SegmentCard({
  segment,
  alignment,
  onSeek,
  onText,
  onClass,
}: {
  segment: TranscriptSegment | null;
  alignment?: AudioAlignment;
  onSeek?: (time: number) => void;
  onText?: (text: string) => void;
  onClass?: (row: AudioAlignment) => void;
}) {
  const [text, setText] = useState(segment?.text ?? "");
  const [label, setLabel] = useState(alignment?.classification || "uncertain");
  const [explanation, setExplanation] = useState(alignment?.explanation || "");
  if (!segment) {
    return (
      <li className="rounded-md border border-[var(--line)] p-3" aria-hidden="true">
        <Bone width="16rem" />
        <FieldBone textarea className="mt-2" />
        <div className="mt-2"><span className="btn">Save text</span></div>
      </li>
    );
  }
  return (
    <li className="rounded-md border border-[var(--line)] p-3">
      <button className="text-xs text-[var(--teal)]" onClick={() => onSeek?.(segment.start)}>
        {formatTime(segment.start)} – {formatTime(segment.end)} · {humanize(segment.local_id)}
        {segment.language ? ` · ${segment.language}` : ""}
        {segment.confidence != null ? ` · ${segment.confidence.toFixed(2)}` : ""}
      </button>
      <textarea className="field mt-2 w-full" value={text} onChange={(event) => setText(event.target.value)} />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button className="btn" onClick={() => onText?.(text)}>Save text</button>
        {segment.corrected && <span className="tag">corrected · model text kept</span>}
        {alignment && <span className={`text-xs ${TONE[alignment.classification] || ""}`}>{humanize(alignment.classification)}</span>}
      </div>
      {segment.corrected && segment.text_model !== segment.text && (
        <p className="mt-2 text-xs text-[var(--muted)]">Original model text: {segment.text_model}</p>
      )}
      {alignment && (
        <div className="mt-3 space-y-2 text-xs">
          <p>{alignment.explanation}</p>
          <p className="text-[var(--muted)]">
            Evidence {humanize(alignment.evidence_class)} · {humanize(alignment.provenance)}
            {alignment.speech_is_not_completion ? " · speech does not complete the step" : ""}
            {alignment.time_offset_sec != null ? ` · speech offset ${alignment.time_offset_sec}s` : ""}
          </p>
          <p>
            Actions {alignment.action_local_ids.map(humanize).join(", ") || "none"} · objects {alignment.object_local_ids.map(humanize).join(", ") || "none"} · steps {alignment.step_local_ids.map(humanize).join(", ") || "none"}
          </p>
          <div className="flex flex-wrap gap-2">
            {alignment.frame_timestamps.map((time) => (
              <button key={time} className="btn" onClick={() => onSeek?.(time)}>{formatTime(time)}</button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            <select className="field" value={label} onChange={(event) => setLabel(event.target.value)}>
              {FILTERS.filter((item) => item !== "all").map((item) => <option key={item} value={item}>{humanize(item)}</option>)}
            </select>
            <input className="field" value={explanation} onChange={(event) => setExplanation(event.target.value)} />
            <button className="btn" onClick={() => onClass?.({ ...alignment, classification: label, explanation })}>Save classification</button>
          </div>
        </div>
      )}
    </li>
  );
}

function ComparisonReport({ report }: { report: Record<string, unknown> }) {
  const actions = report.actions as { only_visual?: string[]; only_audio_assisted?: string[]; shared?: number } | undefined;
  const objects = report.objects as { only_visual?: string[]; only_audio_assisted?: string[] } | undefined;
  const steps = report.steps as { only_visual?: string[]; only_audio_assisted?: string[] } | undefined;
  return (
    <div className="rounded-md border border-[var(--line)] p-3 text-xs">
      <p>{String(report.note || "")}</p>
      <p className="mt-2">Workflow document changed: {report.workflow_document_changed ? "yes" : "no"}</p>
      <p>Actions only in visual: {(actions?.only_visual || []).map(humanize).join(", ") || "none"}</p>
      <p>Actions only in audio-assisted: {(actions?.only_audio_assisted || []).map(humanize).join(", ") || "none"}</p>
      <p>Objects only in one side: {[...(objects?.only_visual || []), ...(objects?.only_audio_assisted || [])].map(humanize).join(", ") || "none"}</p>
      <p>Steps only in one side: {[...(steps?.only_visual || []), ...(steps?.only_audio_assisted || [])].map(humanize).join(", ") || "none"}</p>
      <p>Latency (s): {JSON.stringify(report.latency_sec)}</p>
      <p>Cost: {JSON.stringify(report.cost_usd)}</p>
      <p>Speech classes: {JSON.stringify(report.speech_classifications)}</p>
      {report.ground_truth ? <p>Ground truth: {JSON.stringify(report.ground_truth)}</p> : null}
    </div>
  );
}
