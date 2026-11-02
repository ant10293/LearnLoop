import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { diagnosticEntries, rawRowsNear } from "../lib/analysis";
import { api, formatTime } from "../lib/api";
import { humanize } from "../lib/humanize";
import type { Bundle, PerceptionHypothesis } from "../lib/types";
import { useInspector } from "../stores/ui";

export function PerceptionPanel({ bundle }: { bundle: Bundle }) {
  const perception = bundle.perception;
  const client = useQueryClient();
  const setTime = useInspector((state) => state.setTime);
  const time = useInspector((state) => state.time);
  const [error, setError] = useState("");
  const [keep, setKeep] = useState("");
  const [drop, setDrop] = useState("");
  if (!perception) return null;
  const instances = perception.instances ?? [];
  const hypotheses = perception.hypotheses ?? [];
  const diagnostics = perception.diagnostics ?? {};

  async function refresh(work: Promise<unknown>) {
    setError("");
    try {
      await work;
      await client.invalidateQueries({ queryKey: ["run", bundle.run.id] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "The identity edit was not saved.");
    }
  }

  return (
    <section className="panel space-y-3 p-4 text-sm">
      <div>
        <h2 className="text-base">Tracked instances</h2>
        <p className="text-xs text-[var(--muted)]">
          {humanize(perception.strategy)} · detector {perception.detector_id || "unrecorded"}
          {perception.encoder_id ? ` · encoder ${perception.encoder_id}` : ""}
          {perception.encoder_status ? ` (${perception.encoder_status})` : ""}
          {perception.segmenter_id ? ` · ${perception.segmenter_id}` : ""}. Scores are uncalibrated and are not probabilities.
        </p>
        {perception.encoder_detail && <p className="text-xs text-[var(--amber)]">{perception.encoder_detail}</p>}
        {perception.segmenter_id === "sam-3.1" && (
          <p className="text-xs text-[var(--muted)]">
            SAM 3.1 supplies boxes, masks, and tracks. The primary model still owns actions, relations, and workflow judgments. A gap in a track is unobserved, not absence.
          </p>
        )}
        {perception.notes && <p className="text-xs text-[var(--muted)]">{perception.notes}</p>}
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-[var(--muted)]">
        {diagnosticEntries(diagnostics).map(([key, value]) => (
          <span key={key}>{humanize(key)} {value}</span>
        ))}
      </div>
      <ul className="space-y-2">
        {instances.map((item) => (
          <li key={item.instance_id} className="flex items-center gap-3 rounded border border-[var(--line)] p-2">
            <div className="min-w-0 flex-1">
              <button className="font-medium" onClick={() => setTime(item.first_seen)}>{humanize(item.instance_id)}</button>
              <span className="ml-2 text-xs text-[var(--muted)]">
                {item.status} · {formatTime(item.first_seen)}–{formatTime(item.last_seen)} · {item.observation_count} observations
                {item.uncertain ? " · uncertain" : ""}
                {item.semantic_label ? ` · ${item.semantic_label}` : " · no semantic class"}
              </span>
              <div className="text-xs text-[var(--muted)]">
                Trusted references {item.trusted_references} · candidate references {item.candidate_references}
                {item.human_initialized ? " · seeded by a person" : ""}
              </div>
            </div>
            <button className="btn shrink-0 whitespace-nowrap" onClick={() => refresh(api.splitInstance(bundle.run.id, item.instance_id, [time]))}>Split at playhead</button>
          </li>
        ))}
      </ul>
      {hypotheses.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm">Identity hypotheses</h3>
          {hypotheses.map((item) => (
            <HypothesisRow key={item.hypothesis_id} item={item} onSeek={setTime} onDecide={(decision) => refresh(api.decideHypothesis(bundle.run.id, item.hypothesis_id, decision))} />
          ))}
        </div>
      )}
      {instances.length > 1 && (
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-xs text-[var(--muted)]">Keep
            <select className="field" value={keep} onChange={(event) => setKeep(event.target.value)}>
              <option value="">Instance</option>
              {instances.map((item) => <option key={item.instance_id} value={item.instance_id}>{humanize(item.instance_id)}</option>)}
            </select>
          </label>
          <label className="text-xs text-[var(--muted)]">Merge into it
            <select className="field" value={drop} onChange={(event) => setDrop(event.target.value)}>
              <option value="">Instance</option>
              {instances.map((item) => <option key={item.instance_id} value={item.instance_id}>{humanize(item.instance_id)}</option>)}
            </select>
          </label>
          <button className="btn" disabled={!keep || !drop || keep === drop} onClick={() => refresh(api.mergeInstances(bundle.run.id, keep, drop))}>Merge instances</button>
        </div>
      )}
      <RawRows bundle={bundle} time={time} onSeek={setTime} />
      {error && <p className="text-xs text-[var(--rose)]">{error}</p>}
    </section>
  );
}

function RawRows({ bundle, time, onSeek }: { bundle: Bundle; time: number; onSeek: (time: number) => void }) {
  const raw = bundle.raw_observations;
  if (!raw || !raw.total) return null;
  const near = rawRowsNear(raw.rows, time);
  return (
    <div className="space-y-1" data-testid="raw-observations">
      <h3 className="text-sm">Raw evidence</h3>
      <p className="text-xs text-[var(--muted)]">
        {raw.total} stored {raw.total === 1 ? "row is" : "rows are"} hidden from the overlay because a canonical row, a conflict, or a nearer estimate covers that time.
        {raw.truncated ? " The list is truncated." : ""}
      </p>
      {near.length > 0 ? (
        <ul className="space-y-1 text-xs">
          {near.map((row, index) => (
            <li key={`${row.object_local_id}-${row.timestamp}-${row.source_model}-${index}`}>
              <button onClick={() => onSeek(row.timestamp)}>{formatTime(row.timestamp)}</button>{" "}
              {humanize(row.object_local_id)} · {row.source_model || row.geometry_authority || "model"} · {humanize(row.geometry_origin || "unknown")}
              {row.detection_support ? ` · detection ${row.detection_support}` : ""}
              {row.association_support ? ` · association ${row.association_support}` : ""}
              {row.association_status === "conflict" ? " · conflict" : ""}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-[var(--muted)]">No raw row is within half a second of the playhead.</p>
      )}
    </div>
  );
}

function HypothesisRow({
  item,
  onSeek,
  onDecide,
}: {
  item: PerceptionHypothesis;
  onSeek: (time: number) => void;
  onDecide: (decision: "confirmed" | "rejected") => void;
}) {
  const pending = item.decision !== "confirmed" && item.decision !== "rejected";
  return (
    <div className="flex items-center gap-3 rounded border border-[var(--line)] p-2">
      <div className="min-w-0 flex-1">
        <button className="text-xs" onClick={() => onSeek(item.timestamp)}>{formatTime(item.timestamp)} · {item.decision}</button>
        <div className="text-xs text-[var(--muted)]">
          {humanize(item.subject_instance_id)} → {humanize(item.candidate_instance_id)}
          {item.runner_up_instance_id ? ` · runner-up ${humanize(item.runner_up_instance_id)}` : ""}
        </div>
        <div className="text-xs text-[var(--muted)]">
          Appearance {score(item.appearance_cosine)} · motion {score(item.motion_agreement)} · association {score(item.association_score)} · margin {score(item.margin)}
        </div>
        <p className="text-xs">{item.note}</p>
      </div>
      {pending && (
        <div className="flex shrink-0 gap-2">
          <button className="btn whitespace-nowrap" onClick={() => onDecide("confirmed")}>Confirm</button>
          <button className="btn whitespace-nowrap" onClick={() => onDecide("rejected")}>Reject</button>
        </div>
      )}
    </div>
  );
}

function score(value: number | null) {
  return value == null ? "—" : value.toFixed(2);
}
