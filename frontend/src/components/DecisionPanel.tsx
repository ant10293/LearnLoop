import { decisionsNear } from "../lib/analysis";
import { formatTime } from "../lib/api";
import { humanize } from "../lib/humanize";
import type { Bundle } from "../lib/types";
import { useInspector } from "../stores/ui";

export function DecisionPanel({ bundle }: { bundle: Bundle }) {
  const decisions = bundle.box_decisions;
  const time = useInspector((state) => state.time);
  const setTime = useInspector((state) => state.setTime);
  if (!decisions || !decisions.total) return null;
  const near = decisionsNear(decisions.rows, time).slice(0, 12);
  return (
    <section className="panel space-y-3 p-4 text-sm" data-testid="box-decisions">
      <div>
        <h2 className="text-base">Dropped, moved, and unmeasured boxes</h2>
        <p className="text-xs text-[var(--muted)]">
          {decisions.total} {decisions.total === 1 ? "record" : "records"} from this run's pipeline steps. Each box is shown as it was before the step changed it.
          {decisions.truncated ? " The list is truncated. The counts are complete." : ""}
        </p>
      </div>
      <table className="w-full text-left text-xs">
        <thead className="text-[var(--muted)]">
          <tr><th className="font-normal">Step</th><th className="font-normal">Reason</th><th className="font-normal">Outcome</th><th className="text-right font-normal">Boxes</th></tr>
        </thead>
        <tbody>
          {decisions.counts.map((row) => (
            <tr key={`${row.step}-${row.reason_code}-${row.outcome}`}>
              <td>{humanize(row.step)}</td>
              <td>{humanize(row.reason_code)}</td>
              <td>{humanize(row.outcome)}</td>
              <td className="text-right">{row.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="space-y-1">
        <h3 className="text-sm">Near the playhead</h3>
        {near.length > 0 ? (
          <ul className="space-y-1 text-xs">
            {near.map((row, index) => (
              <li key={`${row.step}-${row.reason_code}-${row.object_local_id}-${row.timestamp}-${index}`}>
                <button onClick={() => setTime(row.timestamp ?? 0)}>{formatTime(row.timestamp ?? 0)}</button>{" "}
                {row.object_local_id} · {humanize(row.reason_code)}
                {row.confidence != null ? ` · ${Math.round(row.confidence * 100)}%` : ""}
                {row.detail ? ` · ${row.detail}` : ""}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-[var(--muted)]">No record is within half a second of the playhead.</p>
        )}
      </div>
    </section>
  );
}
