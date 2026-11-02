import { Star } from "lucide-react";
import { Link } from "react-router-dom";
import { formatDuration, formatUsd, formatWhen } from "../lib/api";
import { humanize, humanizeText } from "../lib/humanize";
import type { RunSummary } from "../lib/types";
import { Bone } from "./Skeleton";

const COLUMNS: { label: string; className?: string }[] = [
  { label: "Run id" },
  { label: "Research / video type" },
  { label: "Model" },
  { label: "Status" },
  { label: "Cost", className: "num" },
  { label: "Objects", className: "num" },
  { label: "Actions", className: "num" },
  { label: "Date", className: "date" },
  { label: "Duration", className: "num" },
];

const PLACEHOLDER_WIDTHS = ["16rem", "12rem", "8rem", "5rem", "5rem", "3rem", "3rem", "7rem", "4rem"];

export function RunsTable({ runs }: { runs: (RunSummary | null)[] }) {
  return (
    <table className="runs-table">
      <thead>
        <tr>
          {COLUMNS.map((column) => (
            <th key={column.label} className={column.className} scope="col">
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {runs.map((run, index) => (
          <tr key={run?.id ?? `pending-${index}`}>
            {run ? <RunCells run={run} /> : <PlaceholderCells />}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function researchVideoType(run: RunSummary): string {
  if (run.research_mode) return humanize(run.research_mode);
  if (run.modality) return humanize(run.modality);
  return "—";
}

function RunCells({ run }: { run: RunSummary }) {
  const when = formatWhen(run.created_at);
  const type = researchVideoType(run);
  const name = humanizeText(run.name);
  return (
    <>
      <th className="mono" scope="row">
        {run.id}
        {run.starred && <Star className="ml-1.5 inline-block align-[-2px] text-[var(--amber)]" size={14} fill="currentColor" role="img" aria-label="Starred" />}
      </th>
      <td>
        <Link className="run-link" to={`/runs/${run.id}`}>
          <span className="block">{type}</span>
          {name && name !== type && <span className="block text-xs font-normal text-[var(--muted)]">{name}</span>}
        </Link>
      </td>
      <td>
        <span className="block">{humanize(run.model_id) || "—"}</span>
        {run.provider_id && <span className="block text-xs text-[var(--muted)]">{humanize(run.provider_id)}</span>}
      </td>
      <td>
        <span className="tag" title={run.status === "failed" && run.error ? run.error : undefined}>{humanize(run.status)}</span>
      </td>
      <td className="num mono">
        {formatUsd(run.cost_usd)}
        {run.unpriced_calls > 0 ? <span className="block text-xs text-[var(--muted)]">{run.unpriced_calls} unpriced</span> : null}
      </td>
      <td className="num mono">{run.object_count}</td>
      <td className="num mono">{run.action_count}</td>
      <td className="date">{when ? <time dateTime={run.created_at}>{when}</time> : "—"}</td>
      <td className="num mono">{formatDuration(run.duration_sec)}</td>
    </>
  );
}

function PlaceholderCells() {
  return (
    <>
      {PLACEHOLDER_WIDTHS.map((width, index) => (
        <td key={index} className={COLUMNS[index]?.className} aria-hidden="true">
          <Bone width={width} />
        </td>
      ))}
    </>
  );
}
