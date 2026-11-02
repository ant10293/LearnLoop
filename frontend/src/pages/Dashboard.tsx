import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RunsTable } from "../components/RunsTable";
import { Bone } from "../components/Skeleton";
import { api, formatUsd } from "../lib/api";
import { humanize } from "../lib/humanize";
import type { Dashboard } from "../lib/types";
import { AnalysisPending, RunAnalysis } from "./RunPages";

const RECENT_PLACEHOLDERS = 4;

const STATS: { label: string; bone: string; value: (data: Dashboard) => string }[] = [
  { label: "Videos", bone: "2ch", value: (data) => String(data.videos) },
  { label: "Completed runs", bone: "2ch", value: (data) => String(data.runs_completed) },
  { label: "Active runs", bone: "2ch", value: (data) => String(data.runs_active) },
  { label: "Objects", bone: "2ch", value: (data) => String(data.objects) },
  { label: "Actions", bone: "2ch", value: (data) => String(data.actions) },
  { label: "Workflows", bone: "2ch", value: (data) => String(data.workflows) },
  { label: "Ambiguities", bone: "2ch", value: (data) => String(data.ambiguities) },
  { label: "Connected providers", bone: "2ch", value: (data) => String(data.providers_connected.length) },
  { label: "Estimated cost", bone: "7ch", value: (data) => formatUsd(data.cost_usd) },
];

export function DashboardPage() {
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 3000 });
  const data = dash.data;
  const loading = data == null && !dash.isError;
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-medium">Research dashboard</h1>
          <p className="text-sm text-[var(--muted)]">
            What models claim about an unfamiliar procedure, and which claims the pictures support.
          </p>
        </div>
        <Link className="btn primary" to="/projects?new=1">New project</Link>
      </header>
      {dash.isError && !data && <p className="text-[var(--rose)]">{(dash.error as Error).message}</p>}
      {(data || loading) && (
        <>
          {loading && <p className="sr-only">Loading dashboard</p>}
          <section className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-busy={loading || undefined}>
            {STATS.map((stat) => (
              <div key={stat.label} className="panel p-3">
                <div className="text-xs uppercase tracking-wide text-[var(--muted)]">{stat.label}</div>
                <div className="mono text-2xl">{data ? stat.value(data) : <Bone width={stat.bone} />}</div>
              </div>
            ))}
          </section>
          <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="panel relative min-w-0">
              {/* Out of flow on large screens so the chart column sets the row height and this list scrolls inside it. */}
              <div className="flex max-h-[32rem] flex-col p-4 lg:absolute lg:inset-0 lg:max-h-none">
                <h2 className="mb-3 shrink-0 text-sm font-medium">Recent experiments</h2>
                <div className="min-h-0 min-w-0 flex-1 overflow-auto" aria-busy={loading || undefined}>
                  {loading && <p className="sr-only">Loading experiments</p>}
                  {data && !data.recent_runs.length ? (
                    <p className="text-sm text-[var(--muted)]">No experiments yet. Start one, or let the fixture laboratory finish seeding.</p>
                  ) : (
                    <RunsTable runs={data ? data.recent_runs : Array.from({ length: RECENT_PLACEHOLDERS }, () => null)} />
                  )}
                </div>
              </div>
            </div>
            <SpendPanel data={data ?? null} />
          </section>
        </>
      )}
      <LatestAnalysis />
    </div>
  );
}

function SpendPanel({ data }: { data: Dashboard | null }) {
  return (
    <div className="panel p-4">
      <h2 className="mb-2 text-sm font-medium">Actions per recent run</h2>
      <div className="h-52">
        {data ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.recent_runs.map((run) => ({ name: humanize(run.model_id), actions: run.action_count, objects: run.object_count }))}>
              <XAxis dataKey="name" stroke="#8b95a8" fontSize={11} />
              <YAxis stroke="#8b95a8" fontSize={11} allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="actions" fill="#3ddc97" />
              <Bar dataKey="objects" fill="#7eb6ff" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full overflow-hidden rounded-md" aria-hidden="true">
            <Bone className="bone-fill" />
          </div>
        )}
      </div>
      <div className="mt-3 space-y-1 text-xs text-[var(--muted)]">
        <div className="flex justify-between gap-3 text-[var(--text)]">
          <span>Estimated API cost</span>
          <span className="mono">{data ? formatUsd(data.cost_usd) : <Bone width="7ch" />}</span>
        </div>
        {data ? data.cost_by_provider.map((row) => (
          <div key={row.provider_id} className="flex justify-between gap-3">
            <span>{humanize(row.provider_id)} · {row.calls} {row.calls === 1 ? "call" : "calls"}</span>
            <span className="mono">
              {row.unpriced_calls > 0 ? `${row.unpriced_calls} unpriced · ` : ""}
              {formatUsd(row.cost_usd)}
            </span>
          </div>
        )) : (
          Array.from({ length: 2 }, (_, index) => (
            <div key={index} className="flex justify-between gap-3" aria-hidden="true">
              <Bone width="46%" />
              <Bone width="7ch" />
            </div>
          ))
        )}
        {data && data.unitemized_usd > 0 && (
          <div className="flex justify-between gap-3">
            <span>Stored on runs without a call record</span>
            <span className="mono">{formatUsd(data.unitemized_usd)}</span>
          </div>
        )}
        <p>
          {data ? (
            <>
              Tokens {data.input_tokens + data.output_tokens}. Catalog list prices, not an invoice.
              Connected means a validation request succeeded: {data.providers_connected.map(humanize).join(", ") || "none yet"}.
            </>
          ) : (
            <Bone width="100%" />
          )}
        </p>
      </div>
    </div>
  );
}

function LatestAnalysis() {
  const runs = useQuery({
    queryKey: ["runs", "completed"],
    queryFn: () => api.runs({ status: "completed" }),
    refetchInterval: 3000,
  });
  const first = runs.data?.[0];
  const loading = runs.data === undefined && !runs.isError;
  return (
    <section className="panel space-y-3 p-4">
      <h2 className="text-sm font-medium">Latest analysis</h2>
      {runs.isError && <p className="text-sm text-[var(--rose)]">{(runs.error as Error).message}</p>}
      {loading && <AnalysisPending embedded tab="objects" />}
      {runs.isSuccess && !first && <p className="text-sm text-[var(--muted)]">No completed analysis yet.</p>}
      {first && <RunAnalysis key={first.id} runId={first.id} embedded />}
    </section>
  );
}
