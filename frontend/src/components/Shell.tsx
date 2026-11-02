import { NavLink, Outlet } from "react-router-dom";
import { Activity, Clapperboard, FlaskConical, FolderKanban, GitCompare, ScrollText, Settings2, Waypoints } from "lucide-react";

const LINKS = [
  ["/", "Dashboard", Activity],
  ["/projects", "Projects", FolderKanban],
  ["/videos", "Video library", Clapperboard],
  ["/experiments/new", "New experiment", FlaskConical],
  ["/compare", "Comparison lab", GitCompare],
  ["/providers", "Providers", Settings2],
  ["/prompts", "Prompts", ScrollText],
  ["/runs", "Runs", Waypoints],
] as const;

export function Shell() {
  return (
    <div className="grid min-h-screen grid-cols-1 md:grid-cols-[220px_1fr]">
      <aside className="border-b border-[var(--line)] bg-[#0e1117] p-4 md:border-b-0 md:border-r">
        <div className="mb-6">
          <div className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">LearnLoop</div>
          <div className="text-lg font-medium">Research Lab</div>
        </div>
        <nav className="flex gap-2 overflow-auto md:flex-col">
          {LINKS.map(([to, label, Icon]) => (
            <NavLink
              key={label}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2 rounded-md px-2 py-1.5 text-sm ${isActive ? "bg-[#1a221d] text-[var(--teal)]" : "text-[var(--muted)]"}`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="min-w-0 p-4 md:p-6">
        <Outlet />
      </main>
    </div>
  );
}
