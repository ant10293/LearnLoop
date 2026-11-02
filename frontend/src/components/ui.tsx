import { useEffect, useRef, type ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";
import { clock, reviewLabels, verdictLabels, verdictTone, type ReviewState } from "../lib/learnloop";

export function Layout({ children, procedureId, title }: { children: ReactNode; procedureId?: string; title?: string }) {
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">LearnLoop</Link>
        {title && <span className="crumb">{title}</span>}
        {procedureId && (
          <nav className="flow" aria-label="Procedure steps">
            <NavLink to={`/procedures/${procedureId}/teach`}>Teach</NavLink>
            <NavLink to={`/procedures/${procedureId}/procedure`}>Procedure</NavLink>
            <NavLink to={`/procedures/${procedureId}/verify`}>Verify</NavLink>
          </nav>
        )}
      </header>
      <main className="page">{children}</main>
    </div>
  );
}

export function VerdictTag({ verdict, large }: { verdict: string; large?: boolean }) {
  return <span className={`verdict ${verdictTone(verdict)} ${large ? "large" : ""}`}>{verdictLabels[verdict] ?? verdict}</span>;
}

export function ReviewTag({ state }: { state: ReviewState }) {
  return state ? <span className={`review ${state}`}>{reviewLabels[state]}</span> : null;
}

export function ErrorNote({ message }: { message: string }) {
  return message ? <p className="error" role="alert">{message}</p> : null;
}

/** A video that jumps to `time` whenever a new request arrives, including the same time twice. */
export function Film({ src, time, seek, caption, label }: { src: string; time: number | null | undefined; seek: number; caption: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node || time == null) return;
    const jump = () => {
      node.pause();
      node.currentTime = time;
    };
    if (node.readyState >= 1) jump();
    else node.addEventListener("loadedmetadata", jump, { once: true });
  }, [src, time, seek]);
  return (
    <figure className="film">
      <figcaption><strong>{label}</strong><span className="mono">{clock(time)}</span></figcaption>
      {src ? <video ref={ref} src={src} controls muted playsInline aria-label={label} /> : <div className="film-empty">No recording</div>}
      <p className="muted small">{caption}</p>
    </figure>
  );
}
