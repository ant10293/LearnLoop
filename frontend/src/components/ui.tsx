import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { Link, NavLink } from "react-router-dom";
import { boxBetween, clock, smoothTrack, reviewLabels, verdictLabels, verdictTone, type FilmTrack, type ReviewState } from "../lib/learnloop";
import { Bone } from "./Skeleton";

export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18.36 5.64A9 9 0 1 1 12 3" />
      <path d="M9.6 .8 12.2 3 9.6 5.2" />
      <path d="m8 12.2 2.8 2.8 5.2-5.6" />
    </svg>
  );
}

export function Layout({ children, procedureId, title, titleLoading }: { children: ReactNode; procedureId?: string; title?: string; titleLoading?: boolean }) {
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand"><Logo />LearnLoop</Link>
        {(title || titleLoading) && <span className="crumb">{title ?? <Bone width="9rem" />}</span>}
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

/** A video that jumps to `time` whenever a new request arrives, including the same time twice. Without `src` it is still loading. */
const BOX_COLORS = ["#ffb020", "#22d3ee", "#f472b6", "#a3e635", "#c084fc", "#fb7185", "#60a5fa", "#fde047"];

/** Tracked boxes over the picture. A video letterboxes inside its element, so they follow the picture, not the element. */
function FilmBoxes({ video, tracks }: { video: RefObject<HTMLVideoElement | null>; tracks: FilmTrack[] }) {
  const [view, setView] = useState({ left: 0, top: 0, width: 0, height: 0, time: 0 });
  const smoothed = useMemo(() => tracks.map((item) => smoothTrack(item.track)), [tracks]);
  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const node = video.current;
      if (node && node.videoWidth) {
        const scale = Math.min(node.clientWidth / node.videoWidth, node.clientHeight / node.videoHeight);
        const width = node.videoWidth * scale;
        const height = node.videoHeight * scale;
        const next = { left: node.offsetLeft + (node.clientWidth - width) / 2, top: node.offsetTop + (node.clientHeight - height) / 2, width, height, time: node.currentTime };
        setView((current) => (Object.keys(next) as (keyof typeof next)[]).every((key) => current[key] === next[key]) ? current : next);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [video]);
  return (
    <div className="film-boxes" aria-hidden="true" style={{ left: view.left, top: view.top, width: view.width, height: view.height }}>
      {tracks.map((item, index) => {
        const box = boxBetween(smoothed[index], view.time);
        if (!box) return null;
        const color = BOX_COLORS[index % BOX_COLORS.length];
        return (
          <div key={item.label + index} className="film-box" style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%`, borderColor: color }}>
            <span style={{ background: color }}>{item.label}</span>
          </div>
        );
      })}
    </div>
  );
}

export function Film({ src, time, seek, caption, label, tracks }: { src?: string; time?: number | null; seek?: number; caption: ReactNode; label: string; tracks?: FilmTrack[] }) {
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
      <figcaption><strong>{label}</strong><span className="mono">{src === undefined ? <Bone width="3ch" /> : clock(time)}</span></figcaption>
      {src ? <div className="film-stage"><video ref={ref} src={src} controls muted playsInline aria-label={label} />{tracks && tracks.length > 0 && <FilmBoxes video={ref} tracks={tracks} />}</div> : <div className="film-empty">{src === undefined ? <Bone className="bone-fill" /> : "No recording"}</div>}
      <p className="muted small">{caption}</p>
    </figure>
  );
}
