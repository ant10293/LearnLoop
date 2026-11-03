import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from "react";
import { Maximize, Minimize, Pause, Play, Volume2, VolumeX } from "lucide-react";
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
          <div key={item.label + index} className={`film-box${box.status === "provisional" || box.status === "ambiguous" ? " unconfirmed" : ""}`} title={box.status === "ambiguous" ? "Appearance matches a similar part as well" : box.status === "provisional" ? "Not confirmed by appearance" : undefined} style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%`, borderColor: color }}>
            <span style={{ background: color }}>{item.label}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Play, scrub, mute and fullscreen for a tracked video. Native controls would fullscreen only the <video>, leaving the boxes behind. */
function FilmControls({ video, full, onFull }: { video: RefObject<HTMLVideoElement | null>; full: boolean; onFull: () => void }) {
  const [state, setState] = useState({ playing: false, muted: true, time: 0, duration: 0 });
  useEffect(() => {
    const node = video.current;
    if (!node) return;
    const sync = () => setState({ playing: !node.paused && !node.ended, muted: node.muted, time: node.currentTime, duration: Number.isFinite(node.duration) ? node.duration : 0 });
    const events = ["play", "pause", "ended", "timeupdate", "seeked", "loadedmetadata", "durationchange", "volumechange"];
    events.forEach((name) => node.addEventListener(name, sync));
    sync();
    return () => events.forEach((name) => node.removeEventListener(name, sync));
  }, [video]);
  const toggle = () => {
    const node = video.current;
    if (!node) return;
    if (node.paused || node.ended) void node.play();
    else node.pause();
  };
  const toggleMute = () => {
    if (video.current) video.current.muted = !video.current.muted;
  };
  const time = Math.min(state.time, state.duration || 0);
  const fill = state.duration ? (time / state.duration) * 100 : 0;
  return (
    <div className={`film-controls${state.playing ? "" : " paused"}`}>
      <div className="film-controls-row">
        <button type="button" onClick={toggle} aria-label={state.playing ? "Pause" : "Play"} title={state.playing ? "Pause" : "Play"}>{state.playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button>
        <span className="film-time">{clock(state.time)} / {clock(state.duration)}</span>
        <button type="button" className="film-mute" onClick={toggleMute} aria-label={state.muted ? "Unmute" : "Mute"} title={state.muted ? "Unmute" : "Mute"}>{state.muted ? <VolumeX size={18} /> : <Volume2 size={18} />}</button>
        <button type="button" onClick={onFull} aria-label={full ? "Exit full screen" : "Full screen"} title={full ? "Exit full screen" : "Full screen"}>{full ? <Minimize size={18} /> : <Maximize size={18} />}</button>
      </div>
      <input type="range" className="film-seek" min={0} max={state.duration || 0} step={0.01} value={time} aria-label="Seek" style={{ "--fill": `${fill}%` } as CSSProperties} onChange={(event) => { if (video.current) video.current.currentTime = Number(event.target.value); }} />
    </div>
  );
}

export function Film({ src, time, seek, caption, label, tracks }: { src?: string; time?: number | null; seek?: number; caption: ReactNode; label: string; tracks?: FilmTrack[] }) {
  const ref = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [full, setFull] = useState(false);
  const boxed = Boolean(tracks && tracks.length > 0);
  useEffect(() => {
    const sync = () => setFull(document.fullscreenElement !== null && document.fullscreenElement === stage.current);
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);
  const toggleFull = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.current?.requestFullscreen?.();
  };
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
      {src ? (
        <div ref={stage} className={`film-stage${boxed ? " boxed" : ""}`}>
          <video ref={ref} src={src} controls={!boxed} muted playsInline aria-label={label} onClick={boxed ? () => { const node = ref.current; if (node) void (node.paused ? node.play() : node.pause()); } : undefined} onDoubleClick={boxed ? toggleFull : undefined} />
          {boxed && <FilmBoxes video={ref} tracks={tracks!} />}
          {boxed && <FilmControls video={ref} full={full} onFull={toggleFull} />}
        </div>
      ) : <div className="film-empty">{src === undefined ? <Bone className="bone-fill" /> : "No recording"}</div>}
      <p className="muted small">{caption}</p>
    </figure>
  );
}
