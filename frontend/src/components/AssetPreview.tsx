import { Download, EyeOff, ExternalLink, Maximize2, Minimize2, Search, SquareDashed, Undo2, X } from "lucide-react";
import { useEffect, useRef, useState, type MouseEvent, type PointerEvent, type SyntheticEvent } from "react";
import { createPortal } from "react-dom";

/** Where the object sits in its full frame, as fractions of the picture. */
export type PreviewBox = { x: number; y: number; w: number; h: number };
/**
 * src is the object's crop, or "" when the object was not found on this frame and only the full frame exists.
 * correction says what a reviewer did on this frame, if anything.
 */
export type PreviewImage = {
  src: string;
  filename: string;
  confidence?: number | null;
  full?: { src: string; box: PreviewBox | null };
  correction?: { label: string };
};

/** Lets a reviewer redraw the object's box on the shown frame, mark it not in the frame, or undo their correction. */
export type PreviewEditor = {
  onSave: (image: PreviewImage, box: PreviewBox | null) => Promise<void>;
  onUndo: (image: PreviewImage) => Promise<void>;
};

const MIN_SIDE = 0.005;
/** Corners and edges of the box being drawn. Each letter names the side a handle moves. */
const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
type Handle = (typeof HANDLES)[number];
/** What a press on the frame does: start a new box, move the box, or move some of its sides. */
type Gesture = { kind: "new"; from: { x: number; y: number } } | { kind: "move" | Handle; from: { x: number; y: number }; box: PreviewBox };

function clamp(value: number) {
  return Math.min(Math.max(value, 0), 1);
}

function rounded(box: PreviewBox): PreviewBox {
  const round = (value: number) => Math.round(value * 10000) / 10000;
  return { x: round(box.x), y: round(box.y), w: round(box.w), h: round(box.h) };
}

/** The box after a gesture has moved the pointer to `to`. Sides dragged past each other swap. */
function reshaped(gesture: Gesture, to: { x: number; y: number }): PreviewBox {
  if (gesture.kind === "new") {
    const { from } = gesture;
    return rounded({ x: Math.min(from.x, to.x), y: Math.min(from.y, to.y), w: Math.abs(to.x - from.x), h: Math.abs(to.y - from.y) });
  }
  const { box, from } = gesture;
  if (gesture.kind === "move") {
    const x = Math.min(Math.max(box.x + to.x - from.x, 0), 1 - box.w);
    const y = Math.min(Math.max(box.y + to.y - from.y, 0), 1 - box.h);
    return rounded({ ...box, x, y });
  }
  let left = box.x;
  let right = box.x + box.w;
  let top = box.y;
  let bottom = box.y + box.h;
  if (gesture.kind.includes("w")) left = to.x;
  if (gesture.kind.includes("e")) right = to.x;
  if (gesture.kind.includes("n")) top = to.y;
  if (gesture.kind.includes("s")) bottom = to.y;
  return rounded({ x: Math.min(left, right), y: Math.min(top, bottom), w: Math.abs(right - left), h: Math.abs(bottom - top) });
}

/** Largest box with the image's own aspect ratio that fits the expanded preview. */
function fittedPreviewSize(naturalWidth: number, naturalHeight: number, viewportWidth: number, viewportHeight: number) {
  if (naturalWidth <= 0 || naturalHeight <= 0 || viewportWidth <= 0 || viewportHeight <= 0) return null;
  const scale = Math.min((viewportWidth * 0.9) / naturalWidth, (viewportHeight * 0.85) / naturalHeight);
  return { width: naturalWidth * scale, height: naturalHeight * scale };
}

function PreviewNav({ direction, label, onClick }: { direction: "prev" | "next"; label: string; onClick: () => void }) {
  return (
    <button type="button" className={`asset-preview-nav is-${direction}`} title={label} aria-label={label} onClick={onClick}>
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <polyline points={direction === "prev" ? "15 18 9 12 15 6" : "9 6 15 12 9 18"} />
      </svg>
    </button>
  );
}

/** Full-screen image preview with zoom, previous/next, download, and an explicit new-tab action. From vision-test-bench. */
export function AssetPreview({ images, startIndex = 0, onClose, editor }: { images: PreviewImage[]; startIndex?: number; onClose: () => void; editor?: PreviewEditor }) {
  const [requested, setCurrent] = useState(() => Math.min(Math.max(0, startIndex), Math.max(0, images.length - 1)));
  // A correction can remove the shown frame from the list. The same position then shows the next one.
  const current = Math.min(requested, Math.max(0, images.length - 1));
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<PreviewBox | null>(null);
  const [busy, setBusy] = useState(false);
  const [editError, setEditError] = useState("");
  const gesture = useRef<Gesture | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const drawingRef = useRef(false);
  drawingRef.current = drawing;
  const [zoom, setZoom] = useState(1);
  const [box, setBox] = useState<{ width: number; height: number } | null>(null);
  const [showFull, setShowFull] = useState(false);
  const natural = useRef({ width: 0, height: 0 });
  const onCloseRef = useRef(onClose);
  const closeRef = useRef<HTMLButtonElement>(null);
  onCloseRef.current = onClose;
  const image = images[current];
  // A frame where the object was not found has no crop, so its full frame is all there is to show.
  const full = showFull || drawing || (image && !image.src) ? image?.full : undefined;
  const src = full?.src ?? image?.src;
  const fittedSrc = useRef(src);
  if (src !== fittedSrc.current) {
    fittedSrc.current = src;
    natural.current = { width: 0, height: 0 };
    setBox(null);
  }

  function fitToPreview() {
    setBox(fittedPreviewSize(natural.current.width, natural.current.height, window.innerWidth, window.innerHeight));
  }

  function onImageLoad(event: SyntheticEvent<HTMLImageElement>) {
    natural.current = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
    fitToPreview();
  }

  useEffect(() => {
    window.addEventListener("resize", fitToPreview);
    return () => window.removeEventListener("resize", fitToPreview);
  }, []);

  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        if (drawingRef.current) {
          setDrawing(false);
          setDraft(null);
          return;
        }
        onCloseRef.current();
        return;
      }
      if (images.length < 2 || drawingRef.current) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setZoom(1);
        setCurrent((index) => (index - 1 + images.length) % images.length);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setZoom(1);
        setCurrent((index) => (index + 1) % images.length);
      }
    }
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [images.length]);

  if (!image) return null;

  function go(delta: number) {
    setZoom(1);
    setEditError("");
    setCurrent((index) => (index + delta + images.length) % images.length);
  }

  function point(event: PointerEvent) {
    const rect = imgRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    return { x: clamp((event.clientX - rect.left) / rect.width), y: clamp((event.clientY - rect.top) / rect.height) };
  }

  function onPointerDown(event: PointerEvent<HTMLSpanElement>) {
    if (!drawing || busy) return;
    const from = point(event);
    if (!from) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    // A handle resizes the box and its body moves it. Anywhere else starts a new box.
    const target = event.target instanceof HTMLElement ? event.target : null;
    const handle = target?.dataset.handle as Handle | undefined;
    if (draft && handle) {
      gesture.current = { kind: handle, from, box: draft };
    } else if (draft && target?.classList.contains("is-draft")) {
      gesture.current = { kind: "move", from, box: draft };
    } else {
      gesture.current = { kind: "new", from };
      setDraft({ x: from.x, y: from.y, w: 0, h: 0 });
    }
  }

  function onPointerMove(event: PointerEvent<HTMLSpanElement>) {
    const current = gesture.current;
    const to = drawing && current ? point(event) : null;
    if (!current || !to) return;
    setDraft(reshaped(current, to));
  }

  function onPointerUp() {
    gesture.current = null;
  }

  async function edit(action: () => Promise<void>) {
    setBusy(true);
    setEditError("");
    try {
      await action();
      setDrawing(false);
      setDraft(null);
    } catch (exc) {
      setEditError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const drawn = draft && draft.w >= MIN_SIDE && draft.h >= MIN_SIDE ? draft : null;

  function setZoomAction(action: "in" | "out" | "fit") {
    setZoom((value) => {
      if (action === "in") return Math.min(value * 1.3, 5);
      if (action === "out") return Math.max(value / 1.3, 0.2);
      return 1;
    });
  }

  function download() {
    const link = document.createElement("a");
    link.href = src;
    link.download = image.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function onBackdrop(event: MouseEvent<HTMLDivElement>) {
    const target = event.target;
    if (target === event.currentTarget || (target instanceof HTMLElement && target.classList.contains("asset-preview-viewport"))) {
      onClose();
    }
  }

  return createPortal(
    <div className="asset-preview" role="dialog" aria-modal="true" aria-label={image.filename} onClick={onBackdrop}>
      <div className="asset-preview-topbar">
        <button ref={closeRef} type="button" className="asset-preview-close" title="Close" aria-label="Close" onClick={onClose}>
          <X size={24} strokeWidth={1.8} />
        </button>
        <div className="asset-preview-filename">{image.filename}</div>
      </div>
      {images.length > 1 && !drawing && (
        <>
          <PreviewNav direction="prev" label="Previous" onClick={() => go(-1)} />
          <PreviewNav direction="next" label="Next" onClick={() => go(1)} />
        </>
      )}
      {/* Actions on the picture itself sit in one capsule, centered. Download and open stay at the right. */}
      <div className="asset-preview-tools">
        {editor && image.full && !drawing && (
          <>
            {image.correction && (
              <button type="button" className="asset-preview-action has-label" disabled={busy} onClick={() => edit(() => editor.onUndo(image))}>
                <Undo2 size={18} strokeWidth={1.8} />
                Undo correction
              </button>
            )}
            <button
              type="button"
              className="asset-preview-action has-label"
              disabled={busy}
              onClick={() => {
                setZoom(1);
                setEditError("");
                // The current box is the starting point, so it can be adjusted instead of drawn again.
                setDraft(image.full?.box ? rounded(image.full.box) : null);
                setShowFull(true);
                setDrawing(true);
              }}
            >
              <SquareDashed size={18} strokeWidth={1.8} />
              {image.full.box ? "Redraw box" : "Draw box"}
            </button>
            {image.src && (
              <button type="button" className="asset-preview-action has-label" disabled={busy} onClick={() => edit(() => editor.onSave(image, null))}>
                <EyeOff size={18} strokeWidth={1.8} />
                Not in frame
              </button>
            )}
          </>
        )}
        {editor && drawing && (
          <>
            <span className="asset-preview-hint">{draft ? "Drag a corner or edge to resize, inside to move, or outside to start over" : "Drag across the whole object"}</span>
            <button
              type="button"
              className="asset-preview-action has-label"
              onClick={() => {
                setDrawing(false);
                setDraft(null);
              }}
            >
              Cancel
            </button>
            <button type="button" className="asset-preview-action has-label is-primary" disabled={!drawn || busy} onClick={() => drawn && edit(() => editor.onSave(image, drawn))}>
              Save box
            </button>
          </>
        )}
        {image.full && image.src && !drawing && (
          <button
            type="button"
            className="asset-preview-action has-label"
            onClick={() => {
              setZoom(1);
              setShowFull((value) => !value);
            }}
          >
            {full ? <Minimize2 size={18} strokeWidth={1.8} /> : <Maximize2 size={18} strokeWidth={1.8} />}
            {full ? "Object only" : "Full image"}
          </button>
        )}
      </div>
      <div className="asset-preview-actions">
        <button type="button" className="asset-preview-action" title="Download" aria-label="Download" onClick={download}>
          <Download size={20} strokeWidth={1.8} />
        </button>
        <button type="button" className="asset-preview-action" title="Open in new tab" aria-label="Open in new tab" onClick={() => window.open(src, "_blank", "noopener,noreferrer")}>
          <ExternalLink size={20} strokeWidth={1.8} />
        </button>
      </div>
      {(editError || image.correction) && (
        <div className={`asset-preview-note${editError ? " is-error" : ""}`} role={editError ? "alert" : "status"}>
          {editError || image.correction?.label}
        </div>
      )}
      <div className="asset-preview-viewport">
        <span
          className={`shot-frame${drawing ? " is-drawing" : ""}`}
          style={{ transform: zoom === 1 ? undefined : `scale(${zoom})`, ...(box ? { width: box.width, maxWidth: "none" } : {}) }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          data-testid="preview-frame"
        >
          <img
            ref={imgRef}
            src={src}
            alt={image.filename}
            className="asset-preview-img"
            style={box ? { width: box.width, height: box.height, minWidth: 0, minHeight: 0, maxWidth: "none", maxHeight: "none" } : undefined}
            draggable={false}
            onLoad={onImageLoad}
          />
          {full?.box && box && (
            <span
              className={`asset-preview-box${drawing ? " is-replaced" : ""}`}
              style={{ left: `${full.box.x * 100}%`, top: `${full.box.y * 100}%`, width: `${full.box.w * 100}%`, height: `${full.box.h * 100}%` }}
            />
          )}
          {drawing && draft && (
            <span
              className="asset-preview-box is-draft"
              aria-label="New box"
              style={{ left: `${draft.x * 100}%`, top: `${draft.y * 100}%`, width: `${draft.w * 100}%`, height: `${draft.h * 100}%` }}
            >
              {HANDLES.map((handle) => (
                <span key={handle} className={`asset-preview-handle is-${handle}`} data-handle={handle} aria-label={`Resize ${handle}`} />
              ))}
            </span>
          )}
          {image.confidence != null && <span className="shot-score">{Math.round(image.confidence * 100)}%</span>}
        </span>
      </div>
      <div className="asset-preview-controls">
        <button type="button" className="asset-preview-zoom" title="Zoom out" aria-label="Zoom out" onClick={() => setZoomAction("out")}>
          −
        </button>
        <button type="button" className="asset-preview-zoom" title="Fit to screen" aria-label="Fit to screen" onClick={() => setZoomAction("fit")}>
          <Search size={16} strokeWidth={2} />
        </button>
        <button type="button" className="asset-preview-zoom" title="Zoom in" aria-label="Zoom in" onClick={() => setZoomAction("in")}>
          +
        </button>
      </div>
    </div>,
    document.body,
  );
}
