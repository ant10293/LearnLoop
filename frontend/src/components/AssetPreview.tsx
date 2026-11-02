import { Download, ExternalLink, Search, X } from "lucide-react";
import { useEffect, useRef, useState, type MouseEvent, type SyntheticEvent } from "react";
import { createPortal } from "react-dom";
import { fittedPreviewSize } from "../lib/previewFit";
import { ShotFrame } from "./ShotFrame";

export type PreviewImage = { src: string; filename: string; confidence?: number | null };

/** Round previous/next control shared by the full-screen preview and the inspect frame. */
export function PreviewNav({
  direction,
  label,
  onClick,
}: {
  direction: "prev" | "next";
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`asset-preview-nav is-${direction}`}
      title={label}
      aria-label={label}
      onPointerDown={(event) => event.stopPropagation()}
      onClick={onClick}
    >
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <polyline points={direction === "prev" ? "15 18 9 12 15 6" : "9 6 15 12 9 18"} />
      </svg>
    </button>
  );
}

/**
 * Full-screen image preview from potheads-ceramics' admin image overlay:
 * zoom, previous/next, download, and an explicit new-tab action.
 */
export function AssetPreview({
  images,
  startIndex = 0,
  onClose,
}: {
  images: PreviewImage[];
  startIndex?: number;
  onClose: () => void;
}) {
  const [current, setCurrent] = useState(() =>
    Math.min(Math.max(0, startIndex), Math.max(0, images.length - 1)),
  );
  const [zoom, setZoom] = useState(1);
  const [box, setBox] = useState<{ width: number; height: number } | null>(null);
  const natural = useRef({ width: 0, height: 0 });
  const onCloseRef = useRef(onClose);
  const closeRef = useRef<HTMLButtonElement>(null);
  onCloseRef.current = onClose;
  const image = images[current];
  const fittedSrc = useRef(image?.src);
  if (image?.src !== fittedSrc.current) {
    fittedSrc.current = image?.src;
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
        onCloseRef.current();
        return;
      }
      if (images.length < 2) return;
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
    setCurrent((index) => (index + delta + images.length) % images.length);
  }

  function setZoomAction(action: "in" | "out" | "fit") {
    setZoom((value) => {
      if (action === "in") return Math.min(value * 1.3, 5);
      if (action === "out") return Math.max(value / 1.3, 0.2);
      return 1;
    });
  }

  function download() {
    const link = document.createElement("a");
    link.href = image.src;
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
      {images.length > 1 && (
        <>
          <PreviewNav direction="prev" label="Previous" onClick={() => go(-1)} />
          <PreviewNav direction="next" label="Next" onClick={() => go(1)} />
        </>
      )}
      <div className="asset-preview-actions">
        <button type="button" className="asset-preview-action" title="Download" aria-label="Download" onClick={download}>
          <Download size={20} strokeWidth={1.8} />
        </button>
        <button
          type="button"
          className="asset-preview-action"
          title="Open in new tab"
          aria-label="Open in new tab"
          onClick={() => window.open(image.src, "_blank", "noopener,noreferrer")}
        >
          <ExternalLink size={20} strokeWidth={1.8} />
        </button>
      </div>
      <div className="asset-preview-viewport">
        <ShotFrame
          src={image.src}
          alt={image.filename}
          confidence={image.confidence}
          imgClassName="asset-preview-img"
          imgStyle={box ? { width: box.width, height: box.height, maxWidth: "none", maxHeight: "none" } : undefined}
          draggable={false}
          onLoad={onImageLoad}
          frameStyle={{ transform: zoom === 1 ? undefined : `scale(${zoom})` }}
        />
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
