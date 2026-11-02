import { useCallback, useEffect, useRef, useState, type PointerEvent, type RefObject } from "react";
import { clampStageView, MAX_STAGE_ZOOM, MIN_STAGE_ZOOM, panStage, stageViewAt, stageZoomByWheel, type StageView } from "../lib/stageZoom";
import { ZoomStepper } from "./ZoomStepper";

const STAGE_ZOOM_STEP = 0.25;

/** Scroll, step, and drag-to-pan for a `.stage-frame`. `picture` is the letterboxed image size inside it. */
export function useStageZoom(frameRef: RefObject<HTMLDivElement | null>, picture: { w: number; h: number }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const viewRef = useRef<StageView>({ zoom: 1, panX: 0, panY: 0 });
  const pictureRef = useRef(picture);
  pictureRef.current = picture;
  const dragRef = useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number } | null>(null);
  const commitView = useCallback((next: StageView) => {
    viewRef.current = next;
    setZoom(next.zoom);
    setPan({ x: next.panX, y: next.panY });
  }, []);

  function pictureSize(frameWidth: number, frameHeight: number) {
    const current = pictureRef.current;
    return {
      width: current.w > 1 ? current.w : frameWidth,
      height: current.h > 1 ? current.h : frameHeight,
    };
  }

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = frame.getBoundingClientRect();
      const size = pictureSize(rect.width, rect.height);
      commitView(stageZoomByWheel(
        viewRef.current,
        event.clientX - rect.left,
        event.clientY - rect.top,
        rect.width,
        rect.height,
        event.deltaY,
        event.deltaMode,
        size.width,
        size.height,
      ));
    };
    const observer = new ResizeObserver(() => {
      const rect = frame.getBoundingClientRect();
      const size = pictureSize(rect.width, rect.height);
      const next = clampStageView(
        viewRef.current,
        rect.width,
        rect.height,
        size.width,
        size.height,
      );
      const current = viewRef.current;
      if (next.zoom !== current.zoom || next.panX !== current.panX || next.panY !== current.panY) commitView(next);
    });
    frame.addEventListener("wheel", onWheel, { passive: false });
    observer.observe(frame);
    return () => {
      frame.removeEventListener("wheel", onWheel);
      observer.disconnect();
    };
  }, [commitView, frameRef]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame || picture.w <= 1 || picture.h <= 1) return;
    const rect = frame.getBoundingClientRect();
    const next = clampStageView(viewRef.current, rect.width, rect.height, picture.w, picture.h);
    const current = viewRef.current;
    if (next.zoom !== current.zoom || next.panX !== current.panX || next.panY !== current.panY) commitView(next);
  }, [picture.w, picture.h, commitView, frameRef]);

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || viewRef.current.zoom <= 1) return;
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      panX: viewRef.current.panX,
      panY: viewRef.current.panY,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setPanning(true);
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const size = pictureSize(rect.width, rect.height);
    commitView(panStage(
      viewRef.current,
      drag.panX + event.clientX - drag.x,
      drag.panY + event.clientY - drag.y,
      rect.width,
      rect.height,
      size.width,
      size.height,
    ));
  }

  function endPan(event: PointerEvent<HTMLDivElement>) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setPanning(false);
  }

  function stepZoom(direction: number) {
    const frame = frameRef.current;
    if (!frame) return;
    const rect = frame.getBoundingClientRect();
    const size = pictureSize(rect.width, rect.height);
    const nextZoom = Math.min(MAX_STAGE_ZOOM, Math.max(MIN_STAGE_ZOOM, viewRef.current.zoom + direction * STAGE_ZOOM_STEP));
    commitView(stageViewAt(
      viewRef.current,
      rect.width / 2,
      rect.height / 2,
      rect.width,
      rect.height,
      nextZoom,
      size.width,
      size.height,
    ));
  }

  return {
    zoom,
    pan,
    panning,
    commitView,
    stepZoom,
    frameClass: `${zoom > 1 ? " is-zoomed" : ""}${panning ? " is-panning" : ""}`,
    frameHandlers: { onPointerDown, onPointerMove, onPointerUp: endPan, onPointerCancel: endPan },
    frameTransform: { transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "center" },
  };
}

/** The − / + pinned to the frame's bottom-right corner. */
export function StageZoomControl({ zoom, onStep }: { zoom: number; onStep: (direction: number) => void }) {
  return (
    <div
      className="stage-zoom"
      onPointerDown={(event) => event.stopPropagation()}
      onWheel={(event) => event.stopPropagation()}
    >
      <ZoomStepper
        canZoomOut={zoom > MIN_STAGE_ZOOM}
        canZoomIn={zoom < MAX_STAGE_ZOOM}
        onZoomOut={() => onStep(-1)}
        onZoomIn={() => onStep(1)}
      />
    </div>
  );
}
