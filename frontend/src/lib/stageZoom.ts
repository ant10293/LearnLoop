/** Inspect-stage zoom. `translate(pan) scale(zoom)` uses the frame center as its origin. */

export type StageView = {
  zoom: number;
  panX: number;
  panY: number;
};

export const MIN_STAGE_ZOOM = 1;
export const MAX_STAGE_ZOOM = 4;

const WHEEL_PIXELS = 0.0016;

export function clampStageView(
  view: StageView,
  frameWidth: number,
  frameHeight: number,
  pictureWidth = frameWidth,
  pictureHeight = frameHeight,
): StageView {
  const zoom = Math.min(MAX_STAGE_ZOOM, Math.max(MIN_STAGE_ZOOM, view.zoom));
  if (zoom <= MIN_STAGE_ZOOM) return { zoom: MIN_STAGE_ZOOM, panX: 0, panY: 0 };
  if (frameWidth <= 0 || frameHeight <= 0 || pictureWidth <= 0 || pictureHeight <= 0) return { ...view, zoom };
  // Larger than the frame: stop at the last position that still covers it.
  // Smaller than the frame: stop when the picture touches the frame edge.
  const limitX = Math.abs(zoom * pictureWidth - frameWidth) / 2;
  const limitY = Math.abs(zoom * pictureHeight - frameHeight) / 2;
  return {
    zoom,
    panX: Math.min(limitX, Math.max(-limitX, view.panX)),
    panY: Math.min(limitY, Math.max(-limitY, view.panY)),
  };
}

/** Zoom toward a point in frame coordinates, keeping that point fixed on screen. */
export function stageViewAt(
  view: StageView,
  cursorX: number,
  cursorY: number,
  width: number,
  height: number,
  nextZoom: number,
  pictureWidth = width,
  pictureHeight = height,
): StageView {
  if (width <= 0 || height <= 0) return view;
  const zoom = Math.min(MAX_STAGE_ZOOM, Math.max(MIN_STAGE_ZOOM, nextZoom));
  if (zoom <= MIN_STAGE_ZOOM) return { zoom: MIN_STAGE_ZOOM, panX: 0, panY: 0 };
  const currentZoom = view.zoom > 0 ? view.zoom : MIN_STAGE_ZOOM;
  const originX = width / 2;
  const originY = height / 2;
  const scale = zoom / currentZoom;
  return clampStageView(
    {
      zoom,
      panX: cursorX - originX - scale * (cursorX - originX - view.panX),
      panY: cursorY - originY - scale * (cursorY - originY - view.panY),
    },
    width,
    height,
    pictureWidth,
    pictureHeight,
  );
}

export function stageZoomByWheel(
  view: StageView,
  cursorX: number,
  cursorY: number,
  width: number,
  height: number,
  deltaY: number,
  deltaMode = 0,
  pictureWidth = width,
  pictureHeight = height,
): StageView {
  if (deltaY === 0 || width <= 0 || height <= 0) return view;
  const pixels = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * height : deltaY;
  return stageViewAt(view, cursorX, cursorY, width, height, view.zoom * Math.exp(-pixels * WHEEL_PIXELS), pictureWidth, pictureHeight);
}

export function panStage(
  view: StageView,
  panX: number,
  panY: number,
  width: number,
  height: number,
  pictureWidth = width,
  pictureHeight = height,
): StageView {
  return clampStageView({ zoom: view.zoom, panX, panY }, width, height, pictureWidth, pictureHeight);
}
