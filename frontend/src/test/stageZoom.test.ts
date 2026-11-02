import { describe, expect, it } from "vitest";
import { MAX_STAGE_ZOOM, clampStageView, panStage, stageViewAt, stageZoomByWheel, type StageView } from "../lib/stageZoom";

const frame = { width: 800, height: 450 };

function project(point: { x: number; y: number }, view: StageView, width: number, height: number) {
  const originX = width / 2;
  const originY = height / 2;
  return {
    x: originX + view.panX + view.zoom * (point.x - originX),
    y: originY + view.panY + view.zoom * (point.y - originY),
  };
}

function unproject(cursor: { x: number; y: number }, view: StageView, width: number, height: number) {
  const originX = width / 2;
  const originY = height / 2;
  return {
    x: originX + (cursor.x - originX - view.panX) / view.zoom,
    y: originY + (cursor.y - originY - view.panY) / view.zoom,
  };
}

describe("stageZoomByWheel", () => {
  it("keeps the point under the cursor fixed", () => {
    const before: StageView = { zoom: 1, panX: 0, panY: 0 };
    const cursor = { x: 640, y: 80 };
    const after = stageZoomByWheel(before, cursor.x, cursor.y, frame.width, frame.height, -120);
    const point = unproject(cursor, before, frame.width, frame.height);
    const placed = project(point, after, frame.width, frame.height);
    expect(after.zoom).toBeGreaterThan(before.zoom);
    expect(placed.x).toBeCloseTo(cursor.x);
    expect(placed.y).toBeCloseTo(cursor.y);
  });

  it("does not pan when the cursor is on the center", () => {
    const after = stageZoomByWheel({ zoom: 1, panX: 0, panY: 0 }, 400, 225, frame.width, frame.height, -100);
    expect(after.zoom).toBeGreaterThan(1);
    expect(after.panX).toBeCloseTo(0);
    expect(after.panY).toBeCloseTo(0);
  });

  it("shifts toward the side being zoomed", () => {
    const after = stageZoomByWheel({ zoom: 1, panX: 0, panY: 0 }, 700, 225, frame.width, frame.height, -100);
    expect(after.panX).toBeLessThan(0);
  });

  it("returns to a fitted frame when zoomed all the way out", () => {
    const after = stageZoomByWheel({ zoom: 2.4, panX: 80, panY: -40 }, 120, 300, frame.width, frame.height, 8000);
    expect(after).toEqual({ zoom: 1, panX: 0, panY: 0 });
  });

  it("treats a line delta the same direction as a pixel delta", () => {
    const after = stageZoomByWheel({ zoom: 1, panX: 0, panY: 0 }, 500, 100, frame.width, frame.height, -3, 1);
    expect(after.zoom).toBeGreaterThan(1);
    expect(after.panX).not.toBe(0);
  });

  it("leaves the view alone when the wheel does not move", () => {
    const before: StageView = { zoom: 1.5, panX: 12, panY: -4 };
    expect(stageZoomByWheel(before, 10, 10, frame.width, frame.height, 0)).toEqual(before);
  });
});

describe("stageViewAt", () => {
  it("stops at the maximum zoom and still anchors to the cursor", () => {
    const before: StageView = { zoom: 2, panX: 40, panY: -20 };
    const cursor = { x: 180, y: 360 };
    const after = stageViewAt(before, cursor.x, cursor.y, frame.width, frame.height, 12);
    const point = unproject(cursor, before, frame.width, frame.height);
    const placed = project(point, after, frame.width, frame.height);
    expect(after.zoom).toBe(MAX_STAGE_ZOOM);
    expect(placed.x).toBeCloseTo(cursor.x);
    expect(placed.y).toBeCloseTo(cursor.y);
  });
});

describe("letterboxed picture", () => {
  it("zooms toward a point on the picture without sliding it out of the frame", () => {
    const before: StageView = { zoom: 1, panX: 0, panY: 0 };
    const cursor = { x: 620, y: 90 };
    const after = stageZoomByWheel(before, cursor.x, cursor.y, 1000, 600, -120, 0, 400, 600);
    const point = unproject(cursor, before, 1000, 600);
    const placed = project(point, after, 1000, 600);
    const left = 1000 / 2 - (after.zoom * 400) / 2 + after.panX;
    const right = left + after.zoom * 400;
    expect(after.zoom).toBeGreaterThan(1);
    expect(placed.x).toBeCloseTo(cursor.x);
    expect(placed.y).toBeCloseTo(cursor.y);
    expect(left).toBeGreaterThanOrEqual(-0.01);
    expect(right).toBeLessThanOrEqual(1000.01);
  });
});

describe("panStage", () => {
  it("keeps the picture covering the frame", () => {
    expect(panStage({ zoom: 2, panX: 0, panY: 0 }, 5000, -5000, frame.width, frame.height)).toEqual({
      zoom: 2,
      panX: frame.width / 2,
      panY: -frame.height / 2,
    });
  });

  it("keeps a letterboxed picture on screen", () => {
    const panned = panStage({ zoom: 2, panX: 0, panY: 0 }, -5000, 5000, 1000, 600, 300, 600);
    expect(panned).toEqual({ zoom: 2, panX: -200, panY: 300 });
    const left = 1000 / 2 - (2 * 300) / 2 + panned.panX;
    const top = 600 / 2 - (2 * 600) / 2 + panned.panY;
    expect(left).toBe(0);
    expect(top).toBe(0);
  });

  it("drops pan when the frame is fitted", () => {
    expect(clampStageView({ zoom: 1, panX: 40, panY: 10 }, frame.width, frame.height)).toEqual({
      zoom: 1,
      panX: 0,
      panY: 0,
    });
  });
});
