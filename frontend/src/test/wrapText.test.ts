import { describe, expect, it } from "vitest";
import { labelSpan, wrapLines } from "../lib/wrapText";

const width = (value: string) => value.length;

describe("wrapLines", () => {
  it("keeps a short label on one line", () => {
    expect(wrapLines("Object 4: hex nut", 20, width)).toEqual(["Object 4: hex nut"]);
  });

  it("wraps on word boundaries once the line exceeds the column", () => {
    expect(wrapLines("Object 1: hex-head bolt (threaded fastener)", 24, width)).toEqual([
      "Object 1: hex-head bolt",
      "(threaded fastener)",
    ]);
  });

  it("leaves a word that is wider than the column intact", () => {
    expect(wrapLines("supercalifragilistic", 8, width)).toEqual(["supercalifragilistic"]);
  });

  it("returns nothing for blank text", () => {
    expect(wrapLines("   ", 10, width)).toEqual([]);
  });
});

describe("labelSpan", () => {
  const boxes = [
    { x: 691.2, width: 129.6 },
    { x: 842.4, width: 108 },
    { x: 993.6, width: 108 },
    { x: 1123.2, width: 86.4 },
  ];

  it("hangs past both edges of an open box and stops before the next one", () => {
    const span = labelSpan(boxes[0].x, boxes[0].width, boxes.slice(1));
    expect(span.x).toBeLessThan(boxes[0].x);
    expect(span.x + span.width).toBeGreaterThan(boxes[0].x + boxes[0].width);
    expect(span.x + span.width).toBeLessThan(boxes[1].x);
  });

  it("keeps a narrow box's label from running into the neighbor on its left", () => {
    const span = labelSpan(boxes[3].x, boxes[3].width, boxes.slice(0, 3));
    expect(span.x).toBeLessThan(boxes[3].x);
    expect(span.x).toBeGreaterThan(boxes[2].x + boxes[2].width);
    expect(span.x + span.width).toBeGreaterThan(boxes[3].x + boxes[3].width);
  });
});
