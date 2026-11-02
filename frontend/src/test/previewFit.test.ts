import { describe, expect, it } from "vitest";
import { fittedPreviewSize } from "../lib/previewFit";

describe("expanded image size", () => {
  it("enlarges a small crop to the preview without changing its shape", () => {
    const size = fittedPreviewSize(200, 400, 1400, 900);
    expect(size).toEqual({ width: 382.5, height: 765 });
  });

  it("shrinks a large picture so it still fits", () => {
    const size = fittedPreviewSize(4000, 2000, 1000, 800);
    expect(size).toEqual({ width: 900, height: 450 });
  });
});
