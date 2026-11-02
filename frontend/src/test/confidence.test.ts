import { describe, expect, it } from "vitest";
import { formatConfidence } from "../lib/confidence";

describe("observation confidence", () => {
  it("shows a model score as a percent", () => {
    expect(formatConfidence(0.42)).toBe("42%");
    expect(formatConfidence(1)).toBe("100%");
    expect(formatConfidence(0)).toBe("0%");
  });
});
