import { describe, expect, it } from "vitest";
import { formatDuration } from "../lib/api";

describe("formatDuration", () => {
  it("rounds to the nearest second and skips units that are zero", () => {
    expect(formatDuration(4198.85)).toBe("1h 9m 59s");
    expect(formatDuration(9.89)).toBe("10s");
    expect(formatDuration(59.6)).toBe("1m");
    expect(formatDuration(3600)).toBe("1h");
    expect(formatDuration(3630)).toBe("1h 30s");
    expect(formatDuration(0)).toBe("0s");
  });

  it("leaves an unknown duration blank", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(undefined)).toBe("—");
  });
});
