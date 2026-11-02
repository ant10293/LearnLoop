import { describe, expect, it } from "vitest";
import { formatLogText } from "../lib/formatLog";

describe("formatLogText", () => {
  it("breaks a one-line JSON payload into indented lines", () => {
    const formatted = formatLogText('{"objects":[{"label":"bolt"}]}');
    expect(formatted.split("\n").length).toBeGreaterThan(1);
    expect(formatted).toContain('  "objects"');
    expect(formatted).toContain('    "label": "bolt"');
  });

  it("pretty-prints JSON wrapped in a markdown fence", () => {
    const formatted = formatLogText('```json\n{"ok":true}\n```');
    expect(formatted).toBe('{\n  "ok": true\n}');
  });

  it("leaves plain error text unchanged", () => {
    expect(formatLogText("provider timed out")).toBe("provider timed out");
  });
});
