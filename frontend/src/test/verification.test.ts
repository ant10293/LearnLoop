import { describe, expect, it } from "vitest";
import { markerColor, markerKind, recordingsForVerification, verdictClass } from "../lib/verification";

describe("verification display", () => {
  it("keeps pass, fail, and unverified visually distinct", () => {
    expect(verdictClass("PASS")).toBe("observed");
    expect(verdictClass("FAIL")).toBe("unknown");
    expect(verdictClass("CANNOT_VERIFY")).toBe("inferred");
    expect(verdictClass("INCONCLUSIVE")).toBe("inferred");
  });

  it("marks a corrected pass separately from a failure", () => {
    expect(markerKind({ verdict: "PASS", execution_state: "corrected" })).toBe("corrected");
    expect(markerKind({ verdict: "FAIL", error_category: "MISSING_COMPONENT" })).toBe("failure");
    expect(markerKind({ verdict: "CANNOT_VERIFY" })).toBe("uncertain");
    expect(markerKind({ verdict: "PASS" })).toBe("verified");
    expect(markerColor("failure")).not.toBe(markerColor("uncertain"));
  });

  it("keeps an unanalyzed video and prefers the newest completed run", () => {
    const recordings = recordingsForVerification(
      [
        { id: "video-a", filename: "a.mp4", duration_sec: 12, playback_url: "/a" },
        { id: "video-b", filename: "b.mp4", duration_sec: 4, playback_url: "/b" },
      ],
      [
        { id: "run-new", video_id: "video-a" },
        { id: "run-old", video_id: "video-a" },
      ],
    );
    expect(recordings.map((item) => item.run_id)).toEqual(["run-new", ""]);
    expect(recordings[1].filename).toBe("b.mp4");
  });
});
