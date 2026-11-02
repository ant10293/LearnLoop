import { describe, expect, it } from "vitest";
import { captureToLocalInput, localInputToCapture } from "../lib/captureTime";

function localFrom(instant: string): string {
  const date = new Date(instant);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

describe("capture time field", () => {
  it("shows an aware timestamp in local time", () => {
    expect(captureToLocalInput("2026-09-18T19:37:23.000000Z", "2026-09-18T19:37:23+00:00")).toBe(localFrom("2026-09-18T19:37:23Z"));
  });

  it("reads an offset written without a colon", () => {
    expect(captureToLocalInput("2026-10-21T18:39:25-0500", "")).toBe(localFrom("2026-10-21T18:39:25-05:00"));
  });

  it("keeps a timestamp with no timezone as wall-clock time", () => {
    expect(captureToLocalInput("2026-03-01T10:00:00", "")).toBe("2026-03-01T10:00:00");
    expect(captureToLocalInput("2026-03-01 10:00:00", "")).toBe("2026-03-01T10:00:00");
  });

  it("stores a picker value with the local offset", () => {
    const saved = localInputToCapture("2026-09-18T14:37:23");
    const date = new Date("2026-09-18T14:37:23");
    const offset = -date.getTimezoneOffset();
    const sign = offset >= 0 ? "+" : "-";
    const abs = Math.abs(offset);
    const pad = (value: number) => String(value).padStart(2, "0");
    expect(saved).toBe(`2026-09-18T14:37:23${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`);
    expect(new Date(saved).getTime()).toBe(date.getTime());
  });
});
