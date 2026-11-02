import { describe, expect, it } from "vitest";
import { droppedPath, pathsForDrop } from "../lib/droppedPath";

function transfer(data: Record<string, string>) {
  return { getData: (type: string) => data[type] ?? "" };
}

describe("dropped file paths", () => {
  it("reads an absolute path the browser attached to the file", () => {
    const file = new File(["x"], "clip.mov");
    Object.defineProperty(file, "path", { value: "/Users/a/Videos/clip.mov" });
    expect(droppedPath(file)).toBe("/Users/a/Videos/clip.mov");
    expect(pathsForDrop([file], transfer({ "text/uri-list": "file:///elsewhere/clip.mov" }))).toEqual([
      "/Users/a/Videos/clip.mov",
    ]);
  });

  it("reads file urls from the drop", () => {
    const file = new File(["x"], "My clip.mov");
    expect(pathsForDrop([file], transfer({ "text/uri-list": "file:///Users/a/Videos/My%20clip.mov" }))).toEqual([
      "/Users/a/Videos/My clip.mov",
    ]);
  });

  it("reads a windows file url", () => {
    const file = new File(["x"], "clip.mov");
    expect(pathsForDrop([file], transfer({ "text/plain": "file:///C:/Users/a/clip.mov" }))).toEqual([
      "C:/Users/a/clip.mov",
    ]);
  });

  it("leaves the path blank when the drop does not include one", () => {
    const file = new File(["x"], "clip.mov");
    expect(droppedPath(file)).toBe("");
    expect(pathsForDrop([file], transfer({}))).toEqual([""]);
  });
});
