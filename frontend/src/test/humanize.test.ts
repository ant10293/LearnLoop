import { describe, expect, it } from "vitest";
import { actionPhrase, humanize, humanizeText, labeledId, nameObjects } from "../lib/humanize";

describe("humanize", () => {
  it("splits snake_case into a sentence-style label", () => {
    expect(humanize("scene_analysis")).toBe("Scene analysis");
    expect(humanize("visually_supported")).toBe("Visually supported");
    expect(humanize("high_detail")).toBe("High detail");
    expect(humanize("pick_up")).toBe("Pick up");
  });

  it("splits kebab-case and keeps a hyphen between numbers", () => {
    expect(humanize("claude-sonnet-4-5")).toBe("Claude sonnet 4-5");
    expect(humanize("gpt-4.1")).toBe("GPT 4.1");
    expect(humanize("constrained_head_mounted")).toBe("Constrained head-mounted");
  });

  it("splits camelCase and screaming snake case", () => {
    expect(humanize("researchMode")).toBe("Research mode");
    expect(humanize("sampledFrames")).toBe("Sampled frames");
    expect(humanize("DIRECTLY_OBSERVED")).toBe("Directly observed");
    expect(humanize("AI_PREDICTION")).toBe("AI prediction");
    expect(humanize("CV_DETECTION")).toBe("CV detection");
    expect(humanize("INFERRED")).toBe("Inferred");
    expect(humanize("manual_json")).toBe("Manual JSON");
    expect(humanize("openai_compatible")).toBe("OpenAI compatible");
  });

  it("reads local ids as words", () => {
    expect(humanize("Object_1")).toBe("Object 1");
    expect(humanize("track_1")).toBe("Track 1");
  });

  it("puts a colon between an object id and its name", () => {
    expect(labeledId("Object_1", "threaded bolt")).toBe("Object 1: threaded bolt");
    expect(labeledId("Object_2", "")).toBe("Object 2");
  });

  it("formats identifier tokens inside a title and leaves prose hyphenation", () => {
    expect(humanizeText("fixed_camera · claude-sonnet-4-5")).toBe("Fixed camera · Claude sonnet 4-5");
    expect(humanizeText("Unconstrained head-mounted — two places")).toBe("Unconstrained head-mounted — two places");
    expect(humanizeText("Step 4 precedes Step_5, but physical necessity is not established.")).toBe(
      "Step 4 precedes Step 5, but physical necessity is not established.",
    );
  });

  it("capitalizes a single word that needs no other splitting", () => {
    expect(humanize("completed")).toBe("Completed");
    expect(humanize("align")).toBe("Align");
    expect(humanize("standard")).toBe("Standard");
    expect(humanize("inferred")).toBe("Inferred");
    expect(humanize("openai")).toBe("OpenAI");
    expect(humanize("Completed")).toBe("Completed");
  });

  it("names objects inside a step sentence", () => {
    const labels = {
      Object_6: "lower light spoked disk",
      Object_7: "lower dark toothed ring",
      Object_4: "light irregular structural piece",
    };
    expect(nameObjects("Insert Object_6 into Object_7.", labels)).toBe(
      "Insert the lower light spoked disk into the lower dark toothed ring.",
    );
    expect(
      nameObjects("Object_6's separate visible extent decreases and its spoked face is centered within Object_7.", labels),
    ).toBe(
      "The lower light spoked disk's separate visible extent decreases and its spoked face is centered within the lower dark toothed ring.",
    );
    expect(nameObjects("Object_6 is visibly seated inside Object_7 by 13 seconds.", labels)).toBe(
      "The lower light spoked disk is visibly seated inside the lower dark toothed ring by 13 seconds.",
    );
    expect(nameObjects("Attach the Object_6/Object_7 combination to the structure", labels)).toBe(
      "Attach the lower light spoked disk and lower dark toothed ring combination to the structure",
    );
    expect(nameObjects("The rod's separate visible extent ends at Object_4.", labels)).toBe(
      "The rod's separate visible extent ends at the light irregular structural piece.",
    );
    expect(nameObjects("Pick up Object_3.", labels)).toBe("Pick up Object 3.");
    expect(nameObjects("Both pieces are held.", labels)).toBe("Both pieces are held.");
  });

  it("names the parts on an action so two lifts are not the same words", () => {
    const labels = new Map([
      ["Object_1", "Light-gray molded body"],
      ["Object_7", "Small clear round part"],
    ]);
    expect(actionPhrase({ action_type: "pick_up", object_local_ids: ["Object_1", "Object_7"] }, labels)).toBe(
      "Pick up · Light-gray molded body, Small clear round part",
    );
    expect(actionPhrase({ action_type: "rotate", object_local_ids: [] }, labels)).toBe("Rotate");
  });

  it("leaves prose, JSON, and uuids alone", () => {
    expect(humanize("lying on the work surface")).toBe("lying on the work surface");
    expect(humanize("Corrected label")).toBe("Corrected label");
    expect(humanize("threaded bolt")).toBe("threaded bolt");
    expect(humanize('{"scene_analysis":true}')).toBe('{"scene_analysis":true}');
    expect(humanize("43968c2c-595b-4f43-b303-e57028811d29")).toBe("43968c2c-595b-4f43-b303-e57028811d29");
    expect(humanize("")).toBe("");
  });
});
