import { describe, expect, it } from "vitest";
import { objectScene } from "../lib/sceneSlice";
import type { Bundle } from "../lib/types";

const bundle = {
  states: [
    { id: "s1", object_local_id: "Object_1", time_start: 0, time_end: 2, state_label: "down", description: "", evidence_class: "DIRECTLY_OBSERVED", provenance: "HUMAN_ANNOTATION" },
    { id: "s2", object_local_id: "Object_2", time_start: 0, time_end: 2, state_label: "flat", description: "", evidence_class: "DIRECTLY_OBSERVED", provenance: "HUMAN_ANNOTATION" },
  ],
  transitions: [
    { id: "t1", object_local_id: "Object_1", timestamp: 2, from_state: "down", to_state: "up", description: "", evidence_class: "DIRECTLY_OBSERVED", certainty: 0.8, provenance: "HUMAN_ANNOTATION" },
    { id: "t2", object_local_id: "Object_2", timestamp: 4, from_state: "flat", to_state: "stacked", description: "", evidence_class: "DIRECTLY_OBSERVED", certainty: 0.8, provenance: "HUMAN_ANNOTATION" },
  ],
  relationships: [
    { id: "r1", subject_local_id: "Object_2", other_local_id: "Object_1", kind: "around", time_start: 5, time_end: 8, description: "ring around shaft", evidence_class: "DIRECTLY_OBSERVED", provenance: "HUMAN_ANNOTATION" },
    { id: "r2", subject_local_id: "Object_4", other_local_id: "Object_1", kind: "touching", time_start: 7, time_end: 8, description: "nut touches end", evidence_class: "DIRECTLY_OBSERVED", provenance: "HUMAN_ANNOTATION" },
  ],
  actions: [
    { local_id: "Action_1", object_local_ids: ["Object_1"], start: 0, end: 1 },
  ],
} as Pick<Bundle, "states" | "transitions" | "relationships" | "actions">;

describe("objectScene", () => {
  it("shows one object's transitions and relationships", () => {
    const first = objectScene({ observations: [{ object_local_id: "Object_1" }, { object_local_id: "Object_2" }] }, bundle, "Object_1", 0);
    const second = objectScene(undefined, bundle, "Object_2", 0);
    expect(first.transitions.map((item) => item.id)).toEqual(["t1"]);
    expect(second.transitions.map((item) => item.id)).toEqual(["t2"]);
    expect(first.relationships.map((item) => item.id)).toEqual(["r1", "r2"]);
    expect(second.relationships.map((item) => item.id)).toEqual(["r1"]);
    expect(first.states.map((item) => item.id)).toEqual(["s1"]);
    expect(second.states.map((item) => item.id)).toEqual(["s2"]);
    expect(first.observations).toEqual([{ object_local_id: "Object_1" }]);
    expect(JSON.stringify(first)).not.toBe(JSON.stringify(second));
  });
});
