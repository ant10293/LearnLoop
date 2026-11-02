import { describe, expect, it } from "vitest";
import { clipPosition } from "../lib/taskOrder";

describe("partial clip order", () => {
  it("marks a missing timestamp as ambiguous and a tie as unordered inside the group", () => {
    const order = {
      ranked: [["early"], ["tied-a", "tied-b"]],
      unordered: ["loose"],
      ambiguous_ids: ["tied-a", "tied-b", "loose"],
    };
    expect(clipPosition(order, "early")).toEqual({ placed: true, tied: false, ambiguous: false });
    expect(clipPosition(order, "tied-a").tied).toBe(true);
    expect(clipPosition(order, "tied-a").ambiguous).toBe(true);
    expect(clipPosition(order, "loose")).toEqual({ placed: false, tied: false, ambiguous: true });
  });
});
