import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DecisionPanel } from "../components/DecisionPanel";
import { decisionsNear } from "../lib/analysis";
import type { BoxDecision, Bundle } from "../lib/types";

const row = (timestamp: number | null, object = "Object_1"): BoxDecision => ({
  step: "identity_reconcile",
  reason_code: "extent_growth_exceeded",
  outcome: "dropped",
  object_local_id: object,
  timestamp,
  bbox: { x: 0.1, y: 0.1, w: 0.2, h: 0.2 },
  confidence: 0.4,
  other_object_id: null,
  detail: "6.0 times this part's established area.",
});

describe("box decisions", () => {
  it("finds rows near the playhead, nearest first, and skips rows with no time", () => {
    const rows = [row(1.3, "far"), row(1.05, "near"), row(4), row(null)];
    expect(decisionsNear(rows, 1).map((item) => item.object_local_id)).toEqual(["near", "far"]);
  });

  it("renders nothing for a run with no records", () => {
    const { container } = render(<DecisionPanel bundle={{} as Bundle} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows counts and the recorded reason", () => {
    const bundle = {
      box_decisions: {
        total: 1,
        truncated: false,
        counts: [{ step: "identity_reconcile", reason_code: "extent_growth_exceeded", outcome: "dropped", count: 3 }],
        rows: [row(0)],
      },
    } as unknown as Bundle;
    render(<DecisionPanel bundle={bundle} />);
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText(/6\.0 times this part's established area/)).toBeInTheDocument();
  });
});
