import { afterEach, describe, expect, it } from "vitest";
import { useInspector } from "../stores/ui";

const span = { start: 6, end: 18 };

afterEach(() => {
  useInspector.setState({
    time: 0,
    playing: false,
    selectedAction: null,
    actionSpan: null,
    loop: null,
  });
});

describe("action selection follows the frame", () => {
  it("keeps the action while the playhead stays inside it", () => {
    const inspector = useInspector.getState();
    inspector.setTime(6);
    inspector.selectAction("attach", span);
    inspector.setTime(12);
    expect(useInspector.getState().selectedAction).toBe("attach");
  });

  it("clears the action and its loop when the frame leaves the range", () => {
    const inspector = useInspector.getState();
    inspector.setTime(6);
    inspector.selectAction("attach", span);
    inspector.setLoop(span);
    inspector.setTime(19);
    const state = useInspector.getState();
    expect(state.selectedAction).toBeNull();
    expect(state.actionSpan).toBeNull();
    expect(state.loop).toBeNull();
    expect(state.time).toBe(19);
  });

  it("leaves a wider loop alone when the action ends", () => {
    const inspector = useInspector.getState();
    inspector.setTime(6);
    inspector.selectAction("attach", span);
    inspector.setLoop({ start: 0, end: 30 });
    inspector.setTime(20);
    const state = useInspector.getState();
    expect(state.selectedAction).toBeNull();
    expect(state.loop).toEqual({ start: 0, end: 30 });
  });

  it("does not select an action the current frame misses", () => {
    useInspector.getState().setTime(2);
    useInspector.getState().selectAction("attach", span);
    expect(useInspector.getState().selectedAction).toBeNull();
  });
});
