import { create } from "zustand";
import { actionCoversTime } from "../lib/analysis";

export type ActionSpan = { start: number; end: number };

interface InspectorState {
  time: number;
  playing: boolean;
  speed: number;
  overlay: boolean;
  layers: {
    tracks: boolean;
    trajectory: boolean;
    uncertain: boolean;
    confidence: boolean;
    conflicts: boolean;
    drops: boolean;
  };
  seedMode: boolean;
  selectedObject: string | null;
  selectedAction: string | null;
  actionSpan: ActionSpan | null;
  selectedStep: string | null;
  loop: { start: number; end: number } | null;
  setTime: (time: number) => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (speed: number) => void;
  setOverlay: (overlay: boolean) => void;
  setLayer: (key: keyof InspectorState["layers"], value: boolean) => void;
  setSeedMode: (seedMode: boolean) => void;
  selectObject: (id: string | null) => void;
  selectAction: (id: string | null, span?: ActionSpan) => void;
  selectStep: (id: string | null) => void;
  setLoop: (loop: { start: number; end: number } | null) => void;
}

export const useInspector = create<InspectorState>((set) => ({
  time: 0,
  playing: false,
  speed: 1,
  overlay: true,
  layers: {
    tracks: false,
    trajectory: false,
    uncertain: false,
    confidence: true,
    conflicts: false,
    drops: false,
  },
  seedMode: false,
  selectedObject: null,
  selectedAction: null,
  actionSpan: null,
  selectedStep: null,
  loop: null,
  setTime: (time) => set((state) => {
    const span = state.actionSpan;
    if (!state.selectedAction || !span || actionCoversTime(span, time)) return { time };
    const loop = state.loop;
    const loopWasThisAction = loop != null && Math.abs(loop.start - span.start) < 0.001 && Math.abs(loop.end - span.end) < 0.001;
    return {
      time,
      selectedAction: null,
      actionSpan: null,
      ...(loopWasThisAction ? { loop: null } : {}),
    };
  }),
  setPlaying: (playing) => set({ playing }),
  setSpeed: (speed) => set({ speed }),
  setOverlay: (overlay) => set({ overlay }),
  setLayer: (key, value) => set((state) => ({ layers: { ...state.layers, [key]: value } })),
  setSeedMode: (seedMode) => set({ seedMode }),
  selectObject: (selectedObject) => set({ selectedObject }),
  selectAction: (selectedAction, span) => set((state) => {
    if (!selectedAction) return { selectedAction: null, actionSpan: null };
    const actionSpan = span ?? null;
    if (actionSpan && !actionCoversTime(actionSpan, state.time)) return { selectedAction: null, actionSpan: null };
    return { selectedAction, actionSpan };
  }),
  selectStep: (selectedStep) => set({ selectedStep }),
  setLoop: (loop) => set({ loop }),
}));
