import type { Bundle } from "./types";

type ScenePayload = {
  actions?: Bundle["actions"];
  observations?: { object_local_id?: string }[];
};

/** Scene JSON for the object selected in the states tab.

 *  Transitions and relationships come from that object, so choosing another
 *  object changes the payload. States, actions, and observations stay at `time`.
 */
export function objectScene(
  scene: ScenePayload | Record<string, unknown> | undefined,
  bundle: Pick<Bundle, "states" | "transitions" | "relationships" | "actions">,
  localId: string,
  time: number,
) {
  const covers = (start: number, end: number) => start - 1e-6 <= time && time <= end + 1e-6;
  const data = (scene ?? {}) as ScenePayload;
  return {
    t: Number(time.toFixed(2)),
    object_local_id: localId,
    states: bundle.states.filter((item) => item.object_local_id === localId && covers(item.time_start, item.time_end)),
    transitions: bundle.transitions.filter((item) => item.object_local_id === localId),
    relationships: bundle.relationships.filter(
      (item) => item.subject_local_id === localId || item.other_local_id === localId,
    ),
    actions: (data.actions ?? bundle.actions).filter(
      (item) => item.object_local_ids.includes(localId) && covers(item.start, item.end),
    ),
    observations: (data.observations ?? []).filter((item) => item.object_local_id === localId),
  };
}
