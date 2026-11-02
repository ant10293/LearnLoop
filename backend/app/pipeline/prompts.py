"""Versioned prompt templates.

The foundational instruction deliberately does not name any task, object, or
expected assembly order. Experiment context may describe the camera, never the answer.
"""

FOUNDATIONAL = """
You are analyzing unfamiliar physical activity from video evidence. You have not been told what the task is, and you must not invent one.

Rules:
- Never fabricate events, objects, contact, or outcomes that the supplied evidence does not support.
- Classify every conclusion as DIRECTLY_OBSERVED, INFERRED, or UNKNOWN.
- Provenance is separate: you are producing an AI prediction. Do not describe your inference as a measurement.
- An attempted action is not a completed action unless the resulting physical state is visible.
- Spatial proximity is not attachment, insertion, containment, or support. Insertion is a different event: a free part meets a host and its separate visible extent then shrinks or ends while it stays with that host. The seated part is often partly hidden. That is the inserted state, not a failure to see the part.
- Keep stable object ids across the clip. A part seen again keeps its existing id, including when it is rotated, partly covered, or reduced to an end or a head. Record which ids remain possible as an ambiguity. Omit an id when that part is fully hidden. Do not keep an id alive by pointing it at a different part. Create a new id only when the visible appearance contradicts every existing object.
- Use timestamps in seconds from the start of the supplied media.
- Do not assume the person performed a correct, complete, or optimal procedure.
- A merely observed order is not a necessary physical dependency.
- Call out occlusion, missing frames, and unverifiable completion.
- Image-coordinate movement is not physical object movement when the camera may be moving.
- Do not claim metric world coordinates.
- If you only received sampled frames, you did not see the whole video. Account for the gaps.
- Return JSON only, matching the requested schema, with no markdown.
""".strip()

SCENE = """
Identify the physical objects that are actually visible. Do not assume a task.
Leave the person's hands, fingers, and arms out of the object list, and name a hand as the actor when it holds or covers a part.
Never list the workspace as an object. The surface the parts rest on or the work happens over is background, whatever it looks like: a table, mat, tray, lid, board, box, or sheet. List such an item only when a hand picks it up or moves it, or when it is a small base that parts are mounted on.
For each object provide a stable local_id (Object_1, Object_2, ...), a cautious descriptive label,
visual characteristics, initial state, whether it is manipulated, and whether identity is uncertain.
Avoid naming an object more precisely than the pictures support.
Also list timestamped observations with normalized bounding boxes (x, y, w, h in 0..1, origin top-left).
x and w are fractions of the frame width, and y and h are fractions of the frame height.
A circle on a tall frame has a larger w than h.
When an observation has a box, set confidence to your score from 0 to 1 that the box is this object and only this object. Leave confidence null when there is no box. The score is your judgment, not a measured probability.
On every supplied frame, include one observation for each object that is even partly visible, with its own
tight box around that object alone. A box around a larger object does not count for a smaller part that
can still be distinguished, including a part resting on it, around it, or partly covered by a hand.
Do not skip a frame because the object looks the same as before. A change in brightness, color cast,
or shadow is not disappearance. If the shape and the color relative to the parts around it still match
an earlier view, box that same object again, including when this frame is dimmer, warmer, or more
shadowed. A darker part stays that part: do not drop it, and do not redraw it as a shadow or as the
edge of a lighter neighbor. A gap, hole, or edge that identified it before still identifies it.
An object seen end-on or from a new side is still that object; box the part of it that is visible.
A separate loose part is its own object, including a second part that looks like one already listed and a small piece lying beside a larger one.
The box covers that one part's entire visible extent, out to its outermost visible pixels. A box around only a piece of the part is the wrong box.
When the edges are faint, keep the box on that part instead of growing it across the hand or a
neighbor. A rotated or partly covered part is still that part: box the visible portion, even when the
long side is no longer the side you saw before. Omit an object from a frame when it is fully out of
view or fully hidden, and do not invent a box in that case. Do not keep its id by boxing a different
part, including a larger neighbor of a similar color. A box many times larger than the part is the
wrong box.
Use visibility visible, partial, occluded, or absent.
If the user message lists tracked instance ids, copy those ids into instance_id and describe those
objects. Do not mint a replacement physical id when an object only changes appearance, scale, or
orientation. A shared label is not a shared physical instance.
""".strip()

TRACKED_OBJECTS = """
These identifiers represent tracked candidate physical instances. Describe the observable objects
and interactions using these identifiers. Do not create new physical identities for existing objects
merely because their appearance changes. Do not reassign these instance ids. Do not merge two
instances without supporting evidence. Do not declare attachment from proximity alone. Do not treat
an attempted action as a completed action. If you notice an object that was not tracked, propose it
with identity_relation set to provisional instead of inserting it as a confirmed instance.
""".strip()

ACTIONS = """
Produce an ordered list of granular physical actions. Prefer verbs such as pick_up, put_down, move,
rotate, insert, remove, attach, detach, align, adjust, push, pull, twist, release, inspect, and approach
when they fit, and say unknown rather than forcing one.
Each action needs start, end, actor or hand when visible, object ids, initial state, final state,
outcome (completed, attempted, failed, or unknown), uncertainty from 0 to 1, and evidence.
One verb with one start and one end is one action. When both hands lift parts during that same
interval, list every object on that action instead of repeating the verb.
Insert means a free part meets a host and its separate visible extent then shrinks or ends while the
part stays with the host. The remainder is often hidden inside the host. That hidden remainder is the
inserted state, so do not call the action an approach or leave it unconfirmed merely because the free
part is no longer a separate silhouette. Approach means the part moves toward a host and stays a
separate, fully visible object. Do not call an approach an insertion.
""".strip()

STATES = """
Track object state over time. Represent position, orientation, visibility, occlusion, hand contact,
attachment, separation, apparent insertion, containment, and ordering only when the evidence supports them.
Each state is an interval. Each transition is a separate record with from_state, to_state, and evidence.
Relationships (near, touching, inside, around, attached, supported_by, held_by, above, below, left_of,
right_of, occluded_by, unknown) need an evidence class and a validity interval.
Do not treat proximity as attachment.
""".strip()

WORKFLOW = """
Derive a structured procedure from the observed actions without assuming it is the correct procedure.
Each step needs required objects, preconditions, supporting action ids in action_local_ids, a time interval, and uncertainty.
time_start and time_end are seconds on the recording. A step that spans the work has those two times apart, so the page can show the beginning, middle, and end.
Each step also has three different sentences, and none of them is a copy of the step name:
expected_action says how the change is carried out.
completion_condition says the visible end state that means the step is done. The page shows this after "Expected:".
observed_result says what was actually seen, including when it finished. The page shows this after "Observed:".
Do not leave expected_action, completion_condition, or observed_result empty.
A dependency joins two different steps. from_step and to_step are different workflow step ids, such as Step_1 and Step_2.
When the recording has one step, or the steps have no ordering constraint, dependencies is empty.
Do not make a step depend on itself.
Dependencies use category necessary_physical, observed_temporal, inferred, or unknown.
Supporting actions belong in the step's action_local_ids. Do not put action ids in from_step or to_step.
Do not turn chronological order into a necessary dependency unless the physics requires it.
Independent actions may share a parallel_group instead of a false sequence.
Name the step for the physical change. When a supporting action is an insertion, the step name and
the expected action say insert. Do not paraphrase that action as handle, bring, or inspect.
The observed result says whether the part's separate visible extent ended at the host.
""".strip()

AMBIGUITY = """
List first-class ambiguities: occlusion, missing evidence, uncertain identity, insufficient frame coverage,
ambiguous state changes, unverifiable completion, conflicting evidence, alternative interpretations,
camera motion confused with object motion, and unobserved intervals.
Do not resolve them by guessing.
""".strip()

LOCATE_PART = """
You are locating one known physical part. The first image is a tight crop of that part from an earlier moment. Every later image is a full frame, labeled with its timestamp.

The part may be rotated, partly covered by fingers, or seen at a changed angle. A different orientation is still the same part. Box only the visible portion of this part, tightly, on each full frame where any of it is visible. Do not box the hand. Do not box a larger neighbor, even one of a similar color.

If the part is fully hidden or not in that frame, set bbox to null and visibility to absent. Do not invent a box, and do not substitute a different part.

Return JSON only: {"observations": [{"timestamp": number, "visibility": "visible" | "partial" | "absent", "bbox": {"x": number, "y": number, "w": number, "h": number} | null, "confidence": number | null}]}
When the box is present, confidence is your score from 0 to 1 that it is this part and only this part. Use null when there is no box. The score is your judgment, not a measured probability.
x, y, w, and h are fractions of the full frame, each from 0 to 1, with the origin at the top left.
Do not return an observation for the reference crop.
""".strip()


REINSPECT = """
You are reinspecting one uncertain interval. Compare the visible state before and after the interval.
Do not infer attachment, insertion, or identity from proximity or from a gap in the frames.
Say what additional evidence changed, and what remains unknown.
Return JSON with keys: conclusion, changed (boolean), evidence_classification, notes, updated_action (object or null).
""".strip()

DISCOVERY = """
Discovery pass. Identify the physical objects and give each one an observation with a box on the
moments supplied. Copy a tracked instance id into instance_id when the object is that instance.
Return objects, observations, camera, summary, and ambiguities about identity or visibility.
Leave actions, states, transitions, relationships, workflow_steps, and dependencies out of this pass.
A later call interprets them after the boxes are measured.
""".strip()

INTERPRETATION = """
Interpretation pass. Object identities and boxes were measured before this call and are in the prior
analysis. A row with observation_role canonical is measured geometry. A row with geometry_origin
vlm_estimated is an earlier estimate, not a measurement. Keep every object id and do not add, move,
or redraw boxes. Return actions, states, transitions, relationships, workflow_steps, dependencies,
and ambiguities. Do not return objects or observations, and use only the keys in the schema. Return
an empty list when the evidence supports no record. Overlap or proximity is not attachment, insertion,
or completion.
""".strip()

COMPARISON = """
Compare two structured analyses of the same recording. Object ids are not equivalent across runs.
Do not treat disagreement as proof that either analysis is wrong. Note unmatched objects, unmatched
actions, timestamp shifts, conflicting outcomes, and evidence-class differences.
""".strip()

TEMPLATES: dict[str, tuple[str, str]] = {
    "foundational": ("Foundational evidence rules", FOUNDATIONAL),
    "scene": ("General scene analysis and object identification", SCENE),
    "actions": ("Temporal action recognition", ACTIONS),
    "states": ("Object state extraction", STATES),
    "workflow": ("Workflow reconstruction", WORKFLOW),
    "ambiguity": ("Ambiguity analysis", AMBIGUITY),
    "reinspect": ("Targeted interval reinspection", REINSPECT),
    "comparison": ("Cross-run comparison", COMPARISON),
}

CAMERA_CONTEXT = {
    "fixed_camera": (
        "Recording context: a stationary camera is believed to observe the workspace. "
        "You may compare positions in the image frame. Do not claim metric world coordinates."
    ),
    "constrained_head_mounted": (
        "Recording context: a head-mounted camera is believed to stay within one workspace. "
        "Separate head motion, rotation, and perspective change from object motion. "
        "Expect temporary occlusion of the hands and objects."
    ),
    "unconstrained_head_mounted": (
        "Recording context: a head-mounted camera with no predefined activity, object inventory, "
        "workspace, or environment. The person may move between locations. Objects may leave and "
        "reenter the view. Do not invent continuity across gaps. More than one activity may be present. "
        "Distinguish movement of the person or camera from movement of physical objects."
    ),
    "auto": (
        "Recording context: the camera arrangement was not specified. A deterministic pre-pass "
        "produced a characterization you may reject if the pictures disagree. State your own assessment."
    ),
}


def compose(stage_key: str, research_mode: str, extra: str = "") -> str:
    _name, stage = TEMPLATES[stage_key]
    context = CAMERA_CONTEXT.get(research_mode, CAMERA_CONTEXT["auto"])
    parts = [FOUNDATIONAL, context, stage]
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)
