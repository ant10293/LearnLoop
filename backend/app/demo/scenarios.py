"""Synthetic research videos and the authored analyses that match their pixels.

These documents are fixtures. They are not model output and must stay labeled as such.
The prompts in app.pipeline.prompts do not mention the object names used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.schemas.analysis import (
    ActionPrediction,
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    CameraAssessment,
    Coverage,
    DependencyPrediction,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
    RelationshipPrediction,
    StatePrediction,
    TransitionPrediction,
    WorkflowStepPrediction,
)
from app.schemas.enums import (
    ActionOutcome,
    AmbiguityKind,
    DependencyCategory,
    EvidenceClass,
    Provenance,
    RelationKind,
    Visibility,
)

WIDTH = 640
HEIGHT = 360
FPS = 10


@dataclass
class Key:
    t: float
    x: int
    y: int
    w: int
    h: int
    state: str


@dataclass
class Prop:
    local_id: str
    label: str
    description: str
    visual: str
    color: tuple[int, int, int]
    shape: str
    keys: list[Key]
    uncertain: bool = False
    manipulated: bool = True


@dataclass
class AuthoredAction:
    local_id: str
    action_type: str
    description: str
    start: float
    end: float
    objects: list[str]
    initial_state: str
    final_state: str
    outcome: ActionOutcome
    uncertainty: float
    classification: EvidenceClass
    note: str


@dataclass
class Scenario:
    key: str
    title: str
    research_mode: str
    viewpoint: str
    task_constraint: str
    duration: float
    summary: str
    props: list[Prop]
    actions: list[AuthoredAction]
    ambiguities: list[AmbiguityPrediction]
    dependencies: list[tuple[str, str, DependencyCategory, str]]
    background: str
    camera_motion: str = "none"
    cut_at: float | None = None
    notes: str = ""
    steps_parallel: dict[str, str] = field(default_factory=dict)


def _ev(classification: EvidenceClass, start: float, end: float | None = None, note: str = "") -> Evidence:
    return Evidence(
        classification=classification,
        provenance=Provenance.HUMAN_ANNOTATION,
        note=note,
        time_start=start,
        time_end=end if end is not None else start,
        frame_timestamps=[start],
    )


def _interp(keys: list[Key], t: float) -> tuple[int, int, int, int, str]:
    if t <= keys[0].t:
        k = keys[0]
        return k.x, k.y, k.w, k.h, k.state
    if t >= keys[-1].t:
        k = keys[-1]
        return k.x, k.y, k.w, k.h, k.state
    for left, right in zip(keys, keys[1:], strict=False):
        if left.t <= t <= right.t:
            span = max(right.t - left.t, 1e-6)
            mix = (t - left.t) / span
            return (
                int(left.x + (right.x - left.x) * mix),
                int(left.y + (right.y - left.y) * mix),
                int(left.w + (right.w - left.w) * mix),
                int(left.h + (right.h - left.h) * mix),
                right.state if mix > 0.5 else left.state,
            )
    k = keys[-1]
    return k.x, k.y, k.w, k.h, k.state


def camera_offset(scenario: Scenario, t: float) -> tuple[int, int]:
    if scenario.camera_motion == "sway":
        return int(18 * math.sin(t * 1.7)), int(8 * math.sin(t * 1.1))
    if scenario.camera_motion == "walk" and scenario.cut_at and t >= scenario.cut_at:
        return 6, 0
    return 0, 0


def fixed_assembly() -> Scenario:
    bolt = Prop(
        "Object_1",
        "Threaded bolt",
        "Long silver cylinder with a visible helical ridge.",
        "metallic cylinder, helical ridge, rounded head",
        (180, 186, 196),
        "rect",
        [
            Key(0, 70, 150, 18, 78, "lying on the work surface"),
            Key(1.2, 70, 150, 18, 78, "lying on the work surface"),
            Key(2.2, 250, 120, 18, 78, "held above the work surface"),
            Key(3.0, 300, 168, 18, 78, "lying on the work surface"),
        ],
    )
    lock = Prop(
        "Object_2",
        "Split lock washer",
        "Small thick ring with a visible split.",
        "small split ring, darker metal",
        (90, 90, 110),
        "ellipse",
        [
            Key(0, 150, 220, 34, 34, "lying on the work surface"),
            Key(3.3, 150, 220, 34, 34, "lying on the work surface"),
            Key(4.3, 292, 150, 34, 34, "held near the bolt shaft; contact is partly hidden"),
            Key(5.1, 292, 176, 34, 34, "visible around the bolt shaft"),
        ],
    )
    flat = Prop(
        "Object_3",
        "Flat washer",
        "Thin flat ring, wider than the split ring.",
        "thin wide ring",
        (150, 150, 160),
        "ellipse",
        [
            Key(0, 210, 230, 46, 16, "lying flat on the work surface"),
            Key(5.3, 210, 230, 46, 16, "lying flat on the work surface"),
            Key(6.2, 286, 196, 46, 16, "stacked against the split ring"),
        ],
    )
    nut = Prop(
        "Object_4",
        "Hexagonal nut",
        "Six-sided block with a dark central hole.",
        "hexagonal outline, central hole",
        (40, 70, 140),
        "hex",
        [
            Key(0, 250, 110, 36, 36, "resting on the work surface"),
            Key(6.5, 250, 110, 36, 36, "resting on the work surface"),
            Key(7.5, 286, 150, 36, 36, "touching the bolt end; thread engagement is not visible"),
        ],
    )
    actions = [
        AuthoredAction("Action_1", "pick_up", "A hand lifts the threaded cylinder off the surface.", 1.2, 2.0, ["Object_1"], "on surface", "held", ActionOutcome.COMPLETED, 0.2, EvidenceClass.DIRECTLY_OBSERVED, "The cylinder is clear of the surface at 2.2s."),
        AuthoredAction("Action_2", "put_down", "The cylinder is set down toward the right side of the surface.", 2.2, 3.0, ["Object_1"], "held", "on surface", ActionOutcome.COMPLETED, 0.2, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_3", "pick_up", "A hand lifts the split ring.", 3.3, 4.1, ["Object_2"], "on surface", "held", ActionOutcome.COMPLETED, 0.25, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_4", "align", "The split ring is moved toward the cylinder. The hand hides the contact.", 4.1, 4.8, ["Object_2", "Object_1"], "held", "near cylinder", ActionOutcome.ATTEMPTED, 0.45, EvidenceClass.INFERRED, "Proximity is visible. Seating is not."),
        AuthoredAction("Action_5", "place", "The split ring is later visible around the cylinder shaft.", 4.8, 5.2, ["Object_2", "Object_1"], "near cylinder", "around shaft", ActionOutcome.COMPLETED, 0.3, EvidenceClass.DIRECTLY_OBSERVED, "The around-shaft state is visible at 5.1s. The motion that produced it was partly occluded."),
        AuthoredAction("Action_6", "pick_up", "A hand lifts the flat ring.", 5.3, 5.8, ["Object_3"], "on surface", "held", ActionOutcome.COMPLETED, 0.25, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_7", "place", "The flat ring is set against the split ring.", 5.8, 6.3, ["Object_3", "Object_2"], "held", "stacked against split ring", ActionOutcome.COMPLETED, 0.3, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_8", "twist", "The hexagonal block is turned against the cylinder end.", 6.6, 7.6, ["Object_4", "Object_1"], "on surface", "touching the end", ActionOutcome.ATTEMPTED, 0.55, EvidenceClass.INFERRED, "Contact is visible. Thread engagement is not."),
    ]
    ambiguities = [
        AmbiguityPrediction(
            local_id="Ambiguity_1",
            kind=AmbiguityKind.OCCLUSION,
            description="The hand hides whether the split ring actually passed onto the shaft between 4.3s and 5.0s.",
            time_start=4.3,
            time_end=5.0,
            object_local_ids=["Object_2", "Object_1"],
            action_local_ids=["Action_4"],
            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 4.3, 5.0, "Hand overlaps the contact region."),
        ),
        AmbiguityPrediction(
            local_id="Ambiguity_2",
            kind=AmbiguityKind.UNVERIFIABLE_COMPLETION,
            description="The hexagonal block touches the cylinder, but engagement of the threads is not observable.",
            time_start=7.2,
            time_end=7.6,
            object_local_ids=["Object_4", "Object_1"],
            action_local_ids=["Action_8"],
            alternative="The block may only rest against the end.",
            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 7.5, 7.6),
        ),
    ]
    deps = [
        ("Step_1", "Step_4", DependencyCategory.OBSERVED_TEMPORAL, "The ring was moved after the cylinder was put down. Nothing visible proves the order is physically required."),
        ("Step_4", "Step_5", DependencyCategory.NECESSARY_PHYSICAL, "The ring can only be observed around the shaft after it has been brought to the shaft."),
        ("Step_5", "Step_7", DependencyCategory.OBSERVED_TEMPORAL, "The flat ring was stacked afterward. Another order might be physically possible."),
        ("Step_7", "Step_8", DependencyCategory.INFERRED, "The block was handled last in this demonstration. Necessity is not established."),
    ]
    return Scenario(
        key="fixed_assembly",
        title="Fixed camera — small hardware stack",
        research_mode="fixed_camera",
        viewpoint="fixed",
        task_constraint="constrained",
        duration=8.0,
        summary="A stationary view of a hand moving four small metal parts into a stack. One placement is occluded and one final engagement is not verifiable.",
        props=[bolt, lock, flat, nut],
        actions=actions,
        ambiguities=ambiguities,
        dependencies=deps,
        background="bench",
        notes="Authored fixture for a stationary camera. Not a model prediction.",
    )


def constrained_workspace() -> Scenario:
    cup = Prop(
        "Object_1",
        "Open cup",
        "Upright container with a visible rim.",
        "tall cylinder, open top",
        (70, 90, 180),
        "rect",
        [
            Key(0, 80, 140, 50, 70, "upright on the desk"),
            Key(8, 80, 140, 50, 70, "upright on the desk"),
        ],
        manipulated=False,
    )
    lid = Prop(
        "Object_2",
        "Lid",
        "Flat disc matching the cup rim.",
        "flat disc",
        (40, 40, 40),
        "ellipse",
        [
            Key(0, 180, 210, 54, 18, "lying beside the cup"),
            Key(2.0, 180, 210, 54, 18, "lying beside the cup"),
            Key(3.2, 78, 128, 54, 18, "held above the rim"),
            Key(4.2, 78, 136, 54, 18, "resting on the rim"),
        ],
    )
    spoon = Prop(
        "Object_3",
        "Spoon",
        "Long handle with an oval end.",
        "thin handle, oval bowl",
        (190, 190, 200),
        "rect",
        [
            Key(0, 280, 200, 90, 16, "lying on the desk"),
            Key(4.6, 280, 200, 90, 16, "lying on the desk"),
            Key(5.6, 110, 180, 90, 16, "held over the cup"),
            Key(6.4, 96, 168, 90, 16, "handle sticking out of the cup; the bowl is hidden"),
        ],
    )
    actions = [
        AuthoredAction("Action_1", "pick_up", "The lid is lifted from the desk.", 2.0, 2.8, ["Object_2"], "on desk", "held", ActionOutcome.COMPLETED, 0.2, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_2", "place", "The lid is set on the cup rim.", 2.8, 4.2, ["Object_2", "Object_1"], "held", "on rim", ActionOutcome.COMPLETED, 0.25, EvidenceClass.DIRECTLY_OBSERVED, "The lid overlaps the rim at 4.2s."),
        AuthoredAction("Action_3", "pick_up", "The spoon is lifted.", 4.6, 5.4, ["Object_3"], "on desk", "held", ActionOutcome.COMPLETED, 0.2, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_4", "insert", "The spoon moves toward the cup. The bowl ends hidden inside or behind the cup.", 5.4, 6.4, ["Object_3", "Object_1"], "held", "partially hidden by the cup", ActionOutcome.ATTEMPTED, 0.5, EvidenceClass.INFERRED, "Disappearance of the bowl is not proof it entered the cup."),
    ]
    ambiguities = [
        AmbiguityPrediction(
            local_id="Ambiguity_1",
            kind=AmbiguityKind.CAMERA_MOTION,
            description="The whole desk shifts in the image while the head-mounted camera sways. Object motion has to be separated from that shift.",
            time_start=0,
            time_end=8,
            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 0, 8, "Background texture translates across the frame."),
        ),
        AmbiguityPrediction(
            local_id="Ambiguity_2",
            kind=AmbiguityKind.UNVERIFIABLE_COMPLETION,
            description="The spoon bowl is hidden at the end. Containment inside the cup is not directly observed.",
            time_start=6.0,
            time_end=6.4,
            object_local_ids=["Object_3", "Object_1"],
            action_local_ids=["Action_4"],
            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 6.4, 6.4),
        ),
    ]
    deps = [
        ("Step_1", "Step_2", DependencyCategory.NECESSARY_PHYSICAL, "The lid is on the rim only after it is moved there."),
        ("Step_2", "Step_3", DependencyCategory.OBSERVED_TEMPORAL, "The spoon was handled after the lid in this take. The order is not shown to be required."),
    ]
    return Scenario(
        key="constrained_head_mounted",
        title="Constrained head-mounted — desk workspace",
        research_mode="constrained_head_mounted",
        viewpoint="head_mounted",
        task_constraint="constrained",
        duration=8.0,
        summary="First-person view with camera sway inside one workspace. A lid is placed on a cup. A spoon is moved toward the cup, and its final containment is hidden.",
        props=[cup, lid, spoon],
        actions=actions,
        ambiguities=ambiguities,
        dependencies=deps,
        background="desk",
        camera_motion="sway",
        notes="Authored fixture. Camera motion is real in the pixels; world coordinates are not claimed.",
    )


def unconstrained_move() -> Scenario:
    cube = Prop(
        "Object_1",
        "Red block",
        "Small red cube.",
        "red cube",
        (40, 40, 200),
        "rect",
        [
            Key(0.0, 80, 180, 40, 40, "on a blue surface"),
            Key(2.4, 180, 170, 40, 40, "in the hand"),
            Key(3.2, 200, 160, 40, 40, "leaving the view"),
        ],
    )
    book = Prop(
        "Object_2",
        "Green notebook",
        "Flat green rectangle.",
        "green flat rectangle",
        (60, 140, 50),
        "rect",
        [
            Key(0, 300, 200, 90, 50, "on the blue surface"),
            Key(3.2, 300, 200, 90, 50, "on the blue surface"),
        ],
        manipulated=False,
    )
    bottle = Prop(
        "Object_3",
        "Yellow bottle",
        "Tall yellow container that appears only after the scene change.",
        "tall yellow cylinder",
        (40, 200, 220),
        "rect",
        [Key(3.6, 120, 140, 36, 90, "standing on a green surface"), Key(8, 120, 140, 36, 90, "standing on a green surface")],
        manipulated=False,
    )
    maybe = Prop(
        "Object_4",
        "Red block, identity uncertain",
        "A red cube enters after the cut. It resembles Object_1, but the gap was not observed.",
        "red cube",
        (40, 40, 200),
        "rect",
        [Key(6.0, 360, 180, 40, 40, "entering from the right"), Key(7.4, 280, 180, 40, 40, "resting on the green surface")],
        uncertain=True,
    )
    actions = [
        AuthoredAction("Action_1", "pick_up", "The red block is lifted in the first room.", 1.2, 2.2, ["Object_1"], "on surface", "held", ActionOutcome.COMPLETED, 0.25, EvidenceClass.DIRECTLY_OBSERVED, ""),
        AuthoredAction("Action_2", "move", "The block moves toward the edge of the view.", 2.2, 3.2, ["Object_1"], "held", "leaving view", ActionOutcome.COMPLETED, 0.3, EvidenceClass.DIRECTLY_OBSERVED, "It is not seen being set down."),
        AuthoredAction("Action_3", "transfer", "A red block is carried into the second room. Whether it is the same block is unknown.", 6.0, 7.2, ["Object_4"], "entering", "on the new surface", ActionOutcome.UNKNOWN, 0.7, EvidenceClass.UNKNOWN, "No frames cover the interval between rooms."),
    ]
    ambiguities = [
        AmbiguityPrediction(
            local_id="Ambiguity_1",
            kind=AmbiguityKind.UNOBSERVED_INTERVAL,
            description="The camera moves through an unobserved transition between two places from about 3.3s to 3.6s.",
            time_start=3.3,
            time_end=3.6,
            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 3.3, 3.6, "Background color changes abruptly."),
        ),
        AmbiguityPrediction(
            local_id="Ambiguity_2",
            kind=AmbiguityKind.UNCERTAIN_IDENTITY,
            description="Object_4 is not identified with Object_1. The first block left view and a similar block appeared later.",
            time_start=3.2,
            time_end=6.0,
            object_local_ids=["Object_1", "Object_4"],
            alternative="They may be the same block, or a different one.",
            evidence=_ev(EvidenceClass.UNKNOWN, 3.2, 6.0),
        ),
        AmbiguityPrediction(
            local_id="Ambiguity_3",
            kind=AmbiguityKind.ALTERNATIVE_INTERPRETATION,
            description="The clip may contain two independent activities: handling a block in the first place, and passing a bottle that is never touched.",
            time_start=0,
            time_end=8,
            object_local_ids=["Object_1", "Object_3"],
            evidence=_ev(EvidenceClass.INFERRED, 0, 8),
        ),
    ]
    deps = [
        ("Step_1", "Step_2", DependencyCategory.NECESSARY_PHYSICAL, "The block can leave the hand's reach only after it is held."),
        ("Step_2", "Step_3", DependencyCategory.UNKNOWN, "The later red block is not established as the same object, so no dependency is claimed."),
    ]
    return Scenario(
        key="unconstrained_head_mounted",
        title="Unconstrained head-mounted — two places",
        research_mode="unconstrained_head_mounted",
        viewpoint="head_mounted",
        task_constraint="unconstrained",
        duration=8.0,
        summary="First-person footage changes rooms. A red block leaves view. A similar block appears later and is kept as a separate, uncertain identity. A bottle is present and never manipulated.",
        props=[cube, book, bottle, maybe],
        actions=actions,
        ambiguities=ambiguities,
        dependencies=deps,
        background="rooms",
        camera_motion="walk",
        cut_at=3.4,
        notes="Authored fixture. Continuity across the cut is deliberately not asserted.",
    )


SCENARIOS = {
    "fixed_camera": fixed_assembly,
    "constrained_head_mounted": constrained_workspace,
    "unconstrained_head_mounted": unconstrained_move,
}


def build_document(scenario: Scenario) -> AnalysisDocument:
    objects = []
    observations = []
    states = []
    transitions = []
    relationships = []
    for prop in scenario.props:
        objects.append(
            ObjectPrediction(
                local_id=prop.local_id,
                label=prop.label,
                description=prop.description,
                visual_characteristics=prop.visual,
                initial_state=prop.keys[0].state,
                final_state=prop.keys[-1].state,
                uncertain_identity=prop.uncertain,
                manipulated=prop.manipulated,
                confidence=None,
            )
        )
        sample_times = [round(t * 0.5, 2) for t in range(int(scenario.duration * 2) + 1)]
        last_state = None
        state_start = sample_times[0]
        for timestamp in sample_times:
            if scenario.cut_at and prop.keys[0].t > scenario.cut_at and timestamp < prop.keys[0].t:
                continue
            if prop.keys[-1].t < scenario.duration and timestamp > prop.keys[-1].t + 0.2 and prop.local_id == "Object_1" and scenario.cut_at:
                if timestamp > prop.keys[-1].t:
                    observations.append(
                        ObservationPrediction(
                            object_local_id=prop.local_id,
                            timestamp=timestamp,
                            bbox=None,
                            visibility=Visibility.ABSENT,
                            description="Not visible after leaving the first view.",
                            evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, timestamp, note="Absent from the frame."),
                            continuity="uncertain",
                        )
                    )
                    continue
            x, y, w, h, state = _interp(prop.keys, timestamp)
            ox, oy = camera_offset(scenario, timestamp)
            x += ox
            y += oy
            if x + w < 0 or x > WIDTH or y + h < 0 or y > HEIGHT:
                visibility = Visibility.ABSENT
                bbox = None
            else:
                visibility = Visibility.VISIBLE
                bbox = BBox(
                    x=max(0, x) / WIDTH,
                    y=max(0, y) / HEIGHT,
                    w=max(1, min(WIDTH, x + w) - max(0, x)) / WIDTH,
                    h=max(1, min(HEIGHT, y + h) - max(0, y)) / HEIGHT,
                )
            continuity = "uncertain" if prop.uncertain else "continuous"
            observations.append(
                ObservationPrediction(
                    object_local_id=prop.local_id,
                    timestamp=timestamp,
                    bbox=bbox,
                    visibility=visibility,
                    description=state,
                    evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, timestamp, note="Authored against the rendered frame."),
                    continuity=continuity,
                    coordinate_frame="image",
                    camera_compensated=False,
                )
            )
            if last_state is None:
                last_state = state
                state_start = timestamp
            elif state != last_state:
                states.append(
                    StatePrediction(
                        object_local_id=prop.local_id,
                        time_start=state_start,
                        time_end=timestamp,
                        state_label=last_state,
                        description=last_state,
                        evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, state_start, timestamp),
                    )
                )
                transitions.append(
                    TransitionPrediction(
                        object_local_id=prop.local_id,
                        timestamp=timestamp,
                        from_state=last_state,
                        to_state=state,
                        description=f"{last_state} → {state}",
                        evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, timestamp),
                        certainty=0.4 if prop.uncertain else 0.8,
                    )
                )
                last_state = state
                state_start = timestamp
        if last_state is not None:
            states.append(
                StatePrediction(
                    object_local_id=prop.local_id,
                    time_start=state_start,
                    time_end=scenario.duration,
                    state_label=last_state,
                    description=last_state,
                    evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, state_start, scenario.duration),
                )
            )
    if scenario.key == "fixed_assembly":
        relationships.append(
            RelationshipPrediction(
                subject_local_id="Object_2",
                other_local_id="Object_1",
                kind=RelationKind.AROUND,
                time_start=5.1,
                time_end=8,
                description="The split ring is visible around the cylinder shaft.",
                evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 5.1, 8),
            )
        )
        relationships.append(
            RelationshipPrediction(
                subject_local_id="Object_4",
                other_local_id="Object_1",
                kind=RelationKind.TOUCHING,
                time_start=7.5,
                time_end=8,
                description="The hexagonal block touches the cylinder end. Attachment is not claimed.",
                evidence=_ev(EvidenceClass.DIRECTLY_OBSERVED, 7.5, 8, "Touching is visible. Fastening is not."),
            )
        )
    actions = [
        ActionPrediction(
            local_id=item.local_id,
            action_type=item.action_type,
            description=item.description,
            start=item.start,
            end=item.end,
            actor="visible hand",
            object_local_ids=item.objects,
            initial_state=item.initial_state,
            final_state=item.final_state,
            outcome=item.outcome,
            uncertainty=item.uncertainty,
            evidence=_ev(item.classification, item.start, item.end, item.note),
        )
        for item in scenario.actions
    ]
    steps = []
    for index, action in enumerate(scenario.actions, start=1):
        steps.append(
            WorkflowStepPrediction(
                local_id=f"Step_{index}",
                name=action.description[:80],
                required_objects=action.objects,
                preconditions=[action.initial_state] if action.initial_state else [],
                expected_action=action.action_type,
                completion_condition=action.final_state or "visible end state",
                observed_result=action.final_state,
                action_local_ids=[action.local_id],
                time_start=action.start,
                time_end=action.end,
                uncertainty=action.uncertainty,
                evidence=_ev(action.classification, action.start, action.end, action.note),
                parallel_group=scenario.steps_parallel.get(f"Step_{index}", ""),
            )
        )
    dependencies = [
        DependencyPrediction(
            from_step=src,
            to_step=dst,
            category=category,
            rationale=reason,
            evidence=_ev(EvidenceClass.INFERRED if category != DependencyCategory.NECESSARY_PHYSICAL else EvidenceClass.INFERRED, 0, scenario.duration, reason),
        )
        for src, dst, category, reason in scenario.dependencies
    ]
    return AnalysisDocument(
        summary=scenario.summary,
        fixture=True,
        fixture_label=scenario.title,
        camera=CameraAssessment(
            viewpoint=scenario.viewpoint,
            task_constraint=scenario.task_constraint,
            research_mode=scenario.research_mode,
            coordinate_frame="image",
            world_coordinates_supported=False,
            camera_motion_compensated=False,
            compensation_model="none",
            characterization_source="user",
            notes=scenario.notes,
        ),
        coverage=Coverage(
            modality="sampled_frames",
            sampling_preset="detailed",
            nominal_fps=2,
            frames_extracted=int(scenario.duration * 2) + 1,
            frames_submitted=int(scenario.duration * 2) + 1,
            frame_timestamps=[round(t * 0.5, 2) for t in range(int(scenario.duration * 2) + 1)],
            selection_reasons={"authored_fixture": int(scenario.duration * 2) + 1},
            duration_sec=scenario.duration,
            native_video_submitted=False,
            temporal_coverage_note=(
                "Fixture observations are sampled at 2 Hz from the rendered clip. "
                "They are not a claim that every source frame was interpreted by a model."
            ),
        ),
        objects=objects,
        observations=observations,
        states=states,
        transitions=transitions,
        relationships=relationships,
        actions=actions,
        workflow_steps=steps,
        dependencies=dependencies,
        ambiguities=scenario.ambiguities,
    )


def render_scenario(scenario: Scenario) -> list[np.ndarray]:
    frames = []
    count = int(scenario.duration * FPS)
    for index in range(count):
        t = index / FPS
        frame = _background(scenario, t)
        ox, oy = camera_offset(scenario, t)
        for prop in scenario.props:
            if t + 1e-6 < prop.keys[0].t or t > prop.keys[-1].t + 0.05 and scenario.cut_at and prop.local_id == "Object_1" and t > prop.keys[-1].t:
                continue
            if prop.keys[-1].t < t and not (scenario.cut_at and prop.local_id == "Object_1"):
                if t > prop.keys[-1].t:
                    continue
            x, y, w, h, _state = _interp(prop.keys, min(t, prop.keys[-1].t))
            _draw(frame, prop.shape, x + ox, y + oy, w, h, prop.color)
        _draw_hand(frame, scenario, t, ox, oy)
        frames.append(frame)
    return frames


def _background(scenario: Scenario, t: float) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ox, oy = camera_offset(scenario, t)
    if scenario.background == "rooms" and scenario.cut_at and t >= scenario.cut_at:
        image[:] = (70, 110, 70)
        cv2.rectangle(image, (0, 230), (WIDTH, HEIGHT), (50, 80, 50), -1)
    elif scenario.background == "rooms":
        image[:] = (120, 80, 50)
        cv2.rectangle(image, (0, 230), (WIDTH, HEIGHT), (90, 60, 40), -1)
    else:
        image[:] = (90, 86, 80)
        cv2.rectangle(image, (0, 210), (WIDTH, HEIGHT), (70, 74, 78), -1)
    for x in range(-40, WIDTH + 40, 40):
        cv2.line(image, (x + ox, 210), (x + 20 + ox, HEIGHT), (60, 60, 60), 1)
    cv2.putText(image, f"{t:0.1f}s", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return image


def _draw(image, shape, x, y, w, h, color) -> None:
    if shape == "ellipse":
        cv2.ellipse(image, (x + w // 2, y + h // 2), (max(w // 2, 1), max(h // 2, 1)), 0, 0, 360, color, -1)
    elif shape == "hex":
        cx, cy = x + w // 2, y + h // 2
        pts = []
        for k in range(6):
            ang = math.radians(60 * k - 30)
            pts.append((int(cx + w * 0.5 * math.cos(ang)), int(cy + h * 0.5 * math.sin(ang))))
        cv2.fillConvexPoly(image, np.array(pts, dtype=np.int32), color)
        cv2.circle(image, (cx, cy), max(3, w // 8), (20, 20, 20), -1)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)


def _draw_hand(image, scenario: Scenario, t: float, ox: int, oy: int) -> None:
    target = None
    for action in scenario.actions:
        if action.start <= t <= action.end and action.objects:
            prop = next(item for item in scenario.props if item.local_id == action.objects[0])
            target = _interp(prop.keys, min(t, prop.keys[-1].t))
            break
    if target is None:
        return
    x, y, w, h, _state = target
    cv2.ellipse(image, (x + ox + w // 2, y + oy - 8), (22, 14), 0, 0, 360, (80, 120, 180), -1)
