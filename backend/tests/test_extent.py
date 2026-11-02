"""Extent keeps an id on one part, and insertion is read from that part's track."""

from app.perception.assembly import apply_assembly, find_joins
from app.perception.continuity import reconcile_identity
from app.perception.locate import apply_located, box_fits, choose_locate_times
from app.schemas.analysis import (
    ActionPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
    RelationshipPrediction,
    WorkflowStepPrediction,
)
from app.schemas.enums import ActionOutcome, EvidenceClass, Provenance, RelationKind


def _evidence(timestamp: float = 0) -> Evidence:
    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.AI_PREDICTION,
        frame_timestamps=[timestamp],
    )


def _obs(object_id: str, timestamp: float, x: float, y: float, w: float, h: float) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=x, y=y, w=w, h=h),
        evidence=_evidence(timestamp),
    )


def _doc(observations, actions=None, steps=None) -> AnalysisDocument:
    return AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_1", label="short rod", manipulated=True),
            ObjectPrediction(local_id="Object_5", label="gray body", manipulated=True),
            ObjectPrediction(local_id="Object_4", label="clear piece", manipulated=True),
        ],
        observations=observations,
        actions=actions or [],
        workflow_steps=steps or [],
    )


def test_a_box_many_times_larger_than_the_part_is_removed():
    rod = [_obs("Object_1", float(t), 0.55, 0.25, 0.08, 0.02) for t in range(5)]
    body = [_obs("Object_5", float(t), 0.40, 0.40, 0.20, 0.15) for t in range(6)]
    stolen = _obs("Object_1", 5, 0.40, 0.40, 0.20, 0.16)
    document = reconcile_identity(_doc([*rod, stolen, *body]))
    late = [item for item in document.observations if item.object_local_id == "Object_1" and item.timestamp == 5]
    assert late == []
    assert any(item.object_local_id == "Object_5" and item.timestamp == 5 for item in document.observations)
    assert any(item.local_id == "Ambiguity_extent_Object_1" for item in document.ambiguities)


def test_a_medium_box_is_not_handed_to_a_smaller_part():
    rod = [_obs("Object_1", float(t), 0.55, 0.25, 0.08, 0.02) for t in range(4)]
    clear = [_obs("Object_4", float(t), 0.10, 0.70, 0.03, 0.03) for t in range(4)]
    foreign = _obs("Object_4", 4, 0.30, 0.40, 0.10, 0.046)
    document = reconcile_identity(_doc([*rod, *clear, foreign]))
    assert not any(item.object_local_id == "Object_1" and item.timestamp == 4 for item in document.observations)


def test_a_modest_scale_change_stays_with_the_part():
    observations = [_obs("Object_1", float(t), 0.2, 0.2, 0.10, 0.04) for t in range(4)]
    observations.append(_obs("Object_1", 4, 0.2, 0.2, 0.14, 0.06))
    document = reconcile_identity(_doc(observations))
    assert any(item.object_local_id == "Object_1" and item.timestamp == 4 for item in document.observations)


def test_a_swapped_small_box_returns_to_the_part_it_fits():
    rod = [_obs("Object_1", float(t), 0.55, 0.25, 0.08, 0.02) for t in range(4)]
    body = [_obs("Object_5", float(t), 0.40, 0.40, 0.20, 0.15) for t in range(4)]
    clear = [_obs("Object_4", float(t), 0.10, 0.70, 0.03, 0.03) for t in range(4)]
    # The body id is now on the clear piece, far from where the body was.
    swapped = _obs("Object_5", 4, 0.10, 0.72, 0.03, 0.03)
    document = reconcile_identity(_doc([*rod, *body, *clear, swapped]))
    moved = [item for item in document.observations if item.timestamp == 4 and abs(item.bbox.w - 0.03) < 1e-6]
    assert [item.object_local_id for item in moved] == ["Object_4"]


def test_insertion_is_the_free_part_ending_at_the_host():
    observations = []
    for timestamp in range(6):
        observations.append(_obs("Object_5", float(timestamp), 0.45, 0.40, 0.18, 0.16))
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.08, 0.02),
            _obs("Object_1", 1, 0.12, 0.20, 0.08, 0.02),
            _obs("Object_1", 2, 0.20, 0.25, 0.08, 0.02),
            _obs("Object_1", 3, 0.42, 0.42, 0.08, 0.02),
        ]
    )
    action = ActionPrediction(
        local_id="Action_9",
        action_type="approach",
        description="The rod is held near the assembly.",
        start=2,
        end=4,
        object_local_ids=["Object_1", "Object_5"],
        outcome=ActionOutcome.UNKNOWN,
        uncertainty=0.7,
        evidence=_evidence(3),
    )
    step = WorkflowStepPrediction(
        local_id="Step_6",
        name="Handle rod and inspect assembly",
        expected_action="Pick up the rod and bring it to the assembly",
        completion_condition="Rod visibly inserted or attached",
        observed_result="Held near the assembly. Completion unverified.",
        action_local_ids=["Action_9"],
        time_start=2,
        time_end=5,
        uncertainty=0.7,
        evidence=_evidence(3),
    )
    document = apply_assembly(_doc(observations, [action], [step]))
    assert document.actions[0].action_type == "insert"
    # Box geometry infers the join. It keeps the model's own outcome and never confirms completion.
    assert document.actions[0].outcome == ActionOutcome.UNKNOWN
    assert document.actions[0].evidence.classification.value == "INFERRED"
    assert document.workflow_steps[0].uncertainty >= 0.5
    assert document.workflow_steps[0].name.startswith("Insert the short rod")
    assert "bring" not in document.workflow_steps[0].expected_action.lower()
    assert "separate visible extent" in document.workflow_steps[0].observed_result
    assert find_joins(document)[0].host_ids == ("Object_5",)
    assert find_joins(document)[0].contact_time == 3.0
    assert find_joins(document)[0].kind == "inside"


def test_sitting_beside_a_part_at_the_start_is_not_the_moment_of_insertion():
    observations = []
    for timestamp in range(6):
        observations.append(_obs("Object_5", float(timestamp), 0.55, 0.45, 0.18, 0.16))
        observations.append(_obs("Object_4", float(timestamp), 0.10, 0.18, 0.04, 0.04))
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.24, 0.08, 0.02),
            _obs("Object_1", 1, 0.10, 0.24, 0.08, 0.02),
            _obs("Object_1", 2, 0.12, 0.24, 0.08, 0.02),
            _obs("Object_1", 3, 0.52, 0.46, 0.08, 0.02),
        ]
    )
    event = find_joins(_doc(observations))[0]
    assert event.contact_time == 3.0
    assert "Object_4" not in event.host_ids
    assert event.host_ids[0] == "Object_5"


def test_a_part_that_comes_back_is_not_an_insertion():
    observations = [_obs("Object_5", float(t), 0.45, 0.40, 0.18, 0.16) for t in range(6)]
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.08, 0.02),
            _obs("Object_1", 1, 0.12, 0.20, 0.08, 0.02),
            _obs("Object_1", 2, 0.42, 0.42, 0.08, 0.02),
            _obs("Object_1", 5, 0.10, 0.20, 0.08, 0.02),
        ]
    )
    assert find_joins(_doc(observations)) == []


def _host(object_id: str = "Object_5", frames: int = 10):
    return [_obs(object_id, float(t), 0.45, 0.40, 0.18, 0.16) for t in range(frames)]


def test_an_inserted_part_that_is_pulled_out_is_removed():
    observations = _host()
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.08, 0.02),
            _obs("Object_1", 1, 0.15, 0.22, 0.08, 0.02),
            _obs("Object_1", 2, 0.42, 0.42, 0.06, 0.02),
            _obs("Object_1", 3, 0.44, 0.43, 0.05, 0.02),
            _obs("Object_1", 4, 0.44, 0.43, 0.05, 0.02),
            _obs("Object_1", 5, 0.44, 0.43, 0.05, 0.02),
            _obs("Object_1", 6, 0.20, 0.25, 0.08, 0.02),
            _obs("Object_1", 7, 0.12, 0.22, 0.08, 0.02),
            _obs("Object_1", 8, 0.12, 0.22, 0.08, 0.02),
            _obs("Object_1", 9, 0.12, 0.22, 0.08, 0.02),
        ]
    )
    claimed = RelationshipPrediction(
        subject_local_id="Object_1",
        other_local_id="Object_5",
        kind=RelationKind.INSIDE,
        time_start=2,
        time_end=9,
        evidence=_evidence(2),
    )
    source = _doc(observations).model_copy(update={"relationships": [claimed]})
    joins = find_joins(source)
    assert len(joins) == 1
    assert joins[0].kind == "inside"
    assert joins[0].until == 5.0
    assert joins[0].released == 6.0
    assert joins[0].free_until == 9.0

    document = apply_assembly(source)
    assert [action.action_type for action in document.actions] == ["insert", "remove"]
    assert len(document.relationships) == 1
    assert document.relationships[0].time_end == 5.0
    assert document.objects[0].final_state.startswith("Removed from the gray body at 6s")
    labels = [(state.state_label, state.time_start, state.time_end) for state in document.states]
    assert ("removed", 6.0, 9.0) in labels
    assert all(end <= 5.0 for label, _start, end in labels if label == "inserted")
    assert any(item.from_state == "inserted" and item.to_state == "removed" for item in document.transitions)

    again = apply_assembly(document)
    assert len(again.actions) == 2
    assert len(again.relationships) == 1
    assert len(again.transitions) == len(document.transitions)


def test_a_part_that_keeps_its_size_against_the_host_is_attached():
    observations = _host(frames=8)
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.06, 0.04),
            _obs("Object_1", 1, 0.14, 0.22, 0.06, 0.04),
            _obs("Object_1", 2, 0.40, 0.42, 0.06, 0.04),
            *[_obs("Object_1", float(t), 0.40, 0.42, 0.06, 0.04) for t in range(3, 8)],
        ]
    )
    step = WorkflowStepPrediction(
        local_id="Step_1",
        name="Bring the handle to the body",
        action_local_ids=["Action_1"],
        evidence=_evidence(2),
    )
    action = ActionPrediction(
        local_id="Action_1",
        action_type="move",
        start=1,
        end=3,
        object_local_ids=["Object_1"],
        evidence=_evidence(2),
    )
    document = apply_assembly(_doc(observations, [action], [step]))
    assert document.actions[0].action_type == "attach"
    assert document.relationships[0].kind == RelationKind.ATTACHED
    assert document.workflow_steps[0].name == "Attach the short rod to the gray body"
    assert document.objects[0].final_state.startswith("Attached to the gray body")
    assert any(state.state_label == "attached" for state in document.states)


def test_a_part_can_be_attached_detached_and_attached_again():
    observations = _host(frames=14)
    seated = (0.40, 0.42, 0.06, 0.04)
    free = (0.10, 0.20, 0.06, 0.04)
    track = [free, (0.14, 0.22, 0.06, 0.04), seated, seated, seated, seated, free, free, free, seated, seated, seated, seated, seated]
    observations.extend(_obs("Object_1", float(t), *box) for t, box in enumerate(track))
    joins = find_joins(_doc(observations))
    assert [(join.kind, join.contact_time, join.released) for join in joins] == [
        ("attached", 2.0, 6.0),
        ("attached", 9.0, None),
    ]
    assert joins[0].free_until == 9.0
    document = apply_assembly(_doc(observations))
    assert [action.action_type for action in document.actions] == ["attach", "detach", "attach"]
    windows = sorted((item.time_start, item.time_end) for item in document.relationships)
    assert windows == [(2.0, 5.0), (9.0, 13.0)]
    assert document.objects[0].final_state.startswith("Attached to the gray body")


def test_a_tool_pressed_on_and_lifted_away_is_not_joined():
    observations = _host(frames=7)
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.06, 0.04),
            _obs("Object_1", 1, 0.20, 0.25, 0.06, 0.04),
            _obs("Object_1", 2, 0.48, 0.38, 0.06, 0.04),
            _obs("Object_1", 3, 0.48, 0.38, 0.06, 0.04),
            _obs("Object_1", 4, 0.25, 0.25, 0.06, 0.04),
            _obs("Object_1", 5, 0.10, 0.20, 0.06, 0.04),
            _obs("Object_1", 6, 0.10, 0.20, 0.06, 0.04),
        ]
    )
    assert find_joins(_doc(observations)) == []
    assert apply_assembly(_doc(observations)).actions == []


def test_a_removed_part_set_down_beside_the_host_is_not_joined_again():
    observations = _host(frames=12)
    seated = (0.44, 0.43, 0.05, 0.02)
    free = (0.10, 0.20, 0.08, 0.02)
    beside = (0.40, 0.58, 0.08, 0.02)
    track = [free, (0.15, 0.22, 0.08, 0.02), seated, seated, seated, seated, free, free, beside, beside, beside, beside]
    observations.extend(_obs("Object_1", float(t), *box) for t, box in enumerate(track))
    joins = find_joins(_doc(observations))
    assert [(join.contact_time, join.released) for join in joins] == [(2.0, 6.0)]
    assert apply_assembly(_doc(observations)).objects[0].final_state.startswith("Removed from")


def test_a_placement_the_model_already_described_stands():
    observations = _host(frames=8)
    observations.extend(
        [
            _obs("Object_1", 0, 0.10, 0.20, 0.06, 0.04),
            _obs("Object_1", 1, 0.14, 0.22, 0.06, 0.04),
            *[_obs("Object_1", float(t), 0.40, 0.42, 0.06, 0.04) for t in range(2, 8)],
        ]
    )
    placed = ActionPrediction(
        local_id="Action_1",
        action_type="place",
        start=1.5,
        end=2.5,
        object_local_ids=["Object_1", "Object_5"],
        outcome=ActionOutcome.COMPLETED,
        evidence=_evidence(2),
    )
    source = _doc(observations, [placed])
    assert find_joins(source) == []
    assert apply_assembly(source) == source


def test_a_removal_the_model_described_keeps_its_wording_and_ends_the_relation():
    observations = _host()
    seated = (0.40, 0.42, 0.06, 0.04)
    free = (0.10, 0.20, 0.06, 0.04)
    track = [free, (0.14, 0.22, 0.06, 0.04), seated, seated, seated, seated, free, free, free, free]
    observations.extend(_obs("Object_1", float(t), *box) for t, box in enumerate(track))
    actions = [
        ActionPrediction(
            local_id="Action_1",
            action_type="place",
            start=1.5,
            end=2.5,
            object_local_ids=["Object_1", "Object_5"],
            evidence=_evidence(2),
        ),
        ActionPrediction(
            local_id="Action_2",
            action_type="pick_up",
            start=5.5,
            end=6.5,
            object_local_ids=["Object_1"],
            evidence=_evidence(6),
        ),
    ]
    around = RelationshipPrediction(
        subject_local_id="Object_1",
        other_local_id="Object_5",
        kind=RelationKind.AROUND,
        time_start=2,
        time_end=9,
        evidence=_evidence(2),
    )
    source = _doc(observations, actions).model_copy(update={"relationships": [around]})
    document = apply_assembly(source)
    assert [(action.local_id, action.action_type) for action in document.actions] == [
        ("Action_1", "place"),
        ("Action_2", "pick_up"),
    ]
    assert [(item.kind, item.time_start, item.time_end) for item in document.relationships] == [
        (RelationKind.AROUND, 2, 5.0)
    ]
    assert document.objects[0].final_state.startswith("Detached from the gray body at 6s")


def test_one_stray_box_away_from_the_host_is_not_a_removal():
    observations = _host(frames=8)
    seated = (0.44, 0.43, 0.05, 0.02)
    track = [(0.10, 0.20, 0.08, 0.02), (0.15, 0.22, 0.08, 0.02), seated, seated, seated, (0.10, 0.80, 0.08, 0.02), seated, seated]
    observations.extend(_obs("Object_1", float(t), *box) for t, box in enumerate(track))
    joins = find_joins(_doc(observations))
    assert len(joins) == 1
    assert joins[0].released is None


def test_locate_asks_for_the_hole_and_the_start_of_the_gap():
    kept = [float(t) for t in range(17)]
    all_times = [float(t) for t in range(22)]
    assert choose_locate_times(kept, all_times) == [17.0, 18.0, 19.0]


def test_a_second_arm_at_a_new_angle_stays_the_same_part():
    from app.perception.roster import align_to_roster, drop_split_objects

    roster = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_3", label="spoked wheel with black tire"),
            ObjectPrediction(local_id="Object_5", label="light-gray body piece"),
            ObjectPrediction(local_id="Object_6", label="black fork or frame piece"),
        ],
        observations=[
            _obs("Object_3", 18, 0.23, 0.42, 0.13, 0.07),
            _obs("Object_5", 18, 0.39, 0.38, 0.15, 0.15),
            _obs("Object_6", 18, 0.39, 0.39, 0.18, 0.14),
        ],
    )
    later = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_5", label="black strut lower"),
            ObjectPrediction(local_id="Object_4", label="black strut upper"),
        ],
        observations=[
            _obs("Object_5", 18, 0.40, 0.42, 0.16, 0.10),
            _obs("Object_5", 21, 0.387, 0.492, 0.16, 0.087),
            _obs("Object_4", 21, 0.289, 0.450, 0.16, 0.043),
        ],
    )
    aligned = align_to_roster(roster, later)
    assert {obj.local_id for obj in aligned.objects} == {"Object_3", "Object_5", "Object_6"}
    at_21 = [item for item in aligned.observations if item.timestamp == 21]
    assert len(at_21) == 1
    assert at_21[0].object_local_id == "Object_6"
    assert at_21[0].bbox is not None
    assert at_21[0].bbox.y < 0.46
    assert at_21[0].bbox.y + at_21[0].bbox.h > 0.55

    merged = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_3", label="spoked wheel with black tire"),
            ObjectPrediction(local_id="Object_5", label="light-gray body piece"),
            ObjectPrediction(local_id="Object_6", label="black fork or frame piece (black strut)"),
            ObjectPrediction(local_id="Object_7", label="black strut (upper-left one)"),
        ],
        observations=[
            _obs("Object_6", 18, 0.39, 0.39, 0.18, 0.14),
            _obs("Object_3", 21, 0.231, 0.427, 0.129, 0.072),
            _obs("Object_5", 21, 0.213, 0.470, 0.245, 0.110),
            _obs("Object_6", 21, 0.387, 0.492, 0.16, 0.087),
            _obs("Object_7", 21, 0.289, 0.450, 0.16, 0.043),
        ],
    )
    closed = drop_split_objects(roster, merged)
    assert {obj.local_id for obj in closed.objects} == {"Object_3", "Object_5", "Object_6"}
    fork = next(item for item in closed.observations if item.object_local_id == "Object_6" and abs(item.timestamp - 21) < 1e-6)
    assert fork.bbox is not None and fork.bbox.y < 0.46 and fork.bbox.y + fork.bbox.h > 0.55
    wheel = next(item for item in closed.observations if item.object_local_id == "Object_3" and abs(item.timestamp - 21) < 1e-6)
    assert wheel.bbox is not None and wheel.bbox.w < 0.2


def test_a_reused_id_for_a_second_arm_does_not_join_the_wheel():
    from app.perception.roster import align_to_roster

    roster = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_3", label="spoked wheel with black tire"),
            ObjectPrediction(local_id="Object_4", label="clear round stud or plate"),
            ObjectPrediction(local_id="Object_6", label="black fork or frame piece"),
        ],
        observations=[
            _obs("Object_3", 18, 0.23, 0.42, 0.13, 0.07),
            _obs("Object_4", 18, 0.50, 0.57, 0.03, 0.02),
            _obs("Object_6", 18, 0.40, 0.40, 0.16, 0.10),
        ],
    )
    later = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_3", label="spoked wheel with black tire (upper/left one)"),
            ObjectPrediction(local_id="Object_4", label="black fork/strut (upper-left one)"),
            ObjectPrediction(local_id="Object_5", label="black fork/strut (lower-right one)"),
            ObjectPrediction(local_id="Object_6", label="small transparent round stud/plate"),
        ],
        observations=[
            _obs("Object_3", 18, 0.23, 0.42, 0.13, 0.07),
            _obs("Object_6", 18, 0.50, 0.57, 0.03, 0.02),
            _obs("Object_5", 18, 0.40, 0.40, 0.16, 0.10),
            _obs("Object_3", 21, 0.231, 0.427, 0.129, 0.072),
            _obs("Object_4", 21, 0.289, 0.450, 0.160, 0.043),
            _obs("Object_5", 21, 0.387, 0.492, 0.160, 0.087),
            _obs("Object_6", 21, 0.487, 0.573, 0.025, 0.020),
        ],
    )
    aligned = align_to_roster(roster, later)
    assert {obj.local_id for obj in aligned.objects} == {"Object_3", "Object_4", "Object_6"}
    at_21 = {item.object_local_id: item.bbox for item in aligned.observations if item.timestamp == 21}
    assert set(at_21) == {"Object_3", "Object_4", "Object_6"}
    fork = at_21["Object_6"]
    assert fork is not None and fork.y < 0.46 and fork.y + fork.h > 0.55
    wheel = at_21["Object_3"]
    assert wheel is not None and wheel.w < 0.2


def test_a_separate_part_is_not_folded_into_a_neighbor():
    from app.perception.roster import align_to_roster

    roster = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_6", label="black fork")],
        observations=[_obs("Object_6", 18, 0.1, 0.1, 0.1, 0.1)],
    )
    later = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_9", label="separate screw")],
        observations=[_obs("Object_9", 21, 0.8, 0.8, 0.05, 0.05)],
    )
    aligned = align_to_roster(roster, later)
    assert {obj.local_id for obj in aligned.objects} == {"Object_6", "Object_9"}


def test_a_pixel_box_is_read_as_a_fraction_of_the_frame():
    from app.pipeline.locate_pass import _fraction_box

    pixels = _fraction_box({"x": 425, "y": 868, "w": 125, "h": 60}, 2160, 3840)
    assert pixels is not None
    assert abs(pixels.x - 425 / 2160) < 1e-6
    assert abs(pixels.y - 868 / 3840) < 1e-6
    fraction = _fraction_box({"x": 0.2, "y": 0.3, "w": 0.1, "h": 0.1}, 2160, 3840)
    assert fraction is not None
    assert abs(fraction.x - 0.2) < 1e-6


def test_a_follow_up_box_on_a_different_surface_is_rejected():
    import numpy as np
    from app.perception.locate import same_surface

    def solid(blue: int, green: int, red: int) -> np.ndarray:
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :] = (blue, green, red)
        return image

    gray = solid(190, 192, 194)
    assert same_surface(gray, solid(192, 197, 200))
    assert not same_surface(gray, solid(117, 64, 82))
    assert not same_surface(gray, solid(178, 201, 225))


def test_a_located_box_keeps_its_confidence_and_a_bad_box_is_still_rejected():
    from app.pipeline.locate_pass import _parse

    scored = _parse(
        '{"observations":[{"timestamp":6,"visibility":"visible","bbox":{"x":0.5,"y":0.2,"w":0.08,"h":0.025},"confidence":0.75}]}',
        {6.0},
    )
    assert scored == [(6.0, BBox(x=0.5, y=0.2, w=0.08, h=0.025), "visible", 0.75)]
    observations = [_obs("Object_1", float(t), 0.5, 0.2, 0.08, 0.025) for t in range(4)]
    document = apply_located(_doc(observations), "Object_1", scored)
    added = next(item for item in document.observations if item.timestamp == 6)
    assert added.confidence == 0.75

    glare = _parse(
        '{"observations":[{"timestamp":6,"visibility":"partial","bbox":{"x":0.5,"y":0.2,"w":0.08,"h":0.025},"confidence":0.61}]}',
        {6.0},
    )
    left_out = apply_located(_doc(observations), "Object_1", glare)
    assert all(item.timestamp != 6 for item in left_out.observations)

    at_floor = [(6.0, BBox(x=0.5, y=0.2, w=0.08, h=0.025), "visible", 0.70)]
    kept = apply_located(_doc(observations), "Object_1", at_floor)
    assert next(item for item in kept.observations if item.timestamp == 6).confidence == 0.70

    unscored = [(6.0, BBox(x=0.5, y=0.2, w=0.08, h=0.025), "visible")]
    still = apply_located(_doc(observations), "Object_1", unscored)
    assert next(item for item in still.observations if item.timestamp == 6).confidence is None

    too_big = _parse(
        '{"observations":[{"timestamp":6,"visibility":"visible","bbox":{"x":0.4,"y":0.4,"w":0.2,"h":0.15},"confidence":0.9}]}',
        {6.0},
    )
    rejected = apply_located(_doc(observations), "Object_1", too_big)
    assert all(item.timestamp != 6 for item in rejected.observations)

    absent = _parse(
        '{"observations":[{"timestamp":6,"visibility":"absent","bbox":null,"confidence":0.8}]}',
        {6.0},
    )
    assert absent[0][3] is None


def test_a_located_box_cannot_be_a_larger_part():
    assert box_fits(0.001, 0.002)
    assert not box_fits(0.03, 0.002)
    observations = [_obs("Object_1", float(t), 0.5, 0.2, 0.08, 0.025) for t in range(4)]
    document = apply_located(
        _doc(observations),
        "Object_1",
        [(6, BBox(x=0.4, y=0.4, w=0.2, h=0.15), "visible")],
    )
    assert all(item.timestamp != 6 for item in document.observations)
