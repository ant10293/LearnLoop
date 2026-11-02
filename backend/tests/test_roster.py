"""Object ids follow the previous window, through the frames those windows share."""

from app.perception.roster import align_to_roster, drop_contained_objects, omit_workspace_and_hands
from app.schemas.analysis import (
    ActionPrediction,
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
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
)


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


def _window(specs: list[tuple[str, str]], observations: list[ObservationPrediction]) -> AnalysisDocument:
    return AnalysisDocument(
        objects=[ObjectPrediction(local_id=object_id, label=label) for object_id, label in specs],
        observations=observations,
    )


def _chain(windows: list[AnalysisDocument]) -> list[AnalysisDocument]:
    aligned: list[AnalysisDocument] = []
    for window in windows:
        if aligned:
            window = align_to_roster(aligned[-1], window)
        aligned.append(window)
    return aligned


def _box(document: AnalysisDocument, object_id: str, timestamp: float) -> tuple[float, float, float, float]:
    found = [
        item
        for item in document.observations
        if item.object_local_id == object_id and abs(item.timestamp - timestamp) < 1e-6 and item.bbox is not None
    ]
    assert len(found) == 1
    box = found[0].bbox
    assert box is not None
    return (box.x, box.y, box.w, box.h)


def _ids(document: AnalysisDocument) -> set[str]:
    return {obj.local_id for obj in document.objects}


def test_three_windows_align_through_the_shared_boundary():
    rod = (0.10, 0.10, 0.16, 0.08)
    wheel = (0.10, 0.40, 0.16, 0.16)
    connector = (0.70, 0.70, 0.10, 0.10)
    # The parts move between boundaries. Window 3 never shares a frame with window 1.
    moved_rod = (0.62, 0.12, 0.16, 0.08)
    moved_wheel = (0.20, 0.62, 0.16, 0.16)
    moved_connector = (0.40, 0.20, 0.10, 0.10)
    window1 = _window(
        [("Object_1", "gray rod"), ("Object_2", "spoked wheel"), ("Object_3", "black connector")],
        [
            _obs("Object_1", 0, *rod),
            _obs("Object_2", 0, *wheel),
            _obs("Object_3", 0, *connector),
            _obs("Object_1", 1, *rod),
            _obs("Object_2", 1, *wheel),
            _obs("Object_3", 1, *connector),
        ],
    )
    window2 = _window(
        [("Object_1", "spoked wheel"), ("Object_2", "gray rod"), ("Object_3", "black connector")],
        [
            _obs("Object_2", 1, *rod),
            _obs("Object_1", 1, *wheel),
            _obs("Object_3", 1, *connector),
            _obs("Object_2", 2, *moved_rod),
            _obs("Object_1", 2, *moved_wheel),
            _obs("Object_3", 2, *moved_connector),
        ],
    )
    window3 = _window(
        [("Object_1", "black connector"), ("Object_2", "spoked wheel"), ("Object_3", "gray rod")],
        [
            _obs("Object_3", 2, *moved_rod),
            _obs("Object_2", 2, *moved_wheel),
            _obs("Object_1", 2, *moved_connector),
            _obs("Object_3", 3, *moved_rod),
            _obs("Object_2", 3, *moved_wheel),
            _obs("Object_1", 3, *moved_connector),
        ],
    )

    aligned = _chain([window1, window2, window3])
    assert _box(aligned[1], "Object_1", 2) == moved_rod
    assert _box(aligned[1], "Object_2", 2) == moved_wheel
    assert _box(aligned[1], "Object_3", 2) == moved_connector
    assert _box(aligned[2], "Object_1", 2) == moved_rod
    assert _box(aligned[2], "Object_2", 2) == moved_wheel
    assert _box(aligned[2], "Object_3", 2) == moved_connector
    assert _box(aligned[2], "Object_1", 3) == moved_rod
    assert _ids(aligned[2]) == {"Object_1", "Object_2", "Object_3"}

    # The same window against the first window has no shared frame to follow.
    direct = align_to_roster(window1, window3)
    direct_rod = {
        item.object_local_id
        for item in direct.observations
        if item.bbox is not None and abs(item.timestamp - 2) < 1e-6 and abs(item.bbox.x - moved_rod[0]) < 1e-6
    }
    assert direct_rod != {"Object_1"}


def test_identity_propagates_across_four_windows_without_resetting_ids():
    places = [
        (0.05, 0.10, 0.12, 0.08),
        (0.40, 0.10, 0.12, 0.08),
        (0.70, 0.10, 0.12, 0.08),
        (0.40, 0.55, 0.12, 0.08),
        (0.05, 0.55, 0.12, 0.08),
    ]
    wheel_places = [(x, y + 0.25, w, 0.12) for x, y, w, _h in places]
    raw_ids = [("Object_1", "Object_2"), ("Object_2", "Object_1"), ("Object_1", "Object_2"), ("Object_2", "Object_1")]
    windows = []
    for index, (rod_id, wheel_id) in enumerate(raw_ids):
        start = places[index]
        end = places[index + 1]
        wheel_start = wheel_places[index]
        wheel_end = wheel_places[index + 1]
        windows.append(
            _window(
                [(rod_id, "gray rod"), (wheel_id, "spoked wheel")],
                [
                    _obs(rod_id, index, *start),
                    _obs(wheel_id, index, *wheel_start),
                    _obs(rod_id, index + 1, *end),
                    _obs(wheel_id, index + 1, *wheel_end),
                ],
            )
        )

    aligned = _chain(windows)
    assert len(aligned) == 4
    for window in aligned:
        assert _ids(window) == {"Object_1", "Object_2"}
        assert len(window.objects) == len(_ids(window))
    assert _box(aligned[3], "Object_1", 4) == places[4]
    assert _box(aligned[3], "Object_2", 4) == wheel_places[4]
    assert _box(aligned[0], "Object_1", 0) == places[0]


def test_an_object_found_in_the_second_window_keeps_a_new_global_id():
    wheel = (0.10, 0.10, 0.16, 0.16)
    rod = (0.50, 0.10, 0.20, 0.08)
    connector = (0.80, 0.75, 0.08, 0.08)
    window1 = _window(
        [("Object_1", "spoked wheel"), ("Object_2", "gray rod")],
        [_obs("Object_1", 1, *wheel), _obs("Object_2", 1, *rod)],
    )
    window2 = _window(
        [("Object_1", "spoked wheel"), ("Object_2", "gray rod"), ("Object_3", "small connector")],
        [
            _obs("Object_1", 1, *wheel),
            _obs("Object_2", 1, *rod),
            _obs("Object_3", 1, *connector),
            _obs("Object_1", 2, *wheel),
            _obs("Object_2", 2, *rod),
            _obs("Object_3", 2, *connector),
        ],
    )
    window3 = _window(
        [("Object_1", "small connector"), ("Object_2", "spoked wheel"), ("Object_3", "gray rod")],
        [
            _obs("Object_2", 2, *wheel),
            _obs("Object_3", 2, *rod),
            _obs("Object_1", 2, *connector),
            _obs("Object_2", 3, *wheel),
            _obs("Object_3", 3, *rod),
            _obs("Object_1", 3, *connector),
        ],
    )

    aligned = _chain([window1, window2, window3])
    assert _box(aligned[1], "Object_3", 2) == connector
    assert _box(aligned[2], "Object_3", 2) == connector
    assert _box(aligned[2], "Object_3", 3) == connector
    assert _box(aligned[2], "Object_1", 3) == wheel
    assert _box(aligned[2], "Object_2", 3) == rod
    assert _ids(aligned[2]) == {"Object_1", "Object_2", "Object_3"}


def test_a_rotated_part_keeps_its_id_when_the_shared_frame_still_overlaps():
    before = (0.40, 0.40, 0.20, 0.20)
    after = (0.41, 0.41, 0.18, 0.18)
    roster = _window(
        [("Object_1", "gray rod")],
        [_obs("Object_1", 1, *before)],
    )
    later = _window(
        [("Object_2", "gray rod end-on")],
        [_obs("Object_2", 1, *after), _obs("Object_2", 2, *after)],
    )

    aligned = align_to_roster(roster, later)
    assert _ids(aligned) == {"Object_1"}
    assert _box(aligned, "Object_1", 1) == after
    assert _box(aligned, "Object_1", 2) == after


def test_touching_parts_stay_distinct_and_a_tie_is_not_a_match():
    rod = (0.10, 0.20, 0.20, 0.20)
    wheel = (0.28, 0.20, 0.20, 0.20)
    roster = _window(
        [("Object_1", "small black rod"), ("Object_2", "small black wheel")],
        [_obs("Object_1", 1, *rod), _obs("Object_2", 1, *wheel)],
    )
    later = _window(
        [("Object_2", "small black rod"), ("Object_1", "small black wheel"), ("Object_3", "small black stud")],
        [
            _obs("Object_2", 1, *rod),
            _obs("Object_1", 1, *wheel),
            _obs("Object_2", 2, *rod),
            _obs("Object_1", 2, *wheel),
            # Sits on both parts and shares a word with both names.
            _obs("Object_3", 2, 0.20, 0.20, 0.10, 0.10),
        ],
    )

    aligned = align_to_roster(roster, later)
    assert _box(aligned, "Object_1", 2) == rod
    assert _box(aligned, "Object_2", 2) == wheel
    studs = [item for item in aligned.observations if abs(item.timestamp - 2) < 1e-6 and item.bbox is not None and abs(item.bbox.w - 0.10) < 1e-6]
    assert len(studs) == 1
    assert studs[0].object_local_id not in {"Object_1", "Object_2"}
    assert studs[0].bbox is not None
    assert (studs[0].bbox.x, studs[0].bbox.y, studs[0].bbox.w, studs[0].bbox.h) == (0.20, 0.20, 0.10, 0.10)


def test_a_part_hidden_on_one_frame_keeps_its_id_when_it_is_seen_again():
    full = (0.20, 0.20, 0.20, 0.20)
    partial = (0.20, 0.20, 0.20, 0.12)
    other = (0.70, 0.70, 0.10, 0.10)
    elsewhere = (0.10, 0.70, 0.10, 0.10)
    window1 = _window(
        [("Object_1", "gray rod"), ("Object_2", "clear stud")],
        [_obs("Object_1", 1, *full), _obs("Object_2", 1, *other)],
    )
    window2 = _window(
        [("Object_1", "gray rod"), ("Object_2", "clear stud")],
        [
            _obs("Object_1", 1, *partial),
            _obs("Object_2", 1, *other),
            # Fully hidden at t=2: no observation and no invented box.
            _obs("Object_1", 3, *partial),
            _obs("Object_2", 3, *other),
        ],
    )
    window3 = _window(
        [("Object_1", "gray rod"), ("Object_2", "clear stud")],
        [
            _obs("Object_1", 3, *partial),
            _obs("Object_2", 3, *elsewhere),
            _obs("Object_1", 4, *full),
            _obs("Object_2", 4, *elsewhere),
        ],
    )

    aligned = _chain([window1, window2, window3])
    assert all(abs(item.timestamp - 2) > 1e-6 for item in aligned[1].observations)
    assert _box(aligned[2], "Object_1", 3) == partial
    assert _box(aligned[2], "Object_1", 4) == full
    stray = [item for item in aligned[2].observations if abs(item.timestamp - 3) < 1e-6 and item.bbox is not None and abs(item.bbox.x - 0.10) < 1e-6]
    assert len(stray) == 1
    assert stray[0].object_local_id not in {"Object_1", "Object_2"}
    assert stray[0].bbox is not None
    assert (stray[0].bbox.x, stray[0].bbox.y, stray[0].bbox.w, stray[0].bbox.h) == elsewhere


def test_remapped_references_follow_the_same_global_id():
    rod = (0.10, 0.10, 0.20, 0.08)
    wheel = (0.50, 0.40, 0.16, 0.16)
    roster = _window(
        [("Object_1", "gray rod"), ("Object_2", "spoked wheel")],
        [_obs("Object_1", 1, *rod), _obs("Object_2", 1, *wheel)],
    )
    evidence = _evidence(1)
    later = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_1", label="spoked wheel"),
            ObjectPrediction(local_id="Object_2", label="gray rod"),
        ],
        observations=[_obs("Object_1", 1, *wheel), _obs("Object_2", 1, *rod)],
        states=[
            StatePrediction(
                object_local_id="Object_1",
                time_start=1,
                time_end=2,
                state_label="held",
                evidence=evidence,
            )
        ],
        transitions=[
            TransitionPrediction(
                object_local_id="Object_2",
                timestamp=1,
                from_state="resting",
                to_state="held",
                evidence=evidence,
            )
        ],
        relationships=[
            RelationshipPrediction(
                subject_local_id="Object_1",
                other_local_id="Object_2",
                kind=RelationKind.TOUCHING,
                time_start=1,
                time_end=2,
                evidence=evidence,
            )
        ],
        actions=[
            ActionPrediction(
                local_id="Action_1",
                action_type="hold",
                start=1,
                end=2,
                object_local_ids=["Object_1", "Object_2"],
                evidence=evidence,
                outcome=ActionOutcome.COMPLETED,
            )
        ],
        workflow_steps=[
            WorkflowStepPrediction(
                local_id="Step_1",
                name="seat the wheel",
                required_objects=["Object_2", "Object_1"],
                evidence=evidence,
            )
        ],
        ambiguities=[
            AmbiguityPrediction(
                local_id="Ambiguity_1",
                kind=AmbiguityKind.OCCLUSION,
                description="The rod is partly covered.",
                object_local_ids=["Object_2"],
                evidence=evidence,
            )
        ],
        dependencies=[
            DependencyPrediction(
                from_step="Step_1",
                to_step="Step_2",
                category=DependencyCategory.OBSERVED_TEMPORAL,
                evidence=evidence,
            )
        ],
    )

    aligned = align_to_roster(roster, later)
    assert _box(aligned, "Object_1", 1) == rod
    assert _box(aligned, "Object_2", 1) == wheel
    assert aligned.states[0].object_local_id == "Object_2"
    assert aligned.transitions[0].object_local_id == "Object_1"
    assert aligned.relationships[0].subject_local_id == "Object_2"
    assert aligned.relationships[0].other_local_id == "Object_1"
    assert aligned.actions[0].object_local_ids == ["Object_2", "Object_1"]
    assert aligned.workflow_steps[0].required_objects == ["Object_1", "Object_2"]
    assert aligned.ambiguities[0].object_local_ids == ["Object_1"]
    assert aligned.dependencies[0].from_step == "Step_1"
    assert aligned.dependencies[0].to_step == "Step_2"
    known = _ids(aligned)
    referenced = {item.object_local_id for item in aligned.observations}
    referenced.update(item.object_local_id for item in aligned.states)
    referenced.update(item.object_local_id for item in aligned.transitions)
    for item in aligned.relationships:
        referenced.add(item.subject_local_id)
        if item.other_local_id:
            referenced.add(item.other_local_id)
    for item in aligned.actions:
        referenced.update(item.object_local_ids)
    for item in aligned.workflow_steps:
        referenced.update(item.required_objects)
    for item in aligned.ambiguities:
        referenced.update(item.object_local_ids)
    assert referenced <= known


def test_a_shifted_box_is_not_absorbed_by_the_part_it_touches():
    frame = (0.40, 0.40, 0.20, 0.20)
    stud = (0.50, 0.50, 0.04, 0.02)
    shifted = (0.51, 0.49, 0.04, 0.03)
    roster = _window(
        [("Object_1", "small clear stud"), ("Object_2", "small gray frame")],
        [_obs("Object_1", 1, *stud), _obs("Object_2", 1, *frame)],
    )
    later = _window(
        [("Object_1", "small clear stud"), ("Object_2", "small gray frame")],
        [_obs("Object_1", 1, *shifted), _obs("Object_2", 1, *frame)],
    )

    aligned = align_to_roster(roster, later)
    assert _box(aligned, "Object_1", 1) == shifted
    assert _box(aligned, "Object_2", 1) == frame
    assert _ids(aligned) == {"Object_1", "Object_2"}


def test_a_box_that_never_leaves_another_object_is_not_shown_separately():
    tire = (0.50, 0.29, 0.08, 0.05)
    spoke = (0.52, 0.30, 0.05, 0.03)
    document = _window(
        [("Object_3", "black tire"), ("Object_4", "spoked wheel center")],
        [
            _obs("Object_3", 0, *tire),
            _obs("Object_4", 0, *spoke),
            _obs("Object_3", 1, *tire),
            _obs("Object_4", 1, *spoke),
        ],
    )
    kept = drop_contained_objects(document)
    assert _ids(kept) == {"Object_3"}
    assert _box(kept, "Object_3", 0) == tire
    assert all(item.object_local_id != "Object_4" for item in kept.observations)
    assert any("never seen apart" in item.description for item in kept.ambiguities)


def test_a_part_on_the_table_is_not_swallowed_by_the_table():
    surface = (0.0, 0.0, 0.99, 0.70)
    body = (0.40, 0.30, 0.12, 0.06)
    document = _window(
        [("Object_1", "light gray body"), ("Object_7", "purple flat support surface")],
        [
            _obs("Object_1", 0, *body),
            _obs("Object_7", 0, *surface),
            _obs("Object_1", 1, *body),
            _obs("Object_7", 1, *surface),
        ],
    )
    kept = drop_contained_objects(document)
    assert _ids(kept) == {"Object_1", "Object_7"}
    assert _box(kept, "Object_1", 0) == body


def test_hands_and_the_resting_surface_are_left_out():
    surface = (0.0, 0.0, 0.99, 0.70)
    body = (0.40, 0.30, 0.12, 0.06)
    hand = (0.20, 0.20, 0.15, 0.12)
    base = (0.55, 0.40, 0.20, 0.10)
    document = _window(
        [
            ("Object_1", "light gray body"),
            ("Object_7", "purple flat support surface"),
            ("Object_8", "left hand"),
            ("Object_9", "wooden base"),
        ],
        [
            _obs("Object_1", 0, *body),
            _obs("Object_7", 0, *surface),
            _obs("Object_8", 0, *hand),
            _obs("Object_9", 0, *base),
        ],
    )
    document = document.model_copy(
        update={
            "objects": [
                document.objects[0].model_copy(update={"manipulated": True, "initial_state": "Resting on the purple surface"}),
                document.objects[1],
                document.objects[2],
                document.objects[3],
            ]
        }
    )
    kept = omit_workspace_and_hands(document)
    assert _ids(kept) == {"Object_1", "Object_9"}
    assert _box(kept, "Object_1", 0) == body
    assert all(item.object_local_id != "Object_7" for item in kept.observations)
    assert all(item.object_local_id != "Object_8" for item in kept.observations)


def test_a_part_seen_apart_before_it_is_inside_stays_its_own_object():
    body = (0.40, 0.40, 0.30, 0.20)
    apart = (0.10, 0.10, 0.04, 0.04)
    inside = (0.50, 0.48, 0.04, 0.04)
    document = _window(
        [("Object_1", "gray body"), ("Object_7", "clear stud")],
        [
            _obs("Object_1", 0, *body),
            _obs("Object_7", 0, *apart),
            _obs("Object_1", 1, *body),
            _obs("Object_7", 1, *inside),
        ],
    )
    kept = drop_contained_objects(document)
    assert _ids(kept) == {"Object_1", "Object_7"}


def test_a_large_still_box_is_the_surface_whatever_it_is_called():
    lid = (0.0, 0.0, 1.0, 0.69)
    part = (0.40, 0.40, 0.10, 0.10)
    document = _window(
        [("Object_1", "purple flat lid or board"), ("Object_2", "gray body")],
        [item for time in (0, 1, 2) for item in (_obs("Object_1", time, *lid), _obs("Object_2", time, *part))],
    )
    assert _ids(omit_workspace_and_hands(document)) == {"Object_2"}


def test_a_large_part_that_moves_is_not_the_surface():
    document = _window(
        [("Object_1", "large gray panel")],
        [_obs("Object_1", time, 0.05 + 0.15 * time, 0.1, 0.55, 0.55) for time in (0, 1, 2)],
    )
    assert _ids(omit_workspace_and_hands(document)) == {"Object_1"}


def test_a_small_mounted_base_stays():
    document = _window(
        [("Object_1", "wooden base")],
        [_obs("Object_1", time, 0.3, 0.3, 0.3, 0.3) for time in (0, 1, 2)],
    )
    assert _ids(omit_workspace_and_hands(document)) == {"Object_1"}

