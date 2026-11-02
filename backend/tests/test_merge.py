"""Window merges must not drop states, transitions, or relationships."""

import json

from app.perception.assignment import iou
from app.pipeline.merge import restore_omitted_records, retain_prior_records
from app.schemas.analysis import AnalysisDocument
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance


def _evidence():
    return {
        "classification": EvidenceClass.DIRECTLY_OBSERVED.value,
        "provenance": Provenance.AI_PREDICTION.value,
        "frame_timestamps": [0.0],
    }


def _doc(**fields) -> AnalysisDocument:
    return AnalysisDocument.model_validate(fields)


def test_empty_merge_keeps_window_records_and_rewrites_reused_ids():
    first = _doc(
        objects=[{"local_id": "Object_1", "label": "threaded bolt"}, {"local_id": "Object_3", "label": "first washer"}],
        states=[
            {
                "object_local_id": "Object_1",
                "time_start": 0,
                "time_end": 1,
                "state_label": "on the surface",
                "evidence": _evidence(),
            }
        ],
        transitions=[
            {
                "object_local_id": "Object_1",
                "timestamp": 1,
                "from_state": "on the surface",
                "to_state": "held",
                "evidence": _evidence(),
            }
        ],
        relationships=[
            {
                "subject_local_id": "Object_3",
                "other_local_id": "Object_1",
                "kind": "near",
                "time_start": 0,
                "time_end": 1,
                "description": "washer near the bolt",
                "evidence": _evidence(),
            }
        ],
        workflow_steps=[
            {
                "local_id": "Step_1",
                "name": "Acquire bolt",
                "required_objects": ["Object_1"],
                "evidence": _evidence(),
            }
        ],
    )
    second = _doc(
        objects=[{"local_id": "Object_1", "label": "threaded bolt"}, {"local_id": "Object_3", "label": "white work surface"}],
        states=[
            {
                "object_local_id": "Object_3",
                "time_start": 20,
                "time_end": 21,
                "state_label": "supporting the work",
                "evidence": _evidence(),
            }
        ],
        transitions=[
            {
                "object_local_id": "Object_1",
                "timestamp": 21,
                "from_state": "held",
                "to_state": "threaded",
                "evidence": _evidence(),
            }
        ],
        relationships=[
            {
                "subject_local_id": "Object_3",
                "other_local_id": "Object_1",
                "kind": "supported_by",
                "time_start": 20,
                "time_end": 22,
                "description": "bolt above the surface",
                "evidence": _evidence(),
            }
        ],
        workflow_steps=[
            {
                "local_id": "Step_1",
                "name": "Position nut",
                "required_objects": ["Object_3"],
                "evidence": _evidence(),
            }
        ],
        dependencies=[
            {"from_step": "Step_1", "to_step": "Step_1", "category": "observed_temporal", "evidence": _evidence()}
        ],
    )
    merged = _doc(
        objects=[
            {"local_id": "Object_1", "label": "threaded bolt"},
            {"local_id": "Object_3", "label": "first washer"},
            {"local_id": "Object_8", "label": "white work surface"},
        ],
        actions=[{"local_id": "Action_1", "action_type": "pick_up", "start": 0, "end": 1, "evidence": _evidence()}],
    )

    restored = restore_omitted_records(merged, [first, second])

    assert [item.object_local_id for item in restored.states] == ["Object_1", "Object_8"]
    assert [item.timestamp for item in restored.transitions] == [1, 21]
    assert restored.relationships[0].subject_local_id == "Object_3"
    assert restored.relationships[1].subject_local_id == "Object_8"
    assert [item.local_id for item in restored.workflow_steps] == ["Step_1", "Step_2"]
    assert restored.workflow_steps[1].required_objects == ["Object_8"]
    assert restored.dependencies[0].from_step == "Step_2"
    assert restored.dependencies[0].to_step == "Step_2"
    assert restored.actions[0].local_id == "Action_1"


def test_merge_that_already_has_records_is_left_in_place():
    partial = _doc(
        objects=[{"local_id": "Object_1", "label": "bolt"}],
        transitions=[
            {
                "object_local_id": "Object_1",
                "timestamp": 1,
                "from_state": "a",
                "to_state": "b",
                "evidence": _evidence(),
            }
        ],
    )
    merged = _doc(
        objects=[{"local_id": "Object_1", "label": "bolt"}],
        transitions=[
            {
                "object_local_id": "Object_1",
                "timestamp": 3,
                "from_state": "kept",
                "to_state": "kept",
                "evidence": _evidence(),
            }
        ],
    )
    restored = restore_omitted_records(merged, [partial, partial])
    assert [item.timestamp for item in restored.transitions] == [3]


def test_a_renumbered_window_does_not_replace_the_merged_part():
    first = _doc(
        objects=[
            {"local_id": "Object_1", "label": "light-gray bar"},
            {"local_id": "Object_2", "label": "spoked wheel"},
        ],
        observations=[
            {
                "object_local_id": "Object_1",
                "timestamp": 14,
                "bbox": {"x": 0.5, "y": 0.2, "w": 0.04, "h": 0.04},
                "visibility": "visible",
                "evidence": _evidence(),
            }
        ],
    )
    second = _doc(
        objects=[
            {"local_id": "Object_1", "label": "toy motorcycle"},
            {"local_id": "Object_2", "label": "loose light-grey bar"},
        ],
        observations=[
            {
                "object_local_id": "Object_2",
                "timestamp": 15,
                "bbox": {"x": 0.6, "y": 0.2, "w": 0.02, "h": 0.02},
                "visibility": "partial",
                "description": "grey bar",
                "evidence": _evidence(),
            },
            {
                "object_local_id": "Object_1",
                "timestamp": 15,
                "bbox": {"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.1},
                "visibility": "visible",
                "description": "whole motorcycle",
                "evidence": _evidence(),
            },
        ],
    )
    merged = _doc(
        objects=[
            {"local_id": "Object_1", "label": "light-gray bar"},
            {"local_id": "Object_2", "label": "spoked wheel"},
            {"local_id": "Object_4", "label": "light-gray body"},
        ],
        observations=[
            {
                "object_local_id": "Object_1",
                "timestamp": 15,
                "bbox": {"x": 0.6, "y": 0.2, "w": 0.02, "h": 0.02},
                "visibility": "partial",
                "description": "the bar",
                "evidence": _evidence(),
            },
            {
                "object_local_id": "Object_4",
                "timestamp": 15,
                "bbox": {"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.1},
                "visibility": "visible",
                "description": "assembly",
                "evidence": _evidence(),
            },
        ],
    )
    restored = restore_omitted_records(merged, [first, second])
    by_object = {(item.object_local_id, item.timestamp): item for item in restored.observations}
    assert by_object[("Object_1", 15)].description == "the bar"
    assert ("Object_2", 15) not in by_object
    assert by_object[("Object_4", 15)].description == "assembly"
    assert ("Object_1", 14) in by_object


def test_merge_keeps_a_box_the_window_saw_when_the_merge_omits_that_frame():
    boxed = {
        "object_local_id": "Object_2",
        "timestamp": 12,
        "bbox": {"x": 0.5, "y": 0.4, "w": 0.04, "h": 0.04},
        "visibility": "partial",
        "evidence": _evidence(),
    }
    first = _doc(
        objects=[{"local_id": "Object_1", "label": "threaded fastener"}, {"local_id": "Object_2", "label": "dark ring"}],
        observations=[
            boxed,
            {
                "object_local_id": "Object_1",
                "timestamp": 12,
                "visibility": "visible",
                "evidence": _evidence(),
            },
        ],
    )
    second = _doc(
        objects=[{"local_id": "Object_1", "label": "threaded fastener"}, {"local_id": "Object_2", "label": "dark ring"}],
        observations=[
            {
                "object_local_id": "Object_1",
                "timestamp": 18,
                "bbox": {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.1},
                "visibility": "visible",
                "evidence": _evidence(),
            }
        ],
    )
    merged = _doc(
        objects=[{"local_id": "Object_1", "label": "threaded fastener"}, {"local_id": "Object_2", "label": "dark ring"}],
        observations=[
            {
                "object_local_id": "Object_1",
                "timestamp": 12,
                "bbox": {"x": 0.42, "y": 0.4, "w": 0.2, "h": 0.12},
                "visibility": "visible",
                "evidence": _evidence(),
            }
        ],
    )
    restored = restore_omitted_records(merged, [first, second])
    by_object = {(item.object_local_id, item.timestamp): item for item in restored.observations}
    assert by_object[("Object_2", 12)].bbox is not None
    assert by_object[("Object_1", 12)].bbox is not None
    assert by_object[("Object_1", 18)].bbox is not None


ROD = {"x": 0.342, "y": 0.41, "w": 0.053, "h": 0.035}
UPPER = {"x": 0.44, "y": 0.367, "w": 0.065, "h": 0.028}


def _box(values):
    return {"x": values["x"], "y": values["y"], "w": values["w"], "h": values["h"]}


def _observation(bbox, classification, visibility="partial", note="", description=""):
    return {
        "object_local_id": "Object_1",
        "timestamp": 16.902,
        "bbox": None if bbox is None else _box(bbox),
        "visibility": visibility,
        "description": description,
        "evidence": {
            "classification": classification.value,
            "provenance": Provenance.AI_PREDICTION.value,
            "note": note,
            "frame_timestamps": [16.902],
        },
    }


def _window(observations, uncertain=False):
    return _doc(
        objects=[{"local_id": "Object_1", "label": "light-gray rod", "uncertain_identity": uncertain}],
        observations=observations,
    )


def _at(document):
    return {(item.object_local_id, item.timestamp): item for item in document.observations}


def _conflict_payloads(document):
    payloads = []
    for item in document.ambiguities:
        if item.kind != AmbiguityKind.CONFLICTING_EVIDENCE:
            continue
        payloads.append(json.loads(item.evidence.note))
    return payloads


def test_reported_boundary_keeps_the_image_backed_rod():
    """Window 2 saw the rod. The text-only merge must not replace it with window 3's box."""
    correct = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)
    wrong = _observation(
        UPPER,
        EvidenceClass.INFERRED,
        note="grey rod ends at top, identity uncertain",
    )
    assert iou((ROD["x"], ROD["y"], ROD["w"], ROD["h"]), (UPPER["x"], UPPER["y"], UPPER["w"], UPPER["h"])) == 0
    windows = [
        _window([]),
        _window([correct]),
        _window([wrong], uncertain=True),
    ]
    merged = _window([
        _observation(UPPER, EvidenceClass.INFERRED, note="window 3 grey rod-like feature at the top; identity uncertain")
    ])
    before = [window.model_dump() for window in windows]
    merged_before = merged.model_dump()

    restored = restore_omitted_records(merged, windows)
    again = restore_omitted_records(merged, windows)

    kept = _at(restored)[("Object_1", 16.902)]
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert kept.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    assert [window.model_dump() for window in windows] == before
    assert merged.model_dump() == merged_before
    assert again.model_dump() == restored.model_dump()
    payload = _conflict_payloads(restored)[0]
    assert payload["object_local_id"] == "Object_1"
    assert payload["timestamp"] == 16.902
    assert payload["candidate_count"] == 2
    assert payload["unresolved"] is False
    assert payload["selected"]["window_index"] == 1
    assert payload["selected"]["bbox"] == [ROD["x"], ROD["y"], ROD["w"], ROD["h"]]
    assert payload["selected"]["evidence"] == EvidenceClass.DIRECTLY_OBSERVED.value
    assert [item["bbox"] for item in payload["candidates"]] == [
        [ROD["x"], ROD["y"], ROD["w"], ROD["h"]],
        [UPPER["x"], UPPER["y"], UPPER["w"], UPPER["h"]],
    ]
    assert payload["candidates"][1]["identity_uncertain"] is True
    assert "directly observed" in payload["reason"].lower()


def test_text_only_merge_cannot_override_an_image_box():
    image = _observation(ROD, EvidenceClass.INFERRED, note="seen on the frame")
    text = _observation(UPPER, EvidenceClass.DIRECTLY_OBSERVED, note="merge copy")
    restored = restore_omitted_records(_window([text]), [_window([image]), _window([])])
    kept = _at(restored)[("Object_1", 16.902)]
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert kept.evidence.classification == EvidenceClass.INFERRED
    assert _conflict_payloads(restored) == []


def test_direct_observation_beats_a_conflicting_inference():
    inferred = _observation(UPPER, EvidenceClass.INFERRED)
    direct = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)
    restored = restore_omitted_records(_window([]), [_window([inferred]), _window([direct])])
    kept = _at(restored)[("Object_1", 16.902)]
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert kept.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    payload = _conflict_payloads(restored)[0]
    assert payload["unresolved"] is False
    assert payload["selected"]["evidence"] == EvidenceClass.DIRECTLY_OBSERVED.value


def test_equally_supported_contradictions_stay_unresolved():
    first = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED, description="rod")
    second = _observation(UPPER, EvidenceClass.DIRECTLY_OBSERVED, description="other part")
    restored = restore_omitted_records(_window([]), [_window([first]), _window([second])])
    again = restore_omitted_records(_window([]), [_window([first]), _window([second])])
    kept = _at(restored)[("Object_1", 16.902)]
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert kept.description == "rod"
    assert kept.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    assert again.model_dump() == restored.model_dump()
    payload = _conflict_payloads(restored)[0]
    assert payload["unresolved"] is True
    assert "unresolved" in payload["reason"].lower()
    assert "conclus" not in payload["reason"].lower()
    selected = tuple(payload["selected"]["bbox"])
    assert selected in {
        (ROD["x"], ROD["y"], ROD["w"], ROD["h"]),
        (UPPER["x"], UPPER["y"], UPPER["w"], UPPER["h"]),
    }
    midpoint = ((ROD["x"] + UPPER["x"]) / 2, (ROD["y"] + UPPER["y"]) / 2)
    assert (kept.bbox.x, kept.bbox.y) != midpoint


def test_agreeing_boxes_do_not_raise_a_conflict():
    first = _observation(
        {"x": 0.50, "y": 0.40, "w": 0.20, "h": 0.20},
        EvidenceClass.DIRECTLY_OBSERVED,
        description="rod",
    )
    second = _observation(
        {"x": 0.51, "y": 0.41, "w": 0.19, "h": 0.19},
        EvidenceClass.DIRECTLY_OBSERVED,
        description="shifted",
    )
    left = (0.50, 0.40, 0.20, 0.20)
    right = (0.51, 0.41, 0.19, 0.19)
    assert iou(left, right) >= 0.5
    restored = restore_omitted_records(_window([]), [_window([first]), _window([second])])
    rows = [item for item in restored.observations if item.timestamp == 16.902]
    assert len(rows) == 1
    assert rows[0].description == "rod"
    assert rows[0].object_local_id == "Object_1"
    assert rows[0].evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    assert (rows[0].bbox.x, rows[0].bbox.y, rows[0].bbox.w, rows[0].bbox.h) == left
    assert _conflict_payloads(restored) == []


def test_a_later_box_fills_a_missing_one_and_absence_is_not_silent():
    missing = _observation(None, EvidenceClass.DIRECTLY_OBSERVED, visibility="visible")
    boxed = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)
    filled = restore_omitted_records(_window([]), [_window([missing]), _window([boxed])])
    assert _at(filled)[("Object_1", 16.902)].bbox is not None
    assert _conflict_payloads(filled) == []

    absent = _observation(None, EvidenceClass.DIRECTLY_OBSERVED, visibility="absent")
    blocked = restore_omitted_records(_window([]), [_window([boxed]), _window([absent])])
    kept = _at(blocked)[("Object_1", 16.902)]
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert _conflict_payloads(blocked)[0]["unresolved"] is True

    inferred_box = _observation(UPPER, EvidenceClass.INFERRED)
    absence_wins = restore_omitted_records(_window([]), [_window([absent]), _window([inferred_box])])
    winner = _at(absence_wins)[("Object_1", 16.902)]
    assert winner.bbox is None
    assert winner.visibility.value == "absent"
    assert winner.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    recorded = _conflict_payloads(absence_wins)[0]
    assert recorded["unresolved"] is False
    assert recorded["selected"]["visibility"] == "absent"

    inferred_absence = _observation(None, EvidenceClass.INFERRED, visibility="absent")
    box_wins = restore_omitted_records(_window([]), [_window([inferred_absence]), _window([boxed])])
    recovered = _at(box_wins)[("Object_1", 16.902)]
    assert (recovered.bbox.x, recovered.bbox.y, recovered.bbox.w, recovered.bbox.h) == (
        ROD["x"],
        ROD["y"],
        ROD["w"],
        ROD["h"],
    )
    assert _conflict_payloads(box_wins)[0]["unresolved"] is False


def test_merge_only_observation_is_kept():
    only = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED, description="from the merge")
    restored = restore_omitted_records(_window([only]), [_window([]), _window([])])
    kept = _at(restored)[("Object_1", 16.902)]
    assert kept.description == "from the merge"
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (ROD["x"], ROD["y"], ROD["w"], ROD["h"])
    assert _conflict_payloads(restored) == []

    earlier = _observation(ROD, EvidenceClass.DIRECTLY_OBSERVED, description="first copy")
    later = _observation(UPPER, EvidenceClass.INFERRED, description="later copy")
    duplicate = _doc(
        objects=[{"local_id": "Object_1", "label": "light-gray rod"}],
        observations=[earlier, later],
    )
    restored = restore_omitted_records(duplicate, [_window([]), _window([])])
    kept = _at(restored)[("Object_1", 16.902)]
    assert kept.description == "later copy"
    assert (kept.bbox.x, kept.bbox.y, kept.bbox.w, kept.bbox.h) == (UPPER["x"], UPPER["y"], UPPER["w"], UPPER["h"])


def _action(local_id: str = "Action_1") -> dict:
    return {
        "local_id": local_id,
        "action_type": "pick_up",
        "start": 1,
        "end": 2,
        "object_local_ids": ["Object_1"],
        "evidence": _evidence(),
    }


def test_a_later_stage_keeps_objects_actions_and_boxes_it_omitted():
    prior = _doc(
        objects=[{"local_id": "Object_1", "label": "light-gray rod"}],
        observations=[_observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)],
        actions=[_action()],
    )
    kept = retain_prior_records(_doc(), prior)
    assert [item.local_id for item in kept.objects] == ["Object_1"]
    assert [item.local_id for item in kept.actions] == ["Action_1"]
    assert kept.observations[0].bbox.x == ROD["x"]


def test_a_later_stage_keeps_scene_boxes_when_the_new_ids_do_not_match():
    prior = _doc(
        objects=[{"local_id": "Object_1", "label": "light-gray rod"}],
        observations=[_observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)],
        actions=[_action()],
    )
    renamed = dict(_observation(UPPER, EvidenceClass.INFERRED))
    renamed["object_local_id"] = "obj_body_gray"
    incoming = _doc(
        objects=[{"local_id": "obj_body_gray", "label": "light-gray elongated body"}],
        observations=[renamed],
        actions=[_action("Action_9")],
    )
    kept = retain_prior_records(incoming, prior)
    assert [item.local_id for item in kept.objects] == ["Object_1"]
    assert kept.observations[0].object_local_id == "Object_1"
    assert kept.observations[0].bbox.x == ROD["x"]
    assert [item.local_id for item in kept.actions] == ["Action_9"]


def test_a_later_stage_keeps_a_slice_that_uses_the_scene_ids():
    prior = _doc(
        objects=[{"local_id": "Object_1", "label": "light-gray rod"}],
        observations=[_observation(ROD, EvidenceClass.DIRECTLY_OBSERVED)],
    )
    incoming = _doc(actions=[_action()])
    kept = retain_prior_records(incoming, prior)
    assert [item.local_id for item in kept.objects] == ["Object_1"]
    assert kept.observations[0].bbox.w == ROD["w"]
    assert [item.local_id for item in kept.actions] == ["Action_1"]
