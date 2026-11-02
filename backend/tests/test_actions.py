"""Same verb on the same clock interval is one action."""

from app.pipeline.actions import collapse_repeated_actions
from app.schemas.analysis import AnalysisDocument
from app.schemas.enums import ActionOutcome, EvidenceClass


def _evidence(kind: str = "DIRECTLY_OBSERVED"):
    return {
        "classification": kind,
        "provenance": "AI_PREDICTION",
        "frame_timestamps": [1.0],
        "note": kind,
    }


def _action(local_id: str, action_type: str, start: float, end: float, objects: list[str], **extra):
    payload = {
        "local_id": local_id,
        "action_type": action_type,
        "description": extra.pop("description", local_id),
        "start": start,
        "end": end,
        "object_local_ids": objects,
        "outcome": extra.pop("outcome", "completed"),
        "uncertainty": extra.pop("uncertainty", 0.4),
        "evidence": extra.pop("evidence", _evidence()),
    }
    payload.update(extra)
    return payload


def test_two_pickups_in_the_opening_second_become_one_action():
    document = AnalysisDocument.model_validate(
        {
            "actions": [
                _action("Action_1", "pick_up", 1, 2, ["Object_1"], description="Right hand lifts the gray body."),
                _action("Action_2", "pick_up", 1, 2, ["Object_7"], description="Left hand picks up the clear part."),
                _action("Action_3", "rotate", 2, 3, ["Object_1"]),
            ],
            "workflow_steps": [
                {
                    "local_id": "Step_1",
                    "name": "Lift the parts",
                    "action_local_ids": ["Action_1", "Action_2", "Action_3"],
                    "evidence": _evidence(),
                }
            ],
            "ambiguities": [
                {
                    "local_id": "Ambiguity_1",
                    "kind": "uncertain_identity",
                    "description": "Which hand.",
                    "action_local_ids": ["Action_2"],
                    "evidence": _evidence(),
                }
            ],
        }
    )
    collapsed = collapse_repeated_actions(document)
    assert [item.local_id for item in collapsed.actions] == ["Action_1", "Action_3"]
    opening = collapsed.actions[0]
    assert opening.object_local_ids == ["Object_1", "Object_7"]
    assert "gray body" in opening.description
    assert "clear part" in opening.description
    assert collapsed.workflow_steps[0].action_local_ids == ["Action_1", "Action_3"]
    assert collapsed.ambiguities[0].action_local_ids == ["Action_1"]


def test_a_later_pickup_with_its_own_interval_stays():
    document = AnalysisDocument.model_validate(
        {
            "actions": [
                _action("Action_1", "pick_up", 1, 2, ["Object_1"]),
                _action("Action_2", "pick_up", 4, 5, ["Object_2"]),
                _action("Action_3", "pick_up", 1.0, 2.4, ["Object_3"]),
            ]
        }
    )
    collapsed = collapse_repeated_actions(document)
    assert [item.local_id for item in collapsed.actions] == ["Action_1", "Action_2", "Action_3"]


def test_hundredths_that_display_as_the_same_range_collapse_and_keep_the_stronger_outcome():
    document = AnalysisDocument.model_validate(
        {
            "actions": [
                _action(
                    "Action_1",
                    "pick_up",
                    1.001,
                    2.004,
                    ["Object_1"],
                    outcome="attempted",
                    uncertainty=0.8,
                    evidence=_evidence("INFERRED"),
                ),
                _action(
                    "Action_2",
                    "Pick_Up",
                    1.004,
                    1.996,
                    ["Object_1", "Object_2"],
                    outcome="completed",
                    uncertainty=0.2,
                    evidence=_evidence("DIRECTLY_OBSERVED"),
                ),
            ]
        }
    )
    collapsed = collapse_repeated_actions(document)
    assert len(collapsed.actions) == 1
    action = collapsed.actions[0]
    assert action.local_id == "Action_1"
    assert action.outcome == ActionOutcome.COMPLETED
    assert action.uncertainty == 0.2
    assert action.object_local_ids == ["Object_1", "Object_2"]
    assert action.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    assert action.evidence.frame_timestamps == [1.0]


def test_a_single_action_is_left_alone():
    document = AnalysisDocument.model_validate(
        {"actions": [_action("Action_1", "release", 3, 4, ["Object_7"])]}
    )
    assert collapse_repeated_actions(document) == document
