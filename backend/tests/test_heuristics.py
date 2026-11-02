"""Each retired repair is fenced to model estimates, and its replacement wins where evidence exists."""

import asyncio
from types import SimpleNamespace

import pytest
from app.perception.assembly import apply_assembly, find_joins
from app.perception.continuity import reconcile_identity
from app.perception.resolve import fence_repair, mark_estimates
from app.perception.roster import drop_contained_objects, drop_split_objects
from app.pipeline import infer
from app.pipeline.locate_pass import _targets
from app.pipeline.merge import retain_prior_records
from app.schemas.analysis import (
    ActionPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
    WorkflowStepPrediction,
)
from app.schemas.enums import ActionOutcome, EvidenceClass, Provenance


def _evidence(timestamp: float = 0) -> Evidence:
    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.AI_PREDICTION,
        frame_timestamps=[timestamp],
    )


def _estimate(object_id: str, timestamp: float, box: tuple[float, float, float, float]) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
        evidence=_evidence(timestamp),
        geometry_origin="vlm_estimated",
        geometry_authority="vlm",
        detection_support="none",
    )


def _canonical(object_id: str, timestamp: float, box: tuple[float, float, float, float], track: str) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
        evidence=_evidence(timestamp),
        track_ref=track,
        source_model="tracker",
        observation_role="canonical",
        geometry_origin="detected",
        geometry_authority="manual",
        detection_support="detected",
        association_support="associated",
        temporal_support="repeated",
    )


def _doc(objects, observations, **extra) -> AnalysisDocument:
    return AnalysisDocument(objects=objects, observations=observations, **extra)


def _ids(document: AnalysisDocument) -> set[str]:
    return {obj.local_id for obj in document.objects}


BODY = (0.40, 0.40, 0.30, 0.20)
STUD = (0.50, 0.48, 0.04, 0.04)


def _nested(bound: bool) -> AnalysisDocument:
    objects = [
        ObjectPrediction(local_id="Object_1", label="gray body", instance_id="inst-1" if bound else None),
        ObjectPrediction(local_id="Object_7", label="clear stud", instance_id="inst-2" if bound else None),
    ]
    if bound:
        rows = [_canonical("Object_1", float(t), BODY, "inst-1") for t in range(3)]
        rows += [_canonical("Object_7", float(t), STUD, "inst-2") for t in range(3)]
    else:
        rows = [_estimate("Object_1", float(t), BODY) for t in range(3)]
        rows += [_estimate("Object_7", float(t), STUD) for t in range(3)]
    return _doc(objects, rows)


def test_two_canonical_nested_instances_both_remain():
    document = fence_repair(_nested(bound=True), reconcile_identity)
    kept = drop_contained_objects(document)
    assert _ids(kept) == {"Object_1", "Object_7"}
    assert sum(1 for row in kept.observations if row.object_local_id == "Object_7") == 3
    assert all(row.observation_role == "canonical" for row in kept.observations)


def test_a_nested_estimate_without_an_instance_is_still_merged():
    kept = drop_contained_objects(_nested(bound=False))
    assert _ids(kept) == {"Object_1"}


def test_a_vlm_only_size_glitch_is_still_repaired():
    objects = [
        ObjectPrediction(local_id="Object_1", label="short rod"),
        ObjectPrediction(local_id="Object_5", label="gray body"),
    ]
    rod = [_estimate("Object_1", float(t), (0.55, 0.25, 0.08, 0.02)) for t in range(5)]
    body = [_estimate("Object_5", float(t), (0.40, 0.40, 0.20, 0.15)) for t in range(6)]
    stolen = _estimate("Object_1", 5, (0.40, 0.40, 0.20, 0.16))
    document = fence_repair(_doc(objects, [*rod, stolen, *body]), reconcile_identity)
    assert not any(row.object_local_id == "Object_1" and row.timestamp == 5 for row in document.observations)


def test_reconcile_never_edits_a_canonical_row():
    objects = [
        ObjectPrediction(local_id="Object_1", label="short rod", instance_id="inst-1"),
        ObjectPrediction(local_id="Object_5", label="gray body"),
    ]
    rod = [_canonical("Object_1", float(t), (0.55, 0.25, 0.08, 0.02), "inst-1") for t in range(5)]
    # The tracker measured the rod large at t=5. Only a measurement can overrule that.
    grown = _canonical("Object_1", 5, (0.40, 0.40, 0.20, 0.16), "inst-1")
    body = [_estimate("Object_5", float(t), (0.40, 0.40, 0.20, 0.15)) for t in range(6)]
    document = fence_repair(_doc(objects, [*rod, grown, *body]), reconcile_identity)
    kept = [row for row in document.observations if row.object_local_id == "Object_1" and row.timestamp == 5]
    assert len(kept) == 1 and kept[0].bbox.w == pytest.approx(0.20)


def _touching(bound: bool) -> tuple[AnalysisDocument, AnalysisDocument]:
    roster = _doc([ObjectPrediction(local_id="Object_1", label="gray block")], [])
    merged = _doc(
        [
            ObjectPrediction(local_id="Object_1", label="gray block"),
            ObjectPrediction(local_id="Object_2", label="gray block", instance_id="inst-2" if bound else None),
        ],
        [
            _estimate("Object_1", 0, (0.20, 0.20, 0.10, 0.10)),
            _estimate("Object_2", 0, (0.30, 0.20, 0.10, 0.10)),
        ],
    )
    return roster, merged


def test_two_touching_identical_instances_stay_two():
    roster, merged = _touching(bound=True)
    assert _ids(drop_split_objects(roster, merged)) == {"Object_1", "Object_2"}


def test_a_touching_split_without_an_instance_is_still_closed():
    roster, merged = _touching(bound=False)
    assert _ids(drop_split_objects(roster, merged)) == {"Object_1"}


def test_a_canonical_track_causes_no_locate_call():
    objects = [
        ObjectPrediction(local_id="Object_1", label="gray body", instance_id="inst-1", manipulated=True),
        ObjectPrediction(local_id="Object_2", label="clear stud", manipulated=True),
        ObjectPrediction(local_id="Object_3", label="base"),
    ]
    rows = [_canonical("Object_1", float(t), BODY, "inst-1") for t in range(3)]
    rows += [_estimate("Object_2", float(t), STUD) for t in range(3)]
    rows += [_estimate("Object_3", float(t), (0.1, 0.7, 0.2, 0.1)) for t in range(8)]
    targets = {object_id for object_id, _times in _targets(_doc(objects, rows))}
    assert "Object_1" not in targets
    assert "Object_2" in targets


def test_canonical_rows_skip_the_reground_call(monkeypatch):
    async def refuse(*_args, **_kwargs):
        raise AssertionError("a frame box call was made for an object with canonical geometry")

    monkeypatch.setattr(infer, "_call", refuse)
    document = _nested(bound=True)
    frames = [SimpleNamespace(id=f"f{t}", timestamp=float(t)) for t in range(3)]
    run = SimpleNamespace(id="run", config_json={})
    video = SimpleNamespace(display_width=640, display_height=480)
    result = asyncio.run(infer._reground_video_boxes(None, run, None, document, frames, video, SimpleNamespace()))
    assert result == document


def test_a_later_stage_box_does_not_replace_a_canonical_row():
    objects = [ObjectPrediction(local_id="Object_1", label="gray body", instance_id="inst-1")]
    step = WorkflowStepPrediction(
        local_id="Step_1",
        name="Place the body",
        expected_action="place",
        completion_condition="The body rests on the base",
        observed_result="The body rests on the base",
        time_start=0,
        time_end=2,
        evidence=_evidence(1),
    )
    prior = _doc(objects, [_canonical("Object_1", 1.0, BODY, "inst-1")], workflow_steps=[step])
    later = mark_estimates(_doc(objects, [ObservationPrediction(
        object_local_id="Object_1",
        timestamp=1.0,
        bbox=BBox(x=0.05, y=0.05, w=0.1, h=0.1),
        evidence=_evidence(1),
    )]))
    merged = retain_prior_records(later, prior)
    shown = [row for row in merged.observations if row.object_local_id == "Object_1"]
    assert len(shown) == 1 and shown[0].observation_role == "canonical"
    assert shown[0].bbox.x == pytest.approx(BODY[0])
    assert any(row.geometry_authority == "vlm_later_stage" for row in merged.raw_observations)
    assert [item.local_id for item in merged.workflow_steps] == ["Step_1"]


def _insertion(origin: str) -> AnalysisDocument:
    def row(object_id: str, timestamp: float, box) -> ObservationPrediction:
        return ObservationPrediction(
            object_local_id=object_id,
            timestamp=timestamp,
            bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
            evidence=_evidence(timestamp),
            geometry_origin=origin,
        )

    rows = [row("Object_5", float(t), (0.45, 0.40, 0.18, 0.16)) for t in range(6)]
    rows += [
        row("Object_1", 0, (0.10, 0.20, 0.08, 0.02)),
        row("Object_1", 1, (0.12, 0.20, 0.08, 0.02)),
        row("Object_1", 2, (0.20, 0.25, 0.08, 0.02)),
        row("Object_1", 3, (0.42, 0.42, 0.08, 0.02)),
    ]
    action = ActionPrediction(
        local_id="Action_1",
        action_type="approach",
        description="The rod is brought to the body.",
        start=2,
        end=4,
        object_local_ids=["Object_1", "Object_5"],
        outcome=ActionOutcome.UNKNOWN,
        uncertainty=0.7,
        evidence=_evidence(3),
    )
    return _doc(
        [
            ObjectPrediction(local_id="Object_1", label="short rod", manipulated=True),
            ObjectPrediction(local_id="Object_5", label="gray body", manipulated=True),
        ],
        rows,
        actions=[action],
    )


def test_vlm_only_boxes_make_no_assembly_claim():
    document = _insertion("vlm_estimated")
    assert find_joins(document) == []
    assert apply_assembly(document) == document


def test_sustained_contact_stays_inferred_and_does_not_complete():
    document = apply_assembly(_insertion("detected"))
    assert find_joins(document)
    assert all(action.outcome != ActionOutcome.COMPLETED for action in document.actions)
    relations = [item for item in document.relationships if item.subject_local_id == "Object_1"]
    assert relations
    assert all(item.evidence.classification == EvidenceClass.INFERRED for item in relations)
