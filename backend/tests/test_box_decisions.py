"""Each removed box leaves a record with its original geometry and a reason."""

from app.api.serialize import _box_decisions
from app.perception.continuity import reconcile_identity
from app.perception.decisions import merged_decisions
from app.perception.locate import apply_located
from app.perception.roster import drop_contained_objects, omit_workspace_and_hands
from app.pipeline.frame_boxes import drop_copied_boxes, merge_frame_boxes
from app.pipeline.locate_pass import _on_the_same_surface
from app.pipeline.merge import restore_omitted_records, retain_prior_records
from app.pipeline.parsing import parse_analysis
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    BoxDecision,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
)
from app.schemas.enums import EvidenceClass, Provenance, Visibility


def _evidence(classification=EvidenceClass.DIRECTLY_OBSERVED) -> Evidence:
    return Evidence(classification=classification, provenance=Provenance.AI_PREDICTION, note="test")


def _object(local_id: str, label: str = "rod") -> ObjectPrediction:
    return ObjectPrediction(local_id=local_id, label=label, evidence=_evidence())


def _obs(object_id: str, timestamp: float, x=0.1, y=0.1, w=0.1, h=0.1, confidence=None, classification=EvidenceClass.DIRECTLY_OBSERVED):
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=x, y=y, w=w, h=h),
        visibility=Visibility.VISIBLE,
        confidence=confidence,
        evidence=_evidence(classification),
    )


def _reasons(document: AnalysisDocument) -> list[str]:
    return [row.reason_code for row in document.box_decisions]


def test_field_stays_out_of_the_model_schema():
    assert "box_decisions" not in AnalysisDocument.model_json_schema()["properties"]


def test_model_answer_cannot_write_decisions():
    document = parse_analysis('{"summary": "x", "box_decisions": [{"step": "a", "reason_code": "b", "object_local_id": "c"}]}')
    assert document.box_decisions == []


def test_workspace_drop_records_every_box():
    document = AnalysisDocument(
        objects=[_object("Object_1", "table"), _object("Object_2")],
        observations=[_obs("Object_1", 0.0), _obs("Object_1", 1.0), _obs("Object_2", 0.0)],
    )
    result = omit_workspace_and_hands(document)
    assert _reasons(result) == ["workspace_or_hand", "workspace_or_hand"]
    assert result.box_decisions[0].bbox == document.observations[0].bbox


def test_object_with_no_boxes_still_gets_a_record():
    result = omit_workspace_and_hands(AnalysisDocument(objects=[_object("Object_1", "hand")]))
    assert [(row.reason_code, row.bbox) for row in result.box_decisions] == [("workspace_or_hand", None)]


def test_contained_object_names_its_host():
    document = AnalysisDocument(
        objects=[_object("Object_1"), _object("Object_2")],
        observations=[
            row
            for timestamp in (0.0, 1.0, 2.0)
            for row in (_obs("Object_1", timestamp, w=0.4, h=0.4), _obs("Object_2", timestamp, x=0.15, y=0.15, w=0.05, h=0.05))
        ],
    )
    result = drop_contained_objects(document)
    assert [obj.local_id for obj in result.objects] == ["Object_1"]
    assert {row.reason_code for row in result.box_decisions} == {"contained_in_object"}
    assert {row.other_object_id for row in result.box_decisions} == {"Object_1"}


def test_oversized_box_records_extent_drop():
    document = AnalysisDocument(
        objects=[_object("Object_1")],
        observations=[
            _obs("Object_1", 0.0),
            _obs("Object_1", 1.0),
            _obs("Object_1", 2.0),
            _obs("Object_1", 3.0, w=0.6, h=0.6),
        ],
    )
    result = reconcile_identity(document)
    assert _reasons(result) == ["extent_growth_exceeded"]
    row = result.box_decisions[0]
    assert row.timestamp == 3.0 and row.bbox.w == 0.6
    assert "36.0 times" in row.detail


def test_copied_box_keeps_the_higher_score_and_names_it():
    document = AnalysisDocument(
        observations=[_obs("Object_1", 0.0, confidence=0.9), _obs("Object_2", 0.0, confidence=0.4)],
    )
    result = drop_copied_boxes(document)
    assert _reasons(result) == ["duplicate_copied_box"]
    row = result.box_decisions[0]
    assert (row.object_local_id, row.other_object_id, row.confidence) == ("Object_2", "Object_1", 0.4)
    assert row.bbox is not None


def test_frame_answer_absence_records_the_cleared_box():
    absent = ObservationPrediction(
        object_local_id="Object_1", timestamp=0.0, bbox=None, visibility=Visibility.ABSENT, evidence=_evidence()
    )
    result = merge_frame_boxes(AnalysisDocument(observations=[_obs("Object_1", 0.0)]), AnalysisDocument(observations=[absent]))
    assert result.observations[0].bbox is None
    assert _reasons(result) == ["frame_answer_absent"]
    assert result.box_decisions[0].bbox is not None


def test_located_box_refusals_carry_reasons():
    other = _obs("Object_2", 1.0, x=0.5, y=0.5)
    references = [_obs("Object_1", timestamp) for timestamp in (0.0, 7.0, 8.0, 9.0)]
    document = AnalysisDocument(objects=[_object("Object_1"), _object("Object_2")], observations=[*references, other])
    good = BBox(x=0.1, y=0.1, w=0.1, h=0.1)
    found = [
        (2.0, good, "visible", 0.2),
        (3.0, BBox(x=0.1, y=0.1, w=0.8, h=0.8), "visible", 0.9),
        (1.0, BBox(x=0.5, y=0.5, w=0.1, h=0.1), "visible", 0.9),
        (0.0, good, "visible", 0.9),
        (4.0, good, "absent", 0.9),
        (5.0, None, "absent", None),
        (6.0, None, "visible", None, good, 0.8),
    ]
    result = apply_located(document, "Object_1", found)
    assert _reasons(result) == [
        "low_confidence",
        "extent_mismatch",
        "duplicate_overlap",
        "slot_occupied",
        "box_marked_absent",
        "surface_mismatch",
    ]
    assert result.box_decisions[2].other_object_id == "Object_2"
    assert result.box_decisions[5].confidence == 0.8
    assert len(result.observations) == 5


def test_surface_refusal_keeps_the_box():
    box = BBox(x=0.1, y=0.1, w=0.1, h=0.1)
    kept = _on_the_same_surface(None, [], [(1.0, box, "visible", 0.8)])
    assert kept == [(1.0, None, "visible", None, box, 0.8)]


def test_window_merge_records_losers_and_unaligned_ids():
    merged = AnalysisDocument(objects=[_object("Object_1")], observations=[_obs("Object_1", 0.0, x=0.7, y=0.7, classification=EvidenceClass.INFERRED)])
    first = AnalysisDocument(objects=[_object("Object_1")], observations=[_obs("Object_1", 0.0)])
    second = AnalysisDocument(objects=[_object("Object_1")], observations=[_obs("Object_1", 0.0, x=0.4, y=0.4, classification=EvidenceClass.INFERRED)])
    result = restore_omitted_records(merged, [first, second])
    reasons = _reasons(result)
    assert "window_conflict_lost" in reasons
    assert "text_merge_overridden" in reasons


def test_records_survive_a_later_stage_without_duplicates():
    row = BoxDecision(step="locate", reason_code="low_confidence", object_local_id="Object_1", timestamp=1.0)
    prior = AnalysisDocument(box_decisions=[row])
    result = retain_prior_records(AnalysisDocument(box_decisions=[row]), prior)
    assert result.box_decisions == [row]
    assert merged_decisions([row], [row]) == [row]


def test_serialized_counts_cover_truncated_rows():
    row = BoxDecision(step="locate", reason_code="low_confidence", object_local_id="Object_1").model_dump(mode="json")
    payload = _box_decisions({"box_decisions": [row, row]})
    assert payload["total"] == 2
    assert payload["counts"] == [{"step": "locate", "reason_code": "low_confidence", "outcome": "dropped", "count": 2}]
    assert _box_decisions({}) == {"total": 0, "truncated": False, "counts": [], "rows": []}


def test_records_survive_the_stored_document_round_trip():
    row = BoxDecision(step="locate", reason_code="low_confidence", object_local_id="Object_1", bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1))
    stored = AnalysisDocument(box_decisions=[row]).model_dump(mode="json")
    assert AnalysisDocument.model_validate(stored).box_decisions == [row]
    assert _box_decisions(stored)["rows"][0]["reason_code"] == "low_confidence"
