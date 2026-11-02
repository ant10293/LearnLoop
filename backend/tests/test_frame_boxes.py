import asyncio
from types import SimpleNamespace

from app.pipeline import infer
from app.pipeline.frame_boxes import (
    FRAME_BOX_ATTEMPTS,
    UNMEASURED_AUTHORITY,
    frame_box_rejection,
    frame_boxes_usable,
    mark_unmeasured,
    merge_frame_boxes,
    parse_frame_boxes,
    square_normalized,
)
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
)
from app.schemas.enums import EvidenceClass, Provenance, Visibility


def _obs(object_id: str, timestamp: float, w: float, h: float, x: float = 0.2, y: float = 0.2) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=x, y=y, w=w, h=h),
        evidence=Evidence(
            classification=EvidenceClass.DIRECTLY_OBSERVED,
            provenance=Provenance.AI_PREDICTION,
        ),
    )


def test_square_fractions_on_a_tall_frame_need_another_pass():
    boxes = [_obs("Object_1", float(index), 0.05, 0.05) for index in range(8)]
    assert square_normalized(boxes, 2160, 3840)


def test_portrait_sized_boxes_stay():
    boxes = [_obs("Object_1", float(index), 0.08, 0.045) for index in range(8)]
    assert square_normalized(boxes, 2160, 3840) is False


def test_a_short_square_answer_is_not_copied_onto_a_tall_frame():
    boxes = [_obs("Object_1", float(index), 0.05, 0.05) for index in range(3)]
    assert square_normalized(boxes, 2160, 3840) is False
    assert frame_boxes_usable(boxes, 2160, 3840) is False


def test_a_square_frame_can_have_square_boxes():
    boxes = [_obs("Object_1", float(index), 0.05, 0.05) for index in range(8)]
    assert square_normalized(boxes, 1000, 1000) is False


def test_pixel_corners_become_fractions_of_the_frame():
    text = '{"observations":[{"object_local_id":"Object_1","timestamp":0,"bbox":{"x":940,"y":1318,"w":442,"h":250}}]}'
    document = parse_frame_boxes(text, 2160, 3840)
    box = document.observations[0].bbox
    assert box is not None
    assert round(box.x, 3) == round(940 / 2160, 3)
    assert round(box.y, 3) == round(1318 / 3840, 3)


def test_a_box_shared_by_three_parts_is_not_copied():
    boxes = [_obs(f"Object_{index}", 0.0, 0.08, 0.045, x=0.4, y=0.3) for index in range(1, 4)]
    assert frame_box_rejection(boxes, 2160, 3840, {item.object_local_id for item in boxes})


def test_an_absent_part_clears_the_video_box():
    original = AnalysisDocument(observations=[_obs("Object_1", 1.0, 0.08, 0.045)])
    fresh = AnalysisDocument(
        observations=[
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=1.0,
                bbox=None,
                visibility="absent",
                confidence=None,
                evidence=Evidence(classification=EvidenceClass.DIRECTLY_OBSERVED, provenance=Provenance.AI_PREDICTION),
            )
        ]
    )
    merged = merge_frame_boxes(original, fresh)
    assert merged.observations[0].bbox is None
    assert merged.observations[0].confidence is None


def test_two_ids_on_one_rectangle_keep_the_higher_score():
    original = AnalysisDocument(
        observations=[
            _obs("Object_1", 1.0, 0.08, 0.045, x=0.4, y=0.3),
            _obs("Object_2", 1.0, 0.08, 0.045, x=0.4, y=0.3),
        ]
    )
    original.observations[0].confidence = 0.4
    original.observations[1].confidence = 0.8
    fresh = AnalysisDocument(observations=original.observations)
    merged = merge_frame_boxes(original, fresh)
    kept = [item for item in merged.observations if item.bbox is not None]
    assert [item.object_local_id for item in kept] == ["Object_2"]


def test_frame_boxes_replace_the_same_moment_only():
    original = AnalysisDocument(observations=[_obs("Object_1", 1.0, 0.05, 0.05), _obs("Object_1", 2.0, 0.05, 0.05)])
    fresh = AnalysisDocument(observations=[_obs("Object_1", 1.0, 0.09, 0.05, x=0.4, y=0.3)])
    merged = merge_frame_boxes(original, fresh)
    first, second = merged.observations
    assert first.bbox is not None and (first.bbox.x, first.bbox.w, first.bbox.h) == (0.4, 0.09, 0.05)
    assert second.bbox is not None and second.bbox.w == 0.05


def test_frame_box_contract_only_allows_the_roster_ids():
    from app.pipeline.contracts import validate_strict_schema
    from app.pipeline.frame_boxes import FRAME_BOX_SCHEMA, frame_box_schema

    schema = frame_box_schema(["Object_2", "Object_3"])
    validate_strict_schema(schema)
    fields = schema["properties"]["observations"]["items"]["properties"]
    assert fields["object_local_id"]["enum"] == ["Object_2", "Object_3"]
    assert fields["visibility"]["enum"] == ["visible", "partial", "occluded", "absent"]
    assert FRAME_BOX_SCHEMA["properties"]["observations"]["items"]["properties"]["object_local_id"] == {
        "type": "string"
    }



def test_a_frame_answer_adds_a_moment_the_video_answer_left_out():
    video_only_first_frame = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="part")],
        observations=[_obs("Object_1", 0.0, 0.05, 0.05)],
    )
    fresh = AnalysisDocument(
        observations=[
            _obs("Object_1", 4.0, 0.09, 0.05, x=0.4, y=0.3),
            _obs("Object_9", 4.0, 0.09, 0.05, x=0.1, y=0.1),
        ]
    )
    merged = merge_frame_boxes(video_only_first_frame, fresh)
    assert [(item.object_local_id, item.timestamp) for item in merged.observations] == [
        ("Object_1", 0.0),
        ("Object_1", 4.0),
    ]
    added = merged.observations[1]
    assert added.bbox is not None and added.bbox.x == 0.4
    assert added.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
    assert added.evidence.frame_timestamps == [4.0]


def _absent(object_id: str, timestamp: float) -> ObservationPrediction:
    return _obs(object_id, timestamp, 0.08, 0.045).model_copy(update={"bbox": None, "visibility": Visibility.ABSENT})


def test_an_answer_without_boxes_names_what_went_wrong():
    assert frame_box_rejection([_absent("Object_1", 0.0)], 2160, 3840, {"Object_1"}) == "the answer called every part absent"
    empty = _obs("Object_1", 0.0, 0.08, 0.045).model_copy(update={"bbox": None})
    assert frame_box_rejection([empty], 2160, 3840, {"Object_1"}) == "the answer contained no boxes"
    assert frame_box_rejection([], 2160, 3840, {"Object_1"}) == "the answer contained no boxes"


def test_only_boxes_no_answer_settled_are_marked_unmeasured():
    document = AnalysisDocument(
        observations=[
            _obs("Object_1", 0.0, 0.08, 0.045),
            _obs("Object_1", 4.0, 0.08, 0.045),
            _obs("Object_1", 9.0, 0.08, 0.045),
            _obs("Object_2", 0.0, 0.08, 0.045),
            _absent("Object_1", 1.0),
        ]
    )
    failed = [(0.0, 3.0, "the answer contained no boxes")]
    result = mark_unmeasured(document, {"Object_1"}, {("Object_1", 4.0)}, failed)
    authority = [item.geometry_authority for item in result.observations]
    assert authority == [UNMEASURED_AUTHORITY, "", UNMEASURED_AUTHORITY, "", ""]
    assert all(row.outcome == "kept" for row in result.box_decisions)
    assert "the answer contained no boxes" in result.box_decisions[0].detail
    assert "no frame answer settled" in result.box_decisions[1].detail


def _window_run(monkeypatch, answers):
    calls = []

    async def answer(_session, _run, _stage, _system, user, *_args, **_kwargs):
        calls.append(user)
        return answers(user, len(calls))

    monkeypatch.setattr(infer, "_call", answer)
    objects = [ObjectPrediction(local_id=name, label=name) for name in ("Object_1", "Object_2")]
    document = AnalysisDocument(
        objects=objects,
        observations=[_obs(name, 0.0, 0.08, 0.045) for name in ("Object_1", "Object_2")],
    )
    frames = [SimpleNamespace(id="f0", timestamp=0.0)]
    run = SimpleNamespace(id="run", config_json={})
    video = SimpleNamespace(display_width=2160, display_height=3840)
    coverage = SimpleNamespace(temporal_coverage_note="")
    result = asyncio.run(infer._reground_video_boxes(None, run, "scene_analysis", document, frames, video, coverage))
    return result, calls, coverage


def _measured() -> AnalysisDocument:
    return AnalysisDocument(
        observations=[
            _obs("Object_1", 0.0, 0.1, 0.05, x=0.5),
            _obs("Object_2", 0.0, 0.1, 0.05, x=0.7),
        ]
    )


def test_a_refused_window_is_asked_again_and_the_second_answer_counts(monkeypatch):
    empty = AnalysisDocument(observations=[_absent("Object_1", 0.0), _absent("Object_2", 0.0)])
    result, calls, coverage = _window_run(monkeypatch, lambda _user, count: empty if count == 1 else _measured())
    assert len(calls) == 2
    assert "refused because the answer called every part absent" in calls[1]
    assert "refused" not in calls[0]
    assert {item.bbox.x for item in result.observations} == {0.5, 0.7}
    assert {item.geometry_authority for item in result.observations} == {""}
    assert coverage.temporal_coverage_note == ""


def test_a_window_refused_twice_keeps_the_video_boxes_marked(monkeypatch):
    empty = AnalysisDocument(observations=[_absent("Object_1", 0.0), _absent("Object_2", 0.0)])
    result, calls, coverage = _window_run(monkeypatch, lambda _user, _count: empty)
    assert len(calls) == FRAME_BOX_ATTEMPTS
    assert {item.bbox.x for item in result.observations} == {0.2}
    assert {item.geometry_authority for item in result.observations} == {UNMEASURED_AUTHORITY}
    assert {row.reason_code for row in result.box_decisions} == {"frame_box_unmeasured"}
    assert "the answer called every part absent" in coverage.temporal_coverage_note
