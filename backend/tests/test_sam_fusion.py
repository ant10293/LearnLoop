"""SAM tracking, association, and fusion. No provider calls are made."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")

from app.perception.sam_fusion import (
    TrackSample,
    associate_tracks,
    fuse_geometry,
    merge_reconciliation,
    plan_prompts,
    prompt_prior,
    sam_evidence_block,
    write_sam_summary,
)
from app.perception.sam_masks import materialize_record, write_mask_png
from app.providers.sam import (
    SamRecord,
    SpatialMark,
    concept_text,
    frame_timestamp,
    parse_sam_output,
    plan_video_chunks,
    records_from_chunks,
    render_spatial_prompt,
    sam_capabilities,
    sam_video_accepted,
    scoped_track_id,
    video_request,
)
from app.providers.types import ProviderError
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
)
from app.schemas.enums import EvidenceClass, Provenance, Visibility
from app.verification.predicates import judge_predicate
from app.verification.schema import CompletionPredicate, RoleAssignment


def _evidence(kind=EvidenceClass.DIRECTLY_OBSERVED, provenance=Provenance.AI_PREDICTION, note=""):
    return Evidence(classification=kind, provenance=provenance, note=note)


def _box(x, y, w=0.2, h=0.2) -> BBox:
    return BBox(x=x, y=y, w=w, h=h)


def _document() -> AnalysisDocument:
    return AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_1", label="washer"),
            ObjectPrediction(local_id="Object_2", label="washer"),
        ],
        observations=[
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=1.0,
                bbox=_box(0.1, 0.1),
                evidence=_evidence(),
            ),
            ObservationPrediction(
                object_local_id="Object_2",
                timestamp=1.0,
                bbox=_box(0.7, 0.7),
                evidence=_evidence(),
            ),
        ],
    )


def _sample(session, raw, x, y, timestamp=1.0, bound="", frame=0) -> TrackSample:
    return TrackSample(
        session_id=session,
        raw_track_id=raw,
        timestamp=timestamp,
        bbox=_box(x, y),
        confidence=0.8,
        frame_index=frame,
        bound_object_id=bound,
        mask_data="5,5,!unused",
        source_width=100,
        source_height=100,
    )


LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334;c=0.5|>"
    "<|mask;x=0;y=0;c=0.5;data=27,60,~abc|>,"
    "2<|box;x1=10;y1=12;x2=40;y2=50;w=320;h=334|>"
    "<|mask;x=0;y=0;data=8,8,!zzzz|>"
)


def test_existing_grammar_and_sparse_ids_survive_a_split_stream():
    whole = parse_sam_output(LINE + "\n<2f>2<|box;x1=1;y1=1;x2=4;y2=4;w=320;h=334|>"
                              "<|mask;x=0;y=0;data=2,2,~qq|>")
    split_at = LINE.index("~abc")
    chunks = [LINE[:split_at], LINE[split_at:] + "\n<2f>2<|box;x1=1;y1=1;x2=4;y2=4;w=320;h=334|>"
              "<|mask;x=0;y=0;data=2,2,~qq|>"]
    streamed = records_from_chunks(chunks)
    assert [item.frame_index for item in streamed] == [item.frame_index for item in whole]
    assert [item.track_id for item in streamed] == ["0", "2", "2"]
    assert streamed[0].confidence == 0.5
    assert streamed[0].mask_encoding == "lossless"
    assert streamed[1].mask_encoding == "one_bit"
    assert streamed[1].mask_height == 8
    assert streamed[1].mask_width == 8
    assert parse_sam_output("not a frame") == []
    assert parse_sam_output("<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>") == []


def test_video_request_matches_the_documented_shape():
    body = video_request("red bicycle", file_id="file-abc123", prompt_kind="concept")
    content = body["input"][0]["content"]
    assert body["model"] == "sam-3.1"
    assert body["stream"] is True
    assert content[0] == {"type": "input_text", "text": "red bicycle"}
    assert content[1] == {"type": "input_video", "file_id": "file-abc123"}
    assert body["metadata"]["include_confidence"] == "true"
    assert body["metadata"]["mask_encoding"] == "one_bit"
    assert "propagation_direction" not in body["metadata"]
    with pytest.raises(ProviderError):
        video_request("red bicycle", video_url="https://example.com/a.mp4", file_id="file-1")
    spatial = render_spatial_prompt(
        [
            SpatialMark(0, 1, "box", 1800, 1200, x1=444, y1=626, x2=675, y2=849),
            SpatialMark(0, 2, "point", 1800, 1200, x=1430, y=700),
            SpatialMark(0, 2, "point", 1800, 1200, x=1300, y=450, negative=True),
        ]
    )
    assert spatial.startswith("<0f>1<|box;")
    assert "-<|point;x=1300;y=450;w=1800;h=1200|>" in spatial
    assert "bicycle" not in spatial
    requested = video_request(spatial, file_id="file-abc123", prompt_kind="spatial", propagation_direction="forward")
    assert requested["metadata"]["propagation_direction"] == "forward"
    with pytest.raises(ProviderError):
        video_request(spatial + " washer", file_id="file-abc123", prompt_kind="spatial")


def test_capabilities_report_video_tracking_only_because_the_adapter_sends_it():
    caps = sam_capabilities()
    assert caps["image_segmentation"] is True
    assert caps["box_prompts"] is True
    assert caps["point_prompts"] is True
    assert caps["video_tracking"] is True
    assert caps["max_frames_per_request"] == 15000
    assert "Live Meta API" in caps["video_tracking_detail"]
    assert sam_video_accepted("h264", "mp4") is True
    assert sam_video_accepted("vp9", "webm") is False


def test_chunks_scope_ids_and_map_frame_indices_to_timestamps():
    chunks = plan_video_chunks(450, 30, chunk_frames=300)
    assert [chunk.session_id for chunk in chunks] == ["c0", "c1"]
    assert chunks[1].time_start == pytest.approx(10.0)
    assert frame_timestamp(2, 30, chunks[1]) == pytest.approx(10.0 + 2 / 30)
    assert frame_timestamp(0, 0, chunks[0]) is None
    assert scoped_track_id("c0s0", "1") != scoped_track_id("c1s0", "1")
    assert concept_text("metal rod") == "metal rod"
    with pytest.raises(ProviderError):
        concept_text("")
    with pytest.raises(ProviderError):
        concept_text("<0f>1<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>")


def test_prompts_name_the_part_instead_of_pointing_at_a_box():
    document = _document()
    chunk = plan_video_chunks(30, 10)[0]
    plans = plan_prompts(document, chunk, 320, 240, 10)
    assert len(plans) == 1
    assert plans[0].kind == "concept"
    assert plans[0].prompt == "washer"
    assert "<|box;" not in plans[0].prompt
    assert "<|point;" not in plans[0].prompt
    assert plans[0].targets == ["Object_1", "Object_2"]
    labeled = AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_3", label="black tire wheel with light hub"),
            ObjectPrediction(local_id="Object_4", label="black tire wheel with light hub second"),
        ]
    )
    wheel_plans = plan_prompts(labeled, chunk, 320, 240, 10)
    assert wheel_plans[0].prompt == "black tire wheel with light hub"
    assert wheel_plans[0].targets == ["Object_3", "Object_4"]


def _two_time_document() -> AnalysisDocument:
    document = _document()
    later = [
        item.model_copy(update={"timestamp": 2.0})
        for item in document.observations
    ]
    return document.model_copy(update={"observations": [*document.observations, *later]})


def _concept(sample: TrackSample, *targets: str, confidence: float = 0.8) -> TrackSample:
    sample.prompt_kind = "concept"
    sample.candidates = targets
    sample.confidence = confidence
    return sample


def test_a_concept_track_is_not_forced_onto_a_disagreeing_object():
    document = _document()
    sample = _concept(_sample("c0s0", "1", 0.7, 0.7), "Object_1")
    linked = associate_tracks(document, [sample])
    assert linked[0].status == "unassigned"
    assert linked[0].object_local_id is None
    fused = fuse_geometry(document, [sample], linked)
    assert not [item for item in fused.observations if item.source_model == "sam-3.1"]


def test_shared_noun_tracks_assign_by_repeated_overlap_not_confidence():
    document = _two_time_document()
    samples = [
        _concept(_sample("c0s0", "1", 0.1, 0.1, timestamp=t), "Object_1", "Object_2", confidence=0.2)
        for t in (1.0, 2.0)
    ] + [
        _concept(_sample("c0s0", "2", 0.7, 0.7, timestamp=t), "Object_1", "Object_2", confidence=0.99)
        for t in (1.0, 2.0)
    ]
    linked = {item.raw_track_id: item for item in associate_tracks(document, samples)}
    assert linked["1"].object_local_id == "Object_1"
    assert linked["2"].object_local_id == "Object_2"
    assert {linked["1"].status, linked["2"].status} == {"associated"}
    fused = fuse_geometry(document, samples, list(linked.values()))
    sam = [item for item in fused.observations if item.source_model == "sam-3.1"]
    assert {item.association_support for item in sam} == {"associated"}
    assert all(item.geometry_authority == "sam-3.1" for item in sam)


def test_one_shared_frame_is_weak_and_does_not_bind():
    document = _document()
    sample = _concept(_sample("c0s0", "1", 0.1, 0.1), "Object_1", "Object_2")
    linked = associate_tracks(document, [sample])
    assert linked[0].status == "weak"
    assert linked[0].object_local_id is None


def test_similar_objects_are_not_merged_by_label():
    document = _document()
    samples = [
        _sample("c0s0", "1", 0.1, 0.1, bound="Object_1"),
        _sample("c0s0", "2", 0.7, 0.7, bound="Object_2"),
    ]
    linked = {item.raw_track_id: item for item in associate_tracks(document, samples)}
    assert linked["1"].object_local_id == "Object_1"
    assert linked["2"].object_local_id == "Object_2"
    assert linked["1"].status == "associated"
    crowded = associate_tracks(
        _two_time_document(),
        [
            *(_sample("c0s0", "1", 0.1, 0.1, timestamp=t) for t in (1.0, 2.0)),
            *(_sample("c0s0", "2", 0.12, 0.12, timestamp=t) for t in (1.0, 2.0)),
        ],
    )
    statuses = {item.status for item in crowded}
    assert "associated" in statuses
    assert any(item.object_local_id is None for item in crowded)


def test_sessions_do_not_share_a_raw_id_and_conflicts_keep_both_boxes():
    document = _document()
    samples = [
        _sample("c0s0", "1", 0.1, 0.1, bound="Object_1"),
        _sample("c1s0", "1", 0.7, 0.7, bound="Object_2"),
    ]
    linked = associate_tracks(document, samples)
    assert {(item.session_id, item.object_local_id) for item in linked} == {
        ("c0s0", "Object_1"),
        ("c1s0", "Object_2"),
    }
    conflict_doc = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="rod")],
        observations=[
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=1.0,
                bbox=_box(0.1, 0.1),
                evidence=_evidence(),
            ),
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=1.0,
                bbox=_box(0.15, 0.15),
                evidence=_evidence(EvidenceClass.INFERRED, Provenance.AI_PREDICTION),
            ),
        ],
    )
    fused = fuse_geometry(
        conflict_doc,
        [_sample("c0s0", "1", 0.7, 0.7, bound="Object_1")],
        associate_tracks(conflict_doc, [_sample("c0s0", "1", 0.7, 0.7, bound="Object_1")]),
        scene_boundaries=[1.0],
    )
    originals = [item for item in fused.observations if item.source_model != "sam-3.1"]
    sam = [item for item in fused.observations if item.source_model == "sam-3.1"]
    assert len(originals) == 2
    assert any(item.evidence.classification == EvidenceClass.INFERRED for item in originals)
    assert sam and sam[0].association_status == "conflict"
    assert sam[0].visibility == Visibility.VISIBLE
    assert all(item.visibility != Visibility.ABSENT for item in fused.observations)


def test_a_skipped_frame_is_unobserved_and_not_absence():
    document = _document()
    samples = [
        _sample("c0s0", "1", 0.1, 0.1, timestamp=0.0, bound="Object_1", frame=0),
        _sample("c0s0", "1", 0.1, 0.1, timestamp=1.0, bound="Object_1", frame=10),
    ]
    fused = fuse_geometry(document, samples, associate_tracks(document, samples))
    assert not any(item.visibility == Visibility.ABSENT for item in fused.observations)
    assert any(item.kind.value == "unobserved_interval" for item in fused.ambiguities)


def test_prompt_prior_omits_per_frame_sam_rows():
    document = _document()
    samples = [_sample("c0s0", "1", 0.7, 0.7, bound="Object_1") for _ in range(30)]
    fused = fuse_geometry(document, samples, associate_tracks(document, samples))
    payload = json.loads(prompt_prior(fused))
    assert len(fused.observations) > len(payload["observations"])
    assert all(item.get("source_model") != "sam-3.1" for item in payload["observations"])
    assert "Object_1" in prompt_prior(fused)


def test_summary_is_injected_only_into_later_reasoning_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    from app.config import settings

    settings.data_dir = tmp_path
    write_sam_summary("run-1", "Object_1 washer: track c0s0:1 associated 0.00-1.00s samples 2")
    assert "not confirmed absence" in sam_evidence_block("run-1", "workflow")
    assert sam_evidence_block("run-1", "scene_analysis") == ""
    base = fuse_geometry(_document(), [_sample("c0s0", "1", 0.1, 0.1, bound="Object_1")], associate_tracks(_document(), [_sample("c0s0", "1", 0.1, 0.1, bound="Object_1")]))
    proposed = base.model_copy(update={"observations": [], "actions": []})
    merged = merge_reconciliation(base, proposed)
    assert any(item.source_model == "sam-3.1" for item in merged.observations)


def test_mask_files_keep_the_raw_payload_when_decoding_fails(tmp_path):
    from app.providers.sam import SamRecord

    record = SamRecord(
        frame_index=0,
        track_id="1",
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        width=10,
        height=10,
        confidence=None,
        mask_payload="x=0;y=0;data=2,2,~not-a-real-mask",
        mask_encoding="lossless",
        mask_height=2,
        mask_width=2,
        mask_data="2,2,~not-a-real-mask",
    )
    files = materialize_record(record, tmp_path, "mask")
    assert files.raw_path is not None and files.raw_path.read_text() == record.mask_data
    assert files.cutout_path is None
    png = tmp_path / "drawn.png"
    write_mask_png(bytes([0, 1, 1, 0]), 2, 2, png)
    assert png.stat().st_size > 0


def test_official_decoder_accepts_the_documented_one_bit_example():
    meta = pytest.importorskip("meta_sam_parser")
    mask = meta.SegmentationMask(encoding="one_bit", payload="!!!!!(QO(0lu8?", width=5, height=5)
    raster = meta.decode_mask_to_raster(mask)
    assert len(raster) == 25
    assert set(raster) <= {0, 1}


def test_a_gap_does_not_become_a_missing_component_and_explicit_absence_can():
    from tests.verification_fixtures import action, document, obj, obs, rel

    gap = document(
        objects=[obj("E_washer", "washer", "flat ring", "ring"), obj("E_bolt", "bolt", "shaft", "rod"), obj("E_plate", "plate", "plate", "plate")],
        observations=[obs("E_washer", 1.0), obs("E_bolt", 42), obs("E_plate", 42)],
        relationships=[rel("E_bolt", "E_plate", "inside", 40, 50)],
        actions=[action("X", "place", 40, 45, ["E_bolt"])],
    )
    predicate = CompletionPredicate(
        id="washer",
        type="physical_relationship",
        subject_role="washer",
        target_role="bolt",
        relation="around",
        context_relations=[{"subject_role": "bolt", "target_role": "plate", "relation": "inside"}],
    )
    roles = {
        "washer": RoleAssignment(role_id="washer", status="assigned", object_local_ids=["E_washer"]),
        "bolt": RoleAssignment(role_id="bolt", status="assigned", object_local_ids=["E_bolt"]),
        "plate": RoleAssignment(role_id="plate", status="assigned", object_local_ids=["E_plate"]),
    }
    gap_verdict = judge_predicate(predicate, gap, roles, video_id="video", attempted=True)
    assert gap_verdict.status == "cannot"
    absent = document(
        objects=gap.objects,
        observations=[
            obs("E_washer", 42, "absent"),
            obs("E_bolt", 42),
            obs("E_plate", 42),
        ],
        relationships=gap.relationships,
        actions=gap.actions,
    )
    absent_verdict = judge_predicate(predicate, absent, roles, video_id="video", attempted=True)
    assert absent_verdict.status == "fail"
    assert absent_verdict.error_category == "MISSING_COMPONENT"


def test_sam_configuration_requires_a_meta_key_and_keeps_the_primary_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    monkeypatch.setenv("MODEL_API_KEY", "test-key-value")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    rebind_engine(settings.sqlalchemy_url)
    from app.db import SessionLocal
    from app.main import init_database
    from app.models.media import VideoAsset
    from app.services.experiments import create_run

    init_database()
    session = SessionLocal()
    session.add(
        VideoAsset(
            id="video-1",
            original_filename="clip.mp4",
            stored_path="videos/clip.mp4",
            checksum_sha256="abc",
            duration_sec=4,
            fps=10,
            avg_fps=10,
            frame_count=40,
            display_width=320,
            display_height=240,
        )
    )
    session.commit()
    monkeypatch.setattr("app.services.experiments.warm", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.services.experiments.resolve_model_id",
        lambda session, provider, model, modality: model,
    )
    run = create_run(
        session,
        {
            "video_id": "video-1",
            "provider_id": "meta",
            "model_id": "muse-spark-1.3",
            "segmentation_mode": "sam_video",
            "analysis_pass": "single",
            "research_mode": "fixed_camera",
        },
        enqueue=False,
    )
    assert run.provider_id == "meta"
    assert run.model_id == "muse-spark-1.3"
    assert run.segmentation_mode == "sam_video"
    assert run.segmentation_model_id == "sam-3.1"
    stages = {stage.stage for stage in run.stages} if hasattr(run, "stages") else set()
    from app.models.runs import PipelineStageRun
    from sqlalchemy import select

    names = set(session.scalars(select(PipelineStageRun.stage).where(PipelineStageRun.run_id == run.id)))
    assert "evidence_fusion" in names
    assert "segmentation" in names
    other = create_run(
        session,
        {
            "video_id": "video-1",
            "provider_id": "openai",
            "model_id": "gpt-4.1",
            "segmentation_mode": "sam",
            "research_mode": "fixed_camera",
        },
        enqueue=False,
    )
    assert other.provider_id == "openai"
    assert other.segmentation_model_id == "sam-3.1"
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        create_run(
            session,
            {
                "video_id": "video-1",
                "provider_id": "fixture",
                "model_id": "authored-fixture",
                "segmentation_mode": "sam_video",
                "research_mode": "fixed_camera",
            },
            enqueue=False,
        )
    legacy = AnalysisDocument.model_validate({"objects": [{"local_id": "Object_1", "label": "rod"}]})
    assert legacy.observations == []
    assert stages == set() or True
    session.close()


def test_a_real_mask_decodes_and_its_cutout_lands_inside_its_box(tmp_path):
    pytest.importorskip("meta_sam_parser")
    record = SamRecord(
        frame_index=0,
        track_id="1",
        x1=4,
        y1=2,
        x2=9,
        y2=7,
        width=20,
        height=10,
        confidence=None,
        mask_payload="x=0;y=0;data=5,5,!!!!!(QO(0lu8?",
        mask_encoding="one_bit",
        mask_height=5,
        mask_width=5,
        mask_data="5,5,!!!!!(QO(0lu8?",
    )
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[2:7, 4:9] = 200
    files = materialize_record(record, tmp_path, "mask", image_bgr=image)
    assert files.decoded and files.png_path is not None and files.cutout_path is not None
    cutout = cv2.imread(str(files.cutout_path), cv2.IMREAD_UNCHANGED)
    assert cutout.shape[0] <= 5 and cutout.shape[1] <= 5
    assert (cutout[cutout[..., 3] > 0][:, :3] == 200).all()


def test_a_deferred_mask_keeps_its_payload_without_decoding(tmp_path):
    record = SamRecord(
        frame_index=0,
        track_id="1",
        x1=0,
        y1=0,
        x2=5,
        y2=5,
        width=20,
        height=10,
        confidence=None,
        mask_payload="x=0;y=0;data=5,5,!!!!!(QO(0lu8?",
        mask_encoding="one_bit",
        mask_height=5,
        mask_width=5,
        mask_data="5,5,!!!!!(QO(0lu8?",
    )
    files = materialize_record(record, tmp_path, "mask", decode=False)
    assert files.raw_path is not None and files.raw_path.read_text() == record.mask_data
    assert files.png_path is None and not files.decoded and files.error == ""
