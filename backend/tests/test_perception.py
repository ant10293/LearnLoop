"""Identity tracking, embeddings, and evaluation without a paid model or a weight download."""

from __future__ import annotations

import os

import certifi
import numpy as np
import pytest
from app.perception.assignment import linear_assignment
from app.perception.encoder import (
    Dinov2Encoder,
    HistogramEncoder,
    cache_key,
    cached_encode,
    load_encoder,
    probe_encoder,
)
from app.perception.identity_eval import evaluate_identity
from app.perception.legacy import reassociate_archived
from app.perception.memory import gallery_cosine, reference_state
from app.perception.presets import config_from_mapping, preset
from app.perception.tracker import track_frames
from app.perception.types import Detection, EncoderUnavailable, FrameView, TrackPoint
from app.pipeline.merge import restore_omitted_records
from app.schemas.analysis import AnalysisDocument


class ScriptedDetector:
    detector_id = "scripted"

    def __init__(self, frames):
        self.frames = frames

    def detect(self, frame, context):
        return self.frames.get(context.index, [])


class MeanEncoder:
    encoder_id = "test_mean_color"
    encoder_version = "v1"
    preprocess_version = "raw-mean-v1"
    dimension = 3

    def encode(self, crops):
        rows = []
        for crop in crops:
            if crop.size == 0:
                rows.append(np.zeros(3, dtype=np.float32))
                continue
            mean = crop.reshape(-1, crop.shape[-1])[:, :3].mean(axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean))
            rows.append(mean / norm if norm else mean)
        return np.stack(rows)


class BrokenEncoder(MeanEncoder):
    encoder_id = "dinov2_vits14"

    def encode(self, crops):
        raise EncoderUnavailable("PyTorch is not installed. Histogram features were not substituted.")


def _frame(index, timestamp, boxes, color=(30, 30, 30), size=96, scene=False):
    image = np.full((size, size, 3), 230, np.uint8)
    detections = []
    for box in boxes:
        x, y, w, h = box
        image[y : y + h, x : x + w] = color
        detections.append(
            Detection(
                timestamp=timestamp,
                bbox=(x / size, y / size, w / size, h / size),
                detector_id="scripted",
            )
        )
    view = FrameView(timestamp=timestamp, image=image, index=index, scene_change=scene)
    return view, detections


def _run(specs, encoder=None, **config):
    views = []
    detections = {}
    for index, spec in enumerate(specs):
        view, found = _frame(index, spec["t"], spec["boxes"], color=spec.get("color", (30, 30, 30)), scene=spec.get("scene", False))
        views.append(view)
        detections[index] = found
    result = track_frames(
        views,
        config=config_from_mapping({"strategy": "temporal", "detector": "scripted", "tentative_hits": 2, **config}),
        detector=ScriptedDetector(detections),
        encoder=encoder,
        research_mode="fixed_camera",
    )
    return result


def _ids_at(result, timestamp):
    return {point.track_ref for point in result.points if abs(point.timestamp - timestamp) < 1e-6 and point.continuity != "uncertain"}


def test_hungarian_is_optimal_and_one_to_one():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    pairs = linear_assignment(cost)
    assert sorted(left for left, _right in pairs) == [0, 1, 2]
    assert sorted(right for _left, right in pairs) == [0, 1, 2]
    chosen = {(left, right) for left, right in pairs}
    assert chosen == {(0, 1), (1, 0), (2, 2)}


def test_motion_scale_and_rotation_keep_one_id():
    motion = _run(
        [
            {"t": 0.0, "boxes": [(10, 40, 16, 16)]},
            {"t": 0.1, "boxes": [(22, 40, 16, 16)]},
            {"t": 0.2, "boxes": [(34, 40, 16, 16)]},
            {"t": 0.3, "boxes": [(46, 40, 16, 16)]},
        ]
    )
    assert len({point.track_ref for point in motion.points}) == 1
    grown = _run(
        [
            {"t": 0.0, "boxes": [(30, 30, 14, 14)]},
            {"t": 0.1, "boxes": [(28, 28, 20, 20)]},
            {"t": 0.2, "boxes": [(26, 26, 26, 26)]},
        ]
    )
    assert len({point.track_ref for point in grown.points}) == 1
    rotated = _run(
        [
            {"t": 0.0, "boxes": [(30, 40, 28, 10)]},
            {"t": 0.1, "boxes": [(36, 32, 16, 24)]},
            {"t": 0.2, "boxes": [(40, 28, 10, 30)]},
        ]
    )
    assert len({point.track_ref for point in rotated.points}) == 1


def test_occlusion_keeps_identity_and_a_long_gap_does_not_confirm_it():
    occluded = _run(
        [
            {"t": 0.0, "boxes": [(20, 40, 16, 16)]},
            {"t": 0.1, "boxes": [(28, 40, 16, 16)]},
            {"t": 0.2, "boxes": []},
            {"t": 0.4, "boxes": [(40, 40, 16, 16)]},
        ]
    )
    visible = [point.track_ref for point in occluded.points if point.visibility == "visible"]
    assert len(set(visible)) == 1
    gap = _run(
        [
            {"t": 0.0, "boxes": [(20, 40, 16, 16)]},
            {"t": 0.1, "boxes": [(28, 40, 16, 16)]},
            {"t": 3.0, "boxes": [(30, 40, 16, 16)]},
        ]
    )
    early = {point.track_ref for point in gap.points if point.timestamp < 1}
    late = {point.track_ref for point in gap.points if point.timestamp > 1}
    assert early.isdisjoint(late)
    assert all(item.decision != "confirmed" for item in gap.perception.hypotheses)


def test_crossing_objects_do_not_swap_and_scene_change_keeps_history():
    crossed = _run(
        [
            {"t": 0.0, "boxes": [(8, 20, 14, 14), (70, 60, 14, 14)]},
            {"t": 0.1, "boxes": [(24, 20, 14, 14), (54, 60, 14, 14)]},
            {"t": 0.2, "boxes": [(40, 20, 14, 14), (38, 60, 14, 14)]},
            {"t": 0.3, "boxes": [(56, 20, 14, 14), (22, 60, 14, 14)]},
        ]
    )
    by_track = {}
    for point in crossed.points:
        by_track.setdefault(point.track_ref, []).append(point)
    assert len(by_track) == 2
    left = min(by_track, key=lambda ref: by_track[ref][0].x)
    right = max(by_track, key=lambda ref: by_track[ref][0].x)
    assert by_track[left][-1].x > by_track[left][0].x
    assert by_track[right][-1].x < by_track[right][0].x
    changed = _run(
        [
            {"t": 0.0, "boxes": [(20, 30, 16, 16)]},
            {"t": 0.1, "boxes": [(32, 30, 16, 16)]},
            {"t": 0.2, "boxes": [(48, 50, 16, 16)], "scene": True},
        ]
    )
    archived = [item for item in changed.perception.instances if item.status == "archived"]
    assert archived
    assert len(changed.perception.instances) == 2
    assert all(item.decision != "confirmed" for item in changed.perception.hypotheses)


def test_similar_objects_are_not_silently_confirmed_and_do_not_poison_memory():
    encoder = MeanEncoder()
    result = _run(
        [
            {"t": 0.0, "boxes": [(8, 20, 18, 18), (60, 60, 18, 18)], "color": (20, 20, 200)},
            {"t": 0.1, "boxes": [(12, 20, 18, 18), (64, 60, 18, 18)], "color": (20, 20, 200)},
            {"t": 0.2, "boxes": [(16, 20, 18, 18), (68, 60, 18, 18)], "color": (20, 20, 200)},
            {"t": 2.0, "boxes": [(40, 40, 18, 18)], "color": (20, 20, 200)},
        ],
        encoder=encoder,
        strategy="appearance",
    )
    late = [point for point in result.points if point.timestamp > 1]
    early = {point.track_ref for point in result.points if point.timestamp < 1}
    assert late
    assert late[0].continuity != "continuous" or late[0].track_ref not in early
    assert all(item.decision != "confirmed" for item in result.perception.hypotheses)
    assert any(item.trusted_references >= 1 for item in result.perception.instances if item.instance_id in early)
    assert all(
        item.timestamp < 1
        for item in result.perception.appearance_references
        if item.state == "trusted"
    )


def test_reference_rules_and_embedding_cache(tmp_path):
    assert reference_state(human=False, temporal_strong=False, assigned_reid=False) is None
    assert reference_state(human=False, temporal_strong=True, assigned_reid=False) == "trusted"
    assert reference_state(human=False, temporal_strong=False, assigned_reid=True) == "candidate"
    trusted = [np.array([1.0, 0.0], dtype=np.float32)]
    assert gallery_cosine(trusted, np.array([1.0, 0.0], dtype=np.float32)) == 1
    encoder = HistogramEncoder()
    crop = np.full((8, 8, 3), 40, np.uint8)
    calls = {"n": 0}
    original = encoder.encode

    def counting(crops):
        calls["n"] += 1
        return original(crops)

    encoder.encode = counting
    first = cached_encode(encoder, [crop], [None], tmp_path)
    second = cached_encode(encoder, [crop], [None], tmp_path)
    assert calls["n"] == 1
    assert first.shape == (1, 24)
    assert abs(float(np.linalg.norm(first[0])) - 1) < 1e-5
    assert np.allclose(first, second)
    other = cache_key(crop, None, encoder.encoder_id, "other-version", encoder.preprocess_version)
    current = cache_key(crop, None, encoder.encoder_id, encoder.encoder_version, encoder.preprocess_version)
    assert other != current
    broken = BrokenEncoder()
    try:
        broken.encode([crop])
        raised = False
    except EncoderUnavailable:
        raised = True
    assert raised
    status = probe_encoder("dinov2_vits14")
    assert status["weights_loaded"] is False
    try:
        load_encoder("missing-encoder")
        missing = False
    except EncoderUnavailable:
        missing = True
    assert missing


def test_dinov2_load_verifies_downloads_without_a_prompt(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = {}

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

    def fake_load(repo, name, **kwargs):
        seen.update(kwargs, repo=repo, name=name, cert=os.environ.get("SSL_CERT_FILE"))
        return Model()

    monkeypatch.setenv("SSL_CERT_FILE", "placeholder")
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.setattr(torch.hub, "load", fake_load)
    Dinov2Encoder()._load()
    assert seen["repo"] == "facebookresearch/dinov2"
    assert seen["trust_repo"] is True
    assert seen["cert"] == certifi.where()


def test_legacy_scene_history_does_not_replace_the_new_id():
    fresh = [{"ref": "track_2", "sig": np.ones(8, dtype=np.float32)}]
    archived = [{"ref": "track_1", "sig": np.ones(8, dtype=np.float32)}]
    points = [
        TrackPoint("track_1", 0.0, 0.1, 0.1, 0.2, 0.2, 0, "new", False, 0, status="archived"),
        TrackPoint("track_2", 1.0, 0.4, 0.4, 0.2, 0.2, 0, "new", False, 1),
    ]
    reassociate_archived(points, fresh, archived, 1)
    assert points[1].track_ref == "track_2"
    assert points[1].continuity == "reassociated"
    assert "track_1" in points[1].note
    weak = [{"ref": "track_4", "sig": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)}]
    prior = [{"ref": "track_3", "sig": np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)}]
    later = [TrackPoint("track_4", 2.0, 0.1, 0.1, 0.2, 0.2, 0, "new", False, 2)]
    reassociate_archived(later, weak, prior, 2)
    assert later[0].continuity == "uncertain"
    assert later[0].track_ref == "track_4"


def test_labels_do_not_merge_distinct_instances_and_unknown_ids_stay_provisional():
    from app.perception.evidence import evidence_text
    from app.perception.ground import apply_perception
    from app.schemas.enums import EvidenceClass, Provenance

    evidence = {
        "classification": EvidenceClass.DIRECTLY_OBSERVED.value,
        "provenance": Provenance.AI_PREDICTION.value,
        "frame_timestamps": [0.0],
    }
    partial = AnalysisDocument.model_validate(
        {
            "objects": [
                {"local_id": "Object_1", "label": "red brick", "instance_id": "track_1"},
                {"local_id": "Object_2", "label": "red brick", "instance_id": "track_2"},
            ],
            "states": [
                {
                    "object_local_id": "Object_2",
                    "time_start": 0,
                    "time_end": 1,
                    "state_label": "separate",
                    "evidence": evidence,
                }
            ],
        }
    )
    merged = AnalysisDocument.model_validate(
        {
            "objects": [
                {"local_id": "Object_9", "label": "red brick", "instance_id": "track_1"},
                {"local_id": "Object_8", "label": "red brick", "instance_id": "track_2"},
            ]
        }
    )
    restored = restore_omitted_records(merged, [partial, partial])
    states = {item.object_local_id for item in restored.states}
    assert states == {"Object_8"}
    spatial = AnalysisDocument.model_validate(
        {
            "perception": {
                "strategy": "temporal",
                "detector_id": "proposals",
                "used_for_identity": True,
                "instances": [
                    {
                        "instance_id": "track_1",
                        "status": "tracked",
                        "first_seen": 0,
                        "last_seen": 1,
                    }
                ],
            }
        }
    )
    grounded = apply_perception(
        AnalysisDocument.model_validate(
            {"objects": [{"local_id": "Object_3", "label": "unknown part", "instance_id": "track_9"}]}
        ),
        spatial,
    )
    assert grounded.objects[0].identity_relation == "provisional"
    assert grounded.objects[0].uncertain_identity is True
    text = evidence_text(spatial)
    assert "track_1" in text
    assert "provisional" in text


def test_identity_metrics_count_switches_merges_and_withhold_labels():
    truth = {
        "observations": [
            {"instance_id": "gold_a", "timestamp": 0.0, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
            {"instance_id": "gold_a", "timestamp": 1.0, "bbox": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}},
            {"instance_id": "gold_b", "timestamp": 0.0, "bbox": {"x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2}},
        ]
    }
    predicted = {
        "observations": [
            {"instance_id": "track_1", "timestamp": 0.0, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}, "continuity": "continuous"},
            {"instance_id": "track_2", "timestamp": 1.0, "bbox": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}, "continuity": "uncertain"},
            {"instance_id": "track_1", "timestamp": 0.0, "bbox": {"x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2}, "continuity": "continuous"},
        ],
        "perception": {"hypotheses": [{"decision": "ambiguous"}, {"decision": "rejected"}]},
    }
    report = evaluate_identity(predicted, truth)
    assert report["evaluated"] is True
    assert report["identity_switches"] == 1
    assert report["false_merges"] == 1
    assert report["abstention_rate"] > 0
    assert report["hypothesis_resolution"]["ambiguous"] == 1
    assert any(item["timestamp"] == 1.0 for item in report["failures"])
    withheld = evaluate_identity(
        {"observations": [{"instance_id": "track_1", "timestamp": 0, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]},
        {"objects": [{"local_id": "Object_1", "label": "red brick"}], "observations": []},
    )
    assert withheld["evaluated"] is False
    again = evaluate_identity(predicted, truth)
    assert again == report


def test_confirmed_hypothesis_promotes_only_that_candidate():
    from app.perception.corrections import _promote_reference
    from app.schemas.analysis import AppearanceRefRecord, HypothesisRecord, InstanceRecord

    hypothesis = HypothesisRecord(
        hypothesis_id="hyp_1",
        timestamp=2.0,
        subject_instance_id="track_1",
        candidate_instance_id="track_1",
        association_score=0.8,
        decision="proposed",
    )
    references = [
        AppearanceRefRecord(instance_id="track_1", timestamp=2.0, state="candidate"),
        AppearanceRefRecord(instance_id="track_1", timestamp=0.2, state="candidate"),
        AppearanceRefRecord(instance_id="track_2", timestamp=2.0, state="candidate"),
    ]
    instances = [InstanceRecord(instance_id="track_1", status="tracked", first_seen=0, last_seen=2, candidate_references=2)]
    updated, rows = _promote_reference(hypothesis, references, instances)
    assert [item.state for item in updated] == ["trusted", "candidate", "candidate"]
    assert updated[0].confirmed is True
    assert rows[0].trusted_references == 1
    assert rows[0].candidate_references == 1


def test_presets_are_distinct_and_do_not_substitute():
    legacy = preset("A_legacy_motion")
    temporal = preset("B_temporal")
    appearance = preset("C_appearance")
    segmented = preset("D_appearance_segmented")
    assert legacy.encoder is None and legacy.strategy == "legacy_motion"
    assert temporal.encoder is None and temporal.detector == "proposals"
    assert appearance.encoder == "dinov2_vits14" and appearance.segment is False
    assert segmented.segment is True
    try:
        preset("yolo_magic")
        failed = False
    except ValueError:
        failed = True
    assert failed


def test_proposal_detector_shrinks_phone_frames():
    import time

    from app.perception.detect import PROPOSAL_MAX_EDGE, ProposalDetector, _mser_view
    from app.perception.types import DetectionContext

    small, scale = _mser_view(np.zeros((360, 640), np.uint8))
    assert scale == 1.0
    assert small.shape == (360, 640)

    view, scale = _mser_view(np.zeros((3840, 2160), np.uint8))
    assert scale < 1
    assert max(view.shape) == PROPOSAL_MAX_EDGE

    frame = np.zeros((3840, 2160, 3), np.uint8)
    frame[800:2800, 600:1500] = 220
    detector = ProposalDetector(False)
    started = time.perf_counter()
    found = detector.detect(
        frame,
        DetectionContext(
            timestamp=1.0,
            index=1,
            scene_chunk=0,
            research_mode="fixed_camera",
        ),
    )
    assert time.perf_counter() - started < 1
    assert all(0 <= item.bbox[0] <= 1 and 0 <= item.bbox[1] <= 1 for item in found)
