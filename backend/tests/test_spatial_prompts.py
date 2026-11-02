"""Seeds become exact prompt tokens on exact frames of the submitted media."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from app.perception.sam_fusion import plan_prompts
from app.perception.seeds import Seed
from app.perception.spatial_prompts import (
    MediaMap,
    SeedMappingError,
    billed_frames,
    box_mark,
    build_spatial_plan,
    choose_direction,
    is_spatial_track,
    plan_for_strategy,
)
from app.providers.sam import VideoChunk, video_request
from app.schemas.analysis import AnalysisDocument, ObjectPrediction
from app.services import recovery

BOX = {"x": 0.25, "y": 0.5, "w": 0.25, "h": 0.25}


def _seed(local_id="Object_1", time=1.0, box=BOX, points=None, trust="human_verified") -> Seed:
    return Seed(object_local_id=local_id, timestamp=time, bbox=box, points=points or [], trust=trust, source="human")


def _media(width=640, height=360, count=10, step=0.5, start=0.0) -> MediaMap:
    return MediaMap(width, height, [start + index * step for index in range(count)])


def test_a_single_box_uses_the_documented_grammar_and_inclusive_corners():
    plan = build_spatial_plan([(1, _seed())], _media(), "forward", "rec1r0")
    assert plan.prompt == "<2f>1<|box;x1=160;y1=180;x2=319;y2=269;w=640;h=360|>"
    assert plan.kind == "spatial"
    assert plan.propagation == "forward"
    assert plan.bindings == {1: "Object_1"}
    body = video_request(plan.prompt, file_id="f", prompt_kind="spatial", propagation_direction="forward")
    assert body["metadata"]["propagation_direction"] == "forward"


def test_a_box_touching_the_edge_ends_on_the_last_pixel():
    mark = box_mark({"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}, 0, 1, 640, 360)
    assert (mark.x2, mark.y2) == (639, 359)


def test_normalized_boxes_map_to_the_pixels_they_cover_at_any_size():
    for width, height in [(640, 360), (1920, 1080), (1080, 1920)]:
        mark = box_mark(BOX, 0, 1, width, height)
        assert mark.x1 == round(0.25 * width)
        assert mark.x2 - mark.x1 + 1 == round(0.25 * width)
        assert mark.y2 - mark.y1 + 1 == round(0.25 * height)


def test_a_rotated_recording_uses_the_displayed_size():
    """A 1920x1080 file rotated 90 degrees displays as 1080x1920. Tokens carry that size."""
    plan = build_spatial_plan([(1, _seed())], _media(width=1080, height=1920), "both", "s")
    assert "w=1080;h=1920" in plan.prompt


def test_identical_labels_keep_distinct_object_ids():
    plan = build_spatial_plan([(1, _seed("Object_1")), (2, _seed("Object_2", box={"x": 0, "y": 0, "w": 0.1, "h": 0.1}))], _media(), "both", "s")
    assert plan.prompt.startswith("<2f>1<|box;")
    assert ",2<|box;" in plan.prompt
    assert plan.bindings == {1: "Object_1", 2: "Object_2"}


def test_one_sam_id_cannot_stand_for_two_objects():
    with pytest.raises(SeedMappingError):
        build_spatial_plan([(1, _seed("Object_1")), (1, _seed("Object_2", time=2.0))], _media(), "both", "s")


def test_multiple_seed_frames_for_one_object_share_its_id_in_order():
    plan = build_spatial_plan([(1, _seed(time=3.0)), (1, _seed(time=1.0))], _media(), "forward", "s")
    assert plan.prompt.index("<2f>") < plan.prompt.index("<6f>")
    assert plan.prompt.count("1<|box;") == 2


def test_positive_and_negative_points_serialize():
    points = [{"x": 0.5, "y": 0.5}, {"x": 0.1, "y": 0.1, "negative": True}]
    plan = build_spatial_plan([(1, _seed(box=None, points=points))], _media(), "both", "s")
    assert plan.prompt == "<2f>1<|point;x=320;y=180;w=640;h=360|>-<|point;x=64;y=36;w=640;h=360|>"


def test_only_a_negative_point_names_nothing():
    with pytest.raises(SeedMappingError):
        build_spatial_plan([(1, _seed(box=None, points=[{"x": 0.5, "y": 0.5, "negative": True}]))], _media(), "both", "s")


def test_an_out_of_bounds_seed_fails_with_a_reason():
    with pytest.raises(SeedMappingError, match="outside the frame"):
        box_mark({"x": 0.9, "y": 0.0, "w": 0.3, "h": 0.1}, 0, 1, 640, 360)


def test_frame_index_is_relative_to_the_clip_not_the_recording():
    plan = build_spatial_plan([(1, _seed(time=12.5))], _media(start=12.0), "forward", "s")
    assert plan.prompt.startswith("<1f>")


def test_variable_frame_rate_uses_real_frame_times_not_a_nominal_rate():
    media = MediaMap(640, 360, [0.0, 0.1, 0.15, 0.9, 1.0])
    assert media.frame_for(0.9) == 3


def test_a_time_between_frames_is_refused_rather_than_moved():
    with pytest.raises(SeedMappingError, match="nearest submitted frame"):
        MediaMap(640, 360, [0.0, 0.1, 0.9]).frame_for(0.5)
    with pytest.raises(SeedMappingError):
        _media(start=12.0).frame_for(1.0)


def test_estimated_frame_times_cannot_place_a_seed():
    with pytest.raises(SeedMappingError, match="nominal rate"):
        MediaMap(640, 360, [0.0, 1.0], time_basis="container").frame_for(0.0)


def test_direction_follows_the_seed_relative_to_the_gap():
    assert choose_direction(1.0, 2.0, 3.0) == "forward"
    assert choose_direction(4.0, 2.0, 3.0) == "backward"
    assert choose_direction(2.5, 2.0, 3.0) == "both"


def test_billing_estimate_counts_tracked_frames_with_the_documented_floor():
    assert billed_frames(30, 200, "forward") == 170
    assert billed_frames(120, 200, "backward") == 121
    assert billed_frames(30, 200, "both") == 200
    assert billed_frames(5, 20, "forward") == 50


def test_backward_needs_a_prompt_after_the_first_frame():
    with pytest.raises(SeedMappingError, match="after the first"):
        build_spatial_plan([(1, _seed(time=0.0))], _media(), "backward", "s")


def test_prompted_track_ids_are_recognized_and_concept_ids_are_not():
    assert is_spatial_track("c0p0:1")
    assert is_spatial_track("rec12abr0:1")
    assert not is_spatial_track("c0s0:1")


def _document() -> AnalysisDocument:
    return AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="washer"), ObjectPrediction(local_id="Object_2", label="bolt")]
    )


def _chunk(times=(0.0, 0.5, 1.0, 1.5)) -> VideoChunk:
    return VideoChunk(0, 0, len(times), times[0], times[-1] + 0.5, "c0", list(times))


def test_concept_strategy_matches_the_existing_planner():
    plans, outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "concept", [_seed()])
    assert [(item.kind, item.prompt, item.session_id) for item in plans] == [
        (item.kind, item.prompt, item.session_id) for item in plan_prompts(_document(), _chunk(), 640, 360, 2.0)
    ]
    assert outcomes == []


def test_trusted_spatial_prompts_seeded_objects_and_leaves_the_rest_unobserved():
    plans, outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "trusted_spatial", [_seed()])
    assert [item.kind for item in plans] == ["spatial"]
    assert {item["object_local_id"]: item["prompt"] for item in outcomes} == {"Object_1": "spatial", "Object_2": "unobserved"}


def test_hybrid_keeps_concept_and_spatial_in_separate_requests_without_duplicate_objects():
    plans, _outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "hybrid", [_seed()])
    assert [item.kind for item in plans] == ["spatial", "concept"]
    assert plans[1].prompt == "bolt"
    assert "<" not in plans[1].prompt
    assert plans[1].targets == ["Object_2"]


def test_hybrid_falls_back_when_a_seed_cannot_be_placed():
    plans, outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "hybrid", [_seed(time=0.25)])
    assert [item.kind for item in plans] == ["concept", "concept"]
    assert any(item["prompt"] == "seed_rejected" for item in outcomes)


def test_a_candidate_is_never_prompted_automatically():
    plans, _outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "trusted_spatial", [_seed(trust="candidate")])
    assert plans == []


def test_a_seed_on_a_later_chunk_does_not_prompt_this_one():
    plans, _outcomes = plan_for_strategy(_document(), _chunk(), 640, 360, 2.0, "trusted_spatial", [_seed(time=30.0)])
    assert plans == []


ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)


def _index_frame(index: int) -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :] = (index * 7 % 255, 40, 200 - index % 150)
    return image


def _vfr_source(path: Path, count: int = 60) -> list[float]:
    """Frames at uneven times. The second half runs at half the rate of the first."""
    from app.demo.encode import write_clip
    from app.video.probe import presentation_timestamps

    plain = path.with_name("plain.mp4")
    write_clip([_index_frame(item) for item in range(count)], plain, fps=30)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(plain),
            "-vf", "setpts='if(lt(N,30),N/(30*TB),(30+(N-30)*2)/(30*TB))'",
            "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )  # fmt: skip
    raw = presentation_timestamps(path)
    return [round(item - raw[0], 6) for item in raw]


@ffmpeg
def test_a_variable_frame_rate_clip_keeps_its_source_mapping(tmp_path: Path):
    source = tmp_path / "vfr.mp4"
    times = _vfr_source(source)
    assert times[40] - times[39] > 1.5 * (times[10] - times[9])
    wanted = times[20:45]
    media = recovery.prepare_clip(source, wanted, tmp_path / "clip.mp4", 640, 360)
    assert media["frames"] == 25
    assert media["source_start"] == wanted[0]


@ffmpeg
def test_a_clip_whose_frames_do_not_line_up_is_refused(tmp_path: Path):
    source = tmp_path / "vfr.mp4"
    times = _vfr_source(source)
    clip = tmp_path / "clip.mp4"
    recovery.prepare_clip(source, times[20:45], clip, 640, 360)
    with pytest.raises(recovery.RecoveryError, match="ambiguous"):
        recovery.verify_clip(clip, times[20:44], 640, 360)
    wrong_spacing = [times[20] + index * 0.2 for index in range(25)]
    with pytest.raises(recovery.RecoveryError, match="ambiguous"):
        recovery.verify_clip(clip, wrong_spacing, 640, 360)


@ffmpeg
def test_a_rotated_source_is_cut_upright_at_its_displayed_size(tmp_path: Path):
    from app.demo.encode import write_clip
    from app.video.probe import presentation_timestamps, probe

    plain = tmp_path / "plain.mp4"
    write_clip([_index_frame(item) for item in range(40)], plain, fps=30)
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(plain), "-c", "copy", str(rotated)],
        check=True,
    )
    metadata = probe(rotated)
    assert abs(metadata.rotation_deg) % 180 == 90
    width, height = metadata.display_width, metadata.display_height
    assert (width, height) == (360, 640)
    raw = presentation_timestamps(rotated)
    times = [round(item - raw[0], 6) for item in raw][5:25]
    recovery.prepare_clip(rotated, times, tmp_path / "clip.mp4", width, height)
    cut = probe(tmp_path / "clip.mp4")
    assert (cut.width, cut.height, cut.rotation_deg % 360) == (360, 640, 0)


@ffmpeg
def test_a_clip_of_the_wrong_size_is_refused(tmp_path: Path):
    source = tmp_path / "vfr.mp4"
    times = _vfr_source(source)
    with pytest.raises(recovery.RecoveryError, match="displays at"):
        recovery.prepare_clip(source, times[:20], tmp_path / "clip.mp4", 1280, 720)
