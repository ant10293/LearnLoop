"""Track loss, visual segments, and identity. Synthetic sequences with no provider and no accuracy claim."""

from __future__ import annotations

from types import SimpleNamespace

from app.perception.recovery_validate import (
    ValidationPolicy,
    combine_passes,
    validate_track,
)
from app.perception.track_segments import link_identity, track_report, unresolved_spans
from app.schemas.analysis import AnalysisDocument, ObjectPrediction
from app.services.recovery import mask_stats, merge_proposal

SEED = {"seed_id": "seed-a", "timestamp": 0.0, "bbox": {"x": 0.1, "y": 0.4, "w": 0.1, "h": 0.1}, "trust": "human_verified"}


def _s(time, x=0.1, y=0.4, w=0.1, h=0.1, stats=None):
    return {
        "timestamp": time,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "confidence": 0.9,
        "frame_index": round(time * 10),
        "source_width": 640,
        "source_height": 360,
        "mask_data": "1,1,!A",
        "mask_payload": "",
        "mask_encoding": "one_bit",
        "mask_height": 1,
        "mask_stats": stats,
    }


def _v(samples, *, seed=SEED, anchors=(), neighbors=(), boundaries=None, policy=None):
    return validate_track(
        samples,
        seed=seed,
        association="associated",
        anchors=list(anchors),
        neighbors=list(neighbors),
        media_size=(640, 360),
        boundaries=boundaries,
        policy=policy,
    )


def _anchor(time, x=0.1):
    return {"timestamp": time, "bbox": {"x": x, "y": 0.4, "w": 0.1, "h": 0.1}, "source": "tracker"}


def _pass(rows, seed=SEED, name="recabc:1:seed-a", frames=None, direction="forward", session="recabc"):
    for row in rows:
        row["seed_id"] = seed["seed_id"]
        row["segment_id"] = f"{name}/{row['segment']}" if row["segment"] else ""
    return {
        "rows": rows,
        "seed": seed,
        "pass_id": name,
        "experiment_id": "e1",
        "request_key": "k1",
        "session_id": session,
        "raw_track_id": "1",
        "request_id": "r",
        "direction": direction,
        "frame_times": frames or [row["timestamp"] for row in rows],
    }


TARGET = {"object_local_id": "Object_6", "instance_id": "instance_6", "time_start": 0.0, "time_end": 2.0}
STEADY = [_s(t / 10, x=0.1 + t * 0.002) for t in range(6)]


# ---- Loss detection ----


def test_smooth_movement_stays_tracked():
    rows = _v(STEADY)
    assert {row["lifecycle"] for row in rows} == {"tracked"}
    assert not any(row["failure"] for row in rows)


def test_one_noisy_box_is_an_outlier_and_does_not_end_the_track():
    samples = [*STEADY, _s(0.6, x=0.8), _s(0.7, x=0.114), _s(0.8, x=0.116)]
    rows = _v(samples)
    assert not any(row["failure"] for row in rows)
    assert rows[6]["outcome"] == "conflicting" and rows[6]["lifecycle"] == "degraded"
    assert rows[7]["outcome"] == "supported" and rows[7]["segment"] == "f0"


def test_a_sustained_jump_closes_the_segment_and_records_why():
    samples = [*STEADY, *[_s(0.6 + t / 10, x=0.8) for t in range(6)]]
    rows = _v(samples)
    failure = rows[6]["failure"]
    assert failure["reason"] == "large_spatial_jump" and failure["last_good_time"] == 0.5
    assert all(row["lifecycle"] == "lost" and row["segment"] is None and row["outcome"] == "conflicting" for row in rows[6:])
    assert [row["outcome"] for row in rows[:6]] == ["supported"] * 6


def test_motion_is_measured_against_time_so_uneven_frames_do_not_look_like_jumps():
    fast = _v([_s(t / 10, x=0.1 + t * 0.01) for t in range(6)] + [_s(1.5, x=0.25)])
    assert not any(row["failure"] for row in fast)


def test_a_fast_step_over_a_short_interval_is_a_jump_even_at_a_higher_frame_rate():
    samples = [_s(t / 100, x=0.1 + t * 0.0001) for t in range(8)] + [_s(0.08 + t / 100, x=0.6) for t in range(6)]
    assert any(row["failure"] for row in _v(samples))


def test_a_box_growing_over_another_part_is_flagged_and_keeps_the_washer_separate():
    neighbor = [{"timestamp": 0.1 * t, "bbox": {"x": 0.2, "y": 0.4, "w": 0.1, "h": 0.1}, "object_local_id": "rod"} for t in range(1, 6)]
    rows = _v([_s(0.0), *[_s(t / 10, w=0.3, h=0.2) for t in range(1, 6)]], neighbors=neighbor)
    assert rows[1]["failure"]["reason"] == "absorbed_neighbor"
    assert all(row["outcome"] == "conflicting" for row in rows[1:])
    assert rows[1]["bbox"]["w"] == 0.3
    identity = link_identity(SEED, rows[:1] + rows[1:], began="")
    assert identity["status"] == "ambiguous"


def test_a_gradual_perspective_change_does_not_end_the_track():
    rows = _v([_s(t / 10, w=0.1 + t * 0.03, h=0.1 + t * 0.03) for t in range(8)])
    assert not any(row["failure"] for row in rows)


def test_large_growth_alone_is_a_warning_and_a_poor_mask_as_well_ends_the_segment():
    good = {"pixels": 100, "fill": 0.8, "components": 1}
    poor = {"pixels": 110, "fill": 0.05, "components": 6}
    grown = [_s(t / 10, w=0.1 + t * 0.03, h=0.1 + t * 0.03, stats=good) for t in range(8)]
    assert not any(row["failure"] for row in _v(grown))
    assert any(item["check"] == "size_drift" and not item["passed"] for item in _v(grown)[-1]["checks"])
    bad = [*grown[:7], *[_s(0.7 + t / 10, w=0.34, h=0.34, stats=poor) for t in range(5)]]
    assert next(row["failure"] for row in _v(bad) if row["failure"])["reason"] == "expanded_beyond_object"


def test_a_missing_mask_is_not_evidence_of_a_lost_object():
    samples = [{**_s(t / 10), "mask_data": "", "mask_encoding": "", "mask_height": 0} for t in range(4)]
    rows = _v(samples)
    assert not any(row["failure"] for row in rows)
    assert {row["outcome"] for row in rows} == {"provisional"}


def test_leaving_at_the_edge_is_a_hypothesis_not_a_finding():
    samples = [_s(t / 10, x=0.7 + t * 0.04) for t in range(6)] + [_s(0.6 + t / 10, x=0.1) for t in range(5)]
    failure = next(row["failure"] for row in _v(samples, seed={**SEED, "bbox": {"x": 0.7, "y": 0.4, "w": 0.1, "h": 0.1}}) if row["failure"])
    assert failure["hypothesis"]["state"] == "out_of_view"
    assert "may" in failure["hypothesis"]["basis"]


def test_hiding_behind_another_part_keeps_identity_and_invents_no_box():
    cover = [{"timestamp": 0.5, "bbox": {"x": 0.05, "y": 0.35, "w": 0.3, "h": 0.3}, "object_local_id": "hand"}]
    frames = [t / 10 for t in range(12)]
    rows = _v([_s(t / 10) for t in range(6)], neighbors=cover)
    report = track_report(TARGET, [_pass(rows, frames=frames)], cover)
    failure = report["failures"][0]
    assert failure["reason"] == "no_detection" and failure["state"] == "occluded"
    assert failure["physical_instance_id"] == "instance_6"
    assert report["no_detection_frames"] == 6
    assert len(rows) == 6


def test_a_scene_cut_does_not_carry_trust_across_it():
    samples = [_s(t / 10) for t in range(5)] + [_s(1.6 + t / 10) for t in range(5)]
    rows = _v(samples, boundaries=[1.0], policy=ValidationPolicy(boundary_window=0.01))
    after = rows[5:]
    assert after[0]["segment"] == "f1" and after[0]["segment_began"] == "scene_cut_unverified"
    assert all(row["outcome"] != "supported" for row in after)
    report = track_report(TARGET, [_pass(rows)], [])
    second = next(item for item in report["segments"] if item["segment_id"].endswith("f1"))
    assert second["status"] == "ambiguous" and second["identity"]["status"] == "hypothesis"


def test_a_jump_across_a_scene_cut_names_the_cut():
    samples = [_s(t / 10) for t in range(5)] + [_s(1.6 + t / 10, x=0.8) for t in range(5)]
    rows = _v(samples, boundaries=[1.0], policy=ValidationPolicy(boundary_window=0.01))
    assert rows[5]["failure"]["reason"] == "scene_cut"


# ---- Segments and reacquisition ----


def _broken_then_confirmed():
    samples = [*STEADY, *[_s(0.6 + t / 10, x=0.8) for t in range(4)], *[_s(1.0 + t / 10, x=0.5) for t in range(5)]]
    return _v(samples, anchors=[_anchor(1.0, x=0.5)])


def test_a_measurement_starts_a_new_segment_that_inherits_no_failure_and_no_support():
    rows = _broken_then_confirmed()
    assert rows[10]["segment"] == "f1" and rows[10]["lifecycle"] == "reacquired" and rows[10]["outcome"] == "supported"
    # The later boxes have only the track to vouch for them, so the old seed does not lift them.
    assert {row["outcome"] for row in rows[11:]} == {"provisional"}
    assert not any(item["check"] == "track_intact" for row in rows[10:] for item in row["checks"])


def test_a_track_splits_into_segments_with_the_gap_left_unresolved():
    rows = _broken_then_confirmed()
    report = track_report(TARGET, [_pass(rows, frames=[t / 10 for t in range(15)])], [])
    names = [item["segment_id"] for item in report["segments"]]
    assert names == ["recabc:1:seed-a/f0", "recabc:1:seed-a/f1"]
    assert report["segments"][0]["status"] == "lost" and report["segments"][1]["status"] == "reacquired"
    assert report["track_breaks"] == 1
    assert report["unresolved"] == [
        {"start": 0.6, "end": 0.9, "frames": 4, "no_detection_frames": 0, "unusable_frames": 4}
    ]
    assert all(item["physical_instance_id"] == "instance_6" for item in report["segments"])


def test_a_stable_wrong_track_never_earns_trust_by_repeating():
    samples = [*STEADY, *[_s(0.6 + t / 10, x=0.8) for t in range(30)]]
    rows = _v(samples)
    assert all(row["outcome"] == "conflicting" for row in rows[6:])
    assert len([row for row in rows if row["failure"]]) == 1


def test_a_new_pass_that_starts_off_its_seed_is_rejected_as_this_object():
    rows = _v([_s(t / 10, x=0.7) for t in range(5)])
    assert rows[0]["failure"]["reason"] == "seed_mismatch"
    assert {row["outcome"] for row in rows} == {"conflicting"}
    assert not any(row["segment"] for row in rows)


def test_repeated_contradiction_by_measurements_ends_a_segment():
    samples = [_s(t / 10) for t in range(5)]
    rows = _v(samples, anchors=[_anchor(0.2, x=0.7), _anchor(0.3, x=0.7)])
    assert rows[3]["failure"]["reason"] == "anchor_contradiction"


def test_a_backward_pass_walks_toward_the_loss_and_records_its_side():
    seed = {**SEED, "timestamp": 1.0, "seed_id": "seed-b"}
    samples = [_s(0.3 + t / 10, x=0.8) for t in range(3)] + [_s(0.6 + t / 10) for t in range(5)]
    rows = _v(samples, seed=seed)
    failure = next(row["failure"] for row in rows if row["failure"])
    assert failure["time"] == 0.5 and failure["last_good_time"] == 0.6
    report = track_report(TARGET, [_pass(rows, seed=seed, name="recb:1:seed-b", direction="backward")], [])
    assert report["failures"][0]["direction"] == "backward"
    assert any(item["direction"] == "backward" for item in report["segments"])


def test_two_passes_from_separate_seeds_agree_and_from_one_seed_do_not_lift_each_other():
    far = 5.0
    first = _v([_s(far)])
    second = _v([_s(far, x=0.101)])
    for row in first:
        row["seed_id"] = "a"
    for row in second:
        row["seed_id"] = "b"
    merged = combine_passes([first, second])[0]
    assert merged["outcome"] == "supported" and len(merged["alternates"]) == 1
    for row in second:
        row["seed_id"] = "a"
    assert combine_passes([first, second])[0]["outcome"] == "provisional"


def test_a_lost_pass_does_not_contaminate_a_valid_segment_and_stays_inspectable():
    lost = _v([*STEADY, *[_s(0.6 + t / 10, x=0.8) for t in range(4)]])
    good_seed = {**SEED, "timestamp": 0.8, "seed_id": "b"}
    valid = _v([_s(0.8), _s(0.9)], seed=good_seed)
    merged = combine_passes([lost, valid])
    row = next(item for item in merged if abs(item["timestamp"] - 0.8) < 1e-6)
    assert row["outcome"] == "supported" and row["lifecycle"] != "lost"
    assert row["alternates"][0]["bbox"]["x"] == 0.8 and row["alternates"][0]["lifecycle"] == "lost"


def test_disagreeing_passes_keep_both_boxes():
    one = _v([_s(5.0)])
    two = _v([_s(5.0, x=0.7)])
    merged = combine_passes([one, two])[0]
    assert merged["outcome"] == "conflicting" and merged["alternates"][0]["bbox"]["x"] == 0.7


# ---- Identity ----


def test_identity_needs_evidence_and_a_label_or_stable_box_is_not_any():
    rows = _v([_s(t / 10) for t in range(3)], seed={**SEED, "trust": "candidate"})
    assert link_identity({**SEED, "trust": "candidate"}, rows, began="")["status"] == "hypothesis"
    assert link_identity(SEED, _v(STEADY), began="")["status"] == "linked"


def test_a_competing_look_alike_leaves_identity_open():
    other = [{"timestamp": t / 10, "bbox": {"x": 0.1, "y": 0.4, "w": 0.1, "h": 0.1}, "object_local_id": "Object_2"} for t in range(3)]
    rows = _v([_s(t / 10) for t in range(3)], neighbors=other)
    assert link_identity(SEED, rows, began="")["status"] == "ambiguous"
    report = track_report(TARGET, [_pass(rows)], other)
    assert report["segments"][0]["status"] == "ambiguous" and report["identity_ambiguities"] == 1


def test_a_sam_id_is_scoped_to_its_request_and_never_names_the_object():
    one = track_report(TARGET, [_pass(_v(STEADY), name="reca:1:x", session="reca")], [])["segments"][0]
    two = track_report(TARGET, [_pass(_v(STEADY), name="recb:1:x", session="recb")], [])["segments"][0]
    assert one["raw_track_id"] == two["raw_track_id"] == "1"
    assert one["segment_id"] != two["segment_id"]
    assert one["physical_instance_id"] == two["physical_instance_id"] == "instance_6"


def test_a_reacquired_object_holds_several_segments_under_one_identity():
    report = track_report({**TARGET, "reacquires": {"failure_id": "f"}}, [_pass(_v(STEADY))], [])
    assert report["segments"][0]["status"] == "reacquired"
    untrusted = track_report(
        {**TARGET, "reacquires": {"failure_id": "f"}},
        [_pass(_v(STEADY, seed={**SEED, "trust": "candidate"}), seed={**SEED, "trust": "candidate"})],
        [],
    )
    assert untrusted["segments"][0]["status"] == "reacquisition_candidate"


# ---- Frames and merging ----


def test_no_detection_and_rejected_frames_are_counted_apart():
    spans = unresolved_spans([0.0, 0.1, 0.2, 0.3], {0.0}, {0.0, 0.1}, 0.0, 0.3)
    assert spans == [{"start": 0.1, "end": 0.3, "frames": 3, "no_detection_frames": 2, "unusable_frames": 1}]


def _proposal(rows):
    for row in rows:
        row.update({"session_id": "recabc", "raw_track_id": "1", "request_key": "k" * 32, "request_id": "r", "context": False, "seed_trust": "human_verified"})
    return SimpleNamespace(object_local_id="Object_6", rows_json=rows, metrics_json={"remaining_gaps": []})


def test_accepting_keeps_lost_rows_visible_as_conflicts_and_applying_twice_adds_nothing():
    rows = _broken_then_confirmed()
    document = AnalysisDocument(objects=[ObjectPrediction(local_id="Object_6", label="washer")])
    merge_proposal(document, _proposal(rows), {}, "e1")
    lost = [row for row in document.raw_observations if 0.59 <= row.timestamp <= 0.91]
    assert len(lost) == 4 and {row.association_status for row in lost} == {"conflict"}
    assert all(row.observation_role != "canonical" for row in lost)
    assert any("after the track broke" in item.description for item in document.ambiguities)
    count = len(document.raw_observations)
    ambiguities = len(document.ambiguities)
    merge_proposal(document, _proposal(rows), {}, "e1")
    assert len(document.raw_observations) == count and len(document.ambiguities) == ambiguities


def test_one_bad_box_is_an_outlier_even_when_the_part_returns_at_a_different_size():
    samples = [*STEADY, _s(0.6, x=0.5, w=0.2), *[_s(0.7 + t / 10, x=0.13, w=0.04, h=0.1) for t in range(5)]]
    rows = _v(samples)
    assert not any(row["failure"] for row in rows)
    assert rows[6]["outcome"] == "conflicting" and rows[6]["segment"] == "f0"
    assert all(row["lifecycle"] != "lost" for row in rows)


def test_mask_stats_decode_a_real_mask_or_report_nothing():
    assert mask_stats(SimpleNamespace(mask_data="", mask_payload="", mask_height=0, mask_width=0)) is None
    assert mask_stats(SimpleNamespace(mask_data="1,1,notamask", mask_payload="", mask_height=1, mask_width=1)) in (
        None,
        {"pixels": 0, "fill": 0.0, "components": 0},
        {"pixels": 1, "fill": 1.0, "components": 1},
    )


def test_replaying_the_same_rows_is_deterministic():
    assert _broken_then_confirmed() == _broken_then_confirmed()
