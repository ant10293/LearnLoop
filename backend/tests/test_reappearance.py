"""A lost object is found again, identified, and tracked as a new segment. SAM is mocked. Nothing reaches a network."""

from __future__ import annotations

import re

import pytest
from app.perception.reappearance import (
    AMBIGUOUS_CANDIDATE,
    NO_DETECTION,
    PROBABLE_REAPPEARANCE,
    REJECTED_CANDIDATE,
    VERIFIED_REAPPEARANCE,
    ReappearancePolicy,
    candidate_id,
    diagnose_failure,
    search_result,
    search_window,
    verify_candidates,
)
from app.perception.recovery_validate import validate_track
from app.perception.seeds import Seed
from app.perception.spatial_prompts import (
    MediaMap,
    MultiSeedPolicy,
    SeedMappingError,
    build_spatial_plan,
    plan_for_strategy,
    select_prompt_seeds,
)
from app.providers.sam import VideoChunk, video_request
from app.schemas.analysis import AnalysisDocument, ObjectPrediction

from tests.test_recovery_execution import (
    BOX,
    FakeSam,
    _counts,
    _install,
    _live,
    _setup,
)
from tests.test_recovery_execution import client as client  # noqa: F401, E402
from tests.test_recovery_execution import no_language_model as no_language_model  # noqa: F401, E402


@pytest.fixture(autouse=True)
def no_appearance_model(monkeypatch):
    """Whether weights sit on this machine must not change a result here. Appearance tests pass scores in."""
    from app.services import recovery_reacquisition

    monkeypatch.setattr(recovery_reacquisition, "_appearance", lambda *_args: None)


A = {"x": 0.6, "y": 0.5, "w": 0.1, "h": 0.1}
B = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}


def _seed(time, box=BOX, trust="human_verified", source="human", local_id="Object_1"):
    return Seed(object_local_id=local_id, timestamp=time, bbox=box, trust=trust, source=source)


def _candidate(time=5.0, box=A, track="7", **extra):
    return {
        "candidate_id": candidate_id("Object_1", "cached_concept", time, box, track),
        "source": "cached_concept",
        "family": "sam",
        "trust": "candidate",
        "timestamp": time,
        "bbox": box,
        "raw_track_id": track,
        **extra,
    }


def _context(**extra):
    return {
        "object_local_id": "Object_1",
        "failure_time": 2.0,
        "last_good_time": 1.9,
        "anchors": {},
        "boundaries": [],
        "same_label": ["Object_1"],
        **extra,
    }


# ---- Multi-frame prompting ----


def test_meta_multi_frame_grammar_repeats_one_id_on_two_frames():
    media = MediaMap(640, 480, [index * 0.1 for index in range(120)])
    one = _seed(1.0, {"x": 100 / 640, "y": 100 / 480, "w": 50 / 640, "h": 50 / 480})
    two = _seed(9.5, {"x": 300 / 640, "y": 200 / 480, "w": 55 / 640, "h": 50 / 480})
    plan = build_spatial_plan([(1, two), (1, one)], media, "forward", "s")
    assert plan.prompt == (
        "<10f>1<|box;x1=100;y1=100;x2=149;y2=149;w=640;h=480|>"
        "<95f>1<|box;x1=300;y1=200;x2=354;y2=249;w=640;h=480|>"
    )
    assert plan.bindings == {1: "Object_1"}
    assert video_request(plan.prompt, file_id="f", prompt_kind="spatial", propagation_direction="forward")


def test_a_later_prompt_cannot_start_a_backward_pass():
    media = MediaMap(640, 480, [index * 0.1 for index in range(120)])
    with pytest.raises(SeedMappingError, match="earliest prompt"):
        build_spatial_plan([(1, _seed(1.0)), (1, _seed(9.5))], media, "backward", "s")
    assert build_spatial_plan([(1, _seed(9.5))], media, "backward", "s").prompt.startswith("<95f>")
    assert build_spatial_plan([(1, _seed(1.0)), (1, _seed(9.5))], media, "both", "s").propagation == "both"


def test_one_object_has_one_prompt_per_frame_and_two_objects_never_share_an_id():
    media = MediaMap(640, 480, [index * 0.1 for index in range(20)])
    same = build_spatial_plan([(1, _seed(1.0)), (1, _seed(1.0))], media, "both", "s")
    assert same.prompt.count("<|box;") == 1
    with pytest.raises(SeedMappingError, match="two different prompts"):
        build_spatial_plan([(1, _seed(1.0)), (1, _seed(1.0, B))], media, "both", "s")
    with pytest.raises(SeedMappingError, match="two SAM ids"):
        build_spatial_plan([(1, _seed(1.0)), (2, _seed(1.5))], media, "both", "s")


def test_selection_prefers_both_sides_of_a_gap_over_many_near_identical_boxes():
    near = [_seed(0.1 * index, trust="independently_supported", source="tracker") for index in range(10)]
    after = _seed(8.0, A, trust="independently_supported", source="tracker")
    chosen = select_prompt_seeds([*near, after])
    assert [seed.timestamp for seed in chosen][-1] == 8.0
    assert len(chosen) == 2


def test_selection_adds_a_near_seed_only_when_the_view_changed():
    wide = {"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.05}
    chosen = select_prompt_seeds([_seed(0.0), _seed(1.5, wide, trust="independently_supported", source="tracker"), _seed(1.6, trust="independently_supported", source="tracker")])
    assert [seed.timestamp for seed in chosen] == [0.0, 1.5]


def test_selection_never_uses_drift_candidates_or_a_prompted_track():
    drift = _seed(8.0, trust="candidate", source="vlm")
    own = _seed(9.0, trust="independently_supported", source="sam_spatial")
    assert [seed.timestamp for seed in select_prompt_seeds([_seed(0.0), drift, own])] == [0.0]


def test_selection_honors_the_per_object_limit():
    seeds = [_seed(float(index) * 5) for index in range(6)]
    assert len(select_prompt_seeds(seeds, MultiSeedPolicy(max_prompts_per_object=2))) == 2


def _chunk():
    times = [index * 0.1 for index in range(120)]
    return VideoChunk(0, 0, 120, 0.0, 12.0, "c0", times)


def _doc():
    return AnalysisDocument(summary="a", objects=[ObjectPrediction(local_id="Object_1", label="washer"), ObjectPrediction(local_id="Object_2", label="washer")])


def test_the_pipeline_keeps_one_seed_per_object_unless_multi_seed_is_selected():
    seeds = [_seed(1.0), _seed(9.5, A), _seed(2.0, B, local_id="Object_2")]
    single, _ = plan_for_strategy(_doc(), _chunk(), 640, 480, 10.0, "trusted_spatial", seeds)
    assert single[0].prompt.count("<|box;") == 2
    multi, outcomes = plan_for_strategy(_doc(), _chunk(), 640, 480, 10.0, "trusted_spatial", seeds, multi_seed=True)
    assert multi[0].prompt.count("<|box;") == 3
    assert multi[0].prompt.count("1<|box;") == 2
    assert multi[0].bindings == {1: "Object_1", 2: "Object_2"}
    assert sum(item["object_local_id"] == "Object_1" for item in outcomes) == 2


def test_the_request_prompt_limit_still_gives_every_object_its_first_prompt():
    seeds = [_seed(1.0), _seed(9.5, A), _seed(2.0, B, local_id="Object_2")]
    plans, _ = plan_for_strategy(_doc(), _chunk(), 640, 480, 10.0, "trusted_spatial", seeds, multi_seed=True, multi_policy=MultiSeedPolicy(max_prompts_per_request=2))
    assert plans[0].prompt.count("<|box;") == 2 and set(plans[0].bindings.values()) == {"Object_1", "Object_2"}


# ---- Identity ----


def test_an_independent_measurement_verifies_a_new_sam_id():
    anchors = {"Object_1": [{"timestamp": 5.0, "bbox": A, "source": "tracker", "object_local_id": "Object_1"}]}
    judged = verify_candidates([_candidate()], _context(anchors=anchors))
    assert judged[0]["outcome"] == VERIFIED_REAPPEARANCE


def test_a_measurement_from_the_same_family_verifies_nothing():
    anchors = {"Object_1": [{"timestamp": 5.0, "bbox": A, "source": "sam", "object_local_id": "Object_1"}]}
    judged = verify_candidates([_candidate()], _context(anchors=anchors))
    assert judged[0]["outcome"] == PROBABLE_REAPPEARANCE


def test_two_identical_washers_stop_for_a_person():
    judged = verify_candidates([_candidate()], _context(same_label=["Object_1", "Object_2"]))
    assert judged[0]["outcome"] == AMBIGUOUS_CANDIDATE and judged[0]["needs_confirmation"]


def test_a_detection_measured_as_the_other_washer_is_rejected():
    anchors = {"Object_2": [{"timestamp": 5.0, "bbox": A, "source": "tracker", "object_local_id": "Object_2"}]}
    judged = verify_candidates([_candidate()], _context(anchors=anchors, same_label=["Object_1", "Object_2"]))
    assert judged[0]["outcome"] == REJECTED_CANDIDATE


def test_of_two_detections_the_supported_one_wins_and_the_rival_is_rejected():
    anchors = {"Object_1": [{"timestamp": 5.0, "bbox": A, "source": "tracker", "object_local_id": "Object_1"}]}
    judged = verify_candidates([_candidate(), _candidate(box=B, track="8")], _context(anchors=anchors, same_label=["Object_1", "Object_2"]))
    assert [item["outcome"] for item in judged] == [VERIFIED_REAPPEARANCE, REJECTED_CANDIDATE]
    assert judged[0]["raw_track_id"] == "7"


def test_a_turned_part_is_not_rejected_for_its_new_shape():
    turned = {"x": 0.6, "y": 0.5, "w": 0.3, "h": 0.02}
    anchors = {"Object_1": [{"timestamp": 5.0, "bbox": turned, "source": "human", "object_local_id": "Object_1"}]}
    judged = verify_candidates([_candidate(box=turned)], _context(anchors=anchors))
    assert judged[0]["outcome"] == VERIFIED_REAPPEARANCE


def test_appearance_alone_verifies_only_a_unique_part_with_meta_weights():
    meta = {"encoder": "dinov2_vits14", "score": 0.9}
    assert verify_candidates([_candidate(appearance=meta)], _context())[0]["outcome"] == VERIFIED_REAPPEARANCE
    hist = {"encoder": "histogram_v1", "score": 0.99}
    assert verify_candidates([_candidate(appearance=hist)], _context())[0]["outcome"] == PROBABLE_REAPPEARANCE
    assert verify_candidates([_candidate()], _context())[0]["outcome"] == PROBABLE_REAPPEARANCE
    assert verify_candidates([_candidate(appearance=meta)], _context(same_label=["Object_1", "Object_2"]))[0]["outcome"] == AMBIGUOUS_CANDIDATE


def test_a_scene_cut_stops_appearance_from_verifying():
    meta = {"encoder": "dinov2_vits14", "score": 0.9}
    judged = verify_candidates([_candidate(appearance=meta)], _context(boundaries=[3.0]))
    assert judged[0]["outcome"] == PROBABLE_REAPPEARANCE
    assert any(check["check"] == "scene_continuity" and not check["passed"] for check in judged[0]["checks"])


def test_nothing_found_reports_no_detection():
    assert search_result(verify_candidates([], _context())) == NO_DETECTION


def test_a_long_occlusion_past_the_window_needs_approval_to_widen():
    frames = [index * 0.1 for index in range(200)]
    hidden = [2.0 + index * 0.3 for index in range(30)]
    policy = ReappearancePolicy(search_window_sec=5.0)
    narrow = search_window(2.0, frames, hidden, policy)
    assert narrow["limited"] and not narrow["frames"] and "approval" in narrow["note"]
    wide = search_window(2.0, frames, hidden[:20], policy, expand=True)
    assert not wide["limited"] and wide["frames"]


def test_the_diagnosis_separates_new_ids_from_rejected_boxes():
    failure = {"last_good_time": 1.9, "last_supported_time": 1.9}
    new_id = diagnose_failure(failure, rows_after=[], frames_after=[2.0, 2.1], candidates=[], other_tracks=[{"raw_track_id": "7"}], anchors=[], clip_end=3.0, video_end=3.0, missing_accepted=0, planned=False, executed=False)
    assert [item["code"] for item in new_id["categories"]] == ["B_new_track_id"]
    assert new_id["first_no_detection_time"] == 2.0
    stopped = diagnose_failure(failure, rows_after=[], frames_after=[2.0], candidates=[], other_tracks=[], anchors=[], clip_end=2.0, video_end=9.0, missing_accepted=0, planned=False, executed=False)
    assert stopped["categories"][0]["code"] == "A_sam_stopped" and stopped["clip_ended_early"]
    row = {"timestamp": 2.0, "bbox": A, "outcome": "conflicting"}
    right = diagnose_failure(failure, rows_after=[row], frames_after=[2.0], candidates=[], other_tracks=[], anchors=[{"timestamp": 2.0, "bbox": A}], clip_end=2.0, video_end=2.0, missing_accepted=0, planned=False, executed=False)
    assert right["categories"][0]["code"] == "C_rejected_by_validation"
    wrong = diagnose_failure(failure, rows_after=[row], frames_after=[2.0], candidates=[], other_tracks=[], anchors=[{"timestamp": 2.0, "bbox": B}], clip_end=2.0, video_end=2.0, missing_accepted=0, planned=False, executed=False)
    assert wrong["categories"][0]["code"] == "D_followed_wrong_object"
    unknown = diagnose_failure(failure, rows_after=[row], frames_after=[2.0], candidates=[], other_tracks=[], anchors=[], clip_end=2.0, video_end=2.0, missing_accepted=2, planned=False, executed=False)
    assert [item["code"] for item in unknown["categories"]] == ["C_or_D_undetermined", "E_not_in_analysis"]


def test_a_reacquired_segment_that_drifts_onto_a_neighbor_is_cut_at_the_right_frame():
    seed = {"seed_id": "s", "timestamp": 0.0, "bbox": A, "trust": "independently_supported"}

    def sample(time, box):
        return {"timestamp": time, "bbox": box, "confidence": 0.9, "frame_index": round(time * 10), "source_width": 640, "source_height": 360, "mask_data": "", "mask_payload": "", "mask_encoding": "", "mask_height": 0}

    samples = [sample(index / 10, A) for index in range(6)] + [sample(0.6 + index / 10, B) for index in range(6)]
    rows = validate_track(samples, seed=seed, association="associated", anchors=[], neighbors=[], media_size=(640, 360))
    assert rows[6]["failure"]["reason"] == "large_spatial_jump" and rows[6]["failure"]["last_good_time"] == 0.5
    assert all(row["outcome"] != "supported" for row in rows[6:])


# ---- End to end through the API, with SAM mocked ----


def _text(lines):
    return "\n".join(lines)


def _pixels(box, w, h):
    return round(box["x"] * w), round(box["y"] * h), round((box["x"] + box["w"]) * w) - 1, round((box["y"] + box["h"]) * h) - 1


def _token(track, box, w, h):
    x1, y1, x2, y2 = _pixels(box, w, h)
    return f"{track}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c=0.9|><|mask;data=1,1,!A|>"


def _first_pass(prompt, _direction):
    """Object A is followed for 20 frames, then SAM returns nothing for it. Track 2 is competing object B."""
    w, h = (int(item) for item in re.search(r"w=(\d+);h=(\d+)", prompt).groups())
    first = int(re.match(r"<(\d+)f>", prompt).group(1))
    return _text(f"<{first + index}f>{_token(1, BOX, w, h)}" for index in range(20))


def _history(client, run_id):
    return next(item for item in client.get(f"/api/runs/{run_id}/recovery/reacquisition").json()["objects"] if item["object_local_id"] == "Object_1")


def _start(client, run_id, times):
    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={"manual_targets": [{"object_local_id": "Object_1", "time_start": times[2], "time_end": times[2] + 2.5, "direction": "forward", "seed": {"timestamp": times[2], "bbox": BOX}}], "limits": {"max_requests": 4, "padding_sec": 0}},
    ).json()
    return created


def test_a_lost_object_is_verified_reinitialized_validated_and_accepted(client, tmp_path, monkeypatch, no_language_model):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRevision

    run_id, revision_id, times = _setup(client, tmp_path)
    before = _counts()
    fake = _install(monkeypatch, FakeSam(answer=_first_pass))
    first = _start(client, run_id, times)
    done = _live(client, first).json()
    assert done["status"] == "completed", done["error"]
    assert len(fake.calls) == 1

    # Step 1: the loss is reported with its kind, and planning sends nothing.
    history = _history(client, run_id)
    plan = history["plans"][0]
    assert plan["failure"]["reason"] == "no_detection"
    assert plan["status"] == "open"
    assert plan["diagnosis"]["raw_boxes_after_failure"] == 0
    assert plan["diagnosis"]["first_no_detection_time"] is not None
    assert plan["automatic_start_allowed"] is False
    assert [stage["stage"] for stage in plan["workflow"]][:3] == ["track_lost", "cached_search", "identity"]
    assert len(fake.calls) == 1

    # Step 2: a stored independent box after the loss is a verified place to start.
    candidate = next(item for item in plan["search"]["candidates"] if item["outcome"] == VERIFIED_REAPPEARANCE)
    assert candidate["bbox"] == BOX
    created = client.post(
        f"/api/runs/{run_id}/recovery/reacquisition/plan",
        json={"object_local_id": "Object_1", "failure_id": plan["failure"]["failure_id"], "candidate_id": candidate["candidate_id"], "limits": {"max_requests": 4}},
    )
    assert created.status_code == 200, created.text
    second = created.json()
    assert second["status"] == "planned"
    assert len(fake.calls) == 1, "A plan is free."
    spec = second["plan"]["requests"][0]
    assert spec["objects"][0]["prompt_role"] == "reappearance_seed"
    assert spec["objects"][0]["seed"]["source"] == "reappearance"
    assert spec["source_end"] - spec["source_start"] <= 6.0 + 0.1

    # Step 3: the request goes out only with this plan's approval, and starts a separate segment.
    refused = client.post(f"/api/recovery/experiments/{second['id']}/live", json={"plan_hash": second["plan"]["plan_hash"]})
    assert refused.status_code == 400 and len(fake.calls) == 1
    fake.answer = lambda prompt, _direction: _text(
        f"<{index}f>{_token(1, BOX, *(int(item) for item in re.search(r'w=(\d+);h=(\d+)', prompt).groups()))}"
        for index in range(int(re.match(r"<(\d+)f>", prompt).group(1)), len(spec["frame_times"]))
    )
    ran = client.post(f"/api/recovery/experiments/{second['id']}/live", json={"confirm": True, "plan_hash": second["plan"]["plan_hash"]}).json()
    assert ran["status"] == "completed", ran["error"]
    assert len(fake.calls) == 2
    proposal = ran["proposals"][0]
    segment = proposal["metrics"]["track"]["segments"][0]
    assert segment["reacquires"]["failure_id"] == plan["failure"]["failure_id"]
    assert segment["status"] == "reacquired" and segment["identity"]["status"] == "linked"
    assert segment["physical_instance_id"] == "track_1"
    assert ran["targets"][0]["reacquires"]["record"]["seed_outcome"] == VERIFIED_REAPPEARANCE

    # Nothing changed in the analysis yet.
    session = SessionLocal()
    original = session.get(AnalysisRevision, revision_id).document_json
    session.close()
    assert _counts()["revisions"] == before["revisions"]

    # Step 4: replay repeats the result for free.
    replayed = client.post(f"/api/recovery/experiments/{second['id']}/replay").json()
    assert len(fake.calls) == 2
    fresh = replayed["proposals"][0]
    assert fresh["metrics"]["counts"] == proposal["metrics"]["counts"]

    # Step 5: accepting makes a new revision holding the new boxes. The old one is untouched.
    accepted = client.post(
        f"/api/recovery/experiments/{second['id']}/proposals/decide",
        json={"decision": "accepted", "proposal_ids": [fresh["id"]], "include_provisional": True},
    ).json()
    assert accepted["applied_revision_id"]
    session = SessionLocal()
    assert session.get(AnalysisRevision, revision_id).document_json == original
    applied = AnalysisDocument.model_validate(session.get(AnalysisRevision, accepted["applied_revision_id"]).document_json)
    session.close()
    returned = [row for row in applied.observations if row.source_model == "sam-3.1" and row.object_local_id == "Object_1" and row.timestamp >= segment["start"]]
    assert returned, "The returned object is in the accepted geometry."
    assert {row.continuity.value for row in returned if row.association_status == "associated"} <= {"reassociated", "continuous"}
    assert len(fake.calls) == 2
    assert _history(client, run_id)["plans"][0]["status"] == "resolved"


def test_a_concept_search_with_several_candidates_stops_for_a_person(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision_id, times = _setup(client, tmp_path)
    from app.services.storage import data_path

    # No independent evidence after the loss.
    path = data_path("runs", run_id, "spatial.json")
    spatial = AnalysisDocument.model_validate_json(path.read_text())
    spatial.observations = [row for row in spatial.observations if row.timestamp <= times[2] + 0.01]
    path.write_text(spatial.model_dump_json())
    fake = _install(monkeypatch, FakeSam(answer=_first_pass))
    _live(client, _start(client, run_id, times))
    plan = _history(client, run_id)["plans"][0]
    assert len(fake.calls) == 1
    concept = client.post(
        f"/api/runs/{run_id}/recovery/reacquisition/plan",
        json={"object_local_id": "Object_1", "failure_id": plan["failure"]["failure_id"], "action": "concept", "limits": {"max_requests": 4}},
    )
    assert concept.status_code == 200, concept.text
    experiment = concept.json()
    spec = experiment["plan"]["requests"][0]
    assert spec["prompt_kind"] == "concept" and spec["prompt"] == "washer" and "<|" not in spec["prompt"]
    assert len(fake.calls) == 1, "Planning a concept pass sends nothing."

    fake.answer = lambda _prompt, _direction: _text(
        f"<{index}f>{_token(1, A, spec['width'], spec['height'])},{_token(2, B, spec['width'], spec['height'])}".replace(",2<", "\n" + f"<{index}f>2<")
        for index in range(len(spec["frame_times"]))
    )
    ran = client.post(f"/api/recovery/experiments/{experiment['id']}/live", json={"confirm": True, "plan_hash": experiment["plan"]["plan_hash"]}).json()
    assert ran["status"] == "completed", ran["error"]
    assert len(fake.calls) == 2
    after = _history(client, run_id)["plans"][0]
    found = [item for item in after["search"]["candidates"] if item["source"] == "cached_concept"]
    assert len(found) == 2 and {item["outcome"] for item in found} == {AMBIGUOUS_CANDIDATE}
    assert all(item["outcome"] != VERIFIED_REAPPEARANCE for item in found)
    assert len(fake.calls) == 2, "No spatial request starts by itself."

    # An unconfirmed candidate cannot become a seed. A confirmed one can, as a person's seed.
    refused = client.post(
        f"/api/runs/{run_id}/recovery/reacquisition/plan",
        json={"object_local_id": "Object_1", "failure_id": after["failure"]["failure_id"], "candidate_id": found[0]["candidate_id"]},
    )
    assert refused.status_code == 400 and "confirmation" in refused.json()["detail"]
    confirmed = client.post(
        f"/api/runs/{run_id}/recovery/reacquisition/plan",
        json={"object_local_id": "Object_1", "failure_id": after["failure"]["failure_id"], "candidate_id": found[0]["candidate_id"], "confirm": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    role = confirmed.json()["plan"]["requests"][0]["objects"][0]
    assert role["prompt_role"] == "manual_seed" and role["seed"]["trust"] == "human_verified"
    assert confirmed.json()["targets"][0]["reacquires"]["record"]["confirmed_by_person"] is True

    # A second concept search is refused without its own approval, so a failure cannot loop.
    again = client.post(
        f"/api/runs/{run_id}/recovery/reacquisition/plan",
        json={"object_local_id": "Object_1", "failure_id": after["failure"]["failure_id"], "action": "concept"},
    )
    assert again.status_code == 400 and "already had" in again.json()["detail"]
    assert len(fake.calls) == 2


def test_an_object_that_never_returns_stays_unresolved_and_asks_for_nothing_automatically(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision_id, times = _setup(client, tmp_path)
    from app.services.storage import data_path

    path = data_path("runs", run_id, "spatial.json")
    spatial = AnalysisDocument.model_validate_json(path.read_text())
    spatial.observations = [row for row in spatial.observations if row.timestamp <= times[2] + 0.01]
    path.write_text(spatial.model_dump_json())
    fake = _install(monkeypatch, FakeSam(answer=_first_pass))
    _live(client, _start(client, run_id, times))
    plan = _history(client, run_id)["plans"][0]
    assert plan["status"] == "open"
    assert not [item for item in plan["search"]["candidates"] if item["outcome"] == VERIFIED_REAPPEARANCE]
    assert plan["automatic_start_allowed"] is False
    assert next(stage for stage in plan["workflow"] if stage["stage"] == "reinitialization")["status"] == "blocked"
    assert len(fake.calls) == 1


def test_grouped_frames_must_be_an_unbroken_run_of_the_source():
    from app.services.recovery import _check_contiguous

    source = [0.0, 0.1, 0.2, 0.3, 0.4]
    _check_contiguous([0.1, 0.2, 0.3], source)
    with pytest.raises(SeedMappingError, match="skip source frames"):
        _check_contiguous([0.1, 0.3], source)


def test_two_seeds_on_different_frames_never_share_a_backward_request():
    from app.services.recovery import DEFAULT_LIMITS, _groupable

    first = {"seed": _seed(5.0), "direction": "backward", "frames": [index * 0.1 for index in range(60)], "target": {}, "tolerance": None}
    second = {"seed": _seed(4.0, local_id="Object_2"), "direction": "backward", "frames": [index * 0.1 for index in range(50)], "target": {}, "tolerance": None}
    group = {"direction": "backward", "frames": first["frames"], "entries": [first]}
    assert not _groupable(group, second, DEFAULT_LIMITS)
    forward = {**group, "direction": "forward", "entries": [{**first, "direction": "forward"}]}
    assert _groupable(forward, {**second, "direction": "forward"}, DEFAULT_LIMITS)


def test_a_multi_frame_target_goes_out_as_one_request_with_one_id(client, tmp_path):
    run_id, _revision_id, times = _setup(client, tmp_path)
    candidates = client.get(f"/api/runs/{run_id}/recovery/candidates").json()
    trusted = sorted((seed for seed in candidates["seeds"] if seed["object_local_id"] == "Object_1" and seed["trust"] == "independently_supported"), key=lambda seed: seed["timestamp"])
    chosen = [trusted[0]["seed_id"], trusted[2]["seed_id"]]
    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={"manual_targets": [{"object_local_id": "Object_1", "time_start": times[0], "time_end": times[3], "direction": "forward", "seed_ids": chosen}], "limits": {"max_frames_per_request": 300}},
    ).json()
    plan = client.post(f"/api/recovery/experiments/{created['id']}/dry-run").json()["plan"]
    assert not plan["blocked"], plan["blocked"]
    assert len(plan["requests"]) == 1
    spec = plan["requests"][0]
    assert spec["prompt"].count("1<|box;") == 2 and spec["bindings"] == {"1": "Object_1"}
    assert [item["prompt_role"] for item in spec["objects"]] == ["initial_seed", "additional_prompt"]
    assert spec["prompt"].index(f"<{spec['objects'][0]['seed_frame_index']}f>") < spec["prompt"].index(f"<{spec['objects'][1]['seed_frame_index']}f>")


def test_a_multi_frame_request_splits_rows_between_its_prompts(client, tmp_path, monkeypatch):
    run_id, _revision_id, times = _setup(client, tmp_path)
    candidates = client.get(f"/api/runs/{run_id}/recovery/candidates").json()
    trusted = sorted((seed for seed in candidates["seeds"] if seed["object_local_id"] == "Object_1" and seed["trust"] == "independently_supported"), key=lambda seed: seed["timestamp"])
    chosen = [trusted[0]["seed_id"], trusted[2]["seed_id"]]
    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={"manual_targets": [{"object_local_id": "Object_1", "time_start": times[0], "time_end": times[3], "direction": "forward", "seed_ids": chosen}], "limits": {"max_frames_per_request": 300}},
    ).json()
    plan = client.post(f"/api/recovery/experiments/{created['id']}/dry-run").json()["plan"]
    spec = plan["requests"][0]
    fake = _install(monkeypatch, FakeSam(answer=lambda _prompt, _direction: _text(f"<{index}f>{_token(1, BOX, spec['width'], spec['height'])}" for index in range(len(spec["frame_times"])))))
    data = client.post(f"/api/recovery/experiments/{created['id']}/live", json={"confirm": True, "plan_hash": plan["plan_hash"]}).json()
    assert data["status"] == "completed", data["error"]
    rows = data["proposals"][0]["rows"]
    times_seen = [row["timestamp"] for row in rows]
    assert len(times_seen) == len(set(times_seen)), "One box per frame, not one per prompt."
    assert {row["seed_id"] for row in rows} == set(chosen)
    assert data["proposals"][0]["metrics"]["usage"]["requests"] == 1
    assert len(fake.calls) == 1
