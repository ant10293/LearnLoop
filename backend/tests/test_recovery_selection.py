"""Trust levels, recovery targets, validation, and resolver protection. Synthetic sequences, not accuracy."""

from __future__ import annotations

from types import SimpleNamespace

from app.perception.recovery_targets import find_targets
from app.perception.recovery_validate import combine_passes, remaining_gaps, validate_track
from app.perception.resolve import resolve_document
from app.perception.seeds import collect_seeds, manual_seed, rank_for_interval, seeds_around
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    BoxDecision,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
    RelationshipPrediction,
)
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance, RelationKind, Visibility
from app.services.recovery import merge_proposal


def _evidence(provenance=Provenance.AI_PREDICTION):
    return Evidence(classification=EvidenceClass.DIRECTLY_OBSERVED, provenance=provenance)


def _box(x=0.1, y=0.1, w=0.1, h=0.1):
    return BBox(x=x, y=y, w=w, h=h)


def _vlm(time, box=None, local="Object_1", **extra):
    return ObservationPrediction(
        object_local_id=local,
        timestamp=time,
        bbox=box or _box(),
        evidence=_evidence(),
        geometry_origin="vlm_estimated",
        geometry_authority="vlm",
        **extra,
    )


def _tracker(time, box=None, local="Object_1", ref="track_1", **extra):
    values = {
        "association_support": "associated",
        "detection_support": "detected",
        "temporal_support": "repeated",
        **extra,
    }
    return ObservationPrediction(
        object_local_id=local,
        timestamp=time,
        bbox=box or _box(),
        evidence=_evidence(Provenance.CV_DETECTION),
        source_model="tracker",
        track_ref=ref,
        geometry_origin="detected",
        geometry_authority="motion",
        **values,
    )


def _sam(time, box=None, local="Object_1", ref="c0s0:1"):
    return ObservationPrediction(
        object_local_id=local,
        timestamp=time,
        bbox=box or _box(),
        evidence=_evidence(Provenance.CV_DETECTION),
        source_model="sam-3.1",
        track_ref=ref,
        association_status="associated",
        association_support="associated",
        detection_support="mask",
        geometry_origin="detected",
        geometry_authority="sam-3.1",
    )


def _doc(*rows, objects=("Object_1", "Object_2"), **extra) -> AnalysisDocument:
    return AnalysisDocument(
        objects=[ObjectPrediction(local_id=item, label="washer") for item in objects],
        observations=list(rows),
        **extra,
    )


def _trust(document, time=1.0, source=None, local="Object_1"):
    seeds = collect_seeds(document)
    return next(
        seed.trust
        for seed in seeds
        if seed.object_local_id == local and abs(seed.timestamp - time) < 1e-6 and (source is None or seed.source == source)
    )


# ---- Trust ----


def test_a_confident_model_box_alone_is_only_a_candidate():
    assert _trust(_doc(_vlm(1.0, confidence=0.99))) == "candidate"


def test_a_repeated_model_box_is_still_one_opinion():
    document = _doc(_vlm(1.0), _vlm(1.0, _box(x=0.101)), _vlm(1.5))
    assert _trust(document) == "candidate"


def test_a_human_box_is_verified():
    row = _vlm(1.0).model_copy(update={"geometry_authority": "human", "evidence": _evidence(Provenance.HUMAN_CORRECTION)})
    assert _trust(_doc(row)) == "human_verified"


def test_a_model_box_a_tracker_agrees_with_is_independently_supported():
    assert _trust(_doc(_vlm(1.0), _tracker(1.0)), source="vlm") == "independently_supported"


def test_disagreeing_producers_do_not_support_each_other():
    assert _trust(_doc(_vlm(1.0), _tracker(1.0, _box(x=0.6))), source="tracker") == "candidate"


def test_a_prompted_track_cannot_vouch_for_a_box_or_for_itself():
    document = _doc(_vlm(1.0), _sam(1.0, ref="c0p0:1"))
    assert _trust(document, source="vlm") == "candidate"
    assert _trust(document, source="sam_spatial") == "candidate"


def test_an_associated_concept_track_can_support_a_model_box():
    assert _trust(_doc(_vlm(1.0), _sam(1.0)), source="sam") == "independently_supported"


def test_single_detection_trackers_do_not_count_as_measured():
    assert _trust(_doc(_vlm(1.0), _tracker(1.0, temporal_support="single")), source="vlm") == "candidate"


def test_an_occluded_conflicting_or_flow_row_is_rejected():
    assert _trust(_doc(_vlm(1.0, occlusion=0.8))) == "rejected"
    assert _trust(_doc(_vlm(1.0, association_status="conflict"))) == "rejected"
    flow = _tracker(1.0).model_copy(update={"geometry_origin": "optical_flow"})
    assert _trust(_doc(flow)) == "rejected"


def test_a_box_shared_with_another_object_is_rejected():
    document = _doc(_vlm(1.0), _vlm(1.0, local="Object_2"))
    assert _trust(document) == "rejected"


def test_a_scene_sized_or_unmeasured_box_is_rejected():
    assert _trust(_doc(_vlm(1.0, _box(0, 0, 1, 1)))) == "rejected"
    unmeasured = _vlm(1.0).model_copy(update={"geometry_authority": "vlm_video_unmeasured"})
    assert _trust(_doc(unmeasured)) == "rejected"


def test_a_box_the_pipeline_removed_is_rejected():
    document = _doc(_vlm(1.0))
    document.box_decisions = [BoxDecision(step="roster", reason_code="dup", object_local_id="Object_1", timestamp=1.0, bbox=_box())]
    assert _trust(document) == "rejected"


def test_every_decision_keeps_its_checks():
    seed = collect_seeds(_doc(_vlm(1.0)))[0]
    names = {item["check"] for item in seed.checks}
    assert {"box_size", "visible", "clear_of_neighbors", "independent_support"} <= names
    assert seed.explanation


def test_a_researcher_can_confirm_a_candidate_but_not_a_rejected_box():
    document = _doc(_vlm(1.0))
    seed = collect_seeds(document)[0]
    assert collect_seeds(document, confirmed={seed.seed_id})[0].trust == "human_verified"
    bad = _doc(_vlm(1.0, occlusion=0.9))
    seed = collect_seeds(bad)[0]
    assert collect_seeds(bad, confirmed={seed.seed_id})[0].trust == "rejected"


def test_no_seed_is_an_explicit_empty_result():
    around = seeds_around(collect_seeds(_doc(_vlm(1.0))), "Object_1", 2.0, 3.0)
    assert around["best"] is None and around["usable"] == []


def test_the_nearest_appearance_wins_over_a_more_confident_far_one():
    far = manual_seed({"object_local_id": "Object_1", "timestamp": 50.0, "bbox": _box().model_dump()})
    near = collect_seeds(_doc(_vlm(4.0), _tracker(4.0)))
    ranked = rank_for_interval([far, *near], 5.0, 6.0)
    assert ranked[0].timestamp == 4.0


def test_seeds_on_each_side_of_a_gap_are_both_found():
    document = _doc(_vlm(1.0), _tracker(1.0), _vlm(9.0), _tracker(9.0))
    around = seeds_around(collect_seeds(document), "Object_1", 3.0, 6.0)
    assert around["before"].timestamp == 1.0
    assert around["after"].timestamp == 9.0


# ---- Targets ----


def _steady(times, **kwargs):
    rows = []
    for time in times:
        rows.extend([_vlm(time, **kwargs), _tracker(time, **kwargs)])
    return rows


def _targets(document):
    return find_targets(document, None, collect_seeds(document))


def test_a_gap_between_observations_is_unresolved_and_recoverable():
    document = _doc(*_steady([0, 1, 2, 8, 9, 10]))
    found = [item for item in _targets(document) if "tracking_gap" in item["reasons"]]
    assert len(found) == 1
    target = found[0]
    assert (target["time_start"], target["time_end"]) == (2, 8)
    assert target["state"] == "unresolved"
    assert target["recoverable"]
    assert target["seed_before"] and target["seed_after"]
    assert target["source_observations"]
    assert {"target_id", "instance_id", "label", "difficulty", "support", "seeds"} <= set(target)


def test_no_trustworthy_seed_marks_a_target_unrecoverable():
    document = _doc(*[_vlm(time) for time in [0, 1, 2, 8, 9]])
    target = next(item for item in _targets(document) if "tracking_gap" in item["reasons"])
    assert not target["recoverable"]
    assert "trustworthy seed" in target["blocker"]


def test_a_recorded_absence_is_not_suggested():
    rows = [*_steady([0, 1, 2, 8, 9]), _vlm(5).model_copy(update={"bbox": None, "visibility": Visibility.ABSENT})]
    target = next(item for item in _targets(_doc(*rows)) if "tracking_gap" in item["reasons"])
    assert target["state"] == "recorded_absent"
    assert not target["recoverable"]


def test_a_fully_occluded_object_is_not_suggested():
    rows = [*_steady([0, 1, 2, 8, 9]), _vlm(5).model_copy(update={"bbox": None, "visibility": Visibility.OCCLUDED})]
    target = next(item for item in _targets(_doc(*rows)) if "tracking_gap" in item["reasons"])
    assert target["state"] == "recorded_occluded"
    assert not target["recoverable"]


def test_a_part_that_may_have_left_the_view_is_marked_hard():
    edge = _box(x=0.0, y=0.4)
    document = _doc(*_steady([0, 1, 2, 8, 9], box=edge))
    target = next(item for item in _targets(document) if "tracking_gap" in item["reasons"])
    assert target["state"] == "possibly_out_of_view"
    assert target["difficulty"] == "hard"


def test_an_attachment_gap_stays_unresolved_and_does_not_claim_absence():
    relation = RelationshipPrediction(
        subject_local_id="Object_1",
        other_local_id="Object_2",
        kind=RelationKind.ATTACHED,
        time_start=3,
        time_end=7,
        evidence=_evidence(),
    )
    document = _doc(*_steady([0, 1, 2, 8, 9]), relationships=[relation])
    target = next(item for item in _targets(document) if "tracking_gap" in item["reasons"])
    assert target["state"] == "possibly_hidden"


def test_a_sudden_displacement_among_estimates_is_flagged():
    rows = [_vlm(0), _vlm(0.5, _box(x=0.105)), _vlm(1.0, _box(x=0.7)), _vlm(1.5, _box(x=0.705))]
    assert any("implausible_jump" in item["reasons"] for item in _targets(_doc(*rows)))


def test_tracker_measured_motion_is_not_flagged_as_a_jump():
    rows = [_tracker(0), _tracker(0.5, _box(x=0.105)), _tracker(1.0, _box(x=0.7)), _tracker(1.5, _box(x=0.705))]
    assert not any("implausible_jump" in item["reasons"] for item in _targets(_doc(*rows)))


def test_a_sudden_size_change_is_flagged():
    rows = [_vlm(0), _vlm(0.5), _vlm(1.0, _box(w=0.4, h=0.4)), _vlm(1.5, _box(w=0.4, h=0.4))]
    assert any("sudden_size_change" in item["reasons"] for item in _targets(_doc(*rows)))


def test_a_model_box_contradicting_tracking_is_flagged():
    assert any("vlm_disagrees_with_tracking" in item["reasons"] for item in _targets(_doc(_vlm(1.0, _box(x=0.7)), _tracker(1.0))))


def test_two_objects_exchanging_places_are_flagged_on_both():
    left, right = _box(x=0.1), _box(x=0.7)
    rows = [_tracker(0, left), _tracker(1, right), _tracker(0, right, local="Object_2", ref="track_2"), _tracker(1, left, local="Object_2", ref="track_2")]
    found = [item for item in _targets(_doc(*rows)) if "identity_swap" in item["reasons"]]
    assert {item["object_local_id"] for item in found} == {"Object_1", "Object_2"}


def test_a_box_removed_in_grounding_is_flagged_only_when_nothing_measured_replaces_it():
    document = _doc(_vlm(1.0))
    document.box_decisions = [BoxDecision(step="roster", reason_code="dup", object_local_id="Object_1", timestamp=3.0, bbox=_box())]
    assert any("box_removed_in_grounding" in item["reasons"] for item in _targets(document))
    measured = _doc(_tracker(3.0))
    measured.box_decisions = document.box_decisions
    assert not any("box_removed_in_grounding" in item["reasons"] for item in _targets(measured))


def test_a_recording_wide_ambiguity_does_not_swallow_the_targets():
    """Found on a real run: one 0 to 22 s ambiguity merged every problem into a single useless target."""
    wide = AmbiguityPrediction(
        local_id="sam_unassigned_x",
        kind=AmbiguityKind.UNCERTAIN_IDENTITY,
        description="wide",
        time_start=0.0,
        time_end=22.0,
        object_local_ids=["Object_1"],
        evidence=Evidence(classification=EvidenceClass.UNKNOWN, provenance=Provenance.CV_DETECTION),
    )
    document = _doc(*_steady([0, 1, 2, 8, 9, 10]), ambiguities=[wide])
    assert [(item["time_start"], item["time_end"]) for item in _targets(document)] == [(2, 8)]


def test_an_even_steady_record_is_not_flagged():
    assert _targets(_doc(*_steady([0, 1, 2, 3]))) == []


# ---- Validation ----


def _s(time, x=0.1, w=0.1, h=0.1, confidence=0.5, mask=True, frame=None, source=(640, 360)):
    return {
        "timestamp": time,
        "bbox": {"x": x, "y": 0.1, "w": w, "h": h},
        "confidence": confidence,
        "frame_index": frame if frame is not None else int(time * 10),
        "source_width": source[0],
        "source_height": source[1],
        "mask_data": "1,1,!A" if mask else "",
        "mask_payload": "",
        "mask_encoding": "one_bit" if mask else "",
        "mask_height": 1 if mask else 0,
    }


SEED = {"timestamp": 0.0, "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}, "trust": "human_verified"}


def _validate(samples, *, seed=SEED, association="associated", anchors=(), neighbors=(), boundaries=None):
    return validate_track(
        samples,
        seed=seed,
        association=association,
        anchors=list(anchors),
        neighbors=list(neighbors),
        media_size=(640, 360),
        boundaries=boundaries,
    )


def _outcomes(rows):
    return [row["outcome"] for row in rows]


def _anchor(time, x=0.1):
    return {"timestamp": time, "bbox": {"x": x, "y": 0.1, "w": 0.1, "h": 0.1}, "source": "tracker"}


def test_a_steady_track_near_a_trusted_seed_is_supported():
    assert set(_outcomes(_validate([_s(t / 10) for t in range(5)]))) == {"supported"}


def test_response_order_does_not_change_the_result():
    forward = _validate([_s(t / 10) for t in range(5)])
    shuffled = _validate([_s(t / 10) for t in (3, 0, 4, 1, 2)])
    assert [row["timestamp"] for row in shuffled] == [row["timestamp"] for row in forward]
    assert _outcomes(shuffled) == _outcomes(forward)


def test_a_high_score_never_raises_a_row_and_a_low_score_never_lowers_one():
    far = 5.0
    high = _validate([_s(far, confidence=0.99)], seed={**SEED, "timestamp": 0.0})
    low = _validate([_s(far, confidence=0.01)], seed={**SEED, "timestamp": 0.0})
    assert _outcomes(high) == _outcomes(low) == ["provisional"]


def test_a_row_far_from_the_seed_needs_a_measurement_to_be_supported():
    assert _outcomes(_validate([_s(5.0)], anchors=[_anchor(5.0)])) == ["supported"]


def test_a_disagreeing_measurement_makes_a_conflict_not_an_overwrite():
    assert _outcomes(_validate([_s(1.0)], anchors=[_anchor(1.0, x=0.7)])) == ["conflicting"]


def test_an_untrusted_seed_never_makes_a_row_supported():
    rows = _validate([_s(0.1)], seed={**SEED, "trust": "candidate"})
    assert _outcomes(rows) == ["provisional"]


def test_an_unassociated_track_is_rejected_and_a_contradicted_one_conflicts():
    assert _outcomes(_validate([_s(0.1)], association="unassigned")) == ["rejected"]
    assert _outcomes(_validate([_s(0.1)], association="conflict")) == ["conflicting"]


def test_a_box_off_the_seed_on_the_seed_frame_conflicts():
    assert _outcomes(_validate([_s(0.0, x=0.7)])) == ["conflicting"]


def test_a_tracking_switch_conflicts_until_a_measurement_confirms_it_again():
    samples = [_s(0.0), _s(0.1, x=0.101), _s(0.2, x=0.102), _s(0.3, x=0.7), _s(0.4, x=0.701), _s(0.5, x=0.702)]
    rows = _validate(samples, anchors=[_anchor(0.5, x=0.702)])
    assert _outcomes(rows) == ["supported", "supported", "supported", "conflicting", "conflicting", "supported"]


def test_a_gradual_rotation_stays_supported_and_a_sudden_scale_step_is_only_provisional():
    turning = [_s(t / 10, w=0.1 + t * 0.01) for t in range(5)]
    assert set(_outcomes(_validate(turning))) == {"supported"}
    assert _outcomes(_validate([_s(0.0), _s(0.1, w=0.3)])) == ["supported", "provisional"]


def test_a_box_that_grows_over_another_part_does_not_become_the_assembly():
    neighbor = {"timestamp": 0.1, "bbox": {"x": 0.2, "y": 0.1, "w": 0.1, "h": 0.1}, "object_local_id": "Object_2"}
    rows = _validate([_s(0.0), _s(0.1, w=0.3, h=0.2)], neighbors=[neighbor])
    assert rows[1]["outcome"] == "conflicting"
    assert any(item["check"] == "absorbed_neighbor" for item in rows[1]["checks"])


def test_two_tracks_on_one_box_conflict():
    neighbor = {"timestamp": 0.1, "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}, "object_local_id": "Object_2"}
    assert _outcomes(_validate([_s(0.1)], neighbors=[neighbor])) == ["conflicting"]


def test_partial_visibility_keeps_only_the_visible_box():
    rows = _validate([_s(0.1, w=0.05, h=0.05)])
    assert rows[0]["partial"] is True
    assert rows[0]["bbox"]["w"] == 0.05


def test_malformed_masks_and_wrong_source_sizes_are_rejected():
    bad = {**_s(0.1), "mask_encoding": "unknown"}
    assert _outcomes(_validate([bad])) == ["rejected"]
    assert _outcomes(_validate([_s(0.1, source=(1280, 720))])) == ["rejected"]


def test_a_missing_mask_is_at_most_provisional():
    assert _outcomes(_validate([_s(0.1, mask=False)])) == ["provisional"]


def test_a_scene_cut_nearby_lowers_a_row():
    assert _outcomes(_validate([_s(0.1)], boundaries=[0.2])) == ["provisional"]


def test_forward_and_backward_agreement_lifts_and_disagreement_conflicts():
    forward = _validate([_s(5.0)])
    backward = _validate([_s(5.0, x=0.101)])
    assert _outcomes(combine_passes([forward, backward])) == ["supported"]
    apart = _validate([_s(5.0, x=0.7)])
    assert _outcomes(combine_passes([forward, apart])) == ["conflicting"]


def test_a_returned_nothing_interval_stays_unobserved():
    rows = _validate([_s(0.0), _s(0.1)])
    gaps = remaining_gaps(rows, [0.0, 0.1, 0.2, 0.3, 0.4], 0.0, 0.4)
    assert gaps == [{"start": 0.2, "end": 0.4, "frames": 3, "outcome": "unobserved"}]


def test_a_reappearing_object_reconnects_only_through_evidence():
    rows = _validate([_s(0.0), _s(5.0)], anchors=[_anchor(5.0)])
    assert _outcomes(rows) == ["supported", "supported"]
    assert _outcomes(_validate([_s(0.0), _s(5.0, x=0.13)])) == ["supported", "provisional"]


# ---- Applying through the resolver ----


def _proposal(rows, gaps=(), local="Object_1"):
    return SimpleNamespace(object_local_id=local, rows_json=list(rows), metrics_json={"remaining_gaps": list(gaps)})


def _row(time, outcome="supported", x=0.1):
    return {
        "timestamp": time,
        "bbox": {"x": x, "y": 0.1, "w": 0.1, "h": 0.1},
        "outcome": outcome,
        "confidence": 0.9,
        "session_id": "rec1r0",
        "raw_track_id": "1",
        "request_key": "k" * 40,
        "request_id": "r",
        "frame_index": 3,
        "has_mask": True,
        "seed_trust": "human_verified",
    }


def _apply(document, proposal, **kwargs):
    merge_proposal(document, proposal, {}, "exp", **kwargs)
    resolved = resolve_document(document)
    document.observations = resolved.observations
    document.raw_observations = resolved.raw_observations
    document.ambiguities = resolved.ambiguities
    return document


def test_a_correct_recovered_row_fills_a_gap_as_canonical():
    document = _apply(resolve_document(_doc(_vlm(0.0), _vlm(9.0))), _proposal([_row(5.0)]))
    shown = next(row for row in document.observations if row.timestamp == 5.0)
    assert shown.observation_role == "canonical"
    assert shown.source_model == "sam-3.1"


def test_an_existing_canonical_row_is_not_replaced_by_an_agreeing_or_disagreeing_box():
    base = resolve_document(_doc(_tracker(5.0)))
    original = next(row for row in base.observations if row.timestamp == 5.0)
    assert original.observation_role == "canonical"
    agreed = _apply(base.model_copy(deep=True), _proposal([_row(5.0)]))
    assert next(row for row in agreed.observations if row.timestamp == 5.0).source_model == "tracker"
    assert len([row for row in [*agreed.observations, *agreed.raw_observations] if row.source_model == "sam-3.1"]) == 0
    disagreed = _apply(base.model_copy(deep=True), _proposal([_row(5.0, x=0.7)]))
    assert next(row for row in disagreed.observations if row.timestamp == 5.0).source_model == "tracker"
    kept = next(row for row in disagreed.raw_observations if row.source_model == "sam-3.1")
    assert kept.association_status == "conflict"
    assert any(item.local_id.startswith("recovery_conflict_") for item in disagreed.ambiguities)


def test_conflicting_and_rejected_rows_never_become_canonical():
    document = _apply(resolve_document(_doc(_vlm(0.0))), _proposal([_row(5.0, "conflicting"), _row(6.0, "rejected")]))
    assert not [row for row in document.observations if row.source_model == "sam-3.1"]
    assert [row.timestamp for row in document.raw_observations if row.source_model == "sam-3.1"] == [5.0]


def test_provisional_rows_need_an_explicit_choice_and_stay_non_canonical():
    skipped = _apply(resolve_document(_doc(_vlm(0.0))), _proposal([_row(5.0, "provisional")]))
    assert not [row for row in skipped.observations if row.source_model == "sam-3.1"]
    chosen = _apply(resolve_document(_doc(_vlm(0.0))), _proposal([_row(5.0, "provisional")]), include_provisional=True)
    row = next(item for item in chosen.observations if item.source_model == "sam-3.1")
    assert row.observation_role != "canonical"


def test_no_visible_evidence_adds_no_box_and_records_no_absence():
    gap = {"start": 4.0, "end": 6.0, "frames": 3, "outcome": "unobserved"}
    document = _apply(resolve_document(_doc(_vlm(0.0))), _proposal([], gaps=[gap]))
    assert not [row for row in document.observations if 4.0 <= row.timestamp <= 6.0]
    ambiguity = next(item for item in document.ambiguities if item.local_id.startswith("recovery_gap_"))
    assert "not evidence that it is absent" in ambiguity.description


def test_original_rows_stay_stored_when_a_recovered_row_supersedes_them():
    document = _apply(resolve_document(_doc(_vlm(5.0, _box(x=0.11)))), _proposal([_row(5.0)]))
    assert any(row.geometry_authority == "vlm" for row in document.raw_observations)


def test_applying_twice_adds_nothing_and_authority_does_not_climb():
    document = _apply(resolve_document(_doc(_vlm(0.0))), _proposal([_row(5.0)]))
    first = [row.model_dump() for row in [*document.observations, *document.raw_observations]]
    document = _apply(document, _proposal([_row(5.0)]))
    assert [row.model_dump() for row in [*document.observations, *document.raw_observations]] == first
    document = _apply(document, _proposal([_row(5.0)]))
    assert [row.model_dump() for row in [*document.observations, *document.raw_observations]] == first
