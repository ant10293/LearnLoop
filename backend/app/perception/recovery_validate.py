"""Judge each box a prompted SAM track returned. No box overwrites anything here.

Detection confidence, association, temporal continuity, semantic identity, and
the final evidence class stay separate. A high SAM score never raises an outcome.
A rotating part or one moving toward the camera may change shape, so a large
step on its own only lowers a row to provisional.

Attachment: a box that grows over another known part is reported as conflicting,
so a tracked washer does not quietly turn into the whole assembly. When SAM stops
returning an object the interval stays unobserved. No box is invented for it.

A loss closes the visual segment and leaves the physical object's identity alone.
Boxes after a loss stay stored, in no segment, until an initialization event starts
a new segment. That segment inherits neither the old failure nor the old support.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.perception.assignment import iou

SUPPORTED = "supported"
PROVISIONAL = "provisional"
CONFLICTING = "conflicting"
REJECTED = "rejected"
UNOBSERVED = "unobserved"
# States of one visual track. The physical object keeps its identity through all of them.
TRACKED = "tracked"
DEGRADED = "degraded"
LOST = "lost"
OCCLUDED = "occluded"
OUT_OF_VIEW = "out_of_view"
REACQUISITION_CANDIDATE = "reacquisition_candidate"
REACQUIRED = "reacquired"
AMBIGUOUS = "ambiguous"
_ORDER = {SUPPORTED: 0, PROVISIONAL: 1, CONFLICTING: 2, REJECTED: 3}


@dataclass
class ValidationPolicy:
    same_frame_sec: float = 0.02
    # Reuses the resolver's agreement level and the identity contradiction level.
    anchor_agree_iou: float = 0.5
    anchor_conflict_iou: float = 0.2
    # On its own prompt frame SAM should sit on the seed. Below this it followed something else.
    seed_iou: float = 0.3
    # A step is a tracking switch past this many box sizes and this multiple of the track's usual step.
    jump_sizes: float = 1.0
    jump_factor: float = 5.0
    scale_step: float = 2.0
    neighbor_iou: float = 0.5
    # This much of another object's box inside this one, after growth, means it swallowed that part.
    absorb_containment: float = 0.8
    absorb_growth: float = 1.5
    # Continuity from a trusted seed vouches for a row only this close to the seed.
    supported_horizon_sec: float = 2.0
    boundary_window: float = 0.45
    # An area under this share of the seed keeps only what is visible and is marked partial.
    partial_area: float = 0.5
    # An anomaly followed by a return beside the last good box within this many frames is one outlier.
    outlier_frames: int = 3
    return_scale_step: float = 4.0
    # Recent good steps define usual motion, in frame-rate-independent units of frame widths per second.
    motion_window: int = 8
    min_motion_samples: int = 3
    min_speed: float = 0.01
    # This many measurements in a row placing the object elsewhere end the segment.
    contradictions_to_lose: int = 2
    edge_margin: float = 0.02
    occluder_overlap: float = 0.8
    shrink_ratio: float = 0.5
    shrink_window_sec: float = 0.5
    growth_limit: float = 4.0
    low_mask_fill: float = 0.15
    max_mask_components: int = 4
    mask_extent_step: float = 3.0
    # Reported only. A score never changes an outcome.
    low_score: float = 0.25


def _t(box: dict) -> tuple[float, float, float, float]:
    return (box["x"], box["y"], box["w"], box["h"])


def _center(box: dict) -> tuple[float, float]:
    return (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)


def _area(box: dict) -> float:
    return box["w"] * box["h"]


def _contained(inner: dict, outer: dict) -> float:
    left = max(inner["x"], outer["x"])
    top = max(inner["y"], outer["y"])
    right = min(inner["x"] + inner["w"], outer["x"] + outer["w"])
    bottom = min(inner["y"] + inner["h"], outer["y"] + outer["h"])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top) / max(_area(inner), 1e-9)


def _near(rows: list[dict], timestamp: float, window: float) -> list[dict]:
    return [row for row in rows if abs(row["timestamp"] - timestamp) <= window]


def validate_track(
    samples: list[dict],
    *,
    seed: dict,
    association: str,
    anchors: list[dict],
    neighbors: list[dict],
    media_size: tuple[int, int],
    boundaries: list[float] | None = None,
    policy: ValidationPolicy | None = None,
) -> list[dict]:
    """One row per returned frame, ordered by time. Each carries its outcome and every check.

    `samples` hold timestamp, bbox, confidence, frame_index, source size, and mask
    fields. They can arrive in any order. Walking outward from the seed frame
    compares each box with the one nearer the seed, in time order rather than
    response order. `anchors` and `neighbors` hold timestamp and bbox.
    """
    policy = policy or ValidationPolicy()
    seed_time = float(seed["timestamp"])
    ordered = sorted(samples, key=lambda item: item["timestamp"])
    later = [item for item in ordered if item["timestamp"] >= seed_time]
    earlier = [item for item in reversed(ordered) if item["timestamp"] < seed_time]
    rows = [
        *_walk(later, seed, association, anchors, neighbors, media_size, boundaries, policy, "f"),
        *_walk(earlier, seed, association, anchors, neighbors, media_size, boundaries, policy, "b"),
    ]
    return sorted(rows, key=lambda item: item["timestamp"])


def _speed(previous: dict, sample: dict) -> tuple[float, float, float]:
    """Center step, seconds between the two samples, and normalized speed. Frames need not be evenly spaced."""
    px, py = _center(previous["bbox"])
    cx, cy = _center(sample["bbox"])
    step = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    seconds = max(abs(sample["timestamp"] - previous["timestamp"]), 1e-3)
    return step, seconds, step / seconds


def _median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2] if values else 0.0


def _returns_at(samples: list[dict], index: int, reference: dict, usual_speed: float, policy: ValidationPolicy) -> int | None:
    """Where the track comes back beside `reference` soon after an anomaly, or None for a sustained loss."""
    for ahead in range(index + 1, min(index + 1 + policy.outlier_frames, len(samples))):
        step, seconds, _speed_now = _speed(reference, samples[ahead])
        size = max(_area(reference["bbox"]) ** 0.5, 1e-3)
        ratio = _area(samples[ahead]["bbox"]) / max(_area(reference["bbox"]), 1e-9)
        allowed = policy.jump_sizes * size + policy.jump_factor * usual_speed * seconds
        # A turning part changes area, so a size change only rules out a return when it is large.
        if step <= allowed and max(ratio, 1 / max(ratio, 1e-9)) < policy.return_scale_step:
            return ahead
    return None


def loss_hypothesis(recent: list[dict], neighbors: list[dict], policy: ValidationPolicy) -> dict:
    """What may explain a loss, from the last good boxes. This is a hypothesis and never a finding."""
    if not recent:
        return {"state": LOST, "basis": "No reliable box came before this failure."}
    last = recent[-1]
    box = last["bbox"]
    gaps = (box["x"], box["y"], 1 - box["x"] - box["w"], 1 - box["y"] - box["h"])
    if min(gaps) <= policy.edge_margin:
        return {"state": OUT_OF_VIEW, "basis": "The last reliable box touched the frame edge, so the object may have left the view."}
    for other in _near(neighbors, last["timestamp"], policy.same_frame_sec):
        if _contained(box, other["bbox"]) >= policy.occluder_overlap:
            name = other.get("object_local_id", "another object")
            return {"state": OCCLUDED, "basis": f"The last reliable box was inside {name}, so it may be hidden behind it."}
    window = [row for row in recent if last["timestamp"] - row["timestamp"] <= policy.shrink_window_sec]
    if window and _area(box) <= policy.shrink_ratio * max(_area(row["bbox"]) for row in window):
        return {
            "state": OCCLUDED,
            "basis": "The box shrank quickly before the failure. That fits a part turning away or going out of sight.",
        }
    return {"state": LOST, "basis": "Nothing stored explains this failure."}


def _walk(samples, seed, association, anchors, neighbors, media_size, boundaries, policy, walk) -> list[dict]:
    """Move outward from the seed through explicit states. A segment starts only at an initialization event.

    The seed starts the first segment. A later segment needs a measurement that agrees. A stable box after
    a loss starts nothing, because it can be stable on the wrong object.
    """
    seed_time = float(seed["timestamp"])
    seed_box = seed.get("bbox")
    trusted_seed = seed.get("trust") in {"human_verified", "independently_supported"}
    all_speeds = [_speed(a, b)[2] for a, b in zip(samples, samples[1:], strict=False)]
    previous = None
    segment = 0
    from_seed = True
    lost: dict | None = None
    cut_open = False
    outlier_until = -1
    contradictions = 0
    good: list[dict] = []
    reference_area = _area(seed_box) if seed_box else None
    previous_mask_area = None
    rows: list[dict] = []
    for index, sample in enumerate(samples):
        box = sample["bbox"]
        checks: list[dict] = []
        hard: list[str] = []
        conflict: list[str] = []
        soft: list[str] = []
        began = ""
        failure: dict | None = None

        def note(name: str, passed: bool, detail: str, bucket: list[str] | None = None) -> None:
            checks.append({"check": name, "passed": passed, "detail": detail})
            if not passed and bucket is not None:
                bucket.append(name)

        def lose(reason: str, detail: str) -> None:
            nonlocal lost, failure, from_seed
            failure = {
                "reason": reason,
                "detail": detail,
                "time": sample["timestamp"],
                "last_good_time": good[-1]["timestamp"] if good else None,
                "segment": f"{walk}{segment}",
                "hypothesis": loss_hypothesis(good[-policy.motion_window :], neighbors, policy),
            }
            lost = failure
            from_seed = False

        note("association", association == "associated", f"track association {association}", conflict if association == "conflict" else hard)
        width, height = media_size
        size_ok = not sample.get("source_width") or (sample["source_width"], sample["source_height"]) == (width, height)
        note(
            "source_coordinates",
            size_ok,
            f"SAM reported {sample.get('source_width')}x{sample.get('source_height')} for {width}x{height} media",
            hard,
        )
        has_mask = bool(sample.get("mask_data") or sample.get("mask_payload"))
        malformed = has_mask and (sample.get("mask_encoding") in {"", "unknown"} or int(sample.get("mask_height") or 0) <= 0)
        note("mask", not malformed, "mask header is malformed" if malformed else ("mask present" if has_mask else "no mask returned"), hard)
        if not has_mask:
            soft.append("mask")
        seed_mismatch = False
        if seed_box and abs(sample["timestamp"] - seed_time) <= policy.same_frame_sec:
            overlap = iou(_t(box), _t(seed_box))
            seed_mismatch = overlap < policy.seed_iou
            note("seed_overlap", not seed_mismatch, f"overlap with the seed {overlap:.2f}", conflict)
        anchor_rows = _near(anchors, sample["timestamp"], policy.same_frame_sec)
        agrees = False
        contradicted = False
        for anchor in anchor_rows:
            overlap = iou(_t(box), _t(anchor["bbox"]))
            if overlap >= policy.anchor_agree_iou:
                agrees = True
                note("anchor", True, f"agrees with {anchor.get('source', 'a measurement')} at {overlap:.2f}")
            elif overlap < policy.anchor_conflict_iou:
                contradicted = True
                note("anchor", False, f"disagrees with {anchor.get('source', 'a measurement')} at {overlap:.2f}", conflict)
            else:
                note("anchor", False, f"partly overlaps {anchor.get('source', 'a measurement')} at {overlap:.2f}", soft)
        contradictions = 0 if agrees else contradictions + int(contradicted)
        if agrees and (lost or cut_open):
            note("reassociated", True, "a measurement confirms the track again after the earlier break")
            lost = None
            cut_open = False
            segment += 1
            began = "verified_reassociation"
            good = []
            reference_area = _area(box)
        anomaly = ""
        anomaly_detail = ""
        crossed_cut = None
        if previous is not None:
            step, seconds, speed = _speed(previous, sample)
            size = max(_area(previous["bbox"]) ** 0.5, 1e-3)
            recent = [_speed(a, b)[2] for a, b in zip(good[-policy.motion_window :], good[-policy.motion_window + 1 :], strict=False)]
            usual_speed = _median(recent) if len(recent) >= policy.min_motion_samples else _median(all_speeds)
            jumped = step > policy.jump_sizes * size and speed > policy.jump_factor * max(usual_speed, policy.min_speed)
            note("continuity", not jumped, f"moved {step / size:.2f} box sizes from the neighboring frame")
            if jumped and not agrees:
                anomaly = "large_spatial_jump"
                anomaly_detail = "A large jump broke the track."
            ratio = _area(box) / max(_area(previous["bbox"]), 1e-9)
            changed = max(ratio, 1 / max(ratio, 1e-9)) >= policy.scale_step
            note("scale_step", not changed, f"area changed by {ratio:.2f} from the neighboring frame", soft)
            low, high = sorted((previous["timestamp"], sample["timestamp"]))
            crossed_cut = next((item for item in boundaries or [] if low < item <= high), None)
        for other in _near(neighbors, sample["timestamp"], policy.same_frame_sec):
            overlap = iou(_t(box), _t(other["bbox"]))
            name = other.get("object_local_id", "another object")
            if overlap >= policy.neighbor_iou:
                note("competing_object", False, f"shares this box with {name} at {overlap:.2f}", conflict)
            elif (
                seed_box
                and _contained(other["bbox"], box) >= policy.absorb_containment
                and _area(box) >= policy.absorb_growth * _area(seed_box)
            ):
                note("absorbed_neighbor", False, f"grew over {name}, so it may now cover the assembly", conflict)
                # The more specific reason. A swallowed neighbor usually moves the box center as well.
                anomaly = "absorbed_neighbor"
                anomaly_detail = f"The box grew over {name}."
        stats = sample.get("mask_stats")
        poor_mask = bool(stats) and (stats["fill"] < policy.low_mask_fill or stats["components"] > policy.max_mask_components)
        if stats:
            note("mask_quality", not poor_mask, f"mask fills {stats['fill']:.2f} of its box in {stats['components']} regions", soft)
            if previous_mask_area:
                mask_ratio = stats["pixels"] / max(previous_mask_area, 1)
                note(
                    "mask_extent",
                    max(mask_ratio, 1 / max(mask_ratio, 1e-9)) < policy.mask_extent_step,
                    f"mask area changed by {mask_ratio:.2f} from the neighboring frame",
                    soft,
                )
            previous_mask_area = stats["pixels"]
        if reference_area:
            growth = _area(box) / max(reference_area, 1e-9)
            grown = growth >= policy.growth_limit
            note("size_drift", not grown, f"area is {growth:.2f} times the segment's starting box", soft)
            # Growth alone can be the part coming closer. A poor mask as well means it took in something else.
            if grown and poor_mask and not anomaly:
                anomaly = "expanded_beyond_object"
                anomaly_detail = "The box grew well past the part and its mask no longer fills it."
                note("expanded_beyond_object", False, anomaly_detail, conflict)
        if sample.get("confidence") is not None:
            # A low score is reported and never changes an outcome. A high one never raises it.
            note("score", sample["confidence"] >= policy.low_score, f"SAM score {sample['confidence']:.2f}, uncalibrated")
        near_cut = next((item for item in boundaries or [] if abs(item - sample["timestamp"]) <= policy.boundary_window), None)
        note("scene_boundary", near_cut is None, "no hard cut nearby" if near_cut is None else f"hard cut at {near_cut:.2f}s", soft)
        outlier = index <= outlier_until
        if not lost and not outlier:
            if seed_mismatch:
                lose("seed_mismatch", "SAM did not start on the seed, so this pass followed something else.")
            elif contradictions >= policy.contradictions_to_lose:
                lose("anchor_contradiction", "Measurements keep placing the object elsewhere.")
            elif crossed_cut is not None and anomaly:
                lose("scene_cut", f"The box moved across the hard cut at {crossed_cut:.2f}s.")
            elif anomaly:
                back = _returns_at(samples, index, previous, usual_speed, policy) if previous is not None else None
                if back is None:
                    lose(anomaly, anomaly_detail)
                else:
                    outlier_until = back - 1
                    outlier = True
            elif crossed_cut is not None and not agrees:
                if not cut_open:
                    segment += 1
                    began = "scene_cut_unverified"
                cut_open = True
                from_seed = False
                good = []
                note("scene_continuity", False, f"A hard cut at {crossed_cut:.2f}s lies between these frames, so continuity is unproven.", soft)
        if outlier:
            note("transient_outlier", False, "One abnormal box. The track returns right after it.", conflict)
        if lost:
            note("track_intact", False, lost["detail"], conflict)
        seed_near = (
            from_seed
            and trusted_seed
            and not cut_open
            and abs(sample["timestamp"] - seed_time) <= policy.supported_horizon_sec
        )
        if hard:
            outcome = REJECTED
        elif conflict:
            outcome = CONFLICTING
        elif agrees or (seed_near and not soft):
            outcome = SUPPORTED
        else:
            outcome = PROVISIONAL
        partial = bool(seed_box) and _area(box) < policy.partial_area * _area(seed_box)
        if lost:
            lifecycle = LOST
        elif began == "verified_reassociation":
            lifecycle = REACQUIRED
        elif cut_open or (conflict and not outlier):
            lifecycle = AMBIGUOUS
        elif outlier or [name for name in soft if name != "mask"]:
            lifecycle = DEGRADED
        else:
            lifecycle = TRACKED
        rows.append(
            {
                "timestamp": sample["timestamp"],
                "frame_index": sample.get("frame_index"),
                "bbox": box,
                "confidence": sample.get("confidence"),
                "outcome": outcome,
                "association": association,
                "partial": partial,
                "has_mask": has_mask,
                "mask_stats": stats,
                "anchor_agrees": agrees,
                "checks": checks,
                "lifecycle": lifecycle,
                # Rows after a loss belong to no segment. They stay stored and inspectable.
                "segment": None if lost else f"{walk}{segment}",
                "segment_began": began,
                "failure": failure,
            }
        )
        if outcome != REJECTED and not outlier:
            previous = sample
            if not lost:
                good.append(sample)
    return rows


def _alternate(row: dict) -> dict:
    keys = ("bbox", "outcome", "lifecycle", "segment_id", "seed_id", "direction", "raw_track_id", "session_id", "confidence")
    return {key: row.get(key) for key in keys}


def _independent(first: dict, second: dict) -> bool:
    """Two passes from one seed share an error, so their agreement proves nothing."""
    return not (first.get("seed_id") and first.get("seed_id") == second.get("seed_id"))


def combine_passes(passes: list[list[dict]], policy: ValidationPolicy | None = None) -> list[dict]:
    """One row per frame from separate passes. The other pass's box stays on the row as an alternate.

    Agreement lifts a provisional row to supported only when the passes began from
    separate seeds, and it cannot lift a conflicting or rejected row. A lost row
    cannot contradict a row from a segment that is still valid, and it cannot support one.
    """
    policy = policy or ValidationPolicy()
    if len(passes) == 1:
        return passes[0]
    merged: list[dict] = []
    for rows in passes:
        for row in rows:
            match = next(
                (item for item in merged if abs(item["timestamp"] - row["timestamp"]) <= policy.same_frame_sec),
                None,
            )
            if match is None:
                merged.append({**row, "alternates": []})
                continue
            alternates = [*match.get("alternates", [])]
            overlap = iou(_t(match["bbox"]), _t(row["bbox"]))
            match_lost = match.get("lifecycle") == LOST
            row_lost = row.get("lifecycle") == LOST
            if match_lost != row_lost:
                keep, other = (row, match) if match_lost else (match, row)
                note = {"check": "pass_agreement", "passed": True, "detail": "the other pass had lost the track, so it was not counted"}
                match.update({**keep, "checks": [*keep["checks"], note], "alternates": [*alternates, _alternate(other)]})
            elif overlap >= policy.anchor_agree_iou:
                check = {"check": "pass_agreement", "passed": True, "detail": f"the other pass agrees at {overlap:.2f}"}
                best = min(match, row, key=lambda item: _ORDER[item["outcome"]])
                other = row if best is match else match
                outcome = best["outcome"]
                if _independent(match, row) and max(_ORDER[match["outcome"]], _ORDER[row["outcome"]]) <= _ORDER[PROVISIONAL]:
                    outcome = SUPPORTED
                match.update({**best, "outcome": outcome, "checks": [*best["checks"], check], "alternates": [*alternates, _alternate(other)]})
            elif overlap < policy.anchor_conflict_iou:
                check = {"check": "pass_agreement", "passed": False, "detail": f"the other pass disagrees at {overlap:.2f}"}
                match["outcome"] = REJECTED if match["outcome"] == REJECTED else CONFLICTING
                match["checks"] = [*match["checks"], check]
                match["alternates"] = [*alternates, _alternate(row)]
            else:
                check = {"check": "pass_agreement", "passed": False, "detail": f"the other pass partly overlaps at {overlap:.2f}"}
                if match["outcome"] == SUPPORTED:
                    match["outcome"] = PROVISIONAL
                match["checks"] = [*match["checks"], check]
                match["alternates"] = [*alternates, _alternate(row)]
    return sorted(merged, key=lambda item: item["timestamp"])


def remaining_gaps(rows: list[dict], frame_times: list[float], start: float, end: float) -> list[dict]:
    """Submitted frames in the interval with no usable box. These are unobserved, not absent."""
    wanted = [time for time in frame_times if start - 1e-6 <= time <= end + 1e-6]
    covered = {round(row["timestamp"], 3) for row in rows if row["outcome"] in {SUPPORTED, PROVISIONAL}}
    gaps: list[dict] = []
    current: list[float] = []
    for time in wanted:
        if round(time, 3) in covered:
            if current:
                gaps.append(_gap(current))
                current = []
        else:
            current.append(time)
    if current:
        gaps.append(_gap(current))
    return gaps


def _gap(times: list[float]) -> dict:
    return {"start": times[0], "end": times[-1], "frames": len(times), "outcome": UNOBSERVED}
