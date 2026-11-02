"""Split validated rows into visual segments, and keep them apart from the physical object.

A segment is one continuous run of localizations from one SAM request. SAM ids exist
only inside a request, so a segment never names a physical object by its id. Its link
to an object is a hypothesis that rests on listed evidence. A repeated label, a high
score, or a stable box is not evidence.
"""

from __future__ import annotations

import hashlib

from app.perception.recovery_validate import (
    AMBIGUOUS,
    CONFLICTING,
    DEGRADED,
    LOST,
    PROVISIONAL,
    REACQUIRED,
    REACQUISITION_CANDIDATE,
    REJECTED,
    SUPPORTED,
    TRACKED,
    ValidationPolicy,
    loss_hypothesis,
)

LINKED = "linked"
HYPOTHESIS = "hypothesis"
LINK_AMBIGUOUS = "ambiguous"
LINK_REJECTED = "rejected"
_TRUSTED = {"human_verified": "a person confirmed the seed", "independently_supported": "two producers agreed on the seed"}


def failure_id(object_local_id: str, time: float, reason: str) -> str:
    return hashlib.sha256(f"{object_local_id}|{time:.3f}|{reason}".encode()).hexdigest()[:16]


def _failed(row: dict, name: str) -> bool:
    return any(item["check"] == name and not item["passed"] for item in row["checks"])


def link_identity(seed: dict, rows: list[dict], *, began: str) -> dict:
    """Whether the evidence ties this segment to the physical object. Agreement with itself counts for nothing."""
    basis: list[str] = []
    if not began and seed.get("trust") in _TRUSTED:
        basis.append(_TRUSTED[seed["trust"]])
    measured = sum(1 for row in rows if row.get("anchor_agrees"))
    if measured:
        basis.append(f"{measured} boxes agree with an independent measurement")
    competing = sum(1 for row in rows if _failed(row, "competing_object"))
    absorbed = sum(1 for row in rows if _failed(row, "absorbed_neighbor"))
    if any(_failed(row, "seed_overlap") for row in rows):
        return {"status": LINK_REJECTED, "basis": basis, "detail": "SAM did not start on the seed."}
    if competing or absorbed:
        detail = f"{competing + absorbed} boxes overlap another known object, so which part this is stays open."
        return {"status": LINK_AMBIGUOUS, "basis": basis, "detail": detail}
    if basis:
        return {"status": LINKED, "basis": basis, "detail": "; ".join(basis).capitalize() + "."}
    return {"status": HYPOTHESIS, "basis": [], "detail": "Nothing independent ties this segment to the object yet."}


def _counts(rows: list[dict]) -> dict:
    return {name: sum(1 for row in rows if row["outcome"] == name) for name in (SUPPORTED, PROVISIONAL, CONFLICTING, REJECTED)}


def _status(rows: list[dict], identity: dict, reacquires: dict | None, began: str, closed: bool) -> str:
    if identity["status"] in {LINK_AMBIGUOUS, LINK_REJECTED}:
        return AMBIGUOUS
    if reacquires or began == "verified_reassociation":
        supported = any(row["outcome"] == SUPPORTED for row in rows)
        return REACQUIRED if identity["status"] == LINKED and supported else REACQUISITION_CANDIDATE
    if began == "scene_cut_unverified":
        return AMBIGUOUS
    if closed:
        return LOST
    degraded = sum(1 for row in rows if row["lifecycle"] == DEGRADED)
    return DEGRADED if degraded * 2 > len(rows) else TRACKED


def _ordered(walk: list[dict], name: str) -> list[dict]:
    return sorted(walk, key=lambda row: row["timestamp"], reverse=name.startswith("b"))


def _pass_report(item: dict, object_local_id: str, physical: str, reacquires: dict | None) -> tuple[list[dict], list[dict]]:
    """Segments and failures of one tracking pass."""
    rows = item["rows"]
    seed = item["seed"]
    pass_id = item["pass_id"]
    segments: list[dict] = []
    failures: list[dict] = []
    for name in sorted({row["segment"] for row in rows if row.get("segment")}):
        owned = _ordered([row for row in rows if row.get("segment") == name], name)
        began = next((row["segment_began"] for row in owned if row.get("segment_began")), "")
        walk = name[0]
        following = _ordered([row for row in rows if row["segment"] is None and row.get("failure") and row["failure"].get("segment") == name], name)
        closed = bool(following)
        identity = link_identity(seed, owned, began=began)
        usable = [row["timestamp"] for row in owned if row["outcome"] in {SUPPORTED, PROVISIONAL}]
        supported = [row["timestamp"] for row in owned if row["outcome"] == SUPPORTED]
        scores = [row["confidence"] for row in owned if row.get("confidence") is not None]
        segments.append(
            {
                "segment_id": f"{pass_id}/{name}",
                "physical_instance_id": physical,
                "object_local_id": object_local_id,
                "experiment_id": item["experiment_id"],
                "request_key": item["request_key"],
                "session_id": item["session_id"],
                "raw_track_id": item["raw_track_id"],
                "request_id": item["request_id"],
                "direction": "forward" if walk == "f" else "backward",
                "seed": {key: seed.get(key) for key in ("seed_id", "trust", "source", "timestamp", "bbox")},
                "init_event": began or ("trusted_seed" if seed.get("trust") in _TRUSTED else "candidate_seed"),
                "reacquires": reacquires,
                "start": min(row["timestamp"] for row in owned),
                "end": max(row["timestamp"] for row in owned),
                "last_supported_time": (max(supported) if walk == "f" else min(supported)) if supported else None,
                "last_usable_time": (max(usable) if walk == "f" else min(usable)) if usable else None,
                "status": _status(owned, identity, reacquires, began, closed),
                "identity": identity,
                "counts": _counts(owned),
                "frames": len(owned),
                "mean_confidence": round(sum(scores) / len(scores), 4) if scores else None,
                "failure_id": None,
            }
        )
    for row in rows:
        failure = row.get("failure")
        if not failure:
            continue
        identifier = failure_id(object_local_id, failure["time"], failure["reason"])
        walk = "f" if row["timestamp"] >= float(seed["timestamp"]) else "b"
        last_good = failure.get("last_good_time")
        owner = next(
            (segment for segment in segments if segment["segment_id"].startswith(f"{pass_id}/{walk}") and segment["direction"][0] == walk
             and last_good is not None and segment["start"] - 1e-6 <= last_good <= segment["end"] + 1e-6),
            None,
        )
        if owner:
            owner["failure_id"] = identifier
            if owner["status"] in {TRACKED, DEGRADED}:
                owner["status"] = LOST
        failures.append(
            {
                "failure_id": identifier,
                "physical_instance_id": physical,
                "object_local_id": object_local_id,
                "segment_id": owner["segment_id"] if owner else None,
                "experiment_id": item["experiment_id"],
                "request_key": item["request_key"],
                "time": failure["time"],
                "last_good_time": last_good,
                "last_supported_time": owner["last_supported_time"] if owner else None,
                "direction": "forward" if walk == "f" else "backward",
                "reason": failure["reason"],
                "detail": failure["detail"],
                "state": failure["hypothesis"]["state"],
                "state_basis": failure["hypothesis"]["basis"],
                "reacquisition_required": True,
                "reacquires": reacquires,
            }
        )
    return segments, failures


def _ended_early(item: dict, segments: list[dict], neighbors: list[dict], object_local_id: str, physical: str, policy: ValidationPolicy) -> list[dict]:
    """SAM stopped returning the object before the clip ended. That is a failure with no box to blame."""
    rows = sorted(item["rows"], key=lambda row: row["timestamp"])
    times = item["frame_times"]
    direction = item["direction"]
    if not rows or not times or any(row.get("failure") for row in rows):
        return []
    found = []
    seed_time = float(item["seed"]["timestamp"])
    if direction in {"forward", "both"} and rows[-1]["timestamp"] < times[-1] - 1e-6 and rows[-1]["timestamp"] >= seed_time - policy.same_frame_sec:
        found.append(("forward", rows[-1], rows, next(time for time in times if time > rows[-1]["timestamp"] + 1e-6)))
    if direction in {"backward", "both"} and rows[0]["timestamp"] > times[0] + 1e-6 and rows[0]["timestamp"] <= seed_time + policy.same_frame_sec:
        found.append(("backward", rows[0], rows[::-1], next(time for time in reversed(times) if time < rows[0]["timestamp"] - 1e-6)))
    failures = []
    for walk, last, ordered, when in found:
        hypothesis = loss_hypothesis(ordered[-policy.motion_window :], neighbors, policy)
        owner = next((seg for seg in segments if seg["direction"] == walk and seg["start"] - 1e-6 <= last["timestamp"] <= seg["end"] + 1e-6), None)
        identifier = failure_id(object_local_id, when, "no_detection")
        if owner:
            owner["failure_id"] = identifier
            if owner["status"] in {TRACKED, DEGRADED}:
                owner["status"] = LOST
        failures.append(
            {
                "failure_id": identifier,
                "physical_instance_id": physical,
                "object_local_id": object_local_id,
                "segment_id": owner["segment_id"] if owner else None,
                "experiment_id": item["experiment_id"],
                "request_key": item["request_key"],
                "time": when,
                "last_good_time": last["timestamp"],
                "last_supported_time": owner["last_supported_time"] if owner else None,
                "direction": walk,
                "reason": "no_detection",
                "detail": "SAM stopped returning this object. That does not mean it is gone.",
                "state": hypothesis["state"],
                "state_basis": hypothesis["basis"],
                "reacquisition_required": True,
                "reacquires": item.get("reacquires"),
            }
        )
    return failures


def unresolved_spans(frame_times: list[float], usable: set[float], returned: set[float], start: float, end: float) -> list[dict]:
    """Frames with no validated box, split into what SAM never returned and what was returned and not trusted."""
    spans: list[dict] = []
    for time in frame_times:
        if time < start - 1e-6 or time > end + 1e-6:
            continue
        key = round(time, 3)
        if key in usable:
            if spans and spans[-1].get("open"):
                spans[-1]["open"] = False
            continue
        if not spans or not spans[-1].get("open"):
            spans.append({"start": time, "end": time, "frames": 0, "no_detection_frames": 0, "unusable_frames": 0, "open": True})
        span = spans[-1]
        span["end"] = time
        span["frames"] += 1
        span["unusable_frames" if key in returned else "no_detection_frames"] += 1
    for span in spans:
        span.pop("open", None)
    return spans


def track_report(target: dict, passes: list[dict], neighbors: list[dict], policy: ValidationPolicy | None = None) -> dict:
    """Segments, failures, and unresolved spans for one target. Replaying a stored response repeats this exactly."""
    policy = policy or ValidationPolicy()
    object_local_id = target["object_local_id"]
    physical = target.get("instance_id") or object_local_id
    reacquires = target.get("reacquires")
    segments: list[dict] = []
    failures: list[dict] = []
    for item in passes:
        found, lost = _pass_report(item, object_local_id, physical, reacquires)
        failures.extend(lost)
        failures.extend(_ended_early({**item, "reacquires": reacquires}, found, neighbors, object_local_id, physical, policy))
        segments.extend(found)
    start, end = float(target["time_start"]), float(target["time_end"])
    frames = sorted({time for item in passes for time in item["frame_times"]})
    rows = [row for item in passes for row in item["rows"]]
    usable = {round(row["timestamp"], 3) for row in rows if row["outcome"] in {SUPPORTED, PROVISIONAL}}
    returned = {round(row["timestamp"], 3) for row in rows}
    spans = unresolved_spans(frames, usable, returned, start, end)
    limits = []
    with_masks = [row for row in rows if row.get("has_mask")]
    if not with_masks:
        limits.append("No masks came back, so these checks used boxes only.")
    elif not any(row.get("mask_stats") for row in with_masks):
        limits.append("Masks could not be decoded here, so these checks used boxes only.")
    return {
        "physical_instance_id": physical,
        "object_local_id": object_local_id,
        "segments": sorted(segments, key=lambda item: item["start"]),
        "failures": sorted(failures, key=lambda item: item["time"]),
        "unresolved": spans,
        "track_breaks": len(failures),
        "no_detection_frames": sum(item["no_detection_frames"] for item in spans),
        "unusable_frames": sum(item["unusable_frames"] for item in spans),
        "identity_ambiguities": sum(1 for item in segments if item["identity"]["status"] in {LINK_AMBIGUOUS, LINK_REJECTED}),
        "evidence_limits": limits,
        "state_note": "Occluded and out of view are hypotheses from geometry. They are not findings.",
    }
