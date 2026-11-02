"""Decide whether a later detection is the physical object a track lost. No provider and no model call.

Detection and identity are separate questions. A detection says something is at a place. Only evidence
from another producer, a person, or an appearance match on a part with no look-alike says which part it
is. Being near the last box proves nothing, and a SAM score is never used. Only a verified candidate may
start a seed without a person. The rest stay listed with the reason they stopped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.perception.assignment import iou
from app.perception.resolve import AGREEMENT_IOU, SAME_FRAME_SEC

VERIFIED_REAPPEARANCE = "verified_reappearance"
PROBABLE_REAPPEARANCE = "probable_reappearance"
AMBIGUOUS_CANDIDATE = "ambiguous_candidate"
REJECTED_CANDIDATE = "rejected_candidate"
NO_DETECTION = "no_detection"
# The only appearance encoder allowed to help verify. It is Meta's. A color histogram can only suggest.
TRUSTED_ENCODER = "dinov2_vits14"


@dataclass
class ReappearancePolicy:
    agree_iou: float = AGREEMENT_IOU
    neighbor_iou: float = 0.5
    # Cosines from one encoder. They support an association and never prove one.
    appearance_match: float = 0.8
    appearance_mismatch: float = 0.4
    max_area: float = 0.8
    same_frame_sec: float = SAME_FRAME_SEC
    # Bounds on what a plan may ask for. A later attempt needs its own explicit approval.
    search_window_sec: float = 5.0
    max_redetection_attempts: int = 1
    max_reinitialization_attempts: int = 2
    max_clip_sec: float = 6.0
    forward_after_seed_sec: float = 2.0


def candidate_id(object_local_id: str, source: str, timestamp: float, box: dict, track: str = "") -> str:
    text = f"{object_local_id}|{source}|{timestamp:.3f}|{track}|{sorted(box.items())}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _tuple(box: dict) -> tuple[float, float, float, float]:
    return (box["x"], box["y"], box["w"], box["h"])


def _same_frame(rows: list[dict], time: float, policy: ReappearancePolicy) -> list[dict]:
    return [row for row in rows if abs(row["timestamp"] - time) <= policy.same_frame_sec]


def _check(name: str, passed: bool | None, detail: str) -> dict:
    return {"check": name, "passed": passed, "detail": detail}


def _crossed_cut(boundaries: list[float], start: float, end: float) -> float | None:
    low, high = min(start, end), max(start, end)
    return next((time for time in boundaries if low <= time <= high), None)


def _distinct_rivals(candidate: dict, candidates: list[dict], policy: ReappearancePolicy) -> list[dict]:
    """Other detections on this frame at another place. Each could be a part that looks the same."""
    return [
        other
        for other in _same_frame(candidates, candidate["timestamp"], policy)
        if other is not candidate and iou(_tuple(other["bbox"]), _tuple(candidate["bbox"])) < policy.neighbor_iou
    ]


def verify_candidate(candidate: dict, candidates: list[dict], context: dict, policy: ReappearancePolicy | None = None) -> dict:
    """One candidate with its outcome and every check that decided it."""
    policy = policy or ReappearancePolicy()
    box = candidate["bbox"]
    time = candidate["timestamp"]
    local_id = context["object_local_id"]
    family = candidate.get("family", "")
    checks: list[dict] = []
    if candidate.get("trust") == "human_verified":
        checks.append(_check("human_confirmed", True, "A person placed or confirmed this box."))
        return _result(candidate, VERIFIED_REAPPEARANCE, checks, "A person identified this box.")
    extent = box["w"] * box["h"]
    if extent > policy.max_area:
        checks.append(_check("box_extent", False, f"Covers {extent:.2f} of the frame."))
        return _result(candidate, REJECTED_CANDIDATE, checks, "The box covers most of the frame, so it is not one part.")
    own = [
        row
        for row in _same_frame(context.get("anchors", {}).get(local_id, []), time, policy)
        if row.get("source") != family and iou(_tuple(row["bbox"]), _tuple(box)) >= policy.agree_iou
    ]
    others = [
        row
        for other, rows in context.get("anchors", {}).items()
        if other != local_id
        for row in _same_frame(rows, time, policy)
        if iou(_tuple(row["bbox"]), _tuple(box)) >= policy.neighbor_iou
    ]
    names = sorted({row.get("object_local_id", "") for row in others})
    checks.append(_check("independent_location", bool(own) or None, (
        f"Agrees with {', '.join(sorted({row['source'] for row in own}))} on this frame." if own else "No other producer locates this object here."
    )))
    checks.append(_check("other_objects", not others, f"Overlaps {', '.join(names)}." if others else "No other known object is measured here."))
    cut = _crossed_cut(context.get("boundaries", []), context.get("last_good_time") or context["failure_time"], time)
    checks.append(_check("scene_continuity", cut is None, f"A hard cut at {cut:.2f}s lies between the loss and this box." if cut is not None else "No hard cut since the loss."))
    if own and others:
        return _result(candidate, AMBIGUOUS_CANDIDATE, checks, f"Evidence places both {local_id} and {', '.join(names)} here. A person has to choose.")
    if own:
        return _result(candidate, VERIFIED_REAPPEARANCE, checks, "A separate producer places this object on the same box.")
    if others:
        return _result(candidate, REJECTED_CANDIDATE, checks, f"This box is measured as {', '.join(names)}.")
    rivals = _distinct_rivals(candidate, candidates, policy)
    if any(_has_own_anchor(other, context, policy) for other in rivals):
        checks.append(_check("rival_detections", False, "Another detection on this frame has independent support."))
        return _result(candidate, REJECTED_CANDIDATE, checks, "A different box on this frame is the one with independent support.")
    look_alikes = [item for item in context.get("same_label", []) if item != local_id]
    checks.append(_check("look_alikes", not look_alikes, (
        f"{', '.join(look_alikes)} share this label." if look_alikes else "No other object in this analysis shares the label."
    )))
    checks.append(_check("rival_detections", not rivals, f"{len(rivals)} other detections on this frame." if rivals else "The only detection on this frame."))
    appearance = candidate.get("appearance")
    if appearance is None:
        checks.append(_check("appearance", None, "Appearance was not measured."))
    else:
        checks.append(_check("appearance", appearance["score"] >= policy.appearance_match, f"{appearance['encoder']} cosine {appearance['score']:.2f} against the last supported box."))
    if look_alikes or rivals:
        return _result(candidate, AMBIGUOUS_CANDIDATE, checks, "More than one part could be this detection. A person has to choose.")
    if appearance is not None and appearance["score"] <= policy.appearance_mismatch:
        # A turned part can look different, so this stops short of rejecting it.
        return _result(candidate, PROBABLE_REAPPEARANCE, checks, "It looks different from the last supported box, which a turned part can also do. A person has to confirm it.")
    matches = appearance is not None and appearance["score"] >= policy.appearance_match
    if matches and appearance["encoder"] == TRUSTED_ENCODER and cut is None:
        return _result(candidate, VERIFIED_REAPPEARANCE, checks, "It is the only detection, no other part shares the label, and the appearance matches.")
    return _result(candidate, PROBABLE_REAPPEARANCE, checks, "It could be this object, and nothing independent confirms it. A person has to confirm it.")


def _has_own_anchor(candidate: dict, context: dict, policy: ReappearancePolicy) -> bool:
    return any(
        row.get("source") != candidate.get("family", "") and iou(_tuple(row["bbox"]), _tuple(candidate["bbox"])) >= policy.agree_iou
        for row in _same_frame(context.get("anchors", {}).get(context["object_local_id"], []), candidate["timestamp"], policy)
    )


def _result(candidate: dict, outcome: str, checks: list[dict], reason: str) -> dict:
    return {**candidate, "outcome": outcome, "checks": checks, "reason": reason, "needs_confirmation": outcome in {PROBABLE_REAPPEARANCE, AMBIGUOUS_CANDIDATE}}


def verify_candidates(candidates: list[dict], context: dict, policy: ReappearancePolicy | None = None) -> list[dict]:
    """Every candidate judged, verified first and then earliest. Two verified places at once cancel out."""
    policy = policy or ReappearancePolicy()
    judged = [verify_candidate(item, candidates, context, policy) for item in candidates]
    for item in judged:
        if item["outcome"] != VERIFIED_REAPPEARANCE or item.get("trust") == "human_verified":
            continue
        clash = any(
            other is not item and other["outcome"] == VERIFIED_REAPPEARANCE
            and abs(other["timestamp"] - item["timestamp"]) <= policy.same_frame_sec
            and iou(_tuple(other["bbox"]), _tuple(item["bbox"])) < policy.neighbor_iou
            for other in judged
        )
        if clash:
            item["outcome"] = AMBIGUOUS_CANDIDATE
            item["needs_confirmation"] = True
            item["reason"] = "Two different boxes on this frame both have support. A person has to choose."
    rank = {VERIFIED_REAPPEARANCE: 0, PROBABLE_REAPPEARANCE: 1, AMBIGUOUS_CANDIDATE: 2, REJECTED_CANDIDATE: 3}
    return sorted(judged, key=lambda item: (rank[item["outcome"]], item["timestamp"], item["candidate_id"]))


def search_result(judged: list[dict]) -> str:
    """The best outcome of a search. An empty search reports no detection and never a guess."""
    return judged[0]["outcome"] if judged else NO_DETECTION


def search_window(failure_time: float, frames: list[float], hidden: list[float], policy: ReappearancePolicy, *, expand: bool = False, direction: str = "forward") -> dict:
    """Frames worth searching. Stored rows saying the object is hidden move the start past them."""
    span = policy.search_window_sec * (2 if expand else 1)
    sign = 1 if direction == "forward" else -1
    start = failure_time
    for time in sorted(hidden, key=lambda item: sign * item):
        if sign * (time - start) >= 0 and sign * (time - start) <= 0.5:
            start = time
    limit = failure_time + sign * span
    limited = sign * (start - limit) >= 0
    if limited:
        return {
            "start": min(failure_time, limit),
            "end": max(failure_time, limit),
            "frames": [],
            "limited": True,
            "note": f"Stored rows keep this object hidden past the {span:.1f}s search limit. Widening the search needs its own approval.",
        }
    low, high = min(start, limit), max(start, limit)
    inside = [time for time in frames if low - 1e-6 <= time <= high + 1e-6 and sign * (time - failure_time) >= 0]
    note = "" if inside else "No source frames remain in this search window, so the recording has nothing left to search."
    return {"start": low, "end": high, "frames": inside, "limited": False, "note": note}


def diagnose_failure(failure: dict, *, rows_after: list[dict], frames_after: list[float], candidates: list[dict], other_tracks: list[dict], anchors: list[dict], clip_end: float, video_end: float, missing_accepted: int, planned: bool, executed: bool, policy: ReappearancePolicy | None = None) -> dict:
    """Which kind of failure this is, read from stored rows. A missing frontend box is not assumed to mean SAM returned none."""
    policy = policy or ReappearancePolicy()
    raw_times = [row["timestamp"] for row in rows_after]
    returned = {round(time, 3) for time in raw_times}
    missing = [time for time in frames_after if round(time, 3) not in returned]
    agree = contradict = 0
    for row in rows_after:
        same = _same_frame(anchors, row["timestamp"], policy)
        if not same:
            continue
        if any(iou(_tuple(item["bbox"]), _tuple(row["bbox"])) >= policy.agree_iou for item in same):
            agree += 1
        else:
            contradict += 1
    categories: list[dict] = []

    def add(code: str, detail: str) -> None:
        categories.append({"code": code, "detail": detail})

    if not rows_after and not other_tracks:
        add("A_sam_stopped", "SAM returned no box for this object after the failure.")
    if other_tracks:
        names = ", ".join(sorted({f"SAM track {item['raw_track_id']}" for item in other_tracks}))
        add("B_new_track_id", f"Later detections exist under another id: {names}.")
    if agree:
        add("C_rejected_by_validation", f"{agree} boxes the track kept returning agree with an independent measurement, and validation did not trust them.")
    if contradict:
        add("D_followed_wrong_object", f"{contradict} boxes the track kept returning disagree with an independent measurement.")
    if rows_after and not agree and not contradict:
        add("C_or_D_undetermined", f"SAM kept returning {len(rows_after)} boxes after the failure. No independent measurement can tell a correct box from a drifted one.")
    if missing_accepted:
        add("E_not_in_analysis", f"{missing_accepted} usable boxes are not in the active revision, either because the result was not accepted or provisional boxes were left out.")
    return {
        "categories": categories,
        "last_raw_time": max(raw_times) if raw_times else failure.get("last_good_time"),
        "last_supported_time": failure.get("last_supported_time"),
        "first_no_detection_time": missing[0] if missing else None,
        "raw_boxes_after_failure": len(rows_after),
        "later_track_ids": sorted({str(item["raw_track_id"]) for item in other_tracks}),
        "rejected_by_validation": sum(1 for row in rows_after if row.get("outcome") in {"conflicting", "rejected"}),
        "detected_not_identified": sum(1 for item in candidates if item["outcome"] in {PROBABLE_REAPPEARANCE, AMBIGUOUS_CANDIDATE}),
        "analyzed_through": clip_end,
        "video_end": video_end,
        "clip_ended_early": clip_end < video_end - 1e-3,
        "request_planned": planned,
        "request_executed": executed,
    }
