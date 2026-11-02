"""What is known about each object's visual tracks, and what could reconnect them. Plans only.

Nothing here contacts a provider or changes a row. Every plan names the evidence it
would use. A drifting track is never offered as a seed for itself. An object with no
trustworthy evidence stays unresolved, and the plan says so.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import SpatialTrackPoint
from app.models.media import VideoAsset
from app.models.recovery import RecoveryExperiment, RecoveryProposal, RecoveryRequest
from app.models.runs import AnalysisRun
from app.perception.assignment import iou
from app.perception.encoder import DINOV2_ID, HistogramEncoder, load_encoder, probe_encoder
from app.perception.reappearance import (
    AMBIGUOUS_CANDIDATE,
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
from app.perception.recovery_validate import REACQUIRED
from app.perception.seeds import (
    CANDIDATE,
    HUMAN_VERIFIED,
    INDEPENDENTLY_SUPPORTED,
    Seed,
    collect_seeds,
)
from app.perception.track_segments import LINKED, unresolved_spans
from app.pipeline.sampling_stage import timeline_for
from app.providers.sam import concept_text, parse_sam_output
from app.services.recovery import (
    DEFAULT_LIMITS,
    RecoveryError,
    _boundaries,
    _document,
    _limits,
    _manual_target,
    _measured_anchors,
    _usable_spatial,
    dry_run,
    load_response,
    new_experiment,
    settings,
)
from app.services.recovery_diagnostics import raw_boxes
from app.services.storage import resolve_stored
from app.video.frames import extract_frame, read_bgr
from app.video.probe import VideoError

# How far past the uncertain interval to look for stored boxes and frames worth checking.
SEARCH_AFTER_SEC = 3.0
FORWARD_AFTER_SEED_SEC = 1.0
MAX_CANDIDATES = 8
MAX_FRAMES_TO_CHECK = 5
COMPETING_IOU = 0.5
POLICY = ReappearancePolicy()
_APPEARANCE_CACHE: dict[tuple, dict] = {}
_ENCODER = None
_ENCODER_LOCK = threading.Lock()


def _usable_after(failure: dict, segment: dict) -> bool:
    """A later linked segment with supported boxes, on the far side of the failure from its own seed."""
    if segment["identity"]["status"] != LINKED or not segment["counts"]["supported"] or segment.get("decision") == "rejected":
        return False
    if segment["segment_id"] == failure["segment_id"]:
        return False
    if failure["direction"] == "forward":
        return segment["end"] > failure["time"]
    return segment["start"] < failure["time"]


def _seed_row(seed: Seed, others: list[Seed]) -> dict:
    return {
        "source": seed.source,
        "trust": seed.trust,
        "seed_id": seed.seed_id,
        "timestamp": seed.timestamp,
        "bbox": seed.bbox,
        "explanation": seed.explanation,
        "needs_confirmation": seed.trust == CANDIDATE,
        "may_be": _competitors(seed.bbox, seed.timestamp, others),
    }


def _competitors(box: dict | None, time: float, seeds: list[Seed]) -> list[str]:
    if not box:
        return []
    names = set()
    for other in seeds:
        if abs(other.timestamp - time) > 0.02 or not other.bbox:
            continue
        one = (box["x"], box["y"], box["w"], box["h"])
        two = (other.bbox["x"], other.bbox["y"], other.bbox["w"], other.bbox["h"])
        if iou(one, two) >= COMPETING_IOU:
            names.add(other.object_local_id)
    return sorted(names)


def _usable_frame(rows: list[dict], policy_margin: float = 0.005) -> dict:
    """The first box of a track that does not touch the frame edge, since an entering part shows only a sliver."""
    for row in rows:
        box = row["bbox"]
        if min(box["x"], box["y"], 1 - box["x"] - box["w"], 1 - box["y"] - box["h"]) > policy_margin:
            return row
    return rows[0]


def _cached_tracks(experiments: list[RecoveryExperiment], object_local_id: str, label: str, labels: dict[str, str], start: float, end: float, seeds: list[Seed]) -> tuple[list[dict], int, list[dict]]:
    """One candidate per stored track tied to no object. Nothing is sent. Returns candidates, boxes read, and specs read."""
    found: list[dict] = []
    seen: set[tuple] = set()
    inspected: list[dict] = []
    boxes_read = 0
    for experiment in experiments:
        for spec in (experiment.plan_json or {}).get("requests") or []:
            concept = spec.get("prompt_kind") == "concept"
            # A search for this label may show this object, whichever object first asked for it.
            if concept and labels.get(spec.get("object_local_id", ""), "") != label and spec.get("object_local_id") != object_local_id:
                continue
            if not concept and object_local_id not in spec["bindings"].values():
                continue
            payload = load_response(experiment.run_id, spec["key"])
            if payload is None:
                continue
            inspected.append({"request_key": spec["key"], "prompt_kind": spec.get("prompt_kind", "spatial"), "prompt": spec["prompt"]})
            tracks: dict[str, list[dict]] = {}
            for box in raw_boxes(parse_sam_output(payload.get("text") or ""), spec, {}):
                if box["status"] != "unassigned" or box["bbox"] is None or not start <= box["timestamp"] <= end:
                    continue
                boxes_read += 1
                tracks.setdefault(box["raw_track_id"], []).append(box)
            for track, rows in tracks.items():
                identity = (spec["key"], track)
                if identity in seen:
                    continue
                seen.add(identity)
                ordered = sorted(rows, key=lambda item: item["timestamp"])
                box = _usable_frame(ordered)
                source = "cached_concept" if concept else "cached_unprompted_track"
                found.append(
                    {
                        "candidate_id": candidate_id(object_local_id, source, box["timestamp"], box["bbox"], track),
                        "source": source,
                        "family": "sam",
                        "trust": CANDIDATE,
                        "seed_id": "",
                        "timestamp": box["timestamp"],
                        "bbox": box["bbox"],
                        "raw_track_id": track,
                        "request_key": spec["key"],
                        "track_frames": len(ordered),
                        "track_end": ordered[-1]["timestamp"],
                        "explanation": "SAM found something here. Nothing says it is this physical part.",
                        "needs_confirmation": True,
                        "may_be": _competitors(box["bbox"], box["timestamp"], seeds),
                    }
                )
    return sorted(found, key=lambda item: item["timestamp"]), boxes_read, inspected


def _pipeline_tracks(session: Session, run_id: str, object_local_id: str, start: float, end: float, seeds: list[Seed]) -> tuple[list[dict], int]:
    """SAM tracks the normal pipeline stored and could not assign to any object."""
    rows = session.scalars(
        select(SpatialTrackPoint).where(
            SpatialTrackPoint.run_id == run_id,
            SpatialTrackPoint.detector_id == "sam-3.1",
            SpatialTrackPoint.timestamp >= start,
            SpatialTrackPoint.timestamp <= end,
        )
    ).all()
    tracks: dict[str, list[dict]] = {}
    read = 0
    for row in rows:
        if "association=associated" in row.identity_note:
            continue
        read += 1
        tracks.setdefault(row.track_ref, []).append({"timestamp": row.timestamp, "bbox": row.bbox_json})
    found = []
    for track, items in sorted(tracks.items()):
        ordered = sorted(items, key=lambda item: item["timestamp"])
        box = _usable_frame(ordered)
        found.append(
            {
                "candidate_id": candidate_id(object_local_id, "pipeline_unassigned_track", box["timestamp"], box["bbox"], track),
                "source": "pipeline_unassigned_track",
                "family": "sam",
                "trust": CANDIDATE,
                "seed_id": "",
                "timestamp": box["timestamp"],
                "bbox": box["bbox"],
                "raw_track_id": track,
                "request_key": "",
                "track_frames": len(ordered),
                "track_end": ordered[-1]["timestamp"],
                "explanation": "The pipeline stored this SAM track without an object.",
                "needs_confirmation": True,
                "may_be": _competitors(box["bbox"], box["timestamp"], seeds),
            }
        )
    return found, read


def _appearance_encoder():
    """Loaded once per server process, however many history requests arrive together."""
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            _ENCODER = HistogramEncoder()
            try:
                hub = Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch") / "hub" / "checkpoints"
                if any(hub.glob("dinov2_vits14*")) and probe_encoder(DINOV2_ID)["available"]:
                    _ENCODER = load_encoder(DINOV2_ID)
            except Exception:  # noqa: BLE001 - an unavailable encoder leaves appearance as a hint only.
                _ENCODER = HistogramEncoder()
        return _ENCODER


def _appearance(source, reference: dict | None, candidate: dict) -> dict | None:
    """Cosine of two crops. It uses DINOv2 weights already on this machine, otherwise a histogram that can only suggest."""
    if reference is None or source is None:
        return None
    key = (round(reference["timestamp"], 3), tuple(sorted(reference["bbox"].items())), round(candidate["timestamp"], 3), tuple(sorted(candidate["bbox"].items())))
    if key in _APPEARANCE_CACHE:
        return _APPEARANCE_CACHE[key]
    try:
        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder) / "a.jpg", Path(folder) / "b.jpg"
            extract_frame(source, reference["timestamp"], one)
            extract_frame(source, candidate["timestamp"], two)
            crops = [_crop(read_bgr(one), reference["bbox"]), _crop(read_bgr(two), candidate["bbox"])]
    except (OSError, ValueError, RuntimeError, VideoError):
        return None
    encoder = _appearance_encoder()
    vectors = encoder.encode(crops)
    result = {"encoder": encoder.encoder_id, "score": round(float(np.dot(vectors[0], vectors[1])), 4)}
    _APPEARANCE_CACHE[key] = result
    return result


def _crop(frame: np.ndarray, box: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1 = int(box["x"] * width), int(box["y"] * height)
    x2, y2 = max(int((box["x"] + box["w"]) * width), x1 + 1), max(int((box["y"] + box["h"]) * height), y1 + 1)
    return frame[y1:y2, x1:x2]


def _proposed(failure: dict, evidence: dict | None, strategy: str, interval_start: float, interval_end: float) -> dict:
    target: dict = {
        "object_local_id": failure["object_local_id"],
        "reacquires": {
            "failure_id": failure["failure_id"],
            "failure_time": failure["time"],
            "experiment_id": failure["experiment_id"],
            "strategy": strategy,
        },
    }
    if strategy == "concept_redetection":
        target["strategy"] = "concept"
    if evidence is None:
        return {**target, "time_start": interval_start, "time_end": interval_end, "direction": "auto"}
    time = evidence["timestamp"]
    return {
        **target,
        "time_start": min(interval_start, time),
        "time_end": max(time + FORWARD_AFTER_SEED_SEC, interval_start),
        "direction": "both",
        "seed_id": evidence["seed_id"],
        "confirm_seed": evidence["needs_confirmation"],
    }


def _seed_candidates(window: list[Seed], others: list[Seed]) -> list[dict]:
    """Stored boxes of this object past the failure. Each goes through the same identity checks."""
    return [
        {
            **_seed_row(seed, others),
            "candidate_id": candidate_id(seed.object_local_id, seed.source, seed.timestamp, seed.bbox or {}, seed.seed_id),
            "family": seed.source,
            "raw_track_id": "",
            "request_key": "",
            "track_frames": 1,
            "track_end": seed.timestamp,
        }
        for seed in window
        if seed.bbox
    ]


def _strategies(failure, seeds, cached, phrase, attempted, interval_start, interval_end, verified=None, concept_allowed=True, ruled_out=frozenset()) -> list[dict]:
    # Anywhere past the failure counts, nearest first. The dry run reports whether the clip fits the limits.
    forward = failure["direction"] == "forward"
    window = sorted(
        (
            seed
            for seed in seeds
            if seed.object_local_id == failure["object_local_id"]
            and seed.seed_id not in attempted
            and seed.seed_id not in ruled_out
            # A prompted SAM track cannot vouch for itself, so its own output is never a reacquisition seed.
            and seed.source != "sam_spatial"
            and (seed.timestamp > failure["time"] if forward else seed.timestamp < failure["time"])
        ),
        key=lambda seed: (abs(seed.timestamp - failure["time"]), seed.seed_id),
    )
    independent = [_seed_row(seed, seeds) for seed in window if seed.trust == INDEPENDENTLY_SUPPORTED]
    human = [_seed_row(seed, seeds) for seed in window if seed.trust == HUMAN_VERIFIED]
    candidates = [_seed_row(seed, seeds) for seed in window if seed.trust == CANDIDATE] + cached
    candidates = candidates[:MAX_CANDIDATES]
    has_cached = bool(independent or human or candidates)
    return [
        {
            "strategy": "verified_reappearance",
            "available": bool(verified),
            "requires_remote": True,
            "evidence": (verified or [])[:MAX_CANDIDATES],
            "note": "Stored evidence identifies the object after the failure. A new spatial pass can start from it after you approve the plan.",
        },
        {
            "strategy": "independent_observation",
            "available": bool(independent),
            "requires_remote": True,
            "evidence": independent[:MAX_CANDIDATES],
            "note": "A measurement from another producer locates the object after the failure.",
        },
        {
            "strategy": "human_seed",
            "available": bool(human),
            "requires_remote": True,
            "evidence": human[:MAX_CANDIDATES],
            "note": "A person placed a box on the object after the failure.",
        },
        {
            "strategy": "candidate_detection",
            "available": bool(candidates),
            "requires_remote": True,
            "evidence": candidates,
            "note": "Stored boxes may show the object. A person has to confirm which one, if any.",
        },
        {
            "strategy": "concept_redetection",
            "available": not has_cached and concept_allowed,
            "requires_remote": True,
            "evidence": [],
            "concept": phrase,
            "time_start": interval_start,
            "time_end": interval_end,
            "note": "A short SAM concept pass could find possible locations. It cannot say which physical part each one is.",
        },
        {
            "strategy": "manual",
            "available": True,
            "requires_remote": True,
            "evidence": [],
            "note": "Find a frame where the object is visible again and draw a box on it.",
        },
    ]


def _explanation(failure: dict, raw_after: int, recommended: dict, attempts: list[dict], resolved: dict | None, spans: list[dict]) -> str:
    time = failure["time"]
    if resolved is not None:
        text = (
            f"A new seed at {resolved['seed']['timestamp']:.2f}s started a separate segment with "
            f"{resolved['counts']['supported']} supported boxes."
        )
        if spans:
            text += f" {spans[0]['start']:.2f}–{spans[-1]['end']:.2f}s has frames with no validated box and stays unresolved."
        return text
    if failure["reason"] == "no_detection":
        lead = f"SAM stopped returning this object at {time:.2f}s."
    else:
        lead = f"SAM kept returning boxes ({raw_after} after the failure), and tracking became unreliable at {time:.2f}s."
    if attempts:
        lead += f" {len(attempts)} reacquisition attempts did not produce a verified segment, so stronger evidence is needed."
    if recommended["strategy"] == "verified_reappearance":
        return f"{lead} Stored evidence identifies the object after the failure, so a new spatial pass can start there. Nothing has been sent."
    if recommended["strategy"] in {"independent_observation", "human_seed"}:
        return f"{lead} A trusted seed is available after the failure. Nothing has been sent."
    if recommended["strategy"] == "candidate_detection":
        return f"{lead} Stored boxes may show the object, and a person has to confirm one. Nothing has been sent."
    return f"{lead} No independently verified reappearance seed was available, so automatic reacquisition was not attempted."


def reacquisition(session: Session, run: AnalysisRun, expand: set[str] | None = None) -> dict:
    experiments = list(
        session.scalars(
            select(RecoveryExperiment).where(RecoveryExperiment.run_id == run.id).order_by(RecoveryExperiment.created_at)
        )
    )
    proposals: list[tuple[RecoveryExperiment, RecoveryProposal]] = []
    for experiment in experiments:
        rows = session.scalars(
            select(RecoveryProposal).where(
                RecoveryProposal.experiment_id == experiment.id, RecoveryProposal.generation == experiment.generation
            )
        ).all()
        proposals.extend((experiment, row) for row in rows)
    document = _document(session, run.active_revision_id)
    seeds = collect_seeds(document, _usable_spatial(run.id), boundaries=_boundaries(session, run.id))
    timeline = timeline_for(session.get(VideoAsset, run.video_id), None)
    labels = {obj.local_id: obj.label for obj in document.objects}
    instances = {obj.local_id: obj.instance_id for obj in document.objects}
    video = session.get(VideoAsset, run.video_id)
    shared = {
        "labels": labels,
        "anchors": _measured_anchors(document, _usable_spatial(run.id)),
        "boundaries": _boundaries(session, run.id),
        "document": document,
        "source": _video_path(video),
        "video_end": timeline.pts[-1] if timeline.pts else 0.0,
        "run_id": run.id,
    }
    grouped: dict[str, list[tuple[RecoveryExperiment, RecoveryProposal]]] = {}
    for item in proposals:
        grouped.setdefault(item[1].object_local_id, []).append(item)
    objects = [
        _object_history(session, local_id, items, experiments, seeds, timeline.pts, labels.get(local_id, local_id), instances.get(local_id), shared, expand)
        for local_id, items in sorted(grouped.items())
    ]
    return {
        "run_id": run.id,
        "objects": objects,
        "note": "Plans only. Nothing here was sent, and any new SAM request needs its own dry run and approval.",
        "policy": {"search_window_sec": POLICY.search_window_sec, "max_clip_sec": POLICY.max_clip_sec},
        "price_per_frame_usd": settings.sam_price_per_frame_usd,
    }


def _video_path(video):
    try:
        return resolve_stored(video.stored_path)
    except (OSError, ValueError, TypeError):
        return None


def _hidden_times(document, local_id: str) -> list[float]:
    """Stored rows that say this object could not be seen."""
    return [
        row.timestamp
        for row in [*document.observations, *document.raw_observations]
        if row.object_local_id == local_id and row.visibility.value in {"occluded", "not_visible", "absent"}
    ]


def _search(session, failure, seeds, experiments, local_id, label, shared, frames, attempted, last_box, *, expand=False) -> dict:
    """Look through everything stored past the failure and judge each candidate. Nothing leaves this machine."""
    forward = failure["direction"] == "forward"
    window = search_window(failure["time"], frames, _hidden_times(shared["document"], local_id), POLICY, expand=expand, direction=failure["direction"])
    stored = sorted(
        (
            seed
            for seed in seeds
            if seed.object_local_id == local_id
            and seed.seed_id not in attempted
            and seed.source != "sam_spatial"
            and seed.trust != "rejected"
            and (seed.timestamp > failure["time"] if forward else seed.timestamp < failure["time"])
        ),
        key=lambda seed: (abs(seed.timestamp - failure["time"]), seed.seed_id),
    )[:MAX_CANDIDATES]
    cached, boxes_read, inspected = _cached_tracks(experiments, local_id, label, shared["labels"], window["start"], window["end"], seeds)
    pipeline, pipeline_read = _pipeline_tracks(session, shared["run_id"], local_id, window["start"], window["end"], seeds)
    candidates = [*_seed_candidates(stored, seeds), *cached, *pipeline]
    reference = {"timestamp": failure["last_good_time"], "bbox": last_box} if failure.get("last_good_time") is not None and last_box else None
    for item in candidates:
        anchored = any(
            abs(row["timestamp"] - item["timestamp"]) <= POLICY.same_frame_sec and row.get("source") != item.get("family")
            for row in shared["anchors"].get(local_id, [])
        )
        if not anchored and item.get("trust") != "human_verified":
            item["appearance"] = _appearance(shared["source"], reference, item)
    context = {
        "object_local_id": local_id,
        "failure_time": failure["time"],
        "last_good_time": failure.get("last_good_time"),
        "anchors": shared["anchors"],
        "boundaries": shared["boundaries"],
        "same_label": [other for other, name in shared["labels"].items() if name == label],
    }
    judged = verify_candidates(candidates, context, POLICY)
    return {
        "window": {key: window[key] for key in ("start", "end", "limited", "note")},
        "frames_examined": len(window["frames"]),
        "evidence_inspected": {
            "stored_observations": len(stored),
            "stored_sam_requests": inspected,
            "stored_sam_boxes": boxes_read,
            "pipeline_sam_points": pipeline_read,
        },
        "candidates": judged,
        "result": search_result(judged),
    }


def _workflow(failure, search, resolved, attempts, plan_blocked) -> list[dict]:
    """Where this loss stands. Planning runs on its own, and sending anything waits for a person."""
    ran = [item for item in attempts if item["status"] not in {"draft", "planned"}]
    concept_ran = any(item["strategy"] == "concept_redetection" for item in ran)
    spatial_ran = [item for item in ran if item["strategy"] != "concept_redetection"]
    result = search["result"]
    found = result == VERIFIED_REAPPEARANCE
    def stage(name, status, detail):
        return {"stage": name, "status": status, "detail": detail}
    stages = [
        stage("track_lost", "done", f"Tracking stopped being reliable at {failure['time']:.2f}s."),
        stage("cached_search", "done", f"{len(search['candidates'])} stored candidates in {search['frames_examined']} frames. No request was needed."),
        stage("identity", "done" if found else ("waiting_for_person" if result in {PROBABLE_REAPPEARANCE, AMBIGUOUS_CANDIDATE} else "not_found"), result.replace("_", " ")),
    ]
    if found or result in {PROBABLE_REAPPEARANCE, AMBIGUOUS_CANDIDATE}:
        stages.append(stage("concept_detection", "not_needed", "Stored evidence already shows a possible location."))
    elif concept_ran:
        stages.append(stage("concept_detection", "done", "The concept search already ran and found no usable detection."))
    elif search["window"]["limited"]:
        stages.append(stage("concept_detection", "blocked", search["window"]["note"]))
    else:
        stages.append(stage("concept_detection", "needs_approval", "A bounded concept search can be planned. It is a separate paid request."))
    if resolved is not None:
        stages.append(stage("reinitialization", "done", f"A new segment starts at {resolved['start']:.2f}s."))
        stages.append(stage("acceptance", "done" if resolved.get("decision") == "accepted" else "waiting_for_person", "Accepting creates a new revision."))
    elif spatial_ran:
        stages.append(stage("reinitialization", "failed", "An earlier attempt made no verified segment."))
    elif found:
        stages.append(stage("reinitialization", "needs_approval", "The new spatial request is ready to plan and needs your approval."))
    else:
        stages.append(stage("reinitialization", "blocked", plan_blocked or "It needs a verified or confirmed location first."))
    return stages


def plan_attempt(session: Session, run: AnalysisRun, payload: dict) -> RecoveryExperiment:
    """Turn a reviewed step into a dry-run experiment. It sends nothing, and sending needs the plan's own approval."""
    local_id = str(payload.get("object_local_id") or "")
    failure_id = str(payload.get("failure_id") or "")
    action = payload.get("action") or "reinitialize"
    history = reacquisition(session, run, expand={failure_id} if payload.get("expand_search") else None)
    item = next((plan for entry in history["objects"] if entry["object_local_id"] == local_id for plan in entry["plans"] if plan["failure"]["failure_id"] == failure_id), None)
    if item is None:
        raise RecoveryError("That track failure is not in this run's history. Nothing was planned.")
    attempts = [row for row in item["attempts"] if row["status"] not in {"draft", "planned"}]
    approved = bool(payload.get("approve_new_attempt"))
    document = _document(session, run.active_revision_id)
    known = {obj.local_id: obj for obj in document.objects}
    limits = _limits({**{"padding_sec": 0.0}, **(payload.get("limits") or {})})
    failure = item["failure"]
    record = {
        "trigger": f"{failure['reason']} at {failure['time']:.2f}s",
        "evidence_inspected": item["search"]["evidence_inspected"],
        "candidates_evaluated": [
            {key: row.get(key) for key in ("candidate_id", "source", "timestamp", "outcome", "reason")}
            for row in item["search"]["candidates"]
        ],
    }
    link = {"failure_id": failure_id, "failure_time": failure["time"], "experiment_id": failure["experiment_id"], "strategy": "", "record": record}
    if action == "concept":
        used = sum(1 for row in attempts if row["strategy"] == "concept_redetection")
        if used >= POLICY.max_redetection_attempts and not approved:
            raise RecoveryError(f"This loss already had {used} concept search. Approve a new attempt explicitly to plan another. Nothing was planned.")
        window = item["search"]["window"]
        if window["limited"]:
            raise RecoveryError(f"{window['note']} Nothing was planned.")
        phrase = concept_text(str(payload.get("concept") or known[local_id].label))
        target = _manual_target(
            {
                "object_local_id": local_id,
                "time_start": window["start"],
                "time_end": window["end"],
                "strategy": "concept",
                "concept": phrase,
                "reacquires": {**link, "strategy": "concept_redetection"},
            },
            known,
        )
    else:
        used = sum(1 for row in attempts if row["strategy"] != "concept_redetection")
        if used >= POLICY.max_reinitialization_attempts and not approved:
            raise RecoveryError(f"This loss already had {used} reinitialization attempts. Approve a new attempt explicitly to plan another. Nothing was planned.")
        candidate = next((row for row in item["search"]["candidates"] if row["candidate_id"] == payload.get("candidate_id")), None)
        if candidate is None:
            raise RecoveryError("That candidate is no longer in the search results. Nothing was planned.")
        confirmed = payload.get("confirm") is True
        if candidate["outcome"] == REJECTED_CANDIDATE:
            raise RecoveryError(f"This candidate was rejected. {candidate['reason']} Draw a box instead. Nothing was planned.")
        if candidate["outcome"] != VERIFIED_REAPPEARANCE and not confirmed:
            raise RecoveryError(f"This candidate needs a person's confirmation. {candidate['reason']} Nothing was planned.")
        link["strategy"] = "verified_reappearance" if candidate["outcome"] == VERIFIED_REAPPEARANCE else "confirmed_candidate"
        link["record"] = {**record, "seed_candidate": candidate["candidate_id"], "seed_outcome": candidate["outcome"], "confirmed_by_person": confirmed and candidate["outcome"] != VERIFIED_REAPPEARANCE}
        seed_time = candidate["timestamp"]
        direction = payload.get("direction") or "both"
        forward = POLICY.forward_after_seed_sec
        start = max(min(failure["time"], seed_time), seed_time - (POLICY.max_clip_sec - forward))
        end = min(seed_time + forward, start + POLICY.max_clip_sec)
        if direction == "backward":
            end = seed_time
        elif direction == "forward":
            start = seed_time
        manual = {"object_local_id": local_id, "time_start": start, "time_end": end, "direction": direction, "reacquires": link}
        if candidate["outcome"] == VERIFIED_REAPPEARANCE:
            target = _manual_target(manual, known)
            target["verified_seed"] = Seed(
                object_local_id=local_id,
                timestamp=seed_time,
                bbox=candidate["bbox"],
                trust=candidate["trust"] if candidate["trust"] == "human_verified" else INDEPENDENTLY_SUPPORTED,
                source="reappearance",
                instance_id=known[local_id].instance_id,
                source_ref=candidate["candidate_id"],
                checks=candidate["checks"],
                explanation=candidate["reason"],
            ).as_dict()
        else:
            target = _manual_target({**manual, "seed": {"timestamp": seed_time, "bbox": candidate["bbox"], "points": []}}, known)
    experiment = new_experiment(session, run, [target], limits)
    dry_run(session, experiment)
    return experiment


def _last_good_box(items, failure) -> dict | None:
    time = failure.get("last_good_time")
    if time is None:
        return None
    for _experiment, proposal in items:
        for row in proposal.rows_json or []:
            if row.get("request_key") == failure["request_key"] and abs(row["timestamp"] - time) <= 1e-3:
                return row["bbox"]
    return None


def _missing_from_revision(items, document) -> int:
    """Usable recovered boxes that the active revision does not hold."""
    stored = {(row.object_local_id, round(row.timestamp, 3)) for row in [*document.observations, *document.raw_observations]}
    return sum(
        1
        for _experiment, proposal in items
        for row in proposal.rows_json or []
        if not row.get("context") and row["outcome"] in {"supported", "provisional"}
        and (proposal.object_local_id, round(row["timestamp"], 3)) not in stored
    )


def _object_history(session, local_id, items, experiments, seeds, frames, label, instance_id, shared, expand=None) -> dict:
    segments: dict[tuple, dict] = {}
    failures: dict[str, dict] = {}
    usable: set[float] = set()
    returned: set[float] = set()
    spans_input: list[tuple[float, float]] = []
    raw_frames = associated = supported = provisional = conflicting = rejected = 0
    limits: set[str] = set()
    attempted_targets: dict[str, list[dict]] = {}
    keys: set[str] = set()
    ranges: dict[str, tuple[float, float]] = {}
    for experiment, proposal in items:
        track = (proposal.metrics_json or {}).get("track")
        if not track:
            limits.add("An older result has no segment data. Replay it for free to add it.")
            continue
        rows = proposal.rows_json or []
        spans_input.append((proposal.time_start, proposal.time_end))
        for row in rows:
            key = round(row["timestamp"], 3)
            returned.add(key)
            keys.add(row["request_key"])
            if row["outcome"] in {"supported", "provisional"} and proposal.decision != "rejected":
                usable.add(key)
        raw_frames += len(rows)
        associated += sum(1 for row in rows if row["association"] == "associated")
        counts = (proposal.metrics_json or {}).get("counts", {})
        supported += counts.get("supported", 0)
        provisional += counts.get("provisional", 0)
        conflicting += counts.get("conflicting", 0)
        rejected += counts.get("rejected", 0)
        limits.update(track.get("evidence_limits") or [])
        for segment in track["segments"]:
            identity = (segment["request_key"], segment["raw_track_id"], segment["seed"]["seed_id"], segment["direction"], segment["start"], segment["end"])
            segments.setdefault(identity, {**segment, "decision": proposal.decision, "proposal_id": proposal.id})
        for failure in track["failures"]:
            if failure["failure_id"] not in failures:
                failures[failure["failure_id"]] = {**failure, "decision": proposal.decision}
                last_row = max((row["timestamp"] for row in rows), default=proposal.time_end)
                first_row = min((row["timestamp"] for row in rows), default=proposal.time_start)
                ranges[failure["failure_id"]] = (min(proposal.time_start, first_row), max(proposal.time_end, last_row))
    ordered_segments = sorted(segments.values(), key=lambda item: item["start"])
    frames_used = sorted({time for start, end in spans_input for time in frames if start - 1e-6 <= time <= end + 1e-6})
    spans = unresolved_spans(frames_used, usable, returned, min((s for s, _ in spans_input), default=0.0), max((e for _, e in spans_input), default=0.0))
    # Every run attempt counts, including one that returned nothing and so has no segment to show.
    for experiment in experiments:
        for target in experiment.targets_json or []:
            reacquires = target.get("reacquires")
            if not reacquires or target["object_local_id"] != local_id or experiment.status in {"draft", "planned", "running"}:
                continue
            proposal = next((row for owner, row in items if owner.id == experiment.id and row.target_id == target["target_id"]), None)
            own = [segment for segment in ordered_segments if proposal is not None and segment.get("proposal_id") == proposal.id]
            specs = (experiment.plan_json or {}).get("requests") or []
            attempted_targets.setdefault(reacquires["failure_id"], []).append(
                {
                    "experiment_id": experiment.id,
                    "status": experiment.status,
                    "decision": proposal.decision if proposal is not None else "",
                    "seed_ids": sorted(
                        {
                            entry["seed"]["seed_id"]
                            for spec in specs
                            for entry in spec["objects"]
                            if entry["target_id"] == target["target_id"]
                        }
                    ),
                    "strategy": reacquires.get("strategy", ""),
                    "record": reacquires.get("record") or {},
                    "requests": [
                        {
                            "prompt": spec["prompt"],
                            "prompt_kind": spec.get("prompt_kind", "spatial"),
                            "direction": spec["direction"],
                            "source_start": spec["source_start"],
                            "source_end": spec["source_end"],
                            "frames": len(spec["frame_times"]),
                            "billed_frames": spec["billed_frames"],
                            "cache_key": spec["key"],
                            "cached": bool(spec.get("cached")),
                            "provider_request_id": (load_response(experiment.run_id, spec["key"]) or {}).get("request_id", ""),
                        }
                        for spec in specs
                        if spec.get("target_id") == target["target_id"]
                        or any(entry["target_id"] == target["target_id"] for entry in spec["objects"])
                    ],
                    "segments": [segment["segment_id"] for segment in own],
                    "association_established": any(segment["identity"]["status"] == "linked" for segment in own),
                    "succeeded": any(segment["status"] == REACQUIRED and segment["decision"] != "rejected" for segment in own),
                }
            )
    plans = []
    for failure in sorted(failures.values(), key=lambda item: item["time"]):
        start, end = ranges[failure["failure_id"]]
        if failure["direction"] == "forward":
            interval_start, interval_end = failure["time"], max(end, failure["time"])
        else:
            interval_start, interval_end = min(start, failure["time"]), failure["time"]
        attempts = attempted_targets.get(failure["failure_id"], [])
        resolved = next((segment for segment in ordered_segments if _usable_after(failure, segment)), None)
        attempted = {seed for attempt in attempts if not attempt["succeeded"] for seed in attempt["seed_ids"]}
        last_box = _last_good_box(items, failure)
        search = _search(session, failure, seeds, experiments, local_id, label, shared, frames, attempted, last_box, expand=failure["failure_id"] in (expand or set()))
        cached = [row for row in search["candidates"] if not row["seed_id"] and row["outcome"] != REJECTED_CANDIDATE]
        verified = [row for row in search["candidates"] if row["outcome"] == VERIFIED_REAPPEARANCE and not row["seed_id"]]
        concept_runs = sum(1 for attempt in attempts if attempt["status"] not in {"draft", "planned"} and attempt["strategy"] == "concept_redetection")
        concept_allowed = concept_runs < POLICY.max_redetection_attempts and not search["window"]["limited"]
        strategies = _strategies(failure, seeds, cached, label, attempted, search["window"]["start"], search["window"]["end"], verified, concept_allowed, {row["seed_id"] for row in search["candidates"] if row["outcome"] == REJECTED_CANDIDATE and row["seed_id"]})
        recommended = next(item for item in strategies if item["available"])
        evidence = recommended["evidence"][0] if recommended["evidence"] else None
        if evidence is not None and not evidence["seed_id"]:
            evidence = None
        checks = [item["timestamp"] for item in recommended["evidence"]]
        if not checks:
            after = [time for time in frames if interval_start < time <= interval_end + SEARCH_AFTER_SEC]
            stride = max(len(after) // MAX_FRAMES_TO_CHECK, 1)
            checks = after[::stride][:MAX_FRAMES_TO_CHECK]
        raw_after = sum(
            1
            for _experiment, proposal in items
            for row in proposal.rows_json or []
            if row.get("lifecycle") == "lost" and row.get("request_key") == failure["request_key"]
        )
        window_spans = [item for item in spans if item["end"] >= interval_start and item["start"] <= interval_end]
        rows_after = [
            row
            for _experiment, proposal in items
            for row in proposal.rows_json or []
            if row.get("request_key") == failure["request_key"]
            and (row["timestamp"] >= failure["time"] if failure["direction"] == "forward" else row["timestamp"] <= failure["time"])
        ]
        frames_after = [time for time in frames_used if (time >= failure["time"] if failure["direction"] == "forward" else time <= failure["time"])]
        blocked_reason = "" if recommended["strategy"] in {"verified_reappearance", "independent_observation", "human_seed"} else "No trustworthy evidence locates the object after this failure. A person has to identify it first."
        analyzed = max((end_ for _start, end_ in spans_input), default=failure["time"])
        limit_note = search["window"]["note"]
        plans.append(
            {
                "failure": failure,
                "status": "resolved" if resolved else "open",
                "uncertain_interval": {"start": interval_start, "end": resolved["start"] if resolved and resolved["start"] > interval_start else interval_end},
                "last_supported_time": failure["last_supported_time"],
                "reappearance_frames": checks,
                "search": search,
                "diagnosis": diagnose_failure(
                    failure,
                    rows_after=rows_after,
                    frames_after=frames_after,
                    candidates=search["candidates"],
                    other_tracks=[row for row in search["candidates"] if row.get("raw_track_id")],
                    anchors=shared["anchors"].get(local_id, []),
                    clip_end=analyzed,
                    video_end=shared["video_end"],
                    missing_accepted=_missing_from_revision(items, shared["document"]),
                    planned=bool(attempts),
                    executed=any(row["status"] not in {"draft", "planned"} for row in attempts),
                ),
                "workflow": _workflow(failure, search, resolved, attempts, blocked_reason),
                "limits": {
                    "max_redetection_attempts": POLICY.max_redetection_attempts,
                    "max_reinitialization_attempts": POLICY.max_reinitialization_attempts,
                    "max_clip_sec": POLICY.max_clip_sec,
                    "search_window_sec": POLICY.search_window_sec,
                    "max_requests": DEFAULT_LIMITS["max_requests"],
                    "max_frames_per_request": DEFAULT_LIMITS["max_frames_per_request"],
                    "max_total_billed_frames": DEFAULT_LIMITS["max_total_billed_frames"],
                    "note": limit_note,
                },
                "strategies": strategies,
                "recommended": recommended["strategy"],
                "proposed_target": _proposed(failure, evidence, recommended["strategy"], interval_start, interval_end),
                # Only a person may start this. A candidate also has to be confirmed.
                "automatic_start_allowed": False,
                "blocked_reason": blocked_reason,
                "attempts": attempts,
                "needs_stronger_evidence": bool(attempts) and not any(item["succeeded"] for item in attempts),
                "explanation": _explanation(failure, raw_after, recommended, [a for a in attempts if not a["succeeded"]], resolved, window_spans),
            }
        )
    requests = list(
        session.scalars(
            select(RecoveryRequest).where(
                RecoveryRequest.experiment_id.in_([item[0].id for item in items]), RecoveryRequest.cache_key.in_(keys or {""})
            )
        )
    )
    live = [row for row in requests if row.source == "live"]
    costs = [row.cost_usd for row in live if row.cost_usd is not None]
    every_attempt = [attempt for group in attempted_targets.values() for attempt in group]
    return {
        "object_local_id": local_id,
        "physical_instance_id": instance_id or local_id,
        "label": label,
        "segments": ordered_segments,
        "failures": sorted(failures.values(), key=lambda item: item["time"]),
        "unresolved": spans,
        "plans": plans,
        "stats": {
            "raw_frames_returned": raw_frames,
            "associated": associated,
            "supported": supported,
            "provisional": provisional,
            "conflicting": conflicting,
            "rejected": rejected,
            "no_detection_frames": sum(item["no_detection_frames"] for item in spans),
            "unusable_frames": sum(item["unusable_frames"] for item in spans),
            "track_breaks": len(failures),
            "reacquisition_attempts": len(every_attempt),
            "reacquisitions_succeeded": sum(1 for item in every_attempt if item["succeeded"]),
            "reacquisitions_failed": sum(1 for item in every_attempt if not item["succeeded"]),
            "identity_ambiguities": sum(1 for item in ordered_segments if item["identity"]["status"] in {"ambiguous", "rejected"}),
            "api_requests": len(live),
            "recorded_cost_usd": round(sum(costs), 6) if costs else None,
        },
        "evidence_limits": sorted(limits),
        "coverage": {
            "analyzed_through": max((end for _start, end in spans_input), default=0.0),
            "video_end": shared["video_end"],
        },
    }
