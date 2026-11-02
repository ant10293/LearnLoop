"""Targeted object recovery on a finished analysis.

An experiment reads the run's active revision and writes only its own rows and
its own folder. It has three modes: a dry run plans, a replay re-reads stored SAM
responses, and a live run sends the planned SAM requests after a person approves
that exact plan. No mode can reach a language model, a pipeline stage, or the job
queue. Results stay proposals until a person accepts them. Accepting forks a new
revision through the existing correction path.
"""

from __future__ import annotations

import asyncio
import bisect
import dataclasses
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.domain import SceneChunk
from app.models.media import VideoAsset
from app.models.recovery import (
    RecoveryDismissal,
    RecoveryExperiment,
    RecoveryProposal,
    RecoveryRequest,
)
from app.models.runs import AnalysisRevision, AnalysisRun
from app.models.util import utcnow
from app.perception.recovery_targets import find_targets
from app.perception.recovery_validate import (
    CONFLICTING,
    PROVISIONAL,
    REJECTED,
    SUPPORTED,
    ValidationPolicy,
    combine_passes,
    remaining_gaps,
    validate_track,
)
from app.perception.resolve import is_estimate, resolve_document
from app.perception.sam_fusion import SamRequestPlan, associate_tracks
from app.perception.seeds import REJECTED as SEED_REJECTED
from app.perception.seeds import (
    Seed,
    collect_seeds,
    manual_seed,
    row_family,
    seed_from_dict,
    seeds_around,
)
from app.perception.spatial_prompts import (
    SAME_FRAME_SEC,
    MediaMap,
    SeedMappingError,
    billed_frames,
    build_spatial_plan,
    choose_direction,
)
from app.perception.track_segments import track_report
from app.pipeline.perception_stage import load_spatial_document
from app.pipeline.sam_stage import _encode, _samples_from_result
from app.pipeline.sampling_stage import timeline_for
from app.providers.sam import (
    SamAdapter,
    VideoChunk,
    concept_text,
    decode_mask_raster,
    parse_sam_output,
    sam_video_accepted,
)
from app.providers.types import ProviderError, ProviderResponse, Usage
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    ObservationPrediction,
)
from app.schemas.enums import AmbiguityKind, Continuity, EvidenceClass, Provenance, Visibility
from app.services.corrections import revise
from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
from app.services.credentials import load_secret
from app.services.storage import data_path, relative, resolve_stored
from app.video.frames import extract_frame, read_bgr
from app.video.probe import VideoError, presentation_timestamps, probe

MODEL_ID = "sam-3.1"
# Part of each cache key. Raise it when the clip encoding changes.
MEDIA_VERSION = "recovery-clip-1"
REQUEST_PARAMETERS = {"mask_encoding": "one_bit", "include_confidence": "true"}
STAGE = "targeted_recovery"
DEFAULT_LIMITS = {
    "max_requests": 2,
    "max_frames_per_request": 300,
    "max_total_billed_frames": 600,
    "cost_budget_usd": 1.0,
    "timeout_s": 300.0,
    "max_upload_retries": 1,
    "padding_sec": 1.0,
    "two_sided": True,
    "group_targets": True,
}
# A manual seed is drawn on a paused video, which can sit one frame off. An automatic one cannot.
MANUAL_TOLERANCE_FRAMES = 1.5
CONCEPT_DIRECTION = "concept"
_CLAIM_LOCK = threading.Lock()
_APPLY_LOCK = threading.Lock()


class RecoveryError(ValueError):
    """A refused action. `remote_submitted` says whether a provider received a request."""

    def __init__(self, message: str, *, remote_submitted: bool = False):
        super().__init__(message)
        self.remote_submitted = remote_submitted


class RecoveryConflict(RecoveryError):
    pass


def _adapter(session: Session, timeout_s: float) -> SamAdapter:
    secret, _source = load_secret(session, "meta")
    if not secret:
        raise RecoveryError("No Meta Model API key is configured. Nothing was submitted.")
    return SamAdapter(secret, timeout_s=timeout_s)


# ---- Reading the analysis ----


def _experiment_dir(run_id: str, experiment_id: str) -> Path:
    return data_path("runs", run_id, "recovery", experiment_id)


def _cache_path(run_id: str, key: str) -> Path:
    return data_path("runs", run_id, "recovery", "responses", f"{key}.json")


def _usable_spatial(run_id: str) -> AnalysisDocument | None:
    spatial = load_spatial_document(run_id)
    if spatial is None or spatial.perception is None or not spatial.perception.used_for_identity:
        return None
    return spatial


def _boundaries(session: Session, run_id: str) -> list[float]:
    rows = session.scalars(select(SceneChunk).where(SceneChunk.run_id == run_id)).all()
    return [row.time_start for row in rows if row.scene_change and row.time_start > 0]


def _document(session: Session, revision_id: str | None) -> AnalysisDocument:
    revision = session.get(AnalysisRevision, revision_id) if revision_id else None
    if revision is None:
        raise RecoveryError("This run has no analysis revision to recover from. Nothing was submitted.")
    return AnalysisDocument.model_validate(revision.document_json)


def list_candidates(session: Session, run: AnalysisRun) -> dict:
    document = _document(session, run.active_revision_id)
    spatial = _usable_spatial(run.id)
    seeds = collect_seeds(document, spatial, boundaries=_boundaries(session, run.id))
    dismissed = set(session.scalars(select(RecoveryDismissal.target_id).where(RecoveryDismissal.run_id == run.id)))
    targets = [
        {**item, "dismissed": item["target_id"] in dismissed} for item in find_targets(document, spatial, seeds)
    ]
    return {
        "run_id": run.id,
        "revision_id": run.active_revision_id,
        "targets": targets,
        "seeds": _listed_seeds(seeds),
        "seed_counts": _seed_counts(seeds),
        "objects": [
            {"local_id": obj.local_id, "label": obj.label, "instance_id": obj.instance_id}
            for obj in document.objects
        ],
        "price_per_frame_usd": settings.sam_price_per_frame_usd,
        "default_limits": DEFAULT_LIMITS,
    }


def dismiss_target(session: Session, run: AnalysisRun, target_id: str) -> dict:
    """Hide a suspected problem on this run. No observation or revision changes."""
    current = {item["target_id"]: item for item in list_candidates(session, run)["targets"]}
    target = current.get(target_id)
    if target is None:
        raise RecoveryError(f"Unknown suspected problem {target_id}.")
    if not target["dismissed"]:
        session.add(
            RecoveryDismissal(
                run_id=run.id,
                target_id=target_id,
                object_local_id=target["object_local_id"],
                time_start=target["time_start"],
                time_end=target["time_end"],
                reasons_json=target["reasons"],
            )
        )
        session.commit()
    return list_candidates(session, run)


def restore_target(session: Session, run: AnalysisRun, target_id: str) -> dict:
    row = session.scalar(
        select(RecoveryDismissal).where(RecoveryDismissal.run_id == run.id, RecoveryDismissal.target_id == target_id)
    )
    if row is not None:
        session.delete(row)
        session.commit()
    return list_candidates(session, run)


def _listed_seeds(seeds: list[Seed], per_object: int = 60) -> list[dict]:
    """Enough choices for every object. One densely tracked object does not crowd out the others."""
    listed: list[dict] = []
    counts: dict[str, int] = {}
    for seed in seeds:
        if seed.trust == SEED_REJECTED or counts.get(seed.object_local_id, 0) >= per_object:
            continue
        counts[seed.object_local_id] = counts.get(seed.object_local_id, 0) + 1
        listed.append(seed.as_dict())
    return listed


def _seed_counts(seeds: list[Seed]) -> dict:
    counts: dict[str, int] = {}
    for seed in seeds:
        counts[seed.trust] = counts.get(seed.trust, 0) + 1
    return counts


# ---- Creating and changing an experiment ----


def create_experiment(session: Session, run: AnalysisRun, payload: dict) -> RecoveryExperiment:
    """Store what was asked for. Creating an experiment never runs it."""
    document = _document(session, run.active_revision_id)
    spatial = _usable_spatial(run.id)
    seeds = collect_seeds(document, spatial, boundaries=_boundaries(session, run.id))
    automatic = {item["target_id"]: item for item in find_targets(document, spatial, seeds)}
    known = {obj.local_id: obj for obj in document.objects}
    targets: list[dict] = []
    for item in payload.get("target_ids") or []:
        if item not in automatic:
            raise RecoveryError(f"Unknown recovery candidate {item}.")
        targets.append({**automatic[item], "direction": "auto"})
    for item in payload.get("manual_targets") or []:
        targets.append(_manual_target(item, known))
    if not targets:
        raise RecoveryError("Choose at least one recovery candidate or add a manual target.")
    return new_experiment(session, run, targets, _limits(payload.get("limits")))


def new_experiment(session: Session, run: AnalysisRun, targets: list[dict], limits: dict) -> RecoveryExperiment:
    experiment = RecoveryExperiment(
        run_id=run.id,
        source_revision_id=run.active_revision_id,
        status="draft",
        targets_json=targets,
        limits_json=limits,
    )
    session.add(experiment)
    session.commit()
    return experiment


def _limits(raw: dict | None) -> dict:
    limits = dict(DEFAULT_LIMITS)
    for key, value in (raw or {}).items():
        if key not in DEFAULT_LIMITS:
            raise RecoveryError(f"Unknown recovery limit {key}.")
        if isinstance(DEFAULT_LIMITS[key], bool):
            limits[key] = bool(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RecoveryError(f"The limit {key} must be a number.") from exc
        if number < 0:
            raise RecoveryError(f"The limit {key} cannot be negative.")
        limits[key] = int(number) if isinstance(DEFAULT_LIMITS[key], int) else number
    if limits["padding_sec"] > 10:
        raise RecoveryError("Padding is limited to 10 seconds on each side.")
    return limits


def _manual_target(item: dict, known: dict) -> dict:
    local_id = str(item.get("object_local_id") or "")
    if local_id not in known:
        raise RecoveryError(f"Unknown object {local_id}.")
    try:
        start, end = float(item["time_start"]), float(item["time_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryError("A manual target needs time_start and time_end.") from exc
    if end < start:
        raise RecoveryError("A manual target must end after it starts.")
    obj = known[local_id]
    target = {
        "target_id": hashlib.sha256(f"manual|{local_id}|{start:.3f}|{end:.3f}".encode()).hexdigest()[:16],
        "object_local_id": local_id,
        "instance_id": obj.instance_id,
        "label": obj.label,
        "time_start": start,
        "time_end": end,
        "source": "manual",
        "state": "manual",
        "reasons": ["researcher_selected"],
        "details": [],
        "direction": "auto",
    }
    reacquires = item.get("reacquires")
    if reacquires:
        target["reacquires"] = {
            "failure_id": str(reacquires.get("failure_id") or ""),
            "failure_time": float(reacquires.get("failure_time") or start),
            "experiment_id": str(reacquires.get("experiment_id") or ""),
            "strategy": str(reacquires.get("strategy") or "manual"),
            "record": reacquires.get("record") or {},
        }
        target["reasons"] = ["reacquisition"]
    if item.get("strategy") == "concept":
        # Finds possible locations only. Nothing here says which physical part any of them is.
        target["strategy"] = "concept"
        try:
            target["concept"] = concept_text(str(item.get("concept") or obj.label))
        except ProviderError as exc:
            raise RecoveryError(str(exc)) from exc
        return target
    return _apply_seed_choice(target, item)


def _apply_seed_choice(target: dict, item: dict) -> dict:
    direction = item.get("direction") or target.get("direction") or "auto"
    if direction not in {"auto", "forward", "backward", "both"}:
        raise RecoveryError(f"Unknown propagation direction {direction}.")
    updated = {**target, "direction": direction}
    if item.get("seed"):
        seed = dict(item["seed"])
        seed["object_local_id"] = target["object_local_id"]
        _check_manual_seed(seed)
        updated["seed"] = seed
        updated.pop("seed_id", None)
    elif item.get("seed_ids"):
        updated["seed_ids"] = [str(value) for value in item["seed_ids"]]
        updated.pop("seed", None)
        updated.pop("seed_id", None)
    elif item.get("seed_id"):
        updated["seed_id"] = str(item["seed_id"])
        updated["confirm_seed"] = bool(item.get("confirm_seed"))
        updated.pop("seed", None)
    return updated


def _check_manual_seed(seed: dict) -> None:
    if seed.get("timestamp") is None:
        raise RecoveryError("A manual seed needs the frame timestamp it was drawn on.")
    points = seed.get("points") or []
    if seed.get("bbox"):
        BBox.model_validate(seed["bbox"])
    elif not any(not point.get("negative") for point in points):
        raise RecoveryError("A manual seed needs a box or at least one positive point.")


def update_seeds(session: Session, experiment: RecoveryExperiment, payload: dict) -> RecoveryExperiment:
    """Change a target's seed or direction. The old plan no longer applies, so it is cleared."""
    _not_running(experiment)
    changes = {item.get("target_id"): item for item in payload.get("targets") or []}
    targets = []
    for target in experiment.targets_json or []:
        change = changes.pop(target["target_id"], None)
        targets.append(_apply_seed_choice(target, change) if change else target)
    if changes:
        raise RecoveryError(f"Unknown target {next(iter(changes))}.")
    experiment.targets_json = targets
    experiment.plan_json = {}
    experiment.status = "draft"
    session.commit()
    return experiment


def _not_running(experiment: RecoveryExperiment) -> None:
    if experiment.status == "running":
        raise RecoveryConflict("This recovery is running. Cancel it first.")


# ---- Planning ----


def _chosen_passes(target: dict, seeds: list[Seed], limits: dict) -> tuple[list[tuple[Seed, str]], str]:
    """The seed and direction for each tracking pass, or the reason there are none."""
    start, end = float(target["time_start"]), float(target["time_end"])
    requested = target.get("direction") or "auto"
    if target.get("verified_seed"):
        # Checked when the plan was made. It keeps its own trust, so its evidence stays readable.
        seed = seed_from_dict(target["verified_seed"])
        return [(seed, _direction(requested, seed, start, end))], ""
    if target.get("seed"):
        seed = manual_seed(target["seed"], target.get("instance_id"))
        return [(seed, _direction(requested, seed, start, end))], ""
    if target.get("seed_ids"):
        return _multi_frame_passes(target, seeds, requested, start, end)
    if target.get("seed_id"):
        seed = next((item for item in seeds if item.seed_id == target["seed_id"]), None)
        if seed is None or seed.object_local_id != target["object_local_id"]:
            return [], "The selected seed no longer exists for this object."
        if seed.trust == SEED_REJECTED:
            return [], f"The selected seed was rejected. {seed.explanation}"
        if not seed.automatic and not target.get("confirm_seed"):
            return [], f"The selected seed is a candidate and needs a person's confirmation. {seed.explanation}"
        if not seed.automatic:
            seed = seed_from_dict({**seed.as_dict(), "trust": "human_verified"})
            seed.explanation = f"A person confirmed this {seed.source} box."
            # The person saw it on the displayed frame, so it has the same frame allowance as a drawn seed.
            seed.manual = True
        return [(seed, _direction(requested, seed, start, end))], ""
    around = seeds_around(seeds, target["object_local_id"], start, end)
    if around["best"] is None:
        return [], "No trustworthy seed exists for this object. Confirm a candidate box or draw one."
    if requested == "auto" and limits["two_sided"] and around["before"] and around["after"]:
        return [(around["before"], "forward"), (around["after"], "backward")], ""
    seed = around["best"]
    return [(seed, _direction(requested, seed, start, end))], ""


def _multi_frame_passes(target: dict, seeds: list[Seed], requested: str, start: float, end: float) -> tuple[list[tuple[Seed, str]], str]:
    """Several trusted prompts for one object, sent in one request and tracked from the earliest."""
    known = {item.seed_id: item for item in seeds if item.object_local_id == target["object_local_id"]}
    chosen = [known.get(item) for item in target["seed_ids"]]
    if any(item is None for item in chosen):
        return [], "A selected seed no longer exists for this object."
    if any(not item.automatic for item in chosen):
        return [], "Every prompt in a multi-frame request has to be a trusted seed."
    chosen.sort(key=lambda item: item.timestamp)
    direction = _direction(requested, chosen[0], start, end)
    if direction == "backward":
        return [], "Backward tracking starts at the earliest prompt. Send the later seed in its own request."
    return [(seed, direction) for seed in chosen], ""


def _direction(requested: str, seed: Seed, start: float, end: float) -> str:
    return choose_direction(seed.timestamp, start, end) if requested == "auto" else requested


def _clip_bounds(seed_time: float, start: float, end: float, direction: str, padding: float) -> tuple[float, float]:
    """Forward needs nothing before its seed and backward nothing after it. Those frames add no tracking."""
    if direction == "forward":
        return min(seed_time, start), max(seed_time, end + padding)
    if direction == "backward":
        return min(seed_time, start - padding), max(seed_time, end)
    return min(seed_time, start - padding), max(seed_time, end + padding)


def _clip_frames(times: list[float], low: float, high: float) -> list[float]:
    return [time for time in times if low - 1e-6 <= time <= high + 1e-6]


def cache_key(checksum: str, frame_times: list[float], width: int, height: int, prompt: str, direction: str) -> str:
    """Everything that decides SAM's answer. A changed input can never reuse a stored response."""
    times = hashlib.sha256(json.dumps([round(item, 6) for item in frame_times]).encode()).hexdigest()
    identity = [checksum, times, width, height, MEDIA_VERSION, prompt, direction, MODEL_ID, REQUEST_PARAMETERS]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _tolerance(times: list[float], seed: Seed) -> float | None:
    if not seed.manual or len(times) < 2:
        return None
    return MANUAL_TOLERANCE_FRAMES * max((times[-1] - times[0]) / (len(times) - 1), 1e-3)


def _pass_entry(target: dict, seed: Seed, direction: str, times: list[float], limits: dict) -> dict:
    start, end = float(target["time_start"]), float(target["time_end"])
    tolerance = _tolerance(times, seed)
    limit = int(limits["max_frames_per_request"])
    for padding in (float(limits["padding_sec"]), 0.0):
        low, high = _clip_bounds(seed.timestamp, start, end, direction, padding)
        frames = _clip_frames(times, low, high)
        if frames and len(frames) <= limit:
            return {"target": target, "seed": seed, "direction": direction, "frames": frames, "tolerance": tolerance}
    if not frames:
        raise SeedMappingError("No source frames fall inside the planned interval.")
    raise SeedMappingError(
        f"This interval and seed need {len(frames)} frames, and the limit per request is {limit}. "
        "Choose a nearer seed, shorten the interval, or raise the limit."
    )


def _groupable(group: dict, entry: dict, limits: dict) -> bool:
    if not limits["group_targets"] or group["direction"] != entry["direction"]:
        return False
    # Backward tracking starts at the earliest prompt, so seeds on different frames need separate requests.
    if entry["direction"] == "backward" and any(
        abs(item["seed"].timestamp - entry["seed"].timestamp) > SAME_FRAME_SEC for item in group["entries"]
    ):
        return False
    objects = {item["seed"].object_local_id for item in group["entries"]}
    same_seed = any(item["seed"].seed_id == entry["seed"].seed_id for item in group["entries"])
    # One object may be prompted on several frames, but only once on each frame.
    if not same_seed and any(
        item["seed"].object_local_id == entry["seed"].object_local_id
        and abs(item["seed"].timestamp - entry["seed"].timestamp) <= 0.05
        for item in group["entries"]
    ):
        return False
    if len(objects | {entry["seed"].object_local_id}) > 16:
        return False
    if group["frames"][0] > entry["frames"][-1] or entry["frames"][0] > group["frames"][-1]:
        return False
    union = sorted(set(group["frames"]) | set(entry["frames"]))
    if len(union) > int(limits["max_frames_per_request"]):
        return False
    separate = _billed(group["frames"], group["entries"], group["direction"]) + _billed(
        entry["frames"], [entry], entry["direction"]
    )
    return _billed(union, [*group["entries"], entry], group["direction"]) <= separate or same_seed


def _check_contiguous(frames: list[float], source: list[float]) -> None:
    """The clip is one unbroken run of source frames, so the plan must be too."""
    start = bisect.bisect_left(source, frames[0] - 1e-6)
    if source[start : start + len(frames)] != frames:
        raise SeedMappingError("The grouped frames skip source frames, so they would not match the clip. Send these targets separately.")


def _billed(frames: list[float], entries: list[dict], direction: str) -> int:
    seeds = [item["seed"].timestamp for item in entries]
    index = min(min(range(len(frames)), key=lambda at: abs(frames[at] - time)) for time in seeds)
    return billed_frames(index, len(frames), direction)


def build_plan(session: Session, experiment: RecoveryExperiment) -> dict:
    """The dry run. It reads stored data only and contacts no provider."""
    run = session.get(AnalysisRun, experiment.run_id)
    video = session.get(VideoAsset, run.video_id)
    document = _document(session, experiment.source_revision_id)
    limits = experiment.limits_json or DEFAULT_LIMITS
    timeline = timeline_for(video, None)
    seeds = collect_seeds(document, _usable_spatial(run.id), boundaries=_boundaries(session, run.id))
    blocked: list[dict] = []
    notes = list(timeline.notes)
    groups: list[dict] = []
    width, height = int(video.display_width or 0), int(video.display_height or 0)
    seen: set[tuple] = set()
    concept_targets: list[dict] = []
    for target in experiment.targets_json or []:
        if target.get("strategy") == "concept":
            concept_targets.append(target)
            continue
        passes, reason = _chosen_passes(target, seeds, limits)
        if not passes:
            blocked.append(_blocked(target, reason))
            continue
        if timeline.basis != "presentation":
            blocked.append(_blocked(target, "Frame times are estimates for this video, so a seed cannot be placed exactly."))
            continue
        for seed, direction in passes:
            key = (target["object_local_id"], seed.seed_id, direction, target["time_start"], target["time_end"])
            if key in seen:
                continue
            seen.add(key)
            try:
                entry = _pass_entry(target, seed, direction, timeline.pts, limits)
                probe_media = MediaMap(width, height, entry["frames"])
                build_spatial_plan([(1, seed)], probe_media, direction, "check", entry["tolerance"])
            except (SeedMappingError, ValueError) as exc:
                blocked.append(_blocked(target, str(exc)))
                continue
            group = next((item for item in groups if _groupable(item, entry, limits)), None)
            if group is None:
                groups.append({"direction": direction, "frames": list(entry["frames"]), "entries": [entry]})
            else:
                group["frames"] = sorted(set(group["frames"]) | set(entry["frames"]))
                group["entries"].append(entry)
    requests: list[dict] = []
    prefix = experiment.id.replace("-", "")[:8]
    for index, group in enumerate(groups):
        try:
            _check_contiguous(group["frames"], timeline.pts)
            requests.append(_request_spec(run, video, group, f"rec{prefix}r{index}", width, height))
        except (SeedMappingError, ValueError) as exc:
            blocked.extend(_blocked(item["target"], str(exc)) for item in group["entries"])
    for target in concept_targets:
        frames = _clip_frames(timeline.pts, float(target["time_start"]), float(target["time_end"]))
        if timeline.basis != "presentation":
            blocked.append(_blocked(target, "Frame times are estimates for this video, so results could not be placed exactly."))
        elif not frames:
            blocked.append(_blocked(target, "No source frames fall inside the planned interval."))
        elif len(frames) > int(limits["max_frames_per_request"]):
            blocked.append(_blocked(target, f"This interval has {len(frames)} frames, and the limit per request is {limits['max_frames_per_request']}."))
        else:
            requests.append(_concept_spec(video, target, frames, f"rec{prefix}r{len(requests)}", width, height))
    price = settings.sam_price_per_frame_usd
    for spec in requests:
        spec["cached"] = _cache_path(run.id, spec["key"]).exists()
        spec["estimated_cost_usd"] = round(spec["billed_frames"] * price, 6) if price is not None else None
    remote = [spec for spec in requests if not spec["cached"]]
    billed = sum(spec["billed_frames"] for spec in remote)
    totals = {
        "requests": len(requests),
        "remote_requests": len(remote),
        "cached_requests": len(requests) - len(remote),
        "frames": sum(len(spec["frame_times"]) for spec in requests),
        "billed_frames": billed,
        "estimated_cost_usd": round(billed * price, 6) if price is not None else None,
        "cost_note": _cost_note(price),
    }
    plan = {"requests": requests, "blocked": blocked, "totals": totals, "notes": notes, "limits": limits}
    plan["violations"] = preflight_violations(plan, limits, spent_frames=0, spent_requests=0)
    plan["plan_hash"] = hashlib.sha256(
        json.dumps(
            [[spec["key"], spec["billed_frames"]] for spec in requests], sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    return plan


def _cost_note(price: float | None) -> str:
    if price is None:
        return (
            "SAM is billed per frame with a minimum of 50 per request. No per-frame rate is configured, so the cost "
            "is unknown and the cost budget cannot be checked. Request and frame limits still apply. Set "
            "LEARNLOOP_SAM_PRICE_PER_FRAME_USD from Meta's pricing page to see a cost."
        )
    return (
        f"Estimate at the configured ${price} per frame, using the documented minimum of 50 frames per request. "
        "Meta's invoice decides the real charge. File uploads and any tracking past the planned frames are not included."
    )


def _blocked(target: dict, reason: str) -> dict:
    return {
        "target_id": target["target_id"],
        "object_local_id": target["object_local_id"],
        "time_start": target["time_start"],
        "time_end": target["time_end"],
        "reason": reason,
    }


def _prompt_role(target: dict, seed: Seed, seen: set[str]) -> str:
    """How the Recovery page labels this prompt. Records what was sent and changes nothing about it."""
    first = seed.object_local_id not in seen
    seen.add(seed.object_local_id)
    if target.get("reacquires"):
        return "manual_seed" if seed.manual else "reappearance_seed"
    if seed.manual:
        return "manual_seed"
    return "initial_seed" if first else "additional_prompt"


def _request_spec(run, video, group: dict, session_id: str, width: int, height: int) -> dict:
    frames = group["frames"]
    media = MediaMap(width, height, frames)
    ids: dict[str, int] = {}
    pairs: list[tuple[int, Seed]] = []
    objects: list[dict] = []
    tolerance = max((item["tolerance"] or 0.0 for item in group["entries"]), default=0.0) or None
    used: set[str] = set()
    seen_objects: set[str] = set()
    for entry in sorted(group["entries"], key=lambda item: item["seed"].timestamp):
        seed = entry["seed"]
        object_id = ids.setdefault(seed.object_local_id, len(ids) + 1)
        if seed.seed_id not in used:
            used.add(seed.seed_id)
            pairs.append((object_id, seed))
        objects.append(
            {
                "target_id": entry["target"]["target_id"],
                "object_local_id": seed.object_local_id,
                "sam_object_id": object_id,
                "seed": seed.as_dict(),
                "seed_frame_index": media.frame_for(seed.timestamp, tolerance),
                "prompt_role": _prompt_role(entry["target"], seed, seen_objects),
                "time_start": entry["target"]["time_start"],
                "time_end": entry["target"]["time_end"],
            }
        )
    plan = build_spatial_plan(pairs, media, group["direction"], session_id, tolerance)
    earliest = min(item["seed_frame_index"] for item in objects)
    return {
        "key": cache_key(video.checksum_sha256, frames, width, height, plan.prompt, group["direction"]),
        "session_id": session_id,
        "prompt": plan.prompt,
        "prompt_kind": "spatial",
        "direction": group["direction"],
        "bindings": {str(key): value for key, value in plan.bindings.items()},
        "objects": objects,
        "frame_times": frames,
        "source_start": frames[0],
        "source_end": frames[-1],
        "width": width,
        "height": height,
        "billed_frames": billed_frames(earliest, len(frames), group["direction"]),
        "model_id": MODEL_ID,
    }


def _concept_spec(video, target: dict, frames: list[float], session_id: str, width: int, height: int) -> dict:
    """A concept pass binds no object. Its boxes are only candidates for a person to identify."""
    phrase = target["concept"]
    return {
        "key": cache_key(video.checksum_sha256, frames, width, height, phrase, CONCEPT_DIRECTION),
        "session_id": session_id,
        "prompt": phrase,
        "prompt_kind": "concept",
        "direction": CONCEPT_DIRECTION,
        "bindings": {},
        "objects": [],
        "target_id": target["target_id"],
        "object_local_id": target["object_local_id"],
        "frame_times": frames,
        "source_start": frames[0],
        "source_end": frames[-1],
        "width": width,
        "height": height,
        "billed_frames": billed_frames(0, len(frames), "forward"),
        "model_id": MODEL_ID,
    }


def preflight_violations(plan: dict, limits: dict, *, spent_frames: int, spent_requests: int) -> list[str]:
    """Reasons this plan may not send its remaining requests. Checked before every remote call."""
    problems: list[str] = []
    remote = [spec for spec in plan["requests"] if not spec.get("cached")]
    if spent_requests + len(remote) > int(limits["max_requests"]):
        problems.append(
            f"The plan needs {spent_requests + len(remote)} provider requests and the limit is {limits['max_requests']}."
        )
    for spec in remote:
        if len(spec["frame_times"]) > int(limits["max_frames_per_request"]):
            problems.append(
                f"A request has {len(spec['frame_times'])} frames and the limit is {limits['max_frames_per_request']}."
            )
    total = spent_frames + sum(spec["billed_frames"] for spec in remote)
    if total > int(limits["max_total_billed_frames"]):
        problems.append(f"The plan bills about {total} frames and the limit is {limits['max_total_billed_frames']}.")
    price = settings.sam_price_per_frame_usd
    if price is not None and total * price > float(limits["cost_budget_usd"]):
        problems.append(f"The plan costs about ${total * price:.4f} and the budget is ${limits['cost_budget_usd']:.4f}.")
    return problems


def dry_run(session: Session, experiment: RecoveryExperiment) -> dict:
    _not_running(experiment)
    plan = build_plan(session, experiment)
    experiment.plan_json = plan
    experiment.status = "planned"
    experiment.mode = "dry_run"
    experiment.error = ""
    session.commit()
    return plan


# ---- Media ----


def prepare_clip(source: Path, frame_times: list[float], dest: Path, width: int, height: int) -> dict:
    """Cut and verify the file SAM receives. A clip whose frames do not line up is refused.

    Every source frame is re-encoded with its spacing kept, in displayed
    orientation at the source's displayed size. Normalized seed boxes therefore
    land on the same pixels. Frame `k` must be the source frame at `frame_times[k]`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    steps = [right - left for left, right in zip(frame_times, frame_times[1:], strict=False)]
    start = max(frame_times[0] - 0.0005, 0.0)
    # Ends between the last wanted frame and the next one.
    duration = frame_times[-1] - start + min(steps, default=1 / 30) / 2
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}", "-map", "0:v:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough", "-movflags", "+faststart",
    ]  # fmt: skip
    completed = _encode(command, dest)
    if completed.returncode != 0 or not dest.exists():
        raise RecoveryError(f"The recovery clip could not be cut. Nothing was submitted. {completed.stderr[-300:]}")
    verify_clip(dest, frame_times, width, height, source)
    from app.video.probe import sha256_file

    return {
        "path": relative(dest),
        "checksum_sha256": sha256_file(dest),
        "source_start": frame_times[0],
        "source_end": frame_times[-1],
        "clip_offset": frame_times[0],
        "frames": len(frame_times),
        "width": width,
        "height": height,
        "orientation": "displayed",
        "transform": "No resize or crop. Rotation is applied. Clip frame k is source frame frame_times[k].",
    }


def verify_clip(clip: Path, frame_times: list[float], width: int, height: int, source: Path | None = None) -> None:
    metadata = probe(clip)
    if not sam_video_accepted(metadata.codec, metadata.container.split(",")[0]):
        raise RecoveryError(f"SAM does not accept {metadata.codec} in {metadata.container}. Nothing was submitted.")
    if (metadata.display_width, metadata.display_height) != (width, height):
        raise RecoveryError(
            f"The clip is {metadata.display_width}x{metadata.display_height} and the source displays at "
            f"{width}x{height}. Nothing was submitted."
        )
    actual = presentation_timestamps(clip)
    if len(actual) != len(frame_times):
        raise RecoveryError(
            f"The clip holds {len(actual)} frames and the plan expects {len(frame_times)}. "
            "The mapping is ambiguous, so nothing was submitted."
        )
    steps = [right - left for left, right in zip(frame_times, frame_times[1:], strict=False)]
    allowed = max(min(steps, default=1 / 30) / 2, 1e-3)
    for index, (got, wanted) in enumerate(zip(actual, frame_times, strict=True)):
        if abs((got - actual[0]) - (wanted - frame_times[0])) > allowed:
            raise RecoveryError(
                f"Clip frame {index} does not line up with source time {wanted:.3f}s. "
                "The mapping is ambiguous, so nothing was submitted."
            )
    if source is not None:
        _compare_first_frame(clip, source, frame_times[0])


def _compare_first_frame(clip: Path, source: Path, timestamp: float) -> None:
    """The opening picture must be the source picture at that time. Fine detail may differ after encoding."""
    work = clip.with_suffix("")
    try:
        extract_frame(clip, 0.0, work.with_name(f"{work.name}_clip.jpg"))
        extract_frame(source, timestamp, work.with_name(f"{work.name}_src.jpg"))
        left = read_bgr(work.with_name(f"{work.name}_clip.jpg"))
        right = read_bgr(work.with_name(f"{work.name}_src.jpg"))
    except (VideoError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return
    if left.shape != right.shape:
        raise RecoveryError("The clip's first frame has a different shape from the source. Nothing was submitted.")
    small = (64, 64)
    difference = cv2.absdiff(cv2.resize(left, small), cv2.resize(right, small)).mean()
    if difference > 25:
        raise RecoveryError(
            f"The clip's first frame differs from the source at {timestamp:.3f}s. Nothing was submitted."
        )


# ---- Stored responses ----


def store_response(run_id: str, key: str, payload: dict) -> Path:
    """Write one whole response. A kill during the write cannot leave a partial file."""
    dest = _cache_path(run_id, key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(dest)
    return dest


def load_response(run_id: str, key: str) -> dict | None:
    try:
        payload = json.loads(_cache_path(run_id, key).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and payload.get("cache_key") == key else None


def _validation_policy(experiment: RecoveryExperiment) -> ValidationPolicy:
    return ValidationPolicy(**((experiment.limits_json or {}).get("_validation") or {}))


def _set_validation(experiment: RecoveryExperiment, overrides: dict | None) -> None:
    """Thresholds for a replay. Provider responses and their cache keys do not depend on them."""
    if overrides is None:
        return
    known = {item.name: item.type for item in dataclasses.fields(ValidationPolicy)}
    clean: dict = {}
    for name, value in overrides.items():
        if name not in known:
            raise RecoveryError(f"Unknown validation setting {name}.")
        try:
            clean[name] = int(value) if known[name] == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise RecoveryError(f"The validation setting {name} must be a number.") from exc
    experiment.limits_json = {**(experiment.limits_json or {}), "_validation": clean}


def replay(session: Session, experiment: RecoveryExperiment, validation: dict | None = None) -> dict:
    """Re-run parsing, timing, association, validation, and comparison from stored responses.

    A missing response stays missing. A replay never becomes a live request.
    """
    _not_running(experiment)
    _set_validation(experiment, validation)
    before = (experiment.summary_json or {}).get("totals")
    plan = build_plan(session, experiment)
    experiment.plan_json = plan
    found = {spec["key"]: load_response(experiment.run_id, spec["key"]) for spec in plan["requests"]}
    missing = [key for key, value in found.items() if value is None]
    experiment.mode = "replay"
    if len(missing) == len(found):
        experiment.status = "planned"
        experiment.error = (
            "No stored SAM response matches this plan. Nothing was sent. A changed seed, interval, "
            "or direction needs its own live request."
        )
        session.commit()
        return results(session, experiment)
    summary = evaluate(session, experiment, {key: value for key, value in found.items() if value})
    summary["missing_responses"] = len(missing)
    summary["previous_totals"] = before
    experiment.summary_json = summary
    experiment.status = "partial" if missing else "completed"
    experiment.error = f"{len(missing)} planned requests have no stored response." if missing else ""
    session.commit()
    return results(session, experiment)


# ---- Live, SAM only ----


def authorize_live(session: Session, experiment: RecoveryExperiment, payload: dict) -> dict:
    """Check the approval and claim the experiment. The approved plan hash must match a fresh plan."""
    if payload.get("confirm") is not True:
        raise RecoveryError("A live SAM run needs explicit confirmation. Nothing was submitted.")
    with _CLAIM_LOCK:
        session.refresh(experiment)
        _not_running(experiment)
        run = session.get(AnalysisRun, experiment.run_id)
        if run.status in {"pending", "running"}:
            raise RecoveryConflict("This analysis is still running. Nothing was submitted.")
        if run.active_revision_id != experiment.source_revision_id:
            raise RecoveryConflict(
                "The analysis changed after this experiment was created. Create a new one. Nothing was submitted."
            )
        other = session.scalar(
            select(RecoveryExperiment).where(
                RecoveryExperiment.run_id == experiment.run_id,
                RecoveryExperiment.status == "running",
            )
        )
        if other is not None:
            raise RecoveryConflict("Another recovery is running on this analysis. Nothing was submitted.")
        plan = build_plan(session, experiment)
        if plan["plan_hash"] != payload.get("plan_hash"):
            experiment.plan_json = plan
            experiment.status = "planned"
            session.commit()
            raise RecoveryConflict(
                "The plan changed since it was shown. Review the new estimate and approve again. Nothing was submitted."
            )
        if plan["violations"]:
            raise RecoveryError(" ".join(plan["violations"]) + " Nothing was submitted.")
        if not plan["requests"]:
            raise RecoveryError("The plan has no requests to run. Nothing was submitted.")
        experiment.plan_json = plan
        experiment.status = "running"
        experiment.mode = "live"
        experiment.cancel_requested = False
        experiment.error = ""
        experiment.limits_json = {**experiment.limits_json, "_resubmit": list(payload.get("resubmit_keys") or [])}
        session.commit()
    return plan


def start_live(experiment_id: str) -> None:
    """Run on its own thread, so the API stays responsive. It is never queued or resumed by itself."""

    def work() -> None:
        from app.db import SessionLocal

        session = SessionLocal()
        try:
            asyncio.run(execute_live(session, experiment_id))
        finally:
            session.close()

    if settings.ingest_inline:
        work()
    else:
        threading.Thread(target=work, name=f"recovery-{experiment_id[:8]}", daemon=True).start()


def _request_row(session: Session, experiment: RecoveryExperiment, spec: dict) -> RecoveryRequest:
    row = session.scalar(
        select(RecoveryRequest).where(
            RecoveryRequest.experiment_id == experiment.id,
            RecoveryRequest.cache_key == spec["key"],
        )
    )
    if row is None:
        row = RecoveryRequest(experiment_id=experiment.id, run_id=experiment.run_id, cache_key=spec["key"])
        session.add(row)
        session.flush()
    return row


def _uncertain_rows(session: Session, run_id: str, key: str) -> bool:
    """True when any experiment on this run sent this request and never saw it finish."""
    rows = session.scalars(
        select(RecoveryRequest).where(
            RecoveryRequest.run_id == run_id,
            RecoveryRequest.cache_key == key,
            RecoveryRequest.submission == "submitted",
        )
    ).all()
    return bool(rows)


async def execute_live(session: Session, experiment_id: str, adapter: SamAdapter | None = None) -> None:
    """Send the approved plan in order. Every remote call passes preflight first and is recorded before it leaves."""
    experiment = session.get(RecoveryExperiment, experiment_id)
    if experiment is None or experiment.status != "running":
        return
    run = session.get(AnalysisRun, experiment.run_id)
    video = session.get(VideoAsset, run.video_id)
    limits = experiment.limits_json
    plan = experiment.plan_json
    source = resolve_stored(video.stored_path)
    responses: dict[str, dict] = {}
    spent_frames = 0
    spent_requests = 0
    failure = ""
    canceled = False
    cost = 0.0
    cost_known = settings.sam_price_per_frame_usd is not None
    allowed = set(limits.get("_resubmit") or [])
    try:
        adapter = adapter or _adapter(session, float(limits["timeout_s"]))
    except RecoveryError as exc:
        experiment.status = "failed"
        experiment.error = str(exc)
        session.commit()
        return
    pending = [dict(spec) for spec in plan["requests"]]
    for index, spec in enumerate(pending):
        session.refresh(experiment)
        if experiment.cancel_requested:
            canceled = True
            break
        stored = load_response(run.id, spec["key"])
        row = _request_row(session, experiment, spec)
        if stored is not None:
            responses[spec["key"]] = stored
            row.status, row.source, row.submission = "completed", "cache", row.submission
            session.commit()
            continue
        if _uncertain_rows(session, run.id, spec["key"]) and spec["key"] not in allowed:
            row.status = "failed"
            row.error = (
                "An earlier attempt reached the provider and did not finish, so it may have been billed. "
                "It was not sent again. Approve a resubmission explicitly to retry it."
            )
            failure = failure or row.error
            session.commit()
            continue
        remaining = {"requests": [{**item, "cached": False} for item in pending[index:] if not _cache_path(run.id, item["key"]).exists()]}
        problems = preflight_violations(remaining, limits, spent_frames=spent_frames, spent_requests=spent_requests)
        if problems:
            row.status = "failed"
            row.error = " ".join(problems) + " This request was not submitted."
            failure = row.error
            session.commit()
            break
        outcome = await _send_one(session, experiment, run, spec, row, adapter, source, limits)
        if outcome.get("canceled"):
            canceled = True
            break
        if outcome.get("error"):
            failure = outcome["error"]
            break
        responses[spec["key"]] = outcome["payload"]
        spent_requests += 1
        spent_frames += spec["billed_frames"]
        if row.cost_usd is not None:
            cost += row.cost_usd
    session.refresh(experiment)
    experiment.recorded_cost_usd = round(cost, 6) if cost_known else None
    if responses:
        experiment.summary_json = evaluate(session, experiment, responses)
    experiment.error = failure or ("Canceled. Requests already sent may still be billed." if canceled else "")
    if canceled:
        experiment.status = "canceled"
    elif failure:
        experiment.status = "partial" if responses else "failed"
    else:
        experiment.status = "completed"
    session.commit()


async def _send_one(session, experiment, run, spec, row, adapter, source, limits) -> dict:
    """One provider request. The clip is checked and the row saved before anything leaves this machine."""
    row.status = "running"
    row.billed_frames = spec["billed_frames"]
    session.commit()
    dest = _experiment_dir(run.id, experiment.id) / "media" / f"{spec['key'][:24]}.mp4"
    try:
        row.media_json = await asyncio.to_thread(
            prepare_clip, source, spec["frame_times"], dest, spec["width"], spec["height"]
        )
    except (RecoveryError, VideoError, OSError) as exc:
        row.status = "failed"
        row.error = str(exc)
        session.commit()
        return {"error": f"{exc} This request was not submitted."}
    call = open_provider_call(
        session,
        run,
        stage=STAGE,
        provider_id="meta",
        model_id=MODEL_ID,
        modality="native_video",
        parameters={
            "prompt_kind": spec.get("prompt_kind", "spatial"),
            "experiment_id": experiment.id,
            "cache_key": spec["key"],
            "propagation_direction": spec["direction"],
            "chunk_start": spec["source_start"],
            "chunk_end": spec["source_end"],
            "frames": len(spec["frame_times"]),
            "billed_frames": spec["billed_frames"],
            "bindings": spec["bindings"],
        },
        frame_timestamps=[spec["source_start"], spec["source_end"]],
    )
    row.model_request_id = call.id
    session.commit()
    file_id = ""
    retries = int(limits.get("max_upload_retries") or 0)
    for attempt in range(retries + 1):
        try:
            file_id = await asyncio.wait_for(adapter.upload_video(dest), timeout=float(limits["timeout_s"]))
            break
        except Exception as exc:  # noqa: BLE001 - the upload error is recorded and reported.
            if attempt < retries:
                continue
            fail_provider_call(call, exc)
            row.status = "failed"
            row.error = str(exc)[:2000]
            session.commit()
            return {"error": f"The clip upload failed, so no SAM request was submitted. {exc}"}
    row.submission = "submitted"
    session.commit()
    started = time.perf_counter()
    task = asyncio.create_task(
        adapter.segment_video(
            None,
            spec["prompt"],
            prompt_kind=spec.get("prompt_kind", "spatial"),
            file_id=file_id,
            propagation_direction=None if spec["direction"] == CONCEPT_DIRECTION else spec["direction"],
        )
    )
    deadline = started + float(limits["timeout_s"])
    canceled = False
    while True:
        done, _pending = await asyncio.wait({task}, timeout=0.5)
        if done:
            break
        session.refresh(experiment)
        if experiment.cancel_requested:
            canceled = True
        if canceled or time.perf_counter() > deadline:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            message = (
                "Canceled while the SAM request was in flight."
                if canceled
                else f"The SAM request passed the {limits['timeout_s']:.0f}s limit."
            )
            fail_provider_call(call, RuntimeError(message))
            row.status = "canceled" if canceled else "failed"
            row.error = f"{message} The request was submitted and may be billed. It was not retried."
            session.commit()
            return {"canceled": True} if canceled else {"error": row.error}
    try:
        result = task.result()
    except Exception as exc:  # noqa: BLE001 - submitted calls stay marked, so a retry cannot repeat them silently.
        fail_provider_call(call, exc)
        row.status = "failed"
        row.error = f"{exc}"[:2000]
        session.commit()
        return {"error": f"A SAM request was submitted and failed. It may be billed and was not retried. {exc}"}
    price = settings.sam_price_per_frame_usd
    amount = round(spec["billed_frames"] * price, 6) if price is not None else None
    payload = {
        "cache_key": spec["key"],
        "text": result.text,
        "request_id": result.request_id,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "model_id": MODEL_ID,
        "prompt": spec["prompt"],
        "direction": spec["direction"],
        "frame_times": spec["frame_times"],
        "media": row.media_json,
        "synthetic": False,
    }
    store_response(run.id, spec["key"], payload)
    finish_provider_call(
        session,
        run,
        call,
        ProviderResponse(
            text=result.text,
            raw={
                "id": result.request_id,
                "model": MODEL_ID,
                "file_id": result.file_id,
                "response_file": relative(_cache_path(run.id, spec["key"])),
                "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
                "cost_basis": "planned frames at the configured rate" if amount is not None else "unpriced",
            },
            usage=Usage(result.input_tokens, result.output_tokens, amount or 0.0, amount is not None),
            request_id=result.request_id,
            latency_ms=result.latency_ms,
            model_id=MODEL_ID,
        ),
    )
    row.status = "completed"
    row.submission = "completed"
    row.source = "live"
    row.provider_request_id = result.request_id
    row.latency_ms = result.latency_ms
    row.input_tokens = result.input_tokens
    row.output_tokens = result.output_tokens
    row.cost_usd = amount
    row.error = ""
    session.commit()
    return {"payload": payload}


def cancel(session: Session, experiment: RecoveryExperiment) -> RecoveryExperiment:
    if experiment.status == "running":
        experiment.cancel_requested = True
    elif experiment.status in {"draft", "planned"}:
        experiment.status = "canceled"
    session.commit()
    return experiment


def recover_interrupted(session: Session) -> int:
    """A live recovery does not survive a restart and is never resumed by itself."""
    rows = session.scalars(select(RecoveryExperiment).where(RecoveryExperiment.status == "running")).all()
    for experiment in rows:
        experiment.status = "interrupted"
        experiment.error = (
            "The server stopped during this recovery. Stored responses can be replayed. "
            "A request that was in flight may be billed and is not sent again unless you approve it."
        )
        for request in session.scalars(
            select(RecoveryRequest).where(RecoveryRequest.experiment_id == experiment.id, RecoveryRequest.status == "running")
        ):
            request.status = "failed"
            request.error = "Interrupted by a server restart."
    session.commit()
    return len(rows)


# ---- Evaluation ----


def _measured_anchors(document: AnalysisDocument, spatial: AnalysisDocument | None) -> dict[str, list[dict]]:
    """Boxes that may confirm or contradict a recovered box. A model estimate is neither."""
    from app.perception.seeds import _measured, _pool

    anchors: dict[str, list[dict]] = {}
    for row in _pool(document, spatial):
        family = row_family(row)
        if family == "human" or _measured(row):
            anchors.setdefault(row.object_local_id, []).append(
                {
                    "timestamp": row.timestamp,
                    "bbox": row.bbox.model_dump(),
                    "source": family,
                    "object_local_id": row.object_local_id,
                }
            )
    return anchors


def evaluate(session: Session, experiment: RecoveryExperiment, responses: dict[str, dict]) -> dict:
    """Stored responses become a new generation of proposals. No row of the analysis changes."""
    run_id = experiment.run_id
    document = _document(session, experiment.source_revision_id)
    spatial = _usable_spatial(run_id)
    boundaries = _boundaries(session, run_id)
    anchors = _measured_anchors(document, spatial)
    plan = experiment.plan_json
    policy = _validation_policy(experiment)
    specs = [spec for spec in plan["requests"] if spec["key"] in responses]
    tracked: dict[str, list[dict]] = {}
    reacquisitions = {item["target_id"]: item.get("reacquires") for item in experiment.targets_json or []}
    neighbor_rows: dict[str, list[dict]] = {}
    passes: dict[str, list[list[dict]]] = {}
    gaps_frames: dict[str, list[float]] = {}
    usage: dict[str, dict] = {}
    identity: dict[str, list[dict]] = {}
    dropped: dict[str, int] = {}
    counted_usage: set[tuple[str, str]] = set()
    synthetic = False
    for spec in specs:
        payload = responses[spec["key"]]
        synthetic = synthetic or bool(payload.get("synthetic"))
        samples = _spec_samples(spec, payload)
        associations = associate_tracks(document, samples, spatial=spatial, measured_bindings=True)
        status_by_track = {item.raw_track_id: item for item in associations}
        by_object: dict[str, list] = {}
        unbound = 0
        for sample in samples:
            if sample.bound_object_id:
                by_object.setdefault(sample.bound_object_id, []).append(sample)
            else:
                unbound += 1
        for entry in spec["objects"]:
            local_id = entry["object_local_id"]
            target_id = entry["target_id"]
            window = _prompt_window(entry, spec)
            rows = [item for item in by_object.get(local_id, []) if window[0] - 1e-6 <= item.timestamp < window[1]]
            raw = str(entry["sam_object_id"])
            association = status_by_track[raw].status if raw in status_by_track else "unassigned"
            neighbors = [
                item
                for other, items in anchors.items()
                if other != local_id
                for item in items
            ]
            neighbors += [
                {"timestamp": sample.timestamp, "bbox": sample.bbox.model_dump(), "object_local_id": other}
                for other, others in by_object.items()
                if other != local_id
                for sample in others
            ]
            validated = validate_track(
                [{**_sample_dict(sample), "mask_stats": mask_stats(sample)} for sample in rows],
                seed=entry["seed"],
                association=association,
                anchors=anchors.get(local_id, []),
                neighbors=neighbors,
                media_size=(spec["width"], spec["height"]),
                boundaries=boundaries,
                policy=policy,
            )
            pass_id = f"{spec['session_id']}:{raw}:{entry['seed']['seed_id'][:8]}"
            neighbor_rows.setdefault(target_id, []).extend(neighbors)
            tracked.setdefault(target_id, []).append(
                {
                    "rows": validated,
                    "seed": entry["seed"],
                    "pass_id": pass_id,
                    "experiment_id": experiment.id,
                    "request_key": spec["key"],
                    "session_id": spec["session_id"],
                    "raw_track_id": raw,
                    "request_id": payload.get("request_id", ""),
                    "direction": spec["direction"],
                    "frame_times": [time for time in spec["frame_times"] if window[0] - 1e-6 <= time < window[1]],
                }
            )
            for row in validated:
                row["request_key"] = spec["key"]
                row["session_id"] = spec["session_id"]
                row["raw_track_id"] = raw
                row["request_id"] = payload.get("request_id", "")
                row["direction"] = spec["direction"]
                row["seed_id"] = entry["seed"]["seed_id"]
                row["seed_trust"] = entry["seed"]["trust"]
                row["pass_id"] = pass_id
                row["segment_id"] = f"{pass_id}/{row['segment']}" if row.get("segment") else ""
                row["reacquired"] = bool(reacquisitions.get(target_id))
                row["context"] = not (entry["time_start"] - 1e-6 <= row["timestamp"] <= entry["time_end"] + 1e-6)
            passes.setdefault(target_id, []).append(validated)
            gaps_frames[target_id] = sorted(set(gaps_frames.get(target_id, [])) | set(spec["frame_times"]))
            bucket = usage.setdefault(
                target_id, {"requests": 0, "billed_frames": 0, "latency_ms": 0, "input_tokens": 0, "output_tokens": 0}
            )
            if (target_id, spec["key"]) not in counted_usage:
                counted_usage.add((target_id, spec["key"]))
                bucket["requests"] += 1
                bucket["billed_frames"] += spec["billed_frames"]
                bucket["latency_ms"] += int(payload.get("latency_ms") or 0)
                bucket["input_tokens"] += int(payload.get("input_tokens") or 0)
                bucket["output_tokens"] += int(payload.get("output_tokens") or 0)
            identity.setdefault(target_id, []).append(
                {
                    "session_id": spec["session_id"],
                    "sam_object_id": entry["sam_object_id"],
                    "association": association,
                    "note": status_by_track[raw].note if raw in status_by_track else "SAM returned no rows for this object.",
                    "seed_trust": entry["seed"]["trust"],
                }
            )
            dropped[target_id] = dropped.get(target_id, 0) + unbound
    experiment.generation = int(experiment.generation or 0) + 1
    totals = {"supported": 0, "provisional": 0, "conflicting": 0, "rejected": 0, "unobserved_frames": 0, "proposals": 0}
    for target in experiment.targets_json or []:
        target_id = target["target_id"]
        if target_id not in passes:
            continue
        combined = combine_passes(passes[target_id], policy)
        start, end = float(target["time_start"]), float(target["time_end"])
        gaps = remaining_gaps(combined, gaps_frames[target_id], start, end)
        metrics = _metrics(target, combined, gaps, document, usage[target_id], identity[target_id], dropped[target_id], synthetic)
        metrics["track"] = track_report(target, tracked[target_id], neighbor_rows.get(target_id, []), policy)
        metrics["counts"]["no_detection_frames"] = metrics["track"]["no_detection_frames"]
        metrics["counts"]["unusable_frames"] = metrics["track"]["unusable_frames"]
        session.add(
            RecoveryProposal(
                experiment_id=experiment.id,
                run_id=run_id,
                generation=experiment.generation,
                target_id=target_id,
                object_local_id=target["object_local_id"],
                time_start=start,
                time_end=end,
                rows_json=combined,
                metrics_json=metrics,
            )
        )
        totals["proposals"] += 1
        for name in ("supported", "provisional", "conflicting", "rejected"):
            totals[name] += metrics["counts"][name]
        totals["unobserved_frames"] += metrics["counts"]["unobserved_frames"]
    session.flush()
    return {
        "generation": experiment.generation,
        "totals": totals,
        "synthetic": synthetic,
        "accuracy_note": "Agreement between producers is not accuracy. Accuracy needs human-labeled boxes.",
    }


def _prompt_window(entry: dict, spec: dict) -> tuple[float, float]:
    """The times one prompt of a multi-frame object answers for.

    SAM is told where the object is at each prompt, so agreement there proves nothing. Each prompt
    starts its own stretch, which lets a loss before a later prompt still be seen.
    """
    times = sorted(
        {
            float(item["seed"]["timestamp"])
            for item in spec["objects"]
            if item["object_local_id"] == entry["object_local_id"] and item["target_id"] == entry["target_id"]
        }
    )
    time = float(entry["seed"]["timestamp"])
    if len(times) < 2:
        return float("-inf"), float("inf")
    index = times.index(time)
    return (float("-inf") if index == 0 else time), (float("inf") if index == len(times) - 1 else times[index + 1])


def _spec_samples(spec: dict, payload: dict) -> list:
    plan = SamRequestPlan(
        kind=spec.get("prompt_kind", "spatial"),
        prompt=spec["prompt"],
        session_id=spec["session_id"],
        bindings={int(key): value for key, value in spec["bindings"].items()},
        propagation=spec["direction"],
        labels=[],
        targets=list(spec["bindings"].values()),
    )
    times = spec["frame_times"]
    chunk = VideoChunk(
        index=0,
        start_frame=0,
        end_frame=len(times),
        time_start=times[0],
        time_end=times[-1],
        session_id=spec["session_id"],
        frame_times=times,
    )
    return _samples_from_result(parse_sam_output(payload.get("text") or ""), payload.get("request_id", ""), plan, chunk, 0.0)


def mask_stats(sample) -> dict | None:
    """Mask area, fill of its box, and region count. None when no mask decodes, which proves nothing."""
    data = sample.mask_data if isinstance(sample.mask_data, str) else ""
    if not data:
        return None
    raster = decode_mask_raster(SimpleNamespace(mask_data=data, mask_payload=sample.mask_payload or ""))
    height, width = int(sample.mask_height or 0), int(sample.mask_width or 0)
    if raster is None or height <= 0 or width <= 0:
        return None
    try:
        grid = np.frombuffer(bytes(raster), dtype=np.uint8).reshape(height, width) > 0
    except ValueError:
        return None
    pixels = int(grid.sum())
    regions, _labels = cv2.connectedComponents(grid.astype(np.uint8))
    return {"pixels": pixels, "fill": round(pixels / (height * width), 4), "components": max(regions - 1, 0)}


def _sample_dict(sample) -> dict:
    return {
        "timestamp": sample.timestamp,
        "bbox": sample.bbox.model_dump(),
        "confidence": sample.confidence,
        "frame_index": sample.frame_index,
        "source_width": sample.source_width,
        "source_height": sample.source_height,
        "mask_data": sample.mask_data,
        "mask_payload": sample.mask_payload,
        "mask_encoding": sample.mask_encoding,
        "mask_height": sample.mask_height,
    }


def _metrics(target, rows, gaps, document, usage, identity, unbound, synthetic) -> dict:
    local_id = target["object_local_id"]
    start, end = float(target["time_start"]), float(target["time_end"])
    inside = [row for row in rows if not row.get("context")]
    originals = [
        row
        for row in [*document.observations, *document.raw_observations]
        if row.object_local_id == local_id and row.bbox is not None and start - 0.02 <= row.timestamp <= end + 0.02
    ]
    original_times = [row.timestamp for row in originals]
    counts = {
        name: sum(1 for row in inside if row["outcome"] == name)
        for name in (SUPPORTED, PROVISIONAL, CONFLICTING, REJECTED)
    }
    counts["unobserved_frames"] = sum(item["frames"] for item in gaps)
    previously_missing = sum(
        1 for row in inside if not any(abs(time - row["timestamp"]) <= 0.02 for time in original_times)
    )
    human = [row for row in originals if row_family(row) == "human"]
    overlaps = []
    from app.perception.assignment import iou

    for row in inside:
        for other in human:
            if abs(other.timestamp - row["timestamp"]) <= 0.02:
                box = row["bbox"]
                overlaps.append(iou((box["x"], box["y"], box["w"], box["h"]), (other.bbox.x, other.bbox.y, other.bbox.w, other.bbox.h)))
    return {
        "counts": counts,
        "previously_missing_frames": previously_missing,
        "new_supported_frames": sum(
            1
            for row in inside
            if row["outcome"] == SUPPORTED and not any(abs(time - row["timestamp"]) <= 0.02 for time in original_times)
        ),
        "original_rows_in_interval": len(originals),
        "original_estimates_in_interval": sum(is_estimate(row) for row in originals),
        "remaining_gaps": gaps,
        "identity": identity,
        "unbound_samples": unbound,
        "usage": usage,
        "synthetic_response": synthetic,
        "human_box_overlap": (
            {"frames": len(overlaps), "mean_iou": round(sum(overlaps) / len(overlaps), 4)} if overlaps else None
        ),
        "accuracy_note": (
            "Measured against boxes a person drew on these frames."
            if overlaps
            else "No human-labeled box covers this interval, so accuracy was not measured."
        ),
        "state": target.get("state", ""),
    }


# ---- Decisions ----


def decide(session: Session, experiment: RecoveryExperiment, payload: dict) -> dict:
    """Accept or reject proposals. Accepting forks one revision and leaves every earlier one intact."""
    decision = payload.get("decision")
    if decision not in {"accepted", "rejected"}:
        raise RecoveryError("A decision must be accepted or rejected.")
    ids = list(payload.get("proposal_ids") or [])
    rows = list(
        session.scalars(
            select(RecoveryProposal).where(
                RecoveryProposal.experiment_id == experiment.id,
                RecoveryProposal.id.in_(ids),
            )
        )
    )
    if len(rows) != len(set(ids)) or not rows:
        raise RecoveryError("Unknown recovery proposal.")
    if any(row.decision != "pending" for row in rows):
        raise RecoveryConflict("One of these proposals was already decided.")
    if any(row.generation != experiment.generation for row in rows):
        raise RecoveryConflict("One of these proposals is from an older replay. Decide the current one.")
    if decision == "rejected":
        for row in rows:
            row.decision = "rejected"
            row.decided_at = utcnow()
        session.commit()
        return results(session, experiment)
    with _APPLY_LOCK:
        _apply(session, experiment, rows, include_provisional=bool(payload.get("include_provisional")))
    return results(session, experiment)


def _apply(session: Session, experiment: RecoveryExperiment, proposals: list[RecoveryProposal], *, include_provisional: bool) -> None:
    run = session.get(AnalysisRun, experiment.run_id)
    if run.status in {"pending", "running"}:
        raise RecoveryConflict("This analysis is still running. Nothing was applied.")
    allowed = {experiment.source_revision_id, experiment.applied_revision_id}
    if run.active_revision_id not in allowed:
        raise RecoveryConflict(
            "The analysis changed after these proposals were made. Create a new experiment and replay it for free. "
            "Nothing was applied."
        )
    targets = {item["target_id"]: item for item in experiment.targets_json or []}
    spatial = _usable_spatial(run.id)

    def mutate(document: AnalysisDocument) -> None:
        for proposal in proposals:
            merge_proposal(document, proposal, targets.get(proposal.target_id, {}), experiment.id, include_provisional)
        resolved = resolve_document(document, spatial)
        document.observations = resolved.observations
        document.raw_observations = resolved.raw_observations
        document.ambiguities = resolved.ambiguities
        document.perception = resolved.perception
        for proposal in proposals:
            proposal.decision = "accepted"
            proposal.decided_at = utcnow()

    names = ", ".join(sorted({proposal.object_local_id for proposal in proposals}))
    try:
        revision = revise(session, run, f"accepted targeted recovery for {names}", "recovery", experiment.id, mutate)
    except Exception:
        session.rollback()
        raise
    experiment.applied_revision_id = revision.id
    for proposal in proposals:
        proposal.applied_revision_id = revision.id
    session.commit()


def merge_proposal(
    document: AnalysisDocument,
    proposal: RecoveryProposal,
    target: dict,
    experiment_id: str,
    include_provisional: bool = False,
) -> None:
    """Add a proposal's rows beside the existing evidence. Existing rows stay stored and unedited.

    A row on a frame that already has a canonical box is dropped when it agrees and
    stored as a conflict when it does not, so recovery cannot displace an existing
    canonical row. Applying the same proposal twice adds nothing.
    """
    from app.perception.assignment import iou

    local_id = proposal.object_local_id
    existing = {
        (item.object_local_id, round(item.timestamp, 3), item.track_ref)
        for item in [*document.observations, *document.raw_observations]
    }
    canonical = [
        item
        for item in document.observations
        if item.object_local_id == local_id and item.observation_role == "canonical" and item.source_model != ""
        and item.bbox is not None
    ]
    added: list[ObservationPrediction] = []
    conflict_times: list[float] = []
    lost_times: list[float] = []
    for row in proposal.rows_json or []:
        if row.get("context") or row["outcome"] == REJECTED:
            continue
        # Kept visible as a conflict, which never becomes canonical. Whether it is right is for a person to judge.
        if row.get("lifecycle") == "lost":
            lost_times.append(row["timestamp"])
        if row["outcome"] == PROVISIONAL and not include_provisional:
            continue
        box = row["bbox"]
        outcome = row["outcome"]
        same = next((item for item in canonical if abs(item.timestamp - row["timestamp"]) <= 0.02), None)
        if same is not None:
            overlap = iou((box["x"], box["y"], box["w"], box["h"]), (same.bbox.x, same.bbox.y, same.bbox.w, same.bbox.h))
            if overlap >= 0.5:
                continue
            outcome = CONFLICTING
        track_ref = f"{row['session_id']}:{row['raw_track_id']}"
        if (local_id, round(row["timestamp"], 3), track_ref) in existing:
            continue
        existing.add((local_id, round(row["timestamp"], 3), track_ref))
        if outcome == CONFLICTING:
            conflict_times.append(row["timestamp"])
        added.append(_observation(row, local_id, track_ref, outcome, experiment_id))
    document.raw_observations = [*document.raw_observations, *added]
    seed = target.get("seed")
    if seed and seed.get("bbox"):
        manual = _human_row(seed, local_id, experiment_id)
        key = (local_id, round(manual.timestamp, 3), manual.track_ref)
        if key not in existing:
            document.raw_observations.append(manual)
    gaps = (proposal.metrics_json or {}).get("remaining_gaps") or []
    for gap in gaps:
        _add_ambiguity(
            document,
            f"recovery_gap_{local_id}_{gap['start']:.2f}",
            AmbiguityKind.UNOBSERVED_INTERVAL,
            (
                f"Recovery found no usable box for {local_id} from {gap['start']:.2f}s to {gap['end']:.2f}s. "
                "The object may be hidden. That is not evidence that it is absent."
            ),
            gap["start"],
            gap["end"],
            local_id,
        )
    if lost_times:
        _add_ambiguity(
            document,
            f"recovery_track_lost_{local_id}_{min(lost_times):.2f}",
            AmbiguityKind.CONFLICTING_EVIDENCE,
            (
                f"SAM kept returning boxes for {local_id} from {min(lost_times):.2f}s to {max(lost_times):.2f}s after the "
                "track broke. They are kept as conflicts and were not trusted. The object may be hidden or out of view."
            ),
            min(lost_times),
            max(lost_times),
            local_id,
        )
    if conflict_times:
        _add_ambiguity(
            document,
            f"recovery_conflict_{local_id}_{min(conflict_times):.2f}",
            AmbiguityKind.CONFLICTING_EVIDENCE,
            f"Recovered boxes for {local_id} disagree with other evidence. Both were kept and neither was promoted.",
            min(conflict_times),
            max(conflict_times),
            local_id,
        )


def _add_ambiguity(document, local_id, kind, description, start, end, object_id) -> None:
    if any(item.local_id == local_id for item in document.ambiguities):
        return
    document.ambiguities.append(
        AmbiguityPrediction(
            local_id=local_id,
            kind=kind,
            description=description,
            time_start=start,
            time_end=end,
            object_local_ids=[object_id],
            evidence=Evidence(
                classification=EvidenceClass.UNKNOWN,
                provenance=Provenance.CV_DETECTION,
                note="From targeted object recovery.",
            ),
        )
    )


def _observation(row: dict, local_id: str, track_ref: str, outcome: str, experiment_id: str) -> ObservationPrediction:
    association = {SUPPORTED: "associated", PROVISIONAL: "referenced", CONFLICTING: "conflict"}[outcome]
    continuity = Continuity.CONTINUOUS if outcome == SUPPORTED else Continuity.UNCERTAIN
    if outcome == SUPPORTED and row.get("reacquired"):
        continuity = Continuity.REASSOCIATED
    return ObservationPrediction(
        object_local_id=local_id,
        timestamp=row["timestamp"],
        bbox=BBox.model_validate(row["bbox"]),
        visibility=Visibility.PARTIAL if row.get("partial") else Visibility.VISIBLE,
        description="SAM 3.1 targeted recovery",
        evidence=Evidence(
            classification=EvidenceClass.DIRECTLY_OBSERVED,
            provenance=Provenance.CV_DETECTION,
            note=(
                f"Recovery experiment {experiment_id}, outcome {outcome}, seed {row.get('seed_trust', '')}, "
                f"segment {row.get('segment_id', '')}."
            ),
            frame_timestamps=[row["timestamp"]],
        ),
        confidence=row.get("confidence"),
        continuity=continuity,
        track_ref=track_ref,
        association_status=association,
        source_model=MODEL_ID,
        request_id=str(row.get("request_id") or ""),
        mask_ref=(
            f"recovery:{row['request_key'][:32]}:{row.get('frame_index')}:{row['raw_track_id']}" if row.get("has_mask") else ""
        ),
        geometry_origin="detected",
        geometry_authority=MODEL_ID,
        detection_support="mask" if row.get("has_mask") else "detected",
        association_support=association,
        semantic_support="provisional",
    )


def _human_row(seed: dict, local_id: str, experiment_id: str) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=local_id,
        timestamp=float(seed["timestamp"]),
        bbox=BBox.model_validate(seed["bbox"]),
        description="Seed drawn by the researcher",
        evidence=Evidence(
            classification=EvidenceClass.DIRECTLY_OBSERVED,
            provenance=Provenance.HUMAN_ANNOTATION,
            note=f"Drawn for recovery experiment {experiment_id}.",
            frame_timestamps=[float(seed["timestamp"])],
        ),
        track_ref=f"human:{experiment_id[:8]}",
        geometry_origin="detected",
        geometry_authority="human",
        detection_support="none",
        semantic_support="labeled",
    )


# ---- Reading results ----


def experiment_dict(session: Session, experiment: RecoveryExperiment) -> dict:
    requests = session.scalars(select(RecoveryRequest).where(RecoveryRequest.experiment_id == experiment.id)).all()
    remote_submitted = any(row.submission in {"submitted", "completed"} and row.source != "cache" for row in requests)
    limits = {key: value for key, value in (experiment.limits_json or {}).items() if not key.startswith("_")}
    pending = session.scalar(
        select(func.count())
        .select_from(RecoveryProposal)
        .where(
            RecoveryProposal.experiment_id == experiment.id,
            RecoveryProposal.generation == experiment.generation,
            RecoveryProposal.decision == "pending",
        )
    )
    return {
        "id": experiment.id,
        "run_id": experiment.run_id,
        "source_revision_id": experiment.source_revision_id,
        "status": experiment.status,
        "mode": experiment.mode,
        "generation": experiment.generation,
        "pending_proposals": pending or 0,
        "targets": experiment.targets_json,
        "limits": limits,
        "plan": experiment.plan_json or None,
        "summary": experiment.summary_json or None,
        "applied_revision_id": experiment.applied_revision_id,
        "error": experiment.error,
        "cancel_requested": experiment.cancel_requested,
        "remote_submitted": remote_submitted,
        "recorded_cost_usd": experiment.recorded_cost_usd,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else "",
        "requests": [
            {
                "cache_key": row.cache_key,
                "status": row.status,
                "submission": row.submission,
                "source": row.source,
                "billed_frames": row.billed_frames,
                "latency_ms": row.latency_ms,
                "provider_request_id": row.provider_request_id,
                "cost_usd": row.cost_usd,
                "error": row.error,
                "media": row.media_json,
            }
            for row in requests
        ],
    }


def results(session: Session, experiment: RecoveryExperiment) -> dict:
    rows = session.scalars(
        select(RecoveryProposal)
        .where(RecoveryProposal.experiment_id == experiment.id)
        .order_by(RecoveryProposal.generation.desc(), RecoveryProposal.time_start)
    ).all()
    return {
        **experiment_dict(session, experiment),
        "proposals": [
            {
                "id": row.id,
                "generation": row.generation,
                "current": row.generation == experiment.generation,
                "target_id": row.target_id,
                "object_local_id": row.object_local_id,
                "time_start": row.time_start,
                "time_end": row.time_end,
                "decision": row.decision,
                "rows": row.rows_json,
                "metrics": row.metrics_json,
                "applied_revision_id": row.applied_revision_id,
            }
            for row in rows
        ],
    }
