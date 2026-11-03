"""Object thumbnails and per-object timelines for a demonstration.

One Meta call finds each object on the first frame, and SAM follows those boxes
through a low frame rate copy of the recording. When some object is not on the
first frame, or SAM stopped following one the analysis still uses later, a
second call looks at a few later stills chosen from the analysis, and a second
SAM request follows those boxes forward. Each box is then checked against DINOv2
appearance references (appearance_stage.py): a demonstration learns them here, an attempt
reads its procedure's. A rejected box stays in raw_tracks.json and is not drawn. No box
decides a verdict. A failure leaves the analysis in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config import settings
from app.models.media import VideoAsset
from app.models.runs import AnalysisRun
from app.perception import appearance as app_
from app.perception.sam_masks import cutout_from_mask
from app.pipeline.contracts import GUIDED_LOCATE, GUIDED_LOCATE_LATER
from app.providers.sam import (
    MAX_PROMPTED_OBJECTS,
    SamAdapter,
    SamRecord,
    SpatialMark,
    decode_mask_raster,
    normalized_box,
    parse_sam_output,
    render_spatial_prompt,
)
from app.providers.types import FrameInput, GenerationRequest
from app.schemas.analysis import AnalysisDocument
from app.schemas.generation import GuidedLaterLocateOutput, GuidedLocateOutput
from app.services.storage import data_path, resolve_stored
from app.verification.read import enum_value
from app.video.frames import extract_frame, sharpness_score
from app.video.probe import probe

log = logging.getLogger("learnloop.object_views")

# At 8 a part picked up quickly in a close shot can go from flat on the table to a sliver between fingers in one frame, and SAM lets it go.
SAM_FPS = 12.0
# Seconds between timeline cells. A shown frame sits within half an interval of where an even spacing would put it.
_CELL_INTERVAL = 1.0
_CELL_SPREAD = 0.5
_DEFAULT_SAM = "sam-3.1"
_LONG_SIDE = 1080
_CROP_SIDE = 160
_PADDING = 0.15
_MANIFEST = "manifest.json"
_LOCATE_OUTPUT_TOKENS = 8_000
# A box over most of the frame is the scene, not one part.
_MAX_SEED_AREA = 0.5
LOCATE_SYSTEM = """
You are locating known objects on one still image from the start of a recording. Return one entry per listed object, using its local_id.
bbox is a tight box around that one object alone, out to its outermost visible pixels, with x, y, w, and h as fractions of the image from 0 to 1 and the origin at the top left.
Objects with identical labels are separate physical parts: give each its own box on a different part.
Small parts count. Look closely before deciding a part is not there. A mark on the surface or a shadow is not a part.
If an object is not visible in this image, set bbox to null. Do not guess a location.
Return JSON only.
""".strip()
LATER_SYSTEM = """
You are locating known objects on still images from later in a recording. Each image follows a line giving its timestamp. Return one entry per listed object, using its local_id.
For each object, choose the one image where it is most clearly visible and box it there. timestamp is that image's timestamp, copied exactly.
bbox is a tight box around that one object alone, out to its outermost visible pixels, with x, y, w, and h as fractions of the image from 0 to 1 and the origin at the top left.
A listed object may look like a part that is not listed. Box only a part that fits the listed description, and box a part that is in a hand when it is clearly visible.
If an object is not clearly visible in any image, set timestamp and bbox to null. Do not guess a location.
Return JSON only.
""".strip()
# Later stills sent in one request, and per missing object.
_LATER_FRAMES = 6
_LATER_PER_OBJECT = 2
# Candidate moments closer than this are one moment.
_LATER_MERGE = 0.5
# The sharpest frame this close to a candidate moment is sent instead of it.
_SHARPEST_WITHIN = 0.25
# A track that ends this long before a moment the analysis ties to its object was lost, not finished.
_LOST_AFTER = 1.0


@dataclass
class Seed:
    local_id: str
    object_id: int
    timestamp: float
    box: dict


@dataclass
class Sighting:
    timestamp: float
    box: dict
    record: SamRecord
    # Set by the appearance check: validated, recovered, provisional, ambiguous, or rejected.
    status: str = "provisional"
    # sam for a box from the first requests, recovery for one SAM followed from a box appearance search found.
    provenance: str = "sam"
    reason: str = "unchecked"
    # True when a recovered box for the same frame is drawn instead.
    superseded: bool = False


def estimated_frames(duration_sec: float) -> int:
    """The most frames one request bills. Meta charges by the frame."""
    return int(duration_sec * SAM_FPS) + 1


def views_dir(run_id: str) -> Path:
    return data_path("runs", run_id, "objects")


def read_manifest(run_id: str) -> dict | None:
    try:
        return json.loads((views_dir(run_id) / _MANIFEST).read_text())
    except (OSError, ValueError):
        return None


async def run_object_views(session, run: AnalysisRun, video: VideoAsset, stage, allow_live: bool | None = None, keep_on_empty: bool = False) -> dict:
    """allow_live, when given, replaces the run's own approval. A person approves it again to make views for a finished run.
    keep_on_empty leaves the stored views in place when this pass followed nothing, as a rebuild from stored answers may."""
    from app.pipeline.infer import load_working
    from app.services.credentials import load_secret

    document = load_working(run.id)
    # A fresh analysis sends its own requests. Views made again for it later replay what it stored.
    fresh = allow_live is None and bool((run.config_json or {}).get("fresh"))
    approved = (run.config_json or {}).get("allow_live") if allow_live is None else allow_live
    allow_live = bool(approved) and settings.live_inference
    fresh = fresh and allow_live
    model_id = (run.config_json or {}).get("segmentation_model_id") or _DEFAULT_SAM
    secret, _source = load_secret(session, "meta")
    directory = views_dir(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    # Named by rate, so a copy made at an earlier rate is never read as this one.
    media = _low_rate_copy(resolve_stored(video.stored_path), directory / f"tracked_{SAM_FPS:g}fps.mp4")
    stage.progress_total = 4
    stage.progress_current = 0
    session.commit()
    tracked: dict[str, list[Sighting]] = {}
    seeds, locate_requests, locate_replayed, error = await _locate(session, run, video, document, allow_live, secret, fresh=fresh)
    located = not error
    stage.progress_current = 1
    session.commit()
    text, requests, replayed, failure = await _follow(session, run, video, media, seeds, 0, "both", model_id, allow_live, secret, fresh=fresh)
    error = error or failure
    if text:
        tracked.update(_assign(parse_sam_output(text), seeds))
    stage.progress_current = 2
    session.commit()
    later: list[str] = []
    relocated: list[str] = []
    if located:
        # Without an answer, an empty track says nothing about whether SAM lost its object.
        lost = _lost(document, tracked, seeds, _last_frame_time(video)) if text else {}
        found, later_requests, later_replayed, later_error = await _locate_later(
            session, run, video, document, media, seeds, lost, allow_live, secret, fresh=fresh
        )
        seeds += found
        later = [seed.local_id for seed in found if seed.local_id not in lost]
        relocated = [seed.local_id for seed in found if seed.local_id in lost]
        locate_requests += later_requests
        locate_replayed += later_replayed
        error = error or later_error
        stage.progress_current = 3
        session.commit()
        # A part boxed later is followed forward only, from a copy that starts on its frame. Followed backward,
        # SAM would carry it onto a similar part before it ever came into view, and take that part's pixels.
        start = _first_frame(found)
        text, sent, reused, failure = await _follow(session, run, video, media, found, start, "forward", model_id, allow_live, secret, fresh=fresh)
        requests += sent
        replayed += reused
        error = error or failure
        if text:
            for local_id, sightings in _assign(parse_sam_output(text), found, start).items():
                # A part SAM lost keeps what the first request saw of it, up to where the new box takes over.
                taken = _first(sightings).timestamp
                tracked[local_id] = [item for item in tracked.get(local_id, []) if item.timestamp < taken] + sightings
    stage.progress_current = 4
    session.commit()
    if keep_on_empty and not any(tracked.values()) and read_manifest(run.id):
        return {"error": error or "SAM followed nothing.", "kept_existing": True, "requests": requests, "replayed_requests": replayed}
    checked = _check_appearance(session, run, video, document, media, tracked, seeds)
    _apply_statuses(tracked, checked)
    recovery = await _recover(session, run, video, media, tracked, checked, len(seeds), model_id, allow_live, secret, fresh=fresh)
    requests += recovery.get("requests", 0)
    replayed += recovery.get("replayed_requests", 0)
    _write_raw(run.id, tracked, checked)
    manifest = _render(run.id, document, media, tracked, _last_frame_time(video), checked)
    manifest.update(
        {
            "source": model_id,
            "fps": SAM_FPS,
            "requests": requests,
            "replayed_requests": replayed,
            "locate_requests": locate_requests,
            "locate_replayed": locate_replayed,
            "frames_per_request": estimated_frames(float(video.duration_sec or 0)),
            "seeded_objects": [seed.local_id for seed in seeds],
            "located_later": later,
            "relocated": relocated,
            "error": error,
            "appearance": checked.diagnostics if checked else {"used": False, "error": "Appearance checks are off."},
            "recovery": recovery,
            "gallery": app_.gallery_summary(checked.gallery) if checked and checked.mode == "demonstration" and checked.gallery else (checked.gallery if checked else None),
        }
    )
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    return {key: value for key, value in manifest.items() if key != "objects"}


async def _locate(
    session, run: AnalysisRun, video: VideoAsset, document: AnalysisDocument, allow_live: bool, secret: str, *, fresh: bool = False
) -> tuple[list[Seed], int, int, str]:
    """Boxes for each object on the first frame. A still gives the model sharper pixels than the whole video did."""
    user = _locate_prompt(document)
    key = hashlib.sha256(
        json.dumps({"video": video.checksum_sha256, "model": run.model_id, "system": LOCATE_SYSTEM, "user": user, "schema": GUIDED_LOCATE.hash}, sort_keys=True).encode()
    ).hexdigest()
    text = None if fresh else _cached(key)
    requests, replayed = 0, 0
    if text is not None:
        replayed = 1
    elif allow_live and secret:
        requests = 1
        still = views_dir(run.id) / "first_frame.jpg"
        extract_frame(resolve_stored(video.stored_path), 0.0, still, accurate=True)
        text, error = await _locate_call(session, run, secret, LOCATE_SYSTEM, user, [FrameInput(0.0, still)], GUIDED_LOCATE, "Starting boxes")
        if text is None:
            return [], requests, replayed, error
    else:
        return [], 0, 0, "No stored starting boxes match, and a live call was not approved."
    try:
        answer = GuidedLocateOutput.model_validate_json(text)
    except ValueError as exc:
        return [], requests, replayed, f"The starting boxes were not valid: {str(exc)[:300]}"
    if requests:
        _store(key, text)
    return _seeds(document, answer), requests, replayed, ""


async def _locate_later(
    session,
    run: AnalysisRun,
    video: VideoAsset,
    document: AnalysisDocument,
    media: Path,
    seeds: list[Seed],
    lost: dict[str, float],
    allow_live: bool,
    secret: str,
    *,
    fresh: bool = False,
) -> tuple[list[Seed], int, int, str]:
    """Boxes for objects the first frame did not show, and for objects SAM lost, looked for from the time in lost.
    Sent only when one of those exists, and an object stays blank rather than guessed."""
    seeded = {seed.local_id for seed in seeds}
    after = {item.local_id: _LATER_MERGE for item in document.objects[:MAX_PROMPTED_OBJECTS] if item.local_id not in seeded} | lost
    missing = [item for item in document.objects if item.local_id in after]
    times = _later_times(document, after, _last_frame_time(video), media)
    if not missing or not times:
        return [], 0, 0, ""
    user = _locate_prompt(document.model_copy(update={"objects": missing}))
    key = hashlib.sha256(
        json.dumps(
            {"video": video.checksum_sha256, "model": run.model_id, "system": LATER_SYSTEM, "user": user, "times": times, "schema": GUIDED_LOCATE_LATER.hash},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    text = None if fresh else _cached(key)
    requests, replayed = 0, 0
    if text is not None:
        replayed = 1
    elif allow_live and secret:
        requests = 1
        stills = []
        for index, moment in enumerate(times):
            still = views_dir(run.id) / f"later_{index}.jpg"
            extract_frame(resolve_stored(video.stored_path), moment, still, accurate=True)
            stills.append(FrameInput(moment, still))
        text, error = await _locate_call(session, run, secret, LATER_SYSTEM, user, stills, GUIDED_LOCATE_LATER, "Boxes for later objects")
        if text is None:
            return [], requests, replayed, error
    else:
        return [], 0, 0, "No stored boxes for the objects missing from the first frame match, and a live call was not approved."
    try:
        answer = GuidedLaterLocateOutput.model_validate_json(text)
    except ValueError as exc:
        return [], requests, replayed, f"The boxes for later objects were not valid: {str(exc)[:300]}"
    if requests:
        _store(key, text)
    return _later_seeds(answer, {item.local_id for item in missing}, times, len(seeds)), requests, replayed, ""


def _later_times(document: AnalysisDocument, after: dict[str, float], last_frame: float, media: Path) -> list[float]:
    """Moments the analysis ties to each object from its time in after on, a few per object, each moved to the sharpest nearby frame."""
    per_object = [_object_moments(document, local_id, last_frame, start) for local_id, start in after.items()]
    chosen: list[float] = []
    # Round robin, so one object with many moments does not crowd out the others.
    for rank in range(_LATER_PER_OBJECT):
        for moments in per_object:
            if rank < len(moments) and len(chosen) < _LATER_FRAMES and all(abs(moments[rank] - item) >= _LATER_MERGE for item in chosen):
                chosen.append(moments[rank])
    return sorted(_sharpest_nearby(chosen, media))


def _object_moments(document: AnalysisDocument, local_id: str, last_frame: float, after: float = _LATER_MERGE) -> list[float]:
    moments = [row.timestamp for row in document.observations if row.object_local_id == local_id and enum_value(row.visibility) in {"visible", "partial"}]
    for action in document.actions:
        if local_id in action.object_local_ids:
            moments += [action.start, action.end]
    for step in document.workflow_steps:
        if local_id in step.required_objects:
            moments += [value for value in (step.time_start, step.time_end) if value is not None]
    moments += [row.timestamp for row in document.transitions if row.object_local_id == local_id]
    # The first frame, or the stretch SAM followed, was already searched.
    kept: list[float] = []
    for moment in sorted(min(max(float(value), 0.0), last_frame) for value in moments):
        if moment >= after and all(abs(moment - item) >= _LATER_MERGE for item in kept):
            kept.append(moment)
    return kept


def _sharpest_nearby(moments: list[float], media: Path) -> list[float]:
    """A hand in motion blurs a frame. The sharpest frame a moment ago or later shows the same scene."""
    reach = round(_SHARPEST_WITHIN * SAM_FPS)
    windows = {moment: range(max(round(moment * SAM_FPS) - reach, 0), round(moment * SAM_FPS) + reach + 1) for moment in moments}
    frames = _read_frames(media, {index for window in windows.values() for index in window})
    picked = []
    for moment, window in windows.items():
        options = [index for index in window if index in frames]
        if not options:
            continue
        # Ties go to the frame nearest the moment.
        best = max(options, key=lambda index: (sharpness_score(frames[index]), -abs(index - moment * SAM_FPS)))
        picked.append(round(best / SAM_FPS, 3))
    return list(dict.fromkeys(picked))


def _later_seeds(answer: GuidedLaterLocateOutput, missing: set[str], times: list[float], first_id: int) -> list[Seed]:
    """Only a box on a still that was sent counts, so SAM gets the frame the box was drawn on."""
    seeds: list[Seed] = []
    for item in answer.objects:
        if item.local_id not in missing or item.timestamp is None or any(seed.local_id == item.local_id for seed in seeds):
            continue
        moment = min(times, key=lambda value: abs(value - item.timestamp))
        box = _usable_box(item.bbox)
        if box is None or abs(moment - item.timestamp) > 0.5 / SAM_FPS:
            continue
        seeds.append(Seed(item.local_id, first_id + len(seeds), moment, box))
    return seeds


def _usable_box(box) -> dict | None:
    if box is None:
        return None
    x, y = max(box.x, 0.0), max(box.y, 0.0)
    w, h = min(box.w, 1 - x), min(box.h, 1 - y)
    if w <= 0 or h <= 0 or w * h > _MAX_SEED_AREA:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _locate_prompt(document: AnalysisDocument) -> str:
    entries = [f"- {item.local_id}: {item.label}. {item.visual_characteristics or item.description or ''}".strip() for item in document.objects]
    return "Objects to find:\n" + "\n".join(entries)


async def _locate_call(
    session, run: AnalysisRun, secret: str, system: str, user: str, stills: list[FrameInput], contract, what: str
) -> tuple[str | None, str]:
    from app.providers.registry import get_provider
    from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call

    request = GenerationRequest(
        model_id=run.model_id,
        system_prompt=system,
        user_prompt=user,
        frames=stills,
        modality="sampled_frames",
        json_schema=contract.schema,
        schema_name=contract.name,
        max_output_tokens=_LOCATE_OUTPUT_TOKENS,
    )
    call = open_provider_call(
        session,
        run,
        stage="locate",
        provider_id="meta",
        model_id=run.model_id,
        modality="sampled_frames",
        parameters={"schema_name": contract.name, "strict_output": True, "output_token_limit": _LOCATE_OUTPUT_TOKENS},
        frame_timestamps=[still.timestamp for still in stills],
    )
    session.commit()
    try:
        response = await get_provider("meta", session).generate(secret, request)
    except Exception as exc:
        log.warning("Locating objects failed: %s", exc)
        fail_provider_call(call, exc)
        session.commit()
        return None, str(exc)[:400]
    finish_provider_call(session, run, call, response)
    session.commit()
    if response.incomplete_reason:
        return None, f"{what} stopped early ({response.incomplete_reason})."
    return response.text, ""


def _seeds(document: AnalysisDocument, answer: GuidedLocateOutput) -> list[Seed]:
    """The boxes are the model's estimate, so SAM may still follow the wrong thing."""
    known = {item.local_id for item in document.objects}
    seeds: list[Seed] = []
    taken: set[str] = set()
    for item in answer.objects[:MAX_PROMPTED_OBJECTS]:
        box = _usable_box(item.bbox)
        if item.local_id not in known or item.local_id in taken or box is None:
            continue
        taken.add(item.local_id)
        seeds.append(Seed(item.local_id, len(seeds), 0.0, box))
    return seeds


def _lost(document: AnalysisDocument, tracked: dict[str, list[Sighting]], seeds: list[Seed], last_frame: float) -> dict[str, float]:
    """Boxed objects whose track ends well before a moment the analysis ties them to, each with the time to look again from.
    SAM rarely finds a part again once its memory of it is gone, so another box is the way back."""
    lost = {}
    for seed in seeds:
        after = max((item.timestamp for item in tracked.get(seed.local_id, [])), default=seed.timestamp) + _LOST_AFTER
        if _object_moments(document, seed.local_id, last_frame, after):
            lost[seed.local_id] = after
    return lost


def _first_frame(seeds: list[Seed]) -> int:
    return min((int(round(seed.timestamp * SAM_FPS)) for seed in seeds), default=0)


def _prompt(seeds: list[Seed], media: Path, start: int = 0) -> str:
    """Frame numbers count from the first frame of the copy SAM is sent, which is frame start of the recording."""
    info = probe(media)
    width, height = info.display_width, info.display_height
    last = max(info.frame_count - 1, 0)
    marks = []
    for seed in seeds:
        x1 = min(max(int(seed.box["x"] * width), 0), width - 1)
        y1 = min(max(int(seed.box["y"] * height), 0), height - 1)
        x2 = min(int((seed.box["x"] + seed.box["w"]) * width), width - 1)
        y2 = min(int((seed.box["y"] + seed.box["h"]) * height), height - 1)
        frame = min(max(int(round(seed.timestamp * SAM_FPS)) - start, 0), last)
        marks.append(SpatialMark(frame, seed.object_id, "box", width, height, x1, y1, max(x2, x1), max(y2, y1)))
    return render_spatial_prompt(marks)


async def _follow(
    session, run, video: VideoAsset, media: Path, seeds: list[Seed], start: int, direction: str, model_id: str, allow_live: bool, secret: str, *, fresh: bool = False
) -> tuple[str | None, int, int, str]:
    """One SAM request over the copy from frame start on, as (answer, requests sent, requests replayed, error)."""
    if not seeds:
        return None, 0, 0, ""
    clip = _clip_from(media, start)
    prompt = _prompt(seeds, clip, start)
    key = _cache_key(video.checksum_sha256, prompt, model_id, start, direction)
    text = None if fresh else _cached(key)
    if text is not None:
        return text, 0, 1, ""
    if not (allow_live and secret):
        return None, 0, 0, "No stored SAM answer matches, and a live call was not approved."
    text, failure = await _call(session, run, clip, prompt, secret, model_id, direction)
    # An empty answer may be the prompt or the footage, so a later run may try again.
    if text:
        _store(key, text)
    return text, 1, 0, failure


async def _call(session, run, media: Path, prompt: str, secret: str, model_id: str, direction: str) -> tuple[str | None, str]:
    from app.pipeline.sam_stage import _finish
    from app.services.costs import fail_provider_call, open_provider_call

    call = open_provider_call(
        session,
        run,
        stage="segmentation",
        provider_id="meta",
        model_id=model_id,
        modality="native_video",
        parameters={"prompt_kind": "spatial", "prompt": prompt, "fps": SAM_FPS, "propagation_direction": direction},
    )
    session.commit()
    try:
        result = await SamAdapter(secret, model_id=model_id).segment_video(media, prompt, prompt_kind="spatial", propagation_direction=direction)
    except Exception as exc:
        log.warning("SAM failed: %s", exc)
        fail_provider_call(call, exc)
        session.commit()
        return None, str(exc)[:400]
    _finish(session, run, call, result, model_id)
    session.commit()
    return result.text, ""


def _assign(records: list[SamRecord], seeds: list[Seed], start: int = 0) -> dict[str, list[Sighting]]:
    """SAM answers with the id each box was prompted under, and frame numbers from the start of the copy it was sent."""
    owners = {str(seed.object_id): seed.local_id for seed in seeds}
    tracks: dict[str, list[Sighting]] = {}
    for record in records:
        box = normalized_box(record)
        owner = owners.get(record.track_id)
        if owner and box is not None:
            record.frame_index += start
            tracks.setdefault(owner, []).append(Sighting(record.frame_index / SAM_FPS, box, record))
    return tracks


def _check_appearance(session, run: AnalysisRun, video: VideoAsset, document: AnalysisDocument, media: Path, tracked, seeds):
    """Statuses for every SAM box. None when appearance checks are off; a result with used False when they failed."""
    from app.pipeline.appearance_stage import AppearanceResult, check_tracks

    if not settings.appearance_enabled or not any(tracked.values()):
        return None
    labels = {item.local_id: item.label for item in document.objects}
    template_id = (run.config_json or {}).get("reference_template_id") or ""
    try:
        if not template_id:
            if not settings.auto_gallery:
                return None
            return check_tracks(
                run_id=run.id, media=media, tracked=tracked, seeds=seeds, labels=labels, fps=SAM_FPS, mode="demonstration",
                source={"video_id": video.id, "checksum": video.checksum_sha256},
            )
        reference = execution_reference(session, template_id, document)
        return check_tracks(
            run_id=run.id, media=media, tracked=tracked, seeds=seeds, labels=labels, fps=SAM_FPS, mode="execution", reference=reference
        )
    except Exception as exc:
        # Appearance is an extra check. Its failure leaves SAM's boxes, shown as unchecked.
        log.exception("Appearance checks failed for run %s", run.id)
        return AppearanceResult(mode="execution" if template_id else "demonstration", error=str(exc)[:400], diagnostics={"used": False, "error": str(exc)[:400]})


def execution_reference(session, template_id: str, document: AnalysisDocument):
    """The learned procedure's references, keyed by the objects of this recording that its roles map to.
    A role with no mapped object stays in as a rival under its role id. Nothing here is written back."""
    from app.models.verification import LearnedWorkflowTemplate
    from app.verification.roles import map_roles
    from app.verification.schema import LearnedTemplate

    template = session.get(LearnedWorkflowTemplate, template_id)
    if template is None:
        return None, {}, {}
    manifest, by_role = app_.role_galleries(template.document_json or {}, template.source_run_id)
    if manifest is None or not by_role:
        return manifest, {}, {}
    spec = LearnedTemplate.model_validate(template.document_json)
    galleries: dict = {}
    labels: dict[str, str] = {}
    for assignment in map_roles(spec, document):
        refs = by_role.get(assignment.role_id)
        if refs is None:
            continue
        if assignment.status in {"assigned", "human"} and len(assignment.object_local_ids) == 1:
            galleries[assignment.object_local_ids[0]] = refs
        else:
            galleries[assignment.role_id] = refs
            labels[assignment.role_id] = assignment.label
    return manifest, galleries, labels


def _apply_statuses(tracked: dict[str, list[Sighting]], checked) -> None:
    for local_id, sightings in tracked.items():
        ordered = sorted(sightings, key=lambda item: item.timestamp)
        statuses = (checked.statuses.get(local_id) if checked and checked.used else None) or [("provisional", "unchecked")] * len(ordered)
        for item, (status, reason) in zip(ordered, statuses, strict=True):
            item.status, item.reason = status, reason


async def _recover(session, run, video, media: Path, tracked, checked, next_id: int, model_id, allow_live, secret, *, fresh=False) -> dict:
    """Look for an object where its track was rejected, and follow a clear match with one more SAM request.
    The new boxes are checked against the references again, and replace only boxes that were not validated."""
    from app.pipeline.appearance_stage import check_tracks, propose_recovery

    if not (checked and checked.used and settings.appearance_recovery and checked.diagnostics.get("rejected_intervals")):
        return {"attempted": False}
    try:
        attempts = propose_recovery(media, tracked, checked)
    except Exception as exc:
        log.exception("Recovery search failed for run %s", run.id)
        return {"attempted": True, "error": str(exc)[:400], "attempts": []}
    chosen = [item for item in attempts if item.get("decision") == "reseed"]
    out = {"attempted": True, "attempts": attempts, "requests": 0, "replayed_requests": 0, "recovered_sightings": 0, "sam_budget": settings.recovery_max_sam_requests}
    if not chosen or settings.recovery_max_sam_requests <= 0:
        out["unresolved"] = len(attempts)
        return out
    seeds = [Seed(item["local_id"], next_id + index, item["frame_index"] / SAM_FPS, item["best"]["box"]) for index, item in enumerate(chosen)]
    start = _first_frame(seeds)
    text, sent, reused, failure = await _follow(session, run, video, media, seeds, start, "forward", model_id, allow_live, secret, fresh=fresh)
    out.update(requests=sent, replayed_requests=reused, error=failure)
    if not text:
        for item in chosen:
            item["decision"], item["reason"] = "unresolved", f"sam_not_run: {failure or 'no answer'}"
        out["unresolved"] = len(attempts)
        return out
    followed = _assign(parse_sam_output(text), seeds, start)
    # A recovery is accepted only if the boxes SAM followed from the new seed look like the object.
    again = check_tracks(
        run_id=run.id, media=media, tracked=followed, seeds=seeds, labels=checked.labels, fps=SAM_FPS, mode="execution",
        reference=(None, checked.galleries, {}),
    )
    for item in chosen:
        local_id = item["local_id"]
        new = sorted(followed.get(local_id, []), key=lambda sighting: sighting.timestamp)
        statuses = again.statuses.get(local_id) or []
        kept = 0
        existing = {sighting.record.frame_index: sighting for sighting in tracked.get(local_id, [])}
        for sighting, (status, reason) in zip(new, statuses, strict=True):
            if status == "rejected":
                # The new seed drifted too. Nothing after this is trusted.
                break
            if status != "validated":
                continue
            current = existing.get(sighting.record.frame_index)
            if current is not None and current.status == "validated":
                continue
            if current is not None:
                current.superseded = True
            sighting.status, sighting.reason, sighting.provenance = "recovered", "reseeded_and_validated", "recovery"
            tracked.setdefault(local_id, []).append(sighting)
            kept += 1
        item["decision"] = "recovered" if kept else "unresolved"
        item["reason"] = "reseeded_and_validated" if kept else "followed_boxes_did_not_validate"
        item["recovered_sightings"] = kept
        out["recovered_sightings"] += kept
    out["unresolved"] = sum(item.get("decision") != "recovered" for item in attempts)
    return out


def _write_raw(run_id: str, tracked: dict[str, list[Sighting]], checked) -> None:
    """Every SAM box as it came back, with the status the appearance check gave it. Nothing here is drawn."""
    out = {}
    for local_id, sightings in tracked.items():
        ordered = sorted(sightings, key=lambda item: item.timestamp)
        out[local_id] = [
            {
                "time": round(item.timestamp, 3),
                "frame_index": item.record.frame_index,
                "sam_track_id": item.record.track_id,
                "confidence": item.record.confidence,
                "box": {key: round(value, 4) for key, value in item.box.items()},
                "status": item.status,
                "reason": item.reason,
                "provenance": item.provenance,
                "superseded": item.superseded,
            }
            for item in ordered
        ]
    samples = {}
    if checked and checked.used:
        samples = {local_id: rows for local_id, rows in checked.rows.items()}
    (views_dir(run_id) / "raw_tracks.json").write_text(json.dumps({"sightings": out, "appearance_samples": samples}))


_STATUS_RANK = {"validated": 3, "recovered": 3, "ambiguous": 2, "provisional": 1}


def _shown(sightings: list[Sighting], checked, local_id: str) -> list[Sighting]:
    """Boxes drawn as the object. A rejected box, or one a recovered box replaced, is kept in raw_tracks.json, not here."""
    return [item for item in sorted(sightings, key=lambda item: item.timestamp) if item.status in app_.SHOWN and not item.superseded]


def _render(run_id: str, document: AnalysisDocument, media: Path, tracked: dict[str, list[Sighting]], last_frame: float, checked=None) -> dict:
    directory = views_dir(run_id)
    times = _cell_times(last_frame)
    tracked = {local_id: _shown(sightings, checked, local_id) for local_id, sightings in tracked.items()}
    picks = {local_id: _picks(sightings, times) for local_id, sightings in tracked.items()}
    wanted = set()
    for local_id, sightings in tracked.items():
        wanted.add(_first(sightings).record.frame_index)
        wanted.update(item.record.frame_index for item in picks[local_id] if item)
    frames = _read_frames(media, wanted)
    objects = {}
    for position, item in enumerate(document.objects):
        sightings = tracked.get(item.local_id, [])
        chosen = picks.get(item.local_id) or [None] * len(times)
        cells = []
        for index, (moment, sighting) in enumerate(zip(times, chosen)):
            image = None
            confidence = None
            frame_time = None
            if sighting is not None:
                name = f"o{position}_c{index}.jpg"
                frame = frames.get(sighting.record.frame_index)
                if frame is not None and cv2.imwrite(str(directory / name), _crop(frame, sighting.box)):
                    image = name
                    confidence = sighting.record.confidence
                    frame_time = round(sighting.timestamp, 3)
            cells.append({"time": round(moment, 3), "frame_time": frame_time, "image": image, "confidence": confidence, "status": sighting.status if image else None})
        objects[item.local_id] = {
            "thumbnail": _thumbnail(directory, position, sightings, frames),
            "cells": cells,
            "seen_cells": sum(cell["image"] is not None for cell in cells),
            "first_seen": round(_first(sightings).timestamp, 3) if sightings else None,
            "track": [
                {"time": round(item.timestamp, 3), **{key: round(value, 4) for key, value in item.box.items()}, "status": item.status}
                for item in sorted(sightings, key=lambda item: item.timestamp)
            ],
        }
    return {"objects": objects}


def _cell_times(last_frame: float) -> list[float]:
    """Every interval, then the last frame. A mark closer to the last frame than the shortest allowed gap is dropped,
    since its frame could not sit far enough from the last one."""
    shortest = (1 - _CELL_SPREAD) * _CELL_INTERVAL
    return [step * _CELL_INTERVAL for step in range(math.ceil((last_frame - shortest) / _CELL_INTERVAL))] + [last_frame]


def _last_frame_time(video: VideoAsset) -> float:
    """The container's duration runs one frame past the start of the last frame."""
    duration = float(video.duration_sec or 0)
    return max(duration - 1 / video.fps, 0.0) if video.fps else duration


def _picks(sightings: list[Sighting], times: list[float]) -> list[Sighting | None]:
    """The most confident sighting in each cell's interval, one interval, give or take _CELL_SPREAD of one, after the frame
    shown before it. The last cell is the end of the recording and always shows it, so the cell before it leaves room."""
    picks: list[Sighting | None] = []
    previous: float | None = None
    for index, moment in enumerate(times):
        if index == len(times) - 1:
            pick = _nearest(sightings, moment)
        else:
            allowed = [item for item in sightings if _spaced(item.timestamp, previous)]
            end = times[index + 1]
            if index == len(times) - 2:
                end = min(end, times[-1] - (1 - _CELL_SPREAD) * _CELL_INTERVAL + 1e-9)
            within = [item for item in allowed if moment <= item.timestamp < end]
            # A box the appearance check agreed with comes before one it could not check, then SAM's confidence.
            pick = max(within, key=lambda item: (_STATUS_RANK.get(item.status, 0), item.record.confidence or 0, -item.timestamp), default=None)
        picks.append(pick)
        # An empty cell breaks the chain, so the next one is held only to its own interval.
        previous = pick.timestamp if pick else None
    return picks


def _spaced(timestamp: float, previous: float | None) -> bool:
    if previous is None:
        return True
    gap = timestamp - previous
    return (1 - _CELL_SPREAD) * _CELL_INTERVAL - 1e-9 <= gap <= (1 + _CELL_SPREAD) * _CELL_INTERVAL + 1e-9


def _nearest(sightings: list[Sighting], moment: float) -> Sighting | None:
    if not sightings:
        return None
    best = min(sightings, key=lambda item: abs(item.timestamp - moment))
    # Two sample gaps with no sighting means the object was not found at this moment.
    return best if abs(best.timestamp - moment) <= 2 / SAM_FPS else None


def _thumbnail(directory: Path, position: int, sightings: list[Sighting], frames: dict) -> str | None:
    """The first view of the object, as a cutout when the mask decodes and otherwise a rectangle."""
    if not sightings or _first(sightings).record.frame_index not in frames:
        return None
    first = _first(sightings)
    frame = frames[first.record.frame_index]
    raster = decode_mask_raster(first.record)
    if raster is not None:
        cutout = cutout_from_mask(frame, raster, first.record.mask_height, first.record.mask_width, first.box)
        if cutout is not None:
            name = f"o{position}_thumb.png"
            if cv2.imwrite(str(directory / name), _fit(cutout)):
                return name
    name = f"o{position}_thumb.jpg"
    return name if cv2.imwrite(str(directory / name), _crop(frame, first.box)) else None


def _first(sightings: list[Sighting]) -> Sighting:
    return min(sightings, key=lambda item: item.timestamp)


def _crop(frame, box: dict):
    height, width = frame.shape[:2]
    pad_x, pad_y = box["w"] * _PADDING, box["h"] * _PADDING
    x0 = max(int((box["x"] - pad_x) * width), 0)
    y0 = max(int((box["y"] - pad_y) * height), 0)
    x1 = min(int((box["x"] + box["w"] + pad_x) * width) + 1, width)
    y1 = min(int((box["y"] + box["h"] + pad_y) * height) + 1, height)
    return _fit(frame[y0:y1, x0:x1])


def _fit(image):
    scale = _CROP_SIDE / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _read_frames(media: Path, wanted: set[int]) -> dict:
    capture = cv2.VideoCapture(str(media))
    frames = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames
            if index in wanted:
                frames[index] = frame
            index += 1
    finally:
        capture.release()


def _low_rate_copy(source: Path, dest: Path) -> Path:
    """Frame k shows time k / SAM_FPS, upright, because ffmpeg applies the rotation tag."""
    if dest.exists():
        return dest
    scale = f"scale='if(gt(iw,ih),min({_LONG_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({_LONG_SIDE},ih))'"
    tmp = dest.with_name(f".{dest.stem}.tmp{dest.suffix}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-an", "-vf", f"fps={SAM_FPS},{scale}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)],
        capture_output=True,
        check=True,
    )
    tmp.replace(dest)
    return dest


def _clip_from(media: Path, start: int) -> Path:
    """The low frame rate copy from frame start on, so a request is billed only for frames it can follow."""
    if start <= 0:
        return media
    dest = media.with_name(f"{media.stem}_from_{start}{media.suffix}")
    if dest.exists():
        return dest
    tmp = dest.with_name(f".{dest.stem}.tmp{dest.suffix}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(media), "-an", "-vf", f"trim=start_frame={start},setpts=PTS-STARTPTS", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)],
        capture_output=True,
        check=True,
    )
    tmp.replace(dest)
    return dest


def _cache_key(checksum: str, prompt: str, model_id: str = _DEFAULT_SAM, start: int = 0, direction: str = "both") -> str:
    fields = {"video": checksum, "prompt": prompt, "fps": SAM_FPS, "side": _LONG_SIDE}
    # The first-frame request keeps the key it had when it was the only request.
    if start or direction != "both":
        fields.update({"start_frame": start, "direction": direction})
    # Answers stored before models could be chosen are all from the default, so its key stays the same.
    if model_id != _DEFAULT_SAM:
        fields["model"] = model_id
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _cached(key: str) -> str | None:
    try:
        return json.loads(data_path("inference_cache", "sam", f"{key}.json").read_text())["text"] or None
    except (OSError, ValueError, KeyError):
        return None


def _store(key: str, text: str) -> None:
    path = data_path("inference_cache", "sam", f"{key}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text}))
