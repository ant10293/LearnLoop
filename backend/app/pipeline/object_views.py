"""Object thumbnails and per-object timelines for a demonstration.

One Meta call finds each object on the first frame. When some object is not
there, a second call looks at a few later stills chosen from the analysis. SAM
then follows every box, whatever frame it is on, through a low frame rate copy
of the recording, in one request. This is for looking at, not for judging:
nothing here enters the analysis or a verdict. A failure leaves the analysis in place.
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

SAM_FPS = 8.0
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


async def run_object_views(session, run: AnalysisRun, video: VideoAsset, stage, allow_live: bool | None = None) -> dict:
    """allow_live, when given, replaces the run's own approval. A person approves it again to make views for a finished run."""
    from app.pipeline.infer import load_working
    from app.services.credentials import load_secret

    document = load_working(run.id)
    approved = (run.config_json or {}).get("allow_live") if allow_live is None else allow_live
    allow_live = bool(approved) and settings.live_inference
    model_id = (run.config_json or {}).get("segmentation_model_id") or _DEFAULT_SAM
    secret, _source = load_secret(session, "meta")
    directory = views_dir(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    media = _low_rate_copy(resolve_stored(video.stored_path), directory / "tracked.mp4")
    stage.progress_total = 3
    stage.progress_current = 0
    session.commit()
    tracked: dict[str, list[Sighting]] = {}
    requests = 0
    replayed = 0
    seeds, locate_requests, locate_replayed, error = await _locate(session, run, video, document, allow_live, secret)
    stage.progress_current = 1
    session.commit()
    later: list[str] = []
    if not error:
        found, later_requests, later_replayed, later_error = await _locate_later(
            session, run, video, document, media, seeds, allow_live, secret
        )
        seeds += found
        later = [seed.local_id for seed in found]
        locate_requests += later_requests
        locate_replayed += later_replayed
        error = later_error
    stage.progress_current = 2
    session.commit()
    # A part boxed later is followed forward only, from a copy that starts on its frame. Followed backward,
    # SAM would carry it onto a similar part before it ever came into view, and take that part's pixels.
    later_seeds = [seed for seed in seeds if seed.timestamp > 0]
    groups = [
        (group, start, direction)
        for group, start, direction in (
            ([seed for seed in seeds if seed.timestamp == 0], 0, "both"),
            (later_seeds, _first_frame(later_seeds), "forward"),
        )
        if group
    ]
    for group, start, direction in groups:
        clip = _clip_from(media, start)
        prompt = _prompt(group, clip, start)
        key = _cache_key(video.checksum_sha256, prompt, model_id, start, direction)
        text = _cached(key)
        if text is not None:
            replayed += 1
        elif allow_live and secret:
            requests += 1
            text, failure = await _call(session, run, clip, prompt, secret, model_id, direction)
            error = error or failure
            # An empty answer may be the prompt or the footage, so a later run may try again.
            if text:
                _store(key, text)
        else:
            error = error or "No stored SAM answer matches, and a live call was not approved."
        if text:
            tracked.update(_assign(parse_sam_output(text), group, start))
    stage.progress_current = 3
    session.commit()
    manifest = _render(run.id, document, media, tracked, _last_frame_time(video))
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
            "error": error,
        }
    )
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    return {key: value for key, value in manifest.items() if key != "objects"}


async def _locate(session, run: AnalysisRun, video: VideoAsset, document: AnalysisDocument, allow_live: bool, secret: str) -> tuple[list[Seed], int, int, str]:
    """Boxes for each object on the first frame. A still gives the model sharper pixels than the whole video did."""
    user = _locate_prompt(document)
    key = hashlib.sha256(
        json.dumps({"video": video.checksum_sha256, "model": run.model_id, "system": LOCATE_SYSTEM, "user": user, "schema": GUIDED_LOCATE.hash}, sort_keys=True).encode()
    ).hexdigest()
    text = _cached(key)
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
    session, run: AnalysisRun, video: VideoAsset, document: AnalysisDocument, media: Path, seeds: list[Seed], allow_live: bool, secret: str
) -> tuple[list[Seed], int, int, str]:
    """Boxes for objects the first frame did not show. Sent only when one is missing, and an object stays blank rather than guessed."""
    seeded = {seed.local_id for seed in seeds}
    missing = [item for item in document.objects[:MAX_PROMPTED_OBJECTS] if item.local_id not in seeded]
    times = _later_times(document, [item.local_id for item in missing], _last_frame_time(video), media)
    if not missing or not times:
        return [], 0, 0, ""
    user = _locate_prompt(document.model_copy(update={"objects": missing}))
    key = hashlib.sha256(
        json.dumps(
            {"video": video.checksum_sha256, "model": run.model_id, "system": LATER_SYSTEM, "user": user, "times": times, "schema": GUIDED_LOCATE_LATER.hash},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    text = _cached(key)
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


def _later_times(document: AnalysisDocument, local_ids: list[str], last_frame: float, media: Path) -> list[float]:
    """Moments the analysis ties to each missing object, a few per object, each moved to the sharpest nearby frame."""
    per_object = [_object_moments(document, local_id, last_frame) for local_id in local_ids]
    chosen: list[float] = []
    # Round robin, so one object with many moments does not crowd out the others.
    for rank in range(_LATER_PER_OBJECT):
        for moments in per_object:
            if rank < len(moments) and len(chosen) < _LATER_FRAMES and all(abs(moments[rank] - item) >= _LATER_MERGE for item in chosen):
                chosen.append(moments[rank])
    return sorted(_sharpest_nearby(chosen, media))


def _object_moments(document: AnalysisDocument, local_id: str, last_frame: float) -> list[float]:
    moments = [row.timestamp for row in document.observations if row.object_local_id == local_id and enum_value(row.visibility) in {"visible", "partial"}]
    for action in document.actions:
        if local_id in action.object_local_ids:
            moments += [action.start, action.end]
    for step in document.workflow_steps:
        if local_id in step.required_objects:
            moments += [value for value in (step.time_start, step.time_end) if value is not None]
    moments += [row.timestamp for row in document.transitions if row.object_local_id == local_id]
    # The first frame was already searched.
    kept: list[float] = []
    for moment in sorted(min(max(float(value), 0.0), last_frame) for value in moments):
        if moment >= _LATER_MERGE and all(abs(moment - item) >= _LATER_MERGE for item in kept):
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


def _render(run_id: str, document: AnalysisDocument, media: Path, tracked: dict[str, list[Sighting]], last_frame: float) -> dict:
    directory = views_dir(run_id)
    times = _cell_times(last_frame)
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
            cells.append({"time": round(moment, 3), "frame_time": frame_time, "image": image, "confidence": confidence})
        objects[item.local_id] = {
            "thumbnail": _thumbnail(directory, position, sightings, frames),
            "cells": cells,
            "seen_cells": sum(cell["image"] is not None for cell in cells),
            "first_seen": round(_first(sightings).timestamp, 3) if sightings else None,
            "track": [
                {"time": round(item.timestamp, 3), **{key: round(value, 4) for key, value in item.box.items()}}
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
            pick = max(within, key=lambda item: (item.record.confidence or 0, -item.timestamp), default=None)
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
