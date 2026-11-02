"""Object thumbnails and per-object timelines for a demonstration.

SAM follows every object's first box from the analysis through a low frame rate
copy of the recording, in one request. This is for looking at, not for judging:
nothing here enters the analysis or a verdict. A failure leaves the analysis in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config import settings
from app.models.media import VideoAsset
from app.models.runs import AnalysisRun
from app.perception.sam_masks import cutout_from_mask
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
from app.schemas.analysis import AnalysisDocument
from app.services.storage import data_path, resolve_stored
from app.video.probe import probe

log = logging.getLogger("learnloop.object_views")

SAM_FPS = 8.0
TIMELINE_CELLS = 24
_LONG_SIDE = 1080
_CROP_SIDE = 160
_PADDING = 0.15
_MANIFEST = "manifest.json"


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


async def run_object_views(session, run: AnalysisRun, video: VideoAsset, stage) -> dict:
    from app.pipeline.infer import load_working
    from app.services.credentials import load_secret

    document = load_working(run.id)
    allow_live = bool((run.config_json or {}).get("allow_live")) and settings.live_inference
    secret, _source = load_secret(session, "meta")
    directory = views_dir(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    media = _low_rate_copy(resolve_stored(video.stored_path), directory / "tracked.mp4")
    seeds = _seeds(document)
    stage.progress_total = 1
    stage.progress_current = 0
    session.commit()
    tracked: dict[str, list[Sighting]] = {}
    requests = 0
    replayed = 0
    error = ""
    if seeds:
        prompt = _prompt(seeds, media)
        key = _cache_key(video.checksum_sha256, prompt)
        text = _cached(key)
        if text is not None:
            replayed = 1
        elif allow_live and secret:
            requests = 1
            text, error = await _call(session, run, media, prompt, secret)
            # An empty answer may be the prompt or the footage, so a later run may try again.
            if text:
                _store(key, text)
        else:
            error = "No stored SAM answer matches, and a live call was not approved."
        if text:
            tracked = _assign(parse_sam_output(text), seeds)
    stage.progress_current = 1
    session.commit()
    manifest = _render(run.id, document, media, tracked, float(video.duration_sec or 0))
    manifest.update(
        {
            "source": "sam-3.1",
            "fps": SAM_FPS,
            "requests": requests,
            "replayed_requests": replayed,
            "frames_per_request": estimated_frames(float(video.duration_sec or 0)),
            "seeded_objects": [seed.local_id for seed in seeds],
            "error": error,
        }
    )
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    return {key: value for key, value in manifest.items() if key != "objects"}


def _seeds(document: AnalysisDocument) -> list[Seed]:
    """Each object's earliest box from the analysis. The box is the model's estimate, so SAM may follow the wrong thing."""
    seeds = []
    for item in document.objects[:MAX_PROMPTED_OBJECTS]:
        boxed = [row for row in document.observations if row.object_local_id == item.local_id and row.bbox is not None]
        if boxed:
            first = min(boxed, key=lambda row: row.timestamp)
            seeds.append(Seed(item.local_id, len(seeds), first.timestamp, first.bbox.model_dump()))
    return seeds


def _prompt(seeds: list[Seed], media: Path) -> str:
    info = probe(media)
    width, height = info.display_width, info.display_height
    last = max(info.frame_count - 1, 0)
    marks = []
    for seed in seeds:
        x1 = min(max(int(seed.box["x"] * width), 0), width - 1)
        y1 = min(max(int(seed.box["y"] * height), 0), height - 1)
        x2 = min(int((seed.box["x"] + seed.box["w"]) * width), width - 1)
        y2 = min(int((seed.box["y"] + seed.box["h"]) * height), height - 1)
        frame = min(int(round(seed.timestamp * SAM_FPS)), last)
        marks.append(SpatialMark(frame, seed.object_id, "box", width, height, x1, y1, max(x2, x1), max(y2, y1)))
    return render_spatial_prompt(marks)


async def _call(session, run, media: Path, prompt: str, secret: str) -> tuple[str | None, str]:
    from app.pipeline.sam_stage import _finish
    from app.services.costs import fail_provider_call, open_provider_call

    call = open_provider_call(
        session,
        run,
        stage="segmentation",
        provider_id="meta",
        model_id="sam-3.1",
        modality="native_video",
        parameters={"prompt_kind": "spatial", "prompt": prompt, "fps": SAM_FPS},
    )
    session.commit()
    try:
        result = await SamAdapter(secret).segment_video(media, prompt, prompt_kind="spatial", propagation_direction="both")
    except Exception as exc:
        log.warning("SAM failed: %s", exc)
        fail_provider_call(call, exc)
        session.commit()
        return None, str(exc)[:400]
    _finish(session, run, call, result)
    session.commit()
    return result.text, ""


def _assign(records: list[SamRecord], seeds: list[Seed]) -> dict[str, list[Sighting]]:
    """SAM answers with the id each box was prompted under."""
    owners = {str(seed.object_id): seed.local_id for seed in seeds}
    tracks: dict[str, list[Sighting]] = {}
    for record in records:
        box = normalized_box(record)
        owner = owners.get(record.track_id)
        if owner and box is not None:
            tracks.setdefault(owner, []).append(Sighting(record.frame_index / SAM_FPS, box, record))
    return tracks


def _render(run_id: str, document: AnalysisDocument, media: Path, tracked: dict[str, list[Sighting]], duration: float) -> dict:
    directory = views_dir(run_id)
    times = [duration * (index + 0.5) / TIMELINE_CELLS for index in range(TIMELINE_CELLS)]
    wanted = set()
    for sightings in tracked.values():
        wanted.add(_first(sightings).record.frame_index)
        wanted.update(item.record.frame_index for item in (_nearest(sightings, moment) for moment in times) if item)
    frames = _read_frames(media, wanted)
    objects = {}
    for position, item in enumerate(document.objects):
        sightings = tracked.get(item.local_id, [])
        cells = []
        for index, moment in enumerate(times):
            sighting = _nearest(sightings, moment)
            image = None
            if sighting is not None:
                name = f"o{position}_c{index}.jpg"
                frame = frames.get(sighting.record.frame_index)
                if frame is not None and cv2.imwrite(str(directory / name), _crop(frame, sighting.box)):
                    image = name
            cells.append({"time": round(moment, 3), "image": image})
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


def _cache_key(checksum: str, prompt: str) -> str:
    parts = json.dumps({"video": checksum, "prompt": prompt, "fps": SAM_FPS, "side": _LONG_SIDE}, sort_keys=True)
    return hashlib.sha256(parts.encode()).hexdigest()


def _cached(key: str) -> str | None:
    try:
        return json.loads(data_path("inference_cache", "sam", f"{key}.json").read_text())["text"] or None
    except (OSError, ValueError, KeyError):
        return None


def _store(key: str, text: str) -> None:
    path = data_path("inference_cache", "sam", f"{key}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text}))
