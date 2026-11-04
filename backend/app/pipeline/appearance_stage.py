"""Check SAM sightings against appearance references, inside the object-views pass.

On a demonstration the references are learned here, from the same tracks, and
saved with the run. On an attempt checked against a learned procedure they are
read from that procedure's demonstration and never changed: footage of an attempt
may show the wrong part, so it is not learned from.

Raw SAM boxes are kept. Each one gets a status and a reason, and only the shown
statuses are drawn as the object.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.models.util import utcnow
from app.perception import appearance as app_
from app.perception.encoder import (
    DINOV2_ID,
    DINOV2_VERSION,
    PREPROCESS_VERSION,
    cached_encode,
    shared_encoder,
)
from app.perception.types import EncoderUnavailable
from app.providers.sam import decode_mask_raster
from app.services.storage import data_path

log = logging.getLogger("learnloop.appearance")

# Samples kept per object, so a long recording does not encode every frame.
MAX_SAMPLES_PER_OBJECT = 240


@dataclass
class AppearanceResult:
    """What the pass decided. statuses line up with each object's sightings in time order."""

    used: bool = False
    mode: str = ""
    error: str = ""
    statuses: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    samples: dict[str, list[app_.Sample]] = field(default_factory=dict)
    galleries: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    rows: dict[str, list[dict]] = field(default_factory=dict)
    gallery: dict | None = None
    diagnostics: dict = field(default_factory=dict)


def check_tracks(
    *,
    run_id: str,
    media: Path,
    tracked: dict,
    seeds: list,
    labels: dict[str, str],
    fps: float,
    mode: str,
    source: dict | None = None,
    reference: tuple[dict | None, dict[str, np.ndarray], dict[str, str]] | None = None,
    encoder=None,
) -> AppearanceResult:
    """mode is "demonstration" (learn references from these tracks) or "execution" (use reference).

    reference, for an execution: (gallery manifest, vectors keyed by this run's object ids, labels of those keys).
    A failure returns a result with used False and the reason; the caller draws raw SAM boxes as provisional.
    """
    from app.config import settings

    started = time.perf_counter()
    result = AppearanceResult(mode=mode, labels=dict(labels))
    options = app_.Settings.from_config()
    try:
        encoder = encoder or shared_encoder(DINOV2_ID)
    except EncoderUnavailable as exc:
        result.error = str(exc)
        result.diagnostics = {"used": False, "error": result.error}
        return result
    ordered = {local_id: sorted(items, key=lambda item: item.timestamp) for local_id, items in tracked.items() if items}
    width, height = _frame_size(media)
    aspect = width / height if height else 1.0
    segments = {
        local_id: app_.split_segments([item.timestamp for item in items], [item.box for item in items], aspect)
        for local_id, items in ordered.items()
    }
    seed_times: dict[str, list[float]] = {}
    for seed in seeds:
        seed_times.setdefault(seed.local_id, []).append(float(seed.timestamp))
    step = max(1, round(fps / max(settings.appearance_sample_fps, 0.1)))
    picked: dict[str, list[int]] = {local_id: _sample_indices(items, segments[local_id], seed_times.get(local_id, []), step) for local_id, items in ordered.items()}
    samples = {
        local_id: [
            app_.Sample(
                local_id=local_id,
                timestamp=items[index].timestamp,
                frame_index=items[index].record.frame_index,
                box=items[index].box,
                confidence=items[index].record.confidence,
                segment=segments[local_id][index],
            )
            for index in picked[local_id]
        ]
        for local_id, items in ordered.items()
    }
    records = {(local_id, items[index].record.frame_index): items[index].record for local_id, items in ordered.items() for index in picked[local_id]}
    _fill_crops(media, samples, records)
    others: dict[float, list[tuple[str, dict]]] = {}
    for local_id, items in ordered.items():
        for item in items:
            others.setdefault(round(item.timestamp, 3), []).append((local_id, item.box))
    flat = [sample for items in samples.values() for sample in items]
    app_.score_quality(flat, others, (height, width), options)
    encodable = [sample for sample in flat if sample.crop is not None and sample.crop.size]
    try:
        vectors = cached_encode(encoder, [sample.crop for sample in encodable], [sample.mask for sample in encodable], data_path("inference_cache", "dinov2"))
    except EncoderUnavailable as exc:
        result.error = str(exc)
        result.diagnostics = {"used": False, "error": result.error}
        return result
    for sample, vector in zip(encodable, vectors, strict=True):
        sample.vector = vector
    anchored = {local_id: app_.anchored_segments_of(items, seed_times.get(local_id, [])) for local_id, items in samples.items()}
    if mode == "demonstration":
        selection = app_.select_references(samples, seed_times, labels, options)
        galleries = {
            local_id: np.stack([sample.vector for sample in entry["references"]]) for local_id, entry in selection.items() if entry["references"]
        }
        result.gallery = _save_gallery(run_id, selection, galleries, labels, source or {}, encoder, time.perf_counter() - started)
    else:
        manifest, galleries, reference_labels = reference or (None, {}, {})
        result.labels.update(reference_labels)
        result.gallery = app_.gallery_summary(manifest) if manifest else None
    rows = app_.validate_samples(samples, galleries, result.labels, anchored, options)
    if mode == "execution":
        app_.reject_never_matched(rows, options)
    reach = step / fps + 1e-6
    for local_id, items in ordered.items():
        result.statuses[local_id] = app_.propagate(rows.get(local_id, []), [item.timestamp for item in items], segments[local_id], reach)
        if any(row.get("reason") == app_.NEVER_MATCHED for row in rows.get(local_id, [])):
            # Not one frame looked like the part, so no box of this track is drawn as it.
            result.statuses[local_id] = [("rejected", app_.NEVER_MATCHED)] * len(items)
    result.used = True
    result.samples = samples
    result.galleries = galleries
    result.rows = rows
    result.diagnostics = _diagnostics(result, ordered, segments, encoder, time.perf_counter() - started)
    return result


def _sample_indices(items: list, segments: list[int], seeds: list[float], step: int) -> list[int]:
    """Every step-th sighting, each segment's ends, and the sighting nearest each seed."""
    chosen = set(range(0, len(items), step))
    for index in range(len(items)):
        if index == 0 or segments[index] != segments[index - 1]:
            chosen.add(index)
        if index == len(items) - 1 or segments[index] != segments[index + 1]:
            chosen.add(index)
    for seed in seeds:
        nearest = min(range(len(items)), key=lambda index: abs(items[index].timestamp - seed), default=None)
        if nearest is not None:
            chosen.add(nearest)
    ordered = sorted(chosen)
    if len(ordered) > MAX_SAMPLES_PER_OBJECT:
        stride = len(ordered) / MAX_SAMPLES_PER_OBJECT
        ordered = sorted({ordered[int(index * stride)] for index in range(MAX_SAMPLES_PER_OBJECT)})
    return ordered


def _fill_crops(media: Path, samples: dict[str, list[app_.Sample]], records: dict) -> None:
    """One pass over the low frame rate copy, cropping as frames go by, so frames are never all held at once."""
    wanted: dict[int, list[app_.Sample]] = {}
    for items in samples.values():
        for sample in items:
            wanted.setdefault(sample.frame_index, []).append(sample)
    if not wanted:
        return
    last = max(wanted)
    capture = cv2.VideoCapture(str(media))
    index = 0
    try:
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            for sample in wanted.get(index, []):
                record = records.get((sample.local_id, index))
                raster = decode_mask_raster(record) if record is not None else None
                sample.crop, sample.mask = app_.crop_with_mask(
                    frame, sample.box, raster, getattr(record, "mask_height", 0), getattr(record, "mask_width", 0)
                )
            index += 1
    finally:
        capture.release()


def _frame_size(media: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(media))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    finally:
        capture.release()


def _save_gallery(run_id: str, selection: dict, galleries: dict, labels: dict, source: dict, encoder, runtime: float) -> dict:
    objects = {}
    crops = {}
    for local_id, entry in selection.items():
        references = entry.pop("references")
        crops[local_id] = [sample.crop for sample in references]
        objects[local_id] = {
            **entry,
            "reference_count": len(references),
            "references": [
                {
                    "timestamp": round(sample.timestamp, 3),
                    "frame_index": sample.frame_index,
                    "box": {key: round(value, 4) for key, value in sample.box.items()},
                    "segment": sample.segment,
                    "confidence": sample.confidence,
                    "quality": sample.quality,
                    "masked": sample.mask is not None,
                    "crop": f"{app_._npz_key(local_id)}_{index}.jpg",
                }
                for index, sample in enumerate(references)
            ],
            "association": {
                "evidence": "SAM followed a box the analysis model drew for this object; references come from that stretch only.",
                "seed_times": entry.get("anchor_times") or [],
            },
        }
    learned = [local_id for local_id, entry in objects.items() if entry["status"] in {"ready", "limited"}]
    if not objects:
        status = "unavailable"
    elif len(learned) == len(objects):
        status = "ready"
    elif learned:
        status = "partial"
    else:
        status = "unavailable"
    manifest = {
        "schema": app_.GALLERY_SCHEMA,
        "status": status,
        "encoder_id": DINOV2_ID,
        "encoder_version": DINOV2_VERSION,
        "preprocessing_version": PREPROCESS_VERSION,
        "selection_version": app_.SELECTION_VERSION,
        "source_run_id": run_id,
        "source_video_id": source.get("video_id") or "",
        "video_checksum": source.get("checksum") or "",
        "device": _device(encoder),
        "runtime_sec": round(runtime, 2),
        "created_at": utcnow().isoformat(),
        "objects": objects,
        "warnings": app_.distinguishability(galleries, labels, app_.Settings.from_config().margin),
        "note": "Embeddings of frozen DINOv2 ViT-S/14. No weights were trained. A cosine is a similarity, not a probability.",
    }
    if status == "unavailable":
        manifest["reason"] = "No object had a crop that passed every check."
    return app_.write_gallery(run_id, manifest, galleries, crops)


def _device(encoder) -> str:
    if getattr(encoder, "_model", True) is None:
        return "not loaded: every embedding was cached"
    return getattr(encoder, "device", "cpu")


def _diagnostics(result: AppearanceResult, ordered: dict, segments: dict, encoder, runtime: float) -> dict:
    counts: dict[str, dict[str, int]] = {}
    intervals: dict[str, list[dict]] = {}
    for local_id, statuses in result.statuses.items():
        tally: dict[str, int] = {}
        for status, _reason in statuses:
            tally[status] = tally.get(status, 0) + 1
        counts[local_id] = tally
        intervals[local_id] = _rejected_intervals(ordered[local_id], statuses)
    totals: dict[str, int] = {}
    for tally in counts.values():
        for status, count in tally.items():
            totals[status] = totals.get(status, 0) + count
    return {
        "used": True,
        "mode": result.mode,
        "encoder_id": DINOV2_ID,
        "preprocessing_version": PREPROCESS_VERSION,
        "device": _device(encoder),
        "runtime_sec": round(runtime, 2),
        "samples": sum(len(items) for items in result.samples.values()),
        "sightings": totals,
        "by_object": counts,
        "segments": {local_id: (max(values) + 1 if values else 0) for local_id, values in segments.items()},
        "rejected_intervals": intervals,
        "objects_with_references": sorted(local_id for local_id, refs in result.galleries.items() if len(refs)),
    }


def _rejected_intervals(items: list, statuses: list[tuple[str, str]]) -> list[dict]:
    out: list[dict] = []
    current = None
    for item, (status, reason) in zip(items, statuses, strict=True):
        if status == "rejected":
            if current is None:
                current = {"start": round(item.timestamp, 3), "end": round(item.timestamp, 3), "reason": reason}
                out.append(current)
            current["end"] = round(item.timestamp, 3)
        else:
            current = None
    return out


# ---------------------------------------------------------------- recovery

# Offsets of the search window, in widths and heights of the object's last trusted box.
_OFFSETS = (-1.5, -0.75, 0.0, 0.75, 1.5)
_SCALES = (0.75, 1.33)
# Two candidates this far apart that score within margin of each other are two places the object could be.
_DISTINCT_IOU = 0.3


def propose_recovery(media: Path, tracked: dict, result: AppearanceResult, encoder=None) -> list[dict]:
    """For each rejected stretch, look near the object's last trusted box, and at other tracks' boxes, for a crop
    that matches its references clearly better than every rival and than its surroundings.

    DINOv2 does not find boxes; it ranks these. A stretch is attempted at most once, the search is bounded,
    and an attempt that finds two equally good places, or none, stays unresolved.
    """
    from app.config import settings

    options = app_.Settings.from_config()
    encoder = encoder or shared_encoder(DINOV2_ID)
    intervals = []
    for local_id, items in result.diagnostics.get("rejected_intervals", {}).items():
        if local_id not in result.galleries:
            continue
        for interval in items:
            intervals.append((interval["end"] - interval["start"], local_id, interval))
    intervals.sort(key=lambda item: (-item[0], item[1], item[2]["start"]))
    attempts = []
    for _length, local_id, interval in intervals[: max(settings.recovery_max_intervals, 0)]:
        sightings = sorted(tracked[local_id], key=lambda item: item.timestamp)
        before = [item for item in sightings if item.timestamp < interval["start"] and item.status in {"validated", "recovered"}]
        attempt = {"local_id": local_id, "start": interval["start"], "end": interval["end"], "trigger": interval["reason"]}
        if not before:
            attempts.append({**attempt, "decision": "unresolved", "reason": "no_trusted_box_before"})
            continue
        anchor = before[-1]
        frame_index = next(item.record.frame_index for item in sightings if item.timestamp >= interval["start"])
        attempt["frame_index"] = frame_index
        frame = _read_frame(media, frame_index)
        if frame is None:
            attempts.append({**attempt, "decision": "unresolved", "reason": "frame_unreadable"})
            continue
        candidates = _candidate_boxes(anchor.box, tracked, local_id, frame_index)[: settings.recovery_max_candidates]
        crops, kept = [], []
        for source, box in candidates:
            crop, _mask = app_.crop_with_mask(frame, box, None, 0, 0)
            if crop is not None and min(crop.shape[:2]) >= options.min_crop_px // 2:
                crops.append(crop)
                kept.append((source, box))
        if not crops:
            attempts.append({**attempt, "decision": "unresolved", "reason": "no_candidate"})
            continue
        vectors = cached_encode(encoder, crops, [None] * len(crops), data_path("inference_cache", "dinov2"))
        scored = []
        for (source, box), vector in zip(kept, vectors, strict=True):
            own = app_.top_mean(result.galleries[local_id], vector)
            rival_id, rival = None, -1.0
            for other, refs in result.galleries.items():
                if other != local_id and len(refs):
                    score = app_.top_mean(refs, vector)
                    if score > rival:
                        rival_id, rival = other, score
            scored.append({"source": source, "box": box, "own": round(own, 4), "rival": rival_id, "rival_score": round(rival, 4)})
        attempt["candidates"] = len(scored)
        attempts.append({**attempt, **_decide(scored, local_id, result.labels, options)})
    return attempts


def _decide(scored: list[dict], local_id: str, labels: dict, options: app_.Settings) -> dict:
    ranked = sorted(scored, key=lambda item: -item["own"])
    best = ranked[0]
    surroundings = float(np.median([item["own"] for item in scored]))
    out = {"best": best, "surroundings": round(surroundings, 4)}
    same_kind = best["rival"] and app_._same_kind(labels.get(local_id, ""), labels.get(best["rival"], ""))
    if best["own"] < options.accept:
        return {**out, "decision": "unresolved", "reason": "no_candidate_matches"}
    if best["rival"] and best["rival_score"] > best["own"] - options.margin and not same_kind:
        return {**out, "decision": "unresolved", "reason": f"best_candidate_also_looks_like:{best['rival']}"}
    if best["own"] < surroundings + options.margin and len(scored) > 2:
        # It scores well only because the whole area does: the background, or the colour, not the object.
        return {**out, "decision": "unresolved", "reason": "does_not_stand_out"}
    if str(best["source"]).startswith("track:"):
        return {**out, "decision": "unresolved", "reason": f"best_candidate_is_tracked_as:{best['source'][6:]}"}
    rivals = [item for item in ranked[1:] if app_.iou(item["box"], best["box"]) < _DISTINCT_IOU and item["own"] >= best["own"] - options.margin]
    if rivals:
        return {**out, "decision": "unresolved", "reason": "two_places_match_equally"}
    return {**out, "decision": "reseed", "reason": "clear_match"}


def _candidate_boxes(anchor: dict, tracked: dict, local_id: str, frame_index: int) -> list[tuple[str, dict]]:
    """Windows the size of the last trusted box around where it was, then other tracks' boxes on that frame."""
    boxes = []
    cx, cy = app_.center(anchor)
    for dy in _OFFSETS:
        for dx in _OFFSETS:
            boxes.append(("window", _clamped(cx + dx * anchor["w"], cy + dy * anchor["h"], anchor["w"], anchor["h"])))
    for scale in _SCALES:
        boxes.append(("window", _clamped(cx, cy, anchor["w"] * scale, anchor["h"] * scale)))
    for other, sightings in tracked.items():
        if other == local_id:
            continue
        for item in sightings:
            if item.record.frame_index == frame_index:
                boxes.append((f"track:{other}", item.box))
    return [(source, box) for source, box in boxes if box is not None]


def _clamped(cx: float, cy: float, w: float, h: float) -> dict | None:
    w, h = min(w, 1.0), min(h, 1.0)
    x, y = min(max(cx - w / 2, 0.0), 1.0 - w), min(max(cy - h / 2, 0.0), 1.0 - h)
    if w <= 0 or h <= 0:
        return None
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _read_frame(media: Path, index: int):
    capture = cv2.VideoCapture(str(media))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
