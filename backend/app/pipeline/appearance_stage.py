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
    reach = step / fps + 1e-6
    for local_id, items in ordered.items():
        result.statuses[local_id] = app_.propagate(rows.get(local_id, []), [item.timestamp for item in items], segments[local_id], reach)
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
