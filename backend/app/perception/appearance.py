"""Appearance references learned from a demonstration, and checks of SAM tracks against them.

A gallery holds a few DINOv2 embeddings of crops that SAM followed from a box the
analysis model drew. Its crops come only from the stretch of a track that began at
that box, pass image-quality checks, and still look like the crops next to the box.
A part SAM drifted onto is kept out: looking like the other crops of one drifted
stretch is not evidence of being the right part.

Every number here is a cosine between frozen DINOv2 features. It is not a
probability, and it never decides whether a step was done. It decides only whether
a box is shown as the object, and which stretch of a track to trust.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.perception.encoder import DINOV2_ID, DINOV2_VERSION, PREPROCESS_VERSION

SELECTION_VERSION = "seed-anchored-v1"
GALLERY_SCHEMA = "appearance_gallery_1"

# A track breaks into a new stretch after this many seconds with no box.
SEGMENT_GAP_SEC = 0.75
# Or when its box centre moves this many box diagonals between neighbouring frames.
JUMP_DIAGONALS = 2.5
# Or when its box area changes by more than this factor between neighbouring frames.
JUMP_AREA_RATIO = 4.0
# Crops this close to a seed box describe what the analysis model pointed at.
ANCHOR_SEC = 1.0
# A reference from a stretch SAM did not start at a seed box must look at least this much like the seed crops.
UNANCHORED_ACCEPT = 0.75
# References closer in time than this add little.
MIN_REFERENCE_GAP_SEC = 0.6
# Two references this similar are the same view.
DUPLICATE_COSINE = 0.95
# Boxes of two objects overlapping this much may hold either one.
OVERLAP_IOU = 0.3
OVERLAP_CONTAINED = 0.6
DUPLICATE_IOU = 0.7
MAX_BOX_AREA = 0.5
MIN_MASK_COVERAGE = 0.12
MIN_CONFIDENCE = 0.5
RELATIVE_BLUR = 0.35
ABSOLUTE_BLUR = 12.0
# Statuses drawn as the object. validated and recovered are confirmed by appearance; provisional and ambiguous
# are drawn marked as unconfirmed (two identical parts cannot be told apart by appearance). rejected is never drawn.
SHOWN = frozenset({"validated", "recovered", "provisional", "ambiguous"})


@dataclass
class Settings:
    max_references: int = 6
    min_crop_px: int = 20
    accept: float = 0.6
    margin: float = 0.05
    floor: float = 0.4

    @classmethod
    def from_config(cls) -> Settings:
        from app.config import settings

        return cls(
            max_references=settings.appearance_max_references,
            min_crop_px=settings.appearance_min_crop_px,
            accept=settings.appearance_accept,
            margin=settings.appearance_margin,
            floor=settings.appearance_floor,
        )


@dataclass
class Sample:
    """One SAM sighting picked for an appearance check. Geometry is the raw SAM box."""

    local_id: str
    timestamp: float
    frame_index: int
    box: dict
    confidence: float | None
    segment: int = 0
    crop: np.ndarray | None = None
    mask: np.ndarray | None = None
    vector: np.ndarray | None = None
    quality: dict = field(default_factory=dict)
    rejections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- geometry


def center(box: dict) -> tuple[float, float]:
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def iou(a: dict, b: dict) -> float:
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def containment(a: dict, b: dict) -> float:
    """Share of the smaller box that lies inside the other."""
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
    return inter / smaller if smaller > 0 else 0.0


def overlapping(a: dict, b: dict) -> bool:
    """Two boxes that share enough pixels that a crop of one shows the other: a part on, in, or against another."""
    return iou(a, b) > OVERLAP_IOU or containment(a, b) > OVERLAP_CONTAINED


def is_jump(previous: dict, current: dict, aspect: float = 1.0) -> bool:
    """A box that moves several of its own diagonals, or changes area several times over, between neighbouring frames."""
    (px, py), (cx, cy) = center(previous), center(current)
    # Normalized x and y are scaled by different frame sides, so x is put back in units of y.
    distance = math.hypot((cx - px) * aspect, cy - py)
    diagonal = max(math.hypot(previous["w"] * aspect, previous["h"]), math.hypot(current["w"] * aspect, current["h"]), 1e-6)
    area_a, area_b = previous["w"] * previous["h"], current["w"] * current["h"]
    ratio = max(area_a, area_b) / max(min(area_a, area_b), 1e-9)
    return distance > JUMP_DIAGONALS * diagonal or ratio > JUMP_AREA_RATIO


def split_segments(times: list[float], boxes: list[dict], aspect: float = 1.0) -> list[int]:
    """Segment number for each sighting, in time order. A gap or a jump starts a new one.
    A split says only that SAM's box moved in a way a part rarely does. It rejects nothing by itself."""
    segments: list[int] = []
    current = 0
    for index in range(len(times)):
        if index and (times[index] - times[index - 1] > SEGMENT_GAP_SEC or is_jump(boxes[index - 1], boxes[index], aspect)):
            current += 1
        segments.append(current)
    return segments


# ---------------------------------------------------------------- crops


def crop_with_mask(frame: np.ndarray, box: dict, raster: bytes | None, mask_h: int, mask_w: int, pad: float = 0.08):
    """The box with a little context, and the SAM mask on the same pixels when it decodes.
    A SAM raster covers its box, not the frame."""
    height, width = frame.shape[:2]
    bx0, by0 = box["x"] * width, box["y"] * height
    bx1, by1 = (box["x"] + box["w"]) * width, (box["y"] + box["h"]) * height
    pad_x, pad_y = (bx1 - bx0) * pad, (by1 - by0) * pad
    x0, y0 = max(int(bx0 - pad_x), 0), max(int(by0 - pad_y), 0)
    x1, y1 = min(int(math.ceil(bx1 + pad_x)), width), min(int(math.ceil(by1 + pad_y)), height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None
    crop = frame[y0:y1, x0:x1].copy()
    if raster is None or mask_h <= 0 or mask_w <= 0:
        return crop, None
    array = np.frombuffer(raster, dtype=np.uint8)
    if array.size != mask_h * mask_w:
        return crop, None
    inner_x0, inner_y0 = max(int(round(bx0)), 0), max(int(round(by0)), 0)
    inner_x1 = min(max(int(round(bx1)), inner_x0 + 1), width)
    inner_y1 = min(max(int(round(by1)), inner_y0 + 1), height)
    placed = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    resized = cv2.resize(array.reshape(mask_h, mask_w), (inner_x1 - inner_x0, inner_y1 - inner_y0), interpolation=cv2.INTER_NEAREST)
    oy, ox = inner_y0 - y0, inner_x0 - x0
    region = placed[oy : oy + resized.shape[0], ox : ox + resized.shape[1]]
    region[:] = resized[: region.shape[0], : region.shape[1]]
    return crop, (placed > 0).astype(np.uint8)


def sharpness(crop: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None and mask.shape == gray.shape and mask.sum() > 16:
        # Edges of the object only, so a sharp table behind a blurred part does not count.
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        values = lap[inner > 0] if inner.sum() > 16 else lap[mask > 0]
        return float(values.var())
    return float(lap.var())


def score_quality(samples: list[Sample], others: dict[float, list[tuple[str, dict]]], frame_shape: tuple[int, int], options: Settings) -> None:
    """Fill quality and the reasons a sample cannot be a reference. Sharpness is judged against the object's own median."""
    height, width = frame_shape
    for sample in samples:
        reasons = sample.rejections
        box = sample.box
        w_px, h_px = box["w"] * width, box["h"] * height
        sample.quality["size_px"] = [round(w_px, 1), round(h_px, 1)]
        if sample.crop is None or min(w_px, h_px) < options.min_crop_px:
            reasons.append("too_small")
        if box["w"] * box["h"] > MAX_BOX_AREA:
            reasons.append("too_large")
        edge = 0.004
        if box["x"] <= edge or box["y"] <= edge or box["x"] + box["w"] >= 1 - edge or box["y"] + box["h"] >= 1 - edge:
            reasons.append("clipped")
        if sample.confidence is not None and sample.confidence < MIN_CONFIDENCE:
            reasons.append("low_confidence")
        if sample.mask is not None:
            coverage = float(sample.mask.mean())
            sample.quality["mask_coverage"] = round(coverage, 3)
            if coverage < MIN_MASK_COVERAGE:
                reasons.append("mask_sparse")
        else:
            sample.quality["mask_coverage"] = None
        for other_id, other_box in others.get(round(sample.timestamp, 3), []):
            if other_id == sample.local_id:
                continue
            if iou(box, other_box) > DUPLICATE_IOU:
                # Two tracks on one region: at most one of them is on its own object.
                reasons.append(f"duplicates:{other_id}")
            elif overlapping(box, other_box):
                reasons.append(f"overlaps:{other_id}")
        if sample.crop is not None:
            sample.quality["sharpness"] = round(sharpness(sample.crop, sample.mask), 2)
    by_object: dict[str, list[float]] = {}
    for sample in samples:
        if "sharpness" in sample.quality:
            by_object.setdefault(sample.local_id, []).append(sample.quality["sharpness"])
    for sample in samples:
        value = sample.quality.get("sharpness")
        if value is None:
            continue
        median = float(np.median(by_object[sample.local_id]))
        relative = value / median if median > 0 else 1.0
        sample.quality["relative_sharpness"] = round(relative, 3)
        if value < ABSOLUTE_BLUR or relative < RELATIVE_BLUR:
            sample.rejections.append("blurry")


def quality_rank(sample: Sample) -> float:
    """Higher is a better reference. Only used to order samples that already passed every check."""
    confidence = sample.confidence if sample.confidence is not None else 0.7
    sharp = min(float(sample.quality.get("relative_sharpness") or 1.0), 2.0) / 2.0
    coverage = sample.quality.get("mask_coverage")
    mask_term = 0.85 if coverage is None else min(1.0, 0.6 + coverage)
    return round(confidence * (0.5 + sharp) * mask_term, 4)


# ---------------------------------------------------------------- similarity


def top_mean(references: np.ndarray, query: np.ndarray, k: int = 2) -> float:
    if references is None or len(references) == 0 or query is None:
        return float("nan")
    scores = np.sort(references @ query)[::-1]
    return float(scores[: min(k, len(scores))].mean())


def _centroid(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# ---------------------------------------------------------------- gallery


def select_references(
    samples: dict[str, list[Sample]],
    anchors: dict[str, list[float]],
    labels: dict[str, str],
    options: Settings,
) -> dict[str, dict]:
    """Choose each object's trusted references from embedded samples.

    Candidates come only from segments that contain a seed time, sit within the
    first stretch of that segment that still looks like the crops beside the
    seed, and do not look more like another object's seed crops.
    """
    seed_sets: dict[str, list[Sample]] = {}
    for local_id, items in samples.items():
        near = [
            item for item in items
            if item.vector is not None and not _hard_reject(item)
            and any(0 <= item.timestamp - seed <= ANCHOR_SEC for seed in anchors.get(local_id, []))
        ]
        # The crops nearest each seed, whatever their sharpness, are what the analysis model boxed.
        if not near:
            near = sorted(
                (item for item in items if item.vector is not None and any(abs(item.timestamp - seed) <= ANCHOR_SEC for seed in anchors.get(local_id, []))),
                key=lambda item: min(abs(item.timestamp - seed) for seed in anchors[local_id]),
            )[:2]
        seed_sets[local_id] = near
    seed_centroids = {local_id: _centroid([item.vector for item in near]) for local_id, near in seed_sets.items() if near}
    result: dict[str, dict] = {}
    for local_id, items in samples.items():
        reasons: dict[str, int] = {}
        entry = {"label": labels.get(local_id, local_id), "anchor_times": sorted(anchors.get(local_id, [])), "references": []}
        centroid = seed_centroids.get(local_id)
        if centroid is None:
            entry.update(status="unavailable", reason="No sharp crop of this object near the box the analysis model drew.")
            entry["rejected"] = len(items)
            entry["rejection_reasons"] = {"no_anchor": len(items)}
            result[local_id] = entry
            continue
        anchored_segments = anchored_segments_of(items, anchors.get(local_id, []))
        candidates: list[Sample] = []
        drifted_from: dict[int, float] = {}
        for item in sorted(items, key=lambda value: value.timestamp):
            if item.vector is None:
                reasons["not_encoded"] = reasons.get("not_encoded", 0) + 1
                continue
            own = float(item.vector @ centroid)
            item.quality["seed_similarity"] = round(own, 4)
            # A stretch that did not start at a seed box may still add a view, but only one that looks like the
            # seed crops themselves, by a wider margin than inside the seeded stretch.
            strict = item.segment not in anchored_segments
            if strict and own < UNANCHORED_ACCEPT:
                reasons["unanchored_segment"] = reasons.get("unanchored_segment", 0) + 1
                continue
            if item.segment in drifted_from:
                reasons["after_drift"] = reasons.get("after_drift", 0) + 1
                continue
            if own < options.floor:
                # The stretch that began at the seed no longer looks like it. Nothing later in it is trusted.
                drifted_from[item.segment] = item.timestamp
                reasons["drifted"] = reasons.get("drifted", 0) + 1
                continue
            rival = max(
                ((other, float(item.vector @ vector)) for other, vector in seed_centroids.items() if other != local_id),
                key=lambda pair: pair[1],
                default=(None, -1.0),
            )
            item.quality["closest_rival"] = rival[0]
            item.quality["rival_similarity"] = round(rival[1], 4)
            # Inside the seeded stretch a rival must beat it by margin; outside, it must trail it by twice the margin.
            beaten = rival[1] > own - 2 * options.margin if strict else rival[1] > own + options.margin
            if rival[0] and beaten and not _same_kind(labels.get(local_id, ""), labels.get(rival[0], "")):
                reasons[f"closer_to:{rival[0]}"] = reasons.get(f"closer_to:{rival[0]}", 0) + 1
                continue
            if item.rejections:
                for reason in item.rejections:
                    key = reason.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                continue
            if own < options.accept - options.margin:
                reasons["unlike_seed"] = reasons.get("unlike_seed", 0) + 1
                continue
            candidates.append(item)
        chosen = _diverse(candidates, options.max_references)
        entry["candidates"] = len(candidates)
        entry["rejected"] = len(items) - len(chosen)
        entry["rejection_reasons"] = reasons
        entry["drift_detected_at"] = {str(segment): round(time, 3) for segment, time in drifted_from.items()}
        entry["references"] = chosen
        if len(chosen) >= 3:
            entry["status"] = "ready"
        elif chosen:
            entry["status"] = "limited"
            entry["reason"] = f"Only {len(chosen)} trustworthy crop{'s' if len(chosen) > 1 else ''} passed every check."
        else:
            entry["status"] = "unavailable"
            entry["reason"] = "No crop passed the quality and consistency checks, so no reference was made."
        result[local_id] = entry
    return result


def anchored_segments_of(items: list[Sample], seeds: list[float]) -> set[int]:
    """Segments that began at, or run through, a box the analysis model drew."""
    found = {_segment_at(items, seed) for seed in seeds}
    return {segment for segment in found if segment is not None}


def _segment_at(items: list[Sample], moment: float) -> int | None:
    nearest = min(items, key=lambda item: abs(item.timestamp - moment), default=None)
    if nearest is None or abs(nearest.timestamp - moment) > ANCHOR_SEC:
        return None
    return nearest.segment


def _hard_reject(sample: Sample) -> bool:
    return any(reason in {"too_small", "too_large", "blurry", "mask_sparse"} or reason.startswith(("overlaps:", "duplicates:")) for reason in sample.rejections)


def _diverse(candidates: list[Sample], limit: int) -> list[Sample]:
    """Best first, then the candidate least like those chosen, at least MIN_REFERENCE_GAP_SEC from each. Deterministic."""
    ordered = sorted(candidates, key=lambda item: (-quality_rank(item), item.timestamp))
    chosen: list[Sample] = []
    for _ in range(limit):
        best, best_score = None, -1.0
        for item in ordered:
            if item in chosen or any(abs(item.timestamp - other.timestamp) < MIN_REFERENCE_GAP_SEC for other in chosen):
                continue
            closest = max((float(item.vector @ other.vector) for other in chosen), default=0.0)
            if closest > DUPLICATE_COSINE:
                continue
            score = quality_rank(item) * (1.0 - 0.5 * closest)
            if score > best_score + 1e-9:
                best, best_score = item, score
        if best is None:
            break
        chosen.append(best)
    return sorted(chosen, key=lambda item: item.timestamp)


def _same_kind(a: str, b: str) -> bool:
    return bool(a) and a.strip().lower() == b.strip().lower()


def distinguishability(galleries: dict[str, np.ndarray], labels: dict[str, str], margin: float) -> list[dict]:
    """Pairs of objects whose references look about as alike as each object's own references do."""
    warnings = []
    ids = sorted(local_id for local_id, vectors in galleries.items() if len(vectors))
    within = {}
    for local_id in ids:
        vectors = galleries[local_id]
        if len(vectors) > 1:
            sims = vectors @ vectors.T
            within[local_id] = float(sims[np.triu_indices(len(vectors), 1)].mean())
        else:
            within[local_id] = 1.0
    for index, a in enumerate(ids):
        for b in ids[index + 1 :]:
            cross = float((galleries[a] @ galleries[b].T).max())
            same = _same_kind(labels.get(a, ""), labels.get(b, ""))
            if same or cross >= min(within[a], within[b]) - margin:
                warnings.append(
                    {
                        "objects": [a, b],
                        "kind": "same_kind" if same else "similar_appearance",
                        "cross_similarity": round(cross, 4),
                        "note": (
                            "Same label: appearance cannot tell these two parts apart."
                            if same
                            else "These references look about as alike as each object's own views, so a match between them is weak."
                        ),
                    }
                )
    return warnings


# ---------------------------------------------------------------- validation


# A box this many times the area of the object's validated boxes holds more than the object.
GROWTH_RATIO = 3.0
_UNCHECKABLE = ("too_small", "mask_sparse")


def classify(own: float, rival: float, rival_same_kind: bool, options: Settings) -> str:
    """Per-sample reading before temporal smoothing.

    match: looks like its references and not more like another object's.
    other: looks clearly like another object (rival at least near accept, beating own by margin).
    ambiguous: as close to a same-kind or similar object as to its own.
    unlike: below floor, and nothing else matches either. Occlusion and a hand do this too, so it rejects nothing alone.
    weak: anything in between.
    """
    if math.isnan(own):
        return "unchecked"
    has_rival = not math.isnan(rival)
    if has_rival and abs(own - rival) < options.margin and (rival_same_kind or own >= options.floor):
        return "ambiguous"
    if has_rival and not rival_same_kind and rival >= options.accept - options.margin and rival > own + options.margin:
        return "other"
    if own >= options.accept and (not has_rival or own >= rival + options.margin or rival_same_kind):
        return "match"
    if own < options.floor:
        return "unlike"
    return "weak"


def validate_samples(
    samples: dict[str, list[Sample]],
    galleries: dict[str, np.ndarray],
    labels: dict[str, str],
    anchored: dict[str, set[int]],
    options: Settings,
) -> dict[str, list[dict]]:
    """Status for each sample. Rejection needs positive evidence: two neighbouring samples that look like another
    object, or a box that grew far past the object's validated size while not matching it. A crop too small to
    judge, or one that is merely unfamiliar (a hand over it, a new angle), stays provisional: shown, not validated."""
    out: dict[str, list[dict]] = {}
    for local_id, items in samples.items():
        own_refs = galleries.get(local_id)
        rows = []
        for item in sorted(items, key=lambda value: value.timestamp):
            usable = item.vector is not None and not any(reason.startswith(_UNCHECKABLE) for reason in item.rejections)
            own = top_mean(own_refs, item.vector) if usable and own_refs is not None and len(own_refs) else float("nan")
            rival_id, rival = None, float("nan")
            for other, refs in galleries.items():
                if other == local_id or refs is None or not len(refs) or not usable:
                    continue
                score = top_mean(refs, item.vector)
                if math.isnan(rival) or score > rival:
                    rival_id, rival = other, score
            same = bool(rival_id) and _same_kind(labels.get(local_id, ""), labels.get(rival_id, ""))
            reading = classify(own, rival, same, options)
            overlaps = sorted(reason.split(":", 1)[1] for reason in item.rejections if reason.startswith("overlaps:"))
            if reading == "other" and overlaps:
                # The crop shows the parts it overlaps too, so looking like one of them is expected, not a swap.
                reading = "weak"
            rows.append(
                {
                    "time": round(item.timestamp, 3),
                    "frame_index": item.frame_index,
                    "segment": item.segment,
                    "area": round(item.box["w"] * item.box["h"], 6),
                    "own": None if math.isnan(own) else round(own, 4),
                    "rival": rival_id,
                    "rival_score": None if math.isnan(rival) else round(rival, 4),
                    "reading": reading,
                    "checkable": usable,
                    "overlaps": overlaps,
                    "has_reference": own_refs is not None and len(own_refs) > 0,
                }
            )
        _smooth(rows)
        out[local_id] = rows
    return out


def _smooth(rows: list[dict]) -> None:
    validated_areas = [row["area"] for row in rows if row["reading"] == "match"]
    typical = float(np.median(validated_areas)) if len(validated_areas) >= 2 else None
    def grew(row: dict) -> bool:
        return typical is not None and row["area"] > GROWTH_RATIO * typical

    for index, row in enumerate(rows):
        reading = row["reading"]
        near = [rows[j] for j in (index - 1, index + 1) if 0 <= j < len(rows) and rows[j]["segment"] == row["segment"]]
        neighbours = [item["reading"] for item in near]
        if reading == "match":
            row["status"], row["reason"] = "validated", "matches_references"
        elif reading == "other" and "other" in neighbours:
            row["status"], row["reason"] = "rejected", f"looks_like:{row['rival']}"
        elif grew(row) and any(grew(item) and item["reading"] != "match" for item in near):
            # SAM's box now holds much more than the object it was following, whatever the crop looks like.
            row["status"], row["reason"] = "rejected", "box_grew_beyond_object"
        elif reading == "ambiguous":
            row["status"], row["reason"] = "ambiguous", "matches_a_similar_object_as_well"
        elif reading == "unchecked":
            row["status"] = "provisional"
            row["reason"] = "no_reference" if not row["has_reference"] else "too_small_to_check"
        else:
            reason = {"other": "single_frame_like_another", "unlike": "unlike_references"}.get(reading, "weak_match")
            row["status"], row["reason"] = "provisional", "overlaps_other_object" if row["overlaps"] else reason


def propagate(statuses: list[dict], times: list[float], segments: list[int], reach: float) -> list[tuple[str, str]]:
    """Status for every sighting from the nearest checked sample in the same segment."""
    out = []
    for moment, segment in zip(times, segments, strict=True):
        best = None
        for row in statuses:
            if row["segment"] != segment:
                continue
            gap = abs(row["time"] - moment)
            if gap <= reach and (best is None or gap < abs(best["time"] - moment)):
                best = row
        if best is None:
            out.append(("provisional", "not_sampled"))
        else:
            out.append((best["status"], best["reason"]))
    return out


# ---------------------------------------------------------------- persistence


def gallery_key() -> str:
    return f"{DINOV2_ID}__{PREPROCESS_VERSION}__{SELECTION_VERSION}"


def gallery_dir(run_id: str) -> Path:
    from app.services.storage import data_path

    return data_path("runs", run_id, "appearance", gallery_key())


def write_gallery(run_id: str, manifest: dict, vectors: dict[str, np.ndarray], crops: dict[str, list[np.ndarray]]) -> dict:
    """Write to a temporary folder and rename it into place, so a half-written gallery is never read as ready."""
    final = gallery_dir(run_id)
    tmp = final.with_name(f".{final.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    arrays = {local_id: np.asarray(array, dtype=np.float32) for local_id, array in vectors.items() if len(array)}
    np.savez(tmp / "embeddings.npz", **{_npz_key(local_id): array for local_id, array in arrays.items()})
    digest = hashlib.sha256()
    for local_id in sorted(arrays):
        digest.update(local_id.encode())
        digest.update(arrays[local_id].tobytes())
    digest.update(gallery_key().encode())
    manifest = {**manifest, "gallery_id": digest.hexdigest()[:24], "gallery_key": gallery_key()}
    for local_id, items in crops.items():
        for index, crop in enumerate(items):
            cv2.imwrite(str(tmp / f"{_npz_key(local_id)}_{index}.jpg"), crop)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    backup = final.with_name(f".{final.name}.old")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        os.replace(final, backup)
    os.replace(tmp, final)
    shutil.rmtree(backup, ignore_errors=True)
    return manifest


def read_gallery(run_id: str) -> tuple[dict | None, dict[str, np.ndarray]]:
    """The manifest and vectors, or (None, {}) when no gallery of the current encoder and preprocessing exists."""
    folder = gallery_dir(run_id)
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError):
        return None, {}
    if not compatible(manifest):
        return None, {}
    vectors: dict[str, np.ndarray] = {}
    try:
        with np.load(folder / "embeddings.npz") as data:
            for local_id in manifest.get("objects") or {}:
                key = _npz_key(local_id)
                if key in data.files:
                    vectors[local_id] = np.asarray(data[key], dtype=np.float32)
    except (OSError, ValueError):
        if manifest.get("status") in {"ready", "partial"}:
            return {**manifest, "status": "failed", "reason": "The stored embeddings could not be read."}, {}
    return manifest, vectors


def compatible(manifest: dict) -> bool:
    """Vectors from another encoder, weights, preprocessing, or selection rule are never mixed with these."""
    return (
        manifest.get("schema") == GALLERY_SCHEMA
        and manifest.get("encoder_id") == DINOV2_ID
        and manifest.get("encoder_version") == DINOV2_VERSION
        and manifest.get("preprocessing_version") == PREPROCESS_VERSION
        and manifest.get("selection_version") == SELECTION_VERSION
    )


def _npz_key(local_id: str) -> str:
    return "o_" + "".join(char if char.isalnum() else "_" for char in local_id)


def role_galleries(template_document: dict, run_id: str) -> tuple[dict | None, dict[str, np.ndarray]]:
    """A learned procedure's references by role. A role's references are those of the demonstration objects it came from."""
    manifest, vectors = read_gallery(run_id)
    if manifest is None:
        return None, {}
    by_role: dict[str, np.ndarray] = {}
    for role in template_document.get("roles") or []:
        parts = [vectors[local_id] for local_id in role.get("source_object_ids") or [] if local_id in vectors]
        if parts:
            by_role[role["role_id"]] = np.concatenate(parts, axis=0)
    return manifest, by_role


def gallery_summary(manifest: dict | None) -> dict:
    """Small status for API responses. No vectors."""
    if manifest is None:
        return {"status": "missing"}
    objects = manifest.get("objects") or {}
    learned = sum(item.get("status") in {"ready", "limited"} for item in objects.values())
    return {
        "status": manifest.get("status"),
        "gallery_id": manifest.get("gallery_id"),
        "encoder_id": manifest.get("encoder_id"),
        "objects": len(objects),
        "objects_learned": learned,
        "references": sum(len(item.get("references") or []) for item in objects.values()),
        "unavailable": [local_id for local_id, item in objects.items() if item.get("status") not in {"ready", "limited"}],
        "warnings": len(manifest.get("warnings") or []),
        "reason": manifest.get("reason") or "",
        "created_at": manifest.get("created_at"),
    }
