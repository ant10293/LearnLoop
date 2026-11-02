"""Choose the frames where a known part has to be looked for again.

Scene analysis boxes every part it can name in one pass. A part that rotates
and is partly covered often gets no box, or gets the neighbor's box, which
extent matching then removes. The follow-up look is a different question:
here is this part, and here are the frames where its box is missing. Answer
only for that part. The returned box still has to fit the part's size.
"""

from __future__ import annotations

import numpy as np

from app.perception.continuity import GROWTH, SHRINK
from app.perception.decisions import with_decisions
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    BoxDecision,
    Evidence,
    ObservationPrediction,
)
from app.schemas.enums import Continuity, EvidenceClass, Provenance, Visibility

# A follow-up box may be rotated. It may not be a different surface, such as skin.
SATURATION_GAP = 40
CHROMA_GAP = 30
# The scene pass already left this frame out. A score under 70% is the glare
# case: the second look boxed a bright patch and said it was unsure.
MIN_CONFIDENCE = 0.70

TAIL_FRAMES = 3


def choose_locate_times(kept: list[float], all_times: list[float], limit: int = 4) -> list[float]:
    """Holes between sightings, then the first frames after the part vanishes.

    A hole is the usual place a rotated, partly covered part is still visible.
    The first frames of a trailing gap are where an insertion is still on screen.
    Frames long after the part has gone stay out of the request.
    """
    if not kept or limit <= 0:
        return []
    kept_set = {round(timestamp, 3) for timestamp in kept}
    first, last = min(kept_set), max(kept_set)
    holes = [timestamp for timestamp in all_times if first < timestamp < last and round(timestamp, 3) not in kept_set]
    holes.sort(key=lambda timestamp: min(abs(timestamp - item) for item in kept_set))
    tail = [timestamp for timestamp in all_times if timestamp > last]
    chosen = holes[:limit]
    chosen.extend(tail[: max(0, min(TAIL_FRAMES, limit - len(chosen)))])
    unique: list[float] = []
    for timestamp in chosen:
        if timestamp not in unique:
            unique.append(timestamp)
    return unique


def reference_time(observations: list[ObservationPrediction], object_id: str) -> ObservationPrediction | None:
    """The in-size view used as the picture of the part."""
    rows = [item for item in observations if item.object_local_id == object_id and item.bbox is not None]
    if len(rows) < 3:
        return None
    areas = sorted(item.bbox.w * item.bbox.h for item in rows if item.bbox is not None)
    reference = areas[len(areas) // 2]
    ranked = []
    for item in rows:
        assert item.bbox is not None
        area = item.bbox.w * item.bbox.h
        ratio = max(area, reference) / min(area, reference)
        if ratio > GROWTH:
            continue
        visible = 0 if item.visibility == Visibility.VISIBLE else 1
        ranked.append((visible, ratio, item.timestamp, item))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def same_surface(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """The center of the follow-up box has to be the same material as the reference.

    Area alone accepts a skin-colored box the size of a small part. Saturation and
    the spread between color channels stay put when the part rotates, and they
    change when the box has landed on a different surface.
    """
    left = _surface(reference)
    right = _surface(candidate)
    if left is None or right is None:
        return False
    return abs(left[0] - right[0]) <= SATURATION_GAP and abs(left[1] - right[1]) <= CHROMA_GAP


def _surface(image: np.ndarray) -> tuple[float, float] | None:
    if image is None or getattr(image, "size", 0) == 0 or getattr(image, "ndim", 0) != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        return None
    y0, y1 = height // 4, max(height // 4 + 1, (3 * height) // 4)
    x0, x1 = width // 4, max(width // 4 + 1, (3 * width) // 4)
    pixels = image[y0:y1, x0:x1].reshape(-1, image.shape[2])[:, :3].astype(np.float32)
    median = np.median(pixels, axis=0)
    blue, green, red = (float(median[0]), float(median[1]), float(median[2]))
    peak = max(red, green, blue)
    floor = min(red, green, blue)
    saturation = 0.0 if peak <= 0 else (peak - floor) / peak * 255.0
    return saturation, peak - floor


def box_fits(area: float, reference: float) -> bool:
    """A located box may be the visible portion. It may not be a different, larger part."""
    if area <= 0 or reference <= 0:
        return False
    if area > reference * GROWTH:
        return False
    if area * SHRINK < reference:
        return False
    return True


def apply_located(
    document: AnalysisDocument,
    object_id: str,
    found: list[tuple],
) -> AnalysisDocument:
    """Add boxes the follow-up look returned, when they fit this part.

    A reported score under 70% is left out. A box with no score is still eligible.
    """
    reference = reference_time(document.observations, object_id)
    if reference is None or reference.bbox is None:
        return document
    reference_area = reference.bbox.w * reference.bbox.h
    occupied = {
        (round(item.timestamp, 3), item.object_local_id)
        for item in document.observations
        if item.bbox is not None
    }
    others: dict[float, list[tuple[str, BBox]]] = {}
    for item in document.observations:
        if item.object_local_id == object_id or item.bbox is None:
            continue
        others.setdefault(round(item.timestamp, 3), []).append((item.object_local_id, item.bbox))
    added: list[ObservationPrediction] = []
    decisions: list[BoxDecision] = []

    def refuse(timestamp, bbox, confidence, reason: str, detail: str, other: str | None = None) -> None:
        decisions.append(
            BoxDecision(
                step="locate",
                reason_code=reason,
                object_local_id=object_id,
                timestamp=timestamp,
                bbox=bbox,
                confidence=confidence,
                other_object_id=other,
                detail=detail,
            )
        )

    for item in found:
        timestamp, bbox, visibility, confidence = _with_confidence(item)
        key = round(timestamp, 3)
        if bbox is None and len(item) > 4:
            refuse(timestamp, item[4], item[5], "surface_mismatch", "The pixels in this box do not match the reference crop.")
            continue
        if bbox is None:
            continue
        if visibility == "absent":
            refuse(timestamp, bbox, confidence, "box_marked_absent", "The answer gave a box and also called the part absent.")
            continue
        if confidence is not None and confidence < MIN_CONFIDENCE:
            refuse(
                timestamp, bbox, confidence, "low_confidence", f"Scored {confidence:.2f}, under the {MIN_CONFIDENCE:.2f} limit."
            )
            continue
        if (key, object_id) in occupied:
            refuse(timestamp, bbox, confidence, "slot_occupied", "This part already has a box on this frame.")
            continue
        area = bbox.w * bbox.h
        if not box_fits(area, reference_area):
            refuse(
                timestamp,
                bbox,
                confidence,
                "extent_mismatch",
                f"{area / max(reference_area, 1e-9):.2f} times the reference area. The allowed range is {1 / SHRINK:g} to {GROWTH:g}.",
            )
            continue
        overlap = next(((other_id, _iou(bbox, other)) for other_id, other in others.get(key, []) if _iou(bbox, other) >= 0.55), None)
        if overlap is not None:
            refuse(
                timestamp,
                bbox,
                confidence,
                "duplicate_overlap",
                f"IoU {overlap[1]:.2f} with {overlap[0]}. The limit is 0.55.",
                overlap[0],
            )
            continue
        added.append(
            ObservationPrediction(
                object_local_id=object_id,
                timestamp=timestamp,
                bbox=bbox,
                visibility=Visibility.PARTIAL if visibility == "partial" else Visibility.VISIBLE,
                description=(
                    "Visible portion of this part. A changed angle or a partial cover still counts, "
                    "and the box is only the portion that matches the part's own size."
                ),
                evidence=Evidence(
                    classification=EvidenceClass.DIRECTLY_OBSERVED,
                    provenance=Provenance.AI_PREDICTION,
                    note="Located from a reference crop of this part, on the frame where the scene pass had no fitting box.",
                    frame_timestamps=[timestamp],
                ),
                continuity=Continuity.REASSOCIATED,
                confidence=confidence,
            )
        )
    document = with_decisions(document, decisions)
    if not added:
        return document
    observations = [*document.observations, *added]
    observations.sort(key=lambda item: (item.timestamp, item.object_local_id))
    return document.model_copy(update={"observations": observations})


def _with_confidence(item: tuple) -> tuple[float, BBox | None, str, float | None]:
    """Older callers pass timestamp, box, and visibility. A missing score stays eligible."""
    timestamp, bbox, visibility, *rest = item
    return timestamp, bbox, visibility, rest[0] if rest else None


def _iou(a: BBox, b: BBox) -> float:
    from app.perception.assignment import iou

    return iou((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h))
