"""Geometric relationships. Overlap is not attachment or a completed action."""

from __future__ import annotations

from app.perception.assignment import center, iou
from app.perception.types import TrackPoint
from app.schemas.analysis import Evidence, RelationshipPrediction
from app.schemas.enums import EvidenceClass, Provenance, RelationKind


def geometric_relationships(points: list[TrackPoint], limit: int = 24) -> list[RelationshipPrediction]:
    """Pairwise image relations at shared timestamps. Contact is not claimed."""
    by_time: dict[float, list[TrackPoint]] = {}
    for point in points:
        if point.visibility not in {"visible", "partial"}:
            continue
        by_time.setdefault(round(point.timestamp, 3), []).append(point)
    rows: list[RelationshipPrediction] = []
    seen: set[tuple] = set()
    for timestamp in sorted(by_time):
        group = by_time[timestamp]
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                kind, note, classification = _pair(left, right)
                key = (left.track_ref, right.track_ref, kind)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    RelationshipPrediction(
                        subject_local_id=left.track_ref,
                        other_local_id=right.track_ref,
                        kind=kind,
                        time_start=timestamp,
                        time_end=timestamp,
                        description=note,
                        evidence=Evidence(
                            classification=classification,
                            provenance=Provenance.DETERMINISTIC,
                            note=note,
                            time_start=timestamp,
                            time_end=timestamp,
                            frame_timestamps=[timestamp],
                        ),
                    )
                )
                if len(rows) >= limit:
                    return rows
    return rows


def _pair(left: TrackPoint, right: TrackPoint):
    left_box = (left.x, left.y, left.w, left.h)
    right_box = (right.x, right.y, right.w, right.h)
    overlap = iou(left_box, right_box)
    lx, ly = center(left_box)
    rx, ry = center(right_box)
    if overlap >= 0.05:
        return (
            RelationKind.NEAR,
            "Bounding boxes overlap in the image. Overlap does not establish contact, attachment, or insertion.",
            EvidenceClass.DIRECTLY_OBSERVED,
        )
    if abs(lx - rx) < 0.08 and abs(ly - ry) < 0.08:
        return (
            RelationKind.NEAR,
            "The boxes are close in the image. Proximity is not attachment.",
            EvidenceClass.DIRECTLY_OBSERVED,
        )
    if ly + 0.02 < ry:
        return (
            RelationKind.ABOVE,
            "The subject center is above the other box in the image. This is not a support relation.",
            EvidenceClass.DIRECTLY_OBSERVED,
        )
    if ly > ry + 0.02:
        return (
            RelationKind.BELOW,
            "The subject center is below the other box in the image.",
            EvidenceClass.DIRECTLY_OBSERVED,
        )
    if lx + 0.02 < rx:
        return (
            RelationKind.LEFT_OF,
            "The subject center is left of the other box in the image.",
            EvidenceClass.DIRECTLY_OBSERVED,
        )
    return (
        RelationKind.RIGHT_OF,
        "The subject center is right of the other box in the image.",
        EvidenceClass.DIRECTLY_OBSERVED,
    )
