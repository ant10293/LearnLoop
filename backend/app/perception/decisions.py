"""A record of each box the pipeline removed or moved, and why.

A cleared box otherwise looks the same as a model saying the part was absent.
These rows keep the original box so a run can show what was taken away.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument, BoxDecision, ObservationPrediction


def box_decision(
    step: str,
    reason_code: str,
    observation: ObservationPrediction,
    *,
    outcome: str = "dropped",
    other_object_id: str | None = None,
    detail: str = "",
) -> BoxDecision:
    return BoxDecision(
        step=step,
        reason_code=reason_code,
        outcome=outcome,
        object_local_id=observation.object_local_id,
        timestamp=observation.timestamp,
        bbox=observation.bbox,
        confidence=observation.confidence,
        other_object_id=other_object_id,
        detail=detail,
    )


def merged_decisions(*groups: list[BoxDecision]) -> list[BoxDecision]:
    """Join rows in order. The grounding repairs run twice and can see the same box twice."""
    seen: set[tuple] = set()
    rows: list[BoxDecision] = []
    for group in groups:
        for row in group:
            box = (row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h) if row.bbox else None
            timestamp = round(row.timestamp, 3) if row.timestamp is not None else None
            key = (row.step, row.reason_code, row.object_local_id, timestamp, box)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def with_decisions(document: AnalysisDocument, rows: list[BoxDecision]) -> AnalysisDocument:
    if not rows:
        return document
    return document.model_copy(update={"box_decisions": merged_decisions(document.box_decisions, rows)})


def object_dropped(
    document: AnalysisDocument,
    object_id: str,
    step: str,
    reason_code: str,
    *,
    other_object_id: str | None = None,
    detail: str = "",
) -> list[BoxDecision]:
    """One row per box of a removed object, or one row with no box when it never had any."""
    rows = [
        box_decision(step, reason_code, item, other_object_id=other_object_id, detail=detail)
        for item in document.observations
        if item.object_local_id == object_id and item.bbox is not None
    ]
    if rows:
        return rows
    return [
        BoxDecision(
            step=step,
            reason_code=reason_code,
            object_local_id=object_id,
            other_object_id=other_object_id,
            detail=detail,
        )
    ]
