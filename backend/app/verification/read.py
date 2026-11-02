"""Read canonical analysis documents without treating labels as identity."""

from __future__ import annotations

import re

from app.schemas.analysis import AnalysisDocument

INCIDENTAL_ACTIONS = frozenset(
    {"adjust", "inspect", "hold", "motion", "approach", "put_down", "release"}
)
STRONG_RELATIONS = frozenset(
    {
        "touching",
        "inside",
        "around",
        "attached",
        "supported_by",
        "held_by",
        "above",
        "below",
        "left_of",
        "right_of",
    }
)


def as_document(raw: AnalysisDocument | dict) -> AnalysisDocument:
    if isinstance(raw, AnalysisDocument):
        return raw
    return AnalysisDocument.model_validate(raw)


def enum_value(value) -> str:
    if value is None:
        return ""
    return value.value if hasattr(value, "value") else str(value)


def tokens(text: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(part) > 1}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a <= end_b and start_b <= end_a


def evidence_class(item) -> str:
    evidence = getattr(item, "evidence", None)
    if evidence is None:
        return "UNKNOWN"
    return enum_value(evidence.classification) or "UNKNOWN"


def evidence_note(item) -> str:
    evidence = getattr(item, "evidence", None)
    if evidence is None:
        return ""
    return evidence.note or ""


def frame_times(item) -> list[float]:
    evidence = getattr(item, "evidence", None)
    if evidence is None:
        return []
    return list(evidence.frame_timestamps or [])


def object_map(document: AnalysisDocument) -> dict:
    return {item.local_id: item for item in document.objects}


def signature(document: AnalysisDocument, object_id: str) -> set[str]:
    """Undirected relation signature. Partner identity is the label, not the instance id."""
    labels = {item.local_id: (item.label or "").strip().lower() for item in document.objects}
    found: set[str] = set()
    for relation in document.relationships:
        ends = {relation.subject_local_id, relation.other_local_id}
        if object_id not in ends or relation.other_local_id is None:
            continue
        partner = relation.other_local_id if relation.subject_local_id == object_id else relation.subject_local_id
        kind = enum_value(relation.kind)
        found.add(f"{kind}|{labels.get(partner, '')}")
    return found


def state_labels(document: AnalysisDocument, object_id: str) -> set[str]:
    found = set()
    for state in document.states:
        if state.object_local_id == object_id and state.state_label:
            found.add(state.state_label.strip().lower())
    for transition in document.transitions:
        if transition.object_local_id != object_id:
            continue
        if transition.to_state:
            found.add(transition.to_state.strip().lower())
    return found


def median_area(document: AnalysisDocument, object_id: str) -> float | None:
    areas = []
    for observation in document.observations:
        if observation.object_local_id != object_id or observation.bbox is None:
            continue
        areas.append(observation.bbox.w * observation.bbox.h)
    if not areas:
        return None
    areas.sort()
    return areas[len(areas) // 2]
