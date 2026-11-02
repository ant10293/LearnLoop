"""Keep one id for one physical part across analysis windows.

The first window sees the parts while they are separate and names them.
A later window sees those same parts turned, held, or split into two visible
arms, and it names them again. Label text does not survive that. The shared
frames do: the box that covers the same pixels is the same part. A second box
that only appears afterward and touches that part is the same part at a new
angle, not a new object.
"""

from __future__ import annotations

import re
import statistics

import numpy as np

from app.perception.assignment import iou, linear_assignment
from app.perception.decisions import merged_decisions, object_dropped
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    ObservationPrediction,
)
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance

# A shared frame decides identity when the two boxes cover mostly the same pixels.
MATCH_IOU = 0.5
# Usual overlap of the smaller box with the larger one. One loose frame can be lower.
CONTAINED = 0.9
# Below this, the smaller box is mostly outside and the two were seen apart.
APART = 0.5
# The inner object has to be clearly the smaller one. Similar boxes are not a part inside a part.
CONTAINED_AREA = 0.75
# A box this large, as a fraction of the frame, is the table or the background.
# Parts resting on it are not a second view of that surface.
BACKGROUND_AREA = 0.25
# A later box this close to a known part is another view of that part.
TOUCH_GAP = 0.04
# Pairs below MATCH_IOU stay out of the assignment, so a weak overlap cannot
# trade places with a real one.
_UNMATCHED = 1e3


def align_to_roster(roster: AnalysisDocument, later: AnalysisDocument) -> AnalysisDocument:
    """Rewrite a later window onto the previous window's ids.

    The previous window already carries the global ids. Only frames the two
    windows both observed are evidence. A part that does not overlap a known
    part there keeps its own id, or receives a new one when that id is already
    a different part.
    """
    if not roster.objects or not later.observations:
        return later
    mapping = _mapping(roster, later)
    objects = _objects(roster, later, mapping)
    if not mapping:
        return later.model_copy(update={"objects": objects})
    return _apply(later, mapping, objects=objects)


def drop_split_objects(roster: AnalysisDocument, document: AnalysisDocument) -> AnalysisDocument:
    """Remove an object the merge added for a second view of a roster part.

    The merge keeps the first window's records. An extra id is removed when its
    boxes touch one of those parts, and the box is kept on that part.
    """
    roster_ids = {obj.local_id for obj in roster.objects}
    if not roster_ids:
        return document
    mapping = _touch_map(
        document,
        roster_ids,
        {item: item for item in roster_ids},
        {obj.local_id: obj.label for obj in document.objects},
        {obj.local_id: obj.label for obj in roster.objects},
    )
    # Two touching objects that each name a tracked instance stay two.
    bound = {obj.local_id for obj in document.objects if obj.instance_id}
    mapping = {extra: target for extra, target in mapping.items() if extra not in bound}
    if not mapping:
        return document
    kept = [obj for obj in document.objects if obj.local_id not in mapping]
    return _apply(document, mapping, objects=kept)


def _mapping(roster: AnalysisDocument, later: AnalysisDocument) -> dict[str, str]:
    roster_ids = [obj.local_id for obj in roster.objects]
    later_ids = [obj.local_id for obj in later.objects]
    shared = _times(roster) & _times(later)
    roster_boxes = _boxes(roster)
    later_boxes = _boxes(later)
    scores = _pair_scores(roster_ids, later_ids, shared, roster_boxes, later_boxes)
    matched = _assign(later_ids, roster_ids, scores)
    roster_set = set(roster_ids)
    # A box that is already on a shared frame was compared directly. Touching a
    # neighbor there is not another copy of that neighbor. With no shared frame
    # at all, touch has nothing to attach to.
    if shared:
        present = {
            item.object_local_id
            for item in later.observations
            if round(float(item.timestamp), 3) in shared
        }
        for later_id, target in _touch_map(
            later,
            roster_set,
            matched,
            {obj.local_id: obj.label for obj in later.objects},
            {obj.local_id: obj.label for obj in roster.objects},
        ).items():
            if later_id in matched or later_id in present:
                continue
            matched[later_id] = target
    used = set(roster_set) | set(later_ids) | set(matched.values())
    claimed = set(matched.values())
    for later_id in later_ids:
        if later_id in matched or later_id not in roster_set:
            continue
        same = scores.get((later_id, later_id), 0.0)
        others = max(
            (scores.get((later_id, roster_id), 0.0) for roster_id in roster_ids if roster_id != later_id),
            default=0.0,
        )
        # The same id is still the best explanation, and nobody else took it.
        # A reused id with no overlap, or a worse overlap than another part, is
        # not that part.
        if later_id not in claimed and same > 0.0 and same > others:
            continue
        fresh = _fresh_id(used)
        matched[later_id] = fresh
        used.add(fresh)
        claimed.add(fresh)
    return {later_id: target for later_id, target in matched.items() if later_id != target}


def _pair_scores(
    roster_ids: list[str],
    later_ids: list[str],
    shared: set[float],
    roster_boxes: dict[tuple[str, float], BBox],
    later_boxes: dict[tuple[str, float], BBox],
) -> dict[tuple[str, str], float]:
    """Mean overlap on the frames where both windows boxed that pair.

    One spiked frame does not outvote the rest of the boundary. Two parts that
    cover each other on a single shared frame stay distinguishable when the
    other shared frame puts each box back on its own part.
    """
    scores: dict[tuple[str, str], float] = {}
    for later_id in later_ids:
        for roster_id in roster_ids:
            values: list[float] = []
            for timestamp in shared:
                left = later_boxes.get((later_id, timestamp))
                right = roster_boxes.get((roster_id, timestamp))
                if left is None or right is None:
                    continue
                values.append(iou(_tuple(left), _tuple(right)))
            if values:
                scores[(later_id, roster_id)] = sum(values) / len(values)
    return scores


def _assign(later_ids: list[str], roster_ids: list[str], scores: dict[tuple[str, str], float]) -> dict[str, str]:
    """Give each later part at most one roster part, and each roster part once."""
    if not later_ids or not roster_ids:
        return {}
    cost = np.full((len(later_ids), len(roster_ids)), _UNMATCHED, dtype=float)
    for row, later_id in enumerate(later_ids):
        for column, roster_id in enumerate(roster_ids):
            score = scores.get((later_id, roster_id), 0.0)
            if score >= MATCH_IOU:
                cost[row, column] = 1.0 - score
    matched: dict[str, str] = {}
    for row, column in linear_assignment(cost):
        later_id = later_ids[row]
        roster_id = roster_ids[column]
        if scores.get((later_id, roster_id), 0.0) >= MATCH_IOU:
            matched[later_id] = roster_id
    return matched


def _touch_map(
    document: AnalysisDocument,
    roster_ids: set[str],
    already: dict[str, str] | None,
    labels: dict[str, str],
    roster_labels: dict[str, str],
) -> dict[str, str]:
    """Map an id that never overlapped a shared frame onto the part its box touches.

    A large neighbor often contains the new box. The part it belongs to is the
    touching box whose roster name shares the most words with it.
    """
    assigned = dict(already or {})
    boxes = _boxes(document)
    # shared-word count, target id, and whether another part tied that count.
    best: dict[str, tuple[int, str, bool]] = {}
    for (object_id, timestamp), box in boxes.items():
        # A matching local id is not evidence. Only a part already matched to
        # the roster can lend that identity to a second view.
        target = assigned.get(object_id)
        if target is None or target not in roster_ids:
            continue
        for (other_id, other_time), other in boxes.items():
            if other_time != timestamp or other_id == object_id or other_id in assigned:
                continue
            if _separation(box, other) > TOUCH_GAP:
                continue
            shared = len(_word_set(labels.get(other_id, "")) & _target_words(target, roster_labels, labels))
            current = best.get(other_id)
            if current is None or shared > current[0]:
                best[other_id] = (shared, target, False)
            elif shared == current[0] and target != current[1]:
                best[other_id] = (shared, current[1], True)
    return {
        other_id: target
        for other_id, (shared, target, tied) in best.items()
        if shared > 0 and not tied
    }


def _target_words(target: str, roster_labels: dict[str, str], document_labels: dict[str, str]) -> set[str]:
    """The roster name, plus the document name when it is still describing that part.

    A later window can reuse an id for a different part. Its label is added only
    when it still shares a word with the roster name, so a stud that inherited
    the fork's id cannot lend the fork the word "stud".
    """
    roster_name = roster_labels.get(target, "")
    document_name = document_labels.get(target, "")
    words = _word_set(roster_name)
    extra = _word_set(document_name)
    if words & extra:
        words |= extra
    return words


def _word_set(label: str) -> set[str]:
    skip = {
        "with", "from", "that", "this", "piece", "part",
        "upper", "lower", "left", "right", "first", "second", "front", "back",
    }
    return {word for word in _tokens(label) if word not in skip}


def _tokens(label: str) -> list[str]:
    cleaned = label.casefold().replace("/", " ").replace("-", " ")
    return [word.strip("().,") for word in cleaned.split() if len(word.strip("().,")) > 3]


def _objects(
    roster: AnalysisDocument, later: AnalysisDocument, mapping: dict[str, str]
) -> list:
    by_id = {obj.local_id: obj for obj in roster.objects}
    seen = set(by_id)
    for obj in later.objects:
        target = mapping.get(obj.local_id, obj.local_id)
        if target in seen:
            continue
        seen.add(target)
        by_id[target] = obj.model_copy(update={"local_id": target})
    return list(by_id.values())


def _apply(document: AnalysisDocument, mapping: dict[str, str], objects: list) -> AnalysisDocument:
    if not mapping:
        return document
    return document.model_copy(
        update={
            "objects": objects,
            "observations": _observations(document.observations, mapping),
            "states": [
                item.model_copy(update={"object_local_id": mapping.get(item.object_local_id, item.object_local_id)})
                for item in document.states
            ],
            "transitions": [
                item.model_copy(update={"object_local_id": mapping.get(item.object_local_id, item.object_local_id)})
                for item in document.transitions
            ],
            "relationships": [
                item.model_copy(
                    update={
                        "subject_local_id": mapping.get(item.subject_local_id, item.subject_local_id),
                        "other_local_id": mapping.get(item.other_local_id, item.other_local_id)
                        if item.other_local_id
                        else None,
                    }
                )
                for item in document.relationships
            ],
            "actions": [
                item.model_copy(update={"object_local_ids": _ids(item.object_local_ids, mapping)})
                for item in document.actions
            ],
            "workflow_steps": [
                item.model_copy(update={"required_objects": _ids(item.required_objects, mapping)})
                for item in document.workflow_steps
            ],
            "ambiguities": [
                item.model_copy(update={"object_local_ids": _ids(item.object_local_ids, mapping)})
                for item in document.ambiguities
            ],
        }
    )


def _observations(observations: list[ObservationPrediction], mapping: dict[str, str]) -> list[ObservationPrediction]:
    grouped: dict[tuple[str, float], list[ObservationPrediction]] = {}
    order: list[tuple[str, float]] = []
    for item in observations:
        object_id = mapping.get(item.object_local_id, item.object_local_id)
        key = (object_id, round(float(item.timestamp), 3))
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(item.model_copy(update={"object_local_id": object_id}))
    merged: list[ObservationPrediction] = []
    for key in order:
        rows = grouped[key]
        kept = rows[0]
        box = kept.bbox
        separate: list[ObservationPrediction] = []
        for extra in rows[1:]:
            if box is None:
                box = extra.bbox
                kept = extra
                continue
            if extra.bbox is not None and _separation(box, extra.bbox) <= TOUCH_GAP:
                box = _union(box, extra.bbox)
            else:
                separate.append(extra)
        merged.append(kept if box is kept.bbox else kept.model_copy(update={"bbox": box}))
        merged.extend(separate)
    return merged


def _fresh_id(used: set[str]) -> str:
    number = 1
    while f"Object_{number}" in used:
        number += 1
    return f"Object_{number}"


def _ids(values: list[str], mapping: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        mapped = mapping.get(value, value)
        if mapped not in seen:
            seen.append(mapped)
    return seen


def _times(document: AnalysisDocument) -> set[float]:
    return {round(float(item.timestamp), 3) for item in document.observations}


def _boxes(document: AnalysisDocument) -> dict[tuple[str, float], BBox]:
    found: dict[tuple[str, float], BBox] = {}
    for item in document.observations:
        if item.bbox is None:
            continue
        found[(item.object_local_id, round(float(item.timestamp), 3))] = item.bbox
    return found


def _tuple(box: BBox) -> tuple[float, float, float, float]:
    return (box.x, box.y, box.w, box.h)


def _separation(a: BBox, b: BBox) -> float:
    dx = max(0.0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
    dy = max(0.0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
    return max(dx, dy)


def _union(a: BBox, b: BBox) -> BBox:
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = min(1.0, max(a.x + a.w, b.x + b.w))
    y2 = min(1.0, max(a.y + a.h, b.y + b.h))
    return BBox(x=x1, y=y1, w=max(x2 - x1, 1e-4), h=max(y2 - y1, 1e-4))


# A hand is the actor, not an object. "arm" alone is not matched: a multi-arm piece is a part.
_HAND = re.compile(r"\b(hands?|fingers?|thumbs?|palms?|wrists?|forearms?|arms)\b", re.IGNORECASE)
# The stationary background the parts rest on. A board, tray, or base that is built on stays.
_RESTING_SURFACE = re.compile(r"\b(background|workspace|tabletop|table|desk|bench|floor|surface)\b", re.IGNORECASE)
_BUILT_ON = re.compile(r"\b(board|tray|base)\b", re.IGNORECASE)


def omit_workspace_and_hands(document: AnalysisDocument) -> AnalysisDocument:
    """Drop hands and the stationary surface the parts rest on.

    The scene prompt already tells the model to leave those out of the object
    list. A surface it records anyway is not an object, and its box would
    otherwise cover the parts. A board, tray, or base, and anything picked up,
    stays unless its box shows it is the scene itself.
    """
    reasons: dict[str, tuple[str, str]] = {}
    for obj in document.objects:
        if _workspace_or_hand(obj):
            reasons[obj.local_id] = ("workspace_or_hand", f"The name {obj.label!r} is a hand or a resting surface.")
        elif _fills_the_scene(document, obj):
            reasons[obj.local_id] = ("fills_the_scene", "Its large box never moved, so it is the surface the work happens on.")
    dropped = set(reasons)
    if not dropped:
        return document
    decisions = [
        row
        for object_id, (reason, detail) in reasons.items()
        for row in object_dropped(document, object_id, "workspace_filter", reason, detail=detail)
    ]

    def kept(ids: list[str]) -> list[str]:
        return [item for item in ids if item not in dropped]

    return document.model_copy(
        update={
            "box_decisions": merged_decisions(document.box_decisions, decisions),
            "objects": [obj for obj in document.objects if obj.local_id not in dropped],
            "observations": [item for item in document.observations if item.object_local_id not in dropped],
            "states": [item for item in document.states if item.object_local_id not in dropped],
            "transitions": [item for item in document.transitions if item.object_local_id not in dropped],
            "relationships": [
                item
                for item in document.relationships
                if item.subject_local_id not in dropped and item.other_local_id not in dropped
            ],
            "actions": [
                item.model_copy(update={"object_local_ids": kept(item.object_local_ids)})
                for item in document.actions
            ],
            "workflow_steps": [
                item.model_copy(update={"required_objects": kept(item.required_objects)})
                for item in document.workflow_steps
            ],
            "ambiguities": [
                item.model_copy(update={"object_local_ids": kept(item.object_local_ids)})
                for item in document.ambiguities
            ],
        }
    )


# The name of a surface varies between runs: table, tray, lid, board. Its box does not.
# Picked-up parts move, and even a finished assembly stays well under this share of the frame.
_SCENE_AREA = 0.25
_SCENE_DRIFT = 0.03
_SCENE_MIN_BOXES = 3


def _fills_the_scene(document: AnalysisDocument, obj) -> bool:
    """A large box that never moves is the surface the work happens on, whatever it was called."""
    boxes = [
        item.bbox for item in document.observations if item.object_local_id == obj.local_id and item.bbox is not None
    ]
    if len(boxes) < _SCENE_MIN_BOXES:
        return False
    if statistics.median(box.w * box.h for box in boxes) < _SCENE_AREA:
        return False
    centers_x = [box.x + box.w / 2 for box in boxes]
    centers_y = [box.y + box.h / 2 for box in boxes]
    return max(statistics.pstdev(centers_x), statistics.pstdev(centers_y)) <= _SCENE_DRIFT


def _workspace_or_hand(obj) -> bool:
    label = obj.label or ""
    if _HAND.search(label):
        return True
    if obj.manipulated or _BUILT_ON.search(label):
        return False
    return _RESTING_SURFACE.search(label) is not None


def drop_contained_objects(document: AnalysisDocument) -> AnalysisDocument:
    """Drop an object whose box never leaves the inside of another object.

    A smaller box that stays inside a larger one is the same object seen twice.
    It remains its own object when any shared frame shows the two apart.
    """
    bound = {obj.local_id for obj in document.objects if obj.instance_id}
    # An object bound to a tracked instance is a distinct physical instance even when nested.
    mapping = {inner: outer for inner, outer in _contained_hosts(document).items() if inner not in bound}
    if not mapping:
        return document
    observations = [item for item in document.observations if item.object_local_id not in mapping]
    objects = [obj for obj in document.objects if obj.local_id not in mapping]
    decisions = [
        row
        for inner, outer in sorted(mapping.items())
        for row in object_dropped(
            document,
            inner,
            "contained_objects",
            "contained_in_object",
            other_object_id=outer,
            detail=f"Its box stayed inside {outer} on every frame where both were visible.",
        )
    ]
    ambiguities = list(document.ambiguities)
    used = {item.local_id for item in ambiguities}
    for inner, outer in sorted(mapping.items()):
        marker = f"Ambiguity_contained_{inner}"
        if marker in used:
            continue
        used.add(marker)
        inner_label = _label_of(document, inner)
        outer_label = _label_of(document, outer)
        ambiguities.append(
            AmbiguityPrediction(
                local_id=marker,
                kind=AmbiguityKind.UNCERTAIN_IDENTITY,
                description=(
                    f"{inner} ({inner_label}) was not kept as its own object. Its box stayed inside "
                    f"{outer} ({outer_label}) on every frame where both were visible, and the two were "
                    "never seen apart."
                ),
                object_local_ids=[outer],
                evidence=Evidence(
                    classification=EvidenceClass.INFERRED,
                    provenance=Provenance.DETERMINISTIC,
                    note="Compared the two boxes on the frames where both were present.",
                ),
            )
        )
    return document.model_copy(
        update={
            "box_decisions": merged_decisions(document.box_decisions, decisions),
            "objects": objects,
            "observations": observations,
            "states": [
                item.model_copy(update={"object_local_id": mapping.get(item.object_local_id, item.object_local_id)})
                for item in document.states
            ],
            "transitions": [
                item.model_copy(update={"object_local_id": mapping.get(item.object_local_id, item.object_local_id)})
                for item in document.transitions
            ],
            "relationships": [
                item.model_copy(
                    update={
                        "subject_local_id": mapping.get(item.subject_local_id, item.subject_local_id),
                        "other_local_id": mapping.get(item.other_local_id, item.other_local_id)
                        if item.other_local_id
                        else None,
                    }
                )
                for item in document.relationships
            ],
            "actions": [
                item.model_copy(update={"object_local_ids": _ids(item.object_local_ids, mapping)})
                for item in document.actions
            ],
            "workflow_steps": [
                item.model_copy(update={"required_objects": _ids(item.required_objects, mapping)})
                for item in document.workflow_steps
            ],
            "ambiguities": [
                *[
                    item.model_copy(update={"object_local_ids": _ids(item.object_local_ids, mapping)})
                    for item in ambiguities
                    if item.local_id not in {f"Ambiguity_contained_{inner}" for inner in mapping}
                ],
                *[item for item in ambiguities if item.local_id.startswith("Ambiguity_contained_")],
            ],
        }
    )


def _contained_hosts(document: AnalysisDocument) -> dict[str, str]:
    boxes = _boxes(document)
    areas = _median_area(boxes)
    hosts: dict[str, tuple[float, str]] = {}
    for inner, inner_area in areas.items():
        best: tuple[float, str] | None = None
        for outer, outer_area in areas.items():
            if inner == outer or outer_area >= BACKGROUND_AREA or inner_area > outer_area * CONTAINED_AREA:
                continue
            shared = sorted(
                time for (object_id, time) in boxes if object_id == inner and (outer, time) in boxes
            )
            if not shared:
                continue
            covered = [_overlap(boxes[(inner, time)], boxes[(outer, time)]) for time in shared]
            ordered = sorted(covered)
            usual = ordered[len(ordered) // 2]
            if usual < CONTAINED or any(amount < APART for amount in covered):
                continue
            if best is None or outer_area < best[0]:
                best = (outer_area, outer)
        if best is not None:
            hosts[inner] = best
    resolved = _resolve_hosts({inner: outer for inner, (_area, outer) in hosts.items()})
    # Several objects inside one outline are the parts on that outline, not copies of it.
    counts: dict[str, int] = {}
    for outer in resolved.values():
        counts[outer] = counts.get(outer, 0) + 1
    return {inner: outer for inner, outer in resolved.items() if counts[outer] == 1}


def _resolve_hosts(hosts: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for inner in hosts:
        seen: set[str] = set()
        current = inner
        while current in hosts and current not in seen:
            seen.add(current)
            current = hosts[current]
        if current != inner and current not in hosts:
            resolved[inner] = current
    return resolved


def _median_area(boxes: dict[tuple[str, float], BBox]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for (object_id, _time), box in boxes.items():
        grouped.setdefault(object_id, []).append(box.w * box.h)
    areas: dict[str, float] = {}
    for object_id, values in grouped.items():
        ordered = sorted(values)
        areas[object_id] = ordered[len(ordered) // 2]
    return areas


def _overlap(inner: BBox, outer: BBox) -> float:
    """Share of the inner box that lies inside the outer box."""
    ix1, iy1 = inner.x, inner.y
    ix2, iy2 = inner.x + inner.w, inner.y + inner.h
    ox1, oy1 = outer.x, outer.y
    ox2, oy2 = outer.x + outer.w, outer.y + outer.h
    overlap_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    overlap_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    return (overlap_w * overlap_h) / max(inner.w * inner.h, 1e-9)


def _label_of(document: AnalysisDocument, object_id: str) -> str:
    for obj in document.objects:
        if obj.local_id == object_id:
            return obj.label or object_id
    return object_id
