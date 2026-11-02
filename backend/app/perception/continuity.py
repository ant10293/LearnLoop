"""Bind each object id to one physical extent.

A language model emits a box per id per frame. When a small part rotates or
disappears into an assembly, the model often keeps the id and points it at a
larger neighbor of a similar color. The box then covers the neighbor. Extent
is what makes those boxes the same part or a different one: a part can be
partly covered and look smaller, and it can sit at a new angle, but it does
not become many times larger. A larger box is a different part. It is given
to the part whose established size it matches, or it is dropped.
"""

from __future__ import annotations

import math

from app.perception.assignment import center, iou
from app.perception.decisions import box_decision, merged_decisions
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    ObservationPrediction,
)
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance

# A later view of the same part stays inside this factor of the part's own
# established area. Occlusion and an end-on angle shrink a box. They do not
# multiply its area by this much. Head motion can scale a part, but not by
# enough to turn a short rod into the body it is being fitted to.
GROWTH = 4.0
# A box may move to a different part only when it actually fits that part.
# "Closer than the wrong owner" is not enough: a medium box can be nearer a
# rod than a clear piece and still be neither of them.
ASSIGN_RATIO = 2.0
# Below this, the box is no longer a partial view of this part unless it is
# still sitting where this part was and no other part accounts for it.
SHRINK = 8.0
DUPLICATE_IOU = 0.85
MIN_SAMPLES = 3


def reconcile_identity(document: AnalysisDocument) -> AnalysisDocument:
    """Drop or reassign boxes that cannot be the object they were labeled as."""
    samples = _samples(document)
    references = {object_id: _reference_area(areas) for object_id, areas in samples.items()}
    references = {object_id: area for object_id, area in references.items() if area is not None}
    if not references:
        return document

    by_time: dict[float, list[ObservationPrediction]] = {}
    passthrough: list[ObservationPrediction] = []
    for observation in document.observations:
        if observation.bbox is None or observation.object_local_id not in references:
            passthrough.append(observation)
            continue
        by_time.setdefault(round(observation.timestamp, 3), []).append(observation)

    rebuilt: list[ObservationPrediction] = list(passthrough)
    removed: dict[str, int] = {}
    reassigned: dict[str, int] = {}
    decisions = []
    history = _history(document.observations, references)
    for timestamp in sorted(by_time):
        group = by_time[timestamp]
        kept, oversized, undersized = _split(group, references)
        occupied = {item.object_local_id for item in kept}
        moved_small = _assign(
            undersized, occupied, references, kept, history, timestamp, allow_owner=True
        )
        moved_small_ids = {id(observation) for observation, _target in moved_small}
        for observation, target in moved_small:
            rebuilt.append(_retarget(observation, target))
            reassigned[target] = reassigned.get(target, 0) + 1
            decisions.append(_moved(observation, target))
            occupied.add(target)
            _remember(history, target, timestamp, observation)
        for observation in undersized:
            # Still where this part was, and no other part's size claims it.
            # An end-on or covered view stays with its own id.
            if id(observation) not in moved_small_ids:
                kept.append(observation)
                occupied.add(observation.object_local_id)
        moved_large = _assign(
            oversized, occupied, references, kept, history, timestamp, allow_owner=False
        )
        assigned_ids = {id(observation) for observation, _target in moved_large}
        for observation, target in moved_large:
            rebuilt.append(_retarget(observation, target))
            reassigned[target] = reassigned.get(target, 0) + 1
            decisions.append(_moved(observation, target))
            _remember(history, target, timestamp, observation)
        for observation in oversized:
            if id(observation) not in assigned_ids:
                removed[observation.object_local_id] = removed.get(observation.object_local_id, 0) + 1
                reference = references[observation.object_local_id]
                decisions.append(
                    box_decision(
                        "identity_reconcile",
                        "extent_growth_exceeded",
                        observation,
                        detail=(
                            f"{_area(observation.bbox) / max(reference, 1e-9):.1f} times this part's established area. "
                            f"The limit is {GROWTH:g}, and no other part fit the box."
                        ),
                    )
                )
        rebuilt.extend(kept)
        for observation in kept:
            _remember(history, observation.object_local_id, timestamp, observation)

    ambiguities = list(document.ambiguities)
    for object_id, count in sorted(removed.items()):
        marker = f"Ambiguity_extent_{object_id}"
        if any(item.local_id == marker for item in ambiguities):
            continue
        label = _label(document, object_id)
        ambiguities.append(
            AmbiguityPrediction(
                local_id=marker,
                kind=AmbiguityKind.UNCERTAIN_IDENTITY,
                description=(
                    f"{count} box(es) labeled {object_id} ({label}) were removed because each one was many "
                    "times larger than that part, or it duplicated a part that already had its own box. "
                    "A partly covered or rotated part keeps a box on its visible portion. It does not inherit "
                    "the box of a larger neighbor."
                ),
                object_local_ids=[object_id],
                evidence=Evidence(
                    classification=EvidenceClass.INFERRED,
                    provenance=Provenance.DETERMINISTIC,
                    note="Compared with the area this id kept on the frames where its size was stable.",
                ),
            )
        )
    rebuilt.sort(key=lambda item: (item.timestamp, item.object_local_id))
    if not removed and not reassigned and len(rebuilt) == len(document.observations):
        return document
    return document.model_copy(
        update={
            "observations": rebuilt,
            "ambiguities": ambiguities,
            "box_decisions": merged_decisions(document.box_decisions, decisions),
        }
    )


def _moved(observation: ObservationPrediction, target: str):
    return box_decision(
        "identity_reconcile",
        "size_matched_other_part",
        observation,
        outcome="reassigned",
        other_object_id=target,
        detail=f"The box size fits {target}, not {observation.object_local_id}.",
    )


def _samples(document: AnalysisDocument) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for observation in document.observations:
        if observation.bbox is None:
            continue
        samples.setdefault(observation.object_local_id, []).append(_area(observation.bbox))
    return samples


def _reference_area(areas: list[float]) -> float | None:
    if len(areas) < MIN_SAMPLES:
        return None
    best_group: list[float] = []
    best_key: tuple[int, float] | None = None
    for seed in areas:
        group = [area for area in areas if _ratio(area, seed) <= GROWTH]
        key = (len(group), -seed)
        if best_key is None or key > best_key:
            best_key = key
            best_group = group
    if len(best_group) < MIN_SAMPLES:
        return None
    ordered = sorted(best_group)
    return ordered[len(ordered) // 2]


def _split(
    group: list[ObservationPrediction], references: dict[str, float]
) -> tuple[list[ObservationPrediction], list[ObservationPrediction], list[ObservationPrediction]]:
    kept: list[ObservationPrediction] = []
    oversized: list[ObservationPrediction] = []
    undersized: list[ObservationPrediction] = []
    for observation in group:
        if observation.bbox is None:
            kept.append(observation)
            continue
        reference = references.get(observation.object_local_id)
        area = _area(observation.bbox)
        if reference is None:
            kept.append(observation)
        elif area > reference * GROWTH:
            oversized.append(observation)
        elif reference > area * SHRINK:
            undersized.append(observation)
        else:
            kept.append(observation)
    return kept, oversized, undersized


def _remember(
    history: dict[str, list[tuple[float, tuple[float, float]]]],
    object_id: str,
    timestamp: float,
    observation: ObservationPrediction,
) -> None:
    if observation.bbox is None:
        return
    history.setdefault(object_id, []).append((timestamp, center(_tuple(observation.bbox))))


def _assign(
    pool: list[ObservationPrediction],
    occupied: set[str],
    references: dict[str, float],
    kept: list[ObservationPrediction],
    history: dict[str, list[tuple[float, tuple[float, float]]]],
    timestamp: float,
    allow_owner: bool,
) -> list[tuple[ObservationPrediction, str]]:
    targets = {object_id for object_id in references if object_id not in occupied}
    kept_boxes = {
        item.object_local_id: _tuple(item.bbox) for item in kept if item.bbox is not None
    }
    ranked: list[tuple[float, int, ObservationPrediction, str]] = []
    for index, observation in enumerate(pool):
        if observation.bbox is None:
            continue
        box = _tuple(observation.bbox)
        area = _area(observation.bbox)
        owner_distance = _distance_to(history, observation.object_local_id, timestamp, center(box))
        if any(
            iou(box, kept_box) >= DUPLICATE_IOU and _ratio(area, references.get(target, 0)) <= ASSIGN_RATIO
            for target, kept_box in kept_boxes.items()
        ):
            # A near-copy of a box that already belongs to a part. Another part
            # with a similar size does not get that copy.
            continue
        fits: list[tuple[float, str]] = []
        for target in targets:
            if target == observation.object_local_id and not allow_owner:
                continue
            ratio = _ratio(area, references[target])
            limit = GROWTH if target == observation.object_local_id else ASSIGN_RATIO
            if ratio > limit:
                continue
            if target != observation.object_local_id and allow_owner:
                target_distance = _distance_to(history, target, timestamp, center(box))
                # A shrunken box that is still where this part was is an end-on or
                # covered view, not the other part. Move it only when it sits with
                # the part whose size it matches.
                if owner_distance + 0.02 < target_distance:
                    continue
            fits.append((ratio, target))
        if not fits:
            continue
        fits.sort()
        best_ratio, best_target = fits[0]
        ranked.append((best_ratio, index, observation, best_target))
    ranked.sort()
    used_observations: set[int] = set()
    used_targets: set[str] = set()
    assigned: list[tuple[ObservationPrediction, str]] = []
    for _ratio_value, index, observation, target in ranked:
        if index in used_observations or target in used_targets:
            continue
        used_observations.add(index)
        used_targets.add(target)
        assigned.append((observation, target))
    return assigned


def _history(
    observations: list[ObservationPrediction], references: dict[str, float]
) -> dict[str, list[tuple[float, tuple[float, float]]]]:
    points: dict[str, list[tuple[float, tuple[float, float]]]] = {}
    for observation in observations:
        reference = references.get(observation.object_local_id)
        if observation.bbox is None or reference is None:
            continue
        if _ratio(_area(observation.bbox), reference) > GROWTH:
            continue
        points.setdefault(observation.object_local_id, []).append(
            (round(observation.timestamp, 3), center(_tuple(observation.bbox)))
        )
    for rows in points.values():
        rows.sort()
    return points


def _distance_to(
    history: dict[str, list[tuple[float, tuple[float, float]]]],
    object_id: str,
    timestamp: float,
    point: tuple[float, float],
) -> float:
    rows = history.get(object_id) or []
    if not rows:
        return math.inf
    prior = [item for item in rows if item[0] < timestamp - 1e-6]
    chosen = prior[-1][1] if prior else rows[0][1]
    return math.hypot(point[0] - chosen[0], point[1] - chosen[1])


def _retarget(observation: ObservationPrediction, target: str) -> ObservationPrediction:
    if target == observation.object_local_id:
        return observation
    note = observation.evidence.note
    extra = f"Reassigned from {observation.object_local_id} because the box matches {target}'s established size."
    evidence = observation.evidence.model_copy(update={"note": f"{note} {extra}".strip()})
    return observation.model_copy(update={"object_local_id": target, "evidence": evidence})


def _label(document: AnalysisDocument, object_id: str) -> str:
    for obj in document.objects:
        if obj.local_id == object_id:
            return obj.label or object_id
    return object_id


def _tuple(bbox: BBox) -> tuple[float, float, float, float]:
    return (bbox.x, bbox.y, bbox.w, bbox.h)


def _area(bbox: BBox) -> float:
    return bbox.w * bbox.h


def _ratio(area: float, reference: float) -> float:
    if area <= 0 or reference <= 0:
        return math.inf
    return max(area, reference) / min(area, reference)
