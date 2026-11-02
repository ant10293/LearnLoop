"""Keep records a window returned when a later merge call leaves that list empty.

A model asked to merge several windows can spend its output on objects and
actions and omit states, transitions, and relationships. Those lists are
restored from the windows. Object ids are rewritten when the merge reused an
id for a different label.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.perception.assignment import iou
from app.perception.decisions import box_decision, merged_decisions
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BoxDecision,
    DependencyPrediction,
    Evidence,
    ObservationPrediction,
    RelationshipPrediction,
    StatePrediction,
    TransitionPrediction,
    WorkflowStepPrediction,
)
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance, Visibility

logger = logging.getLogger(__name__)

# Two boxes that cover mostly the same pixels are the same localization.
# This matches the shared-frame overlap already used to treat boxes as one part.
# It is not a detection-sensitivity setting.
AGREEMENT_IOU = 0.5

_EVIDENCE_RANK = {
    EvidenceClass.DIRECTLY_OBSERVED: 2,
    EvidenceClass.INFERRED: 1,
    EvidenceClass.UNKNOWN: 0,
}


_STAGE_LISTS = (
    "objects",
    "observations",
    "states",
    "transitions",
    "relationships",
    "actions",
    "workflow_steps",
    "dependencies",
    "ambiguities",
)


def retain_prior_records(incoming: AnalysisDocument, prior: AnalysisDocument) -> AnalysisDocument:
    """Keep records a later stage left out.

    Action, state, workflow, and ambiguity calls are asked for one slice.
    An empty objects, actions, or observations list is that slice being absent,
    not a decision to delete the scene. A roster that shares no ids with the
    scene is a renamed copy, so the scene ids and their boxes stay.
    """
    update: dict[str, object] = {}
    if prior.box_decisions:
        update["box_decisions"] = merged_decisions(prior.box_decisions, incoming.box_decisions)
    for name in _STAGE_LISTS:
        current = getattr(incoming, name)
        previous = getattr(prior, name)
        if not current and previous:
            update[name] = list(previous)
    if any(item.observation_role for item in prior.observations):
        # Evidence fusion already resolved geometry. A later stage may add raw
        # estimates and semantic lists; it does not replace objects or boxes.
        known_ids = {item.local_id for item in prior.objects}
        later = [
            item.model_copy(
                update={
                    "observation_role": "raw",
                    "geometry_origin": "vlm_estimated" if item.bbox is not None else "unobserved",
                    "geometry_authority": "vlm_later_stage",
                    "detection_support": "none",
                }
            )
            for item in incoming.observations
            if item.object_local_id in known_ids and not item.observation_role
        ]
        kept_raw = [item for item in prior.raw_observations if item.geometry_authority != "vlm_later_stage"]
        update["objects"] = list(prior.objects)
        update["observations"] = list(prior.observations)
        update["raw_observations"] = kept_raw + later
        update["perception"] = prior.perception
        return incoming.model_copy(update=update)
    objects = list(update.get("objects", incoming.objects))
    if incoming.objects and prior.objects:
        incoming_ids = {item.local_id for item in incoming.objects}
        prior_ids = {item.local_id for item in prior.objects}
        if incoming_ids.isdisjoint(prior_ids):
            objects = list(prior.objects)
            update["objects"] = objects
            if prior.observations:
                update["observations"] = list(prior.observations)
    known = {item.local_id for item in objects}
    observations = update.get("observations", incoming.observations)
    if observations and prior.observations and known:
        if not any(item.object_local_id in known for item in observations):
            update["observations"] = list(prior.observations)
    if not update:
        return incoming
    return incoming.model_copy(update=update)


_RESOLVED_AMBIGUITY = ("resolve_", "sam_", "ground_")


def merge_interpretation(base: AnalysisDocument, proposed: AnalysisDocument) -> AnalysisDocument:
    """Take semantic lists from a grounded interpretation call and keep resolved geometry.

    Objects, the display set, and perception come from `base`. Boxes the call
    returned are kept as raw estimates tagged ``vlm_interpretation`` and are
    replaced on retry. Ambiguities from resolution, SAM, and grounding stay.
    """
    known = {item.local_id for item in base.objects}
    raw = [item for item in base.raw_observations if item.geometry_authority != "vlm_interpretation"]
    raw += [
        item.model_copy(
            update={
                "observation_role": "raw",
                "geometry_origin": "vlm_estimated" if item.bbox is not None else "unobserved",
                "geometry_authority": "vlm_interpretation",
                "detection_support": "none",
            }
        )
        for item in proposed.observations
        if item.object_local_id in known
    ]
    kept_ambiguities = [item for item in base.ambiguities if item.local_id.startswith(_RESOLVED_AMBIGUITY)]
    proposed_ambiguities = [item for item in proposed.ambiguities if not item.local_id.startswith(_RESOLVED_AMBIGUITY)]
    return base.model_copy(
        update={
            "summary": proposed.summary or base.summary,
            "actions": list(proposed.actions),
            "states": list(proposed.states),
            "transitions": list(proposed.transitions),
            "relationships": list(proposed.relationships),
            "workflow_steps": list(proposed.workflow_steps),
            "dependencies": list(proposed.dependencies),
            "ambiguities": kept_ambiguities + proposed_ambiguities,
            "raw_observations": raw,
        }
    )


def restore_omitted_records(merged: AnalysisDocument, partials: list[AnalysisDocument]) -> AnalysisDocument:
    if len(partials) < 2:
        return merged
    maps = [_object_id_map(partial, merged) for partial in partials]
    observations, conflicts, decisions = _union_observations(partials, maps, merged)
    decisions = merged_decisions(merged.box_decisions, decisions)
    states = list(merged.states) or _states(partials, maps)
    transitions = list(merged.transitions) or _transitions(partials, maps)
    relationships = list(merged.relationships) or _relationships(partials, maps)
    if merged.workflow_steps or merged.dependencies:
        steps, dependencies = list(merged.workflow_steps), list(merged.dependencies)
    else:
        steps, dependencies = _workflow(partials, maps)
    ambiguities = list(merged.ambiguities) or _ambiguities(partials, maps)
    if conflicts:
        ambiguities = [*ambiguities, *conflicts]
    unchanged = (
        observations == list(merged.observations)
        and states == list(merged.states)
        and transitions == list(merged.transitions)
        and relationships == list(merged.relationships)
        and steps == list(merged.workflow_steps)
        and dependencies == list(merged.dependencies)
        and ambiguities == list(merged.ambiguities)
        and decisions == list(merged.box_decisions)
    )
    if unchanged:
        return merged
    return merged.model_copy(
        update={
            "box_decisions": decisions,
            "observations": observations,
            "states": states,
            "transitions": transitions,
            "relationships": relationships,
            "workflow_steps": steps,
            "dependencies": dependencies,
            "ambiguities": ambiguities,
        }
    )


@dataclass(frozen=True)
class _Candidate:
    item: ObservationPrediction
    window_index: int
    position: int
    uncertain: bool


def _union_observations(
    partials: list[AnalysisDocument],
    maps: list[dict[str, str]],
    merged: AnalysisDocument,
) -> tuple[list[ObservationPrediction], list[AmbiguityPrediction], list[BoxDecision]]:
    """Choose one observation per object and timestamp.

    A box from a window that saw the frame outranks a box from the text-only
    merge, whatever evidence class the merge assigns itself. Image-bearing
    windows that disagree are ranked, not averaged. A window id that was not
    aligned to a merged part is left out: copying it through would put that
    part's box on a different object.
    """
    groups: dict[tuple, list[_Candidate]] = {}
    order: list[tuple] = []
    position = 0
    decisions: list[BoxDecision] = []
    for window_index, (partial, mapping) in enumerate(zip(partials, maps, strict=True)):
        uncertain_ids = {obj.local_id: obj.uncertain_identity for obj in partial.objects}
        for item in partial.observations:
            if item.object_local_id not in mapping:
                if item.bbox is not None:
                    decisions.append(
                        box_decision(
                            "window_merge",
                            "window_id_unaligned",
                            item,
                            detail=f"Window {window_index + 1} used this id, and it matched no merged part.",
                        )
                    )
                continue
            copy = item.model_copy(update={"object_local_id": mapping[item.object_local_id]})
            key = _key(copy)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(
                _Candidate(
                    item=copy,
                    window_index=window_index,
                    position=position,
                    uncertain=bool(uncertain_ids.get(item.object_local_id)) or _identity_uncertain(item),
                )
            )
            position += 1

    used_ids = {item.local_id for item in merged.ambiguities}
    for partial in partials:
        used_ids.update(item.local_id for item in partial.ambiguities)
    chosen: dict[tuple, ObservationPrediction] = {}
    protected: set[tuple] = set()
    conflicts: list[AmbiguityPrediction] = []
    for key in order:
        selected, conflict = _resolve_image_candidates(key, groups[key], used_ids)
        chosen[key] = selected
        if conflict is not None:
            decisions.extend(
                box_decision(
                    "window_merge",
                    "window_conflict_lost",
                    candidate.item,
                    detail=f"Window {candidate.window_index + 1} disagreed with the box that was kept.",
                )
                for candidate in groups[key]
                if candidate.item.bbox is not None
                and candidate.item is not selected
                and not _same_place(candidate.item, selected)
            )
        if _substantive(selected):
            protected.add(key)
        if conflict is not None:
            conflicts.append(conflict)
            used_ids.add(conflict.local_id)

    for item in merged.observations:
        key = _key(item)
        current = chosen.get(key)
        if current is None:
            chosen[key] = item
            order.append(key)
            continue
        # An image-backed box or an explicit absence stays. A text-only merge
        # still supplies the last box when no window localized that moment.
        if key in protected:
            if _disagrees(current, item):
                logger.info(
                    "text-only merge box ignored object=%s timestamp=%s kept=%s rejected=%s",
                    key[0],
                    key[1],
                    _box_list(current),
                    _box_list(item),
                )
                decisions.append(
                    box_decision(
                        "window_merge",
                        "text_merge_overridden",
                        item,
                        detail="The text-only merge disagreed with a window that saw this frame.",
                    )
                )
            continue
        if item.bbox is not None:
            chosen[key] = item
    return [chosen[key] for key in order], conflicts, decisions


def _resolve_image_candidates(
    key: tuple,
    candidates: list[_Candidate],
    used_ids: set[str],
) -> tuple[ObservationPrediction, AmbiguityPrediction | None]:
    substantive = [candidate for candidate in candidates if _substantive(candidate.item)]
    if not substantive:
        return candidates[0].item, None
    best = max(_rank(candidate) for candidate in substantive)
    top = sorted((candidate for candidate in substantive if _rank(candidate) == best), key=_stable)
    chosen = top[0]
    rivals = [
        candidate
        for candidate in substantive
        if candidate is not chosen and not _same_place(candidate.item, chosen.item)
    ]
    if not rivals:
        return chosen.item, None
    unresolved = any(not _same_place(candidate.item, chosen.item) for candidate in top[1:])
    reason = _selection_reason(chosen, rivals, unresolved)
    return chosen.item, _conflict_record(key, substantive, chosen, reason, unresolved, used_ids)


def _rank(candidate: _Candidate) -> tuple[int, int]:
    return (_EVIDENCE_RANK.get(candidate.item.evidence.classification, 0), 0 if candidate.uncertain else 1)


def _stable(candidate: _Candidate) -> tuple[int, int]:
    return (candidate.window_index, candidate.position)


def _selection_reason(chosen: _Candidate, rivals: list[_Candidate], unresolved: bool) -> str:
    if unresolved:
        return (
            "Equally supported image-bearing observations disagree, so the earliest one "
            "is kept and the conflict stays unresolved."
        )
    rival_classes = {candidate.item.evidence.classification for candidate in rivals}
    if (
        chosen.item.evidence.classification == EvidenceClass.DIRECTLY_OBSERVED
        and EvidenceClass.INFERRED in rival_classes
    ):
        note = ""
        if any(candidate.uncertain for candidate in rivals):
            note = " The rejected candidate also marks identity as uncertain."
        return "A directly observed image-bearing box is kept over an inferred one." + note
    if any(candidate.uncertain for candidate in rivals) and not chosen.uncertain:
        return "The identity-uncertain observation was not selected."
    if _explicit_absence(chosen.item):
        return "A directly observed absence is kept over a weaker contradictory box."
    return "The stronger image-bearing observation was kept."


def _conflict_record(
    key: tuple,
    candidates: list[_Candidate],
    chosen: _Candidate,
    reason: str,
    unresolved: bool,
    used_ids: set[str],
) -> AmbiguityPrediction:
    object_id, timestamp = key
    payload = {
        "object_local_id": object_id,
        "timestamp": timestamp,
        "candidate_count": len(candidates),
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
        "selected": _candidate_payload(chosen),
        "reason": reason,
        "unresolved": unresolved,
    }
    listed = "; ".join(
        f"window {candidate.window_index} {candidate.item.evidence.classification.value} "
        f"bbox={_box_list(candidate.item)} uncertain={candidate.uncertain}"
        for candidate in candidates
    )
    return AmbiguityPrediction(
        local_id=_fresh("Ambiguity", used_ids),
        kind=AmbiguityKind.CONFLICTING_EVIDENCE,
        description=(
            f"{object_id} at {timestamp:.3f}s has {len(candidates)} image-bearing observations that "
            f"do not agree. Selected window {chosen.window_index} "
            f"({chosen.item.evidence.classification.value}). {reason} Candidates: {listed}"
        ),
        time_start=timestamp,
        time_end=timestamp,
        object_local_ids=[object_id],
        alternative=reason,
        evidence=Evidence(
            classification=EvidenceClass.UNKNOWN,
            provenance=Provenance.DETERMINISTIC,
            note=json.dumps(payload, sort_keys=True),
            frame_timestamps=[timestamp],
        ),
    )


def _candidate_payload(candidate: _Candidate) -> dict:
    return {
        "window_index": candidate.window_index,
        "bbox": _box_list(candidate.item),
        "evidence": candidate.item.evidence.classification.value,
        "visibility": candidate.item.visibility.value,
        "identity_uncertain": candidate.uncertain,
    }


def _key(item: ObservationPrediction) -> tuple:
    return (item.object_local_id, round(float(item.timestamp), 3))


def _explicit_absence(item: ObservationPrediction) -> bool:
    return item.visibility == Visibility.ABSENT


def _substantive(item: ObservationPrediction) -> bool:
    return item.bbox is not None or _explicit_absence(item)


def _identity_uncertain(item: ObservationPrediction) -> bool:
    note = item.evidence.note.casefold()
    return "identity" in note and "uncertain" in note


def _same_place(left: ObservationPrediction, right: ObservationPrediction) -> bool:
    if _explicit_absence(left) and _explicit_absence(right):
        return True
    if _explicit_absence(left) or _explicit_absence(right):
        return False
    if left.bbox is None or right.bbox is None:
        return False
    return iou(_tuple(left), _tuple(right)) >= AGREEMENT_IOU


def _disagrees(kept: ObservationPrediction, incoming: ObservationPrediction) -> bool:
    if incoming.bbox is None:
        return False
    if _explicit_absence(kept) or kept.bbox is None:
        return True
    return iou(_tuple(kept), _tuple(incoming)) < AGREEMENT_IOU


def _tuple(item: ObservationPrediction) -> tuple[float, float, float, float]:
    box = item.bbox
    assert box is not None
    return (box.x, box.y, box.w, box.h)


def _box_list(item: ObservationPrediction) -> list[float] | None:
    if item.bbox is None or _explicit_absence(item):
        return None
    return [item.bbox.x, item.bbox.y, item.bbox.w, item.bbox.h]


def _object_id_map(partial: AnalysisDocument, merged: AnalysisDocument) -> dict[str, str]:
    from app.perception.ground import object_id_map

    return object_id_map(partial, merged)


def _obj(mapping: dict[str, str], local_id: str) -> str:
    return mapping.get(local_id, local_id)


def _states(partials: list[AnalysisDocument], maps: list[dict[str, str]]) -> list[StatePrediction]:
    seen: set[tuple] = set()
    rows: list[StatePrediction] = []
    for partial, mapping in zip(partials, maps, strict=True):
        for item in partial.states:
            copy = item.model_copy(update={"object_local_id": _obj(mapping, item.object_local_id)})
            key = (copy.object_local_id, round(copy.time_start, 3), round(copy.time_end, 3), copy.state_label)
            if key in seen:
                continue
            seen.add(key)
            rows.append(copy)
    return rows


def _transitions(partials: list[AnalysisDocument], maps: list[dict[str, str]]) -> list[TransitionPrediction]:
    seen: set[tuple] = set()
    rows: list[TransitionPrediction] = []
    for partial, mapping in zip(partials, maps, strict=True):
        for item in partial.transitions:
            copy = item.model_copy(update={"object_local_id": _obj(mapping, item.object_local_id)})
            key = (copy.object_local_id, round(copy.timestamp, 3), copy.from_state, copy.to_state)
            if key in seen:
                continue
            seen.add(key)
            rows.append(copy)
    return rows


def _relationships(partials: list[AnalysisDocument], maps: list[dict[str, str]]) -> list[RelationshipPrediction]:
    seen: set[tuple] = set()
    rows: list[RelationshipPrediction] = []
    for partial, mapping in zip(partials, maps, strict=True):
        for item in partial.relationships:
            other = _obj(mapping, item.other_local_id) if item.other_local_id else None
            copy = item.model_copy(
                update={
                    "subject_local_id": _obj(mapping, item.subject_local_id),
                    "other_local_id": other,
                }
            )
            key = (
                copy.subject_local_id,
                copy.other_local_id,
                copy.kind,
                round(copy.time_start, 3),
                round(copy.time_end, 3),
                copy.description,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(copy)
    return rows


def _fresh(prefix: str, used: set[str]) -> str:
    number = 1
    while f"{prefix}_{number}" in used:
        number += 1
    fresh = f"{prefix}_{number}"
    used.add(fresh)
    return fresh


def _workflow(
    partials: list[AnalysisDocument], maps: list[dict[str, str]]
) -> tuple[list[WorkflowStepPrediction], list[DependencyPrediction]]:
    used: set[str] = set()
    steps: list[WorkflowStepPrediction] = []
    dependencies: list[DependencyPrediction] = []
    for partial, mapping in zip(partials, maps, strict=True):
        step_ids: dict[str, str] = {}
        for item in partial.workflow_steps:
            new_id = item.local_id if item.local_id not in used else _fresh("Step", used)
            used.add(new_id)
            step_ids[item.local_id] = new_id
            steps.append(
                item.model_copy(
                    update={
                        "local_id": new_id,
                        "required_objects": [_obj(mapping, value) for value in item.required_objects],
                    }
                )
            )
        for item in partial.dependencies:
            dependencies.append(
                item.model_copy(
                    update={
                        "from_step": step_ids.get(item.from_step, item.from_step),
                        "to_step": step_ids.get(item.to_step, item.to_step),
                    }
                )
            )
    return steps, dependencies


def _ambiguities(partials: list[AnalysisDocument], maps: list[dict[str, str]]) -> list[AmbiguityPrediction]:
    used: set[str] = set()
    rows: list[AmbiguityPrediction] = []
    for partial, mapping in zip(partials, maps, strict=True):
        for item in partial.ambiguities:
            new_id = item.local_id if item.local_id not in used else _fresh("Ambiguity", used)
            used.add(new_id)
            rows.append(
                item.model_copy(
                    update={
                        "local_id": new_id,
                        "object_local_ids": [_obj(mapping, value) for value in item.object_local_ids],
                    }
                )
            )
    return rows
