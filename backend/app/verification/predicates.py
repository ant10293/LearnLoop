"""Judge completion predicates from tracked evidence.

A model proposal is not an input here. Absence is a failure only when the
anchor is visibly observed and the learned context of that view is present.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument
from app.verification.read import STRONG_RELATIONS, enum_value, evidence_class
from app.verification.schema import (
    CompletionPredicate,
    EvidenceRef,
    PredicateJudgement,
    RoleAssignment,
)


def judge_predicate(
    predicate: CompletionPredicate,
    execution: AnalysisDocument,
    roles: dict[str, RoleAssignment],
    *,
    video_id: str,
    attempted: bool,
) -> PredicateJudgement:
    if predicate.evidence_requirement != "required":
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="pass",
            note="This condition was not a required check.",
        )
    subject = roles.get(predicate.subject_role)
    target = roles.get(predicate.target_role) if predicate.target_role else None
    if _ambiguous(subject) or _ambiguous(target):
        return _cannot(predicate, "A role has competing instances, so the condition was not decided.")
    if predicate.type == "physical_relationship":
        return _relation(predicate, execution, subject, target, roles, video_id)
    return _state(predicate, execution, subject, video_id, attempted)


def _relation(predicate, execution, subject, target, roles, video_id: str) -> PredicateJudgement:
    if target is None or not target.object_local_ids:
        return _cannot(predicate, "The anchor role is unresolved, so a missing component is not claimed.")
    subject_ids = set(subject.object_local_ids if subject else [])
    target_ids = set(target.object_local_ids)
    window = _final_window(execution)
    supports = [
        item
        for item in _relations(execution, predicate.relation, subject_ids, target_ids, direct_only=True)
        if item.time_end >= window[0] - 1e-6
    ]
    if supports:
        corrected = _earlier_other_relation(execution, predicate.relation, subject_ids, target_ids, supports)
        evidence = [_relation_evidence(item, video_id) for item in supports]
        note = _support_note(predicate, corrected)
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="pass",
            note=note,
            corrected=corrected,
            evidence=evidence,
            support_onsets=[item.time_start for item in supports],
            error_category="TEMPORARY_ERROR_CORRECTED" if corrected else "",
        )
    occupants = _occupants(execution, predicate.relation, target_ids, subject_ids)
    if occupants and _visible(execution, target_ids, window):
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="fail",
            error_category="INCORRECT_COMPONENT",
            note=(
                "A different instance occupies the expected relation with the anchor. "
                "It did not match the expected role."
            ),
            evidence=[_relation_evidence(item, video_id) for item in occupants],
            support_onsets=[item.time_start for item in occupants],
        )
    if subject_ids:
        others = _other_strong(execution, predicate.relation, subject_ids, target_ids)
        if others and _visible(execution, target_ids, window):
            return PredicateJudgement(
                predicate_id=predicate.id,
                status="fail",
                error_category="INCORRECT_PLACEMENT",
                note="The mapped instances are in a different physical relation than the learned condition.",
                evidence=[_relation_evidence(item, video_id) for item in others],
                support_onsets=[item.time_start for item in others],
            )
        if _visible(execution, subject_ids, window):
            return _cannot(
                predicate,
                "The expected instance is visible, but the required relation is not directly observed.",
            )
    if _subject_absent(execution, subject_ids, window) and _context_visible(
        predicate, execution, roles, target_ids, window
    ):
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="fail",
            error_category="MISSING_COMPONENT",
            note=(
                "The anchor is visible in the learned context and the expected relation is absent. "
                "The expected role was not observed in that view."
            ),
            evidence=_visible_evidence(execution, target_ids, window, video_id),
        )
    return _cannot(
        predicate,
        "The recording does not show the expected location clearly enough to decide presence or absence.",
    )


def _state(predicate, execution, subject, video_id: str, attempted: bool) -> PredicateJudgement:
    if subject is None or not subject.object_local_ids:
        return _cannot(predicate, "The object role is unresolved, so the state was not judged.")
    expected = (predicate.state_label or predicate.to_state or "").strip().lower()
    object_ids = set(subject.object_local_ids)
    window = _final_window(execution)
    latest = _latest_state(execution, object_ids)
    if (
        latest
        and (latest.state_label or "").strip().lower() != expected
        and _visible(execution, object_ids, window)
    ):
        latest_label = (latest.state_label or "").strip().lower()
        prior = (predicate.from_state or "").strip().lower()
        if attempted or (prior and latest_label == prior):
            category = "INCOMPLETE_OPERATION" if attempted else "MISSING_STEP"
        else:
            category = "INCORRECT_PHYSICAL_STATE"
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="fail",
            error_category=category,
            note=f"The latest directly observed state is {latest.state_label}, not {expected or 'the expected state'}.",
            evidence=[
                EvidenceRef(
                    video_id=video_id,
                    timestamp=latest.time_end,
                    observation=f"Observed state {latest.state_label}",
                    evidence_class="DIRECTLY_OBSERVED",
                    object_local_ids=[latest.object_local_id],
                )
            ],
        )
    supports = []
    for transition in execution.transitions:
        if transition.object_local_id not in object_ids:
            continue
        if evidence_class(transition) != "DIRECTLY_OBSERVED":
            continue
        if (transition.to_state or "").strip().lower() == expected:
            supports.append(transition)
    intervals = [
        state
        for state in execution.states
        if state.object_local_id in object_ids
        and evidence_class(state) == "DIRECTLY_OBSERVED"
        and (state.state_label or "").strip().lower() == expected
    ]
    if supports or intervals:
        evidence = []
        onsets = []
        for transition in supports:
            onsets.append(transition.timestamp)
            evidence.append(
                EvidenceRef(
                    video_id=video_id,
                    timestamp=transition.timestamp,
                    observation=f"State became {transition.to_state}",
                    evidence_class="DIRECTLY_OBSERVED",
                    object_local_ids=[transition.object_local_id],
                )
            )
        for state in intervals:
            onsets.append(state.time_start)
            evidence.append(
                EvidenceRef(
                    video_id=video_id,
                    timestamp=state.time_start,
                    observation=f"State {state.state_label} holds",
                    evidence_class="DIRECTLY_OBSERVED",
                    object_local_ids=[state.object_local_id],
                )
            )
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="pass",
            note="A directly observed state matches the learned completion state.",
            evidence=evidence,
            support_onsets=onsets,
        )
    return _cannot(predicate, "No directly observed state establishes this completion condition.")


def _relations(execution, kind: str, subject_ids: set[str], target_ids: set[str], direct_only: bool):
    found = []
    for relation in execution.relationships:
        if enum_value(relation.kind) != kind or relation.other_local_id is None:
            continue
        if direct_only and evidence_class(relation) != "DIRECTLY_OBSERVED":
            continue
        ends = {relation.subject_local_id, relation.other_local_id}
        if subject_ids and ends == subject_ids | target_ids and subject_ids & ends and target_ids & ends:
            found.append(relation)
        elif subject_ids and subject_ids & ends and target_ids & ends:
            found.append(relation)
    return found


def _occupants(execution, kind: str, target_ids: set[str], subject_ids: set[str]):
    found = []
    for relation in execution.relationships:
        if enum_value(relation.kind) != kind or evidence_class(relation) != "DIRECTLY_OBSERVED":
            continue
        if relation.other_local_id is None:
            continue
        ends = {relation.subject_local_id, relation.other_local_id}
        if not ends & target_ids:
            continue
        other = ends - target_ids
        if not other:
            continue
        if subject_ids and other <= subject_ids:
            continue
        found.append(relation)
    return found


def _other_strong(execution, expected: str, subject_ids: set[str], target_ids: set[str]):
    found = []
    pair = subject_ids | target_ids
    for relation in execution.relationships:
        kind = enum_value(relation.kind)
        if kind == expected or kind not in STRONG_RELATIONS:
            continue
        if evidence_class(relation) != "DIRECTLY_OBSERVED" or relation.other_local_id is None:
            continue
        ends = {relation.subject_local_id, relation.other_local_id}
        if subject_ids & ends and target_ids & ends and ends <= pair | ends:
            found.append(relation)
    return found


def _earlier_other_relation(execution, expected, subject_ids, target_ids, supports) -> bool:
    if not supports:
        return False
    first = min(item.time_start for item in supports)
    return any(item.time_start < first - 1e-6 for item in _other_strong(execution, expected, subject_ids, target_ids))


def _context_visible(predicate, execution, roles: dict[str, RoleAssignment], anchor_ids: set[str], window) -> bool:
    if not predicate.context_relations or not _visible(execution, anchor_ids, window):
        return False
    for context in predicate.context_relations:
        subject = roles.get(context.get("subject_role", ""))
        target = roles.get(context.get("target_role", ""))
        if subject is None or target is None:
            continue
        if not subject.object_local_ids or not target.object_local_ids:
            continue
        matches = [
            item
            for item in _relations(
                execution,
                context.get("relation", ""),
                set(subject.object_local_ids),
                set(target.object_local_ids),
                direct_only=True,
            )
            if item.time_start <= window[1] and item.time_end >= window[0]
        ]
        if matches:
            return True
    return False


def _final_window(execution: AnalysisDocument) -> tuple[float, float]:
    times = [action.end for action in execution.actions]
    times.extend(observation.timestamp for observation in execution.observations)
    end = max(times) if times else 0.0
    if execution.actions:
        start = max(action.start for action in execution.actions)
    else:
        start = end * 0.7
    return start, end


def _visible(execution: AnalysisDocument, object_ids: set[str], window: tuple[float, float]) -> bool:
    return any(_is_visible(observation, object_ids, window) for observation in execution.observations)


def _is_visible(observation, object_ids: set[str], window: tuple[float, float]) -> bool:
    if observation.object_local_id not in object_ids:
        return False
    if evidence_class(observation) != "DIRECTLY_OBSERVED":
        return False
    if enum_value(observation.visibility) != "visible":
        return False
    return window[0] - 1e-6 <= observation.timestamp <= window[1] + 1e-6


def _subject_absent(execution, subject_ids: set[str], window) -> bool:
    """An unresolved role can still be a missing component.

    A mapped object is absent only when a direct observation says so. A gap,
    an occlusion, or a detector that returned nothing is not that observation.
    """
    if not subject_ids:
        return True
    explicit = False
    for observation in execution.observations:
        if observation.object_local_id not in subject_ids:
            continue
        if observation.timestamp < window[0] - 1e-6 or observation.timestamp > window[1] + 1e-6:
            continue
        visibility = enum_value(observation.visibility)
        if visibility in {"visible", "partial"}:
            return False
        if visibility == "absent" and evidence_class(observation) == "DIRECTLY_OBSERVED":
            note = (observation.evidence.note or "").lower()
            if any(token in note for token in ("not detected", "unobserved", "no sam", "empty detection")):
                continue
            explicit = True
    return explicit


def _visible_evidence(execution, object_ids: set[str], window, video_id: str) -> list[EvidenceRef]:
    evidence = []
    for observation in execution.observations:
        if not _is_visible(observation, object_ids, window):
            continue
        # The model saw the anchor, but its box is an estimate and is not cited as a measurement.
        measured = observation.bbox is not None and observation.geometry_origin != "vlm_estimated"
        bbox = observation.bbox.model_dump() if measured else None
        evidence.append(
            EvidenceRef(
                video_id=video_id,
                timestamp=observation.timestamp,
                observation="Anchor is visible in the final interval",
                evidence_class="DIRECTLY_OBSERVED",
                object_local_ids=[observation.object_local_id],
                bbox=bbox,
            )
        )
    return evidence


def _relation_evidence(relation, video_id: str) -> EvidenceRef:
    return EvidenceRef(
        video_id=video_id,
        timestamp=relation.time_start,
        observation=f"Relation {enum_value(relation.kind)} is directly observed",
        evidence_class=evidence_class(relation),
        object_local_ids=[relation.subject_local_id, relation.other_local_id or ""],
    )


def _latest_state(execution, object_ids: set[str]):
    rows = [
        state
        for state in execution.states
        if state.object_local_id in object_ids and evidence_class(state) == "DIRECTLY_OBSERVED"
    ]
    if not rows:
        return None
    return max(rows, key=lambda state: state.time_end)


def _ambiguous(assignment: RoleAssignment | None) -> bool:
    return assignment is not None and assignment.status == "ambiguous"


def _cannot(predicate: CompletionPredicate, note: str) -> PredicateJudgement:
    return PredicateJudgement(
        predicate_id=predicate.id,
        status="cannot",
        error_category="UNVERIFIABLE_OPERATION",
        note=note,
    )


def _support_note(predicate: CompletionPredicate, corrected: bool) -> str:
    if corrected:
        return "An earlier contradicting relation was followed by the expected relation."
    return f"The expected relation {predicate.relation} is directly observed."
