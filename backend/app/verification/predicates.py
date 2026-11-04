"""Judge completion predicates from tracked evidence.

A model proposal is not an input here. Absence is a failure only when the
anchor is visibly observed and the learned context of that view is present.
A part that is never seen at all is not, by that alone, missing.
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
    appearance_seen: set[str] | None = None,
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
    unseen = _never_seen(predicate, execution, subject, target, appearance_seen or set(), video_id)
    if predicate.type == "physical_relationship":
        judgement = _relation(predicate, execution, subject, target, roles, video_id)
    else:
        judgement = _state(predicate, execution, subject, video_id, attempted)
    # Directly observed absence or misplacement still decides. Otherwise say why it stayed open.
    if unseen is not None and judgement.status != "fail":
        return unseen
    return judgement


def _relation(predicate, execution, subject, target, roles, video_id: str) -> PredicateJudgement:
    if target is None or not target.object_local_ids:
        return _cannot(predicate, "The object this part belongs on was not identified, so a missing component is not claimed.")
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
    swapped = [
        item for item in _reversed_relations(execution, predicate.relation, subject_ids, target_ids)
        if item.time_end >= window[0] - 1e-6
    ]
    if swapped and _seen(execution, target_ids, window):
        return PredicateJudgement(
            predicate_id=predicate.id,
            status="fail",
            error_category="INCORRECT_PLACEMENT",
            note="The mapped instances are arranged the opposite way around from the learned condition.",
            evidence=[_relation_evidence(item, video_id) for item in swapped],
            support_onsets=[item.time_start for item in swapped],
        )
    # A part that belongs to another step can share the anchor. Only a part with no role is a wrong component.
    in_procedure = {object_id for role in roles.values() for object_id in role.object_local_ids}
    occupants = [
        item
        for item in _occupants(execution, predicate.relation, target_ids, subject_ids | in_procedure)
        if item.time_end >= window[0] - 1e-6
    ]
    if occupants and _seen(execution, target_ids, window):
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
        if others and _seen(execution, target_ids, window):
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
                "The object this part belongs on is visible, and the rest of the arrangement matches, "
                "but the expected arrangement is absent. The expected part was not seen in that view."
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


_INVERSE = {"above": "below", "below": "above", "left_of": "right_of", "right_of": "left_of"}
# Hand occlusion makes these two hard to tell apart between recordings. Either one means the part is joined.
_JOINED = frozenset({"attached", "inside"})


def _same_result(kind: str, expected: str) -> bool:
    return kind == expected or (kind in _JOINED and expected in _JOINED)


def _directed(relation, kind: str, subject_ids: set[str], target_ids: set[str]) -> str:
    """For a directional kind: same, reversed, or other. "A below B" states "B above A"."""
    forward = enum_value(relation.kind) == kind
    subject, other = relation.subject_local_id, relation.other_local_id
    if (forward and subject in subject_ids and other in target_ids) or (
        not forward and other in subject_ids and subject in target_ids
    ):
        return "same"
    if (forward and subject in target_ids and other in subject_ids) or (
        not forward and other in target_ids and subject in subject_ids
    ):
        return "reversed"
    return "other"


def _reversed_relations(execution, kind: str, subject_ids: set[str], target_ids: set[str]):
    if kind not in _INVERSE or not subject_ids:
        return []
    return [
        relation
        for relation in execution.relationships
        if enum_value(relation.kind) in {kind, _INVERSE[kind]}
        and relation.other_local_id is not None
        and evidence_class(relation) == "DIRECTLY_OBSERVED"
        and _directed(relation, kind, subject_ids, target_ids) == "reversed"
    ]


def _relations(execution, kind: str, subject_ids: set[str], target_ids: set[str], direct_only: bool):
    if kind in _INVERSE and subject_ids:
        return [
            relation
            for relation in execution.relationships
            if enum_value(relation.kind) in {kind, _INVERSE[kind]}
            and relation.other_local_id is not None
            and (not direct_only or evidence_class(relation) == "DIRECTLY_OBSERVED")
            and _directed(relation, kind, subject_ids, target_ids) == "same"
        ]
    found = []
    for relation in execution.relationships:
        if not _same_result(enum_value(relation.kind), kind) or relation.other_local_id is None:
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
        if not _same_result(enum_value(relation.kind), kind) or evidence_class(relation) != "DIRECTLY_OBSERVED":
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
        if _same_result(kind, expected) or kind not in STRONG_RELATIONS:
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


def apply_human(
    step_id: str,
    predicate_ids: list[str],
    judgements: list[PredicateJudgement],
    records: list[dict],
    execution: AnalysisDocument,
    video_id: str,
) -> list[PredicateJudgement]:
    """Fold reviewer observations in as evidence.

    A record counts only when it cites a moment in the final interval, because
    a completion condition is about the end state. An earlier one is noted and
    leaves the machine reading in place.
    """
    start = _final_window(execution)[0]
    by_id = {item.predicate_id: item for item in judgements}
    targets = predicate_ids or [f"human:{step_id}"]
    for record in records:
        chosen = [record["predicate_id"]] if record.get("predicate_id") in targets else targets
        for predicate_id in chosen:
            current = by_id.get(predicate_id)
            if float(record["timestamp"]) < start - 1e-6:
                if current is not None:
                    current.note = (
                        f"{current.note} A reviewer observation at {record['timestamp']:.2f}s was not used: "
                        f"it is before the final interval, which starts at {start:.2f}s."
                    ).strip()
                continue
            by_id[predicate_id] = _human_judgement(predicate_id, record, current, video_id)
    return list(by_id.values())


def apply_reinspection(spec, machine: dict[str, list[PredicateJudgement]], findings: list[dict], video_id: str) -> dict:
    """Fold validated reinspection findings into predicates the first reading could not decide.

    A finding replaces only a "cannot" judgement: it never overturns a reading the first
    analysis already supported or contradicted. Supported and contradicted need directly
    observed, visible evidence at frames that were sent; anything else stays cannot, with
    the reason noted. A reinspected pass carries no onset, so it neither satisfies nor
    breaks an order constraint: the frames show an end state, not when it was reached.
    """
    updated = {step_id: [item.model_copy(deep=True) for item in items] for step_id, items in machine.items()}
    for finding in findings:
        items = updated.get(finding.get("step_id") or "")
        if items is None:
            continue
        for index, current in enumerate(items):
            if current.predicate_id != finding.get("predicate_id") or current.status != "cannot":
                continue
            items[index] = _reinspected_judgement(current, finding, video_id)
    return updated


def _reinspected_judgement(current: PredicateJudgement, finding: dict, video_id: str) -> PredicateJudgement:
    outcome = finding.get("outcome")
    visible = finding.get("visibility") == "visible" or (outcome == "supported" and finding.get("partial_accepted"))
    direct = finding.get("evidence_class") == "DIRECTLY_OBSERVED" and visible
    stamps = [float(value) for value in finding.get("evidence_timestamps") or []]
    model = finding.get("model_id") or "the analysis model"
    reason = (finding.get("reason") or "").strip()
    if outcome in {"supported", "contradicted"} and direct and stamps:
        status = "pass" if outcome == "supported" else "fail"
        words = "directly showed the condition holds" if status == "pass" else "directly showed the condition does not hold"
        note = f"A second look by {model} at {', '.join(f'{value:.2f}s' for value in stamps)} {words}. {reason}".strip()
        category = (current.error_category if current.error_category not in {"", "UNVERIFIABLE_OPERATION"} else "INCORRECT_PHYSICAL_STATE") if status == "fail" else ""
        return PredicateJudgement(
            predicate_id=current.predicate_id,
            status=status,
            error_category=category,
            note=note,
            reinspected=True,
            evidence=[
                EvidenceRef(
                    video_id=video_id,
                    timestamp=value,
                    observation=reason or note,
                    evidence_class="DIRECTLY_OBSERVED",
                    provenance="AI_REINSPECTION",
                )
                for value in stamps
            ],
        )
    limits = "; ".join(finding.get("limitations") or [])
    note = f"{current.note} A second look by {model} did not establish it: {reason or 'no direct view'}{f' ({limits})' if limits else ''}.".strip()
    return current.model_copy(update={"note": note, "reinspected": True})


def _human_judgement(predicate_id: str, record: dict, current, video_id: str) -> PredicateJudgement:
    answer = record["answer"]
    status = {"installed": "pass", "not_installed": "fail"}.get(answer, "cannot")
    words = {
        "installed": "A reviewer confirmed the condition holds",
        "not_installed": "A reviewer confirmed the condition does not hold",
    }.get(answer, "A reviewer could not determine the condition")
    category = ""
    if status == "fail":
        category = (current.error_category if current and current.status == "fail" else "") or "MISSING_COMPONENT"
    elif status == "cannot":
        category = "UNVERIFIABLE_OPERATION"
    note = f"{words} at {record['timestamp']:.2f}s."
    if record.get("note"):
        note += f" {record['note']}"
    return PredicateJudgement(
        predicate_id=predicate_id,
        status=status,
        error_category=category,
        note=note,
        human=True,
        evidence=[
            EvidenceRef(
                video_id=video_id,
                timestamp=float(record["timestamp"]),
                observation=note,
                evidence_class="DIRECTLY_OBSERVED" if status != "cannot" else "UNKNOWN",
                provenance="HUMAN_ANNOTATION",
            )
        ],
    )


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


def _seen(execution: AnalysisDocument, object_ids: set[str], window: tuple[float, float]) -> bool:
    """Enough to judge a relation that is itself directly observed. A hand on the part leaves it partly visible."""
    return any(
        _is_visible(observation, object_ids, window, kinds=("visible", "partial")) for observation in execution.observations
    )


def _is_visible(observation, object_ids: set[str], window: tuple[float, float], kinds: tuple[str, ...] = ("visible",)) -> bool:
    if observation.object_local_id not in object_ids:
        return False
    if evidence_class(observation) != "DIRECTLY_OBSERVED":
        return False
    if enum_value(observation.visibility) not in kinds:
        return False
    return window[0] - 1e-6 <= observation.timestamp <= window[1] + 1e-6


def _never_seen(predicate, execution: AnalysisDocument, subject, target, appearance_seen: set[str], video_id: str):
    """A part that appears nowhere in the attempt is unresolved, not missing.

    Nowhere means: no visible or partial observation at any time, no action, state change, or relation that
    involves it, and no box that matched its learned appearance. Not being seen is not evidence of absence: the
    part may have been fully occluded, outside the frame, or missed by the analysis model. So this never fails a
    condition on its own. It marks the condition CANNOT VERIFY with a NOT_OBSERVED note, which sends the step to
    the second look and the reviewer checkpoint. A FAIL needs a direct observation: the place the part belongs in
    view with an explicit report that it is absent, a positively observed wrong placement, or a reviewer's answer.
    A recording that does not show any of the work returns None here and is judged as usual.
    """
    subject_ids = set(subject.object_local_ids if subject else [])
    target_ids = set(target.object_local_ids if target else [])
    if not subject_ids:
        return None

    def seen(ids: set[str]) -> bool:
        return bool(ids & appearance_seen) or any(_involved(execution, object_id) for object_id in ids)

    others = {item.local_id for item in execution.objects} - subject_ids - target_ids
    if not seen(subject_ids):
        if not any(seen({object_id}) for object_id in others) and not (target_ids and seen(target_ids)):
            return None
        if target_ids and not seen(target_ids):
            subject_label = subject.label or "part"
            target_label = target.label or "part it goes with"
            missing, note = subject_ids | target_ids, (
                f"Neither the {subject_label} nor the {target_label} was seen at any point in the recording, "
                "while other parts of the attempt were. Not being seen is not proof they are missing: "
                "they may be hidden, out of frame, or missed by the analysis."
            )
        else:
            missing, note = subject_ids, (
                "This part was not seen at any point in the recording. Not being seen is not proof it is missing: "
                "it may be hidden, out of frame, or missed by the analysis."
            )
    elif target_ids and not seen(target_ids):
        label = target.label or "part it goes with"
        missing, note = target_ids, (
            f"The {label} was not seen at any point in the recording, so this condition was not decided. "
            "Not being seen is not proof it is missing."
        )
    else:
        return None
    return PredicateJudgement(
        predicate_id=predicate.id,
        status="cannot",
        error_category="UNVERIFIABLE_OPERATION",
        note=note,
        evidence=[
            EvidenceRef(
                video_id=video_id,
                timestamp=None,
                observation=note,
                evidence_class="NOT_OBSERVED",
                provenance="DETERMINISTIC",
                object_local_ids=sorted(missing),
            )
        ],
    )


def _involved(execution: AnalysisDocument, object_id: str) -> bool:
    for observation in execution.observations:
        if observation.object_local_id == object_id and enum_value(observation.visibility) in {"visible", "partial"}:
            return True
    if any(object_id in action.object_local_ids for action in execution.actions):
        return True
    if any(transition.object_local_id == object_id for transition in execution.transitions):
        return True
    return any(object_id in {relation.subject_local_id, relation.other_local_id} for relation in execution.relationships)


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
                observation="The object this part belongs on is visible at the end",
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
