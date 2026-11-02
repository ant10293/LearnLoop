"""Fold a repeated verb on one interval into a single action.

A model often records each hand as its own action even when both happen
between the same two frames. The inspector then shows the opening verb twice
at the same clock time. Those records are one act: the objects and the
wording stay, and later steps point at the action that remains.
"""

from __future__ import annotations

from app.schemas.analysis import (
    ActionPrediction,
    AmbiguityPrediction,
    AnalysisDocument,
    Evidence,
    WorkflowStepPrediction,
)
from app.schemas.enums import ActionOutcome, EvidenceClass

# The inspector prints times to a hundredth of a second. Two spans that
# display as the same range are the same interval.
_SPAN = 2

_OUTCOME_RANK = {
    ActionOutcome.COMPLETED: 3,
    ActionOutcome.ATTEMPTED: 2,
    ActionOutcome.FAILED: 1,
    ActionOutcome.UNKNOWN: 0,
}

_EVIDENCE_RANK = {
    EvidenceClass.DIRECTLY_OBSERVED: 2,
    EvidenceClass.INFERRED: 1,
    EvidenceClass.UNKNOWN: 0,
}


def collapse_repeated_actions(document: AnalysisDocument) -> AnalysisDocument:
    actions = list(document.actions)
    if len(actions) < 2:
        return document
    kept: list[ActionPrediction] = []
    redirect: dict[str, str] = {}
    for action in actions:
        match = next((item for item in kept if _same_act(item, action)), None)
        if match is None:
            kept.append(action)
            continue
        redirect[action.local_id] = match.local_id
        kept[kept.index(match)] = _merge(match, action)
    if not redirect:
        return document
    return document.model_copy(
        update={
            "actions": kept,
            "workflow_steps": [_retarget_step(step, redirect) for step in document.workflow_steps],
            "ambiguities": [_retarget_ambiguity(item, redirect) for item in document.ambiguities],
        }
    )


def _same_act(left: ActionPrediction, right: ActionPrediction) -> bool:
    if left.action_type.strip().lower() != right.action_type.strip().lower():
        return False
    return _stamp(left.start) == _stamp(right.start) and _stamp(left.end) == _stamp(right.end)


def _stamp(value: float) -> float:
    return round(float(value), _SPAN)


def _merge(kept: ActionPrediction, extra: ActionPrediction) -> ActionPrediction:
    outcome = kept.outcome
    if _OUTCOME_RANK.get(extra.outcome, 0) > _OUTCOME_RANK.get(kept.outcome, 0):
        outcome = extra.outcome
    actor = _join_actors(kept.actor, extra.actor)
    return kept.model_copy(
        update={
            "description": _join_text(kept.description, extra.description),
            "actor": actor,
            "object_local_ids": _union(kept.object_local_ids, extra.object_local_ids),
            "initial_state": _join_text(kept.initial_state, extra.initial_state),
            "final_state": _join_text(kept.final_state, extra.final_state),
            "outcome": outcome,
            "uncertainty": min(kept.uncertainty, extra.uncertainty),
            "evidence": _merge_evidence(kept.evidence, extra.evidence),
        }
    )


def _union(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys([*left, *right]))


def _join_text(left: str, right: str) -> str:
    left, right = (left or "").strip(), (right or "").strip()
    if not right or right in left:
        return left
    if not left:
        return right
    return f"{left} {right}"


def _join_actors(left: str | None, right: str | None) -> str | None:
    text = _join_text(left or "", right or "")
    return text or None


def _merge_evidence(kept: Evidence, extra: Evidence) -> Evidence:
    primary, other = (kept, extra)
    if _EVIDENCE_RANK.get(extra.classification, 0) > _EVIDENCE_RANK.get(kept.classification, 0):
        primary, other = extra, kept
    note = _join_text(primary.note, other.note)
    stamps = list(dict.fromkeys([*primary.frame_timestamps, *other.frame_timestamps]))
    return primary.model_copy(update={"note": note, "frame_timestamps": stamps})


def _retarget(ids: list[str], redirect: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for item in ids:
        mapped = redirect.get(item, item)
        if mapped not in seen:
            seen.append(mapped)
    return seen


def _retarget_step(step: WorkflowStepPrediction, redirect: dict[str, str]) -> WorkflowStepPrediction:
    ids = _retarget(step.action_local_ids, redirect)
    if ids == list(step.action_local_ids):
        return step
    return step.model_copy(update={"action_local_ids": ids})


def _retarget_ambiguity(item: AmbiguityPrediction, redirect: dict[str, str]) -> AmbiguityPrediction:
    ids = _retarget(item.action_local_ids, redirect)
    if ids == list(item.action_local_ids):
        return item
    return item.model_copy(update={"action_local_ids": ids})
