"""Align execution events to workflow steps.

The search is dynamic programming over subsets of steps. Absolute timestamps
and durations are not scored. An action can support at most one step.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument
from app.verification.read import enum_value
from app.verification.schema import LearnedTemplate, RoleAssignment

_NEG = -1e9


def align_steps(
    template: LearnedTemplate,
    execution: AnalysisDocument,
    assignments: list[RoleAssignment],
) -> dict[str, dict]:
    by_role = {item.role_id: item for item in assignments}
    steps = list(template.steps)
    candidates = [_candidates(step, execution, by_role) for step in steps]
    chosen = _search(steps, candidates)
    aligned = {}
    for step in steps:
        option = chosen.get(step.step_id)
        aligned[step.step_id] = {
            "action_ids": [option["id"]] if option else [],
            "time": option["time"] if option else None,
            "end": option["end"] if option else None,
            "score": option["score"] if option else 0,
            "outcome": option["outcome"] if option else "",
            "status": "matched" if option else "unmatched",
        }
    return aligned


def _candidates(step, execution: AnalysisDocument, by_role: dict[str, RoleAssignment]) -> list[dict]:
    objects = set()
    for role_id in step.object_roles:
        assignment = by_role.get(role_id)
        if assignment is None:
            continue
        objects.update(assignment.object_local_ids)
    found = []
    for action in execution.actions:
        if not objects.intersection(action.object_local_ids):
            continue
        action_type = enum_value(action.action_type)
        score = 1.0 if step.expected_action_type and action_type == step.expected_action_type else 0.45
        found.append(
            {
                "id": action.local_id,
                "time": action.start,
                "end": action.end,
                "score": score,
                "outcome": enum_value(action.outcome),
            }
        )
    found.sort(key=lambda item: item["score"], reverse=True)
    return found


def _search(steps, candidates: list[list[dict]]) -> dict[str, dict]:
    count = len(steps)
    if count == 0:
        return {}
    if count > 12:
        return _greedy(steps, candidates)
    best = [_NEG] * (1 << count)
    chosen: list[dict | None] = [None] * (1 << count)
    best[0] = 0.0
    chosen[0] = {}
    for mask in range(1 << count):
        current = chosen[mask]
        if current is None:
            continue
        used = {item["id"] for item in current.values()}
        for index, step in enumerate(steps):
            bit = 1 << index
            if mask & bit:
                continue
            for option in candidates[index]:
                if option["id"] in used:
                    continue
                updated = mask | bit
                score = best[mask] + option["score"]
                if score <= best[updated]:
                    continue
                best[updated] = score
                nxt = dict(current)
                nxt[step.step_id] = option
                chosen[updated] = nxt
    winner = max(range(1 << count), key=lambda mask: best[mask])
    return chosen[winner] or {}


def _greedy(steps, candidates: list[list[dict]]) -> dict[str, dict]:
    used: set[str] = set()
    chosen: dict[str, dict] = {}
    pairs = []
    for index, options in enumerate(candidates):
        for option in options:
            pairs.append((option["score"], index, option))
    pairs.sort(reverse=True, key=lambda item: item[0])
    taken_steps: set[int] = set()
    for _score, index, option in pairs:
        if index in taken_steps or option["id"] in used:
            continue
        taken_steps.add(index)
        used.add(option["id"])
        chosen[steps[index].step_id] = option
    return chosen
