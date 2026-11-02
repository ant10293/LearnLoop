"""Map procedure roles onto the instances in another recording.

Instance ids are not compared across recordings. A shared label is one feature
among relations, appearance text, state, and box size. Close scores stay as
competing hypotheses.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument
from app.verification.read import jaccard, median_area, signature, state_labels, tokens
from app.verification.schema import LearnedTemplate, ObjectRole, RoleAssignment, RoleHypothesis

ASSIGN_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.12
HYPOTHESIS_FLOOR = 0.2

_WEIGHTS = {
    "label": 0.25,
    "description": 0.15,
    "visual": 0.15,
    "relation": 0.25,
    "state": 0.15,
    "size": 0.05,
}


def map_roles(
    template: LearnedTemplate,
    execution: AnalysisDocument,
    overrides: dict[str, str] | None = None,
) -> list[RoleAssignment]:
    overrides = overrides or {}
    areas = {item.local_id: median_area(execution, item.local_id) for item in execution.objects}
    ranked: dict[str, list[tuple[float, dict, str]]] = {}
    for role in template.roles:
        options = []
        for item in execution.objects:
            features = _features(role, item, execution, areas.get(item.local_id))
            score = round(sum(_WEIGHTS[name] * features[name] for name in _WEIGHTS), 4)
            options.append((score, features, item.local_id))
        options.sort(key=lambda option: option[0], reverse=True)
        ranked[role.role_id] = options

    claimed: set[str] = set()
    assignments: list[RoleAssignment] = []
    for role in template.roles:
        override = overrides.get(role.role_id)
        if override:
            claimed.add(override)
            assignments.append(
                RoleAssignment(
                    role_id=role.role_id,
                    label=role.label,
                    status="human",
                    object_local_ids=[override],
                    score=1,
                    hypotheses=_hypotheses(ranked.get(role.role_id, [])),
                    cardinality=role.cardinality,
                    note="A person assigned this role. Tracking rows were not rewritten.",
                )
            )
            continue
        assignments.append(_assign(role, ranked.get(role.role_id, []), claimed))
    return assignments


def assignment_index(assignments: list[RoleAssignment]) -> dict[str, RoleAssignment]:
    return {item.role_id: item for item in assignments}


def _assign(role: ObjectRole, options: list[tuple[float, dict, str]], claimed: set[str]) -> RoleAssignment:
    open_options = [option for option in options if option[2] not in claimed]
    hypotheses = _hypotheses(options)
    embedding_note = "Appearance embeddings were not in the analysis document."
    if role.cardinality == "group":
        chosen = [option for option in open_options if option[0] >= ASSIGN_THRESHOLD and _text_support(option[1])]
        for _score, _features, object_id in chosen:
            claimed.add(object_id)
        return RoleAssignment(
            role_id=role.role_id,
            label=role.label,
            status="assigned" if chosen else "unresolved",
            object_local_ids=[option[2] for option in chosen],
            score=chosen[0][0] if chosen else 0,
            hypotheses=hypotheses,
            cardinality="group",
            note=embedding_note if chosen else f"No group of instances cleared the role threshold. {embedding_note}",
        )
    viable = [option for option in open_options if _text_support(option[1])]
    if not viable or viable[0][0] < ASSIGN_THRESHOLD:
        return RoleAssignment(
            role_id=role.role_id,
            label=role.label,
            status="unresolved",
            hypotheses=hypotheses,
            cardinality=role.cardinality,
            note=(
                "No instance matched this role confidently. That is not evidence the component is missing. "
                + embedding_note
            ),
        )
    best = viable[0]
    second = viable[1] if len(viable) > 1 else None
    if second and best[0] - second[0] < AMBIGUITY_MARGIN:
        return RoleAssignment(
            role_id=role.role_id,
            label=role.label,
            status="ambiguous",
            score=best[0],
            hypotheses=hypotheses,
            cardinality=role.cardinality,
            note="Competing instances remain. They were not merged and neither was selected.",
        )
    claimed.add(best[2])
    return RoleAssignment(
        role_id=role.role_id,
        label=role.label,
        status="assigned",
        object_local_ids=[best[2]],
        score=best[0],
        hypotheses=hypotheses,
        cardinality=role.cardinality,
        note=embedding_note,
    )


def _features(role: ObjectRole, item, execution: AnalysisDocument, area: float | None) -> dict:
    label = jaccard(tokens(role.label), tokens(item.label))
    description = jaccard(tokens(role.description), tokens(item.description))
    visual = jaccard(tokens(role.visual_characteristics), tokens(item.visual_characteristics))
    relation = jaccard(set(role.relation_signature), signature(execution, item.local_id))
    state = jaccard(set(role.state_labels), state_labels(execution, item.local_id))
    role_area = role.reference_area
    size = 0.0
    if area is not None and role_area is not None and area > 0 and role_area > 0:
        size = min(area, role_area) / max(area, role_area)
    return {
        "label": label,
        "description": description,
        "visual": visual,
        "relation": relation,
        "state": state,
        "size": size,
    }


def _text_support(features: dict) -> bool:
    return features["label"] >= 0.2 or features["visual"] >= 0.2 or features["description"] >= 0.2


def _hypotheses(options: list[tuple[float, dict, str]]) -> list[RoleHypothesis]:
    kept = []
    for score, features, object_id in options:
        if score < HYPOTHESIS_FLOOR:
            continue
        kept.append(
            RoleHypothesis(
                object_local_id=object_id,
                score=score,
                features={name: round(value, 4) for name, value in features.items()},
            )
        )
    return kept[:6]
