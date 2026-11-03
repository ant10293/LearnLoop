"""Map procedure roles onto the instances in another recording.

Instance ids are not compared across recordings. Identity comes from what a
part is: its label, appearance text, state, and box size. Relations are what
verification checks, so they only break a tie between parts that look alike;
a part placed in the wrong order is still the same part. Close scores stay as
competing hypotheses.

An attempt analyzed against the reference list reuses an entry's label only
for the same kind of part. There the label decides identity: a part whose label
replaces a word of the role's label, other than a color, is a different part.
"""

from __future__ import annotations

import re

from app.schemas.analysis import AnalysisDocument
from app.verification.read import jaccard, median_area, signature, state_labels, tokens
from app.verification.schema import LearnedTemplate, ObjectRole, RoleAssignment, RoleHypothesis

ASSIGN_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.12
HYPOTHESIS_FLOOR = 0.2

_WEIGHTS = {
    "label": 0.35,
    "description": 0.2,
    "visual": 0.2,
    "state": 0.15,
    "size": 0.1,
}
# A color or finish word can change between two copies of the same part.
_COLORS = frozenset(
    {
        "black", "white", "gray", "grey", "silver", "gold", "golden", "brass", "bronze", "copper", "chrome",
        "dark", "light", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "tan", "beige",
        "shiny", "matte", "bright", "dull", "metal", "metallic", "steel", "zinc", "plated", "clear",
    }
)


def map_roles(
    template: LearnedTemplate,
    execution: AnalysisDocument,
    overrides: dict[str, str] | None = None,
) -> list[RoleAssignment]:
    overrides = overrides or {}
    areas = {item.local_id: median_area(execution, item.local_id) for item in execution.objects}
    referenced = _uses_reference_labels(template, execution)
    ranked: dict[str, list[tuple[float, dict, str]]] = {}
    for role in template.roles:
        options = []
        for item in execution.objects:
            if referenced and not same_kind(role.label, item.label):
                continue
            features = _features(role, item, execution, areas.get(item.local_id))
            score = _score(features)
            if referenced and _base_label(role.label) == _base_label(item.label):
                # The analysis reused this entry's label, which it does only for the same kind of part.
                score = max(score, ASSIGN_THRESHOLD)
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
        assignments.append(_assign(role, ranked.get(role.role_id, []), claimed, interchangeable=_has_twin(role, template)))
    return assignments


def assignment_index(assignments: list[RoleAssignment]) -> dict[str, RoleAssignment]:
    return {item.role_id: item for item in assignments}


def _base_label(label: str) -> str:
    return re.sub(r"\s+\d+$", "", label or "").strip().lower()


def same_kind(role_label: str, label: str) -> bool:
    """Words left out are a shorter name for the same part. A word replaced, other than a color, names another part."""
    role_words = tokens(_base_label(role_label)) - _COLORS
    words = tokens(_base_label(label)) - _COLORS
    if not role_words or not words:
        return bool(tokens(_base_label(role_label)) & tokens(_base_label(label)))
    return role_words <= words or words <= role_words


def _uses_reference_labels(template: LearnedTemplate, execution: AnalysisDocument) -> bool:
    """True when the attempt was labeled from this template's reference list."""
    bases = {_base_label(role.label) for role in template.roles if role.label}
    return any(_base_label(item.label) in bases for item in execution.objects)


def _has_twin(role: ObjectRole, template: LearnedTemplate) -> bool:
    """Same part doing the same job. A left pin and a right pin differ by relation, so they are not twins."""
    return any(
        other.role_id != role.role_id
        and _base_label(other.label) == _base_label(role.label)
        and set(other.relation_signature) == set(role.relation_signature)
        for other in template.roles
    )


def _assign(
    role: ObjectRole,
    options: list[tuple[float, dict, str]],
    claimed: set[str],
    *,
    interchangeable: bool = False,
) -> RoleAssignment:
    open_options = [option for option in options if option[2] not in claimed]
    hypotheses = _hypotheses(options)
    embedding_note = "Role matching does not use appearance embeddings."
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
    close = [option for option in viable if best[0] - option[0] < AMBIGUITY_MARGIN]
    if len(close) > 1:
        # Parts that look alike are told apart by where they sit, if one clearly fits the learned arrangement better.
        close.sort(key=lambda option: option[1]["relation"], reverse=True)
        if close[0][1]["relation"] - close[1][1]["relation"] >= AMBIGUITY_MARGIN:
            best, close = close[0], [close[0]]
    second = close[1] if len(close) > 1 else None
    # Identical parts have no right instance to pick, so a tie between them is not a doubt.
    if second and not interchangeable:
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
    features = {
        "label": label,
        "description": description,
        "visual": visual,
        "state": state,
    }
    role_area = role.reference_area
    if area is not None and role_area is not None and area > 0 and role_area > 0:
        features["size"] = min(area, role_area) / max(area, role_area)
    # Kept for breaking ties and for the record. It is not part of the identity score.
    features["relation"] = relation
    return features


def _score(features: dict) -> float:
    """A feature with no measurement is left out. It is not evidence against the match."""
    scored = {name: value for name, value in features.items() if name in _WEIGHTS}
    available = sum(_WEIGHTS[name] for name in scored)
    return round(sum(_WEIGHTS[name] * value for name, value in scored.items()) / available, 4)


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
