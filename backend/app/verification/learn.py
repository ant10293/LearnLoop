"""Build a reusable procedure template from one analyzed demonstration.

Observed order is kept. It becomes mandatory only when the source dependency
is already categorized as a necessary physical prerequisite.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument
from app.verification.read import (
    INCIDENTAL_ACTIONS,
    as_document,
    enum_value,
    evidence_class,
    median_area,
    overlaps,
    signature,
    state_labels,
)
from app.verification.schema import (
    CompletionPredicate,
    LearnedTemplate,
    ObjectRole,
    TemplateDependency,
    TemplateStep,
)

_NOTES = [
    "A single demonstration does not prove that every observed action is required.",
    "Observed temporal order is stored and does not constrain a later execution.",
    "Only dependencies categorized as necessary_physical constrain order.",
    "Passing a step requires an observable completion condition, not an action label alone.",
    "Appearance embeddings are not copied into the template, and role matching does not use them.",
]
_WEAK_RELATIONS = frozenset({"near", "occluded_by", "unknown", "held_by"})


def learn_template(
    document: AnalysisDocument | dict,
    *,
    name: str = "",
    promote_supported: bool = False,
    accept: bool = False,
    source_run_id: str = "",
    source_revision_id: str = "",
    source_video_id: str = "",
    results_first: bool = False,
) -> LearnedTemplate:
    demo = as_document(document)
    roles = _roles(demo)
    role_of = {object_id: role.role_id for role in roles for object_id in role.source_object_ids}
    owners = _relation_owners(demo)
    steps = [_step(demo, item, role_of, owners) for item in demo.workflow_steps]
    if results_first:
        for step in steps:
            _keep_results(step)
    if not steps:
        steps = _steps_from_actions(demo, role_of)
    _attach_dependencies(demo, steps)
    if promote_supported:
        for step in steps:
            if _supported(step):
                step.required = True
                step.requirement_status = "required"
    title = name or _title(demo)
    return LearnedTemplate(
        name=title,
        summary=demo.summary or "",
        source_run_id=source_run_id,
        source_revision_id=source_revision_id,
        source_video_id=source_video_id,
        acceptance_status="accepted" if accept else "draft",
        roles=roles,
        steps=steps,
        notes=list(_NOTES),
    )


def _keep_results(step: TemplateStep) -> None:
    """A placement is the result to check. Weak relations are dropped and a state label only supports it."""
    step.completion_conditions = [
        item
        for item in step.completion_conditions
        if not (item.type == "physical_relationship" and item.relation in _WEAK_RELATIONS)
        and not (item.type == "state_transition" and item.to_state == "other")
    ]
    for item in step.completion_conditions:
        item.context_relations = [c for c in item.context_relations if c.get("relation") not in _WEAK_RELATIONS]
    # A hand often hides the moment of joining. The result can still be checked on a later recording.
    if not any(item.evidence_requirement == "required" for item in step.completion_conditions):
        inferred = [item for item in step.completion_conditions if item.source_evidence_class == "INFERRED"]
        placements = [item for item in inferred if item.type == "physical_relationship"]
        for item in placements or inferred:
            item.evidence_requirement = "required"
    if any(
        item.type == "physical_relationship" and item.evidence_requirement == "required"
        for item in step.completion_conditions
    ):
        for item in step.completion_conditions:
            if item.type != "physical_relationship":
                item.evidence_requirement = "optional"


def _title(demo: AnalysisDocument) -> str:
    if demo.summary:
        return demo.summary.split(".")[0][:180]
    return "Learned procedure"


def _roles(demo: AnalysisDocument) -> list[ObjectRole]:
    needed = _procedure_objects(demo)
    roles = []
    kept = [item for item in demo.objects if item.local_id in needed]
    names = _distinct_labels(kept)
    for item in kept:
        labels = sorted(state_labels(demo, item.local_id))
        relations = sorted(signature(demo, item.local_id))
        roles.append(
            ObjectRole(
                role_id=f"role:{item.local_id}",
                label=names[item.local_id],
                description=item.description or "",
                visual_characteristics=item.visual_characteristics or "",
                source_object_ids=[item.local_id],
                required=True,
                uncertain=bool(item.uncertain_identity),
                cardinality="one",
                state_labels=labels,
                relation_signature=relations,
                reference_area=median_area(demo, item.local_id),
            )
        )
    return roles


def _distinct_labels(objects) -> dict[str, str]:
    """Identical parts share a label, so number them to tell them apart on the page."""
    totals: dict[str, int] = {}
    for item in objects:
        totals[item.label or ""] = totals.get(item.label or "", 0) + 1
    seen: dict[str, int] = {}
    names = {}
    for item in objects:
        label = item.label or ""
        seen[label] = seen.get(label, 0) + 1
        names[item.local_id] = f"{label} {seen[label]}" if label and totals[label] > 1 else label
    return names


def _procedure_objects(demo: AnalysisDocument) -> set[str]:
    needed: set[str] = set()
    for step in demo.workflow_steps:
        needed.update(step.required_objects)
        for action_id in step.action_local_ids:
            action = next((item for item in demo.actions if item.local_id == action_id), None)
            if action:
                needed.update(action.object_local_ids)
    if not demo.workflow_steps:
        for action in demo.actions:
            if enum_value(action.action_type) not in INCIDENTAL_ACTIONS:
                needed.update(action.object_local_ids)
    related = set(needed)
    for relation in demo.relationships:
        ends = {relation.subject_local_id, relation.other_local_id}
        if ends & needed and relation.other_local_id:
            related.update(end for end in ends if end)
    return related


def _step_span(demo: AnalysisDocument, item) -> tuple[list, list[str], float | None, float | None]:
    actions = [action for action in demo.actions if action.local_id in set(item.action_local_ids)]
    objects = list(item.required_objects)
    for action in actions:
        for object_id in action.object_local_ids:
            if object_id not in objects:
                objects.append(object_id)
    start = item.time_start if item.time_start is not None else (actions[0].start if actions else None)
    end = item.time_end if item.time_end is not None else (actions[-1].end if actions else None)
    return actions, objects, start, end


def _relation_owners(demo: AnalysisDocument) -> dict[int, str]:
    """The step that first achieved each arrangement. It stays true later but is not that later step's result."""
    owners: dict[int, str] = {}
    for relation in demo.relationships:
        if relation.other_local_id is None or relation.time_start is None:
            continue
        for item in demo.workflow_steps:
            _actions, objects, start, end = _step_span(demo, item)
            begin, finish = _window(demo, start, end)
            if {relation.subject_local_id, relation.other_local_id} <= set(objects) and begin <= relation.time_start <= finish:
                owners[id(relation)] = item.local_id
                break
    return owners


def _step(demo: AnalysisDocument, item, role_of: dict[str, str], owners: dict[int, str]) -> TemplateStep:
    actions, objects, start, end = _step_span(demo, item)
    action_type = enum_value(actions[0].action_type) if actions else ""
    incidental = bool(action_type) and action_type in INCIDENTAL_ACTIONS
    predicates = _predicates(demo, objects, role_of, start, end, owners, item.local_id)
    classification = evidence_class(item)
    notes = []
    if incidental:
        notes.append("This looks like an incidental motion. It is not treated as a requirement.")
    if classification != "DIRECTLY_OBSERVED":
        notes.append("The demonstration step itself is not directly observed.")
    if not predicates:
        notes.append("No observable completion condition was learned. An action label alone cannot pass this step.")
    status = "uncertain" if incidental or not predicates else "observed"
    uncertainty = 0.8 if incidental or not predicates else (0.35 if classification == "DIRECTLY_OBSERVED" else 0.7)
    return TemplateStep(
        step_id=item.local_id,
        name=item.name,
        required=False,
        requirement_status=status,
        observed=True,
        object_roles=[role_of[object_id] for object_id in objects if object_id in role_of],
        preconditions=list(item.preconditions),
        expected_action=item.expected_action or (actions[0].description if actions else ""),
        expected_action_type=action_type,
        completion_text=item.completion_condition or "",
        observed_text=item.observed_result or "",
        completion_conditions=predicates,
        verification_requirements={
            "action_evidence": "optional",
            "resulting_state_evidence": "required" if predicates else "unavailable",
        },
        uncertainty=max(item.uncertainty, uncertainty) if item.uncertainty else uncertainty,
        uncertainty_notes=" ".join(notes),
        incidental=incidental,
        demo_time_start=start,
        demo_time_end=end,
        parallel_group=item.parallel_group or "",
    )


def _steps_from_actions(demo: AnalysisDocument, role_of: dict[str, str]) -> list[TemplateStep]:
    steps = []
    for index, action in enumerate(demo.actions, start=1):
        action_type = enum_value(action.action_type)
        if action_type in INCIDENTAL_ACTIONS:
            continue
        objects = list(action.object_local_ids)
        predicates = _predicates(demo, objects, role_of, action.start, action.end)
        steps.append(
            TemplateStep(
                step_id=f"step_from_{action.local_id}" if not action.local_id.startswith("Step") else action.local_id,
                name=action.description or action_type or f"Step {index}",
                required=False,
                requirement_status="uncertain" if not predicates else "observed",
                object_roles=[role_of[object_id] for object_id in objects if object_id in role_of],
                expected_action=action.description or action_type,
                expected_action_type=action_type,
                completion_conditions=predicates,
                verification_requirements={
                    "action_evidence": "optional",
                    "resulting_state_evidence": "required" if predicates else "unavailable",
                },
                uncertainty=0.65,
                uncertainty_notes="Learned from an action because the analysis had no workflow steps.",
                demo_time_start=action.start,
                demo_time_end=action.end,
            )
        )
    return steps


def _predicates(demo, objects, role_of, start, end, owners=None, step_id="") -> list[CompletionPredicate]:
    predicates: list[CompletionPredicate] = []
    window = _window(demo, start, end)
    object_ids = set(objects)
    involved = [
        relation
        for relation in demo.relationships
        if relation.other_local_id and overlaps(relation.time_start, relation.time_end, window[0], window[1])
    ]
    relations = [
        relation
        for relation in involved
        if {relation.subject_local_id, relation.other_local_id} <= object_ids
        and (owners or {}).get(id(relation), step_id) == step_id
    ]
    for index, relation in enumerate(relations):
        subject = role_of.get(relation.subject_local_id)
        target = role_of.get(relation.other_local_id or "")
        if not subject or not target:
            continue
        classification = evidence_class(relation)
        kind = _result_kind(demo, relation, window)
        context = []
        for other in involved:
            if other is relation:
                continue
            if evidence_class(other) != "DIRECTLY_OBSERVED":
                continue
            other_subject = role_of.get(other.subject_local_id)
            other_target = role_of.get(other.other_local_id or "")
            if not other_subject or not other_target:
                continue
            shares = {relation.subject_local_id, relation.other_local_id} & {
                other.subject_local_id,
                other.other_local_id,
            }
            if not shares:
                continue
            context.append(
                {
                    "subject_role": other_subject,
                    "target_role": other_target,
                    "relation": enum_value(other.kind),
                }
            )
        predicates.append(
            CompletionPredicate(
                id=f"rel:{index}:{kind}",
                type="physical_relationship",
                subject_role=subject,
                target_role=target,
                relation=kind,
                evidence_requirement="required" if classification == "DIRECTLY_OBSERVED" else "optional",
                source_evidence_class=classification,
                context_relations=context,
            )
        )
    for index, transition in enumerate(demo.transitions):
        if transition.object_local_id not in objects:
            continue
        if not overlaps(transition.timestamp, transition.timestamp, window[0], window[1]):
            continue
        role = role_of.get(transition.object_local_id)
        if not role or not transition.to_state:
            continue
        classification = evidence_class(transition)
        predicates.append(
            CompletionPredicate(
                id=f"transition:{index}",
                type="state_transition",
                subject_role=role,
                from_state=transition.from_state or "",
                to_state=transition.to_state,
                state_label=transition.to_state,
                evidence_requirement="required" if classification == "DIRECTLY_OBSERVED" else "optional",
                source_evidence_class=classification,
            )
        )
    return predicates


def _result_kind(demo: AnalysisDocument, relation, window: tuple[float, float]) -> str:
    """A part the same answer calls inserted during this step sits inside its host."""
    kind = enum_value(relation.kind)
    if kind != "attached":
        return kind
    inserted = any(
        item.object_local_id == relation.subject_local_id
        and item.to_state == "inserted"
        and overlaps(item.timestamp, item.timestamp, window[0], window[1])
        for item in demo.transitions
    )
    return "inside" if inserted else kind


def _window(demo: AnalysisDocument, start, end) -> tuple[float, float]:
    if start is None and end is None:
        times = [action.end for action in demo.actions] or [0.0]
        return 0.0, max(times)
    begin = 0.0 if start is None else float(start)
    finish = begin if end is None else float(end)
    return begin, max(begin, finish)


def _attach_dependencies(demo: AnalysisDocument, steps: list[TemplateStep]) -> None:
    by_id = {step.step_id: step for step in steps}
    for dep in demo.dependencies:
        step = by_id.get(dep.to_step)
        if step is None or dep.from_step not in by_id:
            continue
        category = enum_value(dep.category)
        step.dependencies.append(
            TemplateDependency(
                step_id=dep.from_step,
                category=category,
                constrains_order=category == "necessary_physical",
                rationale=dep.rationale or "",
                evidence_class=evidence_class(dep),
            )
        )


def _supported(step: TemplateStep) -> bool:
    if step.incidental or not step.completion_conditions:
        return False
    return any(
        item.evidence_requirement == "required" and item.source_evidence_class == "DIRECTLY_OBSERVED"
        for item in step.completion_conditions
    )
