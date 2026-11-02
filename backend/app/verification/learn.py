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
    "Appearance embeddings are not copied into the template. Role matching reads them only when a later analysis stored a cosine.",
]


def learn_template(
    document: AnalysisDocument | dict,
    *,
    name: str = "",
    promote_supported: bool = False,
    accept: bool = False,
    source_run_id: str = "",
    source_revision_id: str = "",
    source_video_id: str = "",
) -> LearnedTemplate:
    demo = as_document(document)
    roles = _roles(demo)
    role_of = {object_id: role.role_id for role in roles for object_id in role.source_object_ids}
    steps = [_step(demo, item, role_of) for item in demo.workflow_steps]
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


def _title(demo: AnalysisDocument) -> str:
    if demo.summary:
        return demo.summary.split(".")[0][:180]
    return "Learned procedure"


def _roles(demo: AnalysisDocument) -> list[ObjectRole]:
    needed = _procedure_objects(demo)
    roles = []
    for item in demo.objects:
        if item.local_id not in needed:
            continue
        labels = sorted(state_labels(demo, item.local_id))
        relations = sorted(signature(demo, item.local_id))
        roles.append(
            ObjectRole(
                role_id=f"role:{item.local_id}",
                label=item.label or "",
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


def _step(demo: AnalysisDocument, item, role_of: dict[str, str]) -> TemplateStep:
    actions = [action for action in demo.actions if action.local_id in set(item.action_local_ids)]
    action_type = enum_value(actions[0].action_type) if actions else ""
    incidental = bool(action_type) and action_type in INCIDENTAL_ACTIONS
    objects = list(item.required_objects)
    for action in actions:
        for object_id in action.object_local_ids:
            if object_id not in objects:
                objects.append(object_id)
    start = item.time_start if item.time_start is not None else (actions[0].start if actions else None)
    end = item.time_end if item.time_end is not None else (actions[-1].end if actions else None)
    predicates = _predicates(demo, objects, role_of, start, end)
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


def _predicates(demo, objects, role_of, start, end) -> list[CompletionPredicate]:
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
    ]
    for index, relation in enumerate(relations):
        subject = role_of.get(relation.subject_local_id)
        target = role_of.get(relation.other_local_id or "")
        if not subject or not target:
            continue
        classification = evidence_class(relation)
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
                id=f"rel:{index}:{enum_value(relation.kind)}",
                type="physical_relationship",
                subject_role=subject,
                target_role=target,
                relation=enum_value(relation.kind),
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
