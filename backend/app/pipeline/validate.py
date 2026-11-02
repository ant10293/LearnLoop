"""Programmatic checks. They do not prove that an interpretation is correct."""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument, ValidationIssue
from app.schemas.enums import RECOGNIZED_ACTIONS


def validate_document(document: AnalysisDocument, duration: float) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    from app.perception.verify import audit_claims

    issues.extend(audit_claims(document))
    objects = {item.local_id for item in document.objects}
    actions = {item.local_id for item in document.actions}
    steps = {item.local_id for item in document.workflow_steps}
    if len(objects) != len(document.objects):
        issues.append(
            ValidationIssue(
                severity="error",
                code="duplicate_object_id",
                message="Two objects share a local id.",
                entity_type="object",
            )
        )
    if len(actions) != len(document.actions):
        issues.append(
            ValidationIssue(
                severity="error",
                code="duplicate_action_id",
                message="Two actions share a local id.",
                entity_type="action",
            )
        )

    def time_ok(value: float | None, entity: str, entity_id: str, label: str) -> None:
        if value is None:
            return
        if value < -0.05 or (duration > 0 and value > duration + 0.25):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="timestamp_out_of_range",
                    message=f"{label} {value:.3f}s is outside the recording (0–{duration:.3f}s).",
                    entity_type=entity,
                    entity_id=entity_id,
                )
            )

    for obs in document.observations:
        if obs.object_local_id not in objects:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_object",
                    message=f"Observation references unknown object {obs.object_local_id}.",
                    entity_type="observation",
                    entity_id=obs.object_local_id,
                )
            )
            obs.accepted = False
        time_ok(obs.timestamp, "observation", obs.object_local_id, "Observation time")
        if obs.evidence.classification.value == "DIRECTLY_OBSERVED" and not _anchored(obs.evidence):
            # The sighting time is the frame. Models often leave the evidence list empty.
            if obs.timestamp is not None:
                obs.evidence.frame_timestamps = [obs.timestamp]
            else:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="observation_without_evidence",
                        message="A directly observed claim has no frame or interval reference.",
                        entity_type="observation",
                        entity_id=obs.object_local_id,
                    )
                )

    for action in document.actions:
        time_ok(action.start, "action", action.local_id, "Action start")
        time_ok(action.end, "action", action.local_id, "Action end")
        if action.end + 1e-3 < action.start:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inverted_interval",
                    message="Action end precedes its start.",
                    entity_type="action",
                    entity_id=action.local_id,
                )
            )
            action.accepted = False
        for object_id in action.object_local_ids:
            if object_id not in objects:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_object",
                        message=f"Action {action.local_id} references unknown object {object_id}.",
                        entity_type="action",
                        entity_id=action.local_id,
                    )
                )
                action.accepted = False
        if action.action_type not in RECOGNIZED_ACTIONS:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unrecognized_action",
                    message=f"Action type '{action.action_type}' is outside the core vocabulary and was kept.",
                    entity_type="action",
                    entity_id=action.local_id,
                )
            )
        if action.evidence.classification.value == "DIRECTLY_OBSERVED" and not (
            action.evidence.frame_timestamps or action.evidence.time_start is not None
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="claim_without_evidence",
                    message="Action is marked directly observed without a traceable interval or frame.",
                    entity_type="action",
                    entity_id=action.local_id,
                )
            )
        if action.outcome.value == "completed" and action.evidence.classification.value == "UNKNOWN":
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="completion_without_evidence",
                    message="Completed outcome is paired with UNKNOWN evidence.",
                    entity_type="action",
                    entity_id=action.local_id,
                )
            )

    for step in document.workflow_steps:
        time_ok(step.time_start, "step", step.local_id, "Step start")
        time_ok(step.time_end, "step", step.local_id, "Step end")
        if (
            step.time_start is not None
            and step.time_end is not None
            and step.time_end + 1e-3 < step.time_start
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inverted_interval",
                    message="Workflow step end precedes its start.",
                    entity_type="step",
                    entity_id=step.local_id,
                )
            )
        for action_id in step.action_local_ids:
            if action_id not in actions:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_action",
                        message=f"Step {step.local_id} references unknown action {action_id}.",
                        entity_type="step",
                        entity_id=step.local_id,
                    )
                )
                step.accepted = False
        for object_id in step.required_objects:
            if object_id not in objects:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_object",
                        message=f"Step {step.local_id} requires unknown object {object_id}.",
                        entity_type="step",
                        entity_id=step.local_id,
                    )
                )

    graph: dict[str, list[str]] = {step_id: [] for step_id in steps}
    kept_dependencies = []
    for dep in document.dependencies:
        if _names_action(dep.from_step, actions, steps) or _names_action(dep.to_step, actions, steps):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="dependency_names_action",
                    message=(
                        f"Dependency {dep.from_step} → {dep.to_step} names an action. "
                        "Dependencies connect workflow steps, so this edge was dropped."
                    ),
                    entity_type="dependency",
                    entity_id=f"{dep.from_step}->{dep.to_step}",
                )
            )
            continue
        if dep.from_step not in steps or dep.to_step not in steps:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_step",
                    message=f"Dependency {dep.from_step} → {dep.to_step} references an unknown step.",
                    entity_type="dependency",
                    entity_id=f"{dep.from_step}->{dep.to_step}",
                )
            )
            dep.accepted = False
            kept_dependencies.append(dep)
            continue
        if dep.from_step == dep.to_step:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="self_dependency",
                    message=(
                        f"Step {dep.from_step} depends on itself. "
                        "A step has no ordering constraint against itself, so this edge was dropped."
                    ),
                    entity_type="dependency",
                    entity_id=dep.from_step,
                )
            )
            continue
        kept_dependencies.append(dep)
        graph[dep.from_step].append(dep.to_step)
    document.dependencies = kept_dependencies

    cycles = _cycles(graph)
    for cycle in cycles:
        issues.append(
            ValidationIssue(
                severity="error",
                code="cyclic_dependency",
                message="Workflow dependency cycle: " + " → ".join(cycle),
                entity_type="dependency",
            )
        )

    for state in document.states:
        if state.object_local_id not in objects:
            state.accepted = False
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_object",
                    message=f"State interval references unknown object {state.object_local_id}.",
                    entity_type="state",
                    entity_id=state.object_local_id,
                )
            )
        if state.time_end + 1e-3 < state.time_start:
            state.accepted = False
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inverted_interval",
                    message="State interval ends before it starts.",
                    entity_type="state",
                    entity_id=state.object_local_id,
                )
            )

    if document.coverage and document.coverage.modality == "sampled_frames":
        if document.coverage.frames_submitted <= 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="no_frames_submitted",
                    message="Sampled-frame modality recorded no submitted frames.",
                    entity_type="coverage",
                )
            )
    document.validation_issues = issues
    return issues


def _names_action(endpoint: str, actions: set[str], steps: set[str]) -> bool:
    if endpoint in steps:
        return False
    return endpoint in actions or endpoint.lower().startswith("action_")


def _cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    found: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str, stack: list[str]) -> None:
        if node in visiting:
            if node in stack:
                found.append(stack[stack.index(node) :] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, []):
            walk(nxt, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        walk(node, [])
    return found


def _anchored(evidence) -> bool:
    return bool(evidence.frame_timestamps) or evidence.time_start is not None


def fatal(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == "error" and issue.code == "unparseable" for issue in issues)
