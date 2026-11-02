"""Evidence-based execution verification.

The model may propose a reading. The verdict is the deterministic reading of
tracked relations, states, dependencies, and visibility.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument
from app.verification.align import align_steps
from app.verification.predicates import judge_predicate
from app.verification.read import INCIDENTAL_ACTIONS, as_document, enum_value
from app.verification.roles import assignment_index, map_roles
from app.verification.schema import (
    EvidenceRef,
    LearnedTemplate,
    PredicateJudgement,
    RoleAssignment,
    StepResult,
    TemplateStep,
)

_PRIORITY = {
    "INCORRECT_ORDER": 0,
    "INCORRECT_COMPONENT": 1,
    "INCORRECT_PLACEMENT": 2,
    "INCOMPLETE_OPERATION": 3,
    "INCORRECT_PHYSICAL_STATE": 4,
    "MISSING_COMPONENT": 5,
    "MISSING_STEP": 6,
    "UNEXPECTED_ACTION": 7,
    "UNVERIFIABLE_OPERATION": 8,
    "TEMPORARY_ERROR_CORRECTED": 9,
}
_TIMING = (
    "Duration, pauses, camera angle, and absolute timestamps are not errors. "
    "Order is checked only for dependencies categorized as necessary_physical."
)


def verify_execution(
    template: LearnedTemplate | dict,
    execution: AnalysisDocument | dict,
    *,
    proposals: dict | None = None,
    strategy: str = "hybrid",
    role_overrides: dict[str, str] | None = None,
    execution_video_id: str = "",
    demonstration_video_id: str = "",
    analysis_complete: bool = True,
) -> dict:
    spec = template if isinstance(template, LearnedTemplate) else LearnedTemplate.model_validate(template)
    document = as_document(execution)
    if not analysis_complete:
        return _incomplete(spec, strategy)
    assignments = map_roles(spec, document, role_overrides)
    aligned = align_steps(spec, document, assignments)
    judgements = {
        step.step_id: _judge_step(step, document, assignments, aligned, execution_video_id)
        for step in spec.steps
    }
    steps = [
        _step_result(
            step,
            judgements[step.step_id],
            judgements,
            assignments,
            aligned.get(step.step_id, {}),
            proposals.get(step.step_id) if proposals else None,
            strategy,
            demonstration_video_id,
        )
        for step in spec.steps
    ]
    unexpected = _unexpected(document, aligned, execution_video_id)
    overall = _overall(steps)
    return {
        "status": "completed",
        "overall": overall,
        "strategy": strategy,
        "timing_policy": _TIMING,
        "template_acceptance": spec.acceptance_status,
        "execution_video_id": execution_video_id,
        "demonstration_video_id": demonstration_video_id,
        "score": _score(steps),
        "roles": [item.model_dump() for item in assignments],
        "steps": [item.model_dump() for item in steps],
        "unexpected_actions": unexpected,
        "warnings": _warnings(spec, assignments),
        "provider_invoked": False,
    }


def _judge_step(step: TemplateStep, document, assignments, aligned, video_id: str) -> list[PredicateJudgement]:
    by_role = assignment_index(assignments)
    outcome = (aligned.get(step.step_id) or {}).get("outcome") or ""
    attempted = outcome in {"attempted", "failed"}
    return [
        judge_predicate(predicate, document, by_role, video_id=video_id, attempted=attempted)
        for predicate in step.completion_conditions
    ]


def _step_result(
    step: TemplateStep,
    judgements: list[PredicateJudgement],
    all_judgements: dict[str, list[PredicateJudgement]],
    assignments: list[RoleAssignment],
    alignment: dict,
    proposal: dict | None,
    strategy: str,
    demo_video_id: str,
) -> StepResult:
    by_role = assignment_index(assignments)
    required = [item for item in judgements if _is_required(step, item)]
    order = _order(step, required, all_judgements)
    verdict, category, corrected = _decide(step, required, order)
    explanation = _explanation(step, required, order, verdict)
    machine = verdict
    verdict, category, explanation = _strategy(strategy, verdict, category, explanation, proposal, alignment)
    state = _execution_state(verdict, corrected or order == "corrected", alignment)
    if corrected or order == "corrected":
        if verdict == "PASS":
            category = "TEMPORARY_ERROR_CORRECTED"
    objects = []
    for role_id in step.object_roles:
        assignment = by_role.get(role_id)
        if assignment:
            objects.extend(assignment.object_local_ids)
    evidence = [item for judgement in required for item in judgement.evidence]
    execution_time = evidence[0].timestamp if evidence else alignment.get("time")
    ambiguous = any(
        (by_role.get(role_id) and by_role[role_id].status == "ambiguous") for role_id in step.object_roles
    )
    review = verdict == "CANNOT_VERIFY" or ambiguous or bool(step.uncertainty_notes)
    return StepResult(
        step_id=step.step_id,
        name=step.name,
        required=step.required,
        requirement_status=step.requirement_status,
        verdict=verdict,
        execution_state=state,
        error_category=category,
        expected=_expected(step, by_role),
        observed=_observed(required, verdict),
        explanation=explanation,
        uncertainty_notes=step.uncertainty_notes,
        review_required=review and step.required,
        matched_action_ids=list(alignment.get("action_ids") or []),
        matched_object_ids=objects,
        evidence=evidence,
        demo_evidence=[
            EvidenceRef(
                video_id=demo_video_id,
                timestamp=step.demo_time_start,
                observation=step.expected_action or step.name,
                evidence_class="DIRECTLY_OBSERVED" if step.completion_conditions else "UNKNOWN",
                side="demonstration",
            )
        ],
        corrected=bool(corrected or order == "corrected") and verdict == "PASS",
        alignment=alignment.get("status") or "unmatched",
        demo_time_start=step.demo_time_start,
        demo_time_end=step.demo_time_end,
        execution_time=execution_time,
        predicates=judgements,
        machine_verdict=machine,
    )


def _is_required(step: TemplateStep, judgement: PredicateJudgement) -> bool:
    predicate = next((item for item in step.completion_conditions if item.id == judgement.predicate_id), None)
    return predicate is not None and predicate.evidence_requirement == "required"


def _order(step: TemplateStep, judgements: list[PredicateJudgement], all_judgements) -> str | None:
    onsets = [stamp for item in judgements if item.status == "pass" for stamp in item.support_onsets]
    if not onsets:
        return None
    earliest = min(onsets)
    violated = False
    corrected = False
    for dep in step.dependencies:
        if not dep.constrains_order:
            continue
        pred_onsets = [
            stamp
            for item in all_judgements.get(dep.step_id, [])
            if item.status == "pass"
            for stamp in item.support_onsets
        ]
        if not pred_onsets:
            continue
        pred_time = min(pred_onsets)
        if earliest + 1e-6 < pred_time:
            if any(stamp >= pred_time - 1e-6 for stamp in onsets):
                corrected = True
            else:
                violated = True
    if violated and not corrected:
        return "violated"
    if corrected:
        return "corrected"
    return None


def _decide(step: TemplateStep, judgements: list[PredicateJudgement], order: str | None):
    if step.requirement_status == "optional":
        return "NOT_APPLICABLE", "", False
    if order == "violated":
        return "FAIL", "INCORRECT_ORDER", False
    failures = [item for item in judgements if item.status == "fail"]
    if failures:
        chosen = min(failures, key=lambda item: _PRIORITY.get(item.error_category, 99))
        return "FAIL", chosen.error_category, False
    if not judgements:
        return "CANNOT_VERIFY", "UNVERIFIABLE_OPERATION", False
    if any(item.status == "cannot" for item in judgements):
        return "CANNOT_VERIFY", "UNVERIFIABLE_OPERATION", False
    corrected = any(item.corrected for item in judgements) or order == "corrected"
    return "PASS", "TEMPORARY_ERROR_CORRECTED" if corrected else "", corrected


def _strategy(strategy, verdict, category, explanation, proposal, alignment):
    proposal = proposal or {}
    proposed = proposal.get("verdict") or ""
    evidence = proposal.get("evidence_class") or ""
    if strategy == "tracked":
        return verdict, category, explanation
    if strategy in {"vlm_only", "vlm_align"}:
        if strategy == "vlm_align" and alignment.get("status") != "matched":
            return "CANNOT_VERIFY", "UNVERIFIABLE_OPERATION", "No aligned execution event supports the model proposal."
        if proposed not in {"PASS", "FAIL"} or evidence != "DIRECTLY_OBSERVED":
            return (
                "CANNOT_VERIFY",
                "UNVERIFIABLE_OPERATION",
                "The model proposal has no directly observed support, so the verdict was not taken from it.",
            )
        return proposed, proposal.get("error_category") or "", proposal.get("note") or explanation
    if proposed in {"PASS", "FAIL"} and (evidence != "DIRECTLY_OBSERVED" or verdict == "CANNOT_VERIFY"):
        explanation = (
            explanation
            + " A model proposal was not accepted without directly observed support from the execution."
        ).strip()
    return verdict, category, explanation


def _execution_state(verdict: str, corrected: bool, alignment: dict) -> str:
    if corrected and verdict == "PASS":
        return "corrected"
    if verdict == "PASS":
        return "completed"
    outcome = alignment.get("outcome") or ""
    if outcome in {"attempted", "failed"} or (alignment.get("action_ids") and verdict == "FAIL"):
        return "attempted"
    if verdict == "FAIL":
        return "not_attempted"
    return "unknown"


def _expected(step: TemplateStep, roles: dict[str, RoleAssignment]) -> str:
    parts = []
    for predicate in step.completion_conditions:
        if predicate.evidence_requirement != "required":
            continue
        subject = _label(roles, predicate.subject_role)
        if predicate.type == "physical_relationship":
            target = _label(roles, predicate.target_role)
            parts.append(f"{subject} {predicate.relation} {target}".strip())
        else:
            parts.append(f"{subject} reaches {predicate.state_label or predicate.to_state}".strip())
    return "; ".join(parts) or step.expected_action or step.name


def _label(roles: dict[str, RoleAssignment], role_id: str) -> str:
    assignment = roles.get(role_id)
    if assignment and assignment.label:
        return assignment.label
    return role_id


def _observed(judgements: list[PredicateJudgement], verdict: str) -> str:
    notes = [item.note for item in judgements if item.note]
    if notes:
        return " ".join(notes)
    if verdict == "NOT_APPLICABLE":
        return "The step is optional."
    return "No directly observed completion evidence was assembled."


def _explanation(step, judgements, order, verdict: str) -> str:
    if order == "violated":
        return "A necessary prerequisite was completed after this step's supporting evidence."
    if not step.completion_conditions:
        return "No observable completion condition was learned, so the step cannot pass from an action label."
    notes = [item.note for item in judgements if item.note]
    if notes:
        return " ".join(notes)
    return f"Verdict {verdict} from the learned completion conditions."


def _unexpected(document: AnalysisDocument, aligned: dict, video_id: str) -> list[dict]:
    matched = {action_id for item in aligned.values() for action_id in item.get("action_ids") or []}
    found = []
    for action in document.actions:
        action_type = enum_value(action.action_type)
        if action.local_id in matched or action_type in INCIDENTAL_ACTIONS:
            continue
        found.append(
            {
                "action_id": action.local_id,
                "action_type": action_type,
                "start": action.start,
                "end": action.end,
                "error_category": "UNEXPECTED_ACTION",
                "failure": False,
                "note": "This action is outside the aligned procedure. It is not a failure by itself.",
                "evidence": [
                    {
                        "video_id": video_id,
                        "timestamp": action.start,
                        "observation": action.description or action_type,
                        "evidence_class": enum_value(action.evidence.classification),
                        "side": "execution",
                    }
                ],
            }
        )
    return found


def _overall(steps: list[StepResult]) -> str:
    required = [step for step in steps if step.required and step.verdict != "NOT_APPLICABLE"]
    if any(step.verdict == "FAIL" for step in required):
        return "FAIL"
    if any(step.verdict == "CANNOT_VERIFY" for step in required):
        return "INCONCLUSIVE"
    if required and all(step.verdict == "PASS" for step in required):
        return "PASS"
    return "INCONCLUSIVE"


def _score(steps: list[StepResult]) -> dict:
    required = [step for step in steps if step.required and step.verdict != "NOT_APPLICABLE"]
    verified = sum(step.verdict == "PASS" for step in required)
    failed = sum(step.verdict == "FAIL" for step in required)
    unresolved = sum(step.verdict == "CANNOT_VERIFY" for step in required)
    applicable = [step for step in steps if step.verdict == "NOT_APPLICABLE"]
    denominator = verified + failed + unresolved
    return {
        "verified": verified,
        "failed": failed,
        "unresolved": unresolved,
        "not_applicable": len(applicable),
        "value": round(verified / denominator, 4) if denominator else None,
        "explanation": (
            "Value is verified required steps divided by verified, failed, and unresolved required steps. "
            "CANNOT_VERIFY is unresolved and is not counted as success."
        ),
    }


def _warnings(spec: LearnedTemplate, assignments: list[RoleAssignment]) -> list[str]:
    warnings = []
    if spec.acceptance_status != "accepted":
        warnings.append("This template is still a draft. Accept it before treating the report as a reference.")
    if any(item.status == "ambiguous" for item in assignments):
        warnings.append("At least one role has competing instance hypotheses.")
    if any(item.status == "unresolved" for item in assignments):
        warnings.append("At least one role has no confident instance. That is not a missing-component verdict.")
    return warnings


def _incomplete(spec: LearnedTemplate, strategy: str) -> dict:
    return {
        "status": "incomplete",
        "overall": "INCOMPLETE",
        "strategy": strategy,
        "timing_policy": _TIMING,
        "template_acceptance": spec.acceptance_status,
        "score": {
            "verified": 0,
            "failed": 0,
            "unresolved": 0,
            "not_applicable": 0,
            "value": None,
            "explanation": "Analysis did not finish, so no procedural verdict was produced.",
        },
        "roles": [],
        "steps": [],
        "unexpected_actions": [],
        "warnings": ["The execution analysis is incomplete."],
        "provider_invoked": False,
    }
