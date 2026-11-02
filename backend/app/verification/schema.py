"""Verification template and report models.

These describe a procedure that can be checked in another recording.
They are not a second analysis document.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RequirementStatus = Literal["observed", "required", "optional", "uncertain"]
Verdict = Literal["PASS", "FAIL", "CANNOT_VERIFY", "NOT_APPLICABLE"]
Overall = Literal["PASS", "FAIL", "INCONCLUSIVE", "INCOMPLETE", "NOT_EVALUATED"]
ExecutionState = Literal["not_attempted", "attempted", "completed", "corrected", "unknown"]
RoleStatus = Literal["assigned", "ambiguous", "unresolved", "human"]
ErrorCategory = Literal[
    "MISSING_COMPONENT",
    "MISSING_STEP",
    "INCORRECT_COMPONENT",
    "INCORRECT_ORDER",
    "INCORRECT_PLACEMENT",
    "INCOMPLETE_OPERATION",
    "UNEXPECTED_ACTION",
    "INCORRECT_PHYSICAL_STATE",
    "TEMPORARY_ERROR_CORRECTED",
    "UNVERIFIABLE_OPERATION",
    "",
]


class CompletionPredicate(BaseModel):
    id: str
    type: Literal["physical_relationship", "state_transition", "final_state"]
    subject_role: str
    target_role: str = ""
    relation: str = ""
    from_state: str = ""
    to_state: str = ""
    state_label: str = ""
    evidence_requirement: Literal["required", "optional"] = "required"
    source_evidence_class: str = "UNKNOWN"
    context_relations: list[dict] = Field(default_factory=list)


class TemplateDependency(BaseModel):
    """Predecessor of a step. Only necessary_physical edges constrain order."""

    step_id: str
    category: str
    constrains_order: bool = False
    rationale: str = ""
    evidence_class: str = "UNKNOWN"


class TemplateStep(BaseModel):
    step_id: str
    name: str
    required: bool = False
    requirement_status: RequirementStatus = "observed"
    observed: bool = True
    object_roles: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    expected_action: str = ""
    expected_action_type: str = ""
    completion_conditions: list[CompletionPredicate] = Field(default_factory=list)
    dependencies: list[TemplateDependency] = Field(default_factory=list)
    verification_requirements: dict = Field(default_factory=dict)
    uncertainty: float = 0.5
    uncertainty_notes: str = ""
    incidental: bool = False
    demo_time_start: float | None = None
    demo_time_end: float | None = None
    parallel_group: str = ""


class ObjectRole(BaseModel):
    role_id: str
    label: str = ""
    description: str = ""
    visual_characteristics: str = ""
    source_object_ids: list[str] = Field(default_factory=list)
    required: bool = True
    uncertain: bool = True
    cardinality: Literal["one", "group"] = "one"
    state_labels: list[str] = Field(default_factory=list)
    relation_signature: list[str] = Field(default_factory=list)
    reference_area: float | None = None


class LearnedTemplate(BaseModel):
    schema_version: str = "1.0.0"
    name: str = "Learned procedure"
    summary: str = ""
    source_run_id: str = ""
    source_revision_id: str = ""
    source_video_id: str = ""
    acceptance_status: Literal["draft", "accepted"] = "draft"
    roles: list[ObjectRole] = Field(default_factory=list)
    steps: list[TemplateStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RoleHypothesis(BaseModel):
    object_local_id: str
    score: float
    features: dict = Field(default_factory=dict)


class RoleAssignment(BaseModel):
    role_id: str
    label: str = ""
    status: RoleStatus = "unresolved"
    object_local_ids: list[str] = Field(default_factory=list)
    score: float = 0
    hypotheses: list[RoleHypothesis] = Field(default_factory=list)
    cardinality: Literal["one", "group"] = "one"
    note: str = ""


class EvidenceRef(BaseModel):
    video_id: str = ""
    timestamp: float | None = None
    observation: str = ""
    evidence_class: str = "UNKNOWN"
    provenance: str = "DETERMINISTIC"
    object_local_ids: list[str] = Field(default_factory=list)
    side: Literal["demonstration", "execution"] = "execution"
    bbox: dict | None = None
    frame_id: str | None = None


class PredicateJudgement(BaseModel):
    predicate_id: str
    status: Literal["pass", "fail", "cannot"]
    error_category: str = ""
    note: str = ""
    corrected: bool = False
    evidence: list[EvidenceRef] = Field(default_factory=list)
    support_onsets: list[float] = Field(default_factory=list)


class StepResult(BaseModel):
    step_id: str
    name: str
    required: bool
    requirement_status: str
    verdict: Verdict
    execution_state: ExecutionState = "unknown"
    error_category: str = ""
    expected: str = ""
    observed: str = ""
    explanation: str = ""
    uncertainty_notes: str = ""
    review_required: bool = False
    matched_action_ids: list[str] = Field(default_factory=list)
    matched_object_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    demo_evidence: list[EvidenceRef] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    corrected: bool = False
    alignment: str = "unmatched"
    demo_time_start: float | None = None
    demo_time_end: float | None = None
    execution_time: float | None = None
    predicates: list[PredicateJudgement] = Field(default_factory=list)
    machine_verdict: str = ""
