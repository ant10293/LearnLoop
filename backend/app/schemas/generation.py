"""What a model may return, kept apart from what the application stores.

Every field the model writes is declared here and is required. A value it may
not know is nullable, not absent. Fields the application fills in later, such as
tracking support, provenance, and acceptance, do not exist here, so a model
cannot author them. Bounds such as a box inside 0..1 are checked locally when the
response becomes an AnalysisDocument.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.enums import (
    ActionOutcome,
    AmbiguityKind,
    Continuity,
    DependencyCategory,
    EvidenceClass,
    RelationKind,
    Visibility,
)


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedEvidence(_Closed):
    classification: EvidenceClass
    note: str
    frame_timestamps: list[float]
    time_start: float | None
    time_end: float | None


class GeneratedBox(_Closed):
    x: float
    y: float
    w: float
    h: float


class GeneratedCamera(_Closed):
    viewpoint: str
    task_constraint: str
    research_mode: str
    notes: str


class GeneratedObject(_Closed):
    local_id: str
    label: str
    description: str
    visual_characteristics: str
    initial_state: str
    final_state: str
    uncertain_identity: bool
    manipulated: bool
    confidence: float | None
    instance_id: str | None
    identity_relation: str | None


class GeneratedObservation(_Closed):
    object_local_id: str
    timestamp: float
    bbox: GeneratedBox | None
    visibility: Visibility
    description: str
    evidence: GeneratedEvidence
    occlusion: float | None
    confidence: float | None
    continuity: Continuity
    track_ref: str | None


class GeneratedState(_Closed):
    object_local_id: str
    time_start: float
    time_end: float
    state_label: str
    description: str
    evidence: GeneratedEvidence


class GeneratedTransition(_Closed):
    object_local_id: str
    timestamp: float
    from_state: str
    to_state: str
    description: str
    evidence: GeneratedEvidence
    certainty: float | None


class GeneratedRelationship(_Closed):
    subject_local_id: str
    other_local_id: str | None
    kind: RelationKind
    time_start: float
    time_end: float
    description: str
    evidence: GeneratedEvidence


class GeneratedAction(_Closed):
    local_id: str
    action_type: str
    description: str
    start: float
    end: float
    actor: str | None
    object_local_ids: list[str]
    initial_state: str
    final_state: str
    outcome: ActionOutcome
    uncertainty: float
    evidence: GeneratedEvidence


class GeneratedStep(_Closed):
    local_id: str
    name: str
    required_objects: list[str]
    preconditions: list[str]
    expected_action: str
    completion_condition: str
    observed_result: str
    action_local_ids: list[str]
    time_start: float | None
    time_end: float | None
    uncertainty: float
    evidence: GeneratedEvidence
    parallel_group: str | None


class GeneratedDependency(_Closed):
    from_step: str
    to_step: str
    category: DependencyCategory
    rationale: str
    evidence: GeneratedEvidence


class GeneratedAmbiguity(_Closed):
    local_id: str
    kind: AmbiguityKind
    description: str
    time_start: float | None
    time_end: float | None
    object_local_ids: list[str]
    action_local_ids: list[str]
    alternative: str
    evidence: GeneratedEvidence


class DiscoveryOutput(_Closed):
    """Objects and boxes. Grounded discovery and the multi-pass scene call."""

    summary: str
    camera: GeneratedCamera | None
    objects: list[GeneratedObject]
    observations: list[GeneratedObservation]
    ambiguities: list[GeneratedAmbiguity]


class InterpretationOutput(_Closed):
    """Semantics over resolved geometry. It has no objects or boxes to overwrite."""

    summary: str
    actions: list[GeneratedAction]
    states: list[GeneratedState]
    transitions: list[GeneratedTransition]
    relationships: list[GeneratedRelationship]
    workflow_steps: list[GeneratedStep]
    dependencies: list[GeneratedDependency]
    ambiguities: list[GeneratedAmbiguity]


class SinglePassOutput(_Closed):
    """One call that produces the whole analysis."""

    summary: str
    camera: GeneratedCamera | None
    objects: list[GeneratedObject]
    observations: list[GeneratedObservation]
    states: list[GeneratedState]
    transitions: list[GeneratedTransition]
    relationships: list[GeneratedRelationship]
    actions: list[GeneratedAction]
    workflow_steps: list[GeneratedStep]
    dependencies: list[GeneratedDependency]
    ambiguities: list[GeneratedAmbiguity]


class ActionsOutput(_Closed):
    actions: list[GeneratedAction]


class StatesOutput(_Closed):
    states: list[GeneratedState]
    transitions: list[GeneratedTransition]
    relationships: list[GeneratedRelationship]


class WorkflowOutput(_Closed):
    workflow_steps: list[GeneratedStep]
    dependencies: list[GeneratedDependency]


class AmbiguitiesOutput(_Closed):
    ambiguities: list[GeneratedAmbiguity]


class ReinspectionOutput(_Closed):
    summary: str
    ambiguities: list[GeneratedAmbiguity]
