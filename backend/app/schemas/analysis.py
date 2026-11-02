"""Canonical, versioned analysis document.

Evidence classification and provenance are independent. A human can directly
observe something, and a model can infer something. Neither field implies the other.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from app import SCHEMA_VERSION
from app.schemas.enums import (
    ActionOutcome,
    AmbiguityKind,
    Continuity,
    DependencyCategory,
    EvidenceClass,
    Provenance,
    RelationKind,
    Visibility,
)


class Evidence(BaseModel):
    classification: EvidenceClass
    provenance: Provenance
    note: str = ""
    frame_timestamps: list[float] = Field(default_factory=list)
    time_start: float | None = None
    time_end: float | None = None


class BBox(BaseModel):
    """Normalized image coordinates in displayed video space. Origin is top-left."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @field_validator("x", "y", "w", "h")
    @classmethod
    def finite(cls, value: float) -> float:
        if value != value:
            raise ValueError("bbox values must be finite")
        return value


class CameraAssessment(BaseModel):
    viewpoint: str
    task_constraint: str
    research_mode: str
    global_motion: float = 0
    scene_change_count: int = 0
    coordinate_frame: Literal["image", "scene_relative_image", "world"] = "image"
    world_coordinates_supported: bool = False
    camera_motion_compensated: bool = False
    compensation_model: str = "none"
    characterization_source: Literal["user", "auto", "override"] = "user"
    notes: str = ""


class Coverage(BaseModel):
    modality: Literal["native_video", "sampled_frames", "mixed"]
    sampling_preset: str
    nominal_fps: float | None = None
    frames_extracted: int = 0
    frames_submitted: int = 0
    frame_timestamps: list[float] = Field(default_factory=list)
    selection_reasons: dict[str, int] = Field(default_factory=dict)
    duration_sec: float = 0
    native_video_submitted: bool = False
    temporal_coverage_note: str = ""
    # Everything below is empty on documents written before the source timeline.
    time_start: float | None = None
    time_end: float | None = None
    # "presentation" when frame times came from the file, "container" when estimated.
    time_basis: str = ""
    # Largest unexamined stretch between frames the semantic model received.
    max_gap_sec: float | None = None
    uncovered_gaps: list[list[float]] = Field(default_factory=list)
    motion_frames: int = 0
    candidate_peaks: list[float] = Field(default_factory=list)
    selected_peaks: list[float] = Field(default_factory=list)
    budget_exclusions: int = 0
    scene_boundaries: list[dict] = Field(default_factory=list)
    # "provisional" until tracking has run, then "final".
    semantic_selection: str = ""
    selection_swaps: list[dict] = Field(default_factory=list)
    # "combined" or "grounded". Calls counts language-model requests for semantics.
    semantic_mode: str = ""
    semantic_calls: int | None = None
    tracking: dict = Field(default_factory=dict)
    segmentation: dict = Field(default_factory=dict)
    media_transforms: list[dict] = Field(default_factory=list)


class ObjectPrediction(BaseModel):
    local_id: str
    label: str
    description: str = ""
    visual_characteristics: str = ""
    initial_state: str = ""
    final_state: str = ""
    uncertain_identity: bool = False
    manipulated: bool = False
    confidence: float | None = None
    accepted: bool = True
    instance_id: str | None = None
    identity_relation: str | None = None
    # How the object was bound to a physical instance. Empty on older documents.
    # "associated" needs repeated overlap on shared times; "referenced" means the
    # model named a tracked id it was shown; "ambiguous", "weak", and
    # "unmatched" leave instance_id unset unless the model named one.
    association_status: str = ""
    association_score: float | None = None
    association_rivals: list[str] = Field(default_factory=list)


class ObservationPrediction(BaseModel):
    object_local_id: str
    timestamp: float
    bbox: BBox | None = None
    visibility: Visibility = Visibility.VISIBLE
    description: str = ""
    evidence: Evidence
    sharpness: float | None = None
    occlusion: float | None = None
    # The model's own score that this box is this object. Not a measured probability.
    confidence: float | None = None
    continuity: Continuity = Continuity.CONTINUOUS
    coordinate_frame: Literal["image", "scene_relative_image", "world"] = "image"
    camera_compensated: bool = False
    track_ref: str | None = None
    accepted: bool = True
    # SAM association is scoped to one tracking response. Empty on older documents.
    association_status: str = ""
    source_model: str = ""
    request_id: str = ""
    mask_ref: str = ""
    # Evidence resolution. Every field below is empty on older documents, which
    # still display as they were stored.
    # "raw" is one producer's output. "canonical" is the resolved geometry for
    # an instance at one time.
    observation_role: str = ""
    # "detected", "optical_flow", "vlm_estimated", or "unobserved".
    geometry_origin: str = ""
    # The component that produced the box: a detector id, "sam-3.1", "vlm", or "human".
    geometry_authority: str = ""
    # Separate uncertainties. None of them is a probability.
    # detection: "proposal", "detected", "mask", or "none". Flow is not a detection.
    detection_support: str = ""
    # association to a physical instance: "unmatched", "weak", "ambiguous", "referenced", "associated".
    association_support: str = ""
    # "single", "repeated", "flow_only", or "broken".
    temporal_support: str = ""
    # "unlabeled", "provisional", or "labeled". A label is not an identity.
    semantic_support: str = ""
    # Pixel size of the displayed frame the box was measured in. The box itself
    # stays normalized to the displayed frame.
    source_width: int | None = None
    source_height: int | None = None
    # Stable reference to the raw row a canonical row was resolved from.
    evidence_ref: str = ""


class StatePrediction(BaseModel):
    object_local_id: str
    time_start: float
    time_end: float
    state_label: str
    description: str = ""
    evidence: Evidence
    accepted: bool = True


class TransitionPrediction(BaseModel):
    object_local_id: str
    timestamp: float
    from_state: str
    to_state: str
    description: str = ""
    evidence: Evidence
    certainty: float | None = None
    accepted: bool = True


class RelationshipPrediction(BaseModel):
    subject_local_id: str
    other_local_id: str | None = None
    kind: RelationKind
    time_start: float
    time_end: float
    description: str = ""
    evidence: Evidence
    accepted: bool = True


class ActionPrediction(BaseModel):
    local_id: str
    action_type: str
    description: str = ""
    start: float
    end: float
    actor: str | None = None
    object_local_ids: list[str] = Field(default_factory=list)
    initial_state: str = ""
    final_state: str = ""
    outcome: ActionOutcome = ActionOutcome.UNKNOWN
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    evidence: Evidence
    accepted: bool = True


class WorkflowStepPrediction(BaseModel):
    local_id: str
    name: str
    required_objects: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    expected_action: str = ""
    completion_condition: str = ""
    observed_result: str = ""
    action_local_ids: list[str] = Field(default_factory=list)
    time_start: float | None = None
    time_end: float | None = None
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    evidence: Evidence
    parallel_group: str | None = None
    accepted: bool = True


class DependencyPrediction(BaseModel):
    from_step: str
    to_step: str
    category: DependencyCategory
    rationale: str = ""
    evidence: Evidence
    accepted: bool = True


class AmbiguityPrediction(BaseModel):
    local_id: str
    kind: AmbiguityKind
    description: str
    time_start: float | None = None
    time_end: float | None = None
    object_local_ids: list[str] = Field(default_factory=list)
    action_local_ids: list[str] = Field(default_factory=list)
    alternative: str = ""
    evidence: Evidence
    accepted: bool = True


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    entity_type: str = ""
    entity_id: str = ""


class InstanceRecord(BaseModel):
    """One physical instance. A semantic label is optional and is not the identity."""

    instance_id: str
    semantic_label: str | None = None
    status: str
    first_seen: float
    last_seen: float
    observation_count: int = 0
    trusted_references: int = 0
    candidate_references: int = 0
    uncertain: bool = True
    human_initialized: bool = False
    detector_id: str = ""


class HypothesisRecord(BaseModel):
    """An identity proposal. Scores are uncalibrated and are not probabilities."""

    hypothesis_id: str
    timestamp: float
    subject_instance_id: str
    candidate_instance_id: str
    runner_up_instance_id: str | None = None
    appearance_cosine: float | None = None
    motion_agreement: float | None = None
    association_score: float
    margin: float | None = None
    decision: Literal["proposed", "rejected", "ambiguous", "confirmed"]
    note: str = ""


class AppearanceRefRecord(BaseModel):
    instance_id: str
    timestamp: float
    state: Literal["trusted", "candidate"]
    quality: float = 0
    encoder_id: str = ""
    encoder_version: str = ""
    preprocess_version: str = ""
    bbox: BBox | None = None
    confirmed: bool = False


class PerceptionRecord(BaseModel):
    """Computer-vision identity for one analysis. Vectors are not stored here."""

    strategy: str
    detector_id: str
    encoder_id: str | None = None
    encoder_version: str | None = None
    encoder_status: str = "not_requested"
    encoder_detail: str = ""
    segmenter_id: str | None = None
    tracking_fps: float | None = None
    frames_tracked: int = 0
    world_coordinates_supported: bool = False
    coordinate_frame: Literal["image"] = "image"
    used_for_identity: bool = True
    scores_are_probabilities: bool = False
    diagnostics: dict = Field(default_factory=dict)
    instances: list[InstanceRecord] = Field(default_factory=list)
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    appearance_references: list[AppearanceRefRecord] = Field(default_factory=list)
    manual_corrections: list[dict] = Field(default_factory=list)
    notes: str = ""


class ReinspectionProposal(BaseModel):
    """A later look at an interval. applied stays false until a person accepts it."""

    ambiguity_local_id: str
    summary: str = ""
    evidence_classification: str = "UNKNOWN"
    changed: bool = False
    applied: bool = False
    notes: str = ""
    frame_timestamps: list[float] = Field(default_factory=list)


class InterpretationResponse(BaseModel):
    """What the grounded interpretation call is asked to return. Geometry is already resolved."""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    actions: list[ActionPrediction]
    states: list[StatePrediction]
    transitions: list[TransitionPrediction]
    relationships: list[RelationshipPrediction]
    workflow_steps: list[WorkflowStepPrediction]
    dependencies: list[DependencyPrediction]
    ambiguities: list[AmbiguityPrediction]


class BoxDecision(BaseModel):
    """A box a pipeline step removed, moved, or kept unmeasured. The box is kept as it was before that step."""

    step: str
    reason_code: str
    outcome: Literal["dropped", "reassigned", "kept"] = "dropped"
    object_local_id: str
    timestamp: float | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    other_object_id: str | None = None
    detail: str = ""


class AnalysisDocument(BaseModel):
    schema_version: str = SCHEMA_VERSION
    summary: str = ""
    fixture: bool = False
    fixture_label: str = ""
    camera: CameraAssessment | None = None
    coverage: Coverage | None = None
    objects: list[ObjectPrediction] = Field(default_factory=list)
    # The resolved display set. Every consumer reads this list.
    observations: list[ObservationPrediction] = Field(default_factory=list)
    # Producer outputs that the resolver did not display: superseded model
    # estimates, unassociated or conflicting segmentation. Kept for inspection.
    raw_observations: list[ObservationPrediction] = Field(default_factory=list)
    # Written by the pipeline only, so it stays out of the schema sent to a model.
    box_decisions: SkipJsonSchema[list[BoxDecision]] = Field(default_factory=list)
    states: list[StatePrediction] = Field(default_factory=list)
    transitions: list[TransitionPrediction] = Field(default_factory=list)
    relationships: list[RelationshipPrediction] = Field(default_factory=list)
    actions: list[ActionPrediction] = Field(default_factory=list)
    workflow_steps: list[WorkflowStepPrediction] = Field(default_factory=list)
    dependencies: list[DependencyPrediction] = Field(default_factory=list)
    ambiguities: list[AmbiguityPrediction] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    reinspection_notes: list[str] = Field(default_factory=list)
    reinspection_proposals: list[ReinspectionProposal] = Field(default_factory=list)
    perception: PerceptionRecord | None = None
