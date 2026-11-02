"""Normalized analysis entities. Rows belong to an immutable revision."""

from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class DetectedObject(Base):
    __tablename__ = "detected_objects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_revisions.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    label: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    visual_characteristics: Mapped[str] = mapped_column(Text, default="")
    initial_state: Mapped[str] = mapped_column(Text, default="")
    final_state: Mapped[str] = mapped_column(Text, default="")
    uncertain_identity: Mapped[bool] = mapped_column(Boolean, default=False)
    manipulated: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    color: Mapped[str] = mapped_column(String(16), default="#7EB6FF")
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)
    annotation_status: Mapped[str] = mapped_column(String(32), default="unreviewed")
    primary_image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=0)


class ObjectObservation(Base):
    __tablename__ = "object_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    object_id: Mapped[str] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[float] = mapped_column(Float)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    visibility: Mapped[str] = mapped_column(String(32), default="visible")
    description: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    evidence_note: Mapped[str] = mapped_column(Text, default="")
    frame_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    sharpness: Mapped[float | None] = mapped_column(Float, nullable=True)
    occlusion: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    continuity: Mapped[str] = mapped_column(String(32), default="continuous")
    coordinate_frame: Mapped[str] = mapped_column(String(40), default="image")
    camera_compensated: Mapped[bool] = mapped_column(Boolean, default=False)
    track_ref: Mapped[str] = mapped_column(String(80), default="")
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)
    association_status: Mapped[str] = mapped_column(String(32), default="")
    source_model: Mapped[str] = mapped_column(String(80), default="")
    request_id: Mapped[str] = mapped_column(String(120), default="")
    mask_ref: Mapped[str] = mapped_column(String(200), default="")


class ObjectStateRow(Base):
    __tablename__ = "object_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    object_id: Mapped[str] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    time_start: Mapped[float] = mapped_column(Float)
    time_end: Mapped[float] = mapped_column(Float)
    state_label: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class ObjectStateTransition(Base):
    __tablename__ = "object_state_transitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    object_id: Mapped[str] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[float] = mapped_column(Float)
    from_state: Mapped[str] = mapped_column(String(300))
    to_state: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    certainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class ObjectRelationship(Base):
    __tablename__ = "object_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("detected_objects.id", ondelete="CASCADE"))
    other_id: Mapped[str | None] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(40))
    time_start: Mapped[float] = mapped_column(Float)
    time_end: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class ObjectImage(Base):
    __tablename__ = "object_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    object_id: Mapped[str] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    path: Mapped[str] = mapped_column(String(1000))
    frame_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    timestamp: Mapped[float] = mapped_column(Float, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0)


class SegmentationMask(Base):
    __tablename__ = "segmentation_masks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    object_id: Mapped[str | None] = mapped_column(
        ForeignKey("detected_objects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    frame_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    timestamp: Mapped[float] = mapped_column(Float, default=0)
    track_ref: Mapped[str] = mapped_column(String(80), default="")
    encoding: Mapped[str] = mapped_column(String(32), default="png")
    path: Mapped[str] = mapped_column(String(1000), default="")
    rle_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="cv")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    identity_uncertain: Mapped[bool] = mapped_column(Boolean, default=True)


class ActionRow(Base):
    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    action_type: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    actor: Mapped[str] = mapped_column(String(200), default="")
    object_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    initial_state: Mapped[str] = mapped_column(Text, default="")
    final_state: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(String(32), default="unknown")
    uncertainty: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    evidence_note: Mapped[str] = mapped_column(Text, default="")
    evidence_timestamps: Mapped[list] = mapped_column(JSON, default=list)
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class WorkflowRow(Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    revision_id: Mapped[str] = mapped_column(String(36), index=True, unique=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(300), default="Reconstructed procedure")
    summary: Mapped[str] = mapped_column(Text, default="")


class WorkflowStepRow(Base):
    __tablename__ = "workflow_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    number: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(300))
    required_objects: Mapped[list] = mapped_column(JSON, default=list)
    preconditions: Mapped[list] = mapped_column(JSON, default=list)
    expected_action: Mapped[str] = mapped_column(Text, default="")
    completion_condition: Mapped[str] = mapped_column(Text, default="")
    observed_result: Mapped[str] = mapped_column(Text, default="")
    action_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    time_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    parallel_group: Mapped[str] = mapped_column(String(80), default="")
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class WorkflowDependencyRow(Base):
    __tablename__ = "workflow_dependencies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    from_step: Mapped[str] = mapped_column(String(80))
    to_step: Mapped[str] = mapped_column(String(80))
    category: Mapped[str] = mapped_column(String(40))
    rationale: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class AmbiguityRow(Base):
    __tablename__ = "ambiguities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    revision_id: Mapped[str] = mapped_column(String(36), index=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    time_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    object_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    action_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    alternative: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(40))
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)


class AnnotationRow(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    revision_id: Mapped[str] = mapped_column(String(36))
    target_type: Mapped[str] = mapped_column(String(40))
    target_local_id: Mapped[str] = mapped_column(String(80), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    patch_json: Mapped[dict] = mapped_column(JSON, default=dict)
    author: Mapped[str] = mapped_column(String(120), default="researcher")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class GroundTruthDataset(Base):
    __tablename__ = "ground_truth_datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    revision_id: Mapped[str] = mapped_column(String(36))
    name: Mapped[str] = mapped_column(String(300))
    annotator: Mapped[str] = mapped_column(String(120), default="researcher")
    research_mode: Mapped[str] = mapped_column(String(64), default="")
    document_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    ground_truth_id: Mapped[str] = mapped_column(String(36))
    predicted_revision_id: Mapped[str] = mapped_column(String(36))
    tolerances_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    errors_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ComparisonRow(Base):
    __tablename__ = "comparisons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(300), default="Comparison")
    left_run_id: Mapped[str] = mapped_column(String(36), index=True)
    right_run_id: Mapped[str] = mapped_column(String(36), index=True)
    object_matches_json: Mapped[list] = mapped_column(JSON, default=list)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    video_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(80))
    path: Mapped[str] = mapped_column(String(1000))
    label: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class SpatialTrackPoint(Base):
    __tablename__ = "spatial_track_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    video_id: Mapped[str] = mapped_column(String(36), index=True)
    track_ref: Mapped[str] = mapped_column(String(80))
    timestamp: Mapped[float] = mapped_column(Float)
    bbox_json: Mapped[dict] = mapped_column(JSON)
    motion_score: Mapped[float] = mapped_column(Float, default=0)
    continuity: Mapped[str] = mapped_column(String(32), default="continuous")
    coordinate_frame: Mapped[str] = mapped_column(String(40), default="image")
    camera_compensated: Mapped[bool] = mapped_column(Boolean, default=False)
    scene_chunk: Mapped[int] = mapped_column(Integer, default=0)
    identity_note: Mapped[str] = mapped_column(Text, default="")
    track_status: Mapped[str] = mapped_column(String(32), default="")
    association_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_id: Mapped[str] = mapped_column(String(80), default="")


class SceneChunk(Base):
    __tablename__ = "scene_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    time_start: Mapped[float] = mapped_column(Float)
    time_end: Mapped[float] = mapped_column(Float)
    scene_change: Mapped[bool] = mapped_column(Boolean, default=False)
    global_motion: Mapped[float] = mapped_column(Float, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
