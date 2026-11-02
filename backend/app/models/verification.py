"""Learned procedure templates and execution verification records.

A template revision is immutable once stored. A later edit inserts a new version.
Verification rows point at the template id they used and do not follow later versions.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class LearnedWorkflowTemplate(Base):
    __tablename__ = "learned_workflow_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    family_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_run_id: Mapped[str] = mapped_column(String(36), default="")
    source_revision_id: Mapped[str] = mapped_column(String(36), default="")
    source_video_id: Mapped[str] = mapped_column(String(36), default="")
    name: Mapped[str] = mapped_column(String(300), default="Learned procedure")
    acceptance_status: Mapped[str] = mapped_column(String(32), default="draft")
    document_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class VerificationRunRow(Base):
    __tablename__ = "verification_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    template_id: Mapped[str] = mapped_column(String(36), index=True)
    template_version: Mapped[int] = mapped_column(Integer, default=1)
    execution_video_id: Mapped[str] = mapped_column(String(36), default="")
    execution_run_id: Mapped[str] = mapped_column(String(36), index=True, default="")
    execution_revision_id: Mapped[str] = mapped_column(String(36), default="")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    overall_result: Mapped[str] = mapped_column(String(32), default="NOT_EVALUATED")
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    machine_report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)


class StepVerification(Base):
    __tablename__ = "step_verifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    verification_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str] = mapped_column(String(80))
    execution_state: Mapped[str] = mapped_column(String(32), default="unknown")
    verdict: Mapped[str] = mapped_column(String(32), default="CANNOT_VERIFY")
    error_category: Mapped[str] = mapped_column(String(64), default="")
    matched_action_ids: Mapped[list] = mapped_column(JSON, default=list)
    matched_object_ids: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")
    uncertainty: Mapped[str] = mapped_column(Text, default="")
    evidence_json: Mapped[list] = mapped_column(JSON, default=list)
    history_json: Mapped[list] = mapped_column(JSON, default=list)
    review_required: Mapped[bool] = mapped_column(Boolean, default=False)


class VerificationEvidence(Base):
    __tablename__ = "verification_evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    verification_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str] = mapped_column(String(80), default="")
    video_id: Mapped[str] = mapped_column(String(36), default="")
    timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    object_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence_class: Mapped[str] = mapped_column(String(40), default="UNKNOWN")
    provenance: Mapped[str] = mapped_column(String(40), default="DETERMINISTIC")
    explanation: Mapped[str] = mapped_column(Text, default="")
    side: Mapped[str] = mapped_column(String(32), default="execution")
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class VerificationRoleMapping(Base):
    __tablename__ = "verification_role_mappings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    verification_id: Mapped[str] = mapped_column(String(36), index=True)
    role_id: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(32), default="unresolved")
    object_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    score: Mapped[float] = mapped_column(Float, default=0)
    hypotheses_json: Mapped[list] = mapped_column(JSON, default=list)
    note: Mapped[str] = mapped_column(Text, default="")


class VerificationReinspection(Base):
    __tablename__ = "verification_reinspections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    verification_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str] = mapped_column(String(80), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    frame_timestamps: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="not_sent")
    output_json: Mapped[dict] = mapped_column(JSON, default=dict)
    previous_verdict: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class VerificationGroundTruth(Base):
    __tablename__ = "verification_ground_truth"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    verification_id: Mapped[str] = mapped_column(String(36), default="")
    name: Mapped[str] = mapped_column(String(300), default="Verification labels")
    document_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
