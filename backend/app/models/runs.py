"""Analysis runs, revisions, prompts, and provider bookkeeping."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow

if TYPE_CHECKING:
    from app.models.media import Project


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(80), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)
    foundational: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (UniqueConstraint("key", "version", name="uq_prompt_key_version"),)


class ManualModel(Base):
    __tablename__ = "manual_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    provider_id: Mapped[str] = mapped_column(String(40), index=True)
    model_id: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    __table_args__ = (Index("uq_manual_model_provider_model", "provider_id", "model_id", unique=True),)


class ProviderSecret(Base):
    """Encrypted credential. The plaintext is never selected into API responses."""

    __tablename__ = "provider_secrets"

    provider_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    ciphertext: Mapped[str] = mapped_column(Text)
    hint: Mapped[str] = mapped_column(String(20), default="")
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class ProviderStatus(Base):
    __tablename__ = "provider_status"

    provider_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    last_validated_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    last_error: Mapped[str] = mapped_column(Text, default="")
    extra_base_url: Mapped[str] = mapped_column(String(500), default="")
    spend_limit_usd: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    video_id: Mapped[str] = mapped_column(ForeignKey("video_assets.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    research_mode: Mapped[str] = mapped_column(String(64))
    viewpoint: Mapped[str] = mapped_column(String(32))
    task_constraint: Mapped[str] = mapped_column(String(32))
    provider_id: Mapped[str] = mapped_column(String(40))
    model_id: Mapped[str] = mapped_column(String(200))
    segmentation_mode: Mapped[str] = mapped_column(String(40), default="disabled")
    segmentation_model_id: Mapped[str] = mapped_column(String(200), default="")
    prompt_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompt_versions.id"), nullable=True
    )
    modality: Mapped[str] = mapped_column(String(40))
    sampling_preset: Mapped[str] = mapped_column(String(40))
    analysis_pass: Mapped[str] = mapped_column(String(40), default="single")
    config_json: Mapped[dict] = mapped_column(JSON)
    coverage_json: Mapped[dict] = mapped_column(JSON, default=dict)
    fixture: Mapped[bool] = mapped_column(Boolean, default=False)
    fixture_label: Mapped[str] = mapped_column(String(200), default="")
    starred: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    active_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    app_version: Mapped[str] = mapped_column(String(40), default="")
    git_revision: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    project: Mapped["Project | None"] = relationship(back_populates="runs")
    stages: Mapped[list["PipelineStageRun"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    revisions: Mapped[list["AnalysisRevision"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class AnalysisRevision(Base):
    __tablename__ = "analysis_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    layer: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    document_json: Mapped[dict] = mapped_column(JSON)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    run: Mapped[AnalysisRun] = relationship(back_populates="revisions")


class PipelineStageRun(Base):
    __tablename__ = "pipeline_stage_runs"
    __table_args__ = (UniqueConstraint("run_id", "stage", name="uq_run_stage"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    checkpoint_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="stages")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    from_stage: Mapped[str] = mapped_column(String(64), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=2)
    worker_id: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(nullable=True)


class ModelRequestRow(Base):
    __tablename__ = "model_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    provider_id: Mapped[str] = mapped_column(String(40))
    model_id: Mapped[str] = mapped_column(String(200))
    modality: Mapped[str] = mapped_column(String(40))
    prompt_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    parameters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    frame_timestamps: Mapped[list] = mapped_column(JSON, default=list)
    provider_request_id: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ModelResponseRow(Base):
    __tablename__ = "model_responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(
        ForeignKey("model_requests.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)
    text: Mapped[str] = mapped_column(Text, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    price_known: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
