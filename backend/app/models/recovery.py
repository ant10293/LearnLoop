"""Targeted object recovery. These rows stay apart from a run's stages and revisions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class RecoveryExperiment(Base):
    __tablename__ = "recovery_experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    source_revision_id: Mapped[str] = mapped_column(String(36))
    # draft, planned, running, completed, partial, failed, canceled, or interrupted.
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    # The last mode that produced results: dry_run, replay, or live.
    mode: Mapped[str] = mapped_column(String(16), default="")
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    targets_json: Mapped[list] = mapped_column(JSON, default=list)
    limits_json: Mapped[dict] = mapped_column(JSON, default=dict)
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    applied_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    recorded_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class RecoveryRequest(Base):
    __tablename__ = "recovery_requests"
    __table_args__ = (UniqueConstraint("experiment_id", "cache_key", name="uq_recovery_request_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("recovery_experiments.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    # planned, running, completed, failed, or canceled.
    status: Mapped[str] = mapped_column(String(32), default="planned")
    # not_submitted, uploaded, submitted, or completed. "submitted" without completion may have been billed.
    submission: Mapped[str] = mapped_column(String(32), default="not_submitted")
    source: Mapped[str] = mapped_column(String(16), default="")
    media_json: Mapped[dict] = mapped_column(JSON, default=dict)
    model_request_id: Mapped[str] = mapped_column(String(36), default="")
    provider_request_id: Mapped[str] = mapped_column(String(200), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    billed_frames: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class RecoveryProposal(Base):
    __tablename__ = "recovery_proposals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("recovery_experiments.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    target_id: Mapped[str] = mapped_column(String(32))
    object_local_id: Mapped[str] = mapped_column(String(80))
    time_start: Mapped[float] = mapped_column(Float)
    time_end: Mapped[float] = mapped_column(Float)
    # pending, accepted, or rejected. A rejected proposal is kept for debugging.
    decision: Mapped[str] = mapped_column(String(16), default="pending")
    # Boxes and checks only. Mask payloads stay in the stored response.
    rows_json: Mapped[list] = mapped_column(JSON, default=list)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    applied_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class RecoveryDismissal(Base):
    """A person said a suspected problem is not one. It hides the flag and changes no evidence."""

    __tablename__ = "recovery_dismissals"
    __table_args__ = (UniqueConstraint("run_id", "target_id", name="uq_recovery_dismissal_target"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[str] = mapped_column(String(32))
    object_local_id: Mapped[str] = mapped_column(String(80))
    time_start: Mapped[float] = mapped_column(Float)
    time_end: Mapped[float] = mapped_column(Float)
    # The detector's reasons at the time, so repeated false alarms can be counted later.
    reasons_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
