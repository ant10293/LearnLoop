"""Reliability Lab: repeated, independent runs of one pipeline on one input.

A batch freezes its inputs and settings when it is made. Each repetition points
at the ordinary analysis run (and, for verification, the ordinary report) that
the existing pipeline made for it, and keeps its own normalized result. Ground
truth lives on the batch, apart from every prediction, so changing it never
touches a result.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class ReliabilityBatch(Base):
    __tablename__ = "reliability_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(300), default="")
    # verification or learning
    mode: Mapped[str] = mapped_column(String(32))
    # queued, running, completed, canceled
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    template_id: Mapped[str] = mapped_column(String(36), default="")
    video_id: Mapped[str] = mapped_column(String(36))
    repetitions: Mapped[int] = mapped_column(Integer)
    max_concurrency: Mapped[int] = mapped_column(Integer)
    independent_inference: Mapped[bool] = mapped_column(Boolean, default=True)
    # Everything the runs were made with, copied at creation. Never read from live defaults afterwards.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # {"expected_verdict": "PASS" | "FAIL" | None, "step_labels": {step_id: verdict}}. Kept apart from predictions.
    ground_truth_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)


class ReliabilityRun(Base):
    __tablename__ = "reliability_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    batch_id: Mapped[str] = mapped_column(ForeignKey("reliability_batches.id", ondelete="CASCADE"), index=True)
    # 1-based repetition number within the batch.
    index: Mapped[int] = mapped_column(Integer)
    # queued, running, completed, failed, canceled
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    # The ordinary analysis run the existing pipeline made for this repetition. Set when it is launched.
    analysis_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True, unique=True)
    # The ordinary verification report, in verification mode.
    verification_id: Mapped[str] = mapped_column(String(36), default="")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    # Normalized result for comparison. Raw outputs stay on the linked analysis and report.
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # learning mode: the procedure learned from this repetition, never stored as a procedure version.
    learned_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    # pipeline_error, rate_limited, resource_exhausted, database_busy, missing_recording, invalid_result, canceled, interrupted
    error_kind: Mapped[str] = mapped_column(String(40), default="")
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
