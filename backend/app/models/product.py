"""Records the LearnLoop product adds to the reused schema."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.models.util import new_id, utcnow


class ProcedureRecording(Base):
    """What a recording is for inside a procedure."""

    __tablename__ = "procedure_recordings"

    video_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(32), default="execution")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class HumanObservation(Base):
    """A person's reading of a frame. Append-only: a later row supersedes, nothing is edited."""

    __tablename__ = "human_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    verification_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str] = mapped_column(String(200))
    predicate_id: Mapped[str] = mapped_column(String(200), default="")
    answer: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[float] = mapped_column(Float)
    video_id: Mapped[str] = mapped_column(String(36), default="")
    revision_id: Mapped[str] = mapped_column(String(36), default="")
    reviewer: Mapped[str] = mapped_column(String(200), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
