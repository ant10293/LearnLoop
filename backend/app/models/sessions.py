"""Task sessions, cross-clip identity, and versioned project interpretations."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class RecordingSession(Base):
    __tablename__ = "recording_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(200))
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class CanonicalObject(Base):
    __tablename__ = "canonical_objects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    label: Mapped[str] = mapped_column(String(300), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    color: Mapped[str] = mapped_column(String(16), default="#7EB6FF")
    uncertain: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class IdentityLink(Base):
    __tablename__ = "identity_links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    canonical_id: Mapped[str] = mapped_column(String(36), index=True)
    video_id: Mapped[str] = mapped_column(String(36), index=True)
    run_id: Mapped[str] = mapped_column(String(36), default="")
    object_local_id: Mapped[str] = mapped_column(String(80))
    label: Mapped[str] = mapped_column(String(300), default="")
    state_before: Mapped[str] = mapped_column(Text, default="")
    state_after: Mapped[str] = mapped_column(Text, default="")
    image_url: Mapped[str] = mapped_column(String(300), default="")
    method: Mapped[str] = mapped_column(String(40), default="context")
    score: Mapped[float] = mapped_column(Float, default=0)
    status: Mapped[str] = mapped_column(String(32), default="proposed")
    uncertain: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40), default="INFERRED")
    provenance: Mapped[str] = mapped_column(String(40), default="DETERMINISTIC")


class InterSessionTransition(Base):
    __tablename__ = "inter_session_transitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    canonical_id: Mapped[str] = mapped_column(String(36), index=True)
    from_video_id: Mapped[str] = mapped_column(String(36))
    to_video_id: Mapped[str] = mapped_column(String(36))
    observed_change: Mapped[str] = mapped_column(Text, default="")
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    from_state: Mapped[str] = mapped_column(Text, default="")
    to_state: Mapped[str] = mapped_column(Text, default="")
    evidence_class: Mapped[str] = mapped_column(String(40), default="INFERRED")
    provenance: Mapped[str] = mapped_column(String(40), default="DETERMINISTIC")
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    gap_note: Mapped[str] = mapped_column(Text, default="")


class ProjectRevision(Base):
    __tablename__ = "project_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    trigger: Mapped[str] = mapped_column(String(200), default="")
    change_summary: Mapped[str] = mapped_column(Text, default="")
    document_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ProjectGroundTruth(Base):
    __tablename__ = "project_ground_truth"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(300), default="Researcher order and identity")
    document_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class OrderHypothesis(Base):
    __tablename__ = "order_hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    method: Mapped[str] = mapped_column(String(32))
    ranked_json: Mapped[list] = mapped_column(JSON, default=list)
    evidence_json: Mapped[list] = mapped_column(JSON, default=list)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
