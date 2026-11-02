"""Local transcript rows. The model text is kept when a person edits the display text."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db import Base
from app.models.util import new_id, utcnow


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    video_id: Mapped[str] = mapped_column(String(36), index=True)
    local_id: Mapped[str] = mapped_column(String(80))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text_model: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(16), default="")
    language_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    words_json: Mapped[list] = mapped_column(JSON, default=list)
    model_name: Mapped[str] = mapped_column(String(40), default="")
    device: Mapped[str] = mapped_column(String(20), default="")
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    provenance: Mapped[str] = mapped_column(String(40), default="AI_PREDICTION")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AudioAlignment(Base):
    __tablename__ = "audio_alignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    video_id: Mapped[str] = mapped_column(String(36), index=True)
    segment_id: Mapped[str] = mapped_column(String(36), index=True)
    classification: Mapped[str] = mapped_column(String(40))
    local_classification: Mapped[str] = mapped_column(String(40), default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    action_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    object_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    step_local_ids: Mapped[list] = mapped_column(JSON, default=list)
    frame_timestamps: Mapped[list] = mapped_column(JSON, default=list)
    time_offset_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_class: Mapped[str] = mapped_column(String(40), default="INFERRED")
    provenance: Mapped[str] = mapped_column(String(40), default="DETERMINISTIC")
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    speech_is_not_completion: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
