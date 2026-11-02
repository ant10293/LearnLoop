"""Project, video, and frame tables."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.models.util import new_id, utcnow

if TYPE_CHECKING:
    from app.models.runs import AnalysisRun


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    is_task: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    videos: Mapped[list["VideoAsset"]] = relationship(back_populates="project")
    runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="project")


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    original_filename: Mapped[str] = mapped_column(String(500))
    stored_path: Mapped[str] = mapped_column(String(1000))
    proxy_path: Mapped[str] = mapped_column(String(1000), default="")
    thumbnail_path: Mapped[str] = mapped_column(String(1000), default="")
    contact_sheet_path: Mapped[str] = mapped_column(String(1000), default="")
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="")
    container: Mapped[str] = mapped_column(String(40), default="")
    codec: Mapped[str] = mapped_column(String(80), default="")
    duration_sec: Mapped[float] = mapped_column(Float, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0)
    avg_fps: Mapped[float] = mapped_column(Float, default=0)
    time_base: Mapped[str] = mapped_column(String(40), default="")
    rotation_deg: Mapped[int] = mapped_column(Integer, default=0)
    variable_frame_rate: Mapped[bool] = mapped_column(Boolean, default=False)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    display_width: Mapped[int] = mapped_column(Integer, default=0)
    display_height: Mapped[int] = mapped_column(Integer, default=0)
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    research_mode: Mapped[str] = mapped_column(String(64), default="")
    capture_time_raw: Mapped[str] = mapped_column(String(120), default="")
    capture_time_utc: Mapped[str] = mapped_column(String(40), default="")
    capture_timezone: Mapped[str] = mapped_column(String(40), default="")
    timestamp_source: Mapped[str] = mapped_column(String(40), default="absent")
    timestamp_reliability: Mapped[str] = mapped_column(String(32), default="absent")
    timestamp_note: Mapped[str] = mapped_column(Text, default="")
    file_mtime: Mapped[str] = mapped_column(String(40), default="")
    order_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    order_method: Mapped[str] = mapped_column(String(32), default="")
    order_ambiguous: Mapped[bool] = mapped_column(Boolean, default=False)
    audio_probed: Mapped[bool] = mapped_column(Boolean, default=False)
    has_audio: Mapped[bool] = mapped_column(Boolean, default=False)
    audio_codec: Mapped[str] = mapped_column(String(80), default="")
    audio_channels: Mapped[int] = mapped_column(Integer, default=0)
    ingest_status: Mapped[str] = mapped_column(String(32), default="ready")
    ingest_progress: Mapped[float] = mapped_column(Float, default=100)
    ingest_detail: Mapped[str] = mapped_column(String(200), default="")
    ingest_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    project: Mapped[Project | None] = relationship(back_populates="videos")
    frames: Mapped[list["FrameAsset"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class FrameAsset(Base):
    __tablename__ = "frame_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("video_assets.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    timestamp: Mapped[float] = mapped_column(Float)
    path: Mapped[str] = mapped_column(String(1000))
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str] = mapped_column(String(80), default="uniform")
    strategy: Mapped[str] = mapped_column(String(40), default="uniform")
    motion_score: Mapped[float] = mapped_column(Float, default=0)
    sharpness: Mapped[float] = mapped_column(Float, default=0)
    scene_change: Mapped[bool] = mapped_column(Boolean, default=False)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    video: Mapped[VideoAsset] = relationship(back_populates="frames")
