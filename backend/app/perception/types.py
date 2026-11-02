"""Shared perception records. Embeddings are not stored on these objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.schemas.analysis import PerceptionRecord


@dataclass
class Detection:
    timestamp: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    label: str | None = None
    score: float | None = None
    detector_id: str = ""
    coordinate_frame: str = "image"
    quality: float = 0.0
    human: bool = False


@dataclass
class DetectionContext:
    timestamp: float
    index: int
    scene_chunk: int
    research_mode: str
    previous_bgr: np.ndarray | None = None
    camera_compensated: bool = False


@dataclass
class FrameView:
    timestamp: float
    image: np.ndarray
    scene_change: bool = False
    global_motion: float = 0.0
    motion_score: float = 0.0
    index: int = 0


@dataclass
class TrackPoint:
    track_ref: str
    timestamp: float
    x: float
    y: float
    w: float
    h: float
    motion_score: float
    continuity: str
    camera_compensated: bool
    scene_chunk: int
    note: str = ""
    status: str = "tracked"
    association_score: float | None = None
    detector_id: str = ""
    visibility: str = "visible"


@dataclass
class SpatialReport:
    viewpoint: str
    task_constraint: str
    research_mode: str
    global_motion: float
    scene_change_count: int
    camera_motion_compensated: bool
    compensation_model: str
    coordinate_frame: str
    world_coordinates_supported: bool
    notes: str
    tracks: list[TrackPoint] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    perception: PerceptionRecord | None = None


class EncoderUnavailable(RuntimeError):
    """The requested encoder did not run. Callers must not substitute another model."""


class DetectorUnavailable(RuntimeError):
    """The requested detector did not run. Callers must not substitute another algorithm."""


class TrackingCancelled(Exception):
    """The job was cancelled while frames were still being tracked."""
