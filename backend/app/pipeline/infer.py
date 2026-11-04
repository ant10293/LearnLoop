"""Model calls, windowing, and local document construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path


from app.models.media import VideoAsset
from app.models.runs import AnalysisRun
from app.providers.model_list import capabilities_for
from app.providers.types import (
    UnsupportedMode,
)
from app.schemas.analysis import (
    AnalysisDocument,
    Coverage,
)
from app.services.storage import data_path, resolve_stored

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


class StageSkipped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_working(run_id: str) -> AnalysisDocument:
    path = _working_path(run_id)
    if not path.exists():
        return AnalysisDocument()
    return AnalysisDocument.model_validate(json.loads(path.read_text()))


def save_working(run_id: str, document: AnalysisDocument) -> None:
    path = _working_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))


def _working_path(run_id: str) -> Path:
    return data_path("runs", run_id, "working.json")


SEMANTIC_STAGES = ("scene_analysis", "action_analysis", "state_tracking", "workflow", "ambiguity", "evidence_fusion")
SEMANTIC_LISTS = ("actions", "states", "transitions", "relationships", "workflow_steps", "dependencies")


def save_semantic_raw(run_id: str, document: AnalysisDocument) -> None:
    """The scene call's parsed output, before any repair. Replaced on retry."""
    path = data_path("runs", run_id, "semantic_raw.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=2))


def _video_file(video: VideoAsset, run: AnalysisRun | None = None):
    """The requested-interval subclip when preprocess cut one, else the playback mp4 or the original."""
    if run is not None:
        from app.pipeline.clip_time import load_transform, subclip_path

        if load_transform(run.id) is not None:
            return subclip_path(run.id)
    return resolve_stored(video.proxy_path or video.stored_path)


def _require_video_length(run: AnalysisRun, video: VideoAsset) -> None:
    limit = capabilities_for(run.provider_id, run.model_id).max_video_seconds
    duration = float(video.duration_sec or 0)
    from app.pipeline.clip_time import load_transform

    transform = load_transform(run.id) if getattr(run, "id", None) else None
    if transform is not None:
        duration = float(transform.get("duration") or duration)
    if limit and duration > float(limit):
        raise UnsupportedMode(
            f"{run.model_id} accepts at most {limit:g} seconds of video. "
            f"This recording is {duration:.1f} seconds. The file was not replaced with frames."
        )


def _note_video_coverage(coverage: Coverage, *, timeline_frames: int | None = None) -> None:
    coverage.modality = "native_video"
    coverage.native_video_submitted = True
    if timeline_frames is None:
        coverage.frames_submitted = 0
        extra = "The model receives the video file. Preview frames stay in the inspector."
    else:
        coverage.frames_submitted = timeline_frames
        extra = "The video file is sent with the timeline frames so the whole recording is visible."
    note = coverage.temporal_coverage_note or ""
    if "video file" not in note.lower():
        coverage.temporal_coverage_note = (note + " " + extra).strip()


# Adjacent calls repeat this many frames so a window boundary is a shared moment,
# not a fresh scene. Two frames is enough to show the arrangement on both sides.
WINDOW_OVERLAP = 2


