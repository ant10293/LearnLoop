"""Spatial tracking entry point.

The motion baseline remains available. It is no longer the only identity path.
Neither path emits world coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.perception.legacy import run_legacy
from app.perception.presets import PerceptionConfig, config_from_mapping
from app.perception.tracker import track_frames
from app.perception.types import FrameView, SpatialReport, TrackPoint
from app.video.frames import read_bgr
from app.video.sampling import MotionSignal


def characterize(signals: list[MotionSignal], declared_mode: str) -> dict:
    """Camera profile from motion signals. Only confirmed hard cuts count as scene changes."""
    globals_ = [item.global_motion for item in signals[1:]] or [0.0]
    median_global = float(np.median(globals_))
    scene_changes = sum(1 for item in signals if item.scene_change)
    if declared_mode == "fixed_camera":
        mode, view, constraint = "fixed_camera", "fixed", "constrained"
        source = "user"
    elif declared_mode == "constrained_head_mounted":
        mode, view, constraint = "constrained_head_mounted", "head_mounted", "constrained"
        source = "user"
    elif declared_mode == "unconstrained_head_mounted":
        mode, view, constraint = "unconstrained_head_mounted", "head_mounted", "unconstrained"
        source = "user"
    else:
        source = "auto"
        if scene_changes >= 2:
            mode, view, constraint = "unconstrained_head_mounted", "head_mounted", "unconstrained"
        elif median_global < 0.35:
            mode, view, constraint = "fixed_camera", "fixed", "unknown"
        else:
            mode, view, constraint = "constrained_head_mounted", "head_mounted", "constrained"
    return {
        "research_mode": mode,
        "viewpoint": view,
        "task_constraint": constraint,
        "global_motion": median_global,
        "scene_change_count": scene_changes,
        "source": source,
    }


def analyze_sequence(
    paths: list[Path],
    timestamps: list[float],
    signals: list[MotionSignal],
    declared_mode: str,
    config: PerceptionConfig | dict | None = None,
    detector=None,
    encoder=None,
    cache_dir: Path | None = None,
    should_stop=None,
    *,
    views=None,
    profile_signals: list[MotionSignal] | None = None,
) -> SpatialReport:
    """Track one frame at a time. Images are not retained after each step.

    `views` replaces `paths` with a streamed frame source. `profile_signals`
    characterizes the camera when the tracked frames are not known in advance.
    """
    resolved = config if isinstance(config, PerceptionConfig) else config_from_mapping(config if isinstance(config, dict) else None)
    profile = characterize(profile_signals if profile_signals is not None else signals, declared_mode)
    if views is not None and resolved.strategy in {"legacy_motion", "vlm_only"} and detector is None:
        frames = list(views)
        points, compensated, chunks = run_legacy(frames, profile["research_mode"])
        notes = _legacy_notes(profile, compensated, resolved.strategy)
        perception = _legacy_perception(points, resolved, len(frames), notes)
        return _report(profile, compensated, "affine_partial" if compensated else "none", notes, points, chunks, perception)
    if resolved.strategy in {"legacy_motion", "vlm_only"} and detector is None:
        frames = list(_views(paths, timestamps, signals))
        points, compensated, chunks = run_legacy(frames, profile["research_mode"])
        notes = _legacy_notes(profile, compensated, resolved.strategy)
        perception = _legacy_perception(points, resolved, len(frames), notes)
        return _report(profile, compensated, "affine_partial" if compensated else "none", notes, points, chunks, perception)
    from app.perception.detect import build_detector

    if detector is None:
        detector = build_detector(
            resolved.detector,
            compensate_camera=profile["viewpoint"] == "head_mounted",
            seeds=resolved.manual_regions,
        )
    if encoder is None and resolved.encoder and resolved.strategy not in {"legacy_motion", "vlm_only"}:
        from app.perception.encoder import EncoderUnavailable, load_encoder

        try:
            encoder = load_encoder(resolved.encoder)
        except EncoderUnavailable as exc:
            encoder = _UnavailableEncoder(resolved.encoder, str(exc))
    if views is None:
        views = _views(paths, timestamps, signals)
    views, chunks = _scene_chunks(views)
    result = track_frames(
        views,
        config=resolved,
        detector=detector,
        encoder=encoder,
        research_mode=profile["research_mode"],
        cache_dir=cache_dir,
        should_stop=should_stop,
    )
    if resolved.strategy == "vlm_only":
        result.perception.used_for_identity = False
    notes = result.perception.notes
    if profile["source"] == "auto":
        notes += " Research mode was characterized automatically and can be overridden."
    model = "affine_partial" if result.compensated else "none"
    return _report(profile, result.compensated, model, notes.strip(), result.points, chunks, result.perception)


class _UnavailableEncoder:
    """Fails on encode so the tracker records the miss instead of swapping models."""

    def __init__(self, encoder_id: str, detail: str):
        self.encoder_id = encoder_id
        self.encoder_version = "unavailable"
        self.preprocess_version = "unavailable"
        self.dimension = 1
        self.detail = detail

    def encode(self, crops):
        from app.perception.types import EncoderUnavailable

        raise EncoderUnavailable(self.detail)


def _views(paths: list[Path], timestamps: list[float], signals: list[MotionSignal]):
    for index, path in enumerate(paths):
        signal = signals[index] if index < len(signals) else None
        image = read_bgr(path)
        yield FrameView(
            timestamp=timestamps[index],
            image=image,
            scene_change=bool(signal and signal.scene_change),
            global_motion=float(signal.global_motion) if signal else 0.0,
            motion_score=float(signal.motion) if signal else 0.0,
            index=index,
        )
        del image


def _scene_chunks(views):
    """Scene chunks start only at confirmed hard cuts, never at a processing chunk edge."""
    chunks: list[dict] = []

    def generate():
        for frame in views:
            if not chunks:
                chunks.append(
                    {
                        "ordinal": 0,
                        "time_start": frame.timestamp,
                        "time_end": frame.timestamp,
                        "scene_change": False,
                    }
                )
            elif frame.scene_change:
                chunks.append(
                    {
                        "ordinal": chunks[-1]["ordinal"] + 1,
                        "time_start": frame.timestamp,
                        "time_end": frame.timestamp,
                        "scene_change": True,
                        "global_motion": frame.global_motion,
                    }
                )
            chunks[-1]["time_end"] = frame.timestamp
            yield frame

    return generate(), chunks


def _legacy_notes(profile: dict, compensated: bool, strategy: str) -> str:
    notes = (
        "Tracks are image-plane motion regions. "
        "World coordinates are not estimated because no calibration or visual-geometry solution ran. "
    )
    if profile["viewpoint"] == "head_mounted":
        notes += "Head-mounted baseline subtracts a partial affine camera motion before blob detection. "
    else:
        notes += "Fixed-camera baseline uses a background subtractor against a stable image frame. "
    if strategy == "vlm_only":
        notes += "These tracks are a visual baseline and were not sent to the model as identity constraints. "
    if profile["source"] == "auto":
        notes += "Research mode was characterized automatically and can be overridden."
    return notes.strip()


def _legacy_perception(points: list[TrackPoint], config: PerceptionConfig, frames: int, notes: str):
    from app.schemas.analysis import InstanceRecord, PerceptionRecord

    grouped: dict[str, list[TrackPoint]] = {}
    for point in points:
        grouped.setdefault(point.track_ref, []).append(point)
    instances = []
    for ref, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item.timestamp)
        uncertain = any(item.continuity in {"uncertain", "reassociated"} for item in rows)
        instances.append(
            InstanceRecord(
                instance_id=ref,
                status=rows[-1].status,
                first_seen=rows[0].timestamp,
                last_seen=rows[-1].timestamp,
                observation_count=len(rows),
                uncertain=uncertain or any(item.continuity == "new" for item in rows),
                detector_id="motion",
            )
        )
    return PerceptionRecord(
        strategy=config.strategy,
        detector_id="motion",
        encoder_status="not_requested",
        frames_tracked=frames,
        tracking_fps=config.target_fps,
        world_coordinates_supported=False,
        used_for_identity=config.strategy != "vlm_only",
        scores_are_probabilities=False,
        diagnostics={"frames": frames, "tracks_created": len(instances), "detections": len(points)},
        instances=instances,
        notes=notes,
    )


def _report(profile, compensated, model, notes, points, chunks, perception) -> SpatialReport:
    return SpatialReport(
        viewpoint=profile["viewpoint"],
        task_constraint=profile["task_constraint"],
        research_mode=profile["research_mode"],
        global_motion=profile["global_motion"],
        scene_change_count=profile["scene_change_count"],
        camera_motion_compensated=bool(compensated),
        compensation_model=model,
        coordinate_frame="image",
        world_coordinates_supported=False,
        notes=notes,
        tracks=points,
        chunks=chunks,
        perception=perception,
    )
