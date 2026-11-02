"""Semantic frame selection on the source clock.

`sample_frames` runs a full-interval motion pass and stores a provisional
frame set. `finalize_semantic_frames` runs at the end of spatial tracking and
may trade low-value interior frames for track events. The semantic model only
sees the final set.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from app.models.media import FrameAsset
from app.schemas.analysis import Coverage
from app.services.ingest import sample_run_frames
from app.services.storage import data_path, resolve_stored
from app.video.decode import gray_pass
from app.video.probe import VideoError
from app.video.sampling import (
    MotionSignal,
    SampledFrame,
    annotate_sharpness,
    budget_indices,
    motion_signals,
    plan_interval,
    preset_fps,
    select_frames,
    window_frames,
)
from app.video.timeline import SourceTimeline, coverage_gaps, load_timeline

MOTION_FPS = 4.0
# The motion pass never decodes more frames than this. A longer interval is
# sampled more sparsely across its whole length, not cut short.
MOTION_CAP = 2400


def timeline_for(video, config: dict | None) -> SourceTimeline:
    cache = data_path("originals", f"{video.checksum_sha256[:12]}.pts.json")
    return load_timeline(video, config, cache)


def motion_path(run_id: str) -> Path:
    return data_path("runs", run_id, "motion.json")


def motion_step(span: float) -> float:
    if span <= 0:
        return 1.0 / MOTION_FPS
    return max(1.0 / MOTION_FPS, span / MOTION_CAP)


def run_motion_pass(source: Path, timeline: SourceTimeline, run_id: str) -> tuple[list[MotionSignal], float]:
    """Decode the whole interval at low resolution and keep only the signals."""
    step = motion_step(timeline.span)
    images, times = gray_pass(source, timeline.start, timeline.end, step)
    times = [timeline.snap(item) if timeline.basis == "presentation" else item for item in times]
    signals = motion_signals(images, times, window_frames(times))
    del images
    save_motion(run_id, signals, step, timeline)
    return signals, step


def save_motion(run_id: str, signals: list[MotionSignal], step: float, timeline: SourceTimeline) -> None:
    path = motion_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "step": step,
                "start": timeline.start,
                "end": timeline.end,
                "time_basis": timeline.basis,
                "signals": [
                    {
                        "t": item.timestamp,
                        "motion": item.motion,
                        "global_motion": item.global_motion,
                        "boundary": item.boundary,
                        "reason": item.boundary_reason,
                    }
                    for item in signals
                ],
            }
        )
    )


def load_motion(run_id: str) -> tuple[list[MotionSignal], float]:
    path = motion_path(run_id)
    if not path.exists():
        return [], 0.0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [], 0.0
    signals = [
        MotionSignal(
            timestamp=float(row["t"]),
            motion=float(row.get("motion") or 0),
            scene_change=row.get("boundary") == "hard",
            global_motion=float(row.get("global_motion") or 0),
            boundary=str(row.get("boundary") or "none"),
            boundary_reason=str(row.get("reason") or ""),
        )
        for row in data.get("signals") or []
    ]
    return signals, float(data.get("step") or 0)


def sample_stage(session, run, video, stage) -> None:
    config = run.config_json or {}
    timeline = timeline_for(video, config)
    start, end = timeline.start, timeline.end
    source = resolve_stored(video.stored_path)
    limit = max(int(config.get("max_frames") or 48), 1)
    preset = run.sampling_preset
    native = run.modality == "native_video"
    notes: list[str] = list(timeline.notes)
    signals: list[MotionSignal] = []
    motion_note = ""
    try:
        signals, step = run_motion_pass(source, timeline, run.id)
    except (VideoError, OSError) as exc:
        if preset == "adaptive" and not native:
            raise
        motion_note = f"The motion pass failed ({str(exc)[-200:]}). Tracking density uses the floor rate."
        step = 0.0
    if motion_note:
        notes.append(motion_note)
    selection = None
    if native:
        samples = plan_interval(timeline, 1.0)
        notes.append(
            "The model receives the requested interval as a video file. Preview frames were extracted so the "
            "inspector can show evidence."
        )
    elif preset == "adaptive":
        selection = select_frames(signals, start, end, limit)
        samples = selection.frames
        notes.append(
            f"Adaptive sampling chose {len(samples)} frames from a {len(signals)}-frame motion pass that covered "
            f"{start:.2f}s through {end:.2f}s. This is not full-frame coverage."
        )
    else:
        fps = preset_fps(preset, config.get("custom_fps"))
        samples = plan_interval(timeline, fps)
        span = f" from {samples[0].timestamp:.3f} s through {samples[-1].timestamp:.3f} s" if samples else ""
        notes.append(
            f"Uniform {fps:g} FPS sampling snapped to presentation timestamps{span}. "
            f"{len(samples)} frames will be available. The source has about {video.frame_count} frames."
        )
    if len(samples) > limit:
        samples = [samples[index] for index in budget_indices(len(samples), limit)]
        if native:
            notes.append(f"Inspector previews were limited to {limit}. The model still receives the whole interval.")
        else:
            notes.append(f"The frame budget kept {limit} frames spread across the whole interval.")
    by_time = {round(item.timestamp, 3): item for item in signals}
    for sample in samples:
        signal = by_time.get(round(sample.timestamp, 3))
        if signal is not None:
            sample.motion_score = max(sample.motion_score, signal.motion)
            sample.scene_change = sample.scene_change or signal.scene_change
    rows = _store(session, video, run.id, samples)
    reasons = Counter(sample.reason for sample in samples)
    times = [row.timestamp for row in rows]
    largest, gaps = (None, [])
    if not native:
        largest, gaps = coverage_gaps(times, start, end, _gap_limit(start, end, limit))
    boundaries = selection.boundaries if selection else [
        {"time": round(item.timestamp, 3), "kind": item.boundary, "reason": item.boundary_reason}
        for item in signals
        if item.boundary != "none"
    ]
    coverage = Coverage(
        modality="native_video" if native else "sampled_frames",
        sampling_preset=preset,
        nominal_fps=preset_fps(preset, config.get("custom_fps")) if preset != "adaptive" else MOTION_FPS,
        frames_extracted=len(rows),
        frames_submitted=0 if native else len(rows),
        frame_timestamps=times,
        selection_reasons=dict(reasons),
        duration_sec=video.duration_sec,
        native_video_submitted=native,
        temporal_coverage_note=" ".join(notes),
        time_start=round(start, 3),
        time_end=round(end, 3),
        time_basis=timeline.basis,
        max_gap_sec=largest,
        uncovered_gaps=gaps,
        motion_frames=len(signals),
        candidate_peaks=selection.candidate_peaks if selection else [],
        selected_peaks=selection.selected_peaks if selection else [],
        budget_exclusions=selection.budget_exclusions if selection else 0,
        scene_boundaries=boundaries,
        semantic_selection="provisional",
        semantic_mode=str(config.get("semantic_mode") or "combined"),
        media_transforms=_transforms(run.id),
    )
    run.coverage_json = coverage.model_dump(mode="json")
    stage.progress_total = max(len(rows), 1)
    stage.progress_current = len(rows)
    stage.checkpoint_json = {
        "frames": len(rows),
        "reasons": dict(reasons),
        "motion_frames": len(signals),
        "motion_step": step,
        "scene_changes": [item["time"] for item in boundaries if item["kind"] == "hard"],
        "visual_changes": [item["time"] for item in boundaries if item["kind"] == "soft"],
    }
    session.commit()


def _transforms(run_id: str) -> list[dict]:
    from app.pipeline.clip_time import load_transform

    transform = load_transform(run_id)
    return [transform] if transform else []


def _gap_limit(start: float, end: float, budget: int) -> float:
    """A gap longer than twice the even spacing the budget allows is reported as uncovered."""
    return 2.0 * max(end - start, 0.0) / max(budget, 1)


def _store(session, video, run_id: str, samples: list[SampledFrame]) -> list[FrameAsset]:
    rows = sample_run_frames(session, video, run_id, samples)
    paths = {round(row.timestamp, 3): data_path(*Path(row.path).parts) for row in rows}
    annotate_sharpness(samples, paths)
    for sample, row in zip(samples, rows, strict=False):
        row.sharpness = sample.sharpness
        row.motion_score = sample.motion_score
        row.scene_change = sample.scene_change
    return rows


def track_events(points, minimum_hits: int = 2) -> list[float]:
    """Times a track with repeated non-flow detections appeared or was last detected."""
    grouped: dict[str, list] = {}
    for point in points:
        if point.detector_id == "optical_flow":
            continue
        grouped.setdefault(point.track_ref, []).append(point.timestamp)
    events: set[float] = set()
    for times in grouped.values():
        if len(times) < minimum_hits:
            continue
        events.add(round(min(times), 3))
        events.add(round(max(times), 3))
    return sorted(events)


def plan_swaps(
    current: list[SampledFrame],
    events: list[float],
    start: float,
    end: float,
    budget: int,
    timeline: SourceTimeline,
) -> tuple[list[SampledFrame], list[dict]]:
    """Trade low-value interior frames for track events without opening an uncovered gap."""
    limit = _gap_limit(start, end, budget)
    spacing = max(end - start, 0.0) / max(budget, 1)
    frames = sorted(current, key=lambda item: item.timestamp)
    swaps: list[dict] = []
    for event in events:
        snapped = timeline.snap(event)
        if any(abs(frame.timestamp - snapped) < spacing / 2 for frame in frames):
            continue
        best = None
        for index in range(1, len(frames) - 1):
            frame = frames[index]
            if frame.reason not in {"uniform", "gap_fill"}:
                continue
            trial = [item.timestamp for position, item in enumerate(frames) if position != index] + [snapped]
            largest, _gaps = coverage_gaps(trial, start, end, limit)
            if limit > 0 and largest > limit + 1e-6:
                continue
            key = (frame.motion_score, abs(frame.timestamp - snapped), frame.timestamp)
            if best is None or key < best[0]:
                best = (key, index)
        if best is None:
            continue
        removed = frames.pop(best[1])
        frames.append(SampledFrame(timestamp=snapped, reason="track_event"))
        frames.sort(key=lambda item: item.timestamp)
        swaps.append({"removed": round(removed.timestamp, 3), "added": round(snapped, 3), "reason": "track_event"})
    return frames, swaps


def finalize_semantic_frames(session, run, video, report) -> dict:
    """Last step of spatial tracking. Only adaptive sampled-frame runs trade frames."""
    coverage = Coverage.model_validate(run.coverage_json or {"modality": "sampled_frames", "sampling_preset": run.sampling_preset})
    rows = list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp)))
    swaps: list[dict] = []
    if run.sampling_preset == "adaptive" and run.modality != "native_video" and len(rows) >= 3:
        config = run.config_json or {}
        timeline = timeline_for(video, config)
        budget = max(int(config.get("max_frames") or 48), 1)
        current = [
            SampledFrame(row.timestamp, row.reason, row.motion_score or 0, row.sharpness or 0, bool(row.scene_change))
            for row in rows
        ]
        final, swaps = plan_swaps(current, track_events(report.tracks), timeline.start, timeline.end, budget, timeline)
        if swaps:
            rows = _store(session, video, run.id, final)
    times = [row.timestamp for row in rows]
    updates: dict = {"semantic_selection": "final", "selection_swaps": swaps}
    if swaps:
        start = coverage.time_start if coverage.time_start is not None else 0.0
        end = coverage.time_end if coverage.time_end is not None else float(video.duration_sec or 0)
        budget = max(int((run.config_json or {}).get("max_frames") or 48), 1)
        largest, gaps = coverage_gaps(times, start, end, _gap_limit(start, end, budget))
        updates.update(
            {
                "frame_timestamps": times,
                "frames_extracted": len(rows),
                "frames_submitted": len(rows),
                "selection_reasons": dict(Counter(row.reason for row in rows)),
                "max_gap_sec": largest,
                "uncovered_gaps": gaps,
            }
        )
    run.coverage_json = coverage.model_copy(update=updates).model_dump(mode="json")
    return {"swaps": len(swaps)}


def characterization_signals(run_id: str, peak_fps: float) -> list[MotionSignal] | None:
    """Motion-pass signals with flow expressed per tracking frame. None when no pass was stored."""
    from app.perception.stream import scale_global_motion

    signals, step = load_motion(run_id)
    if not signals:
        return None
    return scale_global_motion(signals, step, 1.0 / max(peak_fps, 1e-3)) if math.isfinite(step) else signals
