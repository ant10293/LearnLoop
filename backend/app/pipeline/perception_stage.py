"""Run tracking without sending those frames to the language model."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import SceneChunk, SpatialTrackPoint
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRun, PipelineStageRun
from app.perception.presets import config_from_mapping
from app.perception.stream import StreamStats, first_chunk, plan_tracking, stream_views
from app.pipeline.deterministic_doc import document_from_spatial
from app.pipeline.sampling_stage import (
    characterization_signals,
    finalize_semantic_frames,
    load_motion,
    timeline_for,
)
from app.schemas.analysis import Coverage
from app.services.storage import data_path, resolve_stored
from app.video.probe import VideoError
from app.video.sampling import distinct_moments, measure_motion
from app.video.spatial import analyze_sequence
from app.video.timeline import coverage_gaps


def run_spatial_stage(session: Session, run: AnalysisRun, video: VideoAsset, stage: PipelineStageRun) -> None:
    config = config_from_mapping((run.config_json or {}).get("perception"))
    profile = characterization_signals(run.id, config.target_fps)

    def should_stop() -> bool:
        session.refresh(run)
        return run.status == "canceled"

    stream = _stream(run, video, config) if config.separate_stream else None
    if stream is not None and stream.get("views") is not None:
        stats: StreamStats = stream["stats"]
        report = analyze_sequence(
            [],
            [],
            [],
            run.research_mode,
            config=config,
            cache_dir=data_path("runs", run.id, "embeddings"),
            should_stop=should_stop,
            views=stream["views"],
            profile_signals=profile if profile is not None else [],
        )
        tracking = stats.summary(stream["plan"], config.occlusion_seconds)
        frames_tracked = stats.frames
    else:
        paths, timestamps = _sampled(session, run, video)
        signals = measure_motion(paths, timestamps) if len(paths) >= 2 else []
        report = analyze_sequence(
            paths,
            timestamps,
            signals,
            run.research_mode,
            config=config,
            cache_dir=data_path("runs", run.id, "embeddings"),
            should_stop=should_stop,
            profile_signals=profile,
        )
        largest, gaps = coverage_gaps(timestamps, *_interval(run, video), config.occlusion_seconds)
        tracking = {
            "frame_source": "sampled_frames",
            "frames": len(paths),
            "first_time": timestamps[0] if timestamps else None,
            "last_time": timestamps[-1] if timestamps else None,
            "max_gap_sec": largest,
            "gaps_over_occlusion": gaps,
            "continuity": "insufficient" if gaps else "supported",
            "notes": ["Tracking ran on the frames sampled for the model."],
        }
        if stream is not None and stream.get("limitation"):
            tracking["limitation"] = stream["limitation"]
        frames_tracked = len(paths)
    if report.perception is not None:
        report.perception.frames_tracked = frames_tracked
        report.perception.diagnostics = {**report.perception.diagnostics, "tracking_continuity": tracking["continuity"]}
    if run.research_mode == "auto":
        run.viewpoint = report.viewpoint
        run.task_constraint = report.task_constraint
        run.research_mode = report.research_mode
        stored = dict(run.config_json or {})
        stored["auto_characterization"] = {
            "mode": report.research_mode,
            "global_motion": report.global_motion,
            "scene_changes": report.scene_change_count,
            "notes": report.notes,
        }
        run.config_json = stored
    session.query(SpatialTrackPoint).filter(SpatialTrackPoint.run_id == run.id).delete()
    session.query(SceneChunk).filter(SceneChunk.run_id == run.id).delete()
    for point in report.tracks:
        session.add(
            SpatialTrackPoint(
                run_id=run.id,
                video_id=video.id,
                track_ref=point.track_ref,
                timestamp=point.timestamp,
                bbox_json={"x": point.x, "y": point.y, "w": point.w, "h": point.h},
                motion_score=point.motion_score,
                continuity=point.continuity,
                coordinate_frame="image",
                camera_compensated=point.camera_compensated,
                scene_chunk=point.scene_chunk,
                identity_note=point.note,
                track_status=point.status,
                association_score=point.association_score,
                detector_id=point.detector_id,
            )
        )
    for chunk in report.chunks:
        session.add(
            SceneChunk(
                run_id=run.id,
                ordinal=chunk["ordinal"],
                time_start=chunk["time_start"],
                time_end=chunk["time_end"],
                scene_change=bool(chunk.get("scene_change")),
                global_motion=float(chunk.get("global_motion") or report.global_motion),
                notes=report.notes,
            )
        )
    finalized = finalize_semantic_frames(session, run, video, report)
    coverage = Coverage.model_validate(run.coverage_json or {"modality": "sampled_frames", "sampling_preset": run.sampling_preset})
    coverage = coverage.model_copy(update={"tracking": tracking})
    run.coverage_json = coverage.model_dump(mode="json")
    document = document_from_spatial(report, coverage, width=video.display_width, height=video.display_height)
    spatial_path = data_path("runs", run.id, "spatial.json")
    spatial_path.parent.mkdir(parents=True, exist_ok=True)
    spatial_path.write_text(document.model_dump_json())
    diagnostics = dict(report.perception.diagnostics) if report.perception else {}
    diagnostics["frame_source"] = tracking["frame_source"]
    if tracking.get("limitation"):
        diagnostics["limitation"] = tracking["limitation"]
    stage.checkpoint_json = {
        "tracks": len({point.track_ref for point in report.tracks}),
        "points": len(report.tracks),
        "global_motion": report.global_motion,
        "scene_changes": report.scene_change_count,
        "world_coordinates_supported": False,
        "compensation_model": report.compensation_model,
        "notes": report.notes,
        "strategy": config.strategy,
        "detector": config.detector,
        "encoder": config.encoder,
        "encoder_status": report.perception.encoder_status if report.perception else "not_requested",
        "encoder_detail": report.perception.encoder_detail if report.perception else "",
        "diagnostics": diagnostics,
        "tracking": tracking,
        "semantic_swaps": finalized["swaps"],
    }
    stage.progress_total = max(frames_tracked, 1)
    stage.progress_current = frames_tracked


def _interval(run: AnalysisRun, video: VideoAsset) -> tuple[float, float]:
    coverage = run.coverage_json or {}
    start = coverage.get("time_start")
    end = coverage.get("time_end")
    if start is None or end is None:
        timeline = timeline_for(video, run.config_json)
        return timeline.start, timeline.end
    return float(start), float(end)


def _stream(run: AnalysisRun, video: VideoAsset, config) -> dict:
    """Plan and open the tracking stream. A decode failure is reported, then sampled frames are used."""
    timeline = timeline_for(video, run.config_json)
    signals, _step = load_motion(run.id)
    plan = plan_tracking(
        signals,
        timeline.start,
        timeline.end,
        config.target_fps,
        config.maximum_frames,
        config.occlusion_seconds,
    )
    source = resolve_stored(video.stored_path)
    work = data_path("runs", run.id, "track_frames")
    try:
        opening = first_chunk(source, plan, work)
    except (VideoError, OSError) as exc:
        return {"limitation": f"Tracking-frame extraction failed ({str(exc)[-300:]}). Sampled frames were used."}
    if len(opening) < 2 and timeline.span > 1.0 / max(plan.floor_fps, 1e-3):
        return {"limitation": "Tracking-frame extraction returned too few frames. Sampled frames were used."}
    stats = StreamStats()
    return {"plan": plan, "stats": stats, "views": stream_views(source, plan, work, stats, opening)}


def _sampled(session: Session, run: AnalysisRun, video: VideoAsset) -> tuple[list[Path], list[float]]:
    sampled = list(
        session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp))
    )
    if len(sampled) < 2:
        sampled = list(
            session.scalars(
                select(FrameAsset)
                .where(FrameAsset.video_id == video.id, FrameAsset.run_id.is_(None))
                .order_by(FrameAsset.timestamp)
            )
        )
    sampled = distinct_moments(sampled)
    return [data_path(*Path(frame.path).parts) for frame in sampled], [frame.timestamp for frame in sampled]


def load_spatial_document(run_id: str):
    path = data_path("runs", run_id, "spatial.json")
    if not path.exists():
        return None
    from app.schemas.analysis import AnalysisDocument

    return AnalysisDocument.model_validate(json.loads(path.read_text()))


__all__ = ["run_spatial_stage", "load_spatial_document"]
