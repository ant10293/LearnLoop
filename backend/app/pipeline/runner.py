"""Durable stage runner. Completed stages are skipped. Failures retry from the failed stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Artifact
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun, Job, PipelineStageRun
from app.models.util import utcnow
from app.perception.types import TrackingCancelled
from app.pipeline.contracts import StructuredOutputError
from app.pipeline.guided import SKIPPED_STAGES, is_guided, run_guided_call
from app.pipeline.infer import (
    BudgetExceeded,
    StageSkipped,
    load_working,
    save_semantic_raw,
    save_working,
)
from app.pipeline.lineage import record_lineage
from app.pipeline.sampling_stage import sample_stage
from app.pipeline.validate import validate_document
from app.providers.types import UnsupportedMode
from app.schemas.analysis import AnalysisDocument
from app.schemas.enums import PIPELINE_STAGES
from app.services.experiments import sends_video_file
from app.services.ingest import playback_path
from app.services.materialize import materialize
from app.services.storage import data_path, relative, resolve_stored
from app.video.frames import contact_sheet

log = logging.getLogger("learnloop.runner")


async def execute_job(session: Session, job: Job) -> None:
    run = session.get(AnalysisRun, job.run_id)
    if run is None or run.status == "canceled":
        job.status = "canceled"
        job.finished_at = utcnow()
        session.commit()
        return
    run_id = run.id
    video = session.get(VideoAsset, run.video_id)
    if video is None:
        _fail_job(session, job, run, "video missing")
        return
    if job.from_stage == OBJECT_VIEWS_JOB:
        await _remake_object_views(session, job, run, video)
        return
    from app.services.appearance import APPEARANCE_JOB, build_gallery_job

    if job.from_stage == APPEARANCE_JOB:
        await build_gallery_job(session, job, run, video)
        return
    if job.from_stage.startswith(REINSPECT_JOB):
        from app.services.reinspection import run_reinspection_job

        await run_reinspection_job(session, job, job.from_stage[len(REINSPECT_JOB) :])
        return
    run.status = "running"
    run.started_at = run.started_at or utcnow()
    run.error = ""
    session.commit()
    started = job.from_stage == ""
    try:
        for name in PIPELINE_STAGES:
            session.refresh(run)
            if run.status == "canceled":
                _cancel_remaining(session, run, name)
                job.status = "canceled"
                job.finished_at = utcnow()
                session.commit()
                return
            if not started:
                if name != job.from_stage:
                    continue
                started = True
            stage = _stage(session, run.id, name)
            if stage.status == "completed":
                continue
            stage.status = "running"
            stage.attempt += 1
            stage.started_at = utcnow()
            stage.error = ""
            stage.progress_current = 0
            stage.progress_total = 1
            job.heartbeat_at = utcnow()
            session.commit()
            try:
                await _dispatch(session, run, video, stage)
            except TrackingCancelled:
                run.status = "canceled"
                _cancel_remaining(session, run, name)
                job.status = "canceled"
                job.finished_at = utcnow()
                session.commit()
                return
            except StageSkipped as skipped:
                stage.status = "skipped"
                stage.checkpoint_json = {"reason": skipped.reason, **(stage.checkpoint_json or {})}
                stage.finished_at = utcnow()
                session.commit()
                continue
            except Exception as exc:
                stage.status = "failed"
                stage.error = _error_text(exc)
                stage.finished_at = utcnow()
                if name in {"segmentation", "evidence_fusion", "reinspection", "transcription", "audio_context"}:
                    session.commit()
                    continue
                raise
            stage.status = "completed"
            stage.progress_current = max(stage.progress_total, 1)
            stage.finished_at = utcnow()
            session.commit()
        run.status = "completed"
        run.finished_at = utcnow()
        job.status = "completed"
        job.finished_at = utcnow()
        session.commit()
        try:
            from app.services.verification_store import settle_waiting

            settle_waiting(session, run.id)
        except Exception:
            session.rollback()
            log.exception("verification waiting on run %s was not settled", run.id)
    except Exception as exc:
        session.rollback()
        if session.get(AnalysisRun, run_id) is None:
            # The run was deleted while it ran, and its job row went with it.
            log.info("run %s was deleted during execution; stopped its job", run_id)
            return
        run.status = "failed"
        run.error = _error_text(exc)
        run.finished_at = utcnow()
        if job.attempt < job.max_attempts and not isinstance(
            exc, (UnsupportedMode, BudgetExceeded, StructuredOutputError)
        ):
            job.status = "pending"
            job.error = _error_text(exc)
            job.worker_id = ""
        else:
            job.status = "failed"
            job.error = _error_text(exc)
            job.finished_at = utcnow()
        session.commit()


def _error_text(exc: BaseException, limit: int = 2000) -> str:
    """Keep the end of a long failure. ffmpeg puts the cause after its banner."""
    text = str(exc).strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _fail_job(session, job, run, message: str) -> None:
    job.status = "failed"
    job.error = message
    job.finished_at = utcnow()
    run.status = "failed"
    run.error = message
    session.commit()


def _cancel_remaining(session, run, current: str) -> None:
    seen = False
    for name in PIPELINE_STAGES:
        if name == current:
            seen = True
        if not seen:
            continue
        stage = _stage(session, run.id, name)
        if stage.status in {"pending", "running"}:
            stage.status = "canceled"
            stage.finished_at = utcnow()


def _stage(session, run_id: str, name: str) -> PipelineStageRun:
    stage = session.scalar(
        select(PipelineStageRun).where(PipelineStageRun.run_id == run_id, PipelineStageRun.stage == name)
    )
    if stage is None:
        stage = PipelineStageRun(run_id=run_id, stage=name, status="pending", checkpoint_json={})
        session.add(stage)
        session.flush()
    return stage


# Not a pipeline stage: a job that only makes object views again for a finished run.
OBJECT_VIEWS_JOB = "object_views"
# Not a pipeline stage: a second look at unresolved steps of one verification, as "reinspect:<verification id>".
REINSPECT_JOB = "reinspect:"


async def _remake_object_views(session, job: Job, run: AnalysisRun, video: VideoAsset) -> None:
    """Live calls were approved for this job. The analysis, its revision, and the run's status are left alone."""
    from app.pipeline.object_views import run_object_views

    stage = _stage(session, run.id, "segmentation")
    stage.status = "running"
    stage.attempt += 1
    stage.started_at = utcnow()
    stage.error = ""
    session.commit()
    try:
        stage.checkpoint_json = await run_object_views(session, run, video, stage, allow_live=True)
        stage.status = "completed"
    except Exception as exc:
        session.rollback()
        stage = _stage(session, run.id, "segmentation")
        stage.status = "skipped"
        stage.checkpoint_json = {"reason": f"Object views were not made: {_error_text(exc, 400)}"}
    stage.finished_at = utcnow()
    job.status = "completed"
    job.finished_at = utcnow()
    session.commit()


async def _object_views(session, run, video, stage) -> None:
    """Pictures for the page. A failure here never costs the person their analysis."""
    from app.pipeline.object_views import run_object_views

    try:
        stage.checkpoint_json = await run_object_views(session, run, video, stage)
    except Exception as exc:
        session.rollback()
        raise StageSkipped(f"Object views were not made: {_error_text(exc, 400)}") from exc


async def _dispatch(session, run: AnalysisRun, video: VideoAsset, stage: PipelineStageRun) -> None:
    name = stage.stage
    if not is_guided(run):
        raise StageSkipped("LearnLoop runs only the hackathon_guided analysis.")
    if name == "segmentation" and (run.config_json or {}).get("object_views"):
        await _object_views(session, run, video, stage)
        return
    if name in SKIPPED_STAGES:
        raise StageSkipped(SKIPPED_STAGES[name])
    if name == "scene_analysis":
        document = await run_guided_call(session, run, video)
        save_semantic_raw(run.id, document)
        save_working(run.id, document)
        _remember(session, run, document, name)
        stage.checkpoint_json = {
            "provider": run.provider_id,
            "model": run.model_id,
            "modality": run.modality,
            "source": (run.config_json or {}).get("inference_source"),
            "objects": len(document.objects),
            "workflow_steps": len(document.workflow_steps),
        }
        return
    if name == "metadata":
        stage.checkpoint_json = {
            "duration_sec": video.duration_sec,
            "fps": video.fps,
            "avg_fps": video.avg_fps,
            "variable_frame_rate": video.variable_frame_rate,
            "width": video.display_width,
            "height": video.display_height,
            "checksum": video.checksum_sha256,
            "rotation_deg": video.rotation_deg,
        }
        stage.progress_total = 1
        stage.progress_current = 1
        return
    if name == "preprocess":
        path = playback_path(video)
        stage.checkpoint_json = {
            "playback": relative(path),
            "proxy": bool(video.proxy_path),
            **_native_subclip(run, video),
        }
        stage.progress_total = 1
        stage.progress_current = 1
        return
    if name == "sample_frames":
        sample_stage(session, run, video, stage)
        return
    if name == "object_extraction":
        _publish_current(session, run)
        stage.checkpoint_json = {"note": "Crops were taken from stored frames during materialization."}
        return
    if name == "validation":
        _validate(session, run, video, stage)
        return
    if name == "artifacts":
        _artifacts(session, run, stage)
        return
    stage.status = "skipped"
    stage.checkpoint_json = {"reason": "unknown stage"}


def _native_subclip(run, video) -> dict:
    """Cut the requested interval for a native-video model. A failed cut fails the stage."""
    from app.pipeline.clip_time import save_transform, subclip_path
    from app.pipeline.sampling_stage import timeline_for
    from app.video.decode import cut_clip

    if not sends_video_file(run):
        return {}
    timeline = timeline_for(video, run.config_json)
    duration = float(video.duration_sec or 0)
    if timeline.start <= 1e-3 and timeline.end >= duration - 1e-3:
        return {}
    dest = subclip_path(run.id)
    transform = cut_clip(resolve_stored(video.stored_path), timeline.start, timeline.end, dest)
    save_transform(run.id, transform)
    return {"subclip": relative(dest), "media_transform": transform}


def _remember(session, run, document: AnalysisDocument, note: str) -> None:
    record_lineage(run, note, document)
    revision = session.scalar(
        select(AnalysisRevision).where(
            AnalysisRevision.run_id == run.id,
            AnalysisRevision.layer == "parsed",
        )
    )
    payload = document.model_dump(mode="json")
    if revision is None:
        revision = AnalysisRevision(
            run_id=run.id,
            layer="parsed",
            version=1,
            document_json=payload,
            note=note,
        )
        session.add(revision)
        session.flush()
    else:
        revision.document_json = payload
        revision.note = note
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id)))
    if not frames:
        frames = list(
            session.scalars(select(FrameAsset).where(FrameAsset.video_id == run.video_id, FrameAsset.run_id.is_(None)))
        )
    materialize(session, run.id, revision.id, document, frames)
    if not _human_active(session, run):
        run.active_revision_id = revision.id
    session.commit()


def _publish_current(session, run) -> None:
    document = load_working(run.id)
    if document.objects or document.actions:
        _remember(session, run, document, "object_extraction")


def _validate(session, run, video, stage) -> None:
    document = load_working(run.id)
    issues = validate_document(document, video.duration_sec)
    errors = [issue for issue in issues if issue.severity == "error"]
    revision = AnalysisRevision(
        run_id=run.id,
        layer="validated",
        version=_next_version(session, run.id, "validated"),
        parent_id=run.active_revision_id,
        document_json=document.model_dump(mode="json"),
        note=f"{len(errors)} validation errors, {len(issues) - len(errors)} warnings",
    )
    session.add(revision)
    session.flush()
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id)))
    materialize(session, run.id, revision.id, document, frames)
    if not _human_active(session, run):
        run.active_revision_id = revision.id
    stage.checkpoint_json = {"errors": len(errors), "warnings": len(issues) - len(errors)}
    stage.progress_total = max(len(issues), 1)
    stage.progress_current = max(len(issues), 1)


def _artifacts(session, run, stage) -> None:
    document = load_working(run.id)
    folder = data_path("runs", run.id)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "analysis": document.model_dump(mode="json"),
        "objects": [item.model_dump(mode="json") for item in document.objects],
        "actions": [item.model_dump(mode="json") for item in document.actions],
        "workflow": {
            "steps": [item.model_dump(mode="json") for item in document.workflow_steps],
            "dependencies": [item.model_dump(mode="json") for item in document.dependencies],
        },
    }
    for kind, payload in files.items():
        path = folder / f"{kind}.json"
        path.write_text(json.dumps(payload, indent=2))
        session.add(Artifact(run_id=run.id, kind=kind, path=relative(path), label=kind))
    frames = list(
        session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp))
    )
    if frames:
        sheet = folder / "run_contact_sheet.jpg"
        contact_sheet([data_path(*Path(frame.path).parts) for frame in frames[:30]], sheet)
        session.add(Artifact(run_id=run.id, kind="contact_sheet", path=relative(sheet), label="Run contact sheet"))
    stage.checkpoint_json = {"files": list(files)}
    stage.progress_total = len(files)
    stage.progress_current = len(files)


def _human_active(session, run) -> bool:
    if not run.active_revision_id:
        return False
    revision = session.get(AnalysisRevision, run.active_revision_id)
    return bool(revision and revision.layer == "human")


def _next_version(session, run_id: str, layer: str) -> int:
    versions = session.scalars(
        select(AnalysisRevision.version).where(
            AnalysisRevision.run_id == run_id,
            AnalysisRevision.layer == layer,
        )
    )
    current = list(versions)
    return (max(current) if current else 0) + 1


