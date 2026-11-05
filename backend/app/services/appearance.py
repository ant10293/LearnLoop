"""Appearance galleries for learned procedures: status, and the job that builds a missing one.

A gallery belongs to the demonstration analysis a procedure was learned from, so
every version of that procedure reads the same references. It is usually built
while the demonstration's object views are made. Learning a procedure makes sure
it exists: when it does not, one queued job rebuilds it from the stored SAM and
locate answers. That job sends nothing to a provider.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.runs import AnalysisRun, Job, PipelineStageRun
from app.perception import appearance as app_

APPEARANCE_JOB = "appearance_gallery"
# Made again although one exists, because a reviewer corrected the boxes it learns from.
APPEARANCE_REBUILD_JOB = "appearance_gallery_rebuild"
_JOBS = (APPEARANCE_JOB, APPEARANCE_REBUILD_JOB)
STAGE = "appearance"


def gallery_status(session: Session, run_id: str) -> dict:
    """What a page shows: ready, partial, unavailable, building, failed, disabled, or missing. Never vectors."""
    if not settings.appearance_enabled or not settings.auto_gallery:
        return {"status": "disabled", "reason": "Appearance references are turned off."}
    job = _latest_job(session, run_id)
    if job is not None and job.from_stage == APPEARANCE_REBUILD_JOB and job.status in {"pending", "running"}:
        return {"status": "building", "rebuilding": True}
    manifest, _vectors = app_.read_gallery(run_id)
    if manifest is not None:
        return app_.gallery_summary(manifest)
    if job is not None and job.status in {"pending", "running"}:
        return {"status": "building"}
    stage = session.scalar(select(PipelineStageRun).where(PipelineStageRun.run_id == run_id, PipelineStageRun.stage == STAGE))
    if job is not None and stage is not None:
        return {"status": "unavailable", "reason": (stage.checkpoint_json or {}).get("reason") or stage.error or "No references were made."}
    segmentation = session.scalar(select(PipelineStageRun).where(PipelineStageRun.run_id == run_id, PipelineStageRun.stage == "segmentation"))
    if segmentation is not None and segmentation.status == "running":
        return {"status": "building"}
    return {"status": "missing"}


def ensure_gallery(session: Session, run_id: str) -> dict:
    """Queue one rebuild when a learned demonstration has no gallery. Calling it again queues nothing more."""
    status = gallery_status(session, run_id)
    if status["status"] != "missing":
        return status
    run = session.get(AnalysisRun, run_id)
    if run is None or not (run.config_json or {}).get("object_views"):
        return {"status": "unavailable", "reason": "This analysis did not follow objects with SAM, so there are no crops to learn from."}
    if _latest_job(session, run_id) is not None:
        return status
    session.add(Job(run_id=run_id, from_stage=APPEARANCE_JOB, max_attempts=1))
    session.commit()
    return {"status": "building"}


def _latest_job(session: Session, run_id: str) -> Job | None:
    return session.scalar(
        select(Job).where(Job.run_id == run_id, Job.from_stage.in_(_JOBS)).order_by(Job.created_at.desc())
    )


def rebuild_after_correction(session: Session, run_id: str) -> dict:
    """A demonstration's references learn from its boxes, so a corrected box makes them again, from stored answers only.
    An attempt's boxes never teach references. One waiting rebuild covers every correction made before it starts."""
    if not settings.appearance_enabled or not settings.auto_gallery:
        return {"status": "disabled"}
    run = session.get(AnalysisRun, run_id)
    if run is None or (run.config_json or {}).get("reference_template_id") or not (run.config_json or {}).get("object_views"):
        return {"status": "unchanged"}
    waiting = session.scalar(
        select(Job.id).where(Job.run_id == run_id, Job.from_stage.in_(_JOBS), Job.status == "pending")
    )
    if waiting is None:
        session.add(Job(run_id=run_id, from_stage=APPEARANCE_REBUILD_JOB, max_attempts=1))
        session.commit()
    return {"status": "building", "rebuilding": True}


async def build_gallery_job(session: Session, job: Job, run: AnalysisRun, video, force: bool = False) -> None:
    """Rebuild object views from stored answers only. A missing answer leaves the views as they were and says why.
    force makes the gallery again although one exists, as after a reviewer corrected a box."""
    from app.models.util import utcnow
    from app.pipeline.object_views import run_object_views
    from app.pipeline.runner import _error_text, _stage

    stage = _stage(session, run.id, STAGE)
    stage.status = "running"
    stage.attempt += 1
    stage.started_at = utcnow()
    stage.error = ""
    session.commit()
    try:
        if not force and app_.read_gallery(run.id)[0] is not None:
            stage.checkpoint_json = {"reason": "A gallery of the current version already existed."}
        else:
            out = await run_object_views(session, run, video, stage, allow_live=False, keep_on_empty=True)
            manifest, _vectors = app_.read_gallery(run.id)
            if manifest is None:
                reason = out.get("error") or (out.get("appearance") or {}).get("error") or "No gallery was written."
                stage.checkpoint_json = {"reason": f"References were not made: {reason}"}
            else:
                stage.checkpoint_json = app_.gallery_summary(manifest)
        stage.status = "completed"
    except Exception as exc:
        session.rollback()
        stage = _stage(session, run.id, STAGE)
        stage.status = "failed"
        stage.error = _error_text(exc, 400)
        stage.checkpoint_json = {"reason": f"References were not made: {_error_text(exc, 400)}"}
    stage.finished_at = utcnow()
    job.status = "completed"
    job.finished_at = utcnow()
    session.commit()
