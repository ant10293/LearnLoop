"""Run Reliability Lab repetitions side by side, up to a bound.

Two parts, both in the API process like the ordinary worker:

tick()            launches queued repetitions into free slots (per batch and overall), and
                  finishes repetitions whose pipeline work ended, recording results as they arrive.
reliability_loop  claims jobs of experiment runs (and only those) and runs each on its own thread
                  with its own event loop and database session, through the ordinary execute_job.
                  At most settings.reliability_max_concurrency run at once.

Each repetition makes its own ordinary analysis run (and verification report), so its
files live under runs/<its own id>/ and no two repetitions share mutable state. The work
overlaps for real: model requests are awaited network calls, and ffmpeg and SAM decoding
are subprocesses, so threads progress while one another wait.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.media import VideoAsset
from app.models.reliability import ReliabilityBatch, ReliabilityRun
from app.models.runs import AnalysisRevision, AnalysisRun, Job
from app.models.util import elapsed_seconds, utcnow
from app.models.verification import LearnedWorkflowTemplate, VerificationRunRow
from app.reliability import normalize
from app.reliability.service import ACTIVE

log = logging.getLogger("learnloop.reliability")


def _session() -> Session:
    from app.db import SessionLocal

    return SessionLocal()


def tick() -> None:
    session = _session()
    try:
        finish_running(session)
        launch_queued(session)
        settle_batches(session)
    finally:
        session.close()


# ---------------------------------------------------------------- launching


def launch_queued(session: Session) -> int:
    cap = settings.reliability_max_concurrency
    running_total = session.scalar(select(func.count()).select_from(ReliabilityRun).where(ReliabilityRun.status == "running")) or 0
    launched = 0
    batches = session.scalars(select(ReliabilityBatch).where(ReliabilityBatch.status.in_(("queued", "running"))).order_by(ReliabilityBatch.created_at)).all()
    for batch in batches:
        running = session.scalar(
            select(func.count()).select_from(ReliabilityRun).where(ReliabilityRun.batch_id == batch.id, ReliabilityRun.status == "running")
        ) or 0
        queued = session.scalars(
            select(ReliabilityRun).where(ReliabilityRun.batch_id == batch.id, ReliabilityRun.status == "queued").order_by(ReliabilityRun.index)
        ).all()
        for run in queued:
            if running >= batch.max_concurrency or running_total >= cap:
                break
            if batch.status == "queued":
                batch.status = "running"
                batch.started_at = batch.started_at or utcnow()
                session.commit()
            try:
                _launch(session, batch, run)
            except Exception as exc:
                session.rollback()
                run = session.get(ReliabilityRun, run.id)
                _finish(run, "failed", f"The run could not be started: {str(exc)[:800]}", normalize.classify_error(str(exc)))
                session.commit()
                log.warning("reliability run %s could not start: %s", run.id, exc)
                continue
            running += 1
            running_total += 1
            launched += 1
    return launched


def _launch(session: Session, batch: ReliabilityBatch, run: ReliabilityRun) -> None:
    """One ordinary analysis run, made from the batch's frozen settings, and its report for verification.
    Its job is added only after the run is recorded as an experiment, so the ordinary worker never claims it."""
    from app.pipeline import guided
    from app.services import verification_store as store

    config = batch.config_json or {}
    video = session.get(VideoAsset, batch.video_id)
    if video is None:
        raise LookupError("The recording was deleted.")
    template_document = config.get("template_document")
    if batch.mode == "verification" and session.get(LearnedWorkflowTemplate, batch.template_id) is None:
        raise LookupError("The procedure version was deleted.")
    run.started_at = utcnow()
    analysis = guided.create_guided_run(
        session,
        video,
        template=template_document,
        template_id=batch.template_id,
        allow_live=bool(config.get("allow_live")),
        model_id=config["model_id"],
        segmentation_model_id=config.get("segmentation_model_id") or "",
        fresh=batch.independent_inference,
        enqueue=False,
        extra_config={
            "reliability_batch_id": batch.id,
            "reliability_run_id": run.id,
            "reliability_index": run.index,
            "cache_policy": config.get("cache_policy") or "shared",
        },
    )
    analysis.name = f"Reliability {batch.id[:8]} #{run.index} · {analysis.name}"
    run.analysis_run_id = analysis.id
    session.commit()
    if batch.mode == "verification":
        report = store.start_verification(session, batch.template_id, {"execution_run_id": analysis.id, "strategy": "tracked"})
        run.verification_id = report["id"]
    # Running and its job become visible together, so a running repetition always has work queued.
    run.status = "running"
    session.add(Job(run_id=analysis.id, max_attempts=1))
    session.commit()


# ---------------------------------------------------------------- finishing


def finish_running(session: Session) -> int:
    finished = 0
    for run in session.scalars(select(ReliabilityRun).where(ReliabilityRun.status == "running")).all():
        batch = session.get(ReliabilityBatch, run.batch_id)
        try:
            if _settle(session, batch, run):
                finished += 1
            session.commit()
        except Exception as exc:
            session.rollback()
            log.exception("reliability run %s could not be finished", run.id)
            run = session.get(ReliabilityRun, run.id)
            _finish(run, "failed", f"The result could not be read: {str(exc)[:800]}", "pipeline_error")
            session.commit()
    return finished


def _pending_jobs(session: Session, analysis_run_id: str) -> list[Job]:
    return list(session.scalars(select(Job).where(Job.run_id == analysis_run_id, Job.status.in_(("pending", "running")))))


def _settle(session: Session, batch: ReliabilityBatch, run: ReliabilityRun) -> bool:
    from app.services import verification_store as store

    analysis = session.get(AnalysisRun, run.analysis_run_id) if run.analysis_run_id else None
    if analysis is None:
        _finish(run, "failed", "The analysis run for this repetition was deleted.", "pipeline_error")
        return True
    jobs = _pending_jobs(session, analysis.id)
    # A cancel that arrives after the work already finished keeps the finished result.
    if run.cancel_requested and not (analysis.status == "completed" and not jobs):
        # Pending work is dropped; work already on a thread is let finish, then the run is recorded as canceled.
        for job in jobs:
            if job.status == "pending":
                job.status = "canceled"
                job.finished_at = utcnow()
        if any(job.status == "running" for job in jobs):
            return False
        run.result_json = normalize.failure_result(session, analysis)
        _finish(run, "canceled", "Canceled from the Reliability Lab.", "canceled", _ended(session, analysis))
        return True
    if analysis.status in {"pending", "running"}:
        if jobs:
            return False
        # No job will ever move this run: its job was stopped by a restart, or never queued. Record why.
        last = session.scalar(select(Job).where(Job.run_id == analysis.id).order_by(Job.created_at.desc()))
        error = (last.error if last is not None else "") or "The run's job stopped before the analysis finished."
        analysis.status = "failed"
        analysis.error = analysis.error or error
        if batch.mode == "verification":
            store.settle_waiting(session, analysis.id)
        run.result_json = normalize.failure_result(session, analysis)
        _finish(run, "failed", error, "interrupted", _ended(session, analysis))
        return True
    if analysis.status in {"failed", "canceled"}:
        if batch.mode == "verification":
            store.settle_waiting(session, analysis.id)
        run.result_json = normalize.failure_result(session, analysis)
        if analysis.status == "canceled":
            _finish(run, "canceled", analysis.error or "The analysis was canceled.", "canceled", _ended(session, analysis))
        else:
            error = analysis.error or "The analysis failed without a message."
            _finish(run, "failed", error, normalize.classify_error(error), _ended(session, analysis))
        return True
    if batch.mode == "verification":
        return _settle_verification(session, run, analysis, jobs)
    return _settle_learning(session, run, analysis, jobs)


def _settle_verification(session: Session, run: ReliabilityRun, analysis: AnalysisRun, jobs: list[Job]) -> bool:
    from app.services import verification_store as store

    row = session.get(VerificationRunRow, run.verification_id) if run.verification_id else None
    if row is None:
        _finish(run, "failed", "The verification report for this repetition is missing.", "pipeline_error")
        return True
    if row.status == "waiting":
        store.settle_waiting(session, analysis.id)
        session.refresh(row)
    if row.status == "failed":
        run.result_json = normalize.failure_result(session, analysis)
        _finish(run, "failed", row.error or "The verification failed.", normalize.classify_error(row.error), _ended(session, analysis))
        return True
    if row.status != "completed":
        return False
    # The second look runs as its own job on this repetition's analysis run. Wait for it.
    if jobs:
        return False
    result = normalize.verification_result(session, analysis, row)
    if (row.report_json or {}).get("reinspection", {}).get("status") == "pending":
        result["diagnostics"]["reinspection"]["incomplete"] = True
    run.result_json = result
    if result["outcome"] == normalize.INVALID:
        _finish(run, "failed", result["invalid_reason"], "invalid_result", _ended(session, analysis))
    else:
        _finish(run, "completed", "", "", _ended(session, analysis))
    return True


def _settle_learning(session: Session, run: ReliabilityRun, analysis: AnalysisRun, jobs: list[Job]) -> bool:
    """Learn the procedure the ordinary way, but keep it with the repetition. It is never added to the procedure."""
    from app.verification.learn import learn_template

    if jobs:
        return False
    revision = session.get(AnalysisRevision, analysis.active_revision_id) if analysis.active_revision_id else None
    if revision is None:
        _finish(run, "failed", "The analysis finished without a revision to learn from.", "invalid_result", _ended(session, analysis))
        return True
    learned = learn_template(
        revision.document_json,
        promote_supported=True,
        results_first=True,
        source_run_id=analysis.id,
        source_revision_id=revision.id,
        source_video_id=analysis.video_id,
    ).model_dump(mode="json")
    run.learned_json = learned
    run.result_json = normalize.learning_result(session, analysis, learned)
    _finish(run, "completed", "", "", _ended(session, analysis))
    return True


def _ended(session: Session, analysis: AnalysisRun):
    """When this repetition's own work ended: the analysis, or a later job on it such as the second look."""
    times = [analysis.finished_at] + list(session.scalars(select(Job.finished_at).where(Job.run_id == analysis.id, Job.finished_at.is_not(None))))
    times = [item for item in times if item is not None]
    if not times:
        return None
    return max(times, key=lambda item: elapsed_seconds(times[0], item))


def _finish(run: ReliabilityRun, status: str, error: str, kind: str, ended=None) -> None:
    run.status = status
    run.error = error
    run.error_kind = kind
    run.finished_at = ended or utcnow()
    if run.started_at:
        run.duration_seconds = round(max(elapsed_seconds(run.started_at, run.finished_at), 0.0), 3)


def settle_batches(session: Session) -> None:
    for batch in session.scalars(select(ReliabilityBatch).where(ReliabilityBatch.status.in_(ACTIVE))).all():
        statuses = list(session.scalars(select(ReliabilityRun.status).where(ReliabilityRun.batch_id == batch.id)))
        if any(status in {"queued", "running"} for status in statuses):
            continue
        batch.status = "canceled" if batch.status == "canceling" else "completed"
        finished = list(session.scalars(select(ReliabilityRun.finished_at).where(ReliabilityRun.batch_id == batch.id, ReliabilityRun.finished_at.is_not(None))))
        batch.finished_at = utcnow()
        if finished and batch.started_at:
            batch.finished_at = max(finished, key=lambda item: elapsed_seconds(batch.started_at, item))
    session.commit()


# ---------------------------------------------------------------- the lane


async def reliability_loop(stop: asyncio.Event) -> None:
    """Claim experiment jobs while slots are free and run each on a thread. The ordinary worker never takes these."""
    from app.jobs.worker import RELIABILITY_LANE, claim_next, run_claimed_job

    worker_id = f"lab-{uuid4().hex[:8]}"
    active: set[asyncio.Task] = set()
    while not stop.is_set():
        try:
            await asyncio.to_thread(tick)
            while len(active) < settings.reliability_max_concurrency:
                job_id = await asyncio.to_thread(claim_next, worker_id, RELIABILITY_LANE)
                if job_id is None:
                    break
                task = asyncio.create_task(asyncio.to_thread(run_claimed_job, job_id))
                active.add(task)
                task.add_done_callback(active.discard)
        except Exception:
            log.exception("reliability lane tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.reliability_poll_seconds)
        except TimeoutError:
            continue
    if active:
        # Threads cannot be interrupted. A restart marks their runs as stopped, and their repetitions fail as interrupted.
        await asyncio.gather(*active, return_exceptions=True)
