"""Claim pending jobs and recover work after a process restart."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from sqlalchemy import select, update

from app.config import settings
from app.models.runs import AnalysisRun, Job, PipelineStageRun
from app.models.util import utcnow
from app.pipeline.runner import execute_job

log = logging.getLogger("learnloop.worker")


def _session():
    from app.db import SessionLocal

    return SessionLocal()


def recover_incomplete() -> int:
    """Runs that were marked running died with the previous process. Requeue them."""
    session = _session()
    recovered = 0
    try:
        jobs = session.scalars(select(Job).where(Job.status == "running")).all()
        for job in jobs:
            run = session.get(AnalysisRun, job.run_id)
            for stage in session.scalars(select(PipelineStageRun).where(PipelineStageRun.run_id == job.run_id)):
                if stage.status == "running":
                    stage.status = "pending"
                    stage.error = "reset after process restart"
            if job.attempt >= job.max_attempts:
                job.status = "failed"
                job.error = "worker stopped before the attempt finished"
                job.finished_at = utcnow()
                if run and run.status == "running":
                    run.status = "failed"
                    run.error = job.error
            else:
                job.status = "pending"
                job.worker_id = ""
                job.error = "requeued after process restart"
                if run and run.status == "running":
                    run.status = "pending"
            recovered += 1
        session.commit()
    finally:
        session.close()
    return recovered


def claim_next(worker_id: str) -> str | None:
    session = _session()
    try:
        pending = session.scalar(select(Job).where(Job.status == "pending").order_by(Job.created_at))
        if pending is None:
            return None
        result = session.execute(
            update(Job)
            .where(Job.id == pending.id, Job.status == "pending")
            .values(
                status="running",
                worker_id=worker_id,
                started_at=utcnow(),
                heartbeat_at=utcnow(),
                attempt=Job.attempt + 1,
            )
        )
        session.commit()
        if result.rowcount != 1:
            return None
        return pending.id
    finally:
        session.close()


def run_claimed_job(job_id: str) -> None:
    """Run one job on this thread, with its own event loop.

    The worker lives in the API process. Tracking and frame work are synchronous
    and can occupy a core for minutes. Running that on the server loop stops the
    process from answering until the stage returns.
    """
    session = _session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return
        asyncio.run(execute_job(session, job))
    except Exception:
        log.exception("job %s crashed", job_id)
    finally:
        session.close()


async def worker_loop(stop: asyncio.Event) -> None:
    worker_id = uuid4().hex[:12]
    recovered = await asyncio.to_thread(recover_incomplete)
    if recovered:
        log.info("recovered %s interrupted jobs", recovered)
    while not stop.is_set():
        job_id = await asyncio.to_thread(claim_next, worker_id)
        if job_id is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
            except TimeoutError:
                continue
            continue
        await asyncio.to_thread(run_claimed_job, job_id)
