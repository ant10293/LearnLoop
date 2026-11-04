"""Pipeline jobs, worker behavior, and file cleanup."""

import asyncio
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    monkeypatch.setenv("LEARNLOOP_WORKER_ENABLED", "0")
    monkeypatch.setenv("LEARNLOOP_AUTO_SEED", "0")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    settings.worker_enabled = False
    settings.auto_seed = False
    settings.ingest_inline = True
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database

    init_database()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _run_job(run_id: str) -> None:
    from app.db import SessionLocal
    from app.models.runs import Job
    from app.pipeline.runner import execute_job
    from sqlalchemy import select

    session = SessionLocal()
    job = session.scalar(select(Job).where(Job.run_id == run_id, Job.status == "pending"))
    assert job is not None
    job.status = "running"
    session.commit()
    asyncio.run(execute_job(session, job))
    session.close()


def test_job_restart_requeues_running_work(client):
    from app.db import SessionLocal
    from app.jobs.worker import recover_incomplete
    from app.models.runs import AnalysisRun, Job, PipelineStageRun
    from app.models.util import new_id

    session = SessionLocal()
    from app.models.media import Project, VideoAsset

    project = Project(name="recovery")
    session.add(project)
    session.flush()
    video = VideoAsset(
        project_id=project.id,
        original_filename="x.mp4",
        stored_path="originals/x.mp4",
        checksum_sha256="abc",
        duration_sec=1,
        width=64,
        height=64,
        display_width=64,
        display_height=64,
    )
    session.add(video)
    session.flush()
    run = AnalysisRun(
        project_id=project.id,
        video_id=video.id,
        name="stuck",
        status="running",
        research_mode="fixed_camera",
        viewpoint="fixed",
        task_constraint="constrained",
        provider_id="fixture",
        model_id="authored-fixture",
        modality="sampled_frames",
        sampling_preset="standard",
        analysis_pass="single",
        config_json={},
    )
    session.add(run)
    session.flush()
    stage = PipelineStageRun(run_id=run.id, stage="scene_analysis", status="running")
    job = Job(id=new_id(), run_id=run.id, status="running", attempt=1, max_attempts=3, worker_id="dead")
    session.add(stage)
    session.add(job)
    session.commit()
    recovered = recover_incomplete()
    assert recovered == 1
    session.expire_all()
    assert session.get(Job, job.id).status == "pending"
    assert session.get(PipelineStageRun, stage.id).status == "pending"
    session.close()


def test_deleting_a_video_keeps_files_another_video_shares(client):
    from app.db import SessionLocal
    from app.models.media import Project, VideoAsset
    from app.services.storage import data_path

    session = SessionLocal()
    project = Project(name="Shared")
    session.add(project)
    session.commit()
    project = {"id": project.id}
    session.close()
    original = data_path("originals", "shared.mov")
    proxy = data_path("proxies", "shared.mp4")
    for path in (original, proxy):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    session = SessionLocal()
    first, second = (
        VideoAsset(project_id=project["id"], original_filename="IMG.mov", stored_path="originals/shared.mov", proxy_path="proxies/shared.mp4", checksum_sha256="c" * 64)
        for _ in range(2)
    )
    session.add_all([first, second])
    session.commit()
    first_id, second_id = first.id, second.id
    session.close()

    assert client.delete(f"/api/videos/{first_id}").status_code == 200
    assert original.exists() and proxy.exists()
    assert client.get(f"/api/media/videos/{second_id}/playback").status_code == 200
    assert client.delete(f"/api/videos/{second_id}").status_code == 200
    assert not original.exists() and not proxy.exists()


def test_worker_keeps_the_server_loop_free_during_a_job(monkeypatch):
    from app.jobs import worker

    def blocking(_job_id: str) -> None:
        time.sleep(0.3)

    claims = {"n": 0}

    def claim(_worker_id: str):
        claims["n"] += 1
        if claims["n"] == 1:
            return "job-1"
        return None

    monkeypatch.setattr(worker, "recover_incomplete", lambda: 0)
    monkeypatch.setattr(worker, "claim_next", claim)
    monkeypatch.setattr(worker, "run_claimed_job", blocking)
    monkeypatch.setattr(worker.settings, "worker_poll_seconds", 0.05)

    async def main():
        stop = asyncio.Event()
        ticks = 0

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.05)
                ticks += 1

        loop_task = asyncio.create_task(worker.worker_loop(stop))
        tick_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.35)
        stop.set()
        await loop_task
        tick_task.cancel()
        return ticks

    ticks = asyncio.run(main())
    assert ticks >= 3


