"""End-to-end local pipeline: synthetic video, fixture provider, corrections, export."""

import asyncio
import os
import time
from pathlib import Path

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


def test_fixture_pipeline_objects_actions_and_correction(client, tmp_path: Path):
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario
    from app.pipeline.infer import register_fixture

    project = client.post("/api/projects", json={"name": "Bench"}).json()
    scenario = SCENARIOS["fixed_camera"]()
    clip = tmp_path / "fixed.mp4"
    write_clip(render_scenario(scenario), clip)
    with clip.open("rb") as handle:
        uploaded = client.post(
            "/api/videos",
            data={"project_id": project["id"]},
            files={"file": ("fixed.mp4", handle, "video/mp4")},
        )
    assert uploaded.status_code == 200, uploaded.text
    video = uploaded.json()
    assert video["duration_sec"] > 1
    detail = client.get(f"/api/videos/{video['id']}").json()
    assert detail["frames"]
    register_fixture(video["checksum"], "fixed_camera")
    created = client.post(
        "/api/runs",
        json={
            "name": "Fixture fixed",
            "video_id": video["id"],
            "provider_id": "fixture",
            "model_id": "authored-fixture",
            "modality": "sampled_frames",
            "research_mode": "fixed_camera",
            "sampling_preset": "detailed",
            "analysis_pass": "single",
            "segmentation_mode": "boxes_only",
            "max_frames": 20,
        },
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["id"]
    _run_job(run_id)
    bundle = client.get(f"/api/runs/{run_id}").json()
    assert bundle["run"]["status"] == "completed"
    assert bundle["run"]["fixture"] is True
    assert len(bundle["objects"]) >= 4
    assert len(bundle["actions"]) >= 4
    assert any(item["outcome"] == "attempted" for item in bundle["actions"])
    assert bundle["coverage"]["modality"] == "sampled_frames"
    assert bundle["coverage"]["frames_extracted"] > 0
    assert any(image["url"] for obj in bundle["objects"] for image in obj["images"])
    renamed = client.patch(
        f"/api/runs/{run_id}/objects/Object_1",
        json={"label": "Researcher label"},
    )
    assert renamed.status_code == 200, renamed.text
    body = renamed.json()
    assert body["revision"]["layer"] == "human"
    labels = {obj["local_id"]: obj["label"] for obj in body["objects"]}
    assert labels["Object_1"] == "Researcher label"
    exported = client.get(f"/api/runs/{run_id}/export")
    assert exported.status_code == 200
    assert exported.content[:2] == b"PK"


def test_native_video_rejected_for_openai(client):
    project = client.post("/api/projects", json={"name": "Mode"}).json()
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    scenario = SCENARIOS["fixed_camera"]()
    path = Path(os.environ["LEARNLOOP_DATA_DIR"]) / "clip.mp4"
    write_clip(render_scenario(scenario)[:5], path)
    with path.open("rb") as handle:
        video = client.post(
            "/api/videos",
            data={"project_id": project["id"]},
            files={"file": ("clip.mp4", handle, "video/mp4")},
        ).json()
    refused = client.post(
        "/api/runs",
        json={
            "name": "should fail",
            "video_id": video["id"],
            "provider_id": "openai",
            "model_id": "gpt-4.1",
            "modality": "native_video",
            "research_mode": "fixed_camera",
        },
    )
    assert refused.status_code == 400
    assert "native video" in refused.json()["detail"].lower()


def test_video_model_defaults_to_the_file_and_honors_frames(client):
    from app.db import SessionLocal
    from app.models.media import Project, VideoAsset
    from app.services.experiments import create_run, sends_video_file

    session = SessionLocal()
    project = Project(name="video-mode")
    session.add(project)
    session.flush()
    video = VideoAsset(
        project_id=project.id,
        original_filename="clip.mp4",
        stored_path="originals/clip.mp4",
        checksum_sha256="videomode",
        duration_sec=8,
    )
    session.add(video)
    session.commit()
    def start(provider_id, model_id, **extra):
        return create_run(
            session,
            {
                "video_id": video.id,
                "provider_id": provider_id,
                "model_id": model_id,
                "research_mode": "fixed_camera",
                **extra,
            },
            enqueue=False,
        )

    default = start("gemini", "gemini-2.5-pro")
    assert default.modality == "native_video"
    assert default.config_json["modality"] == "native_video"
    assert sends_video_file(default)
    chosen = start("gemini", "gemini-2.5-pro", modality="sampled_frames")
    assert chosen.modality == "sampled_frames"
    assert chosen.config_json["modality"] == "sampled_frames"
    assert not sends_video_file(chosen)
    assert start("openai", "gpt-4.1").modality == "sampled_frames"
    assert start("openai", "gpt-4.1", modality="sampled_frames").modality == "sampled_frames"
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


def test_project_video_order_and_delete(client):
    from datetime import UTC, datetime, timedelta

    from app.db import SessionLocal
    from app.models.media import VideoAsset

    project = client.post("/api/projects", json={"name": "Order"}).json()
    task = client.post("/api/tasks", json={"name": "Task order", "description": ""}).json()
    session = SessionLocal()
    now = datetime.now(UTC)
    older = VideoAsset(
        project_id=project["id"],
        original_filename="older.mp4",
        stored_path="originals/older.mp4",
        checksum_sha256="a" * 64,
        created_at=now,
    )
    newer = VideoAsset(
        project_id=project["id"],
        original_filename="newer.mp4",
        stored_path="originals/newer.mp4",
        checksum_sha256="b" * 64,
        created_at=now + timedelta(seconds=5),
    )
    session.add_all([older, newer])
    session.commit()
    older_id, newer_id = older.id, newer.id
    session.close()

    listed = client.get("/api/videos", params={"project_id": project["id"]}).json()
    assert [item["filename"] for item in listed] == ["newer.mp4", "older.mp4"]

    ordered = client.post(
        f"/api/projects/{project['id']}/videos/order",
        json={"video_ids": [older_id, newer_id]},
    )
    assert ordered.status_code == 200, ordered.text
    assert [item["filename"] for item in ordered.json()] == ["older.mp4", "newer.mp4"]
    again = client.get("/api/videos", params={"project_id": project["id"]}).json()
    assert [item["id"] for item in again] == [older_id, newer_id]
    assert [item["order_index"] for item in again] == [0, 1]

    foreign = client.post(
        f"/api/projects/{project['id']}/videos/order",
        json={"video_ids": [older_id, "missing"]},
    )
    assert foreign.status_code == 400
    rejected = client.post(f"/api/projects/{task['project']['id']}/videos/order", json={"video_ids": []})
    assert rejected.status_code == 400

    removed = client.delete(f"/api/videos/{newer_id}")
    assert removed.status_code == 200, removed.text
    remaining = client.get("/api/videos", params={"project_id": project["id"]}).json()
    assert [item["id"] for item in remaining] == [older_id]


def test_upload_analyzes_after_the_response(client, tmp_path: Path):
    from app.config import settings
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    settings.ingest_inline = False
    try:
        project = client.post("/api/projects", json={"name": "Background"}).json()
        clip = tmp_path / "background.mp4"
        write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
        with clip.open("rb") as handle:
            uploaded = client.post(
                "/api/videos",
                data={"project_id": project["id"]},
                files={"file": ("background.mp4", handle, "video/mp4")},
            )
        assert uploaded.status_code == 200, uploaded.text
        video = uploaded.json()
        assert video["ingest_status"] == "processing"
        assert video["ingest_progress"] < 100
        detail = video
        deadline = time.time() + 60
        while time.time() < deadline and detail["ingest_status"] == "processing":
            time.sleep(0.05)
            detail = client.get(f"/api/videos/{video['id']}").json()
        assert detail["ingest_status"] == "ready", detail.get("ingest_error")
        assert detail["duration_sec"] > 1
        assert detail["frames"]
        assert detail["thumbnail_url"]
        assert detail["contact_sheet_url"]
        assert detail["ingest_progress"] == 100
    finally:
        settings.ingest_inline = True


def test_reference_keeps_the_original_file(client, tmp_path: Path):
    from app.config import settings
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    project = client.post("/api/projects", json={"name": "Referenced"}).json()
    clip = tmp_path / "outside.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
    before = clip.stat().st_ino

    missing = client.post("/api/videos/reference", json={"project_id": project["id"], "path": "clip.mp4"})
    assert missing.status_code == 400

    saved = client.put("/api/settings", json={"video_storage": "reference"})
    assert saved.status_code == 200
    assert client.get("/api/settings").json()["video_storage"] == "reference"

    linked = client.post("/api/videos/reference", json={"project_id": project["id"], "path": str(clip)})
    assert linked.status_code == 200, linked.text
    video = linked.json()
    assert video["ingest_status"] == "ready", video.get("ingest_error")
    assert video["thumbnail_url"]
    playback = client.get(video["playback_url"])
    assert playback.status_code == 200
    assert playback.content == clip.read_bytes()
    assert clip.stat().st_ino == before
    copied = [path for path in settings.resolved_data_dir.joinpath("originals").glob("*.mp4")]
    assert copied == []

    removed = client.delete(f"/api/videos/{video['id']}")
    assert removed.status_code == 200
    assert clip.is_file()
    assert clip.stat().st_ino == before


def test_reference_shows_a_still_while_it_processes(client, tmp_path: Path):
    from app.config import settings
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    settings.ingest_inline = False
    try:
        project = client.post("/api/projects", json={"name": "Referenced preview"}).json()
        clip = tmp_path / "preview.mp4"
        write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
        linked = client.post("/api/videos/reference", json={"project_id": project["id"], "path": str(clip)})
        assert linked.status_code == 200, linked.text
        video = linked.json()
        assert video["ingest_status"] == "processing"
        assert video["thumbnail_url"]
        still = client.get(video["thumbnail_url"])
        assert still.status_code == 200
        assert still.content[:3] == b"\xff\xd8\xff"
        detail = video
        deadline = time.time() + 60
        seen = {detail.get("ingest_detail")}
        while time.time() < deadline and detail["ingest_status"] == "processing":
            time.sleep(0.05)
            detail = client.get(f"/api/videos/{video['id']}").json()
            seen.add(detail.get("ingest_detail"))
        assert detail["ingest_status"] == "ready", detail.get("ingest_error")
        assert any(item and item not in {"Hashing", "Queued"} for item in seen)
        assert clip.is_file()
    finally:
        settings.ingest_inline = True


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


def test_cancel_returns_a_duration_when_the_start_has_no_timezone(client):
    from datetime import datetime

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models.media import VideoAsset
    from app.models.runs import AnalysisRun, Job, PipelineStageRun

    session = SessionLocal()
    video = VideoAsset(original_filename="clip.mp4", stored_path="external", checksum_sha256="b" * 64)
    session.add(video)
    session.flush()
    run = AnalysisRun(
        video_id=video.id,
        name="In flight",
        status="running",
        research_mode="fixed_camera",
        viewpoint="fixed",
        task_constraint="constrained",
        provider_id="meta",
        model_id="muse-wonder-1.0",
        modality="native_video",
        sampling_preset="standard",
        config_json={},
        started_at=datetime(2026, 10, 31, 4, 52, 40),
    )
    session.add(run)
    session.flush()
    session.add(Job(run_id=run.id, status="running"))
    session.add(PipelineStageRun(run_id=run.id, stage="scene_analysis", status="running", checkpoint_json={}))
    session.commit()
    run_id = run.id
    session.close()

    assert client.get(f"/api/runs/{run_id}").json()["run"]["starred"] is False
    assert client.patch(f"/api/runs/{run_id}/starred", json={"starred": True}).json()["starred"] is True
    assert client.get("/api/runs").json()[0]["starred"] is True
    assert client.patch(f"/api/runs/{run_id}/starred", json={"starred": False}).json()["starred"] is False
    assert client.patch(f"/api/runs/{run_id}/starred", json={"starred": "yes"}).status_code == 422

    canceled = client.post(f"/api/runs/{run_id}/cancel")
    assert canceled.status_code == 200, canceled.text
    body = canceled.json()
    assert body["status"] == "canceled"
    assert body["duration_sec"] > 0

    session = SessionLocal()
    stage = session.scalar(select(PipelineStageRun).where(PipelineStageRun.run_id == run_id))
    assert stage.status == "canceled"
    session.close()


def test_deleting_a_run_while_it_executes_stops_the_job_without_crashing(client, tmp_path: Path, monkeypatch):
    from app.db import SessionLocal
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario
    from app.models.runs import AnalysisRun, Job
    from app.pipeline import runner
    from app.pipeline.infer import register_fixture
    from sqlalchemy import select

    project = client.post("/api/projects", json={"name": "Bench"}).json()
    clip = tmp_path / "fixed.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
    with clip.open("rb") as handle:
        video = client.post(
            "/api/videos", data={"project_id": project["id"]}, files={"file": ("fixed.mp4", handle, "video/mp4")}
        ).json()
    register_fixture(video["checksum"], "fixed_camera")
    run_id = client.post(
        "/api/runs",
        json={"video_id": video["id"], "provider_id": "fixture", "model_id": "authored-fixture"},
    ).json()["id"]

    async def delete_then_touch_the_run(session, run, _video, _stage):
        other = SessionLocal()
        other.delete(other.get(AnalysisRun, run_id))
        other.commit()
        other.close()
        run.error = "still working"

    monkeypatch.setattr(runner, "_dispatch", delete_then_touch_the_run)
    session = SessionLocal()
    job = session.scalar(select(Job).where(Job.run_id == run_id))
    asyncio.run(runner.execute_job(session, job))
    assert session.get(AnalysisRun, run_id) is None
    session.close()
