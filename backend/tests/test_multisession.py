"""Multi-session ordering, identity, and incremental reconciliation."""

import os

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


def test_capture_time_keeps_timezone_and_ignores_mtime():
    from app.video.capture_time import stamp_from_tags

    stamp = stamp_from_tags(
        {"com.apple.quicktime.creationdate": "2026-03-01T10:00:00-05:00"},
        "2026-10-21T00:00:00+00:00",
    )
    assert stamp.reliability == "reliable"
    assert stamp.source == "quicktime"
    assert stamp.utc_iso.startswith("2026-03-01T15:00:00")
    assert stamp.raw.startswith("2026-03-01T10:00:00-05:00")
    assert stamp.file_mtime == "2026-10-21T00:00:00+00:00"
    assert stamp.utc_iso != stamp.file_mtime


def test_naive_and_conflicting_timestamps_are_not_ordered_as_utc():
    from app.video.capture_time import stamp_from_tags

    naive = stamp_from_tags({"creation_time": "2026-03-01T10:00:00"}, "mtime")
    assert naive.reliability == "weak"
    assert naive.utc_iso == ""
    conflict = stamp_from_tags(
        {
            "com.apple.quicktime.creationdate": "2026-03-01T10:00:00Z",
            "creation_time": "2026-03-01T12:30:00Z",
        },
        "mtime",
    )
    assert conflict.reliability == "conflicting"
    assert conflict.utc_iso == ""


def _cache_object(video_id: str, initial: str, final: str) -> None:
    from app.db import SessionLocal
    from app.models.domain import DetectedObject
    from app.models.media import VideoAsset
    from app.models.runs import AnalysisRevision, AnalysisRun

    session = SessionLocal()
    video = session.get(VideoAsset, video_id)
    assert video is not None
    run = AnalysisRun(
        project_id=video.project_id,
        video_id=video.id,
        name="Cached clip reading",
        status="completed",
        research_mode=video.research_mode or "fixed_camera",
        viewpoint="fixed",
        task_constraint="unknown",
        provider_id="fixture",
        model_id="authored-fixture",
        modality="sampled_frames",
        sampling_preset="standard",
        config_json={},
        fixture=True,
    )
    session.add(run)
    session.flush()
    revision = AnalysisRevision(run_id=run.id, layer="validated", version=1, document_json={"cached": True})
    session.add(revision)
    session.flush()
    run.active_revision_id = revision.id
    session.add(
        DetectedObject(
            revision_id=revision.id,
            run_id=run.id,
            local_id="Object_1",
            label="ceramic figure",
            visual_characteristics="small ceramic figure",
            initial_state=initial,
            final_state=final,
        )
    )
    session.commit()
    session.close()
