"""A video belongs to one project, or to the library."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError

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


def _video(project_id: str | None, name: str, checksum: str):
    from app.db import SessionLocal
    from app.models.media import VideoAsset

    session = SessionLocal()
    row = VideoAsset(
        project_id=project_id,
        original_filename=name,
        stored_path=f"imports/{checksum}.mp4",
        checksum_sha256=checksum,
    )
    session.add(row)
    session.commit()
    video_id = row.id
    session.close()
    return video_id


def test_library_upload_stays_out_of_every_project(client: TestClient, tmp_path: Path):
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    project = client.post("/api/projects", json={"name": "Bench"}).json()
    clip = tmp_path / "loose.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
    with clip.open("rb") as handle:
        uploaded = client.post("/api/videos", files={"file": ("loose.mp4", handle, "video/mp4")})
    assert uploaded.status_code == 200, uploaded.text
    video = uploaded.json()
    assert video["project_id"] is None
    library = client.get("/api/videos", params={"unassigned": True})
    assert library.status_code == 200
    assert [item["id"] for item in library.json()] == [video["id"]]
    owned = client.get("/api/videos", params={"project_id": project["id"]})
    assert owned.json() == []
    both = client.get("/api/videos", params={"unassigned": True, "project_id": project["id"]})
    assert both.status_code == 400


def test_attach_moves_a_video_and_detach_returns_it(client: TestClient):
    project = client.post("/api/projects", json={"name": "Bench"}).json()
    other = client.post("/api/projects", json={"name": "Other"}).json()
    video_id = _video(None, "loose.mp4", "checksum-loose")
    moved = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": [video_id]})
    assert moved.status_code == 200, moved.text
    assert moved.json()[0]["project_id"] == project["id"]
    assert client.get("/api/videos", params={"unassigned": True}).json() == []
    again = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": [video_id]})
    assert again.status_code == 200
    conflict = client.post(f"/api/projects/{other['id']}/videos/attach", json={"video_ids": [video_id]})
    assert conflict.status_code == 409
    returned = client.post(f"/api/videos/{video_id}/detach")
    assert returned.status_code == 200, returned.text
    assert returned.json()["project_id"] is None
    assert returned.json()["session_id"] is None
    library = client.get("/api/videos", params={"unassigned": True}).json()
    assert [item["id"] for item in library] == [video_id]
    assert client.get("/api/videos", params={"project_id": project["id"]}).json() == []


def test_attach_rejects_a_file_the_project_already_has(client: TestClient):
    project = client.post("/api/projects", json={"name": "Bench"}).json()
    _video(project["id"], "kept.mp4", "same-bytes")
    library_id = _video(None, "copy.mp4", "same-bytes")
    attached = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": [library_id]})
    assert attached.status_code == 409
    assert client.get(f"/api/videos/{library_id}").json()["project_id"] is None


def test_attach_to_a_multi_day_project_joins_a_session(client: TestClient):
    created = client.post("/api/projects", json={"name": "Firing", "is_task": True})
    assert created.status_code == 200, created.text
    project = created.json()
    assert project["is_task"] is True
    video_id = _video(None, "day-two.mp4", "day-two")
    attached = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": [video_id]})
    assert attached.status_code == 200, attached.text
    clip = attached.json()[0]
    assert clip["project_id"] == project["id"]
    assert clip["session_id"]
    task = client.get(f"/api/tasks/{project['id']}")
    assert task.status_code == 200
    assert video_id in {item["id"] for item in task.json()["clips"]}


def test_library_run_has_no_project(client: TestClient):
    from app.pipeline.infer import register_fixture

    video_id = _video(None, "loose.mp4", "library-run")
    register_fixture("library-run", "fixed_camera")
    created = client.post(
        "/api/runs",
        json={
            "name": "Library",
            "video_id": video_id,
            "provider_id": "fixture",
            "model_id": "authored-fixture",
            "research_mode": "fixed_camera",
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["project_id"] is None


def test_fixture_run_on_an_unregistered_video_is_rejected_at_launch(client: TestClient):
    video_id = _video(None, "real.mp4", "not-a-fixture")
    created = client.post(
        "/api/runs",
        json={
            "video_id": video_id,
            "provider_id": "fixture",
            "model_id": "authored-fixture",
            "research_mode": "fixed_camera",
        },
    )
    assert created.status_code == 400
    assert "no authored analysis" in created.json()["detail"]


def test_missing_video_and_empty_selection_are_rejected(client: TestClient):
    project = client.post("/api/projects", json={"name": "Bench"}).json()
    missing = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": ["absent"]})
    assert missing.status_code == 404
    empty = client.post(f"/api/projects/{project['id']}/videos/attach", json={"video_ids": []})
    assert empty.status_code == 400
    unknown = client.post("/api/videos/absent/detach")
    assert unknown.status_code == 404


def test_older_sqlite_database_accepts_a_video_with_no_project(tmp_path: Path):
    from app.services.schema_upgrade import allow_missing_project

    engine = create_engine(f"sqlite:///{tmp_path}/old.db")

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE projects (id VARCHAR(36) PRIMARY KEY)"))
        conn.execute(
            text(
                "CREATE TABLE video_assets ("
                "id VARCHAR(36) PRIMARY KEY, "
                "project_id VARCHAR(36) NOT NULL, "
                "name VARCHAR(40) NOT NULL DEFAULT '', "
                "FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE)"
            )
        )
        conn.execute(text("CREATE INDEX ix_video_assets_project_id ON video_assets (project_id)"))
        conn.execute(text("INSERT INTO projects (id) VALUES ('p1')"))
        conn.execute(text("INSERT INTO video_assets (id, project_id, name) VALUES ('v1', 'p1', 'kept')"))

    allow_missing_project(engine, "video_assets")
    allow_missing_project(engine, "video_assets")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO video_assets (id, project_id, name) VALUES ('v2', NULL, 'loose')"))
        with pytest.raises(IntegrityError):
            conn.execute(text("INSERT INTO video_assets (id, project_id, name) VALUES ('v3', 'missing', 'bad')"))
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM projects WHERE id = 'p1'"))
        remaining = [row[0] for row in conn.execute(text("SELECT id FROM video_assets ORDER BY id"))]
    assert remaining == ["v2"]
