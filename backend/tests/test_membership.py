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
