"""Researcher-added model ids are stored and listed once per provider."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
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
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database

    init_database()
    with TestClient(create_app()) as test_client:
        yield test_client


def _ids(models: list[dict]) -> list[str]:
    return [model["model_id"] for model in models]


def test_upgrade_removes_repeated_rows_and_enforces_uniqueness(tmp_path):
    from app.services.schema_upgrade import unique_manual_models

    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE manual_models (id VARCHAR(36) PRIMARY KEY, provider_id VARCHAR(40), "
                "model_id VARCHAR(200), notes TEXT, created_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO manual_models VALUES "
                "('later', 'meta', 'muse-test-9', '', '2026-10-02'), "
                "('first', 'meta', 'muse-test-9', '', '2026-10-01'), "
                "('other', 'fixture', 'muse-test-9', '', '2026-10-03')"
            )
        )

    unique_manual_models(engine)
    unique_manual_models(engine)

    with engine.begin() as conn:
        kept = conn.execute(text("SELECT id FROM manual_models ORDER BY id")).scalars().all()
    assert kept == ["first", "other"]
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text("INSERT INTO manual_models VALUES ('again', 'meta', 'muse-test-9', '', '2026-10-04')")
        )
