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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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


def test_adding_a_model_twice_keeps_one_entry(client: TestClient):
    for model_id in ("claude-test-9", " claude-test-9 "):
        added = client.post("/api/providers/anthropic/models", json={"model_id": model_id})
        assert added.status_code == 200
        assert added.json()["model_id"] == "claude-test-9"

    from app.db import SessionLocal
    from app.models import ManualModel

    with SessionLocal() as session:
        assert session.query(ManualModel).filter_by(provider_id="anthropic").count() == 1

    discovered = client.get("/api/providers/anthropic/models").json()["catalog"]
    assert _ids(discovered).count("claude-test-9") == 1
    anthropic = next(item for item in client.get("/api/providers").json() if item["id"] == "anthropic")
    assert _ids(anthropic["models"]).count("claude-test-9") == 1


def test_blank_model_id_is_rejected(client: TestClient):
    assert client.post("/api/providers/anthropic/models", json={"model_id": "   "}).status_code == 400


def test_merge_lists_each_model_id_once():
    from app.api.routes import with_manual_models

    listed = [{"model_id": "a"}, {"model_id": "b"}, {"model_id": "a"}]
    merged = with_manual_models("anthropic", listed, ["b", "c", "c"])
    assert _ids(merged) == ["a", "b", "c"]
    assert merged[0] is listed[0]


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
                "('later', 'anthropic', 'claude-test-9', '', '2026-10-02'), "
                "('first', 'anthropic', 'claude-test-9', '', '2026-10-01'), "
                "('other', 'openai', 'claude-test-9', '', '2026-10-03')"
            )
        )

    unique_manual_models(engine)
    unique_manual_models(engine)

    with engine.begin() as conn:
        kept = conn.execute(text("SELECT id FROM manual_models ORDER BY id")).scalars().all()
    assert kept == ["first", "other"]
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text("INSERT INTO manual_models VALUES ('again', 'anthropic', 'claude-test-9', '', '2026-10-04')")
        )
