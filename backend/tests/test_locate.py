"""Dropped videos resolve to an on-disk path and are not copied."""

import os
from pathlib import Path

import pytest
from app.video.probe import VideoError
from fastapi.testclient import TestClient


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


def _isolate(monkeypatch, roots: list[Path]):
    from app.services import locate

    monkeypatch.setattr(locate, "spotlight_paths", lambda _name: [])
    monkeypatch.setattr(locate, "default_search_roots", lambda: roots)


def test_locate_uses_name_size_and_mtime(tmp_path, monkeypatch):
    from app.services.locate import locate_dropped_video

    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    kept = first / "clip.mp4"
    other = second / "clip.mp4"
    kept.write_bytes(b"abc")
    other.write_bytes(b"abc")
    os.utime(kept, (1_700_000_000, 1_700_000_000))
    os.utime(other, (1_800_000_000, 1_800_000_000))
    _isolate(monkeypatch, [tmp_path])

    found = locate_dropped_video("clip.mp4", 3, 1_700_000_000_000)
    assert found == kept.resolve()
    assert kept.read_bytes() == b"abc"


def test_locate_accepts_the_only_size_match(tmp_path, monkeypatch):
    from app.services.locate import locate_dropped_video

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"abc")
    os.utime(clip, (1_700_000_000, 1_700_000_000))
    _isolate(monkeypatch, [tmp_path])

    found = locate_dropped_video("clip.mp4", 3, 1_700_000_100_000)
    assert found == clip.resolve()


def test_locate_refuses_two_equal_matches(tmp_path, monkeypatch):
    from app.services.locate import locate_dropped_video

    for folder in ("a", "b"):
        path = tmp_path / folder
        path.mkdir()
        clip = path / "clip.mp4"
        clip.write_bytes(b"abc")
        os.utime(clip, (1_700_000_000, 1_700_000_000))
    _isolate(monkeypatch, [tmp_path])

    with pytest.raises(VideoError, match="Several files match"):
        locate_dropped_video("clip.mp4", 3, 1_700_000_000_000)


def test_locate_prefers_spotlight_and_does_not_walk(tmp_path, monkeypatch):
    from app.services import locate

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"abc")
    os.utime(clip, (1_700_000_000, 1_700_000_000))
    monkeypatch.setattr(locate, "spotlight_paths", lambda _name: [clip])

    def fail() -> list[Path]:
        raise AssertionError("walked")

    monkeypatch.setattr(locate, "default_search_roots", fail)
    found = locate.locate_dropped_video("clip.mp4", 3, 1_700_000_000_000)
    assert found == clip.resolve()


def test_locate_rejects_a_path_in_the_name(monkeypatch):
    from app.services import locate

    def fail(_name: str) -> list[Path]:
        raise AssertionError("searched")

    monkeypatch.setattr(locate, "spotlight_paths", fail)
    with pytest.raises(VideoError, match="file name"):
        locate.locate_dropped_video("../clip.mp4", 3, 0)


def test_locate_endpoint_returns_the_path(client, tmp_path, monkeypatch):
    clip = tmp_path / "learnloop-locate-clip.mp4"
    clip.write_bytes(b"abc")
    os.utime(clip, (1_700_000_000, 1_700_000_000))
    _isolate(monkeypatch, [tmp_path])

    response = client.post(
        "/api/videos/locate",
        json={"name": clip.name, "size": 3, "mtime_ms": 1_700_000_000_000},
    )
    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(clip.resolve())
    assert clip.read_bytes() == b"abc"

    missing = client.post("/api/videos/locate", json={"name": "missing.mp4", "size": 3, "mtime_ms": 0})
    assert missing.status_code == 400
