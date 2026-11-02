"""Provider call costs roll up to the project and the dashboard."""

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


def test_each_provider_call_is_priced_on_the_project_and_dashboard(client: TestClient):
    from app.db import SessionLocal
    from app.models.media import Project, VideoAsset
    from app.models.runs import AnalysisRun
    from app.providers.catalog import lookup
    from app.providers.types import ProviderResponse, Usage, estimate_cost
    from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call

    session = SessionLocal()
    project = Project(name="Spend", description="")
    session.add(project)
    session.flush()
    video = VideoAsset(
        project_id=project.id,
        original_filename="clip.mp4",
        stored_path="external",
        checksum_sha256="a" * 64,
    )
    session.add(video)
    session.flush()
    run = AnalysisRun(
        project_id=project.id,
        video_id=video.id,
        name="Paid pass",
        status="completed",
        research_mode="fixed_camera",
        viewpoint="fixed",
        task_constraint="constrained",
        provider_id="openai",
        model_id="gpt-4.1",
        modality="sampled_frames",
        sampling_preset="standard",
        config_json={},
    )
    session.add(run)
    session.flush()

    priced, known = estimate_cost(lookup("openai", "gpt-4.1"), 1000, 500)
    scene = open_provider_call(
        session,
        run,
        stage="scene_analysis",
        provider_id="openai",
        model_id="gpt-4.1",
        modality="sampled_frames",
    )
    finish_provider_call(
        session,
        run,
        scene,
        ProviderResponse(
            text="{}",
            raw={},
            usage=Usage(1000, 500, priced, known),
            request_id="resp_priced",
            latency_ms=90,
            model_id="gpt-4.1",
        ),
    )
    unpriced_cost, unpriced_known = estimate_cost(lookup("meta", "muse-spark-1.3"), 400, 40)
    audio = open_provider_call(
        session,
        run,
        stage="audio_context",
        provider_id="meta",
        model_id="muse-spark-1.3",
        modality="sampled_frames",
    )
    finish_provider_call(
        session,
        run,
        audio,
        ProviderResponse(
            text="{}",
            raw={},
            usage=Usage(400, 40, unpriced_cost, unpriced_known),
            latency_ms=30,
            model_id="muse-spark-1.3",
        ),
    )
    failed = open_provider_call(
        session,
        run,
        stage="segmentation",
        provider_id="meta",
        model_id="sam-3.1",
        modality="sampled_frames",
    )
    fail_provider_call(failed, RuntimeError("timeout"))
    plan = open_provider_call(
        session,
        run,
        stage="workflow",
        provider_id="openai",
        model_id="gpt-4.1",
        modality="sampled_frames",
        parameters={"openai_transport": "codex"},
    )
    finish_provider_call(
        session,
        run,
        plan,
        ProviderResponse(
            text="{}",
            raw={},
            usage=Usage(10, 5, 0.0, False),
            latency_ms=20,
            model_id="gpt-4.1",
        ),
    )
    session.commit()
    project_id = project.id
    session.close()

    missing = client.get("/api/projects/missing/costs")
    assert missing.status_code == 404

    detail = client.get(f"/api/projects/{project_id}/costs")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["cost_usd"] == priced
    assert body["input_tokens"] == 1410
    assert body["output_tokens"] == 545
    assert body["unpriced_calls"] == 1
    assert len(body["calls"]) == 4
    by_stage = {item["stage"]: item for item in body["calls"]}
    assert by_stage["scene_analysis"]["price_known"] is True
    assert by_stage["scene_analysis"]["api_key"] is True
    assert by_stage["scene_analysis"]["cost_usd"] == priced
    assert by_stage["scene_analysis"]["video_filename"] == "clip.mp4"
    assert by_stage["scene_analysis"]["model_id"] == "gpt-4.1"
    assert by_stage["audio_context"]["price_known"] is False
    assert by_stage["audio_context"]["api_key"] is True
    assert by_stage["audio_context"]["cost_usd"] == 0
    assert by_stage["workflow"]["api_key"] is False
    assert by_stage["workflow"]["price_known"] is False
    assert by_stage["segmentation"]["status"] == "failed"
    assert by_stage["segmentation"]["cost_usd"] == 0

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["cost_usd"] == priced
    assert dashboard["unpriced_calls"] == 1
    providers = {item["provider_id"]: item for item in dashboard["cost_by_provider"]}
    assert providers["openai"]["cost_usd"] == priced
    assert providers["openai"]["calls"] == 2
    assert providers["openai"]["unpriced_calls"] == 1
    assert providers["meta"]["cost_usd"] == 0
    assert providers["meta"]["unpriced_calls"] == 0
    assert providers["meta"]["calls"] == 1
