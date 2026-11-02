"""Optional local transcription. Speech never changes visual action outcomes."""

import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")

from app.audio.align import align_segments
from app.audio.service import CONTEXT_PROMPT, context_request


def _event(local_id, text, start, end, kind="action", video_id="clip-a", occluded=False):
    return {
        "video_id": video_id,
        "kind": kind,
        "local_id": local_id,
        "text": text,
        "start": start,
        "end": end,
        "occluded": occluded,
    }


def _speech(local_id, text, start, end):
    return {"local_id": local_id, "text": text, "start": start, "end": end}


def _one(segments, events, window=4):
    rows = align_segments(segments, events, video_id="clip-a", window_sec=window)
    assert len(rows) == 1
    return rows[0]


def test_supported_contextual_irrelevant_and_contradictory():
    events = [
        _event("Action_1", "insert screw into the base", 2, 4),
        _event("Action_2", "insert screw into the base", 20, 21),
        _event("Action_3", "insert screw into the base", 30, 31),
        _event("Object_1", "screw", 0, 100, kind="object"),
    ]
    supported = _one([_speech("S1", "I'm putting the screw into the base.", 2.2, 3.4)], events)
    assert supported["classification"] == "visually_supported"
    assert supported["action_local_ids"] == ["Action_1"]
    assert supported["speech_is_not_completion"] is True
    assert supported["evidence_class"] == "INFERRED"

    contextual = _one(
        [_speech("S2", "I'm using the shorter screw because the longer one would go through the material.", 2, 4)],
        events,
    )
    assert contextual["classification"] == "contextual"
    assert "not visual proof" in contextual["explanation"]

    irrelevant = _one([_speech("S3", "I need to stop at the grocery store later.", 2, 4)], events)
    assert irrelevant["classification"] == "irrelevant"

    contradictory = _one([_speech("S4", "All four screws are installed.", 8, 9)], events)
    assert contradictory["classification"] == "contradictory"
    assert "camera may have missed" in contradictory["explanation"]
    assert contradictory["speech_is_not_completion"] is True


def test_timing_occlusion_empty_and_other_clip():
    action = _event("Action_1", "insert screw into the base", 10, 12)
    before = _one([_speech("S1", "I'm putting the screw into the base.", 7, 8)], [action])
    assert before["classification"] == "visually_supported"
    assert "precedes" in before["explanation"]
    after = _one([_speech("S2", "I'm putting the screw into the base.", 14, 15)], [action])
    assert after["classification"] == "visually_supported"
    assert "follows" in after["explanation"]
    distant = _one([_speech("S3", "I'm putting the screw into the base.", 40, 41)], [action])
    assert distant["classification"] == "uncertain"
    assert "nearest frame" in distant["explanation"]
    hidden = _event("Action_2", "insert screw into the base", 10, 12, occluded=True)
    uncertain = _one([_speech("S4", "I'm putting the screw into the base.", 10, 12)], [hidden])
    assert uncertain["classification"] == "uncertain"
    assert align_segments([], [action], video_id="clip-a") == []
    other = _event("Action_9", "insert screw into the base", 10, 12, video_id="clip-b")
    crossed = _one([_speech("S5", "I'm putting the screw into the base.", 10, 12)], [other])
    assert crossed["classification"] == "irrelevant"
    assert crossed["action_local_ids"] == []


def test_prompt_treats_transcript_as_untrusted_data():
    banned = ["bolt", "washer", "nut", "hexagonal"]
    lowered = CONTEXT_PROMPT.lower()
    assert not any(word in lowered for word in banned)
    assert "untrusted" in lowered
    assert "not proof" in lowered


def test_context_request_stays_local_for_fixture_and_names_text_for_external():
    class Run:
        provider_id = "fixture"
        video_id = "clip-a"
        config_json = {"use_transcript": True}

    class Segment:
        local_id = "S1"
        start = 0
        end = 1
        text = "hello"

    assert context_request(Run(), [Segment()], []) is None
    Run.provider_id = "openai"
    payload = context_request(Run(), [Segment()], [_event("Action_1", "hello there", 0, 1)])
    assert payload["sends_audio"] is False
    assert payload["sends_transcript_text"] is True
    assert "hello" in payload["user_prompt"]


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
    with TestClient(create_app()) as test_client:
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


def _upload(client, tmp_path: Path) -> dict:
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario
    from app.pipeline.infer import register_fixture

    project = client.post("/api/projects", json={"name": "Audio"}).json()
    scenario = SCENARIOS["fixed_camera"]()
    clip = tmp_path / "silent.mp4"
    write_clip(render_scenario(scenario), clip)
    with clip.open("rb") as handle:
        uploaded = client.post(
            "/api/videos",
            data={"project_id": project["id"]},
            files={"file": ("silent.mp4", handle, "video/mp4")},
        )
    assert uploaded.status_code == 200, uploaded.text
    video = uploaded.json()
    register_fixture(video["checksum"], "fixed_camera")
    return video


def _create(client, video_id: str, **extra):
    payload = {
        "name": "Audio run",
        "video_id": video_id,
        "provider_id": "fixture",
        "model_id": "authored-fixture",
        "modality": "sampled_frames",
        "research_mode": "fixed_camera",
        "sampling_preset": "standard",
        "analysis_pass": "single",
        "segmentation_mode": "disabled",
        "max_frames": 8,
    }
    payload.update(extra)
    created = client.post("/api/runs", json=payload)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_audio_options_rejected_when_transcription_is_off(client, tmp_path):
    video = _upload(client, tmp_path)
    created = client.post(
        "/api/runs",
        json={
            "video_id": video["id"],
            "provider_id": "fixture",
            "model_id": "authored-fixture",
            "research_mode": "fixed_camera",
            "transcribe": False,
            "use_transcript": True,
        },
    )
    assert created.status_code == 400


def test_disabled_audio_skips_whisper_and_keeps_visual_analysis(client, tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("whisper should not load")

    monkeypatch.setattr("app.audio.engine.transcribe_file", boom)
    video = _upload(client, tmp_path)
    run_id = _create(client, video["id"])
    _run_job(run_id)
    bundle = client.get(f"/api/runs/{run_id}").json()
    assert bundle["run"]["status"] == "completed"
    assert bundle["actions"]
    assert bundle["transcripts"] == []
    stages = {item["stage"]: item["status"] for item in bundle["stages"]}
    assert stages["transcription"] == "skipped"
    assert stages["audio_context"] == "skipped"
    assert "Whisper was not loaded" in bundle["stages"][-3]["checkpoint"]["reason"] or any(
        "Whisper was not loaded" in (item["checkpoint"] or {}).get("reason", "") for item in bundle["stages"]
    )
    assert calls["n"] == 0


def test_no_audio_track_transcribes_nothing(client, tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("whisper should not load without an audio track")

    monkeypatch.setattr("app.audio.engine.transcribe_file", boom)
    video = _upload(client, tmp_path)
    run_id = _create(client, video["id"], transcribe=True, use_transcript=True)
    _run_job(run_id)
    bundle = client.get(f"/api/runs/{run_id}").json()
    assert bundle["run"]["status"] == "completed"
    assert any(item["outcome"] == "attempted" for item in bundle["actions"])
    stage = next(item for item in bundle["stages"] if item["stage"] == "transcription")
    assert stage["status"] == "completed"
    assert stage["checkpoint"]["has_audio"] is False
    assert bundle["transcripts"] == []
    assert calls["n"] == 0


def test_transcript_alignment_does_not_change_action_outcomes(client, tmp_path, monkeypatch):
    def fake_transcribe(_path, **_kwargs):
        return {
            "language": "en",
            "language_probability": 0.9,
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "note": "",
            "segments": [
                {
                    "local_id": "Speech_1",
                    "start": 1.2,
                    "end": 2.0,
                    "text": "A hand lifts the threaded cylinder off the surface.",
                    "confidence": 0.8,
                    "words": [],
                },
                {
                    "local_id": "Speech_2",
                    "start": 2.2,
                    "end": 3.4,
                    "text": "I am choosing the shorter piece because the longer one would pass through the material.",
                    "confidence": 0.7,
                    "words": [],
                },
                {
                    "local_id": "Speech_3",
                    "start": 8.0,
                    "end": 9.0,
                    "text": "I should buy milk from the market tomorrow.",
                    "confidence": 0.6,
                    "words": [],
                },
                {
                    "local_id": "Speech_4",
                    "start": 6.6,
                    "end": 7.6,
                    "text": "All four rings are installed.",
                    "confidence": 0.5,
                    "words": [],
                },
                {
                    "local_id": "Speech_5",
                    "start": 0.0,
                    "end": 0.3,
                    "text": "A hand lifts the threaded cylinder off the surface.",
                    "confidence": 0.4,
                    "words": [],
                },
            ],
        }

    monkeypatch.setattr("app.audio.service.transcribe_file", fake_transcribe)
    video = _upload(client, tmp_path)
    from app.db import SessionLocal
    from app.models.media import VideoAsset

    session = SessionLocal()
    row = session.get(VideoAsset, video["id"])
    row.audio_probed = True
    row.has_audio = True
    session.commit()
    session.close()
    run_id = _create(client, video["id"], transcribe=True, use_transcript=True, whisper_model="base")
    _run_job(run_id)
    bundle = client.get(f"/api/runs/{run_id}").json()
    assert bundle["run"]["status"] == "completed"
    outcomes = {item["local_id"]: item["outcome"] for item in bundle["actions"]}
    assert "attempted" in outcomes.values()
    assert "completed" in outcomes.values()
    classes = {item["classification"] for item in bundle["alignments"]}
    assert "visually_supported" in classes
    assert "irrelevant" in classes
    assert "contradictory" in classes
    assert all(item["speech_is_not_completion"] for item in bundle["alignments"])
    context = next(item for item in bundle["stages"] if item["stage"] == "audio_context")
    assert context["checkpoint"]["transcript_text_sent_to_provider"] is False
    segment = next(item for item in bundle["transcripts"] if item["local_id"] == "Speech_3")
    edited = client.patch(f"/api/runs/{run_id}/transcript/{segment['id']}", json={"text": "Corrected aside."})
    assert edited.status_code == 200
    saved = next(item for item in edited.json()["transcripts"] if item["id"] == segment["id"])
    assert saved["text"] == "Corrected aside."
    assert saved["text_model"].startswith("I should buy milk")
    assert saved["corrected"] is True
    visual = client.post(f"/api/runs/{run_id}/visual-only")
    assert visual.status_code == 200
    visual_id = visual.json()["id"]
    _run_job(visual_id)
    report = client.post(
        f"/api/runs/{run_id}/audio-comparison",
        json={"visual_run_id": visual_id, "assisted_run_id": run_id},
    )
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["accuracy"] is None
    assert body["workflow_document_changed"] is False
    assert "ground_truth" not in body
    exported = client.get(f"/api/runs/{run_id}/export")
    assert exported.content[:2] == b"PK"
    import io
    import zipfile

    archive = zipfile.ZipFile(io.BytesIO(exported.content))
    assert "transcripts.json" in archive.namelist()
    assert "alignments.json" in archive.namelist()
