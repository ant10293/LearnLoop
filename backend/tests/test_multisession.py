"""Multi-session ordering, identity, and incremental reconciliation."""

import os
from pathlib import Path

import numpy as np
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


def test_metadata_order_is_partial_when_timestamps_are_missing():
    from app.services.ordering import apply_context_if_unlocked, context_hypothesis, metadata_order

    clips = [
        {"id": "later", "capture_time_utc": "2026-03-04T00:00:00+00:00", "timestamp_reliability": "reliable", "order_locked": False, "order_index": None},
        {"id": "earlier", "capture_time_utc": "2026-03-01T00:00:00+00:00", "timestamp_reliability": "reliable", "order_locked": False, "order_index": None},
        {"id": "loose", "capture_time_utc": "", "timestamp_reliability": "absent", "order_locked": False, "order_index": None},
    ]
    order = metadata_order(clips)
    assert order["fact"] is False
    assert order["ranked"] == [["earlier"], ["later"]]
    assert "loose" in order["unordered"]
    assert "loose" in order["ambiguous_ids"]
    locked = [
        {"id": "a", "capture_time_utc": "2026-03-02T00:00:00+00:00", "timestamp_reliability": "reliable", "order_locked": True, "order_index": 1},
        {"id": "b", "capture_time_utc": "2026-03-01T00:00:00+00:00", "timestamp_reliability": "reliable", "order_locked": True, "order_index": 0},
    ]
    manual = metadata_order(locked)
    hypothesis = context_hypothesis(
        locked,
        {
            "a": [{"label": "ceramic figure", "final_state": "dried clay figure", "initial_state": "dried clay figure"}],
            "b": [{"label": "ceramic figure", "final_state": "wet clay figure", "initial_state": "wet clay figure"}],
        },
    )
    assert hypothesis["fact"] is False
    assert apply_context_if_unlocked(manual, hypothesis)["ranked"] == [["b"], ["a"]]


def test_same_clip_twins_are_not_merged_and_appearance_change_stays_uncertain():
    from app.services.identity import observed_transition, propose_links

    twins = propose_links(
        [
            {
                "video_id": "clip-a",
                "run_id": "run-a",
                "objects": [
                    {"local_id": "Object_1", "label": "red block", "characteristics": "small red block", "initial_state": "still", "final_state": "still"},
                    {"local_id": "Object_2", "label": "red block", "characteristics": "small red block", "initial_state": "still", "final_state": "still"},
                ],
            }
        ]
    )
    assert twins == []
    changed = propose_links(
        [
            {
                "video_id": "day-1",
                "run_id": "r1",
                "objects": [
                    {
                        "local_id": "Object_1",
                        "label": "ceramic figure",
                        "characteristics": "small ceramic figure",
                        "initial_state": "wet clay figure",
                        "final_state": "wet clay figure",
                    }
                ],
            },
            {
                "video_id": "day-4",
                "run_id": "r2",
                "objects": [
                    {
                        "local_id": "Object_1",
                        "label": "ceramic figure",
                        "characteristics": "small ceramic figure",
                        "initial_state": "dried clay figure",
                        "final_state": "dried clay figure",
                    }
                ],
            },
        ]
    )
    assert len(changed) == 1
    assert changed[0]["uncertain"] is True
    assert "Appearance differs" in changed[0]["note"]
    assert {member["video_id"] for member in changed[0]["members"]} == {"day-1", "day-4"}
    gap = observed_transition(
        {"final_state": "unfired figure"},
        {"initial_state": "visually changed figure"},
    )
    assert gap["evidence_class"] != "DIRECTLY_OBSERVED"
    assert gap["hypothesis"] == ""
    assert "unfired figure" in gap["observed_change"]
    dried = observed_transition({"final_state": "wet clay figure"}, {"initial_state": "dried clay figure"})
    assert dried["evidence_class"] == "INFERRED"
    assert "Hypothesis, not an observation" in dried["hypothesis"]
    assert dried["observed_change"] != dried["hypothesis"]


def test_pairwise_order_metric_is_ground_truth_not_confidence():
    from app.evaluation.multisession import evaluate_project

    report = evaluate_project(
        {
            "order": {"ranked": [["a"], ["b"]], "unordered": ["c"], "ambiguous_ids": ["c"]},
            "links": [],
            "transitions": [],
            "sessions": [{"video_ids": ["a", "b"]}],
            "workflow": {"steps": [{"id": "gap", "kind": "unobserved_interval", "video_ids": ["a", "b"]}], "procedural_dependencies": []},
            "clip_run_ids": {"a": "r", "b": "r2", "c": ""},
        },
        {"order": ["a", "b", "c"], "identity_groups": [], "transitions": [], "dependencies": [], "sessions": [{"video_ids": ["a", "b"]}]},
        [0.2],
    )
    assert report["is_accuracy"] is True
    assert report["clip_ordering"]["accuracy"] == 1.0
    assert report["clip_ordering"]["comparable_pairs"] == 1
    assert report["confidence"]["is_accuracy"] is False
    assert report["unresolved_intervals"]["count"] == 1


def test_task_orders_by_capture_time_and_keeps_prior_revisions(client: TestClient, tmp_path: Path):
    from app.db import SessionLocal
    from app.models.media import VideoAsset
    from app.models.runs import AnalysisRun, Job
    from sqlalchemy import func, select

    created = client.post("/api/tasks", json={"name": "Across days", "description": "One figure, several recordings"})
    assert created.status_code == 200, created.text
    project_id = created.json()["project"]["id"]
    early = _clip(tmp_path / "early.mp4", (30, 70, 180), "2026-03-01T15:00:00Z")
    late = _clip(tmp_path / "late.mp4", (20, 140, 50), "2026-03-04T15:00:00Z")
    with early.open("rb") as early_handle, late.open("rb") as late_handle:
        uploaded = client.post(
            f"/api/tasks/{project_id}/clips",
            data={"research_mode": "fixed_camera"},
            files=[
                ("files", ("late.mp4", late_handle, "video/mp4")),
                ("files", ("early.mp4", early_handle, "video/mp4")),
            ],
        )
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    assert body["jobs_created"] == 0
    by_name = {clip["filename"]: clip for clip in body["clips"]}
    assert by_name["early.mp4"]["timestamp_reliability"] == "reliable"
    assert by_name["early.mp4"]["file_mtime"] != by_name["early.mp4"]["capture_time_utc"]
    assert body["order"]["fact"] is False
    assert body["order"]["ranked"] == [[by_name["early.mp4"]["id"]], [by_name["late.mp4"]["id"]]]
    _cache_object(by_name["early.mp4"]["id"], "wet clay figure", "wet clay figure")
    _cache_object(by_name["late.mp4"]["id"], "dried clay figure", "dried clay figure")
    session = SessionLocal()
    jobs_before = session.scalar(select(func.count()).select_from(Job)) or 0
    run_ids = {
        run.video_id: run.id
        for run in session.scalars(select(AnalysisRun).where(AnalysisRun.project_id == project_id))
    }
    session.close()
    reconciled = client.post(f"/api/tasks/{project_id}/reconcile")
    assert reconciled.status_code == 200, reconciled.text
    reading = reconciled.json()
    assert reading["jobs_created"] == 0
    assert reading["revision"]["version"] == 2
    assert len(reading["revisions"]) == 2
    assert reading["revisions"][0]["document"]["clip_run_ids"][by_name["early.mp4"]["id"]] == ""
    assert reading["revision"]["document"]["clip_run_ids"][by_name["early.mp4"]["id"]] == run_ids[by_name["early.mp4"]["id"]]
    assert reading["links"]
    assert all(link["status"] == "proposed" and link["uncertain"] for link in reading["links"])
    assert reading["transitions"]
    assert reading["transitions"][0]["evidence_class"] == "INFERRED"
    assert reading["transitions"][0]["observed_change"] != reading["transitions"][0]["hypothesis"]
    assert "Hypothesis, not an observation" in reading["transitions"][0]["hypothesis"]
    assert reading["context_hypothesis"]["fact"] is False
    assert any(step["kind"] == "unobserved_interval" for step in reading["workflow"]["steps"])
    session = SessionLocal()
    jobs_after = session.scalar(select(func.count()).select_from(Job)) or 0
    assert jobs_after == jobs_before
    session.close()
    reversed_ids = [by_name["late.mp4"]["id"], by_name["early.mp4"]["id"]]
    ordered = client.post(f"/api/tasks/{project_id}/order", json={"video_ids": reversed_ids})
    assert ordered.status_code == 200, ordered.text
    assert ordered.json()["order"]["method"] == "manual"
    assert [group[0] for group in ordered.json()["order"]["ranked"]] == reversed_ids
    loose = _clip(tmp_path / "loose.mp4", (90, 90, 90), None)
    with loose.open("rb") as handle:
        added = client.post(
            f"/api/tasks/{project_id}/clips",
            data={"research_mode": "unconstrained_head_mounted"},
            files=[("files", ("loose.mp4", handle, "video/mp4"))],
        )
    assert added.status_code == 200, added.text
    added_body = added.json()
    assert added_body["jobs_created"] == 0
    assert added_body["revision"]["version"] == 4
    assert len(added_body["revisions"]) == 4
    loose_clip = next(clip for clip in added_body["clips"] if clip["filename"] == "loose.mp4")
    assert loose_clip["order_ambiguous"] is True
    assert loose_clip["research_mode"] == "unconstrained_head_mounted"
    assert run_ids[by_name["early.mp4"]["id"]] == added_body["revision"]["document"]["clip_run_ids"][by_name["early.mp4"]["id"]]
    exported = client.get(f"/api/tasks/{project_id}/export")
    assert exported.status_code == 200
    assert exported.content[:2] == b"PK"
    scored = client.post(
        f"/api/tasks/{project_id}/evaluate",
        json={"ground_truth": {"order": reversed_ids, "identity_groups": [], "sessions": [], "transitions": [], "dependencies": []}},
    )
    assert scored.status_code == 200, scored.text
    assert scored.json()["metrics"]["confidence"]["is_accuracy"] is False
    assert scored.json()["metrics"]["clip_ordering"]["accuracy"] == 1.0
    session = SessionLocal()
    videos = list(session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id)))
    assert len(videos) == 3
    session.close()


def _clip(path: Path, color: tuple[int, int, int], creation_time: str | None) -> Path:
    from app.demo.encode import write_clip
    from app.demo.scenarios import HEIGHT, WIDTH

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = color
    write_clip([frame, frame, frame], path, fps=5, creation_time=creation_time)
    return path


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
