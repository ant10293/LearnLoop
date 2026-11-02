"""Dry run, replay, and live safety through the API. SAM is a fake. Nothing reaches a network."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    InstanceRecord,
    ObjectPrediction,
    ObservationPrediction,
    PerceptionRecord,
)
from app.schemas.enums import EvidenceClass, Provenance
from sqlalchemy import func, select

BOX = {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    settings.worker_enabled = False
    settings.auto_seed = False
    settings.ingest_inline = True
    settings.sam_price_per_frame_usd = None
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database

    init_database()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def no_language_model(monkeypatch):
    """Any route to a language model or the job queue fails the test."""
    from app.pipeline import infer
    from app.providers import registry

    def fail(*_args, **_kwargs):
        raise AssertionError("A language model or pipeline path was reached.")

    monkeypatch.setattr(infer, "produce_document", fail)
    monkeypatch.setattr(registry, "get_provider", fail)


class FakeSam:
    """Records calls and answers with whatever text the test gives for a prompt."""

    def __init__(self, answer=None, error=None, hang=False):
        self.answer = answer
        self.error = error
        self.hang = hang
        self.uploads = 0
        self.calls: list[dict] = []
        self.started = asyncio.Event() if hang else None

    async def upload_video(self, _path):
        self.uploads += 1
        return f"file-{self.uploads}"

    async def segment_video(self, _path, prompt, *, prompt_kind="concept", file_id="", propagation_direction=None):
        from app.providers.sam import SamImageResult, parse_sam_output

        self.calls.append({"prompt": prompt, "kind": prompt_kind, "direction": propagation_direction, "file_id": file_id})
        if self.hang:
            await asyncio.sleep(60)
        if self.error:
            raise self.error
        text = self.answer(prompt, propagation_direction) if self.answer else ""
        return SamImageResult(text, parse_sam_output(text), 12, f"req-{len(self.calls)}", 3, 4, file_id)


def _evidence(provenance=Provenance.CV_DETECTION):
    return Evidence(classification=EvidenceClass.DIRECTLY_OBSERVED, provenance=provenance)


def _setup(client, tmp_path: Path, *, with_seeds=True):
    """A real clip and an authored analysis: Object_1 has a measured gap, Object_2 shares its label."""
    from app.db import SessionLocal
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario
    from app.models.media import VideoAsset
    from app.models.runs import AnalysisRevision
    from app.pipeline.sampling_stage import timeline_for
    from app.services.experiments import create_run
    from app.services.materialize import materialize
    from app.services.storage import data_path

    project = client.post("/api/projects", json={"name": "Recovery"}).json()
    clip = tmp_path / "fixed.mp4"
    frames = render_scenario(SCENARIOS["fixed_camera"]())
    write_clip((frames * (150 // len(frames) + 1))[:150], clip)
    with clip.open("rb") as handle:
        uploaded = client.post(
            "/api/videos", data={"project_id": project["id"]}, files={"file": ("fixed.mp4", handle, "video/mp4")}
        )
    assert uploaded.status_code == 200, uploaded.text
    session = SessionLocal()
    video = session.get(VideoAsset, uploaded.json()["id"])
    run = create_run(
        session,
        {
            "video_id": video.id,
            "provider_id": "deterministic",
            "model_id": "motion-baseline",
            "research_mode": "fixed_camera",
            "modality": "sampled_frames",
            "segmentation_mode": "boxes_only",
        },
        enqueue=False,
    )
    pts = timeline_for(video, None).pts
    times = [pts[index] for index in (0, 15, 30, 110, 125, 140)]
    rows = []
    tracks = []
    for time in times:
        rows.append(
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=time,
                bbox=BBox(**BOX),
                evidence=_evidence(Provenance.AI_PREDICTION),
                geometry_origin="vlm_estimated",
                geometry_authority="vlm",
            )
        )
        tracks.append(
            ObservationPrediction(
                object_local_id="track_1",
                timestamp=time,
                bbox=BBox(**BOX),
                evidence=_evidence(),
                track_ref="track_1",
                geometry_origin="detected",
                geometry_authority="motion",
                detection_support="detected",
                temporal_support="repeated",
            )
        )
    document = AnalysisDocument(
        summary="authored",
        objects=[
            ObjectPrediction(local_id="Object_1", label="washer", instance_id="track_1" if with_seeds else None, association_status="associated"),
            ObjectPrediction(local_id="Object_2", label="washer"),
        ],
        observations=rows,
    )
    spatial = AnalysisDocument(
        observations=tracks,
        perception=PerceptionRecord(
            strategy="temporal",
            detector_id="motion",
            instances=[InstanceRecord(instance_id="track_1", status="active", first_seen=times[0], last_seen=times[-1])],
        ),
    )
    path = data_path("runs", run.id, "spatial.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spatial.model_dump_json())
    revision = AnalysisRevision(run_id=run.id, layer="validated", version=1, document_json=document.model_dump(mode="json"), note="seed")
    session.add(revision)
    session.flush()
    materialize(session, run.id, revision.id, document, [])
    run.active_revision_id = revision.id
    run.status = "completed"
    session.commit()
    ids = (run.id, revision.id, times)
    session.close()
    return ids


def _create(client, run_id: str, limits=None, target_ids=None) -> dict:
    candidates = client.get(f"/api/runs/{run_id}/recovery/candidates").json()
    gap = next(item for item in candidates["targets"] if "tracking_gap" in item["reasons"])
    response = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={"target_ids": target_ids or [gap["target_id"]], "limits": limits or {}},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _answer(width=None, height=None, inside=True, x_shift=0.0):
    """Follow BOX on every planned frame of the prompt's tracked direction."""
    import re

    def build(prompt, direction):
        match = re.search(r"w=(\d+);h=(\d+)", prompt)
        w, h = int(match.group(1)), int(match.group(2))
        first = int(re.match(r"<(\d+)f>", prompt).group(1))
        x1, y1 = round((BOX["x"] + x_shift) * w), round(BOX["y"] * h)
        x2, y2 = round((BOX["x"] + x_shift + BOX["w"]) * w) - 1, round((BOX["y"] + BOX["h"]) * h) - 1
        frames = range(first, first + 60) if direction == "forward" else range(first, max(first - 60, -1), -1)
        line = "1<|box;x1={};y1={};x2={};y2={};w={};h={};c=0.9|><|mask;data=1,1,!A|>".format(x1, y1, x2, y2, width or w, height or h)
        return "\n".join(f"<{frame}f>{line}" for frame in frames)

    return build


def _counts():
    from app.db import SessionLocal
    from app.models.runs import AnalysisRevision, Job, ModelRequestRow, PipelineStageRun

    session = SessionLocal()
    data = {
        "requests": session.scalar(select(func.count()).select_from(ModelRequestRow)),
        "jobs": session.scalar(select(func.count()).select_from(Job)),
        "revisions": session.scalar(select(func.count()).select_from(AnalysisRevision)),
        "stages": sorted((row.stage, row.status) for row in session.scalars(select(PipelineStageRun))),
    }
    session.close()
    return data


def _install(monkeypatch, fake: FakeSam) -> FakeSam:
    from app.services import recovery

    monkeypatch.setattr(recovery, "_adapter", lambda *_args, **_kwargs: fake)
    return fake


def _live(client, experiment: dict, **extra):
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    return client.post(
        f"/api/recovery/experiments/{experiment['id']}/live",
        json={"confirm": True, "plan_hash": plan["plan_hash"], **extra},
    )


def test_candidates_find_the_gap_and_its_seeds(client, tmp_path):
    run_id, _revision, times = _setup(client, tmp_path)
    data = client.get(f"/api/runs/{run_id}/recovery/candidates").json()
    gap = next(item for item in data["targets"] if "tracking_gap" in item["reasons"])
    assert gap["object_local_id"] == "Object_1"
    assert gap["time_start"] == pytest.approx(times[2])
    assert gap["time_end"] == pytest.approx(times[3])
    assert gap["recoverable"]
    assert data["seed_counts"]["independently_supported"] > 0


def test_dismissing_hides_a_problem_survives_a_reload_and_can_be_undone(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRevision

    run_id, revision_id, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    before = _counts()
    session = SessionLocal()
    original = json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True)
    session.close()
    url = f"/api/runs/{run_id}/recovery"
    target = client.get(f"{url}/candidates").json()["targets"][0]
    assert target["dismissed"] is False

    def flag():
        rows = client.get(f"{url}/candidates").json()["targets"]
        return next(item["dismissed"] for item in rows if item["target_id"] == target["target_id"])

    assert client.post(f"{url}/dismissals/{target['target_id']}").status_code == 200
    assert flag() is True
    # Dismissing twice keeps one record.
    assert client.post(f"{url}/dismissals/{target['target_id']}").status_code == 200
    assert client.delete(f"{url}/dismissals/{target['target_id']}").status_code == 200
    assert flag() is False
    assert client.post(f"{url}/dismissals/not-a-target").status_code == 400
    # Nothing but the flag changed, and it does not make a box trustworthy.
    assert _counts() == before and fake.calls == []
    session = SessionLocal()
    assert json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True) == original
    session.close()


def test_a_dismissal_does_not_change_seed_trust(client, tmp_path):
    run_id, _revision, _times = _setup(client, tmp_path)
    url = f"/api/runs/{run_id}/recovery"
    data = client.get(f"{url}/candidates").json()
    client.post(f"{url}/dismissals/{data['targets'][0]['target_id']}")
    assert client.get(f"{url}/candidates").json()["seed_counts"] == data["seed_counts"]


def test_creating_an_experiment_runs_nothing(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    before = _counts()
    experiment = _create(client, run_id)
    assert experiment["status"] == "draft"
    assert experiment["plan"] is None
    assert fake.calls == [] and _counts() == before


def test_dry_run_plans_with_zero_provider_calls_and_estimates(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    before = _counts()
    experiment = _create(client, run_id, limits={"max_requests": 4})
    result = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()
    plan = result["plan"]
    assert fake.calls == [] and fake.uploads == 0
    assert _counts() == before
    assert result["status"] == "planned"
    assert [spec["direction"] for spec in plan["requests"]] == ["forward", "backward"]
    assert plan["totals"]["requests"] == 2
    assert plan["totals"]["billed_frames"] == sum(spec["billed_frames"] for spec in plan["requests"])
    assert plan["totals"]["estimated_cost_usd"] is None
    assert "unknown" in plan["totals"]["cost_note"]
    forward, backward = plan["requests"]
    assert forward["prompt"].startswith("<0f>1<|box;")
    assert backward["prompt"].startswith(f"<{len(backward['frame_times']) - 1}f>")
    assert plan["violations"] == []


def test_a_configured_rate_produces_a_cost_and_enforces_the_budget(client, tmp_path):
    from app.config import settings

    run_id, _revision, _times = _setup(client, tmp_path)
    settings.sam_price_per_frame_usd = 0.01
    experiment = _create(client, run_id, limits={"cost_budget_usd": 0.05})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    assert plan["totals"]["estimated_cost_usd"] > 0.05
    assert any("budget" in item for item in plan["violations"])
    settings.sam_price_per_frame_usd = None


def test_recovery_only_clips_the_relevant_interval(client, tmp_path):
    run_id, _revision, times = _setup(client, tmp_path)
    experiment = _create(client, run_id, limits={"max_requests": 4})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    for spec in plan["requests"]:
        assert spec["source_start"] >= times[2] - 1.01
        assert spec["source_end"] <= times[3] + 1.01
        assert len(spec["frame_times"]) < 150


def test_no_trustworthy_seed_blocks_the_target_before_billing(client, tmp_path):
    run_id, _revision, _times = _setup(client, tmp_path, with_seeds=False)
    experiment = _create(client, run_id)
    result = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()
    assert result["plan"]["requests"] == []
    assert "trustworthy seed" in result["plan"]["blocked"][0]["reason"]


def test_a_candidate_needs_explicit_confirmation_then_overrides_automatic_choice(client, tmp_path):
    run_id, _revision, _times = _setup(client, tmp_path, with_seeds=False)
    experiment = _create(client, run_id)
    candidate = next(
        seed
        for seed in client.get(f"/api/runs/{run_id}/recovery/candidates").json()["seeds"]
        if seed["object_local_id"] == "Object_1" and seed["trust"] == "candidate"
    )
    target = experiment["targets"][0]["target_id"]
    url = f"/api/recovery/experiments/{experiment['id']}"
    client.patch(f"{url}/seeds", json={"targets": [{"target_id": target, "seed_id": candidate["seed_id"]}]})
    blocked = client.post(f"{url}/dry-run").json()["plan"]
    assert "confirmation" in blocked["blocked"][0]["reason"]
    client.patch(f"{url}/seeds", json={"targets": [{"target_id": target, "seed_id": candidate["seed_id"], "confirm_seed": True}]})
    plan = client.post(f"{url}/dry-run").json()["plan"]
    assert plan["requests"][0]["objects"][0]["seed"]["trust"] == "human_verified"


def test_a_manual_seed_box_and_points_override_the_automatic_seed(client, tmp_path):
    run_id, _revision, times = _setup(client, tmp_path)
    response = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={
            "manual_targets": [
                {
                    "object_local_id": "Object_2",
                    "time_start": times[2],
                    "time_end": times[3],
                    "direction": "forward",
                    "seed": {
                        "timestamp": times[2],
                        "bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
                        "points": [{"x": 0.51, "y": 0.51, "negative": True}],
                    },
                }
            ]
        },
    ).json()
    plan = client.post(f"/api/recovery/experiments/{response['id']}/dry-run").json()["plan"]
    spec = plan["requests"][0]
    assert spec["direction"] == "forward"
    assert spec["objects"][0]["seed"]["manual"] is True
    assert "-<|point;" in spec["prompt"]


def test_several_seeds_go_out_in_one_request_when_their_clips_overlap(client, tmp_path):
    run_id, _revision, times = _setup(client, tmp_path)
    box = {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}

    def target(local_id, start, end):
        return {
            "object_local_id": local_id,
            "time_start": start,
            "time_end": end,
            "direction": "forward",
            "seed": {"timestamp": start, "bbox": box},
        }

    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={
            "manual_targets": [
                target("Object_1", times[2], times[2] + 1.0),
                target("Object_2", times[2], times[2] + 1.0),
                target("Object_1", times[2] + 0.5, times[2] + 1.5),
            ],
            "limits": {"padding_sec": 0},
        },
    ).json()
    plan = client.post(f"/api/recovery/experiments/{created['id']}/dry-run").json()["plan"]
    assert plan["blocked"] == []
    assert plan["totals"]["requests"] == 1
    spec = plan["requests"][0]
    assert len(spec["objects"]) == 3
    assert spec["bindings"] == {"1": "Object_1", "2": "Object_2"}
    assert spec["prompt"].count("1<|box;") == 2


def test_frame_limits_refuse_before_any_call(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    experiment = _create(client, run_id, limits={"max_frames_per_request": 5})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    assert plan["requests"] == []
    assert "limit per request" in plan["blocked"][0]["reason"]
    assert fake.calls == []


def test_request_and_total_frame_limits_block_a_live_run(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    experiment = _create(client, run_id, limits={"max_requests": 1})
    response = _live(client, experiment)
    assert response.status_code == 400
    assert "Nothing was submitted" in response.json()["detail"]
    experiment = _create(client, run_id, limits={"max_total_billed_frames": 10, "max_requests": 4})
    response = _live(client, experiment)
    assert response.status_code == 400
    assert "bills about" in response.json()["detail"]
    assert fake.calls == []


def test_live_needs_explicit_confirmation_and_the_approved_plan(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    experiment = _create(client, run_id, limits={"max_requests": 4})
    url = f"/api/recovery/experiments/{experiment['id']}/live"
    assert client.post(url, json={}).status_code == 400
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    assert client.post(url, json={"plan_hash": plan["plan_hash"]}).status_code == 400
    stale = client.post(url, json={"confirm": True, "plan_hash": "not-this-plan"})
    assert stale.status_code == 409
    assert "Nothing was submitted" in stale.json()["detail"]
    assert fake.calls == [] and fake.uploads == 0


def test_replay_without_a_stored_response_stays_offline(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    experiment = _create(client, run_id)
    result = client.post(f"/api/recovery/experiments/{experiment['id']}/replay").json()
    assert fake.calls == [] and fake.uploads == 0
    assert "Nothing was sent" in result["error"]
    assert result["proposals"] == []


def test_live_then_replay_gives_the_same_rows_with_no_new_calls(client, tmp_path, monkeypatch, no_language_model):
    run_id, revision_id, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_answer()))
    before = _counts()
    experiment = _create(client, run_id, limits={"max_requests": 4})
    result = _live(client, experiment)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["status"] == "completed", data["error"]
    assert data["remote_submitted"] is True
    assert len(fake.calls) == 2
    assert {call["direction"] for call in fake.calls} == {"forward", "backward"}
    assert all(call["kind"] == "spatial" for call in fake.calls)
    proposal = data["proposals"][0]
    assert proposal["decision"] == "pending"
    assert data["pending_proposals"] == 1
    assert proposal["metrics"]["counts"]["supported"] > 0
    assert proposal["metrics"]["previously_missing_frames"] > 0
    assert proposal["metrics"]["usage"]["requests"] == 2
    rows = sorted((row["timestamp"], row["outcome"]) for row in proposal["rows"])
    # The source analysis has not changed.
    after = _counts()
    assert after["revisions"] == before["revisions"] and after["stages"] == before["stages"] and after["jobs"] == before["jobs"]
    assert after["requests"] == before["requests"] + 2
    logged = [row for row in client.get(f"/api/runs/{run_id}/requests").json() if row["stage"] == "targeted_recovery"]
    assert len(logged) == 2 and all(row["response_text"].startswith("<") for row in logged)
    client.get(f"/api/runs/{run_id}").json()
    from app.db import SessionLocal
    from app.models.runs import AnalysisRun

    session = SessionLocal()
    assert session.get(AnalysisRun, run_id).active_revision_id == revision_id
    session.close()

    calls = len(fake.calls)
    replayed = client.post(f"/api/recovery/experiments/{experiment['id']}/replay").json()
    assert len(fake.calls) == calls
    assert replayed["status"] == "completed"
    current = next(item for item in replayed["proposals"] if item["current"])
    assert current["generation"] == 2
    assert sorted((row["timestamp"], row["outcome"]) for row in current["rows"]) == rows
    assert any(not item["current"] for item in replayed["proposals"])


def test_returned_times_map_to_source_frames_through_the_clip(client, tmp_path, monkeypatch):
    run_id, _revision, times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    plan = {spec["key"]: spec for spec in data["plan"]["requests"]}
    proposal = data["proposals"][0]
    assert proposal["rows"][0]["timestamp"] >= times[2] - 1e-6
    for row in proposal["rows"]:
        spec = plan[row["request_key"]]
        assert spec["frame_times"][row["frame_index"]] == pytest.approx(row["timestamp"])


def test_a_changed_input_never_reuses_a_stored_response(client, tmp_path, monkeypatch):
    run_id, _revision, times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    _live(client, experiment)
    target = experiment["targets"][0]["target_id"]
    url = f"/api/recovery/experiments/{experiment['id']}"
    client.patch(
        f"{url}/seeds",
        json={"targets": [{"target_id": target, "direction": "forward", "seed": {"timestamp": times[2], "bbox": {"x": 0.21, "y": 0.2, "w": 0.1, "h": 0.1}}}]},
    )
    result = client.post(f"{url}/replay").json()
    assert "Nothing was sent" in result["error"]
    calls = len(fake.calls)
    assert calls == 2


def test_cache_keys_cover_media_prompt_direction_and_times():
    from app.services.recovery import cache_key

    base = cache_key("sum", [0.0, 0.1], 640, 360, "p", "forward")
    assert base == cache_key("sum", [0.0, 0.1], 640, 360, "p", "forward")
    for other in (
        cache_key("other", [0.0, 0.1], 640, 360, "p", "forward"),
        cache_key("sum", [0.0, 0.2], 640, 360, "p", "forward"),
        cache_key("sum", [0.0, 0.1], 320, 360, "p", "forward"),
        cache_key("sum", [0.0, 0.1], 640, 360, "q", "forward"),
        cache_key("sum", [0.0, 0.1], 640, 360, "p", "both"),
    ):
        assert other != base


def test_a_second_experiment_reuses_the_stored_response_for_free(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_answer()))
    _live(client, _create(client, run_id, limits={"max_requests": 4}))
    calls = len(fake.calls)
    second = _create(client, run_id, limits={"max_requests": 4})
    plan = client.post(f"/api/recovery/experiments/{second['id']}/dry-run").json()["plan"]
    assert plan["totals"]["remote_requests"] == 0
    result = client.post(f"/api/recovery/experiments/{second['id']}/replay").json()
    assert result["status"] == "completed"
    assert len(fake.calls) == calls


def test_a_failed_request_keeps_the_analysis_and_is_not_sent_again(client, tmp_path, monkeypatch):
    from app.providers.types import ProviderError

    run_id, revision_id, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(error=ProviderError("provider exploded")))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    assert data["status"] == "failed"
    assert "was submitted" in data["error"]
    assert data["remote_submitted"] is True
    assert len(fake.calls) == 1
    assert data["requests"][0]["submission"] == "submitted"
    from app.db import SessionLocal
    from app.models.runs import AnalysisRun

    session = SessionLocal()
    assert session.get(AnalysisRun, run_id).active_revision_id == revision_id
    session.close()
    # A retry refuses the request that may have been billed, then runs only with approval.
    fake.error = None
    fake.answer = _answer()
    retried = _live(client, experiment).json()
    assert len(fake.calls) >= 1
    assert sum(call["prompt"] == fake.calls[0]["prompt"] for call in fake.calls) == 1
    assert "may have been billed" in retried["error"]
    keys = [item["cache_key"] for item in retried["requests"] if item["submission"] == "submitted"]
    approved = _live(client, experiment, resubmit_keys=keys).json()
    assert approved["status"] == "completed", approved["error"]
    assert sum(call["prompt"] == fake.calls[0]["prompt"] for call in fake.calls) == 2


def test_an_upload_failure_says_nothing_was_submitted(client, tmp_path, monkeypatch):
    class BadUpload(FakeSam):
        async def upload_video(self, _path):
            self.uploads += 1
            raise OSError("network down")

    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, BadUpload())
    experiment = _create(client, run_id, limits={"max_requests": 4, "max_upload_retries": 1})
    data = _live(client, experiment).json()
    assert data["status"] == "failed"
    assert "no SAM request was submitted" in data["error"]
    assert data["remote_submitted"] is False
    assert fake.uploads == 2 and fake.calls == []


def test_partial_responses_keep_what_arrived(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)

    class HalfWay(FakeSam):
        async def segment_video(self, path, prompt, **kwargs):
            if self.calls:
                self.calls.append({"prompt": prompt})
                raise OSError("connection reset")
            return await super().segment_video(path, prompt, **kwargs)

    _install(monkeypatch, HalfWay(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    assert data["status"] == "partial"
    assert data["proposals"][0]["rows"]


def test_malformed_output_rows_are_skipped_not_invented(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=lambda *_args: "<0f>garbage\n<9999f>1<|box;x1=1;y1=1;x2=2;y2=2;w=640;h=360|><|mask;data=1,1,!A|>\nnot a line"))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    assert data["status"] == "completed"
    proposal = data["proposals"][0]
    assert proposal["rows"] == []
    assert proposal["metrics"]["remaining_gaps"]


def test_cancel_before_a_call_sends_nothing(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models.recovery import RecoveryExperiment
    from app.services import recovery

    run_id, _revision, _times = _setup(client, tmp_path)
    fake = FakeSam(answer=_answer())
    experiment = _create(client, run_id, limits={"max_requests": 4})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    session = SessionLocal()
    row = session.get(RecoveryExperiment, experiment["id"])
    recovery.authorize_live(session, row, {"confirm": True, "plan_hash": plan["plan_hash"]})
    recovery.cancel(session, row)
    asyncio.run(recovery.execute_live(session, row.id, fake))
    assert row.status == "canceled"
    assert fake.calls == []
    session.close()


def test_cancel_in_flight_stops_waiting_and_warns_about_billing(client, tmp_path):
    from app.db import SessionLocal
    from app.models.recovery import RecoveryExperiment
    from app.services import recovery

    run_id, _revision, _times = _setup(client, tmp_path)
    fake = FakeSam(hang=True)
    experiment = _create(client, run_id, limits={"max_requests": 4})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    session = SessionLocal()
    row = session.get(RecoveryExperiment, experiment["id"])
    recovery.authorize_live(session, row, {"confirm": True, "plan_hash": plan["plan_hash"]})

    async def scenario():
        fake.started = asyncio.Event()
        work = asyncio.create_task(recovery.execute_live(session, row.id, fake))
        while not fake.calls:
            await asyncio.sleep(0.05)
        other = SessionLocal()
        target = other.get(RecoveryExperiment, row.id)
        target.cancel_requested = True
        other.commit()
        other.close()
        await asyncio.wait_for(work, timeout=20)

    asyncio.run(scenario())
    session.refresh(row)
    assert row.status == "canceled"
    assert "may still be billed" in row.error
    session.close()


def test_a_request_past_its_timeout_is_abandoned_and_not_retried(client, tmp_path):
    from app.db import SessionLocal
    from app.models.recovery import RecoveryExperiment
    from app.services import recovery

    run_id, _revision, _times = _setup(client, tmp_path)
    fake = FakeSam(hang=True)
    experiment = _create(client, run_id, limits={"max_requests": 4, "timeout_s": 0.2})
    plan = client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run").json()["plan"]
    session = SessionLocal()
    row = session.get(RecoveryExperiment, experiment["id"])
    recovery.authorize_live(session, row, {"confirm": True, "plan_hash": plan["plan_hash"]})
    asyncio.run(recovery.execute_live(session, row.id, fake))
    assert row.status == "failed"
    assert "may be billed" in row.error
    assert len(fake.calls) == 1
    session.close()


def test_two_recoveries_cannot_run_on_one_analysis(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models.recovery import RecoveryExperiment

    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam())
    first = _create(client, run_id, limits={"max_requests": 4})
    second = _create(client, run_id, limits={"max_requests": 4})
    session = SessionLocal()
    session.get(RecoveryExperiment, first["id"]).status = "running"
    session.commit()
    session.close()
    response = _live(client, second)
    assert response.status_code == 409
    assert fake.calls == []


def test_interrupted_recovery_is_not_resumed_at_startup(client, tmp_path):
    from app.db import SessionLocal
    from app.models.recovery import RecoveryExperiment
    from app.services.recovery import recover_interrupted

    run_id, _revision, _times = _setup(client, tmp_path)
    experiment = _create(client, run_id)
    session = SessionLocal()
    session.get(RecoveryExperiment, experiment["id"]).status = "running"
    session.commit()
    assert recover_interrupted(session) == 1
    assert session.get(RecoveryExperiment, experiment["id"]).status == "interrupted"
    session.close()


def test_accepting_forks_a_revision_and_leaves_the_original_intact(client, tmp_path, monkeypatch, no_language_model):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRevision, AnalysisRun

    run_id, revision_id, _times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    session = SessionLocal()
    original = json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True)
    session.close()
    proposal = next(item for item in data["proposals"] if item["current"])
    url = f"/api/recovery/experiments/{experiment['id']}/proposals/decide"
    result = client.post(url, json={"decision": "accepted", "proposal_ids": [proposal["id"]]})
    assert result.status_code == 200, result.text
    decided = next(item for item in result.json()["proposals"] if item["id"] == proposal["id"])
    assert decided["decision"] == "accepted"
    session = SessionLocal()
    run = session.get(AnalysisRun, run_id)
    assert run.active_revision_id == decided["applied_revision_id"] != revision_id
    assert json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True) == original
    new = AnalysisDocument.model_validate(session.get(AnalysisRevision, run.active_revision_id).document_json)
    assert session.get(AnalysisRevision, run.active_revision_id).layer == "human"
    old = AnalysisDocument.model_validate(json.loads(original))
    assert any(row.source_model == "sam-3.1" for row in new.observations)
    assert new.actions == old.actions and new.relationships == old.relationships and new.workflow_steps == old.workflow_steps
    session.close()
    again = client.post(url, json={"decision": "accepted", "proposal_ids": [proposal["id"]]})
    assert again.status_code == 409


def test_rejecting_keeps_the_analysis_and_the_proposal(client, tmp_path, monkeypatch):
    run_id, revision_id, _times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    proposal = data["proposals"][0]
    result = client.post(
        f"/api/recovery/experiments/{experiment['id']}/proposals/decide",
        json={"decision": "rejected", "proposal_ids": [proposal["id"]]},
    ).json()
    assert result["proposals"][0]["decision"] == "rejected"
    assert result["proposals"][0]["rows"]
    assert client.get(f"/api/runs/{run_id}").json()["revision"]["id"] == revision_id
    listed = client.get(f"/api/runs/{run_id}/recovery/experiments").json()
    assert listed[0]["pending_proposals"] == 0


def test_accept_refuses_a_changed_analysis(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRun
    from app.services.corrections import update_object

    run_id, _revision, _times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    data = _live(client, experiment).json()
    session = SessionLocal()
    update_object(session, session.get(AnalysisRun, run_id), "Object_1", {"label": "ring"})
    session.close()
    result = client.post(
        f"/api/recovery/experiments/{experiment['id']}/proposals/decide",
        json={"decision": "accepted", "proposal_ids": [data["proposals"][0]["id"]]},
    )
    assert result.status_code == 409


def test_recovery_code_cannot_reach_a_language_model_or_a_stage_reset():
    root = Path(__file__).resolve().parents[1] / "app"
    forbidden = {"produce_document", "retry_from", "execute_job", "get_provider", "run_video_segmentation", "_clear_sam_rows", "create_run"}
    for name in ("services/recovery.py", "api/recovery.py"):
        tree = ast.parse((root / name).read_text())
        used = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not used & forbidden, name


# ---- The normal pipeline ----


def _pipeline(client, tmp_path, monkeypatch, *, mode: str, strategy="concept", fake=None) -> str:
    from app.db import SessionLocal
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, build_document, render_scenario
    from app.models.runs import Job
    from app.pipeline import infer, locate_pass, sam_stage
    from app.pipeline.runner import execute_job
    from app.providers.types import ProviderResponse, Usage
    from app.services import credentials

    text = json.dumps(build_document(SCENARIOS["fixed_camera"]()).model_dump(mode="json"))

    class Authored:
        async def generate(self, _secret, _request):
            return ProviderResponse(text=text, raw={}, usage=Usage(), latency_ms=1)

    for module in (infer, locate_pass):
        monkeypatch.setattr(module, "needs_secret", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(module, "get_provider", lambda *_args, **_kwargs: Authored())
        monkeypatch.setattr(module, "model_for_call", lambda _session, run: (run.model_id, run.model_id))
    monkeypatch.setattr(credentials, "load_secret", lambda *_args, **_kwargs: ("key", "test"))
    monkeypatch.setattr(sam_stage, "load_secret", lambda *_args, **_kwargs: ("key", "test"))
    if fake is not None:
        monkeypatch.setattr(sam_stage, "SamAdapter", lambda *_args, **_kwargs: fake)
    project = client.post("/api/projects", json={"name": mode}).json()
    clip = tmp_path / f"{mode}-{strategy}.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]())[:60], clip)
    with clip.open("rb") as handle:
        video = client.post("/api/videos", data={"project_id": project["id"]}, files={"file": ("fixed.mp4", handle, "video/mp4")}).json()
    created = client.post(
        "/api/runs",
        json={
            "name": mode,
            "video_id": video["id"],
            "provider_id": "openai",
            "model_id": "gpt-4.1",
            "modality": "sampled_frames",
            "research_mode": "fixed_camera",
            "sampling_preset": "detailed",
            "analysis_pass": "single",
            "segmentation_mode": mode,
            "segmentation_model_id": "sam-3.1" if mode.startswith("sam") else "",
            "sam_prompt_strategy": strategy,
            "max_frames": 12,
        },
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["id"]
    session = SessionLocal()
    job = session.scalar(select(Job).where(Job.run_id == run_id, Job.status == "pending"))
    job.status = "running"
    session.commit()
    asyncio.run(execute_job(session, job))
    session.close()
    return run_id


class PipelineSam(FakeSam):
    def __init__(self):
        super().__init__(answer=lambda *_args: "")

    async def segment_video(self, video_path, prompt, *, prompt_kind="concept", file_id="", propagation_direction=None):
        return await super().segment_video(
            video_path, prompt, prompt_kind=prompt_kind, file_id=file_id or "f", propagation_direction=propagation_direction
        )


def _checkpoint(run_id: str) -> dict:
    from app.db import SessionLocal
    from app.models.runs import PipelineStageRun

    session = SessionLocal()
    stage = session.scalar(select(PipelineStageRun).where(PipelineStageRun.run_id == run_id, PipelineStageRun.stage == "segmentation"))
    data = dict(stage.checkpoint_json or {})
    data["status"] = stage.status
    data["error"] = stage.error
    session.close()
    return data


def test_selecting_sam_does_not_change_geometry_before_segmentation(client, tmp_path, monkeypatch):
    """Same mocked upstream answers under two segmentation modes give identical pre-segmentation boxes."""
    from app.pipeline.lineage import load_lineage
    from app.services.storage import data_path

    boxes = _pipeline(client, tmp_path, monkeypatch, mode="boxes_only")
    sam = _pipeline(client, tmp_path, monkeypatch, mode="sam_video", fake=PipelineSam())
    assert _checkpoint(sam)["status"] == "completed"

    def before_segmentation(run_id):
        rows = []
        for row in load_lineage(run_id):
            if row["stage"] in {"segmentation", "evidence_fusion"}:
                break
            rows.append({key: row[key] for key in ("stage", "digest", "boxes", "sources", "upstream_config")})
        return rows

    left, right = before_segmentation(boxes), before_segmentation(sam)
    assert left and left == right
    raw = [json.loads(data_path("runs", item, "semantic_raw.json").read_text()) for item in (boxes, sam)]
    for document in raw:
        # Wall-clock measurements of the tracker. They differ between any two runs.
        document["perception"]["diagnostics"].pop("elapsed_sec", None)
        document["perception"]["diagnostics"].pop("processing_fps", None)
    assert raw[0] == raw[1]


def test_a_stale_sam_summary_cannot_reach_a_stage_before_segmentation(tmp_path, monkeypatch):
    from app.config import settings
    from app.perception.sam_fusion import sam_evidence_block, write_sam_summary

    settings.data_dir = tmp_path
    write_sam_summary("run-1", "Object_1 track")
    assert sam_evidence_block("run-1", "scene_analysis") == ""
    assert sam_evidence_block("run-1", "action_analysis") == ""
    assert sam_evidence_block("run-1", "state_tracking") == ""
    assert sam_evidence_block("run-1", "workflow")


def test_concept_is_the_default_and_a_legacy_run_stays_on_it(client, tmp_path, monkeypatch):
    fake = PipelineSam()
    run_id = _pipeline(client, tmp_path, monkeypatch, mode="sam_video", fake=fake)
    assert fake.calls and all(call["kind"] == "concept" for call in fake.calls)
    checkpoint = _checkpoint(run_id)
    assert checkpoint["prompt_strategy"] == "concept"
    from types import SimpleNamespace

    from app.pipeline.sam_stage import prompt_strategy

    assert prompt_strategy(SimpleNamespace(config_json={})) == "concept"


def test_an_unknown_strategy_is_refused(client, tmp_path):
    from app.services.experiments import _prompt_strategy

    with pytest.raises(ValueError):
        _prompt_strategy({"sam_prompt_strategy": "guess"})


def test_trusted_spatial_with_no_seed_makes_no_concept_call_and_says_so(client, tmp_path, monkeypatch):
    fake = PipelineSam()
    run_id = _pipeline(client, tmp_path, monkeypatch, mode="sam_video", strategy="trusted_spatial", fake=fake)
    checkpoint = _checkpoint(run_id)
    assert checkpoint["prompt_strategy"] == "trusted_spatial"
    assert all(call["kind"] == "spatial" for call in fake.calls)
    assert all(item["prompt"] in {"spatial", "unobserved", "seed_rejected"} for item in checkpoint["prompt_outcomes"])


def test_hybrid_records_each_prompt_kind_separately(client, tmp_path, monkeypatch):
    fake = PipelineSam()
    run_id = _pipeline(client, tmp_path, monkeypatch, mode="sam_video", strategy="hybrid", fake=fake)
    checkpoint = _checkpoint(run_id)
    assert checkpoint["prompt_strategy"] == "hybrid"
    assert sum(checkpoint["prompt_kinds_sent"].values()) == len(fake.calls)
    for call in fake.calls:
        assert ("<|" in call["prompt"]) == (call["kind"] == "spatial")
    assert not any(call["kind"] == "spatial" and call["direction"] is None for call in fake.calls)


# ---- Request diagnostics ----


def _diagnosed(client, tmp_path, monkeypatch, answer) -> dict:
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=answer))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    _live(client, experiment)
    calls = len(fake.calls)
    response = client.get(f"/api/recovery/experiments/{experiment['id']}/diagnostics")
    assert response.status_code == 200, response.text
    assert len(fake.calls) == calls
    return response.json()


def test_diagnostics_before_any_run_send_nothing(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_answer()))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    client.post(f"/api/recovery/experiments/{experiment['id']}/dry-run")
    data = client.get(f"/api/recovery/experiments/{experiment['id']}/diagnostics").json()
    assert {item["verdict"] for item in data["requests"]} == {"not_run"}
    assert fake.calls == []


def test_diagnostics_say_when_boxes_came_back_without_claiming_they_are_right(client, tmp_path, monkeypatch):
    data = _diagnosed(client, tmp_path, monkeypatch, _answer())
    request = data["requests"][0]
    assert request["verdict"] == "boxes_returned" and request["boxes_returned"] is True
    assert "does not show" in request["message"]
    funnel = request["funnel"]
    assert funnel["box_tokens"] == funnel["parsed"] == funnel["usable"] == funnel["assigned"] > 0
    assert request["tracks"][0]["association"] == "associated"
    assert all("mask_data" not in box and "mask_payload" not in box for box in request["boxes"])


def test_diagnostics_name_an_empty_response(client, tmp_path, monkeypatch):
    request = _diagnosed(client, tmp_path, monkeypatch, lambda *_args: "")["requests"][0]
    assert request["verdict"] == "empty_output" and request["boxes_returned"] is False
    assert any(item["code"] == "prompted_track_missing" for item in request["findings"])


def test_diagnostics_tell_no_detection_from_a_parse_failure(client, tmp_path, monkeypatch):
    request = _diagnosed(client, tmp_path, monkeypatch, lambda *_args: "<0f>\n<1f>")["requests"][0]
    assert request["verdict"] == "no_boxes" and request["parsed"]["empty_frame_lines"] == 2


def test_diagnostics_count_lines_and_tokens_the_parser_skipped(client, tmp_path, monkeypatch):
    # A box with no mask is dropped by the parser, and a line with no frame mark is skipped.
    answer = lambda *_args: "nonsense\n<0f>1<|box;x1=1;y1=1;x2=2;y2=2;w=640;h=360|>"  # noqa: E731
    request = _diagnosed(client, tmp_path, monkeypatch, answer)["requests"][0]
    assert request["verdict"] == "parse_failure"
    assert request["parsed"]["unparsed_lines"] == 1 and request["parsed"]["dropped_tokens"] == 1
    assert request["parsed"]["unparsed_samples"][0] == "nonsense"
    assert {"unparsed_lines", "dropped_tokens"} <= {item["code"] for item in request["findings"]}


def test_diagnostics_keep_boxes_from_a_frame_outside_the_clip(client, tmp_path, monkeypatch):
    answer = lambda *_args: "<9999f>1<|box;x1=1;y1=1;x2=2;y2=2;w=640;h=360|><|mask;data=1,1,!A|>"  # noqa: E731
    request = _diagnosed(client, tmp_path, monkeypatch, answer)["requests"][0]
    assert request["verdict"] == "no_usable_boxes"
    assert request["boxes"][0]["status"] == "frame_out_of_range" and request["boxes"][0]["timestamp"] is None


def test_diagnostics_expose_unassigned_tracks_the_proposal_leaves_out(client, tmp_path, monkeypatch):
    base = _answer()

    def answer(prompt, direction):
        return base(prompt, direction).replace("<|mask;data=1,1,!A|>", "<|mask;data=1,1,!A|>,7<|box;x1=1;y1=1;x2=40;y2=40;w=640;h=360;c=0.3|><|mask;data=1,1,!A|>")

    request = _diagnosed(client, tmp_path, monkeypatch, answer)["requests"][0]
    extra = [box for box in request["boxes"] if box["raw_track_id"] == "7"]
    assert extra and all(box["status"] == "unassigned" and box["object_local_id"] == "" for box in extra)
    assert request["funnel"]["unassigned"] == len(extra)
    assert request["funnel"]["assigned"] == request["funnel"]["usable"] - len(extra)
    assert any(item["code"] == "unprompted_track" for item in request["findings"])
    track = next(item for item in request["tracks"] if item["raw_track_id"] == "7")
    assert track["prompted"] is False and track["boxes"] == len(extra)


def test_diagnostics_report_all_unassigned_output(client, tmp_path, monkeypatch):
    answer = lambda *_args: "<0f>7<|box;x1=1;y1=1;x2=40;y2=40;w=640;h=360;c=0.3|><|mask;data=1,1,!A|>"  # noqa: E731
    request = _diagnosed(client, tmp_path, monkeypatch, answer)["requests"][0]
    assert request["verdict"] == "no_assignment"


def test_diagnostics_report_a_failed_request(client, tmp_path, monkeypatch):
    run_id, _revision, _times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(error=RuntimeError("provider said no")))
    experiment = _create(client, run_id, limits={"max_requests": 4})
    _live(client, experiment)
    request = client.get(f"/api/recovery/experiments/{experiment['id']}/diagnostics").json()["requests"][0]
    assert request["verdict"] == "request_failed" and "provider said no" in request["message"]


# ---- Track loss and reacquisition, end to end ----


def _mark(frame, box, width, height):
    x1, y1 = round(box["x"] * width), round(box["y"] * height)
    x2, y2 = round((box["x"] + box["w"]) * width) - 1, round((box["y"] + box["h"]) * height) - 1
    return f"<{frame}f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height};c=0.9|><|mask;data=1,1,!A|>"


def _drifting(prompt, _direction):
    """SAM follows the part for 12 frames, then the object goes and the track settles on another region."""
    import re

    width, height = (int(item) for item in re.search(r"w=(\d+);h=(\d+)", prompt).groups())
    box = {"x": BOX["x"], "y": BOX["y"], "w": BOX["w"], "h": BOX["h"]}
    elsewhere = {**box, "x": min(BOX["x"] + 0.5, 0.85)}
    return "\n".join(_mark(frame, box if frame < 12 else elsewhere, width, height) for frame in range(26))


def _near_seed(prompt, _direction):
    """A new seed finds the part again, for a few frames either side of the seed only."""
    import re

    width, height = (int(item) for item in re.search(r"w=(\d+);h=(\d+)", prompt).groups())
    first = int(re.match(r"<(\d+)f>", prompt).group(1))
    return "\n".join(_mark(frame, BOX, width, height) for frame in range(max(first - 8, 0), first + 9))


def _broken_experiment(client, run_id: str, times: list[float]) -> dict:
    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={
            "manual_targets": [
                {
                    "object_local_id": "Object_1",
                    "time_start": times[2],
                    "time_end": times[2] + 2.5,
                    "direction": "forward",
                    "seed": {"timestamp": times[2], "bbox": BOX},
                }
            ],
            "limits": {"max_requests": 4},
        },
    )
    assert created.status_code == 200, created.text
    return created.json()


def _object_history(client, run_id: str, local_id="Object_1") -> dict:
    response = client.get(f"/api/runs/{run_id}/recovery/reacquisition")
    assert response.status_code == 200, response.text
    return next(item for item in response.json()["objects"] if item["object_local_id"] == local_id)


def test_loss_is_found_and_a_later_seed_reacquires_the_object_without_promoting_the_drift(client, tmp_path, monkeypatch, no_language_model):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRevision

    run_id, revision_id, times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_drifting))
    experiment = _broken_experiment(client, run_id, times)
    data = _live(client, experiment).json()
    assert data["status"] == "completed", data["error"]
    assert len(fake.calls) == 1
    rows = data["proposals"][0]["rows"]
    assert sum(row["lifecycle"] == "lost" for row in rows) > 0
    track = data["proposals"][0]["metrics"]["track"]
    assert track["failures"][0]["reason"] == "large_spatial_jump"
    assert track["segments"][0]["status"] == "lost"
    assert data["proposals"][0]["metrics"]["counts"]["unusable_frames"] > 0

    history = _object_history(client, run_id)
    plan = history["plans"][0]
    assert plan["status"] == "open" and plan["automatic_start_allowed"] is False
    assert len(fake.calls) == 1, "Planning sends nothing."
    assert history["stats"]["track_breaks"] == 1 and history["stats"]["api_requests"] == 1
    assert plan["recommended"] == "independent_observation", plan["strategies"]
    assert "Nothing has been sent" in plan["explanation"]
    used = plan["proposed_target"]
    assert used["seed_id"] and not used["confirm_seed"]
    drift_boxes = [row["bbox"] for row in rows if row["lifecycle"] == "lost"]
    offered = [item["bbox"] for strategy in plan["strategies"] for item in strategy["evidence"]]
    assert not any(box in drift_boxes for box in offered)

    # The new pass is its own experiment. It needs its own plan and approval.
    fake.answer = _near_seed
    created = client.post(f"/api/runs/{run_id}/recovery/experiments", json={"manual_targets": [used], "limits": {"max_requests": 4}}).json()
    assert len(fake.calls) == 1
    refused = client.post(f"/api/recovery/experiments/{created['id']}/live", json={"confirm": False, "plan_hash": "x"})
    assert refused.status_code == 400 and len(fake.calls) == 1
    second = _live(client, created).json()
    assert second["status"] == "completed", second["error"]
    assert len(fake.calls) == 2
    proposal = second["proposals"][0]
    assert all(row["reacquired"] for row in proposal["rows"])
    assert {row["lifecycle"] for row in proposal["rows"]} != {"lost"}
    assert any(row["outcome"] == "supported" for row in proposal["rows"])
    assert proposal["metrics"]["track"]["segments"][0]["status"] == "reacquired"
    assert not any(row["session_id"] == data["plan"]["requests"][0]["session_id"] for row in proposal["rows"])

    history = _object_history(client, run_id)
    resolved = next(item for item in history["plans"] if item["failure"]["reason"] == "large_spatial_jump")
    assert resolved["status"] == "resolved" and resolved["attempts"][0]["succeeded"] is True
    assert "separate segment" in resolved["explanation"]
    assert {item["physical_instance_id"] for item in history["segments"]} == {"track_1"}
    assert len({item["segment_id"] for item in history["segments"]}) >= 2
    assert history["stats"]["reacquisitions_succeeded"] == 1
    assert history["unresolved"], "The interval between the segments stays unresolved."

    # Accepting one result forks a revision, keeps the old one, and promotes no drifted box.
    session = SessionLocal()
    original = json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True)
    session.close()
    first_proposal = data["proposals"][0]
    decided = client.post(
        f"/api/recovery/experiments/{experiment['id']}/proposals/decide",
        json={"decision": "accepted", "proposal_ids": [first_proposal["id"]]},
    ).json()
    applied = decided["proposals"][0]["applied_revision_id"]
    session = SessionLocal()
    assert json.dumps(session.get(AnalysisRevision, revision_id).document_json, sort_keys=True) == original
    new = AnalysisDocument.model_validate(session.get(AnalysisRevision, applied).document_json)
    session.close()
    lost_times = {round(row["timestamp"], 3) for row in rows if row["lifecycle"] == "lost"}
    stored = [row for row in [*new.observations, *new.raw_observations] if row.source_model == "sam-3.1"]
    drifted = [row for row in stored if round(row.timestamp, 3) in lost_times]
    assert drifted and all(row.association_status == "conflict" and row.observation_role != "canonical" for row in drifted)
    assert any("after the track broke" in item.description for item in new.ambiguities)


def test_with_no_later_trusted_seed_the_plan_suggests_and_claims_nothing(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, times = _setup(client, tmp_path, with_seeds=False)
    fake = _install(monkeypatch, FakeSam(answer=_drifting))
    data = _live(client, _broken_experiment(client, run_id, times)).json()
    assert data["status"] == "completed", data["error"]
    plan = _object_history(client, run_id)["plans"][0]
    assert plan["status"] == "open"
    assert plan["recommended"] not in {"independent_observation", "human_seed"}
    assert plan["blocked_reason"]
    assert plan["proposed_target"]["reacquires"]["failure_id"] == plan["failure"]["failure_id"]
    if plan["recommended"] == "candidate_detection":
        assert plan["proposed_target"]["confirm_seed"] is True
    assert "separate segment" not in plan["explanation"]
    assert len(fake.calls) == 1


def test_a_failed_attempt_is_recorded_and_its_seed_is_not_offered_again(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_drifting))
    _live(client, _broken_experiment(client, run_id, times))
    plan = _object_history(client, run_id)["plans"][0]
    used = plan["proposed_target"]
    # The new pass goes wrong straight away. No box lands on its seed.
    fake.answer = lambda *_args: ""
    created = client.post(f"/api/runs/{run_id}/recovery/experiments", json={"manual_targets": [used], "limits": {"max_requests": 4}}).json()
    _live(client, created)
    calls = len(fake.calls)
    again = next(item for item in _object_history(client, run_id)["plans"] if item["failure"]["failure_id"] == plan["failure"]["failure_id"])
    assert again["status"] == "open" and again["needs_stronger_evidence"] is True
    assert again["attempts"][0]["succeeded"] is False
    offered = {item["seed_id"] for strategy in again["strategies"] for item in strategy["evidence"]}
    assert used["seed_id"] not in offered
    assert again["proposed_target"].get("seed_id") != used["seed_id"]
    assert len(fake.calls) == calls


def test_replay_reprocesses_a_stored_response_with_new_thresholds_and_no_calls(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, times = _setup(client, tmp_path)
    fake = _install(monkeypatch, FakeSam(answer=_drifting))
    experiment = _broken_experiment(client, run_id, times)
    live = _live(client, experiment).json()
    stored_key = live["plan"]["requests"][0]["key"]
    from app.services.recovery import load_response

    before = json.dumps(load_response(run_id, stored_key), sort_keys=True)
    calls = len(fake.calls)
    first = client.post(f"/api/recovery/experiments/{experiment['id']}/replay").json()
    second = client.post(f"/api/recovery/experiments/{experiment['id']}/replay").json()
    now = [item for item in first["proposals"] if item["current"]][0]
    later = [item for item in second["proposals"] if item["current"]][0]
    assert now["rows"] == later["rows"] and now["metrics"]["track"] == later["metrics"]["track"]
    assert second["summary"]["previous_totals"] == first["summary"]["totals"]
    # With a huge allowance the same response has no break, and the response did not change.
    relaxed = client.post(
        f"/api/recovery/experiments/{experiment['id']}/replay", json={"validation": {"jump_sizes": 100.0}}
    ).json()
    current = [item for item in relaxed["proposals"] if item["current"]][0]
    assert "large_spatial_jump" in {item["reason"] for item in now["metrics"]["track"]["failures"]}
    assert "large_spatial_jump" not in {item["reason"] for item in current["metrics"]["track"]["failures"]}
    assert relaxed["plan"]["requests"][0]["key"] == stored_key
    assert len(fake.calls) == calls
    assert json.dumps(load_response(run_id, stored_key), sort_keys=True) == before
    bad = client.post(f"/api/recovery/experiments/{experiment['id']}/replay", json={"validation": {"nope": 1}})
    assert bad.status_code == 400


def test_raw_boxes_keep_what_validation_decided(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, times = _setup(client, tmp_path)
    _install(monkeypatch, FakeSam(answer=_drifting))
    experiment = _broken_experiment(client, run_id, times)
    _live(client, experiment)
    request = client.get(f"/api/recovery/experiments/{experiment['id']}/diagnostics").json()["requests"][0]
    lost = [box for box in request["boxes"] if box["lifecycle"] == "lost"]
    assert lost and all(box["outcome"] == "conflicting" and box["segment_id"] == "" for box in lost)
    assert any(box["segment_id"] for box in request["boxes"])


def test_a_concept_pass_is_planned_budgeted_approved_and_binds_nothing(client, tmp_path, monkeypatch, no_language_model):
    run_id, _revision, times = _setup(client, tmp_path)
    answer = "<0f>7<|box;x1=1;y1=1;x2=40;y2=40;w=640;h=360;c=0.3|><|mask;data=1,1,!A|>"
    fake = _install(monkeypatch, FakeSam(answer=lambda *_args: answer))
    created = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={
            "manual_targets": [
                {
                    "object_local_id": "Object_1",
                    "time_start": times[2],
                    "time_end": times[2] + 0.5,
                    "strategy": "concept",
                    "reacquires": {"failure_id": "f1", "failure_time": times[2], "experiment_id": "e", "strategy": "concept_redetection"},
                }
            ],
        },
    ).json()
    plan = client.post(f"/api/recovery/experiments/{created['id']}/dry-run").json()["plan"]
    assert fake.calls == []
    spec = plan["requests"][0]
    assert spec["prompt_kind"] == "concept" and spec["prompt"] == "washer" and spec["bindings"] == {}
    assert plan["totals"]["billed_frames"] == spec["billed_frames"] >= 50
    data = _live(client, created).json()
    assert data["status"] == "completed", data["error"]
    assert fake.calls[0]["kind"] == "concept" and fake.calls[0]["direction"] is None
    assert data["proposals"] == []
    request = client.get(f"/api/recovery/experiments/{created['id']}/diagnostics").json()["requests"][0]
    assert request["verdict"] == "concept_candidates"
    assert request["boxes"][0]["status"] == "unassigned" and request["boxes"][0]["object_local_id"] == ""
    # A blocked limit stops it before any call.
    limited = client.post(
        f"/api/runs/{run_id}/recovery/experiments",
        json={"manual_targets": [{"object_local_id": "Object_1", "time_start": times[2], "time_end": times[2] + 2.0, "strategy": "concept"}], "limits": {"max_frames_per_request": 5}},
    ).json()
    blocked = client.post(f"/api/recovery/experiments/{limited['id']}/dry-run").json()["plan"]
    assert blocked["blocked"] and not blocked["requests"]
