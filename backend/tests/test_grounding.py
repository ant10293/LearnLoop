"""Resolver invariants, one-to-one association, and semantic call counts.

The pipeline cases use a fake provider that returns an authored document. No
provider request leaves the machine.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from app.perception.ground import associate_objects
from app.perception.resolve import resolve_document
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    InstanceRecord,
    ObjectPrediction,
    ObservationPrediction,
    PerceptionRecord,
)
from app.schemas.enums import EvidenceClass, Provenance, Visibility

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")


def _evidence(timestamp: float) -> Evidence:
    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.AI_PREDICTION,
        frame_timestamps=[timestamp],
    )


def _row(object_id: str, timestamp: float, box, **fields) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]) if box else None,
        evidence=_evidence(timestamp),
        **fields,
    )


def _track(ref: str, times, box, detector: str = "manual", origin: str = "detected") -> list[ObservationPrediction]:
    return [
        _row(
            "",
            float(t),
            box,
            track_ref=ref,
            geometry_origin=origin,
            geometry_authority=detector,
            detection_support="none" if origin == "optical_flow" else ("proposal" if detector == "motion" else "detected"),
            temporal_support="flow_only" if origin == "optical_flow" else "repeated",
        )
        for t in times
    ]


def _spatial(*tracks: list[ObservationPrediction]) -> AnalysisDocument:
    rows = [row for track in tracks for row in track]
    refs = sorted({row.track_ref for row in rows})
    instances = [
        InstanceRecord(
            instance_id=ref,
            status="tracked",
            first_seen=min(row.timestamp for row in rows if row.track_ref == ref),
            last_seen=max(row.timestamp for row in rows if row.track_ref == ref),
        )
        for ref in refs
    ]
    return AnalysisDocument(
        observations=rows,
        perception=PerceptionRecord(strategy="test", detector_id="manual", instances=instances),
    )


LEFT = (0.10, 0.30, 0.15, 0.15)
RIGHT = (0.60, 0.30, 0.15, 0.15)


def _two_blocks() -> AnalysisDocument:
    return AnalysisDocument(
        objects=[
            ObjectPrediction(local_id="Object_1", label="gray block"),
            ObjectPrediction(local_id="Object_2", label="gray block"),
        ],
        observations=[
            *[_row("Object_1", float(t), LEFT) for t in range(4)],
            *[_row("Object_2", float(t), RIGHT) for t in range(4)],
        ],
    )


def _ground(document: AnalysisDocument, spatial: AnalysisDocument) -> AnalysisDocument:
    bound = associate_objects(document.model_copy(update={"perception": spatial.perception}), spatial)
    return resolve_document(bound, spatial)


def test_two_identical_labels_remain_two_instances():
    spatial = _spatial(_track("inst-a", range(4), LEFT), _track("inst-b", range(4), RIGHT))
    resolved = _ground(_two_blocks(), spatial)
    ids = {obj.local_id: obj.instance_id for obj in resolved.objects}
    assert ids == {"Object_1": "inst-a", "Object_2": "inst-b"}
    canonical = [row for row in resolved.observations if row.observation_role == "canonical"]
    assert len(canonical) == 8
    # One display row per object per time.
    keys = [(row.object_local_id, row.timestamp) for row in resolved.observations]
    assert len(keys) == len(set(keys))


def test_a_vlm_estimate_never_overwrites_a_canonical_row():
    spatial = _spatial(_track("inst-a", range(4), LEFT), _track("inst-b", range(4), RIGHT))
    document = _two_blocks()
    off = _row("Object_1", 2.1, (0.12, 0.31, 0.15, 0.15))
    document = document.model_copy(update={"observations": [*document.observations, off]})
    resolved = _ground(document, spatial)
    near = [row for row in resolved.observations if row.object_local_id == "Object_1" and abs(row.timestamp - 2.05) < 0.1]
    assert [row.observation_role for row in near] == ["canonical"]
    assert any(row.timestamp == pytest.approx(2.1) for row in resolved.raw_observations)


def test_one_shared_time_is_weak_and_binds_nothing():
    spatial = _spatial(_track("inst-a", [0], LEFT), _track("inst-b", range(10, 14), RIGHT))
    document = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="gray block")],
        observations=[_row("Object_1", 0.0, LEFT)],
    )
    bound = associate_objects(document.model_copy(update={"perception": spatial.perception}), spatial)
    obj = bound.objects[0]
    assert obj.instance_id is None
    assert obj.association_status == "weak"
    assert obj.uncertain_identity is True


def test_a_label_change_does_not_change_the_instance():
    spatial = _spatial(_track("inst-a", range(4), LEFT), _track("inst-b", range(4), RIGHT))
    first = associate_objects(_two_blocks().model_copy(update={"perception": spatial.perception}), spatial)
    renamed = first.model_copy(
        update={"objects": [first.objects[0].model_copy(update={"label": "red brick"}), first.objects[1]]}
    )
    again = associate_objects(renamed, spatial)
    assert again.objects[0].instance_id == "inst-a"
    assert again.objects[0].association_status == "associated"


def test_a_cross_cut_match_is_a_hypothesis_and_not_a_merge():
    spatial = _spatial(_track("inst-a", range(0, 4), LEFT), _track("inst-c", range(10, 14), LEFT))
    document = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="gray block")],
        observations=[_row("Object_1", float(t), LEFT) for t in [*range(0, 4), *range(10, 14)]],
    )
    bound = associate_objects(document.model_copy(update={"perception": spatial.perception}), spatial)
    assert bound.objects[0].instance_id in {"inst-a", "inst-c"}
    hypotheses = [item for item in bound.perception.hypotheses if item.hypothesis_id.startswith("ground-")]
    assert len(hypotheses) == 1
    assert hypotheses[0].decision == "proposed"
    assert {hypotheses[0].subject_instance_id, hypotheses[0].candidate_instance_id} == {"inst-a", "inst-c"}


def test_a_proposal_only_track_stays_provisional():
    spatial = _spatial(_track("inst-m", range(4), LEFT, detector="motion"))
    document = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="block", instance_id="inst-m")],
        perception=spatial.perception,
    )
    resolved = resolve_document(document, spatial)
    assert resolved.observations
    assert all(row.observation_role == "raw" for row in resolved.observations)
    assert all(row.detection_support == "proposal" for row in resolved.observations)
    assert resolved.perception.diagnostics["resolution"]["canonical"] == 0


def test_a_flow_hold_is_not_a_detection_and_not_canonical():
    spatial = _spatial(
        _track("inst-a", range(3), LEFT),
        _track("inst-a", [3, 4], LEFT, origin="optical_flow"),
    )
    document = AnalysisDocument(objects=[ObjectPrediction(local_id="Object_1", label="block", instance_id="inst-a")])
    resolved = resolve_document(document, spatial)
    pool = [*resolved.observations, *resolved.raw_observations]
    flow = [row for row in pool if row.geometry_origin == "optical_flow"]
    assert flow and all(row.detection_support == "none" and row.observation_role != "canonical" for row in flow)


def test_an_unobserved_gap_adds_no_absence_row():
    spatial = _spatial(_track("inst-a", [0, 1, 6, 7], LEFT))
    document = AnalysisDocument(objects=[ObjectPrediction(local_id="Object_1", label="block", instance_id="inst-a")])
    resolved = resolve_document(document, spatial)
    times = sorted(row.timestamp for row in resolved.observations)
    assert times == [0.0, 1.0, 6.0, 7.0]
    assert all(row.visibility != Visibility.ABSENT for row in resolved.observations)


def test_disagreeing_measurements_become_a_conflict_with_both_rows_kept():
    sam = _row(
        "Object_1",
        1.0,
        RIGHT,
        source_model="sam-3.1",
        track_ref="sam-1",
        association_support="associated",
        detection_support="mask",
        mask_ref="m1",
    )
    spatial = _spatial(_track("inst-a", range(3), LEFT))
    document = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="block", instance_id="inst-a")],
        observations=[sam],
    )
    resolved = resolve_document(document, spatial)
    at_one = [row for row in [*resolved.observations, *resolved.raw_observations] if row.timestamp == 1.0]
    assert len(at_one) == 2
    assert all(row.observation_role != "canonical" for row in at_one)
    assert any(item.local_id == "resolve_conflict_Object_1" for item in resolved.ambiguities)


def test_resolution_is_idempotent_and_a_legacy_document_still_displays():
    legacy = AnalysisDocument.model_validate(
        {
            "objects": [{"local_id": "Object_1", "label": "block"}],
            "observations": [
                {
                    "object_local_id": "Object_1",
                    "timestamp": 1.0,
                    "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    "evidence": {"classification": "DIRECTLY_OBSERVED", "provenance": "AI_PREDICTION"},
                }
            ],
        }
    )
    assert legacy.observations[0].observation_role == ""
    once = resolve_document(legacy)
    assert [row.timestamp for row in once.observations] == [1.0]
    assert once.observations[0].geometry_origin == "vlm_estimated"
    twice = resolve_document(once)
    assert twice.model_dump() == once.model_dump()

    spatial = _spatial(_track("inst-a", range(4), LEFT), _track("inst-b", range(4), RIGHT))
    grounded = _ground(_two_blocks(), spatial)
    assert resolve_document(grounded, spatial).model_dump() == grounded.model_dump()


# Pipeline call counts with a fake provider.


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
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database

    init_database()
    with TestClient(create_app()) as test_client:
        yield test_client


class _Authored:
    """Returns the same authored document for every request and records the requests."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def generate(self, _secret, _request):
        from app.providers.types import ProviderResponse, Usage

        self.calls += 1
        return ProviderResponse(text=self.text, raw={}, usage=Usage(), latency_ms=1)


def _install(monkeypatch, provider: _Authored) -> None:
    from app.pipeline import infer, locate_pass

    for module in (infer, locate_pass):
        monkeypatch.setattr(module, "needs_secret", lambda *_args, **_kwargs: False)
        monkeypatch.setattr(module, "get_provider", lambda *_args, **_kwargs: provider)
        monkeypatch.setattr(module, "model_for_call", lambda _session, run: (run.model_id, run.model_id))


def _upload(client, tmp_path: Path) -> dict:
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario

    project = client.post("/api/projects", json={"name": "Grounding"}).json()
    clip = tmp_path / "fixed.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]())[:60], clip)
    with clip.open("rb") as handle:
        uploaded = client.post(
            "/api/videos",
            data={"project_id": project["id"]},
            files={"file": ("fixed.mp4", handle, "video/mp4")},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def _run(client, video: dict, provider_id: str, model_id: str, mode: str) -> dict:
    from app.db import SessionLocal
    from app.models.runs import Job
    from app.pipeline.runner import execute_job
    from sqlalchemy import select

    created = client.post(
        "/api/runs",
        json={
            "name": f"{provider_id} {mode}",
            "video_id": video["id"],
            "provider_id": provider_id,
            "model_id": model_id,
            "modality": "sampled_frames",
            "research_mode": "fixed_camera",
            "sampling_preset": "detailed",
            "analysis_pass": "single",
            "segmentation_mode": "boxes_only",
            "semantic_mode": mode,
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
    return client.get(f"/api/runs/{run_id}").json()


def _requests(run_id: str) -> list[str]:
    from app.db import SessionLocal
    from app.models.runs import ModelRequestRow

    session = SessionLocal()
    stages = [row.stage for row in session.query(ModelRequestRow).filter(ModelRequestRow.run_id == run_id)]
    session.close()
    return stages


def _authored_text() -> str:
    from app.demo.scenarios import SCENARIOS, build_document

    return json.dumps(build_document(SCENARIOS["fixed_camera"]()).model_dump(mode="json"))


def test_combined_adds_no_call_and_grounded_reports_its_second(client, tmp_path, monkeypatch):
    provider = _Authored(_authored_text())
    _install(monkeypatch, provider)
    video = _upload(client, tmp_path)

    combined = _run(client, video, "openai", "gpt-4.1", "combined")
    assert combined["run"]["status"] == "completed", combined["run"]["error"]
    stages = _requests(combined["run"]["id"])
    assert set(stages) == {"scene_analysis"}
    assert combined["coverage"]["semantic_mode"] == "combined"
    assert combined["coverage"]["semantic_calls"] == 1
    fusion = next(item for item in combined["stages"] if item["stage"] == "evidence_fusion")
    assert fusion["checkpoint"]["model_calls"] == 0

    grounded = _run(client, video, "openai", "gpt-4.1", "grounded")
    assert grounded["run"]["status"] == "completed", grounded["run"]["error"]
    stages = _requests(grounded["run"]["id"])
    assert set(stages) == {"scene_analysis", "workflow"}
    assert grounded["coverage"]["semantic_mode"] == "grounded"
    assert grounded["coverage"]["semantic_calls"] == 2
    skipped = {item["stage"] for item in grounded["stages"] if item["status"] == "skipped"}
    assert {"action_analysis", "state_tracking", "ambiguity"} <= skipped


def test_perception_does_not_depend_on_the_provider(client, tmp_path, monkeypatch):
    provider = _Authored(_authored_text())
    _install(monkeypatch, provider)
    video = _upload(client, tmp_path)
    first = _run(client, video, "openai", "gpt-4.1", "combined")
    second = _run(client, video, "anthropic", "claude-sonnet-4-5", "combined")
    assert first["run"]["status"] == second["run"]["status"] == "completed"

    def geometry(bundle: dict) -> list[tuple]:
        return sorted(
            (
                row["object_local_id"],
                round(row["timestamp"], 3),
                row["observation_role"],
                row["geometry_origin"],
                json.dumps(row["bbox"], sort_keys=True),
            )
            for row in bundle["observations"]
        )

    assert geometry(first) == geometry(second)
    assert first["perception"]["instances"] == second["perception"]["instances"]
