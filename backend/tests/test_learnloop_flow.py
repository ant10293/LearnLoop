"""Teach, review, repeat, verify, through the product API with a stubbed Meta client.

Every response here is authored. These tests show the application logic works.
They say nothing about what a Meta model infers from real footage.
"""

import asyncio
import json
import subprocess

import pytest
from fastapi.testclient import TestClient


def _evidence(kind="DIRECTLY_OBSERVED", start=None, end=None, stamps=None):
    return {"classification": kind, "note": "authored", "frame_timestamps": stamps or [], "time_start": start, "time_end": end}


def _object(local_id, label, visual):
    return {
        "local_id": local_id,
        "label": label,
        "description": label,
        "visual_characteristics": visual,
        "initial_state": "separate",
        "final_state": "placed",
        "uncertain_identity": False,
    }


def _observed(local_id, timestamp, visibility="visible", kind="DIRECTLY_OBSERVED"):
    return {
        "object_local_id": local_id,
        "timestamp": timestamp,
        "visibility": visibility,
        "description": "authored",
        "evidence": _evidence(kind, stamps=[timestamp]),
    }


def _rel(subject, other, kind, start, end=9.5, classification="DIRECTLY_OBSERVED"):
    return {
        "subject_local_id": subject,
        "other_local_id": other,
        "kind": kind,
        "time_start": start,
        "time_end": end,
        "description": "authored",
        "evidence": _evidence(classification, start, end, [end]),
    }


def _action(local_id, objects, start, end, outcome="completed"):
    return {
        "local_id": local_id,
        "action_type": "attach",
        "description": f"Place {objects[0]}",
        "start": start,
        "end": end,
        "actor": "hand",
        "object_local_ids": objects,
        "initial_state": "separate",
        "final_state": "placed",
        "outcome": outcome,
        "uncertainty": 0.1,
        "evidence": _evidence(start=start, end=end, stamps=[end]),
    }


def _step(local_id, name, objects, action, start, end):
    return {
        "local_id": local_id,
        "name": name,
        "required_objects": objects,
        "preconditions": [],
        "expected_action": name,
        "completion_condition": f"{objects[0]} is on {objects[1]}",
        "observed_result": "authored",
        "action_local_ids": [action],
        "time_start": start,
        "time_end": end,
        "uncertainty": 0.2,
        "evidence": _evidence(start=start, end=end, stamps=[end]),
        "parallel_group": None,
    }


_OBJECTS = [
    _object("base", "gray base plate", "flat gray rectangle"),
    _object("red", "red cube", "large red cube"),
    _object("blue", "blue cube", "large blue cube"),
    _object("green", "green cube", "large green cube"),
]


def answer(**over):
    payload = {
        "summary": "authored",
        "objects": _OBJECTS,
        "observations": [_observed(item["local_id"], 9.5) for item in _OBJECTS],
        "transitions": [],
        "relationships": [
            _rel("red", "base", "attached", 3.0),
            _rel("blue", "base", "attached", 5.0),
            _rel("green", "base", "attached", 7.5),
        ],
        "actions": [
            _action("a1", ["red", "base"], 2.0, 3.0),
            _action("a2", ["blue", "base"], 4.0, 5.0),
            _action("a3", ["green", "base"], 7.0, 8.0),
        ],
        "workflow_steps": [],
        "dependencies": [],
        "ambiguities": [],
    }
    payload.update(over)
    return json.dumps(payload)


DEMONSTRATION = answer(
    workflow_steps=[
        _step("s1", "Place red cube", ["red", "base"], "a1", 2.0, 3.5),
        _step("s2", "Place blue cube", ["blue", "base"], "a2", 4.0, 5.5),
        _step("s3", "Place green cube", ["green", "base"], "a3", 7.0, 8.5),
    ]
)
CORRECT = answer()
_TWO = [_rel("red", "base", "attached", 3.0), _rel("blue", "base", "attached", 5.0)]
_TWO_ACTIONS = [_action("a1", ["red", "base"], 2.0, 3.0), _action("a2", ["blue", "base"], 4.0, 5.0)]
_ABSENT = [_observed("base", 9.5), _observed("red", 9.5), _observed("blue", 9.5), _observed("green", 9.5, "absent")]
OMITTED = answer(relationships=_TWO, actions=_TWO_ACTIONS, observations=_ABSENT)
MISPLACED = answer(
    relationships=[*_TWO, _rel("green", "base", "left_of", 7.5)],
    actions=[*_TWO_ACTIONS, _action("a3", ["green", "base"], 7.0, 8.0)],
)
HIDDEN = answer(
    relationships=_TWO,
    actions=_TWO_ACTIONS,
    observations=[_observed("base", 9.5), _observed("red", 9.5), _observed("blue", 9.5), _observed("green", 9.5, "occluded", "UNKNOWN")],
)


class StubMeta:
    id = "meta"

    def __init__(self, text=DEMONSTRATION, incomplete=""):
        self.text, self.incomplete, self.calls = text, incomplete, []

    async def generate(self, secret, request):
        from app.providers.types import ProviderResponse, Usage

        self.calls.append(request)
        return ProviderResponse(
            text=self.text,
            raw={"status": "incomplete" if self.incomplete else "completed"},
            usage=Usage(1000, 500, 0.0, False),
            request_id=f"req_{len(self.calls)}",
            model_id=request.model_id,
            incomplete_reason=self.incomplete,
        )


def _video(path, color):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=10:r=10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_API_KEY", "stub-key")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    settings.worker_enabled = False
    settings.auto_seed = False
    settings.ingest_inline = True
    monkeypatch.setattr(settings, "live_inference", True)
    rebind_engine(settings.sqlalchemy_url)
    from app.main import create_app, init_database
    from app.pipeline import guided
    from app.providers import model_list, registry

    monkeypatch.setattr(guided, "REPLAY_DIR", tmp_path / "no_replay")
    monkeypatch.setattr(model_list, "live_options", lambda *args, **kwargs: None)
    stub = StubMeta()
    monkeypatch.setattr(registry, "get_provider", lambda provider_id, session=None: stub)
    init_database()
    with TestClient(create_app()) as client:
        yield client, stub, tmp_path


def _run_job(run_id):
    from app.db import SessionLocal
    from app.models.runs import Job
    from app.pipeline.runner import execute_job
    from sqlalchemy import select

    session = SessionLocal()
    try:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        job.status, job.attempt = "running", job.attempt + 1
        session.commit()
        asyncio.run(execute_job(session, job))
    finally:
        session.close()


def _upload(client, procedure, path, role):
    with path.open("rb") as handle:
        response = client.post(
            f"/api/learnloop/procedures/{procedure}/recordings",
            data={"role": role},
            files={"file": (path.name, handle, "video/mp4")},
        )
    assert response.status_code == 200, response.text
    return response.json()["video"]["id"]


def _analyze(client, video_id):
    response = client.post(f"/api/learnloop/recordings/{video_id}/analyze", json={"allow_live": True})
    assert response.status_code == 200, response.text
    run_id = response.json()["id"]
    _run_job(run_id)
    return client.get(f"/api/learnloop/runs/{run_id}").json()


def _accepted(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube assembly"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    template = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    assert template["review_state"] == "ai_draft"
    accepted = client.patch(f"/api/workflow-templates/{template['id']}", json={"acceptance_status": "accepted"}).json()
    return procedure, accepted


def _check(lab, procedure, template, text, color):
    client, stub, tmp = lab
    stub.text = text
    run = _analyze(client, _upload(client, procedure, _video(tmp / f"{color}.mp4", color), "execution"))
    assert run["status"] == "completed", run["error"]
    created = client.post(f"/api/learnloop/templates/{template['id']}/verify", json={"run_id": run["id"]})
    assert created.status_code == 200, created.text
    return created.json()


def _verdicts(report):
    return {step["step_id"]: step["verdict"] for step in report["report"]["steps"]}


def test_full_flow_passes_a_correct_attempt_and_exports(lab):
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, CORRECT, "blue")
    assert report["overall"] == "PASS"
    assert set(_verdicts(report).values()) == {"PASS"}
    step = report["report"]["steps"][0]
    assert step["basis"] == "model_evidence" and step["evidence"] and step["evidence"][0]["timestamp"] is not None
    client = lab[0]
    exported = client.get(f"/api/verifications/{report['id']}/export").json()
    assert exported["template_version"] == template["version"]
    assert exported["verification"]["execution_revision_id"]
    summary = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert summary["latest_outcome"] == "PASS" and summary["step_count"] == 3 and summary["review_state"] == "accepted"


def test_visible_omission_fails_as_missing_component(lab):
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, OMITTED, "green")
    steps = {step["step_id"]: step for step in report["report"]["steps"]}
    assert report["overall"] == "FAIL"
    assert steps["s3"]["verdict"] == "FAIL" and steps["s3"]["error_category"] == "MISSING_COMPONENT"
    assert steps["s1"]["verdict"] == "PASS" and steps["s2"]["verdict"] == "PASS"


def test_positively_observed_misplacement_fails(lab):
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, MISPLACED, "yellow")
    step = {item["step_id"]: item for item in report["report"]["steps"]}["s3"]
    assert step["verdict"] == "FAIL" and step["error_category"] == "INCORRECT_PLACEMENT"


def test_hidden_component_cannot_be_verified_and_is_not_called_missing(lab):
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, HIDDEN, "white")
    step = {item["step_id"]: item for item in report["report"]["steps"]}["s3"]
    assert step["verdict"] == "CANNOT_VERIFY"
    assert report["overall"] == "INCONCLUSIVE"


def test_human_observation_is_evidence_with_provenance_and_keeps_the_machine_reading(lab):
    procedure, template = _accepted(lab)
    client = lab[0]
    report = _check(lab, procedure, template, HIDDEN, "cyan")
    too_early = client.post(
        f"/api/verifications/{report['id']}/steps/s3/observations",
        json={"answer": "installed", "timestamp": 1.0, "reviewer": "A. Reviewer"},
    ).json()
    assert _verdicts(too_early)["s3"] == "CANNOT_VERIFY"
    confirmed = client.post(
        f"/api/verifications/{report['id']}/steps/s3/observations",
        json={"answer": "installed", "timestamp": 9.5, "reviewer": "A. Reviewer", "note": "Visible after the hand moved."},
    ).json()
    step = {item["step_id"]: item for item in confirmed["report"]["steps"]}["s3"]
    assert step["verdict"] == "PASS" and step["basis"] == "human_confirmed" and step["machine_verdict"] == "CANNOT_VERIFY"
    assert step["evidence"][-1]["provenance"] == "HUMAN_ANNOTATION"
    assert confirmed["report"]["human_assisted"] is True
    assert confirmed["overall"] == "PASS"
    assert [item["reviewer"] for item in confirmed["human_observations"]] == ["A. Reviewer", "A. Reviewer"]
    exported = client.get(f"/api/verifications/{report['id']}/export").json()
    machine = {item["step_id"]: item for item in exported["machine_report"]["steps"]}["s3"]
    assert machine["verdict"] == "CANNOT_VERIFY"
    answer_needs_reviewer = client.post(
        f"/api/verifications/{report['id']}/steps/s3/observations",
        json={"answer": "installed", "timestamp": 9.5, "reviewer": " "},
    )
    assert answer_needs_reviewer.status_code == 400


def test_edits_make_new_versions_and_leave_the_model_version_unchanged(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    draft = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    original = json.dumps(draft["document"], sort_keys=True)
    edited = client.patch(
        f"/api/workflow-templates/{draft['id']}",
        json={
            "steps": [
                {"step_id": "s1", "name": "Seat the red cube", "expected_action": "Press it down"},
                {"step_id": "s2", "removed": True},
                {
                    "step_id": "s3",
                    "completion_conditions": [
                        {"subject_role": "role:green", "target_role": "role:base", "relation": "supported_by"}
                    ],
                },
            ],
            "roles": [{"role_id": "role:red", "label": "crimson cube"}],
        },
    ).json()
    assert edited["version"] == 2 and edited["review_state"] == "human_reviewed"
    steps = {step["step_id"]: step for step in edited["document"]["steps"]}
    assert steps["s1"]["human_edited"] == ["name", "expected_action"]
    assert steps["s2"]["removed"] is True
    assert steps["s3"]["completion_conditions"][0]["origin"] == "human"
    assert edited["document"]["review_log"]
    assert json.dumps(client.get(f"/api/workflow-templates/{draft['id']}").json()["document"], sort_keys=True) == original
    bad = client.patch(
        f"/api/workflow-templates/{edited['id']}",
        json={"steps": [{"step_id": "s1", "completion_conditions": [{"subject_role": "role:none", "relation": "above"}]}]},
    )
    assert bad.status_code == 400
    accepted = client.patch(f"/api/workflow-templates/{edited['id']}", json={"acceptance_status": "accepted"}).json()
    report = _check(lab, procedure, accepted, CORRECT, "blue")
    assert "s2" not in _verdicts(report) and report["report"]["removed_steps"] == ["s2"]
    reedited = client.patch(f"/api/workflow-templates/{accepted['id']}", json={"steps": [{"step_id": "s1", "name": "Again"}]}).json()
    assert reedited["acceptance_status"] == "draft"
    refused = client.post(f"/api/learnloop/templates/{reedited['id']}/verify", json={"run_id": report["execution_run_id"]})
    assert refused.status_code == 409


def test_cached_replay_makes_no_provider_call_and_is_labeled(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    first = _analyze(client, video)
    assert first["source"] == "live" and len(stub.calls) == 1 and first["provider_requests"] == 1
    replay = client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": False}).json()
    _run_job(replay["id"])
    replay = client.get(f"/api/learnloop/runs/{replay['id']}").json()
    assert replay["status"] == "completed" and replay["source"] == "cached" and replay["provider_requests"] == 0
    assert len(stub.calls) == 1


def test_same_filename_different_bytes_does_not_replay(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    first = _upload(client, procedure, _video(tmp / "a" / "clip.mp4", "red"), "demonstration")
    assert _analyze(client, first)["source"] == "live"
    second = _upload(client, procedure, _video(tmp / "b" / "clip.mp4", "blue"), "demonstration")
    refused = client.post(f"/api/learnloop/recordings/{second}/analyze", json={"allow_live": False})
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "live_call_required"
    assert refused.json()["detail"]["plan"]["max_requests"] == 1
    assert len(stub.calls) == 1


def test_live_call_needs_approval_and_the_server_gate(lab, monkeypatch):
    from app.config import settings

    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    assert client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": False}).status_code == 409
    monkeypatch.setattr(settings, "live_inference", False)
    response = client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": True})
    assert response.status_code == 409 and response.json()["detail"]["plan"]["live_enabled"] is False
    assert stub.calls == []


@pytest.mark.parametrize(
    ("text", "incomplete", "needle"),
    [
        (DEMONSTRATION[:200], "max_output_tokens", "stopped before it finished"),
        ("{not json", "", "not valid JSON"),
        ('{"summary": "x"}', "", "broke"),
        (answer(objects=[], observations=[], relationships=[], actions=[]), "", "names no objects"),
        (answer(), "", "no procedure steps"),
    ],
)
def test_a_bad_answer_fails_visibly_once_and_is_not_cached(lab, text, incomplete, needle):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRun, ModelRequestRow, ModelResponseRow

    client, stub, tmp = lab
    stub.text, stub.incomplete = text, incomplete
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    run = _analyze(client, video)
    assert run["status"] == "failed" and needle in run["error"]
    assert len(stub.calls) == 1
    session = SessionLocal()
    row = session.query(ModelRequestRow).filter_by(run_id=run["id"]).one()
    assert session.query(ModelResponseRow).filter_by(request_id=row.id).one().text == text
    assert session.get(AnalysisRun, run["id"]).active_revision_id is None
    session.close()
    assert client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": False}).status_code == 409


def test_the_original_answer_stays_on_its_response_row_after_correction(lab):
    from app.db import SessionLocal
    from app.models.runs import ModelRequestRow, ModelResponseRow

    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    session = SessionLocal()
    request = session.query(ModelRequestRow).first()
    assert session.query(ModelResponseRow).filter_by(request_id=request.id).one().text == DEMONSTRATION
    session.close()


def test_only_meta_is_registered_and_other_providers_are_refused():
    from app.providers.registry import get_provider, provider_ids
    from app.providers.types import ProviderError

    assert "openai" not in provider_ids() and "anthropic" not in provider_ids() and "gemini" not in provider_ids()
    with pytest.raises(ProviderError):
        get_provider("openai")


def test_a_guided_run_uses_one_request_without_sam_or_tracking(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert len(stub.calls) == 1
    request = stub.calls[0]
    assert request.modality == "native_video" and request.model_id == "muse-wonder-1.0"
    assert request.max_output_tokens == 24_000 and request.schema_name == "learnloop_guided_task"
    for stage in ("spatial_tracking", "segmentation", "evidence_fusion", "reinspection", "transcription"):
        assert run["stages"][stage] == "skipped"


def test_the_meta_key_is_read_from_dotenv(tmp_path, monkeypatch):
    from app.services import credentials

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "REPO_ROOT", tmp_path)
    assert credentials.env_secret("meta") == ""
    (tmp_path / ".env").write_text("MODEL_API_KEY=from-dotenv\n")
    assert credentials.env_secret("meta") == "from-dotenv"
