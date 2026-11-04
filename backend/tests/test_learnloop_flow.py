"""Teach, review, repeat, verify, through the product API with a stubbed Meta client.

Every response here is authored. These tests show the application logic works.
They say nothing about what a Meta model infers from real footage.
"""

import asyncio
import json
import subprocess
from types import SimpleNamespace

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
        self.locate_text = '{"objects": []}'
        self.later = lambda request: '{"objects": []}'

    def _reply(self, request):
        if request.schema_name == "learnloop_guided_locate":
            return self.locate_text
        if request.schema_name == "learnloop_guided_locate_later":
            return self.later(request)
        return self.text

    async def generate(self, secret, request):
        from app.providers.types import ProviderResponse, Usage

        self.calls.append(request)
        return ProviderResponse(
            text=self._reply(request),
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
    monkeypatch.setattr(settings, "object_views", False)
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
        job = session.scalar(select(Job).where(Job.run_id == run_id, Job.status == "pending").order_by(Job.created_at))
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
    unnamed = client.post(
        f"/api/verifications/{report['id']}/steps/s3/observations",
        json={"answer": "installed", "timestamp": 9.5},
    )
    assert unnamed.status_code == 200
    assert unnamed.json()["human_observations"][-1]["reviewer"] == ""


def test_deleting_a_learned_procedure_removes_its_versions_reports_and_demonstration(lab):
    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, CORRECT, "blue")
    edited = client.patch(f"/api/workflow-templates/{template['id']}", json={"name": "Renamed"}).json()

    assert client.delete(f"/api/learnloop/templates/{edited['id']}").status_code == 200

    summary = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert summary["templates"] == []
    assert summary["verifications"] == []
    assert [item["role"] for item in summary["recordings"]] == ["execution"]
    assert client.get(f"/api/workflow-templates/{template['id']}").status_code == 404
    assert client.get(f"/api/verifications/{report['id']}").status_code == 404
    assert client.delete(f"/api/learnloop/templates/{edited['id']}").status_code == 404


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


def test_fresh_analysis_skips_stored_answers_after_approval(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    calls = []

    class CountingSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            calls.append(kwargs["propagation_direction"])
            text = "\n".join(_sam_line(frame, [(0, 10)]) for frame in range(4))
            return SimpleNamespace(text=text, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", CountingSam)
    first = json.loads(stub.text)["objects"][0]["local_id"]
    stub.locate_text = json.dumps({"objects": [{"local_id": first, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]})
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    assert client.get(f"/api/learnloop/recordings/{video}").json()["analyzed_before"] is False
    assert _analyze(client, video)["source"] == "live"
    # The same bytes in another procedure count as analyzed.
    other = client.post("/api/learnloop/procedures", json={"name": "Copy"}).json()["id"]
    (tmp / "copy").mkdir()
    copy = _upload(client, other, _video(tmp / "copy" / "demo.mp4", "red"), "demonstration")
    assert client.get(f"/api/learnloop/recordings/{copy}").json()["analyzed_before"] is True
    sent, sam = len(stub.calls), len(calls)
    assert sam == 1

    # Fresh never replays, so it asks first even though an answer is stored.
    refused = client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": False, "fresh": True})
    assert refused.status_code == 409 and refused.json()["detail"]["plan"]["fresh"] is True
    assert len(stub.calls) == sent
    again = client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": True, "fresh": True}).json()
    _run_job(again["id"])
    again = client.get(f"/api/learnloop/runs/{again['id']}").json()
    assert again["status"] == "completed" and again["source"] == "live" and again["provider_requests"] >= 1
    assert len(stub.calls) == 2 * sent and len(calls) == 2 * sam


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
    assert request.max_output_tokens == 65_536 and request.schema_name == "learnloop_guided_task"
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


def test_a_recording_can_be_referenced_and_removed(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    path = _video(tmp / "kept.mp4", "red")
    made = client.post(f"/api/learnloop/procedures/{procedure}/recordings/reference", json={"path": str(path), "role": "demonstration"})
    assert made.status_code == 200, made.text
    video = made.json()["video"]["id"]
    assert made.json()["video"]["thumbnail_url"]
    assert client.get(f"/api/learnloop/videos/{video}/frame?t=1").status_code == 200
    assert client.delete(f"/api/learnloop/recordings/{video}").status_code == 200
    assert path.exists()
    assert client.get(f"/api/learnloop/procedures/{procedure}").json()["recordings"] == []


def test_removing_an_attempt_removes_its_report_but_the_source_recording_stays(lab):
    client = lab[0]
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, CORRECT, "blue")
    assert client.delete(f"/api/learnloop/recordings/{report['execution_video_id']}").status_code == 200
    detail = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert detail["verifications"] == [] and detail["execution_count"] == 0
    assert client.delete(f"/api/learnloop/recordings/{template['source_video_id']}").status_code == 409


def test_frames_come_out_of_a_limited_range_playback_file(tmp_path):
    from app.video.frames import extract_frame, make_proxy
    from app.video.probe import presentation_timestamps

    source = _video(tmp_path / "source.mp4", "red")
    proxy = tmp_path / "proxy.mp4"
    make_proxy(source, proxy)
    dest = tmp_path / "frame.jpg"
    extract_frame(proxy, 1.0, dest)
    assert dest.stat().st_size > 0
    times = presentation_timestamps(source)
    assert len(times) == 100 and times == sorted(times)


def test_learning_twice_from_one_analysis_gives_one_procedure(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    first = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    second = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    assert first["id"] == second["id"]
    assert len(client.get(f"/api/learnloop/procedures/{procedure}").json()["templates"]) == 1


def test_analyzing_a_recording_again_learns_again_and_the_newest_acceptance_wins(lab):
    client, stub, tmp = lab
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    first_run = _analyze(client, video)
    first = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": first_run["id"]}).json()
    client.patch(f"/api/workflow-templates/{first['id']}", json={"name": "Edited"})
    client.patch(f"/api/workflow-templates/{first['id']}", json={"name": "Edited again", "acceptance_status": "accepted"})
    from app.pipeline import guided

    for path in guided.data_path("inference_cache").glob("*.json"):
        path.unlink()
    second_run = _analyze(client, video)
    assert second_run["id"] != first_run["id"]
    second = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": second_run["id"]}).json()
    assert second["id"] != first["id"] and second["source_run_id"] == second_run["id"]
    page = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert page["template"]["id"] == second["id"]
    accepted = client.patch(f"/api/workflow-templates/{second['id']}", json={"acceptance_status": "accepted"}).json()

    from app.api.product import _accepted_template
    from app.db import SessionLocal

    with SessionLocal() as session:
        assert _accepted_template(session, procedure).id == accepted["id"]


def _sam_line(frame: int, boxes: list[tuple[int, int]]) -> str:
    body = "".join(
        f"{track}<|box;x1={left};y1=10;x2={left + 50};y2=60;w=160;h=120;c=0.9|><|mask;data=|>" for track, left in boxes
    )
    return f"<{frame}f>{body}"


def test_a_demonstration_follows_every_box_in_one_sam_request_and_replays_it(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    calls = []
    answer_text = "\n".join(_sam_line(frame, [(0, 10), (2, 100)]) for frame in range(4))

    class FakeSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(text=answer_text, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    ids = [item["local_id"] for item in json.loads(stub.text)["objects"]]
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    stub.locate_text = json.dumps({"objects": [{"local_id": local_id, "bbox": box} for local_id in ids[:3]] + [{"local_id": ids[3], "bbox": None}]})
    procedure = client.post("/api/learnloop/procedures", json={"name": "Views"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    run = _analyze(client, video)
    assert run["status"] == "completed", run["error"]
    assert run["stages"]["segmentation"] == "completed"
    assert len(calls) == 1
    assert [request.schema_name for request in stub.calls].count("learnloop_guided_locate") == 1
    assert calls[0][0].count("<|box;") == 3 and calls[0][1]["prompt_kind"] == "spatial"
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert views["objects"][ids[0]]["seen_cells"] and views["objects"][ids[2]]["seen_cells"]
    assert not views["objects"][ids[1]]["seen_cells"] and not views["objects"][ids[3]]["seen_cells"]
    assert client.get(views["objects"][ids[0]]["thumbnail_url"]).status_code == 200
    assert [row["time"] for row in views["objects"][ids[0]]["track"]] == [round(index / object_views.SAM_FPS, 3) for index in range(4)]
    assert {cell["confidence"] for cell in views["objects"][ids[0]]["cells"] if cell["url"]} == {0.9}
    assert views["objects"][ids[1]]["track"] == []
    assert client.get(f"/api/learnloop/runs/{run['id']}/objects/manifest.json").status_code == 404

    again = _analyze(client, video)
    assert len(calls) == 1
    replayed = client.get(f"/api/learnloop/runs/{again['id']}/objects").json()
    assert [request.schema_name for request in stub.calls].count("learnloop_guided_locate") == 1
    assert replayed["replayed_requests"] == 1 and replayed["locate_replayed"] == 2 and replayed["objects"][ids[0]]["seen_cells"]


def test_an_object_missing_from_the_first_frame_is_found_on_a_later_still(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    prompts = []

    class FakeSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            prompts.append((prompt, kwargs["propagation_direction"], media))
            # Each request answers for every track, so a box from the other request must be ignored.
            text = "\n".join(_sam_line(frame, [(0, 10), (1, 60)]) for frame in range(4))
            return SimpleNamespace(text=text, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    ids = [item["local_id"] for item in json.loads(stub.text)["objects"]]
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    stub.locate_text = json.dumps({"objects": [{"local_id": ids[0], "bbox": box}]})
    sent = []

    def later(request):
        sent.append([frame.timestamp for frame in request.frames])
        listed = [line.split(":")[0].strip("- ") for line in request.user_prompt.splitlines()[1:]]
        picked = request.frames[0].timestamp
        # One object on a sent still, and one on a moment that was never sent.
        return json.dumps({"objects": [
            {"local_id": listed[-1], "timestamp": picked, "bbox": box},
            {"local_id": listed[0], "timestamp": picked + 0.37, "bbox": box},
        ]})

    stub.later = later
    procedure = client.post("/api/learnloop/procedures", json={"name": "Later"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert len(sent) == 1 and 1 <= len(sent[0]) <= 6 and all(moment >= 0.5 for moment in sent[0])
    assert views["located_later"] == [ids[3]]
    frame = round(sent[0][0] * object_views.SAM_FPS)
    # The first-frame boxes keep their request. The later one is followed forward only, on a copy starting at its frame.
    assert len(prompts) == 2 and views["requests"] == 2
    assert prompts[0][0].startswith("<0f>0<|box;") and "1<|box;" not in prompts[0][0] and prompts[0][1] == "both"
    assert prompts[1][0].startswith("<0f>1<|box;") and prompts[1][1] == "forward" and prompts[1][2].name == f"tracked_12fps_from_{frame}.mp4"
    assert [row["time"] for row in views["objects"][ids[3]]["track"]] == [round((frame + index) / object_views.SAM_FPS, 3) for index in range(4)]
    assert [row["time"] for row in views["objects"][ids[0]]["track"]] == [round(index / object_views.SAM_FPS, 3) for index in range(4)]
    assert views["objects"][ids[3]]["first_seen"] == round(frame / object_views.SAM_FPS, 3)
    assert views["objects"][ids[3]]["seen_cells"] and not views["objects"][ids[1]]["seen_cells"]

    again = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert len(sent) == 1 and len(prompts) == 2
    replayed = client.get(f"/api/learnloop/runs/{again['id']}/objects").json()
    assert replayed["located_later"] == [ids[3]] and replayed["replayed_requests"] == 2


def test_an_object_sam_loses_is_boxed_again_after_it_was_lost(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    prompts = []

    class FakeSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            prompts.append((prompt, kwargs["propagation_direction"]))
            if kwargs["propagation_direction"] == "both":
                # Every object to the end, but red only for the first four frames.
                text = "\n".join(_sam_line(frame, [(0, 10), *([(1, 40)] if frame < 4 else []), (2, 70), (3, 100)]) for frame in range(120))
            else:
                text = "\n".join(_sam_line(frame, [(4, 40)]) for frame in range(4))
            return SimpleNamespace(text=text, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    ids = [item["local_id"] for item in json.loads(stub.text)["objects"]]
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    stub.locate_text = json.dumps({"objects": [{"local_id": local_id, "bbox": box} for local_id in ids]})
    sent = []

    def later(request):
        sent.append(([frame.timestamp for frame in request.frames], request.user_prompt))
        return json.dumps({"objects": [{"local_id": ids[1], "timestamp": request.frames[0].timestamp, "bbox": box}]})

    stub.later = later
    procedure = client.post("/api/learnloop/procedures", json={"name": "Lost"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    # Only the lost object is asked about, and only on stills from after its track ended.
    assert len(sent) == 1 and f"- {ids[1]}:" in sent[0][1] and f"- {ids[0]}:" not in sent[0][1]
    assert all(moment >= 3 / object_views.SAM_FPS + 1.0 for moment in sent[0][0])
    assert views["relocated"] == [ids[1]] and views["located_later"] == []
    frame = round(sent[0][0][0] * object_views.SAM_FPS)
    assert len(prompts) == 2 and prompts[1][1] == "forward" and prompts[1][0].startswith("<0f>4<|box;")
    # The first request's sightings stay, and the new box carries the track on from its frame.
    times = [row["time"] for row in views["objects"][ids[1]]["track"]]
    assert times == [round(index / object_views.SAM_FPS, 3) for index in range(4)] + [round((frame + index) / object_views.SAM_FPS, 3) for index in range(4)]
    assert views["objects"][ids[1]]["first_seen"] == 0

    again = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert len(sent) == 1 and len(prompts) == 2
    replayed = client.get(f"/api/learnloop/runs/{again['id']}/objects").json()
    assert replayed["relocated"] == [ids[1]] and replayed["replayed_requests"] == 2


def test_no_later_request_when_the_first_frame_shows_everything(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)

    class FakeSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            return SimpleNamespace(text="", file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    stub.locate_text = json.dumps({"objects": [{"local_id": item["local_id"], "bbox": box} for item in json.loads(stub.text)["objects"]]})
    procedure = client.post("/api/learnloop/procedures", json={"name": "All"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    assert "learnloop_guided_locate_later" not in [request.schema_name for request in stub.calls]


def test_the_thumbnail_is_the_first_view_not_the_largest():
    from app.pipeline import object_views
    from app.providers.sam import SamRecord

    def sighting(time, side, shade):
        box = {"x": 0.1, "y": 0.1, "w": side, "h": side}
        return object_views.Sighting(time, box, SamRecord(int(time * 8), "0", 0, 0, 1, 1, 100, 100, 0.9, "", "", 0, 0, ""))

    assert object_views._first([sighting(2, 0.5, 0), sighting(1, 0.1, 0)]).timestamp == 1


def test_each_second_shows_its_most_confident_sighting():
    from app.pipeline import object_views
    from app.providers.sam import SamRecord

    def sighting(time, confidence):
        record = SamRecord(int(time * 8), "0", 0, 0, 1, 1, 100, 100, confidence, "", "", 0, 0, "")
        return object_views.Sighting(time, {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}, record)

    times = [0.0, 1.0, 2.0, 3.0, 3.93]
    sightings = [sighting(1.0, 0.5), sighting(1.5, 0.95), sighting(1.875, 0.7), sighting(2.0, 0.99), sighting(2.25, 0.6), sighting(3.625, 0.8), sighting(3.875, 0.9)]
    picks = [item.timestamp if item else None for item in object_views._picks(sightings, times)]
    # 3.625 is more than one and a half seconds after 2.0, so that cell is empty and the last frame starts afresh.
    assert picks == [None, 1.5, 2.0, None, 3.875]
    # 2.0 is the most confident in its second but only an eighth of a second after 1.875.
    close = [sighting(1.875, 0.9), sighting(2.0, 0.99), sighting(2.5, 0.5)]
    assert [item.timestamp if item else None for item in object_views._picks(close, [0.0, 1.0, 2.0, 3.0, 3.93])] == [None, 1.875, 2.5, None, None]
    # The end always shows, so the cell before it skips 1.1, which would sit too close to it.
    ending = [sighting(0.5, 0.6), sighting(1.1, 0.99), sighting(1.25, 0.7)]
    assert [item.timestamp if item else None for item in object_views._picks(ending, [0.0, 1.3])] == [0.5, 1.25]


def test_the_timeline_shows_each_second_then_the_end():
    from app.pipeline.object_views import _cell_times

    assert _cell_times(2.93) == [0, 1, 2, 2.93]
    assert _cell_times(3.0) == [0, 1, 2, 3.0]
    assert _cell_times(3.02) == [0, 1, 2, 3.02]
    # 21 would sit too close to the end at 21.3.
    assert _cell_times(2.3) == [0, 1, 2.3]


def test_bad_starting_boxes_send_nothing_to_sam(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)

    class NoSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            raise AssertionError("SAM should not be called")

    monkeypatch.setattr(object_views, "SamAdapter", NoSam)
    stub.locate_text = "not json"
    procedure = client.post("/api/learnloop/procedures", json={"name": "Bad"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    assert "not valid" in client.get(f"/api/learnloop/runs/{run['id']}/objects").json()["error"]


def test_sam_failure_keeps_the_analysis(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)

    class DownSam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            raise RuntimeError("SAM is down")

    monkeypatch.setattr(object_views, "SamAdapter", DownSam)
    first = json.loads(stub.text)["objects"][0]["local_id"]
    stub.locate_text = json.dumps({"objects": [{"local_id": first, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]})
    procedure = client.post("/api/learnloop/procedures", json={"name": "Down"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    assert run["status"] == "completed", run["error"]
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert "SAM is down" in views["error"]
    assert all(not item["seen_cells"] for item in views["objects"].values())


def test_object_views_can_be_made_again_with_approval_and_leave_the_analysis_alone(lab, monkeypatch):
    from app.config import settings
    from app.pipeline import object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    calls = []
    state = {"down": True}

    class FlakySam:
        def __init__(self, secret, timeout_s=180, model_id="sam-3.1"):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            calls.append(kwargs["propagation_direction"])
            if state["down"]:
                raise RuntimeError("SAM is down")
            text = "\n".join(_sam_line(frame, [(0, 10)]) for frame in range(4))
            return SimpleNamespace(text=text, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FlakySam)
    first = json.loads(stub.text)["objects"][0]["local_id"]
    stub.locate_text = json.dumps({"objects": [{"local_id": first, "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]})
    procedure = client.post("/api/learnloop/procedures", json={"name": "Again"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration"))
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert "SAM is down" in views["error"] and views["can_remake"] and not views["running"]

    # Nothing is queued until the person approves the live calls.
    asked = client.post(f"/api/learnloop/runs/{run['id']}/objects/run", json={})
    assert asked.status_code == 409 and asked.json()["detail"]["code"] == "live_call_required"
    assert asked.json()["detail"]["plan"]["object_view_requests"] == 2
    state["down"] = False
    queued = client.post(f"/api/learnloop/runs/{run['id']}/objects/run", json={"allow_live": True})
    assert queued.status_code == 200 and queued.json()["running"] and queued.json()["progress"] == 0
    assert client.post(f"/api/learnloop/runs/{run['id']}/objects/run", json={"allow_live": True}).status_code == 409
    analysis_calls = [request.schema_name for request in stub.calls].count("learnloop_guided_task")
    _run_job(run["id"])

    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert not views["running"] and views["error"] == "" and views["objects"][first]["seen_cells"]
    assert calls == ["both", "both"]
    assert [request.schema_name for request in stub.calls].count("learnloop_guided_task") == analysis_calls
    after = client.get(f"/api/learnloop/runs/{run['id']}").json()
    assert after["status"] == "completed" and after["stages"]["segmentation"] == "completed"


def test_an_attempt_can_have_object_views_made_with_approval(lab, monkeypatch):
    # Attempts are followed with SAM so their boxes can be checked and seen, including attempts analyzed without it.
    from app.config import settings

    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    run = _analyze(client, _upload(client, procedure, _video(tmp / "blue.mp4", "blue"), "execution"))
    assert client.get(f"/api/learnloop/runs/{run['id']}/objects").json()["can_remake"]
    unapproved = client.post(f"/api/learnloop/runs/{run['id']}/objects/run", json={"allow_live": False})
    assert unapproved.status_code == 409 and unapproved.json()["detail"]["code"] == "live_call_required"
    assert unapproved.json()["detail"]["plan"]["segmentation_model_id"] == "sam-3.1"
    monkeypatch.setattr(settings, "appearance_enabled", False)
    assert not client.get(f"/api/learnloop/runs/{run['id']}/objects").json()["can_remake"]
    assert client.post(f"/api/learnloop/runs/{run['id']}/objects/run", json={"allow_live": True}).status_code == 400


def test_models_come_from_the_account_and_segmentation_is_optional(lab, monkeypatch):
    from app.config import settings
    from app.providers import model_list

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    listed = [
        {"model_id": "muse-spark-1.3", "native_video": True},
        {"model_id": "muse-wonder-1.0", "native_video": True},
    ]
    monkeypatch.setattr(model_list, "live_options", lambda *args, **kwargs: listed)
    choices = client.get("/api/learnloop/models").json()
    assert [item["model_id"] for item in choices["inference"]] == ["muse-spark-1.3", "muse-wonder-1.0"]
    # The account did not list SAM, so the catalog's row is offered.
    assert [item["model_id"] for item in choices["segmentation"]] == ["sam-3.1"]
    assert choices["default_inference"] == "muse-wonder-1.0" and choices["default_segmentation"] == "sam-3.1"

    procedure = client.post("/api/learnloop/procedures", json={"name": "Picks"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    url = f"/api/learnloop/recordings/{video}/analyze"
    assert client.post(url, json={"allow_live": True, "model_id": "sam-3.1"}).status_code == 400
    assert client.post(url, json={"allow_live": True, "segmentation_model_id": "muse-spark-1.3"}).status_code == 400
    plan = client.post(url, json={"model_id": "muse-spark-1.3", "segmentation_model_id": ""}).json()["detail"]["plan"]
    assert plan["model_id"] == "muse-spark-1.3" and plan["object_view_requests"] == 0
    run = _analyze_with(client, video, model_id="muse-spark-1.3", segmentation_model_id="")
    assert run["model_id"] == "muse-spark-1.3"
    assert stub.calls[-1].model_id == "muse-spark-1.3"
    assert run["stages"]["segmentation"] == "skipped"


def _analyze_with(client, video_id, **models):
    response = client.post(f"/api/learnloop/recordings/{video_id}/analyze", json={"allow_live": True, **models})
    assert response.status_code == 200, response.text
    _run_job(response.json()["id"])
    return client.get(f"/api/learnloop/runs/{response.json()['id']}").json()


def test_procedure_can_be_renamed(lab):
    client = lab[0]
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    assert client.patch(f"/api/learnloop/procedures/{procedure}", json={"name": "  "}).status_code == 400
    renamed = client.patch(f"/api/learnloop/procedures/{procedure}", json={"name": " Cube assembly "})
    assert renamed.status_code == 200 and renamed.json()["name"] == "Cube assembly"
    assert client.get(f"/api/learnloop/procedures/{procedure}").json()["name"] == "Cube assembly"


def test_deleting_a_procedure_removes_its_recordings_versions_and_reports(lab):
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, CORRECT, "blue")
    client = lab[0]
    detail = client.get(f"/api/learnloop/procedures/{procedure}").json()
    videos = [item["video"]["id"] for item in detail["recordings"]]
    assert client.delete(f"/api/learnloop/procedures/{procedure}").json() == {"deleted": procedure}
    assert client.get(f"/api/learnloop/procedures/{procedure}").status_code == 404
    assert procedure not in [item["id"] for item in client.get("/api/learnloop/procedures").json()]
    assert client.get(f"/api/verifications/{report['id']}/export").status_code == 404
    for video in videos:
        assert client.get(f"/api/learnloop/recordings/{video}").status_code == 404
    assert client.delete(f"/api/learnloop/procedures/{procedure}").status_code == 404
