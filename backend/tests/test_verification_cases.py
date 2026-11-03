"""Remaining verification cases: correction, ambiguity, claims, metrics, and revisions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.schemas.analysis import AnalysisDocument
from app.verification import learn_template, verify_execution
from app.verification.evaluate import compare_strategies, evaluate_report
from app.verification.reinspect import apply_finding, plan_reinspections
from app.verification.roles import map_roles
from fastapi.testclient import TestClient

from tests.verification_fixtures import (
    action,
    assembly_demo,
    assembly_execution,
    dep,
    document,
    obj,
    obs,
    rel,
    state,
    step,
    transition,
)

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")


def _steps(report):
    return {item["step_id"]: item for item in report["steps"]}


def _run(execution, proposals=None, strategy="hybrid"):
    template = learn_template(assembly_demo(), promote_supported=True, accept=True)
    report = verify_execution(
        template,
        execution,
        proposals=proposals,
        strategy=strategy,
        execution_video_id="execution-video",
        demonstration_video_id="demo-video",
    )
    return template, report


def test_h_corrected_mistake_passes_and_keeps_the_mistake():
    demo = document(
        objects=[
            obj("D_bolt", "bolt", "threaded shaft", "long threaded rod"),
            obj("D_washer", "washer", "flat ring", "flat annular ring"),
        ],
        observations=[obs("D_bolt", 18), obs("D_washer", 18)],
        states=[state("D_washer", "seated", 8, 20)],
        transitions=[transition("D_washer", "loose", "seated", 8)],
        relationships=[rel("D_washer", "D_bolt", "around", 8, 20)],
        actions=[action("A_washer", "place", 6, 10, ["D_washer", "D_bolt"])],
        steps=[step("step_washer", "Seat ring", ["D_washer", "D_bolt"], ["A_washer"], 6, 10, "ring around shaft")],
    )
    execution = document(
        objects=[
            obj("E_bolt", "bolt", "threaded shaft", "long threaded rod"),
            obj("E_washer", "washer", "flat ring", "flat annular ring"),
        ],
        observations=[obs("E_bolt", 18), obs("E_washer", 18)],
        states=[state("E_washer", "aside", 1, 4), state("E_washer", "seated", 12, 20)],
        transitions=[transition("E_washer", "loose", "seated", 12)],
        relationships=[
            rel("E_washer", "E_bolt", "left_of", 1, 4),
            rel("E_washer", "E_bolt", "around", 12, 20),
        ],
        actions=[
            action("X_aside", "place", 1, 4, ["E_washer", "E_bolt"]),
            action("X_washer", "place", 12, 16, ["E_washer", "E_bolt"]),
        ],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    report = verify_execution(template, execution)
    washer = _steps(report)["step_washer"]
    assert washer["verdict"] == "PASS", washer
    assert washer["corrected"] is True
    assert washer["execution_state"] == "corrected"
    assert washer["error_category"] == "TEMPORARY_ERROR_CORRECTED"
    assert report["overall"] == "PASS"


def test_j_similar_objects_keep_distinct_roles_and_ties_stay_ambiguous():
    demo = document(
        objects=[
            obj("D_left", "pin", "pin on the left", "short metal pin"),
            obj("D_right", "pin", "pin on the right", "short metal pin"),
            obj("D_plate", "plate", "base plate", "flat plate"),
        ],
        observations=[obs("D_left", 8), obs("D_right", 8), obs("D_plate", 8)],
        relationships=[
            rel("D_left", "D_plate", "left_of", 2, 20),
            rel("D_right", "D_plate", "right_of", 2, 20),
        ],
        actions=[
            action("A_left", "place", 0, 2, ["D_left", "D_plate"]),
            action("A_right", "place", 2, 4, ["D_right", "D_plate"]),
        ],
        steps=[
            step("step_left", "Place left pin", ["D_left", "D_plate"], ["A_left"], 0, 2, "left pin"),
            step("step_right", "Place right pin", ["D_right", "D_plate"], ["A_right"], 2, 4, "right pin"),
        ],
    )
    execution = document(
        objects=[
            obj("E_left", "pin", "pin on the left", "short metal pin"),
            obj("E_right", "pin", "pin on the right", "short metal pin"),
            obj("E_plate", "plate", "base plate", "flat plate"),
        ],
        observations=[obs("E_left", 12), obs("E_right", 12), obs("E_plate", 12)],
        relationships=[
            rel("E_left", "E_plate", "left_of", 2, 20),
            rel("E_right", "E_plate", "right_of", 2, 20),
        ],
        actions=[
            action("X_left", "place", 0, 2, ["E_left", "E_plate"]),
            action("X_right", "place", 2, 4, ["E_right", "E_plate"]),
        ],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    report = verify_execution(template, execution)
    roles = {role["role_id"]: role for role in report["roles"]}
    assert roles["role:D_left"]["object_local_ids"] == ["E_left"], roles
    assert roles["role:D_right"]["object_local_ids"] == ["E_right"]
    assert report["overall"] == "PASS"

    tie_demo = document(
        objects=[
            obj("D_a", "pin", "metal pin", "short metal pin"),
            obj("D_b", "pin", "metal pin", "short metal pin"),
            obj("D_plate", "plate", "base plate", "flat plate"),
        ],
        observations=[obs("D_a", 8), obs("D_b", 8), obs("D_plate", 8)],
        relationships=[
            rel("D_a", "D_plate", "touching", 2, 20),
            rel("D_b", "D_plate", "touching", 2, 20),
        ],
        actions=[
            action("A_a", "place", 0, 2, ["D_a", "D_plate"]),
            action("A_b", "place", 2, 4, ["D_b", "D_plate"]),
        ],
        steps=[
            step("step_a", "Place first pin", ["D_a", "D_plate"], ["A_a"], 0, 2, "pin touching plate"),
            step("step_b", "Place second pin", ["D_b", "D_plate"], ["A_b"], 2, 4, "pin touching plate"),
        ],
    )
    tie_execution = document(
        objects=[
            obj("E_a", "pin", "metal pin", "short metal pin"),
            obj("E_b", "pin", "metal pin", "short metal pin"),
            obj("E_plate", "plate", "base plate", "flat plate"),
        ],
        observations=[obs("E_a", 12), obs("E_b", 12), obs("E_plate", 12)],
        relationships=[
            rel("E_a", "E_plate", "touching", 2, 20),
            rel("E_b", "E_plate", "touching", 2, 20),
        ],
        actions=[
            action("X_a", "place", 0, 2, ["E_a", "E_plate"]),
            action("X_b", "place", 2, 4, ["E_b", "E_plate"]),
        ],
    )
    twins = verify_execution(learn_template(tie_demo, promote_supported=True, accept=True), tie_execution)
    twin_roles = [role for role in twins["roles"] if role["label"].startswith("pin")]
    assert [role["label"] for role in twin_roles] == ["pin 1", "pin 2"]
    assert all(role["status"] == "assigned" for role in twin_roles), twin_roles
    assert twin_roles[0]["object_local_ids"] != twin_roles[1]["object_local_ids"]
    assert all(len(role["hypotheses"]) >= 2 for role in twin_roles)
    assert twins["overall"] == "PASS"

    # One learned pin and two matching instances: nothing says which one it is.
    one_pin = document(
        objects=[obj("D_a", "pin", "metal pin", "short metal pin"), obj("D_plate", "plate", "base plate", "flat plate")],
        observations=[obs("D_a", 8), obs("D_plate", 8)],
        relationships=[rel("D_a", "D_plate", "touching", 2, 20)],
        actions=[action("A_a", "place", 0, 2, ["D_a", "D_plate"])],
        steps=[step("step_a", "Place pin", ["D_a", "D_plate"], ["A_a"], 0, 2, "pin touching plate")],
    )
    tie = verify_execution(learn_template(one_pin, promote_supported=True, accept=True), tie_execution)
    pin_roles = [role for role in tie["roles"] if role["label"] == "pin"]
    assert pin_roles
    assert all(role["status"] == "ambiguous" for role in pin_roles), pin_roles
    assert all(role["object_local_ids"] == [] for role in pin_roles)
    assert all(len(role["hypotheses"]) >= 2 for role in pin_roles)
    assert all(step["verdict"] == "CANNOT_VERIFY" for step in tie["steps"])
    assert tie["overall"] == "INCONCLUSIVE"


def test_k_unsupported_model_claim_cannot_become_a_pass():
    execution = assembly_execution(include_washer=False, bolt_visibility="occluded")
    inferred = {"step_washer": {"verdict": "PASS", "evidence_class": "INFERRED"}}
    _template, hybrid = _run(execution, proposals=inferred, strategy="hybrid")
    washer = _steps(hybrid)["step_washer"]
    assert washer["verdict"] == "CANNOT_VERIFY", washer
    assert "not accepted" in washer["explanation"].lower()
    claimed = {"step_washer": {"verdict": "PASS", "evidence_class": "DIRECTLY_OBSERVED"}}
    _template, still = _run(execution, proposals=claimed, strategy="hybrid")
    assert _steps(still)["step_washer"]["verdict"] == "CANNOT_VERIFY"
    _template, vlm = _run(execution, proposals=claimed, strategy="vlm_only")
    assert _steps(vlm)["step_washer"]["verdict"] == "PASS"
    truth = {
        "steps": [
            {"step_id": "step_washer", "verdict": "CANNOT_VERIFY", "observable": False},
            {"step_id": "step_position", "verdict": "PASS", "observable": True},
            {"step_id": "step_insert", "verdict": "PASS", "observable": True},
            {"step_id": "step_nut", "verdict": "PASS", "observable": True},
        ]
    }
    template = learn_template(assembly_demo(), promote_supported=True, accept=True)
    compared = compare_strategies(template, execution, truth, proposals=claimed)
    assert compared["strategies"]["hybrid"]["metrics"]["incorrect_pass"] == 0
    assert compared["strategies"]["vlm_only"]["metrics"]["incorrect_pass"] >= 1
    assert compared["model_agreement_is_accuracy"] is False


def test_action_label_alone_cannot_pass():
    demo = document(
        objects=[obj("D_part", "part", "loose part", "small block")],
        observations=[obs("D_part", 4)],
        actions=[action("A_move", "move", 0, 2, ["D_part"])],
        steps=[step("step_move", "Move part", ["D_part"], ["A_move"], 0, 2, "part moved")],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    assert template.steps[0].required is False
    template.steps[0].required = True
    template.steps[0].requirement_status = "required"
    execution = document(
        objects=[obj("E_part", "part", "loose part", "small block")],
        observations=[obs("E_part", 4)],
        actions=[action("X_move", "move", 0, 2, ["E_part"], outcome="completed")],
    )
    report = verify_execution(template, execution)
    assert _steps(report)["step_move"]["verdict"] == "CANNOT_VERIFY"
    assert "action label" in _steps(report)["step_move"]["explanation"].lower()


def test_unexpected_action_is_reported_without_failing_a_correct_result():
    extra = action("X_cloth", "move", 40, 42, ["E_cloth"])
    execution = assembly_execution(extra_actions=[extra])
    raw = execution.model_dump(mode="json")
    raw["objects"].append(obj("E_cloth", "cloth", "shop cloth", "folded cloth"))
    raw["observations"].append(obs("E_cloth", 41))
    execution = AnalysisDocument.model_validate(raw)
    _template, report = _run(execution)
    assert report["overall"] == "PASS", report["steps"]
    assert any(item["action_id"] == "X_cloth" and item["failure"] is False for item in report["unexpected_actions"])


def test_reinspection_preserves_history_and_can_stay_unverifiable():
    _template, report = _run(assembly_execution(include_washer=False, bolt_visibility="occluded"))
    plans = plan_reinspections(report, step_id="step_washer")
    assert plans and plans[0]["status"] == "not_sent"
    assert plans[0]["frame_timestamps"]
    held = apply_finding(
        report,
        {
            "step_id": "step_washer",
            "visibility": "insufficient",
            "relation_supported": None,
            "evidence_class": "DIRECTLY_OBSERVED",
            "timestamps": [26.4],
        },
    )
    assert _steps(held)["step_washer"]["verdict"] == "CANNOT_VERIFY"
    changed = apply_finding(
        report,
        {
            "step_id": "step_washer",
            "visibility": "visible",
            "relation_supported": False,
            "evidence_class": "DIRECTLY_OBSERVED",
            "timestamps": [26.4],
            "error_category": "MISSING_COMPONENT",
        },
    )
    washer = _steps(changed)["step_washer"]
    assert washer["verdict"] == "FAIL"
    assert washer["history"][0]["verdict"] == "CANNOT_VERIFY"
    ignored = apply_finding(
        report,
        {
            "step_id": "step_washer",
            "visibility": "visible",
            "relation_supported": True,
            "evidence_class": "INFERRED",
            "timestamps": [26.4],
        },
    )
    assert _steps(ignored)["step_washer"]["verdict"] == "CANNOT_VERIFY"


def test_metrics_count_errors_against_human_labels():
    _template, report = _run(assembly_execution(include_washer=False))
    truth = {
        "steps": [
            {
                "step_id": "step_washer",
                "verdict": "FAIL",
                "error_type": "MISSING_COMPONENT",
                "timestamp": 45,
                "observable": True,
            },
            {"step_id": "step_position", "verdict": "PASS", "observable": True},
            {"step_id": "step_insert", "verdict": "PASS", "observable": True},
            {"step_id": "step_nut", "verdict": "PASS", "observable": True},
        ]
    }
    metrics = evaluate_report(report, truth)
    assert metrics["error_detection"]["tp"] == 1
    assert metrics["error_detection"]["fp"] == 0
    assert metrics["incorrect_pass"] == 0
    assert metrics["timestamp_mean_abs_error"] == 0
    assert metrics["error_confusion"]["MISSING_COMPONENT"]["MISSING_COMPONENT"] == 1
    assert metrics["is_accuracy"] is True


def test_verifier_source_has_no_task_specific_rules():
    root = Path(__file__).resolve().parents[1] / "app" / "verification"
    banned = ["washer", "wingnut", "lego", "hexagonal", "lock washer"]
    for path in root.glob("*.py"):
        text = path.read_text().lower()
        for word in banned:
            assert word not in text, (path.name, word)


def test_perception_claim_audit_is_unchanged():
    from app.perception.verify import audit_claims

    assert callable(audit_claims)


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


def _install(project_id: str, filename: str, document: AnalysisDocument) -> str:
    from app.db import SessionLocal
    from app.models.media import VideoAsset
    from app.models.runs import AnalysisRevision, AnalysisRun

    session = SessionLocal()
    try:
        video = VideoAsset(
            project_id=project_id,
            original_filename=filename,
            stored_path=f"originals/{filename}",
            checksum_sha256=filename,
        )
        session.add(video)
        session.flush()
        run = AnalysisRun(
            project_id=project_id,
            video_id=video.id,
            name=filename,
            status="completed",
            research_mode="fixed_camera",
            viewpoint="fixed",
            task_constraint="constrained",
            provider_id="fixture",
            model_id="fixture",
            modality="sampled_frames",
            sampling_preset="standard",
            config_json={"perception": {"strategy": "B_temporal"}},
            fixture=True,
        )
        session.add(run)
        session.flush()
        revision = AnalysisRevision(
            run_id=run.id,
            layer="validated",
            version=1,
            document_json=document.model_dump(mode="json"),
        )
        session.add(revision)
        session.flush()
        run.active_revision_id = revision.id
        session.commit()
        return run.id
    finally:
        session.close()


def test_same_day_project_can_learn_a_workflow(client):
    created = client.post("/api/projects", json={"name": "Same day"})
    assert created.status_code == 200, created.text
    assert created.json()["is_task"] is False
    project_id = created.json()["id"]
    demo_run = _install(project_id, "same-day-demo.mp4", assembly_demo())
    learned = client.post(
        f"/api/tasks/{project_id}/workflow-templates",
        json={"run_id": demo_run, "name": "Same day procedure"},
    )
    assert learned.status_code == 200, learned.text
    listed = client.get(f"/api/tasks/{project_id}/workflow-templates")
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == [learned.json()["id"]]
    verifications = client.get(f"/api/tasks/{project_id}/verifications")
    assert verifications.status_code == 200
    assert verifications.json() == []
    missing = client.get("/api/tasks/missing-project/workflow-templates")
    assert missing.status_code == 404


def test_l_past_verification_keeps_the_template_revision_it_used(client):
    project_id = client.post("/api/tasks", json={"name": "Verify", "description": ""}).json()["project"]["id"]
    demo_run = _install(project_id, "demo.mp4", assembly_demo())
    execution_run = _install(project_id, "execution.mp4", assembly_execution(include_washer=False))
    created = client.post(
        f"/api/tasks/{project_id}/workflow-templates",
        json={"run_id": demo_run, "promote_supported": True, "accept": True, "name": "Joined plates"},
    )
    assert created.status_code == 200, created.text
    template_id = created.json()["id"]
    assert created.json()["version"] == 1
    started = client.post(
        f"/api/workflow-templates/{template_id}/verifications",
        json={"execution_run_id": execution_run, "strategy": "hybrid"},
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["template_id"] == template_id
    assert body["overall"] == "FAIL", body["report"]["steps"]
    washer = next(step for step in body["report"]["steps"] if step["step_id"] == "step_washer")
    assert washer["error_category"] == "MISSING_COMPONENT"
    exported = client.get(f"/api/verifications/{body['id']}/export")
    assert exported.status_code == 200
    assert "stored_path" not in exported.text
    revised = client.patch(
        f"/api/workflow-templates/{template_id}",
        json={"name": "Joined plates revised", "steps": [{"step_id": "step_washer", "requirement_status": "optional"}]},
    )
    assert revised.status_code == 200, revised.text
    assert revised.json()["id"] != template_id
    assert revised.json()["version"] == 2
    original = client.get(f"/api/workflow-templates/{template_id}")
    assert original.json()["name"] == "Joined plates"
    washer_step = next(step for step in original.json()["document"]["steps"] if step["step_id"] == "step_washer")
    assert washer_step["requirement_status"] == "required"
    saved = client.get(f"/api/verifications/{body['id']}")
    assert saved.json()["template_id"] == template_id
    assert saved.json()["template_version"] == 1
    evidence = client.get(f"/api/verifications/{body['id']}/evidence")
    assert evidence.status_code == 200
    assert evidence.json()


def _pin_and_cap_demo(category: str) -> AnalysisDocument:
    return document(
        objects=[
            obj("D_pin", "pin", "short shaft", "metal pin"),
            obj("D_block", "block", "base block", "square block"),
            obj("D_cap", "cap", "cover", "round cap"),
        ],
        observations=[obs("D_pin", 12), obs("D_block", 12), obs("D_cap", 12)],
        relationships=[rel("D_pin", "D_block", "inside", 4, 30), rel("D_cap", "D_pin", "attached", 8, 30)],
        actions=[
            action("A_ready", "insert", 0, 5, ["D_pin", "D_block"]),
            action("A_lock", "attach", 5, 10, ["D_cap", "D_pin"]),
        ],
        steps=[
            step("step_ready", "Seat pin", ["D_pin", "D_block"], ["A_ready"], 0, 5, "pin inside block"),
            step("step_lock", "Lock cap", ["D_cap", "D_pin"], ["A_lock"], 5, 10, "cap attached to pin"),
        ],
        dependencies=[dep("step_ready", "step_lock", category)],
    )


def _cap_first_execution() -> AnalysisDocument:
    return document(
        objects=[
            obj("E_pin", "pin", "short shaft", "metal pin"),
            obj("E_block", "block", "base block", "square block"),
            obj("E_cap", "cap", "cover", "round cap"),
        ],
        observations=[obs("E_pin", 20), obs("E_block", 20), obs("E_cap", 20)],
        relationships=[rel("E_cap", "E_pin", "attached", 2, 30), rel("E_pin", "E_block", "inside", 12, 30)],
        actions=[
            action("X_lock", "attach", 2, 6, ["E_cap", "E_pin"]),
            action("X_ready", "insert", 12, 18, ["E_pin", "E_block"]),
        ],
    )


def _order_template(client, category: str) -> tuple[str, str]:
    project_id = client.post("/api/tasks", json={"name": "Order", "description": ""}).json()["project"]["id"]
    demo_run = _install(project_id, "demo.mp4", _pin_and_cap_demo(category))
    execution_run = _install(project_id, "execution.mp4", _cap_first_execution())
    created = client.post(
        f"/api/tasks/{project_id}/workflow-templates",
        json={"run_id": demo_run, "promote_supported": True, "accept": True},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"], execution_run


def _lock_result(client, template_id: str, execution_run: str) -> dict:
    started = client.post(
        f"/api/workflow-templates/{template_id}/verifications",
        json={"execution_run_id": execution_run, "strategy": "hybrid"},
    )
    assert started.status_code == 200, started.text
    return _steps(started.json()["report"])["step_lock"]


def test_reviewer_can_require_an_order_the_model_did_not(client):
    template_id, execution_run = _order_template(client, "observed_temporal")
    assert _lock_result(client, template_id, execution_run)["error_category"] != "INCORRECT_ORDER"
    revised = client.patch(
        f"/api/workflow-templates/{template_id}",
        json={"steps": [{"step_id": "step_lock", "order_after": ["step_ready"]}]},
    )
    assert revised.status_code == 200, revised.text
    body = revised.json()
    assert body["acceptance_status"] == "draft"
    lock = next(item for item in body["document"]["steps"] if item["step_id"] == "step_lock")
    assert lock["human_edited"] == ["order_after"]
    assert lock["dependencies"][0]["origin"] == "human"
    assert _lock_result(client, body["id"], execution_run)["error_category"] == "INCORRECT_ORDER"
    original = client.get(f"/api/workflow-templates/{template_id}").json()
    kept = next(item for item in original["document"]["steps"] if item["step_id"] == "step_lock")
    assert kept["dependencies"][0]["constrains_order"] is False


def test_reviewer_can_lift_an_order_the_model_required(client):
    template_id, execution_run = _order_template(client, "necessary_physical")
    assert _lock_result(client, template_id, execution_run)["error_category"] == "INCORRECT_ORDER"
    revised = client.patch(
        f"/api/workflow-templates/{template_id}",
        json={"steps": [{"step_id": "step_lock", "order_after": []}]},
    )
    assert revised.status_code == 200, revised.text
    lock = _lock_result(client, revised.json()["id"], execution_run)
    assert lock["verdict"] == "PASS", lock


def test_reviewer_can_add_an_order_between_unlinked_steps(client):
    template_id, _execution_run = _order_template(client, "observed_temporal")
    revised = client.patch(
        f"/api/workflow-templates/{template_id}",
        json={"steps": [{"step_id": "step_ready", "order_after": ["step_lock"]}]},
    )
    assert revised.status_code == 200, revised.text
    ready = next(item for item in revised.json()["document"]["steps"] if item["step_id"] == "step_ready")
    assert ready["dependencies"][0]["category"] == "human_specified"
    assert ready["dependencies"][0]["constrains_order"] is True


@pytest.mark.parametrize(
    "steps",
    [
        [{"step_id": "step_lock", "order_after": ["step_lock"]}],
        [{"step_id": "step_lock", "order_after": ["step_missing"]}],
        [
            {"step_id": "step_lock", "order_after": ["step_ready"]},
            {"step_id": "step_ready", "order_after": ["step_lock"]},
        ],
    ],
)
def test_impossible_step_orders_are_refused(client, steps):
    template_id, _execution_run = _order_template(client, "observed_temporal")
    revised = client.patch(f"/api/workflow-templates/{template_id}", json={"steps": steps})
    assert 400 <= revised.status_code < 500, revised.text
    assert client.get(f"/api/workflow-templates/{template_id}").json()["version"] == 1


def _hidden_insertion_answer(pin_relation: str = "attached") -> AnalysisDocument:
    """Hand occlusion on the insertions, and an earlier arrangement that stays true to the end."""
    parts = [("clear", "clear piece"), ("body", "gray body"), ("fork", "black fork"), ("wheel_a", "wheel"), ("wheel_b", "wheel"), ("pin", "white pin")]
    return document(
        objects=[obj(local_id, label, label, label) for local_id, label in parts],
        observations=[obs(local_id, 20) for local_id, _label in parts],
        relationships=[
            rel("clear", "body", "attached", 3, 20, "INFERRED"),
            rel("fork", "body", "attached", 6, 20),
            rel("wheel_a", "fork", "attached", 9, 20),
            rel("wheel_b", "fork", "attached", 11, 20),
            rel("pin", "fork", pin_relation, 16, 20, "INFERRED"),
        ],
        transitions=[
            {**transition("clear", "separate", "inserted", 3), "evidence": rel("clear", "body", "attached", 3, 20, "INFERRED")["evidence"]},
            {**transition("pin", "separate", "inserted", 16), "evidence": rel("pin", "fork", "attached", 16, 20, "INFERRED")["evidence"]},
        ],
        steps=[
            step("step_clear", "Insert clear piece", ["clear", "body"], [], 1, 4, "seated"),
            step("step_fork", "Attach fork", ["fork", "body"], [], 4, 7, "joined"),
            step("step_wheel_a", "Attach first wheel", ["wheel_a", "fork", "body"], [], 7, 10, "joined"),
            step("step_wheel_b", "Attach second wheel", ["wheel_b", "fork", "body"], [], 10, 13, "joined"),
            step("step_pin", "Insert pin", ["pin", "body", "fork"], [], 15, 17, "seated"),
        ],
    )


def _required_checks(template, step_id: str) -> list[tuple[str, str, str]]:
    found = next(item for item in template.steps if item.step_id == step_id)
    return [
        (item.subject_role, item.relation, item.target_role)
        for item in found.completion_conditions
        if item.evidence_requirement == "required"
    ]


def test_each_step_checks_only_the_arrangement_it_achieved():
    template = learn_template(_hidden_insertion_answer(), results_first=True)
    assert _required_checks(template, "step_pin") == [("role:pin", "inside", "role:fork")]
    assert _required_checks(template, "step_wheel_a") == [("role:wheel_a", "attached", "role:fork")]


def test_a_hand_hidden_insertion_gets_a_check_but_is_not_made_required():
    template = learn_template(_hidden_insertion_answer(), results_first=True, promote_supported=True)
    assert _required_checks(template, "step_clear") == [("role:clear", "inside", "role:body")]
    assert next(item for item in template.steps if item.step_id == "step_clear").required is False


def test_identical_parts_are_numbered_and_do_not_tie_forever():
    answer = _hidden_insertion_answer()
    template = learn_template(answer, results_first=True)
    labels = {role.role_id: role.label for role in template.roles}
    assert (labels["role:wheel_a"], labels["role:wheel_b"]) == ("wheel 1", "wheel 2")
    mapped = {item.role_id: item for item in map_roles(template, answer)}
    assert mapped["role:wheel_a"].status == mapped["role:wheel_b"].status == "assigned"
    assert mapped["role:wheel_a"].object_local_ids != mapped["role:wheel_b"].object_local_ids


def test_a_seated_part_reported_as_attached_is_not_a_wrong_placement():
    template = learn_template(_hidden_insertion_answer(), results_first=True)
    attempt = _hidden_insertion_answer()
    for item in attempt.relationships:
        item.evidence.classification = "DIRECTLY_OBSERVED"
    report = verify_execution(template, attempt)
    pin = next(item for item in report["steps"] if item["step_id"] == "step_pin")
    assert pin["verdict"] == "PASS"
