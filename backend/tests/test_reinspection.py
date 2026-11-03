"""Automatic second look at unresolved steps. The Meta client is a stub; no test spends credits."""

import json

from app.pipeline.contracts import REINSPECT
from app.schemas.generation import ReinspectOutput
from app.services.reinspection import validate_findings

from tests.test_learnloop_flow import CORRECT, HIDDEN, _accepted, _check, _run_job, _verdicts, lab  # noqa: F401


def _target():
    return {"step_id": "s3", "step": {"verdict": "CANNOT_VERIFY"}, "predicates": [{"predicate_id": "p1", "status": "cannot"}]}


def _finding(**over):
    item = {
        "step_id": "s3",
        "predicate_id": "p1",
        "outcome": "supported",
        "visibility": "visible",
        "evidence_timestamps": [9.5],
        "reason": "The green cube sits on the base.",
        "limitations": [],
        "evidence_class": "DIRECTLY_OBSERVED",
    }
    item.update(over)
    return item


def _validated(*findings):
    return validate_findings(ReinspectOutput.model_validate({"findings": list(findings)}), _target(), [8.0, 9.5])


def test_a_direct_visible_finding_is_kept():
    kept, problems = _validated(_finding())
    assert kept[0]["outcome"] == "supported" and kept[0]["evidence_timestamps"] == [9.5] and not problems


def test_findings_for_conditions_not_asked_or_frames_not_sent_are_rejected():
    kept, problems = _validated(_finding(predicate_id="other"), _finding(evidence_timestamps=[3.0]))
    # The unknown predicate is dropped. The uncited frame leaves no evidence, so supported falls back to inconclusive.
    assert [item["predicate_id"] for item in kept] == ["p1"] and kept[0]["outcome"] == "inconclusive"
    assert any("not asked" in item for item in problems) and any("not sent" in item for item in problems)


def test_an_inferred_or_occluded_claim_cannot_support_or_contradict():
    for over in ({"evidence_class": "INFERRED"}, {"visibility": "occluded"}, {"outcome": "contradicted", "evidence_timestamps": []}):
        kept, _ = _validated(_finding(**over))
        assert kept[0]["outcome"] == "inconclusive"


def test_conflicting_outcomes_for_one_condition_count_as_none():
    kept, problems = _validated(_finding(), _finding(outcome="contradicted"))
    assert kept[0]["outcome"] == "inconclusive" and any("conflicting" in item for item in problems)


def test_the_reinspection_schema_is_strict():
    assert REINSPECT.schema["additionalProperties"] is False


def _stub_reinspect(stub, outcome, visibility="visible", evidence_class="DIRECTLY_OBSERVED", predicate=None):
    sent = []
    original = stub._reply

    def reply(request):
        if request.schema_name == "learnloop_reinspect":
            sent.append(request)
            times = [frame.timestamp for frame in request.frames]
            asked = predicate or request.user_prompt.split("predicate_id ")[1].split(": ")[0]
            return json.dumps(
                {
                    "findings": [
                        {
                            "step_id": "s3",
                            "predicate_id": asked,
                            "outcome": outcome,
                            "visibility": visibility,
                            "evidence_timestamps": [times[-1]] if outcome != "inconclusive" else [],
                            "reason": "authored",
                            "limitations": [],
                            "evidence_class": evidence_class,
                        }
                    ]
                }
            )
        return original(request)

    stub._reply = reply
    return sent


def test_an_unresolved_step_gets_one_second_look_that_can_pass_it_and_keeps_the_first_report(lab):  # noqa: F811
    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    sent = _stub_reinspect(stub, "supported")
    report = _check(lab, procedure, template, HIDDEN, "white")
    assert _verdicts(report)["s3"] == "CANNOT_VERIFY" and report["report"]["reinspection"]["status"] == "pending"
    _run_job(report["execution_run_id"])
    assert len(sent) == 1
    # A short frame sequence from the attempt, not the whole video.
    assert 2 <= len(sent[0].frames) <= 6 and sent[0].video_path is None
    after = client.get(f"/api/verifications/{report['id']}").json()
    step = {item["step_id"]: item for item in after["report"]["steps"]}["s3"]
    assert step["verdict"] == "PASS" and step["basis"] == "model_reinspection" and step["machine_verdict"] == "CANNOT_VERIFY"
    assert step["history"][-1]["verdict"] == "CANNOT_VERIFY"
    assert any(item["provenance"] == "AI_REINSPECTION" for item in step["evidence"])
    assert after["report"]["reinspection"]["changed"] == [{"step_id": "s3", "before": "CANNOT_VERIFY", "after": "PASS"}]
    assert after["overall"] == "PASS"
    exported = client.get(f"/api/verifications/{report['id']}/export").json()
    assert {item["step_id"]: item for item in exported["machine_report"]["steps"]}["s3"]["verdict"] == "CANNOT_VERIFY"

    # A reviewer who saw otherwise still decides.
    reviewed = client.post(
        f"/api/verifications/{report['id']}/steps/s3/observations", json={"answer": "not_installed", "timestamp": 9.5}
    ).json()
    assert _verdicts(reviewed)["s3"] == "FAIL"


def test_a_contradicted_condition_fails_and_an_inconclusive_answer_changes_nothing(lab):  # noqa: F811
    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    _stub_reinspect(stub, "contradicted")
    report = _check(lab, procedure, template, HIDDEN, "white")
    _run_job(report["execution_run_id"])
    assert _verdicts(client.get(f"/api/verifications/{report['id']}").json())["s3"] == "FAIL"

    _stub_reinspect(stub, "inconclusive", visibility="occluded", evidence_class="UNKNOWN")
    report = _check(lab, procedure, template, HIDDEN, "cyan")
    _run_job(report["execution_run_id"])
    after = client.get(f"/api/verifications/{report['id']}").json()
    assert _verdicts(after)["s3"] == "CANNOT_VERIFY" and after["overall"] == "INCONCLUSIVE"


def test_an_inferred_supported_answer_cannot_pass_a_step(lab):  # noqa: F811
    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    _stub_reinspect(stub, "supported", evidence_class="INFERRED")
    report = _check(lab, procedure, template, HIDDEN, "white")
    _run_job(report["execution_run_id"])
    assert _verdicts(client.get(f"/api/verifications/{report['id']}").json())["s3"] == "CANNOT_VERIFY"


def test_nothing_is_sent_without_live_approval_and_the_report_says_why(lab, monkeypatch):  # noqa: F811
    from app.config import settings

    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    sent = _stub_reinspect(stub, "supported")
    run_report = _check(lab, procedure, template, HIDDEN, "white")
    from app.db import SessionLocal
    from app.models.runs import Job

    # Same attempt, checked again with live calls off on the server.
    with SessionLocal() as session:
        session.query(Job).filter(Job.from_stage.like("reinspect:%")).delete(synchronize_session=False)
        session.commit()
    monkeypatch.setattr(settings, "live_inference", False)
    again = client.post(f"/api/learnloop/templates/{template['id']}/verify", json={"run_id": run_report["execution_run_id"]}).json()
    assert again["report"]["reinspection"]["status"] == "unavailable" and "not approved" in again["report"]["reinspection"]["reason"]
    assert _verdicts(again)["s3"] == "CANNOT_VERIFY" and not sent


def test_a_report_with_nothing_unresolved_queues_no_second_look(lab):  # noqa: F811
    client, stub, tmp = lab
    procedure, template = _accepted(lab)
    report = _check(lab, procedure, template, CORRECT, "blue")
    assert report["report"]["reinspection"]["status"] == "not_needed"

