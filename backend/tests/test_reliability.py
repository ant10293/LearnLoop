"""Reliability Lab: bounded concurrent repetitions, independent inference, isolation, and metrics.

Meta is stubbed. These tests show the orchestration and arithmetic work; they say nothing
about how consistent a real model is on real footage.
"""

import asyncio
import json
import threading
import time

import pytest
from sqlalchemy import select

from tests.test_learnloop_flow import (  # noqa: F401  (lab is a fixture)
    CORRECT,
    DEMONSTRATION,
    HIDDEN,
    OMITTED,
    _accepted,
    _analyze,
    _upload,
    _video,
    lab,
)


class Gauge:
    """Counts guided calls in flight at once, across threads and event loops."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.calls = 0
        self.windows: list[tuple[float, float]] = []


@pytest.fixture()
def slow(lab, monkeypatch):  # noqa: F811
    """Guided calls take a moment, so overlap between repetitions is measurable."""
    from app.config import settings

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "reliability_poll_seconds", 0.05)
    gauge = Gauge()
    original = stub.generate

    async def generate(secret, request):
        if request.schema_name not in {"learnloop_guided_locate", "learnloop_guided_locate_later"} and request.schema_name != "learnloop_reinspect":
            with gauge.lock:
                gauge.now += 1
                gauge.calls += 1
                gauge.peak = max(gauge.peak, gauge.now)
            start = time.monotonic()
            try:
                await asyncio.sleep(0.4)
                if getattr(stub, "fail_on", None) and gauge.calls in stub.fail_on:
                    from app.providers.types import ProviderError

                    raise ProviderError("Error code: 429 - rate limit reached for requests")
            finally:
                with gauge.lock:
                    gauge.now -= 1
                    gauge.windows.append((start, time.monotonic()))
        return await original(secret, request)

    monkeypatch.setattr(stub, "generate", generate)
    return client, stub, tmp, gauge


def _drive(batch_id: str, timeout: float = 90.0) -> dict:
    """Run the Reliability lane until the batch settles, as the server would."""
    from app.db import SessionLocal
    from app.models.reliability import ReliabilityBatch
    from app.reliability.executor import reliability_loop

    def settled() -> bool:
        session = SessionLocal()
        try:
            return session.get(ReliabilityBatch, batch_id).status in {"completed", "canceled"}
        finally:
            session.close()

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(reliability_loop(stop))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not await asyncio.to_thread(settled):
            await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(main())
    from app.reliability.service import get_batch

    session = SessionLocal()
    try:
        return get_batch(session, batch_id)
    finally:
        session.close()


def _attempt(client, tmp, procedure, color="blue"):
    return _upload(client, procedure, _video(tmp / f"{color}.mp4", color), "execution")


def _batch(client, **body):
    response = client.post("/api/reliability/batches", json={"allow_live": True, **body})
    assert response.status_code == 200, response.text
    return response.json()


def _cache_files(tmp):
    root = tmp / "inference_cache"
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.json")) if root.exists() else []


# ---------------------------------------------------------------- concurrency


def test_runs_overlap_up_to_the_limit_and_each_is_independent(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    before_calls = gauge.calls
    cache_before = _cache_files(tmp)
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=5, max_concurrency=2)
    assert batch["status"] == "queued" and len(batch["runs"]) == 5
    gauge.peak = 0
    done = _drive(batch["id"])

    assert done["status"] == "completed"
    assert [run["status"] for run in done["runs"]] == ["completed"] * 5
    # Concurrency: two in flight at once, never three.
    assert gauge.peak == 2
    # Independence: one fresh guided request per repetition, none replayed, nothing stored for replay.
    assert gauge.calls - before_calls == 5
    assert _cache_files(tmp) == cache_before
    analysis_ids = {run["analysis_run_id"] for run in done["runs"]}
    verification_ids = {run["verification_id"] for run in done["runs"]}
    assert len(analysis_ids) == 5 and len(verification_ids) == 5
    for run in done["runs"]:
        inference = run["result"]["execution_metadata"]["inference"]
        assert inference["guided_call"] == "live" and inference["fully_fresh"] and inference["cache_policy"] == "isolated"
        assert (tmp / "runs" / run["analysis_run_id"] / "working.json").exists()
    metrics = done["metrics"]
    assert metrics["consistency"]["distribution"] == {"PASS": 5, "FAIL": 0, "INCONCLUSIVE": 0}
    assert metrics["consistency"]["modal_agreement"]["value"] == 1.0
    assert metrics["independence"]["fully_fresh_runs"] == 5
    assert metrics["performance"]["runtime_seconds"]["count"] == 5
    assert metrics["performance"]["batch_wall_clock_seconds"] is not None


def test_queued_runs_start_only_when_a_slot_frees(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    gauge.peak = 0
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=3, max_concurrency=1)
    done = _drive(batch["id"])
    assert gauge.peak == 1
    runs = sorted(done["runs"], key=lambda item: item["index"])
    assert all(run["status"] == "completed" for run in runs)
    for earlier, later in zip(runs, runs[1:]):
        assert later["started_at"] >= earlier["finished_at"]


def test_the_ordinary_worker_never_claims_an_experiment_job(slow):
    from app.db import SessionLocal
    from app.jobs.worker import MAIN_LANE, RELIABILITY_LANE, claim_next
    from app.reliability.executor import launch_queued

    client, stub, tmp, _gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=1, max_concurrency=1)
    session = SessionLocal()
    try:
        assert launch_queued(session) == 1
    finally:
        session.close()
    assert claim_next("main", MAIN_LANE) is None
    assert claim_next("lab", RELIABILITY_LANE) is not None
    assert batch["id"]


# ---------------------------------------------------------------- independence and the shared caches


def test_independent_runs_do_not_read_or_change_stored_answers(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    # An ordinary analysis of this attempt stores its answer.
    ordinary = _analyze(client, video)
    assert ordinary["source"] == "live"
    stored = {name: (tmp / "inference_cache" / name).read_text() for name in _cache_files(tmp)}
    calls = gauge.calls
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=3, max_concurrency=3)["id"])
    assert gauge.calls - calls == 3, "an independent repetition replayed a stored answer"
    assert {name: (tmp / "inference_cache" / name).read_text() for name in _cache_files(tmp)} == stored
    # Ordinary behaviour is unchanged: the attempt still replays its own stored answer.
    replay = client.post(f"/api/learnloop/recordings/{video}/analyze", json={"allow_live": False}).json()
    from tests.test_learnloop_flow import _run_job

    _run_job(replay["id"])
    assert client.get(f"/api/learnloop/runs/{replay['id']}").json()["source"] == "cached"
    assert all(run["result"]["execution_metadata"]["inference"]["fully_fresh"] for run in done["runs"])


def test_a_replay_batch_says_its_answers_were_reused(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    _analyze(client, video)
    calls = gauge.calls
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2, independent_inference=False)["id"])
    assert gauge.calls == calls
    for run in done["runs"]:
        inference = run["result"]["execution_metadata"]["inference"]
        assert inference["guided_call"] == "cached" and not inference["fully_fresh"] and inference["replayed_answers"] >= 1
    assert done["metrics"]["independence"]["fully_fresh_runs"] == 0
    assert done["config"]["shared_deterministic"] == []


def test_independent_batches_need_approval_and_the_server_gate(slow, monkeypatch):
    from app.config import settings

    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    body = {"project_id": procedure, "video_id": video, "template_id": template["id"], "repetitions": 3}
    refused = client.post("/api/reliability/batches", json={**body, "allow_live": False})
    assert refused.status_code == 409 and refused.json()["detail"]["plan"]["repetitions"] == 3
    monkeypatch.setattr(settings, "live_inference", False)
    assert client.post("/api/reliability/batches", json={**body, "allow_live": True}).status_code == 409
    assert client.get("/api/reliability/batches").json() == []


def test_isolated_runs_skip_the_sam_and_second_look_caches(lab, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from app.pipeline import guided, object_views
    from app.services import reinspection

    isolated = SimpleNamespace(config_json={"cache_policy": guided.ISOLATED})
    ordinary = SimpleNamespace(config_json={})
    object_views._store("k1", "answer", isolated)
    assert object_views._cached("k1") is None
    object_views._store("k1", "answer", ordinary)
    assert object_views._cached("k1") == "answer"
    assert guided.shares_cache({}) and not guided.shares_cache({"cache_policy": "isolated"})
    # The second look reads the cache only for runs that share it.
    import inspect

    source = inspect.getsource(reinspection._inspect)
    assert "shares_cache(run.config_json)" in source


# ---------------------------------------------------------------- failures, cancellation, restarts


def test_one_failed_run_does_not_stop_the_others_and_is_not_a_verdict(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    stub.fail_on = {gauge.calls + 2}
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=4, max_concurrency=1)["id"])
    statuses = sorted(run["status"] for run in done["runs"])
    assert statuses == ["completed", "completed", "completed", "failed"]
    failed = next(run for run in done["runs"] if run["status"] == "failed")
    assert failed["error_kind"] == "rate_limited" and "429" in failed["error"]
    assert failed["result"]["outcome"] == "execution_error"
    metrics = done["metrics"]
    assert metrics["counts"]["execution_errors"] == 1 and metrics["counts"]["valid"] == 3
    assert metrics["consistency"]["valid_runs"] == 3 and metrics["consistency"]["distribution"]["INCONCLUSIVE"] == 0
    assert metrics["performance"]["execution_error_rate"] == {"value": 0.25, "numerator": 1, "denominator": 4}


def test_cancel_stops_queued_and_running_runs_through_the_pipeline(slow):
    from app.db import SessionLocal
    from app.models.runs import AnalysisRun
    from app.reliability.executor import launch_queued

    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=3, max_concurrency=1)
    session = SessionLocal()
    try:
        launch_queued(session)
    finally:
        session.close()
    canceled = client.post(f"/api/reliability/batches/{batch['id']}/cancel").json()
    assert canceled["status"] == "canceling"
    running = next(run for run in canceled["runs"] if run["analysis_run_id"])
    session = SessionLocal()
    try:
        assert session.get(AnalysisRun, running["analysis_run_id"]).status == "canceled"
    finally:
        session.close()
    calls = gauge.calls
    done = _drive(batch["id"])
    assert done["status"] == "canceled"
    assert [run["status"] for run in done["runs"]] == ["canceled"] * 3
    assert gauge.calls == calls
    assert done["metrics"]["counts"]["canceled"] == 3 and done["metrics"]["consistency"]["valid_runs"] == 0


def test_a_run_interrupted_by_a_restart_fails_and_the_rest_continue(slow):
    from app.db import SessionLocal
    from app.jobs.worker import RELIABILITY_LANE, claim_next, recover_incomplete
    from app.reliability.executor import launch_queued

    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=1)
    session = SessionLocal()
    try:
        launch_queued(session)
    finally:
        session.close()
    # The job is claimed, then the process dies before it finishes.
    assert claim_next("lab", RELIABILITY_LANE) is not None
    assert recover_incomplete() == 1
    done = _drive(batch["id"])
    first, second = sorted(done["runs"], key=lambda item: item["index"])
    assert first["status"] == "failed" and first["error_kind"] == "interrupted"
    assert second["status"] == "completed", second["error"]
    # The batch is read back from the database, as after a reload.
    again = client.get(f"/api/reliability/batches/{batch['id']}").json()
    assert again["runs"][1]["result"]["verdict"] == "PASS"


# ---------------------------------------------------------------- results, ground truth, isolation from ordinary views


def test_ground_truth_recalculates_accuracy_without_running_again(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure, "green")
    stub.text = OMITTED
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2)["id"])
    assert done["metrics"]["accuracy"] == {"labeled": False, "expected": None}
    assert {run["result"]["verdict"] for run in done["runs"]} == {"FAIL"}
    calls = gauge.calls
    step_ids = [step["step_id"] for step in done["steps"]]
    failing = next(step["step_id"] for step in done["runs"][0]["result"]["steps"] if step["verdict"] == "FAIL")
    labels = {step_id: ("FAIL" if step_id == failing else "PASS") for step_id in step_ids}
    labeled = client.put(f"/api/reliability/batches/{done['id']}/ground-truth", json={"expected_verdict": "FAIL", "step_labels": labels}).json()
    accuracy = labeled["metrics"]["accuracy"]
    assert accuracy["exact_match"]["value"] == 1.0 and accuracy["false_pass"]["value"] == 0.0
    assert accuracy["false_fail"]["applicable"] is False
    step = next(item for item in labeled["metrics"]["steps"] if item["step_id"] == failing)
    assert step["accuracy"]["value"] == 1.0
    relabeled = client.put(f"/api/reliability/batches/{done['id']}/ground-truth", json={"expected_verdict": "PASS"}).json()
    assert relabeled["metrics"]["accuracy"]["exact_match"]["value"] == 0.0
    assert relabeled["metrics"]["accuracy"]["false_fail"]["value"] == 1.0
    assert gauge.calls == calls
    bad = client.put(f"/api/reliability/batches/{done['id']}/ground-truth", json={"step_labels": {"nope": "PASS"}})
    assert bad.status_code == 400


def test_experiments_stay_out_of_the_ordinary_procedure_views(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    before = client.get(f"/api/learnloop/procedures/{procedure}").json()
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2)["id"])
    after = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert after["verifications"] == before["verifications"] == []
    attempt = next(item for item in after["recordings"] if item["video"]["id"] == video)
    assert attempt["run"] is None and attempt["analyzed_before"] is False
    assert after["latest_outcome"] == before["latest_outcome"]
    assert len(after["templates"]) == len(before["templates"])
    # The repetitions' own reports stay readable for drill-down.
    report = client.get(f"/api/verifications/{done['runs'][0]['verification_id']}").json()
    assert report["overall"] == "PASS"


def test_a_second_look_runs_on_the_lane_and_the_run_waits_for_it(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure, "white")
    stub.text = HIDDEN
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2)["id"])
    for run in done["runs"]:
        assert run["status"] == "completed"
        reinspection = run["result"]["diagnostics"]["reinspection"]
        assert reinspection["status"] in {"completed", "failed"} and not reinspection.get("incomplete")
        assert run["result"]["verdict"] == "INCONCLUSIVE"
    assert done["metrics"]["consistency"]["distribution"]["INCONCLUSIVE"] == 2


def test_export_has_one_row_per_run(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    stub.text = CORRECT
    done = _drive(_batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2)["id"])
    text = client.get(f"/api/reliability/batches/{done['id']}/export?format=csv").text
    lines = text.strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("run,status,outcome,verdict")
    assert client.get(f"/api/reliability/batches/{done['id']}/export").json()["id"] == done["id"]
    listed = client.get("/api/reliability/batches").json()
    assert listed[0]["id"] == done["id"] and listed[0]["summary"]["modal_agreement"] == 1.0


# ---------------------------------------------------------------- learning mode


def test_learning_batch_compares_procedures_without_adding_any(slow):
    from app.db import SessionLocal
    from app.models.verification import LearnedWorkflowTemplate

    client, stub, tmp, gauge = slow
    procedure = client.post("/api/learnloop/procedures", json={"name": "Cube"}).json()["id"]
    video = _upload(client, procedure, _video(tmp / "demo.mp4", "red"), "demonstration")
    stub.text = DEMONSTRATION
    calls = gauge.calls
    done = _drive(_batch(client, mode="learning", project_id=procedure, video_id=video, repetitions=3, max_concurrency=3)["id"])
    assert gauge.calls - calls == 3 and gauge.peak >= 2
    assert all(run["status"] == "completed" and run["learned"]["steps"] for run in done["runs"])
    metrics = done["metrics"]
    assert metrics["valid_runs"] == 3 and len(metrics["step_count_distribution"]) == 1
    assert metrics["steps_in_every_run"] >= 1 and metrics["mean_pairwise_label_overlap"] == 1.0
    assert metrics["reviewable"]["value"] == 1.0
    session = SessionLocal()
    try:
        assert session.scalar(select(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.project_id == procedure)) is None
    finally:
        session.close()


def test_learning_alignment_uses_conditions_not_positions():
    from app.reliability.learning import learning_metrics, normalize_label

    assert normalize_label("The Red Bricks!") == "red brick" and normalize_label("glass") == "glass"

    def learned(order, names, extra=None):
        roles = [{"role_id": "r1", "label": "Red brick"}, {"role_id": "r2", "label": "Base plate"}, {"role_id": "r3", "label": "Blue brick"}]
        steps = {
            "red": {"step_id": "s-red", "name": names[0], "completion_conditions": [{"subject_role": "r1", "target_role": "r2", "relation": "attached"}], "demo_time_start": 1.0},
            "blue": {"step_id": "s-blue", "name": names[1], "completion_conditions": [{"subject_role": "r3", "target_role": "r2", "relation": "attached"}], "demo_time_start": 4.0},
        }
        out = [steps[key] for key in order] + (extra or [])
        return {"roles": roles, "steps": out}

    def run(index, doc):
        return {"index": index, "status": "completed", "learned": doc, "result": {"outcome": "verdict", "step_count": len(doc["steps"]), "object_count": 3, "role_count": 3, "validation": {"errors": 0}, "reviewable": True}}

    runs = [
        run(1, learned(["red", "blue"], ["Attach red", "Attach blue"])),
        run(2, learned(["blue", "red"], ["Put blue on", "Put red on"])),
        run(3, learned(["red"], ["Attach red", ""], [{"step_id": "x", "name": "Look around", "completion_conditions": []}])),
    ]
    metrics = learning_metrics(runs, None, None)
    by_signature = {item["signature"]: item for item in metrics["aligned_steps"]}
    red = by_signature["red brick attached base plate"]
    blue = by_signature["blue brick attached base plate"]
    assert red["runs"] == [1, 2, 3] and red["in_every_run"] and red["positions"] == [1, 2]
    assert blue["runs"] == [1, 2] and not blue["in_every_run"]
    assert red["start_seconds"]["range"] == 0.0
    assert metrics["unaligned_steps"] == [{"run": 3, "position": 2, "name": "Look around", "reason": "no checkable condition"}]
    assert metrics["step_count_distribution"] == {2: 3}


# ---------------------------------------------------------------- metric arithmetic


def _run(status="completed", verdict=None, steps=None, error_kind="", duration=10.0):
    result = {}
    if status == "completed":
        result = {"outcome": "verdict" if verdict else "invalid_result", "verdict": verdict, "steps": steps or []}
    return {"status": status, "error_kind": error_kind, "duration_seconds": duration, "started_at": "x", "result": result}


def test_consistency_is_modal_agreement_over_runs_with_a_verdict():
    from app.reliability.metrics import consistency, counts

    runs = [_run(verdict="PASS")] * 8 + [_run(verdict="INCONCLUSIVE")] * 2 + [_run("failed", error_kind="pipeline_error"), _run("canceled")]
    result = consistency(runs)
    assert result["distribution"] == {"PASS": 8, "FAIL": 0, "INCONCLUSIVE": 2}
    assert result["modal_agreement"] == {"value": 0.8, "numerator": 8, "denominator": 10}
    tally = counts(runs)
    assert tally["valid"] == 10 and tally["execution_errors"] == 1 and tally["canceled"] == 1
    assert consistency([])["modal_agreement"]["value"] is None


def test_accuracy_denominators_follow_the_label():
    from app.reliability.metrics import accuracy

    runs = [_run(verdict="PASS")] * 6 + [_run(verdict="FAIL")] * 2 + [_run(verdict="INCONCLUSIVE")] * 2 + [_run("failed")]
    passing = accuracy(runs, {"expected_verdict": "PASS"})
    assert passing["exact_match"] == {"value": 0.6, "numerator": 6, "denominator": 10}
    assert passing["decisive_accuracy"] == {"value": 0.75, "numerator": 6, "denominator": 8}
    assert passing["decisive_coverage"]["value"] == 0.8 and passing["inconclusive_rate"]["value"] == 0.2
    assert passing["false_fail"]["value"] == 0.2 and passing["false_pass"]["value"] is None and not passing["false_pass"]["applicable"]
    failing = accuracy(runs, {"expected_verdict": "FAIL"})
    assert failing["false_pass"] == {"value": 0.6, "numerator": 6, "denominator": 10, "applicable": True}
    assert failing["exact_match"]["value"] == 0.2
    assert accuracy(runs, {}) == {"labeled": False, "expected": None}
    assert accuracy([], {"expected_verdict": "PASS"})["exact_match"]["value"] is None


def test_per_step_figures_handle_missing_steps_and_unlabeled_steps():
    from app.reliability.metrics import per_step

    def step(step_id, verdict, evidence=1, warnings=()):
        return {"step_id": step_id, "verdict": verdict, "evidence_count": evidence, "missing_evidence": evidence == 0, "warnings": list(warnings), "required": True}

    runs = [
        _run(verdict="PASS", steps=[step("s1", "PASS"), step("s2", "PASS")]),
        _run(verdict="INCONCLUSIVE", steps=[step("s1", "PASS"), step("s2", "CANNOT_VERIFY", 0, ["review_required"])]),
        _run(verdict="FAIL", steps=[step("s1", "PASS")]),
        _run("failed"),
    ]
    rows = {row["step_id"]: row for row in per_step(runs, [{"step_id": "s1", "name": "One"}, {"step_id": "s2", "name": "Two"}], {"step_labels": {"s2": "PASS"}})}
    assert rows["s1"]["unstable"] is False and rows["s1"]["agreement"]["value"] == 1.0 and rows["s1"]["accuracy"] is None
    assert rows["s2"]["runs_with_step"] == 2 and rows["s2"]["runs_missing_step"] == 1
    assert rows["s2"]["unstable"] and rows["s2"]["agreement"]["value"] == 0.5
    assert rows["s2"]["missing_evidence"]["value"] == 0.5 and rows["s2"]["with_warnings"]["value"] == 0.5
    assert rows["s2"]["accuracy"] == {"value": 0.5, "numerator": 1, "denominator": 2}


def test_a_report_without_a_verdict_is_invalid_not_inconclusive():
    from app.reliability.normalize import classify_error, overall_verdict

    assert overall_verdict("INCONCLUSIVE") == "INCONCLUSIVE"
    assert overall_verdict("INCOMPLETE") is None and overall_verdict("NOT_EVALUATED") is None and overall_verdict(None) is None
    assert classify_error("Error code: 429") == "rate_limited"
    assert classify_error("CUDA out of memory") == "resource_exhausted"
    assert classify_error("worker stopped before the attempt finished") == "interrupted"
    assert classify_error("boom") == "pipeline_error"


def test_shared_encoder_loads_once_under_concurrent_first_use(monkeypatch):
    from app.perception import encoder

    loads = []

    def load(encoder_id):
        loads.append(encoder_id)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(encoder, "load_encoder", load)
    monkeypatch.setattr(encoder, "_SHARED", {})
    threads = [threading.Thread(target=encoder.shared_encoder, args=("x",)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert loads == ["x"]


def test_batch_validation_rejects_bad_requests(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    video = _attempt(client, tmp, procedure)
    base = {"project_id": procedure, "video_id": video, "template_id": template["id"], "allow_live": True}
    assert client.post("/api/reliability/batches", json={**base, "max_concurrency": 99}).status_code == 400
    assert client.post("/api/reliability/batches", json={**base, "repetitions": 0}).status_code == 400
    assert client.post("/api/reliability/batches", json={**base, "template_id": "missing"}).status_code == 404
    draft = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": template["source_run_id"]}).json()
    assert draft["id"] == template["id"] or draft["acceptance_status"] != "accepted"
    json.dumps(client.get("/api/reliability/options").json())


def test_options_name_each_version_by_its_demonstration(slow):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    entry = next(item for item in client.get("/api/reliability/options").json()["procedures"] if item["id"] == procedure)
    listed = next(item for item in entry["templates"] if item["id"] == template["id"])
    assert listed["source_filename"] == "demo.mp4"
    assert listed["created_at"]


# ---------------------------------------------------------------- missing recordings


def test_a_referenced_recording_that_was_moved_is_named_not_run(slow, tmp_path):
    client, stub, tmp, gauge = slow
    procedure, template = _accepted((client, stub, tmp))
    source = _video(tmp_path / "elsewhere.mp4", "blue")
    response = client.post(f"/api/learnloop/procedures/{procedure}/recordings/reference", json={"path": str(source), "role": "execution"})
    assert response.status_code == 200, response.text
    video = response.json()["video"]["id"]
    stub.text = CORRECT
    batch = _batch(client, project_id=procedure, video_id=video, template_id=template["id"], repetitions=2, max_concurrency=2)
    source.unlink()
    calls = gauge.calls

    # Repetitions already queued fail at launch with a plain reason, before any model call.
    done = _drive(batch["id"])
    assert gauge.calls == calls
    for run in done["runs"]:
        assert run["status"] == "failed" and run["error_kind"] == "missing_recording"
        assert str(source) in run["error"]

    # A new experiment is refused, and the picker says why.
    response = client.post("/api/reliability/batches", json={"allow_live": True, "project_id": procedure, "video_id": video, "template_id": template["id"]})
    assert response.status_code == 400 and str(source) in response.json()["detail"]
    options = client.get("/api/reliability/options").json()
    recording = next(item for item in options["procedures"][0]["recordings"] if item["video_id"] == video)
    assert recording["missing"] is True
