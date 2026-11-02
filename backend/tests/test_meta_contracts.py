"""Meta's output contract, offline. No test here needs an API key or sends a request."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.pipeline.contracts import (
    ALL_CONTRACTS,
    DISCOVERY,
    INTERPRETATION,
    SINGLE_PASS,
    SchemaContractError,
    StructuredOutputError,
    contract_for,
    detect_runaway,
    validate_output,
    validate_strict_schema,
)
from app.pipeline.parsing import parse_analysis
from app.providers.types import GenerationRequest

from tests.test_pipeline import client  # noqa: F401


def evidence(kind="DIRECTLY_OBSERVED"):
    return {"classification": kind, "note": "seen", "frame_timestamps": [1.0], "time_start": None, "time_end": None}


def obj(local_id, label):
    return {
        "local_id": local_id,
        "label": label,
        "description": "",
        "visual_characteristics": "black with a pale hub",
        "initial_state": "loose",
        "final_state": "attached",
        "uncertain_identity": True,
        "manipulated": True,
        "confidence": 0.8,
        "instance_id": None,
        "identity_relation": "provisional",
    }


def observation(local_id, timestamp, box=True, visibility="visible"):
    return {
        "object_local_id": local_id,
        "timestamp": timestamp,
        "bbox": {"x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1} if box else None,
        "visibility": visibility,
        "description": "",
        "evidence": evidence(),
        "occlusion": None,
        "confidence": 0.9 if box else None,
        "continuity": "continuous",
        "track_ref": None,
    }


def semantics():
    return {
        "summary": "Two wheels are attached.",
        "states": [
            {
                "object_local_id": "Object_1",
                "time_start": 0.0,
                "time_end": 1.0,
                "state_label": "loose",
                "description": "",
                "evidence": evidence(),
            }
        ],
        "transitions": [
            {
                "object_local_id": "Object_1",
                "timestamp": 1.0,
                "from_state": "loose",
                "to_state": "held",
                "description": "",
                "evidence": evidence("INFERRED"),
                "certainty": None,
            }
        ],
        "relationships": [
            {
                "subject_local_id": "Object_1",
                "other_local_id": "Object_2",
                "kind": "near",
                "time_start": 0.0,
                "time_end": 1.0,
                "description": "",
                "evidence": evidence(),
            }
        ],
        "actions": [
            {
                "local_id": "Action_1",
                "action_type": "pick_up",
                "description": "Lift the first wheel.",
                "start": 0.5,
                "end": 1.5,
                "actor": "right hand",
                "object_local_ids": ["Object_1"],
                "initial_state": "loose",
                "final_state": "held",
                "outcome": "completed",
                "uncertainty": 0.2,
                "evidence": evidence(),
            }
        ],
        "workflow_steps": [
            {
                "local_id": step,
                "name": f"Attach {step}",
                "required_objects": ["Object_1"],
                "preconditions": ["Loose"],
                "expected_action": "Press it on.",
                "completion_condition": "It stays on.",
                "observed_result": "It stayed on.",
                "action_local_ids": ["Action_1"],
                "time_start": 0.5,
                "time_end": 1.5,
                "uncertainty": 0.3,
                "evidence": evidence(),
                "parallel_group": None,
            }
            for step in ("Step_1", "Step_2")
        ],
        "dependencies": [
            {
                "from_step": "Step_1",
                "to_step": "Step_2",
                "category": "observed_temporal",
                "rationale": "Order only.",
                "evidence": evidence("INFERRED"),
            }
        ],
        "ambiguities": [
            {
                "local_id": "Ambiguity_1",
                "kind": "uncertain_identity",
                "description": "The two wheels look alike.",
                "time_start": None,
                "time_end": None,
                "object_local_ids": ["Object_1", "Object_2"],
                "action_local_ids": [],
                "alternative": "",
                "evidence": evidence("UNKNOWN"),
            }
        ],
    }


def geometry():
    return {
        "camera": None,
        "objects": [obj("Object_1", "wheel"), obj("Object_2", "wheel")],
        "observations": [
            observation("Object_1", 0.0),
            observation("Object_1", 1.0, box=False, visibility="occluded"),
            observation("Object_2", 0.0),
        ],
    }


def single_pass():
    return {**geometry(), **semantics()}


def interpretation():
    return semantics()


def discovery():
    return {**geometry(), "summary": "", "ambiguities": semantics()["ambiguities"]}


def runaway_tail(count=3000):
    return ", ".join(f'"relationships_word{index}": []' for index in range(count))


# Schema correctness


@pytest.mark.parametrize("contract", ALL_CONTRACTS, ids=lambda item: item.name)
def test_every_contract_meets_the_strict_subset(contract):
    validate_strict_schema(contract.schema)
    text = json.dumps(contract.schema)
    for keyword in ('"allOf"', '"oneOf"', '"default"', '"title"', '"minimum"', '"maximum"'):
        assert keyword not in text

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(contract.schema)


@pytest.mark.parametrize("contract", ALL_CONTRACTS, ids=lambda item: item.name)
def test_internal_fields_are_not_model_facing(contract):
    text = json.dumps(contract.schema)
    for field in (
        "perception",
        "coverage",
        "raw_observations",
        "validation_issues",
        "accepted",
        "provenance",
        "geometry_origin",
        "association_status",
        "mask_ref",
    ):
        assert f'"{field}"' not in text


def test_each_stage_only_receives_its_own_lists():
    assert set(DISCOVERY.schema["properties"]) == {"summary", "camera", "objects", "observations", "ambiguities"}
    assert not {"objects", "observations"} & set(INTERPRETATION.schema["properties"])
    assert {"objects", "observations", "actions", "workflow_steps"} <= set(SINGLE_PASS.schema["properties"])


def test_contract_selection_follows_the_call():
    assert contract_for("meta", "scene_analysis", "full", "single") is SINGLE_PASS
    assert contract_for("meta", "scene_analysis", "discovery", "single") is DISCOVERY
    assert contract_for("meta", "workflow", "interpretation", "single") is INTERPRETATION
    assert contract_for("meta", "scene_analysis", "full", "multi") is DISCOVERY
    assert set(contract_for("meta", "workflow", "full", "multi").schema["properties"]) == {
        "workflow_steps",
        "dependencies",
    }
    assert contract_for("openai", "scene_analysis", "full", "single") is None
    with pytest.raises(SchemaContractError, match="not sent unconstrained"):
        contract_for("meta", "made_up_stage", "full", "multi")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.pop("additionalProperties"), "additionalProperties"),
        (lambda s: s["required"].remove("a"), "require every property"),
        (lambda s: s["properties"].update(b={"allOf": []}), "unsupported keywords"),
        (lambda s: s.update(oneOf=[]), "cannot use oneOf"),
        (lambda s: s["properties"].update(b={"type": "number", "minimum": 0}), "unsupported keywords"),
        (lambda s: s["properties"].update(b={"type": "object", "additionalProperties": False}), "no declared"),
    ],
)
def test_validator_rejects_schemas_outside_the_subset(change, message):
    schema = {"type": "object", "additionalProperties": False, "required": ["a"], "properties": {"a": {"type": "string"}}}
    change(schema)
    if "b" in schema["properties"]:
        schema["required"].append("b")
    with pytest.raises(SchemaContractError, match=message):
        validate_strict_schema(schema)


def test_validator_rejects_recursion_and_excess_depth():
    recursive = {
        "type": "object",
        "additionalProperties": False,
        "required": ["next"],
        "properties": {"next": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "additionalProperties": False,
                "required": ["next"],
                "properties": {"next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}},
            }
        },
    }
    with pytest.raises(SchemaContractError, match="recursive"):
        validate_strict_schema(recursive)
    deep = {"type": "string"}
    for _ in range(11):
        deep = {"type": "object", "additionalProperties": False, "required": ["a"], "properties": {"a": deep}}
    with pytest.raises(SchemaContractError, match="nests 11"):
        validate_strict_schema(deep)


def test_every_other_schema_sent_to_meta_is_strict():
    from app.pipeline.frame_boxes import FRAME_BOX_SCHEMA
    from app.pipeline.locate_pass import _SCHEMA as LOCATE_SCHEMA

    validate_strict_schema(FRAME_BOX_SCHEMA)
    validate_strict_schema(LOCATE_SCHEMA)


# Realistic answers


def test_realistic_answer_survives_conversion():
    document = validate_output(SINGLE_PASS, json.dumps(single_pass()))
    assert [item.label for item in document.objects] == ["wheel", "wheel"]
    assert document.objects[0].instance_id is None
    assert document.observations[1].bbox is None
    assert document.observations[1].visibility.value == "occluded"
    assert len(document.states) == len(document.transitions) == len(document.relationships) == 1
    assert document.actions[0].actor == "right hand"
    assert [step.local_id for step in document.workflow_steps] == ["Step_1", "Step_2"]
    assert document.dependencies[0].category.value == "observed_temporal"
    assert document.ambiguities[0].object_local_ids == ["Object_1", "Object_2"]
    assert document.actions[0].evidence.classification.value == "DIRECTLY_OBSERVED"
    assert document.actions[0].evidence.provenance.value == "AI_PREDICTION"


def test_explicit_empty_lists_are_a_valid_answer():
    empty = {key: [] for key in ("actions", "states", "transitions", "relationships")}
    empty |= {"workflow_steps": [], "dependencies": [], "ambiguities": [], "summary": ""}
    assert validate_output(INTERPRETATION, json.dumps(empty)).actions == []


def test_a_missing_required_list_is_not_an_empty_list():
    answer = interpretation()
    del answer["workflow_steps"]
    with pytest.raises(StructuredOutputError) as caught:
        validate_output(INTERPRETATION, json.dumps(answer))
    assert caught.value.code == "missing_fields"
    assert caught.value.diagnostics["missing_examples"] == ["workflow_steps"]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda a: a.update(actions_collapsed=[]), "unexpected_keys"),
        (lambda a: a["actions"][0].update(visibility_status="visible"), "unexpected_keys"),
        (lambda a: a["actions"][0]["evidence"].update(summary="seen"), "unexpected_keys"),
        (lambda a: a["actions"][0].pop("start"), "missing_fields"),
        (lambda a: a["actions"][0].update(start="soon"), "schema_violation"),
        (lambda a: a["actions"][0].update(outcome="maybe"), "schema_violation"),
        (lambda a: a["actions"][0].update(uncertainty=4), "schema_violation"),
    ],
)
def test_contract_violations_are_named(change, code):
    answer = interpretation()
    change(answer)
    with pytest.raises(StructuredOutputError) as caught:
        validate_output(INTERPRETATION, json.dumps(answer), label="workflow call resp_1")
    assert caught.value.code == code
    assert "workflow call resp_1" in str(caught.value)


def test_a_box_outside_the_frame_is_rejected_not_clamped():
    answer = single_pass()
    answer["observations"][0]["bbox"]["w"] = 1.4
    with pytest.raises(StructuredOutputError, match="out of range"):
        validate_output(SINGLE_PASS, json.dumps(answer))


# Runaway output


def test_runaway_keys_in_finished_json_are_rejected_and_named():
    text = json.dumps(single_pass())[:-1] + ", " + runaway_tail() + "}"
    with pytest.raises(StructuredOutputError) as caught:
        validate_output(SINGLE_PASS, text)
    error = caught.value
    assert error.code == "unexpected_keys"
    assert error.raw_text == text
    assert error.diagnostics["runaway"]["stem"] == "relationships"
    assert error.diagnostics["runaway"]["distinct_keys"] == 3000
    assert error.diagnostics["unexpected_key_count"] == 3000


def test_truncated_runaway_cannot_become_a_partial_analysis():
    text = json.dumps(single_pass())[:-1] + ", " + runaway_tail() + ', "relationships_cut'
    with pytest.raises(StructuredOutputError) as caught:
        validate_output(SINGLE_PASS, text, incomplete_reason="max_output_tokens")
    assert caught.value.code == "incomplete_output"
    assert "relationships_*" in str(caught.value)
    with pytest.raises(StructuredOutputError) as malformed:
        validate_output(SINGLE_PASS, text)
    assert malformed.value.code == "malformed_json"


def test_normal_answer_is_not_called_a_runaway():
    assert detect_runaway(json.dumps(single_pass())) == {"detected": False}


def test_legacy_parser_still_reads_old_responses():
    old = {"objects_list": [{"local_id": "Object_1", "label": "wheel"}], "actions": []}
    assert parse_analysis(json.dumps(old)).objects[0].label == "wheel"


# The request Meta receives


def _meta(monkeypatch, response=None, error=None, model="muse-spark-1.3", schema=None):
    from app.providers.meta_provider import MetaProvider

    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        if error is not None:
            raise error
        return response

    class Client:
        responses = SimpleNamespace(create=create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    provider = MetaProvider()
    monkeypatch.setattr(provider, "_client", lambda *_: Client())
    request = GenerationRequest(
        model_id=model,
        system_prompt="s",
        user_prompt="u",
        json_schema=schema or INTERPRETATION.schema,
        schema_name=INTERPRETATION.name,
    )
    return asyncio.run(provider.generate("key", request)), sent


def _response(status="completed", text='{"a":1}', **fields):
    return SimpleNamespace(id="resp_1", output_text=text, usage=None, status=status, **fields)


def test_meta_sends_strict_output_with_the_contract_schema(monkeypatch):
    _result, sent = _meta(monkeypatch, _response())
    assert sent["text"]["format"] == {
        "type": "json_schema",
        "name": "learnloop_interpretation",
        "schema": INTERPRETATION.schema,
        "strict": True,
    }


def test_meta_refuses_to_send_a_permissive_schema(monkeypatch):
    with pytest.raises(SchemaContractError, match="additionalProperties"):
        _meta(monkeypatch, _response(), schema={"type": "object", "properties": {"a": {"type": "string"}}})


def test_meta_does_not_drop_the_schema_for_a_model_without_structured_output(monkeypatch):
    from dataclasses import replace

    from app.providers import meta_provider
    from app.providers.types import UnsupportedMode

    real = meta_provider.capabilities_for
    monkeypatch.setattr(
        meta_provider, "capabilities_for", lambda *args: replace(real(*args), structured_output=False)
    )
    with pytest.raises(UnsupportedMode, match="not sent unconstrained"):
        _meta(monkeypatch, _response())


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_response("incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")), "max_output_tokens"),
        (_response("failed"), "status_failed"),
        (_response(""), "missing_status"),
        (_response(text="  "), "empty_output"),
        (
            _response(output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])]),
            "refusal",
        ),
    ],
)
def test_meta_reports_every_unfinished_terminal_state(monkeypatch, response, reason):
    result, _sent = _meta(monkeypatch, response)
    assert result.incomplete_reason == reason


def test_a_provider_schema_rejection_is_named_and_not_retried(monkeypatch):
    class BadRequest(Exception):
        status_code = 400

    with pytest.raises(SchemaContractError, match="Meta rejected the learnloop_interpretation schema"):
        _meta(monkeypatch, error=BadRequest("invalid text.format schema"))


# Pipeline


class FakeMeta:
    def __init__(self, answers):
        self.answers = answers
        self.requests = []

    async def generate(self, _secret, request):
        from app.providers.types import ProviderResponse, Usage

        self.requests.append(request)
        answer = self.answers.get(request.schema_name, '{"observations": []}')
        reason = ""
        if isinstance(answer, tuple):
            answer, reason = answer
        return ProviderResponse(
            text=answer if isinstance(answer, str) else json.dumps(answer),
            raw={"id": "resp_1", "status": "incomplete" if reason else "completed"},
            usage=Usage(10, 20, 0.0, False),
            request_id="resp_1",
            model_id=request.model_id,
            incomplete_reason=reason,
        )


@pytest.fixture()
def meta_run(client, tmp_path: Path, monkeypatch):  # noqa: F811
    from app.demo.encode import write_clip
    from app.demo.scenarios import SCENARIOS, render_scenario
    from app.pipeline import infer, runner
    from app.pipeline.locate_pass import locate_missing

    clip = tmp_path / "fixed.mp4"
    write_clip(render_scenario(SCENARIOS["fixed_camera"]()), clip)
    project = client.post("/api/projects", json={"name": "Bench"}).json()
    with clip.open("rb") as handle:
        video = client.post(
            "/api/videos", data={"project_id": project["id"]}, files={"file": ("fixed.mp4", handle, "video/mp4")}
        ).json()
    monkeypatch.setattr(infer, "load_secret", lambda *_: ("key", "test"))
    monkeypatch.setattr(runner, "locate_missing", _no_locate)
    assert locate_missing is not None

    def launch(answers, provider=None, **options):
        from app.db import SessionLocal
        from app.models.runs import Job
        from app.services.experiments import create_run
        from sqlalchemy import select

        fake = (provider or FakeMeta)(answers)
        monkeypatch.setattr(infer, "get_provider", lambda *_: fake)
        session = SessionLocal()
        run = create_run(
            session,
            {
                "video_id": video["id"],
                "provider_id": "meta",
                "model_id": "muse-spark-1.3",
                "research_mode": "fixed_camera",
                "segmentation_mode": "boxes_only",
                **options,
            },
        )
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        job.status = "running"
        job.attempt = 1
        session.commit()
        asyncio.run(runner.execute_job(session, job))
        session.refresh(run)
        session.refresh(job)
        return SimpleNamespace(session=session, run=run, job=job, fake=fake)

    return launch


async def _no_locate(_session, _run, document, _frames):
    return document, None


def test_single_pass_sends_its_contract_and_keeps_the_procedure(meta_run):
    outcome = meta_run({SINGLE_PASS.name: single_pass()}, analysis_pass="single", semantic_mode="combined")
    assert outcome.run.status == "completed", outcome.run.error
    first = outcome.fake.requests[0]
    assert first.schema_name == SINGLE_PASS.name
    from app.pipeline.runner import load_working

    document = load_working(outcome.run.id)
    assert [step.local_id for step in document.workflow_steps] == ["Step_1", "Step_2"]
    assert len(document.actions) == 1
    assert len(document.dependencies) == 1
    from app.models.runs import ModelRequestRow, ModelResponseRow
    from sqlalchemy import select

    row = outcome.session.scalars(select(ModelRequestRow).where(ModelRequestRow.run_id == outcome.run.id)).first()
    assert row.parameters_json["strict_output"] is True
    assert row.parameters_json["schema_name"] == SINGLE_PASS.name
    assert row.parameters_json["schema_hash"] == SINGLE_PASS.hash
    assert row.parameters_json["output_token_limit"] == 65_536
    saved = outcome.session.scalar(select(ModelResponseRow).where(ModelResponseRow.request_id == row.id))
    assert saved.raw_json["output_check"]["result"] == "valid"


def test_grounded_interpretation_cannot_replace_geometry(meta_run):
    outcome = meta_run(
        {DISCOVERY.name: discovery(), INTERPRETATION.name: interpretation()},
        analysis_pass="single",
        semantic_mode="grounded",
    )
    assert outcome.run.status == "completed", outcome.run.error
    names = [item.schema_name for item in outcome.fake.requests]
    assert names[0] == DISCOVERY.name
    assert INTERPRETATION.name in names
    from app.pipeline.runner import load_working

    document = load_working(outcome.run.id)
    assert {item.local_id for item in document.objects} == {"Object_1", "Object_2"}
    assert len(document.actions) == 1
    assert len(document.workflow_steps) == 2


def test_multi_pass_stages_each_get_their_own_slice(meta_run):
    answer = semantics()
    outcome = meta_run(
        {
            DISCOVERY.name: discovery(),
            "learnloop_actions": {"actions": answer["actions"]},
            "learnloop_states": {key: answer[key] for key in ("states", "transitions", "relationships")},
            "learnloop_workflow": {key: answer[key] for key in ("workflow_steps", "dependencies")},
            "learnloop_ambiguities": {"ambiguities": answer["ambiguities"]},
        },
        analysis_pass="multi",
        semantic_mode="combined",
    )
    assert outcome.run.status == "completed", outcome.run.error
    from app.pipeline.runner import load_working

    document = load_working(outcome.run.id)
    assert {item.local_id for item in document.objects} == {"Object_1", "Object_2"}
    assert len(document.actions) == 1
    assert len(document.states) == 1
    assert len(document.workflow_steps) == 2


def test_runaway_fails_once_keeps_evidence_and_publishes_nothing(meta_run):
    from app.models.runs import ModelRequestRow, ModelResponseRow
    from sqlalchemy import select

    text = json.dumps(single_pass())[:-1] + ", " + runaway_tail() + ', "relationships_cut'
    outcome = meta_run({SINGLE_PASS.name: (text, "max_output_tokens")}, analysis_pass="single", semantic_mode="combined")
    assert outcome.run.status == "failed"
    assert outcome.job.status == "failed", "a contract failure spends the same budget again, so it does not retry"
    assert "scene_analysis call resp_1" in outcome.run.error
    assert "max_output_tokens" in outcome.run.error
    assert "relationships_*" in outcome.run.error
    assert len(outcome.fake.requests) == 1
    row = outcome.session.scalar(select(ModelRequestRow).where(ModelRequestRow.run_id == outcome.run.id))
    saved = outcome.session.scalar(select(ModelResponseRow).where(ModelResponseRow.request_id == row.id))
    assert saved.text == text
    assert saved.raw_json["output_check"]["result"] == "incomplete_output"
    assert saved.raw_json["output_check"]["runaway"]["stem"] == "relationships"
    assert row.error
    assert outcome.run.active_revision_id is None
    assert outcome.run.status != "completed"


def test_the_surface_never_reaches_the_frame_roster_and_a_bad_window_is_reported(meta_run):
    answer = single_pass()
    answer["objects"].append(obj("Object_9", "purple flat lid or board"))
    answer["objects"][-1]["manipulated"] = False
    answer["observations"] += [
        {**observation("Object_9", time), "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.69}} for time in (0.0, 1.0, 2.0)
    ]
    outcome = meta_run(
        {SINGLE_PASS.name: answer, "learnloop_analysis": {"observations": [
            {"object_local_id": "1", "timestamp": 0.0, "visibility": "visible", "confidence": 0.9,
             "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}}]}},
        analysis_pass="single",
        semantic_mode="combined",
    )
    assert outcome.run.status == "completed", outcome.run.error
    frame_calls = [item for item in outcome.fake.requests if item.schema_name == "learnloop_analysis"]
    assert frame_calls
    for call in frame_calls:
        assert "Object_9" not in call.user_prompt
        allowed = call.json_schema["properties"]["observations"]["items"]["properties"]["object_local_id"]["enum"]
        assert "Object_9" not in allowed and "Object_1" in allowed
    from app.pipeline.runner import load_working

    document = load_working(outcome.run.id)
    assert "Object_9" not in {item.local_id for item in document.objects}
    assert "were not measured on frames" in document.coverage.temporal_coverage_note


def test_a_failed_provider_call_still_retries(meta_run):
    class Down(FakeMeta):
        async def generate(self, _secret, request):
            self.requests.append(request)
            raise RuntimeError("connection reset")

    outcome = meta_run({}, provider=Down, analysis_pass="single", semantic_mode="combined")
    assert outcome.run.status == "failed"
    assert outcome.job.status == "pending"
    assert "connection reset" in outcome.run.error


def test_a_person_can_retry_the_stage_after_a_contract_failure(client, meta_run, monkeypatch):  # noqa: F811
    from app.db import SessionLocal
    from app.models.runs import Job
    from app.pipeline import infer, runner
    from sqlalchemy import select

    failed = meta_run({SINGLE_PASS.name: ("{", "max_output_tokens")}, analysis_pass="single", semantic_mode="combined")
    assert failed.job.status == "failed"
    assert client.post(f"/api/runs/{failed.run.id}/retry/scene_analysis").status_code == 200
    monkeypatch.setattr(infer, "get_provider", lambda *_: FakeMeta({SINGLE_PASS.name: single_pass()}))
    session = SessionLocal()
    job = session.scalars(select(Job).where(Job.run_id == failed.run.id, Job.status == "pending")).first()
    job.status = "running"
    session.commit()
    asyncio.run(runner.execute_job(session, job))
    assert runner.load_working(failed.run.id).workflow_steps
    session.close()


def test_multi_window_calls_and_their_merge_all_use_the_contract(meta_run, monkeypatch):
    from dataclasses import replace

    from app.pipeline import runner
    from app.providers import catalog

    real = catalog.lookup
    monkeypatch.setattr(catalog, "lookup", lambda *args: replace(real(*args), max_images=3))
    outcome = meta_run(
        {SINGLE_PASS.name: single_pass()},
        analysis_pass="single",
        semantic_mode="combined",
        modality="sampled_frames",
    )
    assert outcome.run.status == "completed", outcome.run.error
    contract_calls = [item for item in outcome.fake.requests if item.schema_name == SINGLE_PASS.name]
    assert len(contract_calls) >= 3, "at least two windows and one merge"
    assert all(item.modality == "sampled_frames" and item.video_path is None for item in outcome.fake.requests)
    assert outcome.run.modality == "sampled_frames"
    assert runner.load_working(outcome.run.id).workflow_steps


def test_contract_schemas_are_not_shared_mutable_state():
    before = copy.deepcopy(SINGLE_PASS.schema)
    validate_strict_schema(SINGLE_PASS.schema)
    assert SINGLE_PASS.schema == before
