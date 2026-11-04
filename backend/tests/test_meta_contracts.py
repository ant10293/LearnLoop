"""Meta's output contract, offline. No test here needs an API key or sends a request."""

import asyncio
import copy
import json
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


async def _no_locate(_session, _run, document, _frames):
    return document, None


def test_contract_schemas_are_not_shared_mutable_state():
    before = copy.deepcopy(SINGLE_PASS.schema)
    validate_strict_schema(SINGLE_PASS.schema)
    assert SINGLE_PASS.schema == before
