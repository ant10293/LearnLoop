"""Provider capability routing refuses unsupported modes instead of substituting them."""

from app.providers.catalog import lookup
from app.providers.types import GenerationRequest, estimate_cost


def test_muse_rejects_temperature_capability():
    caps = lookup("meta", "muse-spark-1.3")
    assert caps.native_video is True
    assert caps.supports_temperature is False
    assert "none" not in caps.reasoning_efforts
    assert "high" in caps.reasoning_efforts


def test_sam_is_not_an_analysis_model():
    caps = lookup("meta", "sam-3.1")
    assert caps.structured_output is False
    assert "action" not in caps.notes.lower() or "not an action" in caps.notes.lower()


def test_unpriced_model_does_not_invent_a_zero_invoice():
    caps = lookup("meta", "muse-spark-1.3")
    cost, known = estimate_cost(caps, 1000, 500)
    assert known is False
    assert cost == 0.0


def _meta_response(**fields):
    from types import SimpleNamespace

    return SimpleNamespace(id="resp_1", output_text='{"actions":[]}', usage=None, **fields)


def _meta_generate(monkeypatch, response):
    import asyncio
    from types import SimpleNamespace

    from app.providers.meta_provider import MetaProvider

    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return response

    class Client:
        responses = SimpleNamespace(create=create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    provider = MetaProvider()
    monkeypatch.setattr(provider, "_client", lambda *_: Client())
    request = GenerationRequest(model_id="muse-spark-1.3", system_prompt="s", user_prompt="u")
    return asyncio.run(provider.generate("key", request)), sent


def test_meta_caps_output_and_reports_a_truncated_answer(monkeypatch):
    from types import SimpleNamespace

    from app.providers.meta_provider import MAX_OUTPUT_TOKENS

    truncated = _meta_response(
        status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")
    )
    result, sent = _meta_generate(monkeypatch, truncated)
    assert sent["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert result.incomplete_reason == "max_output_tokens"
    assert result.raw["status"] == "incomplete"


def test_meta_finished_answer_is_not_marked_incomplete(monkeypatch):
    result, _ = _meta_generate(monkeypatch, _meta_response(status="completed"))
    assert result.incomplete_reason == ""


def test_interpretation_schema_forbids_extra_keys_and_requires_the_lists():
    from app.schemas.analysis import InterpretationResponse

    schema = InterpretationResponse.model_json_schema()
    assert schema["additionalProperties"] is False
    assert {"actions", "workflow_steps"} <= set(schema["required"])
    assert "objects" not in schema["properties"]
