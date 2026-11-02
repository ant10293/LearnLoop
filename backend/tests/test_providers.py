"""Provider capability routing refuses unsupported modes instead of substituting them."""

import pytest
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.catalog import lookup
from app.providers.openai_provider import OpenAIProvider
from app.providers.types import GenerationRequest, UnsupportedMode, estimate_cost


def test_openai_catalog_has_no_native_video():
    caps = lookup("openai", "gpt-4.1")
    assert caps.image is True
    assert caps.native_video is False
    assert caps.structured_output is True


def test_video_capable_models_are_not_sent_frames():
    from app.providers.model_list import use_video_file

    assert use_video_file("gemini", "gemini-2.5-pro") is True
    assert use_video_file("meta", "muse-spark-1.3") is True
    assert use_video_file("anthropic", "claude-sonnet-5-5") is False
    assert use_video_file("openai", "gpt-4.1") is False


def test_overlong_video_is_not_replaced_with_frames():
    from types import SimpleNamespace

    from app.pipeline.infer import _require_video_length

    run = SimpleNamespace(provider_id="gemini", model_id="gemini-2.5-flash")
    video = SimpleNamespace(duration_sec=4000)
    with pytest.raises(UnsupportedMode, match="not replaced with frames"):
        _require_video_length(run, video)


def test_openai_refuses_native_video():
    provider = OpenAIProvider()
    request = GenerationRequest(model_id="gpt-4.1", system_prompt="s", user_prompt="u", modality="native_video")
    with pytest.raises(UnsupportedMode):
        import asyncio

        asyncio.run(provider.generate("sk-test", request))


def test_anthropic_sends_reasoning_effort(monkeypatch):
    captured: dict = {}

    class _Usage:
        input_tokens = 3
        output_tokens = 2

    class _Block:
        type = "tool_use"
        name = "record_analysis"
        input = {"ok": True}

    class _Response:
        content = [_Block()]
        usage = _Usage()
        id = "msg_1"

        def model_dump(self):
            return {"id": "msg_1", "model": "claude-sonnet-5-5", "usage": {}, "stop_reason": "tool_use"}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Client:
        messages = _Messages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(AnthropicProvider, "_client", lambda self, secret, timeout: _Client())
    request = GenerationRequest(
        model_id="claude-sonnet-5-5",
        system_prompt="s",
        user_prompt="u",
        reasoning_effort="xhigh",
    )
    import asyncio

    asyncio.run(AnthropicProvider().generate("sk-test", request))
    assert captured["output_config"] == {"effort": "xhigh"}


def test_anthropic_refuses_effort_the_model_does_not_list():
    request = GenerationRequest(
        model_id="claude-sonnet-4-6",
        system_prompt="s",
        user_prompt="u",
        reasoning_effort="xhigh",
    )
    with pytest.raises(UnsupportedMode, match="not available"):
        import asyncio

        asyncio.run(AnthropicProvider().generate("sk-test", request))


def test_anthropic_refuses_native_video():
    provider = AnthropicProvider()
    request = GenerationRequest(
        model_id="claude-sonnet-4-5",
        system_prompt="s",
        user_prompt="u",
        modality="native_video",
    )
    with pytest.raises(UnsupportedMode):
        import asyncio

        asyncio.run(provider.generate("sk-test", request))


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


def test_estimate_cost_uses_catalog_list_rates():
    caps = lookup("openai", "gpt-4.1")
    cost, known = estimate_cost(caps, 1000, 500)
    assert known is True
    assert cost == 0.006


def test_unpriced_model_does_not_invent_a_zero_invoice():
    caps = lookup("meta", "muse-spark-1.3")
    cost, known = estimate_cost(caps, 1000, 500)
    assert known is False
    assert cost == 0.0


def _serve_models():
    """A keep-alive HTTP server, so a client holds an open socket after its request."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            body = json.dumps({"object": "list", "data": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_provider_client_closes_on_the_loop_that_opened_it(monkeypatch):
    """Each job runs on its own loop. A client left open is collected on a later
    loop, where aclose fails with "Event loop is closed"."""
    import asyncio
    import gc

    from app.providers.openai_provider import OpenAIProvider
    from openai import AsyncOpenAI

    server = _serve_models()
    created = []

    def build(self, secret, timeout):
        client = AsyncOpenAI(api_key=secret, base_url=f"http://127.0.0.1:{server.server_port}/v1", timeout=timeout)
        created.append(client)
        return client

    monkeypatch.setattr(OpenAIProvider, "_client", build)
    try:
        asyncio.run(OpenAIProvider().list_models("sk-test"))
        assert created[0].is_closed()
    finally:
        server.shutdown()

    failures = []

    async def later_loop():
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: failures.append(context))
        created.clear()
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(later_loop())
    gc.collect()
    assert failures == []


def _meta_response(**fields):
    from types import SimpleNamespace

    return SimpleNamespace(id="resp_1", output_text='{"actions":[]}', usage=None, **fields)


def _meta_generate(monkeypatch, response):
    import asyncio
    from types import SimpleNamespace

    from app.providers.meta_provider import MetaProvider
    from app.providers.types import GenerationRequest

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
