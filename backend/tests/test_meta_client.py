"""The httpx Meta client against a mocked transport. No network."""

import asyncio
import io
import json

import httpx
import pytest
from app.providers.common import translate
from app.providers.meta_client import MetaAPIError, MetaClient
from app.providers.types import AuthenticationError


def _client(handler) -> MetaClient:
    return MetaClient("key", timeout=5, transport=httpx.MockTransport(handler))


def test_response_joins_output_text_and_keeps_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"a":'}, {"type": "output_text", "text": "1}"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 2}},
            },
        )

    async def run():
        async with _client(handler) as client:
            return await client.responses.create(model="muse-spark-1.3", input="hi")

    response = asyncio.run(run())
    assert seen["url"] == "https://api.meta.ai/v1/responses"
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["model"] == "muse-spark-1.3"
    assert response.output_text == '{"a":1}'
    assert response.usage.input_tokens == 10
    assert response.usage.model_dump()["output_tokens_details"] == {"reasoning_tokens": 2}


def test_stream_yields_event_dicts():
    events = [
        {"type": "response.output_text.delta", "delta": "ab"},
        {"type": "response.completed", "response": {"id": "resp_2"}},
    ]
    body = "".join(f"event: x\ndata: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def run():
        async with _client(handler) as client:
            stream = await client.responses.create(model="sam-3.1", stream=True)
            return [event async for event in stream]

    assert asyncio.run(run()) == events


def test_file_upload_sends_purpose_and_returns_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files"
        assert b'name="purpose"' in request.content and b"user_data" in request.content
        return httpx.Response(200, json={"id": "file_1"})

    async def run():
        async with _client(handler) as client:
            handle = io.BytesIO(b"video")
            handle.name = "clip.mp4"
            return await client.files.create(file=handle, purpose="user_data")

    assert asyncio.run(run()).id == "file_1"


def test_error_status_reaches_the_translator():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    async def run():
        async with _client(handler) as client:
            await client.responses.create(model="muse-spark-1.3")

    with pytest.raises(MetaAPIError) as raised:
        asyncio.run(run())
    assert raised.value.status_code == 401
    assert isinstance(translate(raised.value), AuthenticationError)
