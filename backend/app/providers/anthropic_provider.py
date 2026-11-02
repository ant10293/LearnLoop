"""Anthropic Messages API with timestamped images and tool-shaped JSON output.

Native video is not implemented: current official vision documentation accepts images only.
"""

from __future__ import annotations

import importlib.metadata
import time

import anthropic

from app.providers.claude_code import complete
from app.providers.common import translate
from app.providers.model_list import capabilities_for
from app.providers.types import (
    GenerationRequest,
    ModelCapabilities,
    ProviderResponse,
    UnsupportedMode,
    Usage,
    estimate_cost,
    provider_timeout,
)


class AnthropicProvider:
    id = "anthropic"
    display_name = "Anthropic"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return capabilities_for(self.id, model_id)

    def _client(self, secret: str, timeout: float) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(api_key=secret, timeout=timeout)

    async def validate(self, secret: str) -> None:
        try:
            async with self._client(secret, 30) as client:
                await client.models.list(limit=1)
        except Exception as exc:
            raise translate(exc) from exc

    async def list_models(self, secret: str) -> list[str]:
        try:
            async with self._client(secret, 30) as client:
                page = await client.models.list(limit=40)
        except Exception as exc:
            raise translate(exc) from exc
        found = sorted({item.id for item in page.data})
        return found

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        caps = self.capabilities(request.model_id)
        if request.modality == "native_video" or (request.video_path and not request.frames):
            raise UnsupportedMode(
                "Anthropic's documented Messages API accepts images, not video files. "
                "Native video was not silently replaced with frames."
            )
        if len(request.frames) > caps.max_images:
            raise UnsupportedMode(
                f"{request.model_id} is limited to {caps.max_images} images per request. "
                "The pipeline windows longer sequences before calling the provider."
            )
        effort = _require_effort(request, caps)
        if request.metadata.get("transport") == "claude_code":
            if request.temperature is not None:
                raise UnsupportedMode(
                    "Claude Code does not accept temperature. Leave it unset, "
                    "or switch Anthropic runs to the API key."
                )
            return await complete(request)
        blocks: list[dict] = []
        for frame in request.frames:
            payload = __import__("base64").b64encode(frame.path.read_bytes()).decode("ascii")
            blocks.append({"type": "text", "text": f"Frame timestamp {frame.timestamp:.3f} seconds."})
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": payload},
                }
            )
        blocks.append({"type": "text", "text": request.user_prompt})
        kwargs: dict = {
            "model": request.model_id,
            "max_tokens": 8000,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": blocks}],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if effort:
            kwargs["output_config"] = {"effort": effort}
        schema = request.json_schema or {"type": "object"}
        kwargs["tools"] = [
            {
                "name": "record_analysis",
                "description": "Record the structured physical-procedure analysis.",
                "input_schema": schema,
            }
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": "record_analysis"}
        started = time.perf_counter()
        try:
            async with self._client(secret, provider_timeout(request.timeout_s)) as client:
                response = await client.messages.create(**kwargs)
        except Exception as exc:
            raise translate(exc) from exc
        text = ""
        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == "record_analysis":
                import json

                text = json.dumps(block.input)
                break
        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost, known = estimate_cost(caps, input_tokens, output_tokens)
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        raw.pop("content", None)
        return ProviderResponse(
            text=text,
            raw={"id": raw.get("id"), "model": raw.get("model"), "usage": raw.get("usage"), "stop_reason": raw.get("stop_reason")},
            usage=Usage(input_tokens, output_tokens, cost, known),
            request_id=str(getattr(response, "id", "") or ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=request.model_id,
            sdk_version=_sdk(),
        )


def _require_effort(request: GenerationRequest, caps: ModelCapabilities) -> str | None:
    effort = request.reasoning_effort or None
    if effort and effort not in caps.reasoning_efforts:
        raise UnsupportedMode(
            f"Reasoning effort '{effort}' is not available on {request.model_id}."
        )
    return effort


def _sdk() -> str:
    try:
        return importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        return ""
