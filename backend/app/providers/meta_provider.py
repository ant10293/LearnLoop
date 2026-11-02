"""Meta Model API for Muse Spark scene understanding. SAM is a separate adapter."""

from __future__ import annotations

import importlib.metadata
import time

import httpx
from openai import AsyncOpenAI

from app.providers.common import data_url, translate
from app.providers.model_list import capabilities_for
from app.providers.types import (
    GenerationRequest,
    ModelCapabilities,
    ProviderError,
    ProviderResponse,
    UnsupportedMode,
    Usage,
    estimate_cost,
    provider_timeout,
)

META_BASE = "https://api.meta.ai/v1"
# Above every complete answer recorded so far. Reasoning tokens count toward it.
# A call that repeats itself stops here instead of at the provider's far larger limit.
MAX_OUTPUT_TOKENS = 65_536


class MetaProvider:
    id = "meta"
    display_name = "Meta"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        caps = capabilities_for(self.id, model_id)
        if model_id.startswith("sam"):
            raise UnsupportedMode("SAM is a segmentation model. Select it as the segmentation model, not the analysis model.")
        return caps

    def _client(self, secret: str, timeout: float) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=secret, base_url=META_BASE, timeout=timeout)

    async def validate(self, secret: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                response = await http.get(
                    f"{META_BASE}/models",
                    headers={"Authorization": f"Bearer {secret}"},
                )
                response.raise_for_status()
        except Exception as exc:
            raise translate(exc) from exc

    async def list_models(self, secret: str) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                response = await http.get(
                    f"{META_BASE}/models",
                    headers={"Authorization": f"Bearer {secret}"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise translate(exc) from exc
        data = payload.get("data") or payload.get("models") or []
        names = []
        for item in data:
            model_id = item.get("id") or item.get("name") or ""
            if model_id:
                names.append(model_id)
        return sorted(set(names))

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        caps = self.capabilities(request.model_id)
        if request.modality == "native_video":
            if not caps.native_video:
                raise UnsupportedMode(f"{request.model_id} is not recorded as accepting video.")
            content = await self._video_content(secret, request)
        else:
            if len(request.frames) > caps.max_images:
                raise UnsupportedMode(f"{request.model_id} accepts at most {caps.max_images} images per call.")
            content = [{"type": "input_text", "text": request.user_prompt}]
            for frame in request.frames:
                content.append({"type": "input_text", "text": f"Frame timestamp {frame.timestamp:.3f} seconds."})
                content.append({"type": "input_image", "image_url": data_url(frame.path)})
        kwargs: dict = {
            "model": request.model_id,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": content}],
            "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        if request.reasoning_effort:
            if request.reasoning_effort not in caps.reasoning_efforts:
                raise UnsupportedMode(
                    f"Reasoning effort '{request.reasoning_effort}' is not available on {request.model_id}."
                )
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            raise UnsupportedMode(
                f"{request.model_id} does not accept temperature. Leave temperature unset."
            )
        if request.json_schema:
            from app.pipeline.contracts import validate_strict_schema

            if not caps.structured_output:
                raise UnsupportedMode(
                    f"{request.model_id} is not recorded as supporting structured output. "
                    "The request was not sent unconstrained."
                )
            validate_strict_schema(request.json_schema)
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                }
            }
        started = time.perf_counter()
        try:
            async with self._client(secret, provider_timeout(request.timeout_s)) as client:
                response = await client.responses.create(**kwargs)
        except UnsupportedMode:
            raise
        except Exception as exc:
            if request.json_schema and _rejected_schema(exc):
                from app.pipeline.contracts import SchemaContractError

                raise SchemaContractError(f"Meta rejected the {request.schema_name} schema: {exc}") from exc
            raise translate(exc) from exc
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost, known = estimate_cost(caps, input_tokens, output_tokens)
        text = getattr(response, "output_text", "") or ""
        status = str(getattr(response, "status", "") or "")
        incomplete = _unfinished_reason(response, status, text)
        return ProviderResponse(
            text=text,
            raw={
                "id": getattr(response, "id", ""),
                "model": request.model_id,
                "usage": _usage_dict(usage),
                "status": status,
                "incomplete_reason": incomplete,
            },
            usage=Usage(input_tokens, output_tokens, cost, known),
            request_id=str(getattr(response, "id", "") or ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=request.model_id,
            sdk_version=_sdk(),
            incomplete_reason=incomplete,
        )

    async def _video_content(self, secret: str, request: GenerationRequest) -> list[dict]:
        if request.video_path is None:
            raise ProviderError("native video modality requires a video file")
        size = request.video_path.stat().st_size
        if size <= 8 * 1024 * 1024:
            return [
                {"type": "input_text", "text": request.user_prompt},
                {"type": "input_video", "video_url": data_url(request.video_path, "video/mp4")},
            ]
        try:
            async with self._client(secret, provider_timeout(request.timeout_s)) as client:
                with request.video_path.open("rb") as handle:
                    uploaded = await client.files.create(file=handle, purpose="user_data")
        except Exception as exc:
            raise translate(exc) from exc
        return [
            {"type": "input_text", "text": request.user_prompt},
            {"type": "input_video", "file_id": uploaded.id},
        ]


def _unfinished_reason(response, status: str, text: str) -> str:
    """Why this response is not a finished answer. Empty only when it is one."""
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return str(getattr(details, "reason", "") or "unknown")
    if not status:
        return "missing_status"
    if status != "completed":
        return f"status_{status}"
    if _refused(response):
        return "refusal"
    return "" if text.strip() else "empty_output"


def _refused(response) -> bool:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", "") == "refusal":
                return True
    return False


def _rejected_schema(exc: Exception) -> bool:
    message = str(exc).lower()
    return getattr(exc, "status_code", None) == 400 and ("schema" in message or "text.format" in message)


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {"input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}


def _sdk() -> str:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return ""
