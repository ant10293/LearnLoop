"""OpenAI Responses API. Native video is refused: the public API accepts images."""

from __future__ import annotations

import importlib.metadata
import time

from openai import AsyncOpenAI

from app.providers.codex_cli import complete, load_codex, probe
from app.providers.common import data_url, translate
from app.providers.model_list import capabilities_for
from app.providers.types import (
    AuthenticationError,
    GenerationRequest,
    ModelCapabilities,
    ProviderResponse,
    UnsupportedMode,
    Usage,
    estimate_cost,
    provider_timeout,
)


class OpenAIProvider:
    id = "openai"
    display_name = "OpenAI"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return capabilities_for(self.id, model_id)

    def _client(self, secret: str, timeout: float) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=secret, timeout=timeout)

    async def validate(self, secret: str) -> None:
        if load_codex()["transport"] == "codex":
            status = await probe()
            if not status["installed"]:
                raise AuthenticationError(status["error"] or "Codex executable not found")
            if not status["logged_in"]:
                raise AuthenticationError(
                    status["error"] or "Codex CLI is not signed in with ChatGPT"
                )
            return
        try:
            async with self._client(secret, 30) as client:
                await client.models.list()
        except Exception as exc:
            raise translate(exc) from exc

    async def list_models(self, secret: str) -> list[str]:
        try:
            async with self._client(secret, 30) as client:
                page = await client.models.list()
        except Exception as exc:
            raise translate(exc) from exc
        found = sorted({item.id for item in page.data if _vision_candidate(item.id)})
        return found or [caps.model_id for caps in _defaults()]

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        caps = self.capabilities(request.model_id)
        if request.modality == "native_video" or request.video_path is not None:
            if not caps.native_video:
                raise UnsupportedMode(
                    f"{request.model_id} has no documented native video input. "
                    "Choose sampled frames, or choose another model. The modality was not changed."
                )
        if not caps.image:
            raise UnsupportedMode(f"{request.model_id} does not accept images.")
        if len(request.frames) > caps.max_images:
            raise UnsupportedMode(
                f"{request.model_id} accepts at most {caps.max_images} images in one request."
            )
        if request.metadata.get("transport") == "codex":
            if request.temperature is not None:
                raise UnsupportedMode(
                    "Codex does not accept temperature. Leave it unset, "
                    "or switch OpenAI runs to the API key."
                )
            return await complete(request)
        content: list[dict] = [{"type": "input_text", "text": request.user_prompt}]
        detail = "low" if request.image_detail == "low" else "high"
        for frame in request.frames:
            content.append({"type": "input_text", "text": f"Frame timestamp {frame.timestamp:.3f} seconds."})
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url(frame.path),
                    "detail": detail,
                }
            )
        kwargs: dict = {
            "model": request.model_id,
            "instructions": request.system_prompt,
            "input": [{"role": "user", "content": content}],
        }
        if request.temperature is not None:
            if not caps.supports_temperature:
                raise UnsupportedMode(f"{request.model_id} does not accept temperature.")
            kwargs["temperature"] = request.temperature
        if request.json_schema and caps.structured_output:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "learnloop_analysis",
                    "schema": request.json_schema,
                    "strict": False,
                }
            }
        started = time.perf_counter()
        try:
            async with self._client(secret, provider_timeout(request.timeout_s)) as client:
                response = await client.responses.create(**kwargs)
        except Exception as exc:
            raise translate(exc) from exc
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost, known = estimate_cost(caps, input_tokens, output_tokens)
        text = getattr(response, "output_text", "") or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else {"output_text": text}
        return ProviderResponse(
            text=text,
            raw=_scrub(raw),
            usage=Usage(input_tokens, output_tokens, cost, known),
            request_id=str(getattr(response, "id", "") or ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=request.model_id,
            sdk_version=_sdk(),
        )


def _vision_candidate(model_id: str) -> bool:
    name = model_id.lower()
    blocked = ("embedding", "whisper", "tts", "audio", "realtime", "moderation", "transcribe")
    return not any(part in name for part in blocked)


def _defaults() -> list[ModelCapabilities]:
    from app.providers.catalog import models_for

    return models_for("openai")


def _sdk() -> str:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return ""


def _scrub(payload: dict) -> dict:
    """Drop bulky image echoes if a provider reflects them."""
    text = payload.get("output_text")
    return {"id": payload.get("id"), "model": payload.get("model"), "usage": payload.get("usage"), "output_text": text}
