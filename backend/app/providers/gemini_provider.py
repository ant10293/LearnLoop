"""Gemini adapter. Native video and frame sequences are separate, explicit modalities."""

from __future__ import annotations

import asyncio
import importlib.metadata
import time

from google import genai
from google.genai import types

from app.providers.common import translate
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


class GeminiProvider:
    id = "gemini"
    display_name = "Google Gemini"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return capabilities_for(self.id, model_id)

    def _client(self, secret: str, timeout_s: float | None = None) -> genai.Client:
        if timeout_s is None:
            return genai.Client(api_key=secret)
        return genai.Client(
            api_key=secret,
            http_options=types.HttpOptions(timeout=int(provider_timeout(timeout_s) * 1000)),
        )

    async def validate(self, secret: str) -> None:
        try:
            await asyncio.to_thread(lambda: list(self._client(secret).models.list()))
        except Exception as exc:
            raise translate(exc) from exc

    async def list_models(self, secret: str) -> list[str]:
        try:
            models = await asyncio.to_thread(lambda: list(self._client(secret).models.list()))
        except Exception as exc:
            raise translate(exc) from exc
        names = []
        for model in models:
            name = getattr(model, "name", "") or ""
            short = name.split("/")[-1]
            actions = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", None) or []
            if actions and "generateContent" not in actions and "generate_content" not in str(actions):
                continue
            if "embed" in short or "image" in short and "gemini" not in short:
                continue
            if short:
                names.append(short)
        return sorted(set(names))

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        caps = self.capabilities(request.model_id)
        if request.modality == "native_video":
            if not caps.native_video:
                raise UnsupportedMode(
                    f"{request.model_id} is not recorded as supporting native video. "
                    "The run was not switched to frames."
                )
            if request.video_path is None:
                raise ProviderError("native video modality requires a video file")
            return await asyncio.to_thread(self._generate_video, secret, request, caps)
        if len(request.frames) > caps.max_images:
            raise UnsupportedMode(
                f"{request.model_id} accepts at most {caps.max_images} images per call."
            )
        return await asyncio.to_thread(self._generate_frames, secret, request, caps)

    def _config(self, request: GenerationRequest, caps: ModelCapabilities) -> types.GenerateContentConfig:
        kwargs: dict = {"system_instruction": request.system_prompt}
        if request.temperature is not None:
            if not caps.supports_temperature:
                raise UnsupportedMode(f"{request.model_id} does not accept temperature.")
            kwargs["temperature"] = request.temperature
        if request.json_schema and caps.structured_output:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = request.json_schema
        return types.GenerateContentConfig(**kwargs)

    def _generate_frames(self, secret: str, request: GenerationRequest, caps: ModelCapabilities) -> ProviderResponse:
        parts: list = [request.user_prompt]
        for frame in request.frames:
            parts.append(f"Frame timestamp {frame.timestamp:.3f} seconds.")
            parts.append(types.Part.from_bytes(data=frame.path.read_bytes(), mime_type="image/jpeg"))
        return self._call(secret, request, caps, parts)

    def _generate_video(self, secret: str, request: GenerationRequest, caps: ModelCapabilities) -> ProviderResponse:
        client = self._client(secret, request.timeout_s)
        uploaded = client.files.upload(file=str(request.video_path))
        deadline = time.time() + provider_timeout(request.timeout_s)
        while True:
            state = getattr(getattr(uploaded, "state", None), "name", None) or str(getattr(uploaded, "state", ""))
            if state == "ACTIVE":
                break
            if state in {"FAILED", "STATE_UNSPECIFIED"} and "FAIL" in state:
                raise ProviderError(f"Gemini file processing failed: {state}")
            if time.time() > deadline:
                raise ProviderError("Gemini file processing did not become ACTIVE before the timeout.")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
            state = getattr(getattr(uploaded, "state", None), "name", "") or ""
            if state == "FAILED":
                raise ProviderError("Gemini reported the uploaded video as FAILED.")
        note = (
            "Input modality is native video via the Files API, not a manually chosen frame sample. "
            + request.user_prompt
        )
        return self._call(secret, request, caps, [uploaded, note], client=client)

    def _call(self, secret, request, caps, contents, client=None) -> ProviderResponse:
        client = client or self._client(secret, request.timeout_s)
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=request.model_id,
                contents=contents,
                config=self._config(request, caps),
            )
        except UnsupportedMode:
            raise
        except Exception as exc:
            raise translate(exc) from exc
        usage_meta = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        cost, known = estimate_cost(caps, input_tokens, output_tokens)
        text = getattr(response, "text", "") or ""
        return ProviderResponse(
            text=text,
            raw={
                "model": request.model_id,
                "modality": request.modality,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
            usage=Usage(input_tokens, output_tokens, cost, known),
            request_id=str(getattr(response, "response_id", "") or ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=request.model_id,
            sdk_version=_sdk(),
        )


def _sdk() -> str:
    try:
        return importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        return ""
