"""OpenAI-compatible endpoints. Native video is off unless the model catalog says otherwise."""

from __future__ import annotations

import os

import httpx
from sqlalchemy.orm import Session

from app.models.runs import ProviderStatus
from app.providers.openai_provider import OpenAIProvider
from app.providers.types import (
    GenerationRequest,
    ModelCapabilities,
    ProviderError,
    ProviderResponse,
)


class CompatibleProvider(OpenAIProvider):
    id = "openai_compatible"
    display_name = "OpenAI-compatible"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _client(self, secret: str, timeout: float):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=secret, base_url=self.base_url, timeout=timeout)

    def capabilities(self, model_id: str) -> ModelCapabilities:
        caps = super().capabilities(model_id)
        return ModelCapabilities(
            model_id=model_id,
            provider_id=self.id,
            text=True,
            image=caps.image,
            native_video=False,
            structured_output=True,
            max_images=caps.max_images,
            supports_temperature=True,
            notes=(
                "Custom OpenAI-compatible endpoint. Native video is disabled because compatibility "
                "does not prove the upstream accepts video. "
            )
            + caps.notes,
        )

    async def validate(self, secret: str) -> None:
        if not self.base_url:
            raise ProviderError("Set a base URL before validating this endpoint.")
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {secret}"},
            )
            response.raise_for_status()

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        return await super().generate(secret, request)


def compatible_base_url(session: Session) -> str:
    row = session.get(ProviderStatus, "openai_compatible")
    if row and row.extra_base_url:
        return row.extra_base_url
    return os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")
