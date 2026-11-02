"""Cursor provider. Calls go through the local agent CLI, not a messages API."""

from __future__ import annotations

import asyncio

from app.providers.cursor_cli import complete, list_models_sync, probe_sync
from app.providers.model_list import capabilities_for
from app.providers.types import (
    AuthenticationError,
    GenerationRequest,
    ModelCapabilities,
    ProviderResponse,
    UnsupportedMode,
)


class CursorProvider:
    id = "cursor"
    display_name = "Cursor"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return capabilities_for(self.id, model_id)

    async def validate(self, secret: str) -> None:
        status = await _probe(secret)
        if not status["installed"]:
            raise AuthenticationError(status["error"] or "Cursor CLI executable not found")
        if not status["logged_in"]:
            raise AuthenticationError(status["error"] or "Cursor CLI is not signed in")

    async def list_models(self, secret: str) -> list[str]:
        return await asyncio.to_thread(list_models_sync, secret)

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        caps = self.capabilities(request.model_id)
        sending_video = request.modality == "native_video" or (request.video_path is not None and not request.frames)
        if sending_video and not caps.native_video:
            raise UnsupportedMode(
                f"{request.model_id} does not support native video. "
                "The Cursor CLI was not given frames instead."
            )
        if len(request.frames) > caps.max_images:
            raise UnsupportedMode(
                f"{request.model_id} is limited to {caps.max_images} images per request. "
                "The pipeline windows longer sequences before calling the provider."
            )
        return await complete(request, secret)


async def _probe(secret: str) -> dict:
    return await asyncio.to_thread(probe_sync, secret)
