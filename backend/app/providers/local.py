"""Local providers. Neither one pretends to be a vision-language model."""

from __future__ import annotations

from app.providers.catalog import lookup
from app.providers.types import (
    GenerationRequest,
    ModelCapabilities,
    ProviderError,
    ProviderResponse,
    Usage,
)
from app.schemas.analysis import AnalysisDocument


class DeterministicProvider:
    id = "deterministic"
    display_name = "Deterministic vision baseline"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return lookup(self.id, model_id or "motion-baseline")

    async def validate(self, secret: str) -> None:
        return None

    async def list_models(self, secret: str) -> list[str]:
        return ["motion-baseline"]

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        document = request.metadata.get("document")
        if not isinstance(document, dict):
            raise ProviderError("deterministic baseline did not receive a motion document")
        AnalysisDocument.model_validate(document)
        import json

        return ProviderResponse(
            text=json.dumps(document),
            raw={"provider": self.id, "note": "Local computer-vision baseline, not a VLM response."},
            usage=Usage(price_known=True),
            model_id="motion-baseline",
            sdk_version="opencv",
        )


class FixtureProvider:
    id = "fixture"
    display_name = "Authored fixture"

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return lookup(self.id, model_id or "authored-fixture")

    async def validate(self, secret: str) -> None:
        return None

    async def list_models(self, secret: str) -> list[str]:
        return ["authored-fixture"]

    async def generate(self, secret: str, request: GenerationRequest) -> ProviderResponse:
        document = request.metadata.get("document")
        if not isinstance(document, dict):
            raise ProviderError(
                "This video is not a known fixture. The fixture provider refuses to invent an analysis. "
                "Use the deterministic baseline or a configured model."
            )
        import json

        return ProviderResponse(
            text=json.dumps(document),
            raw={"provider": self.id, "fixture": True},
            usage=Usage(price_known=True),
            model_id="authored-fixture",
            sdk_version="fixture",
        )
