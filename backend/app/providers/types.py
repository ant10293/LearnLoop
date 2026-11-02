"""Provider-neutral request and capability types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ProviderError(RuntimeError):
    pass


class UnsupportedMode(ProviderError):
    """The selected model cannot perform the requested modality. Do not fall back."""


class AuthenticationError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


# Wall-clock deadline for every analysis call. Stored runs still default to 180s,
# and that value is what the UI saves. Each provider raises a shorter request to
# this floor. A configured timeout above it still applies.
PROVIDER_TIMEOUT_S = 30 * 60


# Provider requests one run may send, counted across its retries.
DEFAULT_MAX_REQUESTS = 36


def provider_timeout(requested: float | None = None) -> float:
    return max(float(requested or 0), PROVIDER_TIMEOUT_S)


@dataclass
class FrameInput:
    timestamp: float
    path: Path


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    price_known: bool = False


@dataclass
class ProviderResponse:
    text: str
    raw: dict
    usage: Usage = field(default_factory=Usage)
    request_id: str = ""
    latency_ms: int = 0
    model_id: str = ""
    sdk_version: str = ""
    # Why the provider stopped early, such as max_output_tokens. Empty for a finished answer.
    incomplete_reason: str = ""


@dataclass(frozen=True)
class ModelCapabilities:
    model_id: str
    provider_id: str
    text: bool = True
    image: bool = False
    native_video: bool = False
    structured_output: bool = False
    max_images: int = 16
    max_video_seconds: float | None = None
    context_tokens: int | None = None
    reasoning_efforts: tuple[str, ...] = ()
    supports_temperature: bool = True
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    resolutions: tuple[str, ...] = ("low", "high", "original")
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "text": self.text,
            "image": self.image,
            "native_video": self.native_video,
            "structured_output": self.structured_output,
            "max_images": self.max_images,
            "max_video_seconds": self.max_video_seconds,
            "context_tokens": self.context_tokens,
            "reasoning_efforts": list(self.reasoning_efforts),
            "supports_temperature": self.supports_temperature,
            "input_price_per_mtok": self.input_price_per_mtok,
            "output_price_per_mtok": self.output_price_per_mtok,
            "price_known": self.input_price_per_mtok is not None,
            "resolutions": list(self.resolutions),
            "notes": self.notes,
        }


@dataclass
class GenerationRequest:
    model_id: str
    system_prompt: str
    user_prompt: str
    frames: list[FrameInput] = field(default_factory=list)
    video_path: Path | None = None
    modality: str = "sampled_frames"
    json_schema: dict | None = None
    schema_name: str = "learnloop_analysis"
    temperature: float | None = None
    reasoning_effort: str | None = None
    image_detail: str = "high"
    timeout_s: float = 180
    metadata: dict = field(default_factory=dict)


def estimate_cost(caps: ModelCapabilities, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    if caps.input_price_per_mtok is None or caps.output_price_per_mtok is None:
        return 0.0, False
    cost = (
        input_tokens * caps.input_price_per_mtok + output_tokens * caps.output_price_per_mtok
    ) / 1_000_000
    return round(cost, 6), True


def rough_input_tokens(request: GenerationRequest) -> int:
    text = len(request.system_prompt) + len(request.user_prompt)
    image_tokens = 1100 if request.image_detail == "high" else 400
    if request.modality == "native_video":
        return text // 4 + 8000
    return text // 4 + image_tokens * max(len(request.frames), 1)
