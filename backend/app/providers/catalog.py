"""Static capability catalog.

Prices are approximate public list rates stored so a run can record an estimate.
They are not invoices. A call with no API key stays unpriced. A keyed call without a list rate does not.
"""

from __future__ import annotations

from app.providers.types import ModelCapabilities

_MUSE_NOTE = (
    "Muse Spark accepts text, image, and video on the Meta Responses API. "
    "Reasoning effort 'none' is rejected. Audio input on 1.3 is degraded; this lab does not send audio."
)
_SAM_NOTE = (
    "SAM 3.1 returns boxes, masks, and video tracks. It is not an action-recognition model "
    "and is never substituted for a scene-understanding model. Select it as the segmentation model."
)
MUSE_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def _caps(**kwargs) -> ModelCapabilities:
    return ModelCapabilities(**kwargs)


CATALOG: dict[tuple[str, str], ModelCapabilities] = {
    ("meta", "muse-wonder-1.0"): _caps(
        model_id="muse-wonder-1.0",
        provider_id="meta",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_048_576,
        reasoning_efforts=MUSE_EFFORTS,
        supports_temperature=False,
        input_price_per_mtok=None,
        output_price_per_mtok=None,
        notes=_MUSE_NOTE,
    ),
    ("meta", "muse-spark-1.3"): _caps(
        model_id="muse-spark-1.3",
        provider_id="meta",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_048_576,
        reasoning_efforts=MUSE_EFFORTS,
        supports_temperature=False,
        input_price_per_mtok=None,
        output_price_per_mtok=None,
        notes=_MUSE_NOTE,
    ),
    ("meta", "muse-spark-1.2"): _caps(
        model_id="muse-spark-1.2",
        provider_id="meta",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_048_576,
        reasoning_efforts=("minimal", "low", "medium", "high", "xhigh"),
        supports_temperature=False,
        notes=_MUSE_NOTE + " 1.2 does not advertise the max effort level.",
    ),
    ("meta", "muse-spark-1.1"): _caps(
        model_id="muse-spark-1.1",
        provider_id="meta",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_048_576,
        reasoning_efforts=("low", "medium", "high"),
        supports_temperature=False,
        notes=_MUSE_NOTE,
    ),
    ("meta", "sam-3.1"): _caps(
        model_id="sam-3.1",
        provider_id="meta",
        image=True,
        native_video=True,
        structured_output=False,
        max_images=1,
        max_video_seconds=600,
        supports_temperature=False,
        notes=_SAM_NOTE,
    ),
    ("deterministic", "motion-baseline"): _caps(
        model_id="motion-baseline",
        provider_id="deterministic",
        image=True,
        structured_output=True,
        max_images=10_000,
        supports_temperature=False,
        notes="Local optical-flow and background baseline. Not a vision-language model.",
    ),
    ("fixture", "authored-fixture"): _caps(
        model_id="authored-fixture",
        provider_id="fixture",
        image=True,
        structured_output=True,
        supports_temperature=False,
        notes="Returns a checksum-locked authored analysis. Refuses unknown videos. Not model output.",
    ),
    ("manual_json", "imported-json"): _caps(
        model_id="imported-json",
        provider_id="manual_json",
        text=True,
        structured_output=True,
        supports_temperature=False,
        notes="User-supplied JSON. Not a model call.",
    ),
}

def lookup(provider_id: str, model_id: str) -> ModelCapabilities:
    known = CATALOG.get((provider_id, model_id))
    if known:
        return known
    return _default_manual(provider_id, model_id)


def _default_manual(provider_id: str, model_id: str) -> ModelCapabilities:
    segmentation = model_id.startswith("sam")
    return ModelCapabilities(
        model_id=model_id,
        provider_id=provider_id,
        image=provider_id != "manual_json",
        native_video=provider_id == "meta" and not segmentation,
        structured_output=not segmentation,
        max_images=16,
        supports_temperature=provider_id != "meta",
        notes=(
            "Manually added model id. Capabilities are a conservative default, not a live "
            "provider capability record."
        ),
    )


def models_for(provider_id: str) -> list[ModelCapabilities]:
    return [caps for (provider, _model), caps in CATALOG.items() if provider == provider_id]
