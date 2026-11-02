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
_OPENAI_NOTE = (
    "Public Responses API accepts images, not video files. Native video stays disabled "
    "unless a future capability record explicitly enables it for that model id. "
    "A Codex subscription run attaches those frames in read-only mode and is recorded as unpriced plan usage. "
    "An API key run uses the catalog list rate when one is set."
)
_ANTHROPIC_NOTE = (
    "Claude vision accepts images, not video files. Long sequences are windowed. "
    "A Claude Code subscription run is recorded as unpriced plan usage. "
    "An API key run uses the catalog list rate when one is set."
)
_CURSOR_NOTE = (
    "Cursor CLI ask mode. The agent reads frame files and is not given permission to edit them. "
    "When the answer is not one JSON object, a follow-up call edits answer.json in a temporary "
    "workspace until it parses. The CLI does not enforce the JSON schema; the lab validates the "
    "document after the call. Native video is refused. Linked-account usage is recorded as unpriced "
    "plan usage."
)

# Messages API output_config.effort. xhigh is the "extra high" level.
# Opus 4.6 and Sonnet 4.6 accept max but not xhigh. Opus 4.5 accepts the original three.
# Older 4.5 and 4.1 ids are omitted because the effort parameter is not offered there.
_EFFORT_ALL = ("low", "medium", "high", "xhigh", "max")
_EFFORT_NO_XHIGH = ("low", "medium", "high", "max")
_EFFORT_CORE = ("low", "medium", "high")
_ANTHROPIC_EFFORTS: dict[str, tuple[str, ...]] = {
    "claude-sonnet-5-5": _EFFORT_ALL,
    "claude-opus-5-5": _EFFORT_ALL,
    "claude-haiku-5-5": _EFFORT_ALL,
    "claude-fable-5-1": _EFFORT_ALL,
    "claude-opus-5": _EFFORT_ALL,
    "claude-sonnet-5": _EFFORT_ALL,
    "claude-fable-5": _EFFORT_ALL,
    "claude-opus-4-8": _EFFORT_ALL,
    "claude-opus-4-7": _EFFORT_ALL,
    "claude-sonnet-4-6": _EFFORT_NO_XHIGH,
    "claude-opus-4-6": _EFFORT_NO_XHIGH,
    "claude-opus-4-5-20251101": _EFFORT_CORE,
}

# model id, context, temperature, input $/MTok, output $/MTok, extra note.
# Current ids match the Claude Code /model list and the claude.ai picker.
_ANTHROPIC_MODELS: tuple[tuple, ...] = (
    ("claude-sonnet-5-5", 1_000_000, False, 2.0, 10.0, ""),
    ("claude-opus-5-5", 1_000_000, False, 4.0, 20.0, ""),
    (
        "claude-haiku-5-5",
        1_000_000,
        False,
        0.10,
        0.50,
        "The Haiku 5.5 figures are the short-context list rate. API input over 100k tokens costs more.",
    ),
    ("claude-fable-5-1", 1_000_000, False, 10.0, 50.0, ""),
    ("claude-opus-5", 1_000_000, False, None, None, ""),
    ("claude-sonnet-5", 1_000_000, False, None, None, ""),
    ("claude-fable-5", 1_000_000, False, None, None, ""),
    ("claude-opus-4-8", 1_000_000, False, None, None, ""),
    ("claude-opus-4-7", 1_000_000, False, None, None, ""),
    ("claude-sonnet-4-6", 200_000, False, None, None, ""),
    ("claude-opus-4-6", 200_000, False, None, None, ""),
    ("claude-opus-4-5-20251101", 200_000, True, None, None, ""),
    ("claude-haiku-4-5-20251001", 200_000, True, None, None, ""),
    ("claude-sonnet-4-5-20250929", 200_000, True, None, None, ""),
    ("claude-sonnet-4-5", 200_000, True, 3.0, 15.0, ""),
    ("claude-opus-4-1", 200_000, True, 15.0, 75.0, ""),
    ("claude-haiku-4-5", 200_000, True, 1.0, 5.0, ""),
)

MUSE_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def _caps(**kwargs) -> ModelCapabilities:
    return ModelCapabilities(**kwargs)


CATALOG: dict[tuple[str, str], ModelCapabilities] = {
    ("openai", "gpt-4.1"): _caps(
        model_id="gpt-4.1",
        provider_id="openai",
        image=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_047_576,
        supports_temperature=True,
        input_price_per_mtok=2.0,
        output_price_per_mtok=8.0,
        notes=_OPENAI_NOTE,
    ),
    ("openai", "gpt-4.1-mini"): _caps(
        model_id="gpt-4.1-mini",
        provider_id="openai",
        image=True,
        structured_output=True,
        max_images=16,
        context_tokens=1_047_576,
        input_price_per_mtok=0.4,
        output_price_per_mtok=1.6,
        notes=_OPENAI_NOTE,
    ),
    ("openai", "gpt-4o"): _caps(
        model_id="gpt-4o",
        provider_id="openai",
        image=True,
        structured_output=True,
        max_images=16,
        context_tokens=128_000,
        input_price_per_mtok=2.5,
        output_price_per_mtok=10.0,
        notes=_OPENAI_NOTE,
    ),
    ("gemini", "gemini-2.5-flash"): _caps(
        model_id="gemini-2.5-flash",
        provider_id="gemini",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        max_video_seconds=3600,
        context_tokens=1_048_576,
        input_price_per_mtok=0.3,
        output_price_per_mtok=2.5,
        notes="Native video uses the Gemini Files API and is polled until ACTIVE. Frame mode is a separate modality.",
    ),
    ("gemini", "gemini-2.5-pro"): _caps(
        model_id="gemini-2.5-pro",
        provider_id="gemini",
        image=True,
        native_video=True,
        structured_output=True,
        max_images=16,
        max_video_seconds=3600,
        context_tokens=1_048_576,
        input_price_per_mtok=1.25,
        output_price_per_mtok=10.0,
        notes="Native video uses the Gemini Files API and is polled until ACTIVE. Frame mode is a separate modality.",
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

for _model_id, _context, _temperature, _input_price, _output_price, _extra in _ANTHROPIC_MODELS:
    CATALOG[("anthropic", _model_id)] = _caps(
        model_id=_model_id,
        provider_id="anthropic",
        image=True,
        structured_output=True,
        max_images=20,
        context_tokens=_context,
        reasoning_efforts=_ANTHROPIC_EFFORTS.get(_model_id, ()),
        supports_temperature=_temperature,
        input_price_per_mtok=_input_price,
        output_price_per_mtok=_output_price,
        notes=_ANTHROPIC_NOTE + (f" {_extra}" if _extra else ""),
    )


for _cursor_model in ("composer-2.5", "auto"):
    CATALOG[("cursor", _cursor_model)] = _caps(
        model_id=_cursor_model,
        provider_id="cursor",
        image=True,
        structured_output=True,
        max_images=16,
        supports_temperature=False,
        notes=_CURSOR_NOTE,
    )


def cursor_accepts_video(model_id: str) -> bool:
    """Gemini and Muse accept a video file when Cursor lists those same models.

    Claude, GPT, Grok, Composer, and Auto stay on frames. Grok's chat models take
    text and images. Its video API generates clips and does not watch one.
    """
    token = model_id.lower().removeprefix("cursor-")
    return token.startswith("gemini") or token.startswith("muse-")


def lookup(provider_id: str, model_id: str) -> ModelCapabilities:
    known = CATALOG.get((provider_id, model_id))
    if known:
        return known
    return _default_manual(provider_id, model_id)


def _default_manual(provider_id: str, model_id: str) -> ModelCapabilities:
    native = provider_id in {"gemini", "meta"} and not model_id.startswith("sam")
    if provider_id == "cursor":
        native = cursor_accepts_video(model_id)
    temperature = provider_id not in {"meta", "cursor"}
    if provider_id == "cursor" and native:
        note = (
            "Cursor CLI ask mode. This model accepts the video file. "
            "Linked-account usage is unpriced. An API key run is not. Temperature is not enabled."
        )
    elif provider_id == "cursor":
        note = (
            "Manually added Cursor model. The CLI runs in ask mode. Linked-account usage is "
            "unpriced. An API key run is not. Native video and temperature are not enabled."
        )
    else:
        note = (
            "Manually added model id. Capabilities are a conservative default, not a live "
            "provider capability record. Native video is not enabled for OpenAI or Anthropic defaults."
        )
    return ModelCapabilities(
        model_id=model_id,
        provider_id=provider_id,
        image=provider_id not in {"manual_json"},
        native_video=native,
        structured_output=provider_id != "meta" or not model_id.startswith("sam"),
        max_images=16 if provider_id != "anthropic" else 20,
        supports_temperature=temperature,
        notes=note,
    )


def models_for(provider_id: str) -> list[ModelCapabilities]:
    return [caps for (provider, _model), caps in CATALOG.items() if provider == provider_id]
