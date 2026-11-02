"""The model picker is the account's list, with the static catalog as a fallback.

Every remote provider asks the signed-in account which models it can run.
A known catalog row still supplies price and the limits the live list does
not. An id the catalog has never seen stays unpriced. Local providers are not
accounts, so their single catalog row stays put. The same rows are what a run
validates against.

A generation request whose model is not on that live list uses the account's
marked default, or the first listed model that can take the run's modality.
The static catalog is not a substitute model.
"""

from __future__ import annotations

import re
import threading
import time

from app.providers.catalog import lookup, models_for
from app.providers.types import ModelCapabilities

_LOCAL = {"deterministic", "fixture", "manual_json"}
_TTL_S = 180.0
_LIVE_NOTE = (
    "Reported by the connected account. This model is not in the price catalog, "
    "so the call stays unpriced."
)
_SKIPPED_API = (
    "embedding",
    "whisper",
    "tts",
    "audio",
    "realtime",
    "moderation",
    "transcribe",
    "dall-e",
    "babbage",
    "davinci",
    "imagen",
    "sora",
)

_lock = threading.Lock()
_cache: dict[str, tuple[float, list[dict]]] = {}
_saved: dict[tuple[str, str], dict] = {}


def options_for(
    provider_id: str,
    catalog: list[dict],
    secret: str = "",
    base_url: str = "",
    *,
    refresh: bool = False,
) -> list[dict]:
    """Models to show for this provider. The catalog is returned unchanged when the account cannot be asked."""
    if provider_id in _LOCAL:
        return catalog
    key = _cache_key(provider_id, secret, base_url)
    now = time.monotonic()
    if not refresh:
        with _lock:
            hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    try:
        fresh = _load(provider_id, catalog, secret, base_url)
    except Exception:
        return catalog
    if fresh is None:
        return catalog
    with _lock:
        _cache[key] = (time.monotonic() + _TTL_S, fresh)
    return fresh


def use_video_file(provider_id: str, model_id: str) -> bool:
    """True when this analysis model accepts a video file."""
    return capabilities_for(provider_id, model_id).native_video


def capabilities_for(provider_id: str, model_id: str) -> ModelCapabilities:
    """Capability row for a run. Live fields win once that account list has been read."""
    with _lock:
        saved = _saved.get((provider_id, model_id))
    if saved is None:
        return lookup(provider_id, model_id)
    efforts = saved.get("reasoning_efforts") or ()
    return ModelCapabilities(
        model_id=model_id,
        provider_id=provider_id,
        text=bool(saved.get("text", True)),
        image=bool(saved.get("image", False)),
        native_video=bool(saved.get("native_video", False)),
        structured_output=bool(saved.get("structured_output", False)),
        max_images=int(saved.get("max_images") or 16),
        max_video_seconds=saved.get("max_video_seconds"),
        context_tokens=saved.get("context_tokens"),
        reasoning_efforts=tuple(efforts),
        supports_temperature=bool(saved.get("supports_temperature", True)),
        input_price_per_mtok=saved.get("input_price_per_mtok"),
        output_price_per_mtok=saved.get("output_price_per_mtok"),
        notes=str(saved.get("notes") or ""),
    )


def live_options(
    provider_id: str,
    secret: str = "",
    base_url: str = "",
    *,
    refresh: bool = False,
) -> list[dict] | None:
    """The account's models, or None when that list cannot be read.

    The static catalog is not returned here. A failed account lookup must not
    be treated as the set of models the account can run.
    """
    if provider_id in _LOCAL:
        return None
    catalog = [caps.as_dict() for caps in models_for(provider_id)]
    options = options_for(provider_id, catalog, secret, base_url, refresh=refresh)
    if options is catalog:
        return None
    return options


def choose_fallback(
    options: list[dict],
    *,
    image: bool = False,
    native_video: bool = False,
) -> str:
    """Pick from the account list. A row the account marks as default wins."""
    eligible = [item for item in options if _serves(item, image, native_video)] or list(options)
    for item in eligible:
        if _marked_default(item):
            return str(item["model_id"])
    if not eligible:
        return ""
    return str(eligible[0]["model_id"])


def resolve_model_id(session, provider_id: str, model_id: str, modality: str) -> str:
    """Keep model_id when the account lists it. Otherwise use a live-list fallback.

    When the account list cannot be read, model_id is unchanged.
    """
    if provider_id in _LOCAL:
        return model_id
    secret, base_url = _account(session, provider_id)
    image = modality == "sampled_frames"
    native_video = modality == "native_video"
    options = live_options(provider_id, secret, base_url)
    if options is None:
        return model_id
    if _listed(options, model_id):
        return model_id
    refreshed = live_options(provider_id, secret, base_url, refresh=True)
    if refreshed is not None:
        options = refreshed
    if _listed(options, model_id):
        return model_id
    chosen = choose_fallback(options, image=image, native_video=native_video)
    return chosen or model_id


def _listed(options: list[dict], model_id: str) -> bool:
    return any(str(item.get("model_id")) == model_id for item in options)


def _marked_default(item: dict) -> bool:
    if item.get("default") is True:
        return True
    return bool(re.search(r"\bdefault\b", str(item.get("label") or ""), re.IGNORECASE))


def _serves(item: dict, image: bool, native_video: bool) -> bool:
    if native_video and not item.get("native_video"):
        return False
    if image and item.get("image") is False:
        return False
    return True


def _account(session, provider_id: str) -> tuple[str, str]:
    from app.models.runs import ProviderStatus
    from app.services.credentials import load_secret

    secret, _source = load_secret(session, provider_id)
    status = session.get(ProviderStatus, provider_id)
    return secret, status.extra_base_url if status else ""


def warm(provider_id: str, secret: str = "", base_url: str = "") -> None:
    catalog = [caps.as_dict() for caps in models_for(provider_id)]
    options_for(provider_id, catalog, secret, base_url=base_url)


def overlay_options(provider_id: str, catalog: list[dict], records: list[dict]) -> list[dict]:
    """One account row per live model. Catalog prices stay on ids this lab already knows."""
    by_id = {str(item.get("model_id")): item for item in catalog}
    options: list[dict] = []
    seen: set[str] = set()
    for record in records:
        model_id = str(record.get("model_id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        known = model_id in by_id
        base = dict(by_id[model_id]) if known else lookup(provider_id, model_id).as_dict()
        label = str(record.get("label") or "").strip()
        if label:
            base["label"] = label
        for key in (
            "image",
            "native_video",
            "structured_output",
            "max_images",
            "context_tokens",
            "supports_temperature",
        ):
            if record.get(key) is not None and key in record:
                base[key] = record[key]
        if record.get("reasoning_efforts") is not None:
            base["reasoning_efforts"] = list(record["reasoning_efforts"])
        if not known:
            base["notes"] = str(record.get("notes") or _LIVE_NOTE)
            base["input_price_per_mtok"] = None
            base["output_price_per_mtok"] = None
            base["price_known"] = False
        options.append(base)
    with _lock:
        for item in options:
            _saved[(provider_id, str(item["model_id"]))] = item
    return options


def _cache_key(provider_id: str, secret: str, base_url: str) -> str:
    from app.providers.codex_cli import load_codex
    from app.providers.cursor_cli import load_cursor_cli

    transport = ""
    binary = ""
    if provider_id == "openai":
        settings = load_codex()
        transport = settings["transport"]
        binary = settings["binary"]
    elif provider_id == "cursor":
        settings = load_cursor_cli()
        transport = settings["transport"]
        binary = settings["binary"]
    elif provider_id == "anthropic":
        from app.providers.claude_code import load_claude_code

        settings = load_claude_code()
        transport = settings["transport"]
        binary = settings["binary"]
    tail = secret[-4:] if secret else ""
    return f"{provider_id}|{transport}|{binary}|{base_url}|{tail}"


def _load(provider_id: str, catalog: list[dict], secret: str, base_url: str) -> list[dict] | None:
    if provider_id == "cursor":
        from app.providers.cursor_cli import account_model_options

        options = account_model_options(catalog, secret)
        if options is catalog:
            return None
        return options
    records = _fetch_records(provider_id, secret, base_url)
    if not records:
        return None
    return overlay_options(provider_id, catalog, records)


def _fetch_records(provider_id: str, secret: str, base_url: str) -> list[dict]:
    if provider_id == "openai":
        from app.providers.codex_cli import load_codex, model_records_sync

        if load_codex()["transport"] == "codex":
            return model_records_sync()
        if not secret:
            return []
        return _openai_api_models(secret)
    if provider_id == "openai_compatible":
        if not secret or not base_url:
            return []
        return _openai_api_models(secret, base_url)
    if provider_id == "anthropic":
        if not secret:
            return []
        return _anthropic_models(secret)
    if provider_id == "gemini":
        if not secret:
            return []
        return _gemini_models(secret)
    if provider_id == "meta":
        if not secret:
            return []
        return _meta_models(secret)
    return []


def chat_model(model_id: str) -> bool:
    name = model_id.lower()
    if any(part in name for part in _SKIPPED_API):
        return False
    return name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-", "codex-"))


def _openai_api_models(secret: str, base_url: str = "") -> list[dict]:
    from openai import OpenAI

    kwargs: dict = {"api_key": secret, "timeout": 20}
    if base_url:
        kwargs["base_url"] = base_url
    page = OpenAI(**kwargs).models.list()
    ranked: list[tuple[int, str]] = []
    for item in page.data:
        model_id = str(getattr(item, "id", "") or "")
        if not chat_model(model_id):
            continue
        created = int(getattr(item, "created", 0) or 0)
        ranked.append((created, model_id))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [{"model_id": model_id} for _created, model_id in ranked]


def _anthropic_models(secret: str) -> list[dict]:
    import anthropic

    page = anthropic.Anthropic(api_key=secret, timeout=20).models.list(limit=100)
    records = []
    for item in page.data:
        label = str(getattr(item, "display_name", "") or "")
        records.append({"model_id": item.id, "label": label})
    return records


def _gemini_models(secret: str) -> list[dict]:
    from google import genai

    records = []
    for model in genai.Client(api_key=secret).models.list():
        short = str(getattr(model, "name", "") or "").split("/")[-1]
        actions = list(getattr(model, "supported_actions", None) or [])
        if actions and "generateContent" not in actions:
            continue
        lowered = short.lower()
        if not short.startswith("gemini") or "embed" in lowered or "aqa" in lowered:
            continue
        records.append({"model_id": short, "label": str(getattr(model, "display_name", "") or "")})
    return records


def _meta_models(secret: str) -> list[dict]:
    import httpx

    response = httpx.get(
        "https://api.meta.ai/v1/models",
        headers={"Authorization": f"Bearer {secret}"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or payload.get("models") or []
    records = []
    for item in data:
        if isinstance(item, str):
            model_id, label = item, ""
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or "")
            label = str(item.get("display_name") or item.get("label") or "")
        else:
            continue
        if model_id:
            records.append({"model_id": model_id, "label": label})
    return records
