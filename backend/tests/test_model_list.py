"""A missing generation model is replaced from the account list, not a fixed id."""

from app.providers.model_list import choose_fallback, resolve_model_id


def test_listed_model_is_kept(monkeypatch):
    options = [
        {"model_id": "composer-2.5", "label": "Composer 2.5", "image": True},
        {"model_id": "auto", "label": "Auto (default)", "image": True},
    ]
    monkeypatch.setattr("app.providers.model_list._account", lambda *_args, **_kwargs: ("", ""))
    monkeypatch.setattr("app.providers.model_list.live_options", lambda *_args, **_kwargs: options)
    assert resolve_model_id(None, "cursor", "composer-2.5", "sampled_frames") == "composer-2.5"


def test_missing_model_uses_the_account_default(monkeypatch):
    options = [
        {"model_id": "composer-2.5", "label": "Composer 2.5", "image": True, "native_video": False},
        {"model_id": "auto", "label": "Auto (default)", "image": True, "native_video": False},
    ]
    monkeypatch.setattr("app.providers.model_list._account", lambda *_args, **_kwargs: ("", ""))
    monkeypatch.setattr("app.providers.model_list.live_options", lambda *_args, **_kwargs: options)
    assert resolve_model_id(None, "cursor", "grok-4.7-high-fast", "sampled_frames") == "auto"


def test_missing_model_uses_the_first_listed_model_that_can_take_images():
    options = [
        {"model_id": "text-only", "label": "Text (default)", "image": False, "native_video": False},
        {"model_id": "vision", "label": "Vision", "image": True, "native_video": False},
    ]
    assert choose_fallback(options, image=True) == "vision"


def test_native_video_skips_models_that_cannot_take_video():
    options = [
        {"model_id": "auto", "label": "Auto (default)", "image": True, "native_video": False},
        {"model_id": "video-model", "label": "Video", "image": True, "native_video": True},
    ]
    assert choose_fallback(options, native_video=True) == "video-model"


def test_unknown_account_list_keeps_the_requested_model(monkeypatch):
    monkeypatch.setattr("app.providers.model_list._account", lambda *_args, **_kwargs: ("", ""))
    monkeypatch.setattr("app.providers.model_list.live_options", lambda *_args, **_kwargs: None)
    assert resolve_model_id(None, "cursor", "grok-4.7-high-fast", "sampled_frames") == "grok-4.7-high-fast"
