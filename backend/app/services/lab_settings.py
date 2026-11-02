"""Lab preferences stored beside the media files."""

from __future__ import annotations

import json

from app.services.storage import data_path, ensure_data_dirs

COPY = "copy"
REFERENCE = "reference"
MODES = {COPY, REFERENCE}


def load_settings() -> dict[str, str]:
    ensure_data_dirs()
    path = data_path("settings.json")
    mode = COPY
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            raw = {}
        if isinstance(raw, dict) and raw.get("video_storage") in MODES:
            mode = str(raw["video_storage"])
    return {"video_storage": mode}


def save_settings(video_storage: str) -> dict[str, str]:
    if video_storage not in MODES:
        raise ValueError("video storage must be copy or reference")
    ensure_data_dirs()
    payload = {"video_storage": video_storage}
    data_path("settings.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
