"""SAM responses survive a restart, and saving does not hold every decoded frame."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from app.config import settings
from app.pipeline import sam_stage
from app.services import materialize


def _plan(prompt="wheel"):
    return SimpleNamespace(session_id="c0s1", kind="concept", prompt=prompt, propagation=None)


def _chunk():
    return SimpleNamespace(time_start=0.0, time_end=2.0, frame_count=120)


def test_a_saved_response_comes_back_only_for_the_same_request(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    result = SimpleNamespace(text="records", request_id="req-1")
    sam_stage._save_response("run", _plan(), _chunk(), result)
    assert sam_stage._load_response("run", _plan(), _chunk()) == {"text": "records", "request_id": "req-1"}
    assert sam_stage._load_response("run", _plan("hub"), _chunk()) is None


def test_a_damaged_response_is_requested_again(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    path = sam_stage._response_path("run", _plan(), _chunk())
    path.parent.mkdir(parents=True)
    path.write_text('{"text": ')
    assert sam_stage._load_response("run", _plan(), _chunk()) is None


def test_extract_images_decodes_each_frame_once_while_scoring(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    frames = []
    for index in range(2):
        name = f"f{index}.png"
        cv2.imwrite(str(tmp_path / name), np.random.default_rng(index).integers(0, 255, (80, 80, 3), dtype=np.uint8))
        frames.append(SimpleNamespace(id=f"frame-{index}", path=name))
    observations = [
        SimpleNamespace(
            id=f"observation-{index}",
            frame_id=f"frame-{index % 2}",
            bbox_json={"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
            visibility="visible",
            occlusion=0,
            sharpness=None,
            timestamp=float(index),
        )
        for index in range(40)
    ]
    reads = []
    real_read = materialize.read_bgr
    monkeypatch.setattr(materialize, "read_bgr", lambda path: reads.append(path) or real_read(path))
    session = SimpleNamespace(add=lambda row: None, flush=lambda: None)
    obj = SimpleNamespace(id="object", local_id="Object_1", color="#ff0000", primary_image_id=None)

    materialize._extract_images(session, "run", "revision", obj, observations, frames)

    # Two to score, then one for each of the five curated images.
    assert len(reads) == 2 + 5
    assert all(obs.sharpness is not None for obs in observations)
    assert list((tmp_path / "runs" / "run" / "objects" / "Object_1").glob("primary_*.png"))
