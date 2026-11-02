"""Tracking frames are cached apart from the language-model sample."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def extract_tracking_frames(
    source: Path,
    dest: Path,
    fps: float,
    maximum: int,
    duration: float,
) -> list[tuple[float, Path]]:
    """One ffmpeg pass. A matching manifest is reused and the source is not copied."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    key = {"fps": float(fps), "maximum": int(maximum), "duration": round(float(duration), 3)}
    if manifest_path.exists():
        try:
            saved = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            saved = {}
        if saved.get("key") == key:
            rows = [(float(item["timestamp"]), dest / item["name"]) for item in saved.get("frames") or []]
            if rows and all(path.exists() for _timestamp, path in rows):
                return rows
    for old in dest.glob("*.jpg"):
        old.unlink()
    pattern = dest / "%05d.jpg"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(int(maximum)),
            "-q:v",
            "3",
            str(pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        if len(detail) > 1500:
            detail = detail[-1500:]
        raise RuntimeError(detail or "ffmpeg could not extract tracking frames")
    files = sorted(dest.glob("*.jpg"))
    rows = []
    for index, path in enumerate(files):
        timestamp = round(index / float(fps), 3)
        if duration > 0 and timestamp > duration + 0.05:
            break
        rows.append((timestamp, path))
    manifest_path.write_text(
        json.dumps(
            {
                "key": key,
                "frames": [{"timestamp": timestamp, "name": path.name} for timestamp, path in rows],
            }
        )
    )
    return rows
