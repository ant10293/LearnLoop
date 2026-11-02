"""Detect an audio stream. A silent or missing track is a normal video."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioInfo:
    probed: bool
    has_audio: bool
    codec: str
    channels: int
    note: str


def probe_audio(path: Path) -> AudioInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return AudioInfo(True, False, "", 0, "ffprobe could not read streams. Visual analysis is unchanged.")
    try:
        streams = json.loads(proc.stdout).get("streams") or []
    except json.JSONDecodeError:
        return AudioInfo(True, False, "", 0, "ffprobe returned malformed stream data.")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio is None:
        return AudioInfo(True, False, "", 0, "This recording has no audio track.")
    return AudioInfo(
        True,
        True,
        str(audio.get("codec_name") or ""),
        int(audio.get("channels") or 0),
        "An audio track is present. Speech has not been transcribed yet.",
    )
