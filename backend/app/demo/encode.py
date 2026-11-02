"""Write synthetic clips with ffmpeg so the stored original is browser-playable H.264."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from app.demo.scenarios import FPS, HEIGHT, WIDTH
from app.video.probe import VideoError


def write_clip(frames: list[np.ndarray], dest: Path, fps: int = FPS, creation_time: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if creation_time:
        cmd.extend(["-metadata", f"creation_time={creation_time}"])
    cmd.append(str(dest))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        if frame.shape[0] != HEIGHT or frame.shape[1] != WIDTH:
            raise VideoError("synthetic frame has an unexpected size")
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    error = proc.stderr.read().decode() if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise VideoError(error[-500:] or "ffmpeg could not encode the synthetic clip")
