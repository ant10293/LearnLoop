"""Bounded ffmpeg decode passes on the source clock.

Each pass seeks the requested interval, keeps at most one real frame per time
cell, and reads that frame's presentation time from `showinfo`. Nothing here
holds more than one full-resolution frame in memory.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.video.probe import VideoError

_PTS = re.compile(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*(-?[0-9.]+)")


def _select(step: float) -> str:
    """First real frame in each `step`-second cell. Times keep their source spacing."""
    if step <= 0:
        return "select='1'"
    return f"select='isnan(prev_selected_t)+gte(floor(t/{step:.6f}),floor(prev_selected_t/{step:.6f})+1)'"


def _times(log: str, offset: float) -> list[float]:
    found: list[float] = []
    for line in log.splitlines():
        if "showinfo" not in line:
            continue
        match = _PTS.search(line)
        if match:
            found.append(round(offset + float(match.group(2)), 6))
    return found


def _seek(start: float, end: float) -> list[str]:
    args = []
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    return args


def _limit(start: float, end: float) -> list[str]:
    if end <= start:
        return []
    return ["-t", f"{end - start + 0.001:.3f}"]


def gray_pass(
    source: Path,
    start: float,
    end: float,
    step: float,
    size: tuple[int, int] = (160, 90),
) -> tuple[list[np.ndarray], list[float]]:
    """Small grayscale frames for motion analysis, with source times.

    Frames arrive through a pipe. The full-resolution picture is never stored.
    """
    width, height = size
    with tempfile.TemporaryFile(mode="w+") as log:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            *_seek(start, end),
            "-i",
            str(source),
            *_limit(start, end),
            "-an",
            "-vf",
            f"{_select(step)},showinfo,scale={width}:{height},format=gray",
            "-fps_mode",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        frames: list[np.ndarray] = []
        frame_bytes = width * height
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break
            frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width).copy())
        proc.stdout.close()
        code = proc.wait()
        log.seek(0)
        text = log.read()
    if code != 0:
        raise VideoError(text.strip()[-1500:] or "ffmpeg could not decode the motion pass")
    times = _times(text, start)
    count = min(len(frames), len(times))
    return frames[:count], times[:count]


def jpeg_pass(source: Path, start: float, end: float, step: float, dest: Path) -> list[tuple[float, Path]]:
    """Write one JPEG per time cell into `dest` and return source times with paths."""
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-y",
        *_seek(start, end),
        "-i",
        str(source),
        *_limit(start, end),
        "-an",
        "-vf",
        f"{_select(step)},showinfo",
        "-fps_mode",
        "vfr",
        "-q:v",
        "3",
        str(dest / "%06d.jpg"),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise VideoError(proc.stderr.strip()[-1500:] or "ffmpeg could not extract tracking frames")
    files = sorted(dest.glob("*.jpg"))
    times = _times(proc.stderr, start)
    count = min(len(files), len(times))
    for extra in files[count:]:
        extra.unlink()
    return list(zip(times[:count], files[:count], strict=True))


def chunk_bounds(start: float, end: float, length: float) -> Iterator[tuple[float, float]]:
    """Consecutive processing chunks that cover the interval end to end."""
    if end <= start:
        yield start, end
        return
    cursor = start
    while cursor < end - 1e-6:
        stop = min(end, cursor + length)
        yield cursor, stop
        cursor = stop


def cut_clip(source: Path, start: float, end: float, dest: Path) -> dict:
    """Re-encode `[start, end]` into `dest`, keeping every source frame and the displayed orientation.

    Returns the transform. Clip time zero is source time `offset`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_seek(start, end),
        "-i",
        str(source),
        *_limit(start, end),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dest.exists():
        raise VideoError(proc.stderr.strip()[-1500:] or "ffmpeg could not cut the requested interval")
    return {
        "kind": "subclip",
        "offset": round(start, 6),
        "duration": round(end - start, 6),
        "orientation": "displayed",
        "note": "Clip time zero is source time `offset`. Frames are re-encoded without dropping or duplicating.",
    }
