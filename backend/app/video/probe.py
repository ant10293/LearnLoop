"""ffprobe metadata and presentation-timestamp indexing."""

from __future__ import annotations

import bisect
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class VideoError(RuntimeError):
    pass


@dataclass
class VideoMetadata:
    duration_sec: float
    width: int
    height: int
    display_width: int
    display_height: int
    fps: float
    avg_fps: float
    time_base: str
    rotation_deg: int
    variable_frame_rate: bool
    frame_count: int
    codec: str
    container: str
    pix_fmt: str
    mime_hint: str


def sha256_file(path: Path, on_bytes: Callable[[int, int], None] | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    read = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            read += len(chunk)
            if on_bytes is not None:
                on_bytes(read, total)
    return digest.hexdigest()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _ratio(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: Path) -> VideoMetadata:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise VideoError(proc.stderr.strip() or "ffprobe could not read this file")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoError("ffprobe returned malformed JSON") from exc
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoError("no video stream found")
    fmt = payload.get("format") or {}
    duration = float(video.get("duration") or fmt.get("duration") or 0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    fps = _ratio(video.get("r_frame_rate"))
    avg = _ratio(video.get("avg_frame_rate")) or fps
    vfr = bool(fps and avg and abs(fps - avg) / max(fps, 1e-6) > 0.02)
    frames = int(video.get("nb_frames") or 0)
    if frames <= 0 and avg and duration:
        frames = int(round(avg * duration))
    display_w, display_h = (height, width) if abs(rotation) % 180 == 90 else (width, height)
    container = str(fmt.get("format_name") or "")
    codec = str(video.get("codec_name") or "")
    return VideoMetadata(
        duration_sec=duration,
        width=width,
        height=height,
        display_width=display_w,
        display_height=display_h,
        fps=fps,
        avg_fps=avg,
        time_base=str(video.get("time_base") or ""),
        rotation_deg=rotation,
        variable_frame_rate=vfr,
        frame_count=frames,
        codec=codec,
        container=container,
        pix_fmt=str(video.get("pix_fmt") or ""),
        mime_hint=_mime(container, path),
    )


def _rotation(stream: dict) -> int:
    tags = stream.get("tags") or {}
    raw = tags.get("rotate")
    if raw is not None:
        try:
            return int(raw) % 360
        except ValueError:
            pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    return 0


def _mime(container: str, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".webm" or "webm" in container:
        return "video/webm"
    if suffix == ".mov" or "mov" in container.split(",")[0]:
        return "video/quicktime"
    return "video/mp4"


def parse_presentation_times(stdout: str) -> list[float]:
    """Read ffprobe CSV frame times.

    A frame with side data (Dolby Vision, rotation, ambient metadata) is printed
    as ``0.000000,,,,``. The time is the first field. Later fields are not part
    of the timestamp, and parsing the whole line drops every frame.
    """
    times: list[float] = []
    for line in stdout.splitlines():
        field = line.split(",", 1)[0].strip()
        if not field or field == "N/A":
            continue
        try:
            times.append(float(field))
        except ValueError:
            continue
    return times


def presentation_timestamps(path: Path, cache: Path | None = None) -> list[float]:
    """Actual frame presentation times. Cached because ffprobe frame scans are expensive."""
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if isinstance(data, list) and data:
                return [float(item) for item in data]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise VideoError(_tail(proc.stderr) or "could not read frame timestamps")
    times = parse_presentation_times(proc.stdout)
    if not times:
        sample = next((line.strip() for line in proc.stdout.splitlines() if line.strip()), "")
        detail = f" Unparsed line: {sample[:160]}" if sample else ""
        raise VideoError(f"could not read frame timestamps.{detail}")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(times))
    return times


def _tail(text: str, limit: int = 1500) -> str:
    detail = text.strip()
    if len(detail) <= limit:
        return detail
    return detail[-limit:]


def snap_timestamps(targets: list[float], pts: list[float]) -> list[float]:
    """Map requested times onto real presentation timestamps."""
    if not pts:
        return _unique(targets)
    ordered = sorted(pts)
    snapped: list[float] = []
    for target in targets:
        index = bisect.bisect_left(ordered, target)
        options = [ordered[i] for i in (index - 1, index) if 0 <= i < len(ordered)]
        nearest = min(options, key=lambda item: abs(item - target))
        snapped.append(round(nearest, 6))
    return _unique(snapped)


def _unique(values: list[float]) -> list[float]:
    seen: set[float] = set()
    ordered: list[float] = []
    for value in values:
        key = round(value, 3)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def browser_compatible(meta: VideoMetadata) -> bool:
    h264 = meta.codec in {"h264", "avc1"}
    yuv = meta.pix_fmt in {"yuv420p", "yuvj420p"}
    mp4 = "mp4" in meta.container or "mov" in meta.container
    return h264 and yuv and mp4
