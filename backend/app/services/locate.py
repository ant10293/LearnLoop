"""Find a dropped video's original path without copying the file."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from app.video.probe import VideoError

_SKIP = {"node_modules", ".git", "Library", ".Trash", "venv", ".venv", "__pycache__", "dist", "build"}
_MTIME_TOLERANCE_MS = 2000
_WALK_SECONDS = 4


def default_search_roots() -> list[Path]:
    home = Path.home()
    names = ("Downloads", "Desktop", "Movies", "Documents", "Videos")
    return [home / name for name in names if (home / name).is_dir()]


def spotlight_paths(name: str) -> list[Path]:
    try:
        completed = subprocess.run(
            ["mdfind", "-name", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return [Path(line) for line in completed.stdout.splitlines() if line]


def walk_named(roots: list[Path], name: str) -> list[Path]:
    found: list[Path] = []
    deadline = time.monotonic() + _WALK_SECONDS
    for root in roots:
        if time.monotonic() > deadline:
            break
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if time.monotonic() > deadline:
                break
            dirnames[:] = [item for item in dirnames if item not in _SKIP and not item.startswith(".")]
            if name in filenames:
                found.append(Path(dirpath) / name)
    return found


def _check_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or cleaned != Path(cleaned).name or any(ord(ch) < 32 for ch in cleaned):
        raise VideoError("that drop did not include a video file name")
    return cleaned


def _named_file(path: Path, name: str) -> bool:
    try:
        return path.name == name and path.is_file()
    except OSError:
        return False


def _pick(candidates: list[Path], name: str, size: int, mtime_ms: int) -> Path:
    seen: set[Path] = set()
    sized: list[Path] = []
    for raw in candidates:
        try:
            if not _named_file(raw, name):
                continue
            path = raw.resolve()
            if path in seen:
                continue
            seen.add(path)
            if path.stat().st_size != size:
                continue
            sized.append(path)
        except OSError:
            continue
    if not sized:
        raise VideoError("Could not find the original file")

    def delta(path: Path) -> float:
        return abs(path.stat().st_mtime * 1000 - mtime_ms)

    close = [path for path in sized if delta(path) <= _MTIME_TOLERANCE_MS]
    if len(close) == 1:
        return close[0]
    if len(close) > 1:
        raise VideoError("Several files match that video")
    if len(sized) == 1:
        return sized[0]
    raise VideoError("Several files match that video")


def locate_dropped_video(name: str, size: int, mtime_ms: int) -> Path:
    """Resolve a drag-and-drop to the file on disk. The file is not copied."""
    cleaned = _check_name(name)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise VideoError("Could not find the original file")
    if isinstance(mtime_ms, bool) or not isinstance(mtime_ms, int):
        raise VideoError("Could not find the original file")
    indexed = [path for path in spotlight_paths(cleaned) if _named_file(path, cleaned)]
    if indexed:
        try:
            return _pick(indexed, cleaned, size, mtime_ms)
        except VideoError as exc:
            if "Several" in str(exc):
                raise
    return _pick([*indexed, *walk_named(default_search_roots(), cleaned)], cleaned, size, mtime_ms)
