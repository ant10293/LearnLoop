"""Frame extraction, proxies, thumbnails, and contact sheets."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.video.probe import VideoError


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        if len(detail) > 1500:
            detail = detail[-1500:]
        raise VideoError(detail or f"command failed: {cmd[0]}")


def make_proxy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dest),
        ]
    )


def extract_frame(source: Path, timestamp: float, dest: Path, accurate: bool = True) -> None:
    """Extract one frame at a presentation timestamp. The original file is only read."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{max(timestamp, 0):.3f}"
    if accurate:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            stamp,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-update",
            "1",
            str(dest),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            stamp,
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-update",
            "1",
            str(dest),
        ]
    _run(cmd)


def thumbnail(source: Path, timestamp: float, dest: Path) -> None:
    extract_frame(source, timestamp, dest, accurate=False)
    image = Image.open(dest)
    image.thumbnail((480, 270))
    image.save(dest, format="JPEG", quality=85)


def contact_sheet(frame_paths: list[Path], dest: Path, columns: int = 5) -> None:
    images = []
    for path in frame_paths:
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    if not images:
        raise VideoError("no frames available for a contact sheet")
    thumb_w = 192
    tiles = []
    for image in images:
        image.thumbnail((thumb_w, thumb_w))
        tiles.append(image)
    cell_w = max(tile.width for tile in tiles)
    cell_h = max(tile.height for tile in tiles) + 16
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(tiles):
        col = index % columns
        row = index // columns
        x = col * cell_w + (cell_w - tile.width) // 2
        y = row * cell_h
        sheet.paste(tile, (x, y))
        draw.text((col * cell_w + 4, y + tile.height + 1), str(index + 1), fill=(180, 190, 205))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=85)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise VideoError(f"could not decode image {path.name}")
    return image


def sharpness_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
