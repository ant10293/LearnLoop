"""Write SAM mask rasters and cutouts. A failed decode keeps the raw payload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.providers.sam import SamRecord, decode_mask_raster, normalized_box


@dataclass
class MaskFiles:
    png_path: Path | None
    raw_path: Path | None
    cutout_path: Path | None
    decoded: bool
    error: str = ""


def write_mask_png(raster: bytes, height: int, width: int, dest: Path) -> None:
    array = np.frombuffer(raster, dtype=np.uint8)
    if array.size != height * width:
        raise ValueError(
            f"Mask raster has {array.size} values and the header says {height}x{width}."
        )
    mask = array.reshape(height, width)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., 1] = 210
    image[..., 3] = np.where(mask > 0, 170, 0).astype(np.uint8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), image):
        raise ValueError(f"Could not write {dest}")


def write_raw_mask(payload: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload)


def cutout_from_mask(
    image_bgr: np.ndarray,
    raster: bytes,
    height: int,
    width: int,
    box: dict | None = None,
) -> np.ndarray | None:
    """A SAM raster covers its box, not the frame. `box` is normalized to the frame."""
    array = np.frombuffer(raster, dtype=np.uint8)
    if array.size != height * width:
        return None
    mask = array.reshape(height, width)
    if box is not None:
        frame_h, frame_w = image_bgr.shape[:2]
        x0 = min(max(int(round(box["x"] * frame_w)), 0), frame_w - 1)
        y0 = min(max(int(round(box["y"] * frame_h)), 0), frame_h - 1)
        x1 = min(max(int(round((box["x"] + box["w"]) * frame_w)), x0 + 1), frame_w)
        y1 = min(max(int(round((box["y"] + box["h"]) * frame_h)), y0 + 1), frame_h)
        placed = np.zeros((frame_h, frame_w), dtype=np.uint8)
        placed[y0:y1, x0:x1] = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        mask = placed
    if mask.shape[0] != image_bgr.shape[0] or mask.shape[1] != image_bgr.shape[1]:
        mask = cv2.resize(
            mask,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = image_bgr[y1:y2, x1:x2]
    alpha = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def materialize_record(
    record: SamRecord,
    directory: Path,
    stem: str,
    *,
    image_bgr: np.ndarray | None = None,
    decode: bool = True,
) -> MaskFiles:
    """Decode when possible. Never invent a cutout for a payload that did not decode.

    `decode=False` keeps only the payload and leaves `error` empty. The decoder
    takes about 0.4 s per mask in Python, and a video run can have thousands.
    """
    raw_path = directory / f"{stem}.txt"
    write_raw_mask(record.mask_data or record.mask_payload, raw_path)
    if not decode:
        return MaskFiles(None, raw_path, None, False)
    raster = decode_mask_raster(record)
    if raster is None:
        return MaskFiles(None, raw_path, None, False, "Mask raster was not decoded.")
    height = record.mask_height
    width = record.mask_width
    if height <= 0 or width <= 0:
        return MaskFiles(None, raw_path, None, False, "Mask header did not include a size.")
    png_path = directory / f"{stem}.png"
    try:
        write_mask_png(raster, height, width, png_path)
    except Exception as exc:
        return MaskFiles(None, raw_path, None, False, str(exc))
    cutout_path = None
    if image_bgr is not None:
        cutout = cutout_from_mask(image_bgr, raster, height, width, normalized_box(record))
        if cutout is not None:
            cutout_path = directory / f"{stem}_cutout.png"
            if not cv2.imwrite(str(cutout_path), cutout):
                cutout_path = None
    return MaskFiles(png_path, raw_path, cutout_path, True, "")
