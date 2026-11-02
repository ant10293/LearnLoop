"""Local segmentation. Masks are estimates from a box, not a language-model identity."""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.analysis import BBox


def grabcut_mask(image: np.ndarray, bbox: BBox) -> np.ndarray | None:
    """Full-frame foreground mask. None when GrabCut cannot separate the region."""
    height, width = image.shape[:2]
    x = max(0, int(bbox.x * width))
    y = max(0, int(bbox.y * height))
    w = max(2, min(width - x, int(bbox.w * width)))
    h = max(2, min(height - y, int(bbox.h * height)))
    if w < 4 or h < 4:
        return None
    mask = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, (x, y, w, h), background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    keep = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if int(keep.sum()) < 10:
        return None
    return keep


def grabcut_cutout(image: np.ndarray, bbox: BBox) -> np.ndarray | None:
    mask = grabcut_mask(image, bbox)
    if mask is None:
        return None
    height, width = image.shape[:2]
    x = max(0, int(bbox.x * width))
    y = max(0, int(bbox.y * height))
    w = max(2, min(width - x, int(bbox.w * width)))
    h = max(2, min(height - y, int(bbox.h * height)))
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra[y : y + h, x : x + w]


def crop_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = box
    x0 = max(0, int(x * width))
    y0 = max(0, int(y * height))
    x1 = min(width, max(x0 + 1, int((x + w) * width)))
    y1 = min(height, max(y0 + 1, int((y + h) * height)))
    crop = image[y0:y1, x0:x1].copy()
    if mask is not None and mask.shape[:2] == image.shape[:2] and crop.size:
        crop_mask = mask[y0:y1, x0:x1]
        crop[crop_mask == 0] = 0
    return crop
