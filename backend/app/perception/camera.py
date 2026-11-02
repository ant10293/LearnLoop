"""Image-plane camera motion. This is not a world coordinate frame."""

from __future__ import annotations

import cv2
import numpy as np


def compensate(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, bool]:
    """Warp the previous frame toward the current one with a partial affine model."""
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(previous_gray, maxCorners=80, qualityLevel=0.01, minDistance=7)
    if features is None or len(features) < 6:
        return current, False
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, features, None)
    if nxt is None or status is None:
        return current, False
    good_previous = features[status.flatten() == 1]
    good_next = nxt[status.flatten() == 1]
    if len(good_previous) < 6:
        return current, False
    matrix, inliers = cv2.estimateAffinePartial2D(good_previous, good_next, method=cv2.RANSAC)
    if matrix is None:
        return current, False
    if inliers is not None and int(inliers.sum()) < 6:
        return current, False
    height, width = current.shape[:2]
    warped = cv2.warpAffine(previous, matrix, (width, height))
    return warped, True


def flow_box(
    previous: np.ndarray,
    current: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Shift a box by the median flow inside it. None when the flow is too thin."""
    height, width = current.shape[:2]
    x, y, w, h = box
    x0 = max(0, int(x * width))
    y0 = max(0, int(y * height))
    x1 = min(width, int((x + w) * width))
    y1 = min(height, int((y + h) * height))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    roi = previous_gray[y0:y1, x0:x1]
    points = cv2.goodFeaturesToTrack(roi, maxCorners=24, qualityLevel=0.01, minDistance=3)
    if points is None or len(points) < 4:
        return None
    points = points.reshape(-1, 2)
    points[:, 0] += x0
    points[:, 1] += y0
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points.reshape(-1, 1, 2).astype(np.float32), None
    )
    if nxt is None or status is None:
        return None
    keep = status.flatten() == 1
    if int(keep.sum()) < 4:
        return None
    shift = np.median(nxt.reshape(-1, 2)[keep] - points[keep], axis=0)
    return (
        min(max(x + float(shift[0]) / width, 0.0), 0.98),
        min(max(y + float(shift[1]) / height, 0.0), 0.98),
        w,
        h,
    )
