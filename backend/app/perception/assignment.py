"""One-to-one association. A detection is given to at most one track."""

from __future__ import annotations

import math

import numpy as np


def linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular matrix.

    This is the Jonker–Volgenant shortest-augmenting-path method. Pairs that
    were only padding are omitted.
    """
    if cost.ndim != 2 or cost.size == 0:
        return []
    matrix = np.asarray(cost, dtype=float)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    rows, cols = matrix.shape
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    match = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        match[0] = row
        column = 0
        minimum = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column] = True
            current = match[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                value = matrix[current - 1, candidate - 1] - u[current] - v[candidate]
                if value < minimum[candidate]:
                    minimum[candidate] = value
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = float(minimum[candidate])
                    next_column = candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[match[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if match[column] == 0:
                break
        while True:
            previous = way[column]
            match[column] = match[previous]
            column = previous
            if column == 0:
                break
    pairs: list[tuple[int, int]] = []
    for column in range(1, cols + 1):
        if match[column] == 0:
            continue
        left = int(match[column] - 1)
        right = column - 1
        pairs.append((right, left) if transposed else (left, right))
    return pairs


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def shift_box(
    box: tuple[float, float, float, float], dx: float, dy: float
) -> tuple[float, float, float, float]:
    x = min(max(box[0] + dx, 0.0), 0.98)
    y = min(max(box[1] + dy, 0.0), 0.98)
    w = min(max(box[2], 0.01), 1 - x)
    h = min(max(box[3], 0.01), 1 - y)
    return (x, y, w, h)


def prediction_score(predicted: tuple[float, float, float, float], detected: tuple[float, float, float, float]) -> float:
    dx = center(predicted)[0] - center(detected)[0]
    dy = center(predicted)[1] - center(detected)[1]
    return math.exp(-math.hypot(dx, dy) / 0.22)


def size_score(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    area_a = max(a[2] * a[3], 1e-6)
    area_b = max(b[2] * b[3], 1e-6)
    ratio = max(area_a, area_b) / min(area_a, area_b)
    return max(0.0, 1.0 - (ratio - 1.0) / 2.5)


def motion_agreement(
    previous: tuple[float, float, float, float],
    detected: tuple[float, float, float, float],
    vx: float,
    vy: float,
    dt: float,
) -> float:
    if dt <= 1e-6:
        return 1.0
    implied_vx = (center(detected)[0] - center(previous)[0]) / dt
    implied_vy = (center(detected)[1] - center(previous)[1]) / dt
    gap = math.hypot(implied_vx - vx, implied_vy - vy)
    return math.exp(-gap / 0.8)


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    a = left > 0
    b = right > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum()) / union
