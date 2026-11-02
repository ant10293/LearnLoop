"""Candidate regions. A detection does not require a semantic class."""

from __future__ import annotations

import cv2
import numpy as np

from app.perception.assignment import iou
from app.perception.camera import compensate
from app.perception.types import Detection, DetectionContext, DetectorUnavailable

# What each detector's hits are evidence of. Motion and MSER regions are
# candidate regions; the resolver never makes one canonical geometry. A human
# box is a detection. Flow holds are not detections at all. A detector that is
# not listed here is treated as a proposal source.
DETECTOR_EVIDENCE = {
    "motion": "proposal",
    "proposals": "proposal",
    "grabcut": "proposal",
    "manual": "detected",
    "human": "detected",
    "sam-3.1": "mask",
    "optical_flow": "none",
}


def _blobs(mask: np.ndarray, limit: int = 8) -> list[tuple[int, int, int, int]]:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 80 or w < 4 or h < 4:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda item: item[2] * item[3], reverse=True)
    return boxes[:limit]


def _normalize(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (
        min(max(x / width, 0.0), 0.98),
        min(max(y / height, 0.0), 0.98),
        min(max(w / width, 0.01), 1.0),
        min(max(h / height, 0.01), 1.0),
    )


def non_maximum(boxes: list[tuple[float, float, float, float]], threshold: float = 0.45, limit: int = 8):
    ordered = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
    kept: list[tuple[float, float, float, float]] = []
    for box in ordered:
        if any(iou(box, other) >= threshold for other in kept):
            continue
        kept.append(box)
        if len(kept) >= limit:
            break
    return kept


class MotionDetector:
    """Foreground motion. A blob is a candidate region, not a named object."""

    detector_id = "motion"

    def __init__(self, compensate_camera: bool):
        self.compensate_camera = compensate_camera
        self.subtractor = None
        if not compensate_camera:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=20, varThreshold=16, detectShadows=False
            )
        self._compensated = False

    @property
    def camera_compensated(self) -> bool:
        return self._compensated

    def detect(self, frame: np.ndarray, context: DetectionContext) -> list[Detection]:
        height, width = frame.shape[:2]
        if context.previous_bgr is None:
            return []
        if self.compensate_camera:
            reference, ok = compensate(context.previous_bgr, frame)
            self._compensated = self._compensated or ok
            diff = cv2.absdiff(frame, reference if ok else context.previous_bgr)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _threshold, mask = cv2.threshold(gray, 28, 255, cv2.THRESH_BINARY)
        else:
            assert self.subtractor is not None
            foreground = self.subtractor.apply(frame)
            _threshold, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return [
            Detection(
                timestamp=context.timestamp,
                bbox=_normalize(box, width, height),
                score=None,
                detector_id=self.detector_id,
            )
            for box in _blobs(mask)
        ]


# Phone video is often 4K. MSER walks the whole image, and a few hundred
# full-resolution frames will occupy the process until it stops answering.
PROPOSAL_MAX_EDGE = 960


def _mser_view(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a view whose long side fits the proposal budget, and the scale used."""
    height, width = gray.shape[:2]
    edge = max(height, width)
    if edge <= PROPOSAL_MAX_EDGE or edge <= 0:
        return gray, 1.0
    scale = PROPOSAL_MAX_EDGE / edge
    resized = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


class ProposalDetector:
    """Motion regions plus stable extremal regions. Still not a class label."""

    detector_id = "proposals"

    def __init__(self, compensate_camera: bool):
        self.motion = MotionDetector(compensate_camera)
        self.mser = cv2.MSER_create()

    @property
    def camera_compensated(self) -> bool:
        return self.motion.camera_compensated

    def detect(self, frame: np.ndarray, context: DetectionContext) -> list[Detection]:
        height, width = frame.shape[:2]
        boxes = [item.bbox for item in self.motion.detect(frame, context)]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        view, scale = _mser_view(gray)
        view_height, view_width = view.shape[:2]
        if scale == 1.0:
            self.mser.setMinArea(60)
            self.mser.setMaxArea(14400)
        else:
            view_area = view_width * view_height
            self.mser.setMinArea(max(60, int(view_area * 0.002)))
            self.mser.setMaxArea(max(14400, int(view_area * 0.35)))
        regions, _bboxes = self.mser.detectRegions(view)
        area_min = width * height * 0.002
        area_max = width * height * 0.35
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            if scale != 1.0:
                x = int(round(x / scale))
                y = int(round(y / scale))
                w = int(round(w / scale))
                h = int(round(h / scale))
            if not area_min <= w * h <= area_max:
                continue
            if w < 6 or h < 6:
                continue
            boxes.append(_normalize((x, y, w, h), width, height))
        kept = non_maximum(boxes, threshold=0.5, limit=8)
        return [
            Detection(timestamp=context.timestamp, bbox=box, score=None, detector_id=self.detector_id)
            for box in kept
        ]


class ManualDetector:
    """Only the regions a person marked. Nothing else is invented."""

    detector_id = "manual"

    def detect(self, frame: np.ndarray, context: DetectionContext) -> list[Detection]:
        return []


class SeededDetector:
    """Adds explicit seeds on top of a detector. Seeds are not a silent fallback."""

    def __init__(self, inner, seeds: list[dict]):
        self.inner = inner
        self.seeds = seeds
        self.used: set[int] = set()

    @property
    def detector_id(self) -> str:
        return getattr(self.inner, "detector_id", "manual")

    @property
    def camera_compensated(self) -> bool:
        return bool(getattr(self.inner, "camera_compensated", False))

    def detect(self, frame: np.ndarray, context: DetectionContext) -> list[Detection]:
        found = list(self.inner.detect(frame, context))
        for index, seed in enumerate(self.seeds):
            if index in self.used:
                continue
            if abs(float(seed.get("timestamp") or 0) - context.timestamp) > 0.25:
                continue
            box = (
                float(seed["x"]),
                float(seed["y"]),
                float(seed["w"]),
                float(seed["h"]),
            )
            found.append(
                Detection(
                    timestamp=context.timestamp,
                    bbox=box,
                    score=1.0,
                    detector_id="manual",
                    human=True,
                )
            )
            self.used.add(index)
        return found


def build_detector(name: str, compensate_camera: bool, seeds: list[dict] | None = None):
    if name == "motion":
        inner = MotionDetector(compensate_camera)
    elif name == "proposals":
        inner = ProposalDetector(compensate_camera)
    elif name == "manual":
        inner = ManualDetector()
    else:
        raise DetectorUnavailable(
            f"Detector '{name}' is not available. No other detector was substituted."
        )
    if seeds:
        return SeededDetector(inner, seeds)
    return inner
