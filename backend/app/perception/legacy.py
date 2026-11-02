"""Motion-blob baseline with scene history retained.

Within a scene, association stays greedy intersection-over-union. Across a
scene change the previous tracks are archived instead of deleted, so a later
histogram comparison can see them. A similar histogram does not replace the
new track id.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.perception.assignment import iou
from app.perception.detect import MotionDetector
from app.perception.types import DetectionContext, FrameView, TrackPoint
from app.schemas.enums import TrackStatus


def run_legacy(frames: list[FrameView], research_mode: str) -> tuple[list[TrackPoint], bool, list[dict]]:
    compensate_camera = research_mode in {"constrained_head_mounted", "unconstrained_head_mounted"}
    detector = MotionDetector(compensate_camera)
    active: list[dict] = []
    archived: list[dict] = []
    points: list[TrackPoint] = []
    chunk = 0
    next_id = 1
    previous = None
    chunks = []
    if frames:
        chunks.append(
            {
                "ordinal": 0,
                "time_start": frames[0].timestamp,
                "time_end": frames[0].timestamp,
                "scene_change": False,
            }
        )
    for frame in frames:
        if frame.scene_change and frame.index > 0:
            chunk += 1
            archived.extend(active)
            active = []
            chunks.append(
                {
                    "ordinal": chunk,
                    "time_start": frame.timestamp,
                    "time_end": frame.timestamp,
                    "scene_change": True,
                    "global_motion": frame.global_motion,
                }
            )
        if chunks:
            chunks[-1]["time_end"] = frame.timestamp
        detections = detector.detect(
            frame.image,
            DetectionContext(
                timestamp=frame.timestamp,
                index=frame.index,
                scene_chunk=chunk,
                research_mode=research_mode,
                previous_bgr=previous,
            ),
        )
        used: set[int] = set()
        fresh: list[dict] = []
        for detection in detections:
            signature = _signature(frame.image, detection.bbox)
            best_i = -1
            best_score = 0.15
            for index, track in enumerate(active):
                if index in used:
                    continue
                score = iou(track["box"], detection.bbox)
                if score > best_score:
                    best_score = score
                    best_i = index
            if best_i >= 0:
                used.add(best_i)
                track = active[best_i]
                track["box"] = detection.bbox
                track["sig"] = signature
                continuity = "continuous"
                note = ""
                status = TrackStatus.TRACKED.value
            else:
                track = {
                    "ref": f"track_{next_id}",
                    "box": detection.bbox,
                    "sig": signature,
                    "chunk": chunk,
                    "status": TrackStatus.TENTATIVE.value,
                }
                next_id += 1
                active.append(track)
                fresh.append(track)
                continuity = "new"
                note = "New image-space track. Semantic identity is not established."
                status = TrackStatus.TENTATIVE.value
            points.append(
                TrackPoint(
                    track_ref=track["ref"],
                    timestamp=frame.timestamp,
                    x=detection.bbox[0],
                    y=detection.bbox[1],
                    w=detection.bbox[2],
                    h=detection.bbox[3],
                    motion_score=frame.motion_score,
                    continuity=continuity,
                    camera_compensated=detector.camera_compensated,
                    scene_chunk=chunk,
                    note=note,
                    status=status,
                    association_score=best_score if best_i >= 0 else None,
                    detector_id="motion",
                    visibility="visible",
                )
            )
        if research_mode == "unconstrained_head_mounted" and chunk > 0:
            reassociate_archived(points, fresh, archived, chunk)
        for index, track in enumerate(active):
            if index not in used and track not in fresh:
                track["status"] = TrackStatus.OCCLUDED.value
        previous = frame.image
    archived.extend(active)
    return points, detector.camera_compensated, chunks


def reassociate_archived(points: list[TrackPoint], fresh: list[dict], archived: list[dict], chunk: int) -> None:
    """Annotate a possible link. The new track id is left in place."""
    for track in fresh:
        best = None
        best_corr = -1.0
        for candidate in archived:
            corr = float(
                cv2.compareHist(
                    track["sig"].astype(np.float32),
                    candidate["sig"].astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
            )
            if corr > best_corr:
                best_corr = corr
                best = candidate
        if best is None:
            continue
        for point in points:
            if point.track_ref != track["ref"] or point.scene_chunk != chunk or point.continuity != "new":
                continue
            if best_corr >= 0.92:
                point.continuity = "reassociated"
                point.status = TrackStatus.TRACKED.value
                point.note = (
                    f"Appearance similarity {best_corr:.2f} with earlier track {best['ref']}. "
                    "The intervening interval was not observed, so this is a cautious reassociation, "
                    "not proof of continuous identity. The new track id was kept."
                )
            else:
                point.continuity = "uncertain"
                point.status = TrackStatus.AMBIGUOUS.value
                point.note = (
                    "A visually similar region existed in an earlier scene, but the evidence is too "
                    "weak to treat it as the same object. Identity was not carried across the gap."
                )
            break


def _signature(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x = max(0, int(box[0] * width))
    y = max(0, int(box[1] * height))
    w = max(1, int(box[2] * width))
    h = max(1, int(box[3] * height))
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return np.zeros(24, dtype=np.float32)
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
