"""Track lifecycle helpers. Scores here are uncalibrated association values."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.perception.assignment import (
    center,
    iou,
    linear_assignment,
    mask_iou,
    motion_agreement,
    prediction_score,
    shift_box,
    size_score,
)
from app.perception.encoder import cached_encode
from app.perception.memory import gallery_cosine, store_reference
from app.perception.presets import PerceptionConfig
from app.perception.segment import crop_box, grabcut_mask
from app.perception.types import TrackPoint
from app.schemas.analysis import AppearanceRefRecord, BBox, HypothesisRecord, InstanceRecord
from app.schemas.enums import TrackStatus


@dataclass
class LiveTrack:
    instance_id: str
    status: str
    box: tuple[float, float, float, float]
    vx: float = 0.0
    vy: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
    scene_chunk: int = 0
    human_initialized: bool = False
    detector_id: str = ""
    uncertain: bool = False
    trusted: list = field(default_factory=list)
    trusted_meta: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    candidate_meta: list = field(default_factory=list)
    last_mask: np.ndarray | None = None


def _age(tracks: list[LiveTrack], timestamp: float, config: PerceptionConfig) -> None:
    for track in tracks:
        if track.status == TrackStatus.ARCHIVED.value:
            continue
        gap = timestamp - track.last_seen
        if gap > config.archive_seconds:
            track.status = TrackStatus.ARCHIVED.value
        elif gap > config.occlusion_seconds:
            track.status = TrackStatus.LOST.value


def _short_term(track: LiveTrack, timestamp: float, config: PerceptionConfig) -> bool:
    if track.status not in {
        TrackStatus.TENTATIVE.value,
        TrackStatus.TRACKED.value,
        TrackStatus.OCCLUDED.value,
        TrackStatus.AMBIGUOUS.value,
    }:
        return False
    return timestamp - track.last_seen <= config.occlusion_seconds


def _short_score(track, detection, timestamp, config: PerceptionConfig, camera_scale: float):
    dt = max(timestamp - track.last_seen, 1e-3)
    predicted = shift_box(track.box, track.vx * dt, track.vy * dt)
    predicted_score = prediction_score(predicted, detection.bbox) * camera_scale
    overlap = max(iou(predicted, detection.bbox), iou(track.box, detection.bbox)) * camera_scale
    parts = {
        "prediction": config.weights.prediction * predicted_score,
        "iou": config.weights.iou * overlap,
        "size": config.weights.size * size_score(track.box, detection.bbox),
        "motion": config.weights.motion
        * (1.0 if track.hits < 2 else motion_agreement(track.box, detection.bbox, track.vx, track.vy, dt)),
        "time": config.weights.time * math.exp(-dt / 0.8),
    }
    mask = mask_iou(track.last_mask, detection.mask)
    if mask is not None:
        parts["mask"] = config.weights.mask * mask
    total = sum(config.weights.__dict__[key] for key in parts)
    score = sum(parts.values()) / total if total else 0.0
    geom = max(predicted_score, overlap)
    if geom < 0.12:
        score = 0.0
    return score, {"second": 0.0, "prediction": predicted_score, "cosine": None, "geom": geom}


def _assign(tracks, detections, score_fn, min_score: float):
    if not tracks or not detections:
        return []
    scores = np.zeros((len(tracks), len(detections)), dtype=float)
    extras = {}
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            score, extra = score_fn(track, detection)
            scores[i, j] = score
            extras[(i, j)] = extra
    cost = np.where(scores >= min_score, 1.0 - scores, 1e3)
    chosen = []
    for i, j in linear_assignment(cost):
        if scores[i, j] < min_score:
            continue
        column = np.sort(scores[:, j])[::-1]
        second = float(column[1]) if len(column) > 1 else 0.0
        extras[(i, j)]["second"] = second
        chosen.append((i, j, float(scores[i, j]), float(scores[i, j] - second), extras[(i, j)]))
    return chosen


def _reid(reid_tracks, unmatched, detections, vectors, frame, config, hypotheses):
    if not reid_tracks or not unmatched:
        return []
    scores = np.full((len(reid_tracks), len(unmatched)), -1.0)
    cosines = np.full_like(scores, np.nan)
    motions = np.zeros_like(scores)
    for i, track in enumerate(reid_tracks):
        for j, det_index in enumerate(unmatched):
            cosine_value = gallery_cosine(track.trusted, vectors[j])
            if cosine_value is None:
                continue
            cosines[i, j] = cosine_value
            motions[i, j] = prediction_score(track.box, detections[det_index].bbox)
            if cosine_value >= config.reid_min_cosine:
                scores[i, j] = 0.78 * cosine_value + 0.22 * motions[i, j]
    taken = []
    used_tracks = set()
    order = [(scores[i, j], i, j) for i in range(scores.shape[0]) for j in range(scores.shape[1]) if scores[i, j] >= 0]
    order.sort(reverse=True)
    claimed = set()
    for score, i, j in order:
        if i in used_tracks or j in claimed:
            continue
        column = [scores[row, j] for row in range(scores.shape[0]) if scores[row, j] >= 0]
        column.sort(reverse=True)
        second = column[1] if len(column) > 1 else 0.0
        margin = float(score - second)
        cosine_value = None if math.isnan(cosines[i, j]) else float(cosines[i, j])
        runner = None
        if len(column) > 1:
            for row in range(scores.shape[0]):
                if row != i and scores[row, j] == second:
                    runner = reid_tracks[row].instance_id
                    break
        decision = "proposed" if margin >= config.reid_min_margin else "ambiguous"
        if decision == "ambiguous":
            hypotheses.append(
                _hypothesis(
                    len(hypotheses) + 1,
                    frame.timestamp,
                    reid_tracks[i].instance_id,
                    reid_tracks[i].instance_id,
                    {"cosine": cosine_value, "prediction": float(motions[i, j])},
                    float(score),
                    margin,
                    "ambiguous",
                    "Appearance was similar to more than one instance. The new observation kept its own id.",
                    runner,
                )
            )
            claimed.add(j)
            continue
        used_tracks.add(i)
        claimed.add(j)
        hypotheses.append(
            _hypothesis(
                len(hypotheses) + 1,
                frame.timestamp,
                reid_tracks[i].instance_id,
                reid_tracks[i].instance_id,
                {"cosine": cosine_value, "prediction": float(motions[i, j])},
                float(score),
                margin,
                "proposed",
                "The id was reused from appearance and was not marked confirmed.",
                runner,
            )
        )
        taken.append((unmatched[j], reid_tracks[i], float(score), margin, cosine_value, float(motions[i, j])))
    for j, det_index in enumerate(unmatched):
        if j in claimed:
            continue
        best_i = int(np.nanargmax(cosines[:, j])) if cosines.shape[0] and not np.all(np.isnan(cosines[:, j])) else None
        if best_i is None or math.isnan(cosines[best_i, j]):
            continue
        hypotheses.append(
            _hypothesis(
                len(hypotheses) + 1,
                frame.timestamp,
                reid_tracks[best_i].instance_id,
                reid_tracks[best_i].instance_id,
                {"cosine": float(cosines[best_i, j]), "prediction": float(motions[best_i, j])},
                float(cosines[best_i, j]),
                None,
                "rejected",
                "The best appearance score did not pass the re-identification gate. A new id was kept.",
            )
        )
    return taken


def _update_track(track, detection, frame, score, continuity, chunk) -> None:
    dt = max(frame.timestamp - track.last_seen, 1e-3)
    cx, cy = center(detection.bbox)
    px, py = center(track.box)
    track.vx = 0.5 * ((cx - px) / dt) + 0.5 * track.vx
    track.vy = 0.5 * ((cy - py) / dt) + 0.5 * track.vy
    track.box = detection.bbox
    track.last_seen = frame.timestamp
    track.scene_chunk = chunk
    if detection.detector_id != "optical_flow":
        # A flow hold moves the box. It is not a detection and does not confirm the track.
        track.hits += 1
        track.detector_id = detection.detector_id or track.detector_id
    if detection.mask is not None:
        track.last_mask = detection.mask
    if detection.human:
        track.human_initialized = True
        track.status = TrackStatus.TRACKED.value
    elif track.hits >= 2 and track.status in {TrackStatus.TENTATIVE.value, TrackStatus.OCCLUDED.value, TrackStatus.LOST.value, TrackStatus.ARCHIVED.value}:
        track.status = TrackStatus.TRACKED.value
    elif continuity == "reassociated":
        track.status = TrackStatus.TRACKED.value
    track._last_continuity = continuity
    track._last_score = score


def _store(track, vector, frame, detection, state, config, encoder) -> None:
    if vector is None or encoder is None:
        return
    meta = {
        "timestamp": frame.timestamp,
        "quality": _sharpness(crop_box(frame.image, detection.bbox, detection.mask)),
        "state": state,
        "encoder_id": encoder.encoder_id,
        "encoder_version": encoder.encoder_version,
        "preprocess_version": encoder.preprocess_version,
        "bbox": detection.bbox,
        "confirmed": bool(detection.human),
    }
    if state == "trusted":
        store_reference(track.trusted, track.trusted_meta, vector, meta, config.max_trusted_references)
    else:
        store_reference(track.candidates, track.candidate_meta, vector, meta, config.max_candidate_references)


def _new_track(instance_id, detection, frame, chunk) -> LiveTrack:
    return LiveTrack(
        instance_id=instance_id,
        status=TrackStatus.TRACKED.value if detection.human else TrackStatus.TENTATIVE.value,
        box=detection.bbox,
        first_seen=frame.timestamp,
        last_seen=frame.timestamp,
        hits=1,
        scene_chunk=chunk,
        human_initialized=detection.human,
        detector_id=detection.detector_id,
        uncertain=not detection.human,
    )


def _mark_missed(track, timestamp, config) -> None:
    gap = timestamp - track.last_seen
    if gap > config.archive_seconds:
        track.status = TrackStatus.ARCHIVED.value
    elif gap > config.occlusion_seconds:
        track.status = TrackStatus.LOST.value
    else:
        track.status = TrackStatus.OCCLUDED.value


def _point(track, frame, detection, score, continuity, chunk, cosine_value=None, motion=None, note=""):
    visibility = {
        TrackStatus.OCCLUDED.value: "occluded",
        TrackStatus.LOST.value: "absent",
        TrackStatus.ARCHIVED.value: "absent",
        TrackStatus.AMBIGUOUS.value: "partial",
    }.get(track.status, "visible")
    if not note:
        note = "Tracked in the image frame. This is not a semantic class and not an action."
        if cosine_value is not None:
            note += f" Appearance cosine {cosine_value:.2f} is not a probability."
    return TrackPoint(
        track_ref=track.instance_id,
        timestamp=frame.timestamp,
        x=detection.bbox[0],
        y=detection.bbox[1],
        w=detection.bbox[2],
        h=detection.bbox[3],
        motion_score=frame.motion_score,
        continuity=continuity,
        camera_compensated=False,
        scene_chunk=chunk,
        note=note,
        status=track.status,
        association_score=score,
        detector_id=detection.detector_id,
        visibility=visibility,
    )


def _hypothesis(number, timestamp, subject, candidate, extra, score, margin, decision, note, runner=None):
    return HypothesisRecord(
        hypothesis_id=f"hyp_{number}",
        timestamp=timestamp,
        subject_instance_id=subject,
        candidate_instance_id=candidate,
        runner_up_instance_id=runner,
        appearance_cosine=extra.get("cosine"),
        motion_agreement=extra.get("prediction"),
        association_score=float(score),
        margin=margin,
        decision=decision,
        note=note,
    )


def _instance(track: LiveTrack, points: list[TrackPoint]) -> InstanceRecord:
    count = sum(1 for point in points if point.track_ref == track.instance_id)
    uncertain = track.uncertain or track.status in {TrackStatus.TENTATIVE.value, TrackStatus.AMBIGUOUS.value, TrackStatus.LOST.value}
    return InstanceRecord(
        instance_id=track.instance_id,
        semantic_label=None,
        status=track.status,
        first_seen=track.first_seen,
        last_seen=track.last_seen,
        observation_count=count,
        trusted_references=len(track.trusted),
        candidate_references=len(track.candidates),
        uncertain=uncertain,
        human_initialized=track.human_initialized,
        detector_id=track.detector_id,
    )


def _refs(track: LiveTrack) -> list[AppearanceRefRecord]:
    rows = []
    for meta in track.trusted_meta:
        rows.append(_ref(track.instance_id, meta, "trusted"))
    for meta in track.candidate_meta:
        rows.append(_ref(track.instance_id, meta, "candidate"))
    return rows


def _ref(instance_id, meta, state) -> AppearanceRefRecord:
    box = meta.get("bbox")
    return AppearanceRefRecord(
        instance_id=instance_id,
        timestamp=float(meta.get("timestamp") or 0),
        state=state,
        quality=float(meta.get("quality") or 0),
        encoder_id=meta.get("encoder_id") or "",
        encoder_version=meta.get("encoder_version") or "",
        preprocess_version=meta.get("preprocess_version") or "",
        bbox=BBox(x=box[0], y=box[1], w=min(box[2], 1 - box[0]), h=min(box[3], 1 - box[1])) if box else None,
        confirmed=bool(meta.get("confirmed")),
    )


def _segment_detections(image, detections) -> int:
    written = 0
    for detection in detections[:6]:
        x, y, w, h = detection.bbox
        if w <= 0 or h <= 0 or x + w > 1.001 or y + h > 1.001:
            continue
        mask = grabcut_mask(image, BBox(x=x, y=y, w=min(w, 1 - x), h=min(h, 1 - y)))
        if mask is not None:
            detection.mask = mask
            written += 1
    return written


def _encode(encoder, image, detections, cache_dir):
    started = time.perf_counter()
    crops = [crop_box(image, detection.bbox, detection.mask) for detection in detections]
    masks = [detection.mask for detection in detections]
    vectors = cached_encode(encoder, crops, masks, cache_dir)
    return vectors, time.perf_counter() - started


def _sharpness(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
