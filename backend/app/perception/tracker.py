"""Temporal association. Instance ids are not semantic labels and are not action verdicts."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.perception.camera import flow_box
from app.perception.memory import reference_state
from app.perception.presets import PerceptionConfig
from app.perception.tracker_ops import (
    LiveTrack,
    _age,
    _assign,
    _encode,
    _hypothesis,
    _instance,
    _mark_missed,
    _new_track,
    _point,
    _refs,
    _reid,
    _segment_detections,
    _short_score,
    _short_term,
    _store,
    _update_track,
)
from app.perception.types import (
    Detection,
    DetectionContext,
    EncoderUnavailable,
    FrameView,
    TrackingCancelled,
    TrackPoint,
)
from app.schemas.analysis import AppearanceRefRecord, HypothesisRecord, PerceptionRecord
from app.schemas.enums import TrackStatus


@dataclass
class TrackResult:
    points: list[TrackPoint]
    perception: PerceptionRecord
    compensated: bool


def track_frames(
    frames: list[FrameView],
    *,
    config: PerceptionConfig,
    detector,
    encoder=None,
    research_mode: str,
    cache_dir=None,
    should_stop=None,
) -> TrackResult:
    """Associate detections over time. Weak matches stay unmatched."""
    started = time.perf_counter()
    frame_count = 0
    tracks: list[LiveTrack] = []
    points: list[TrackPoint] = []
    hypotheses: list[HypothesisRecord] = []
    references: list[AppearanceRefRecord] = []
    next_id = 1
    next_hyp = 1
    chunk = 0
    previous = None
    compensated = bool(getattr(detector, "camera_compensated", False))
    detections_n = 0
    ambiguous_n = 0
    reid_proposed = 0
    encoder_error = ""
    encoder_status = "not_requested" if encoder is None else "ready"
    encoder_seconds = 0.0
    segmented = 0
    active_encoder = encoder

    def stopped() -> None:
        if should_stop is not None and should_stop():
            raise TrackingCancelled("Tracking stopped because the job was cancelled.")

    for frame in frames:
        frame_count += 1
        stopped()
        if frame.scene_change and frame.index > 0:
            chunk += 1
            for track in tracks:
                track.status = TrackStatus.ARCHIVED.value
        _age(tracks, frame.timestamp, config)
        context_previous = previous
        found = detector.detect(
            frame.image,
            DetectionContext(
                timestamp=frame.timestamp,
                index=frame.index,
                scene_chunk=chunk,
                research_mode=research_mode,
                previous_bgr=context_previous,
                camera_compensated=compensated,
            ),
        )
        memory_jobs: list[tuple[LiveTrack, Detection, str]] = []
        if config.segment and frame.index % 4 == 0:
            segmented += _segment_detections(frame.image, found)
        detections_n += len(found)
        compensated = compensated or bool(getattr(detector, "camera_compensated", False))
        camera_scale = 0.75 if frame.global_motion > 0.55 and not compensated else 1.0
        short_term = [track for track in tracks if _short_term(track, frame.timestamp, config)]
        matched_det: set[int] = set()
        matched_tracks: set[int] = set()
        assignments = _assign(
            short_term,
            found,
            lambda track, detection: _short_score(track, detection, frame.timestamp, config, camera_scale),
            config.min_score,
        )
        for track_index, det_index, score, margin, extra in assignments:
            track = short_term[track_index]
            detection = found[det_index]
            uncertain = margin < config.min_margin and extra["second"] >= config.min_score
            if uncertain:
                ambiguous_n += 1
                track.uncertain = True
                track.status = TrackStatus.AMBIGUOUS.value
                hypotheses.append(
                    _hypothesis(
                        next_hyp,
                        frame.timestamp,
                        track.instance_id,
                        track.instance_id,
                        extra,
                        score,
                        margin,
                        "ambiguous",
                        "Short-gap scores were too close to treat the match as settled.",
                    )
                )
                next_hyp += 1
            continuity = "uncertain" if uncertain else "continuous"
            _update_track(track, detection, frame, score, continuity, chunk)
            if not uncertain:
                track.uncertain = False
            state = reference_state(
                human=detection.human,
                temporal_strong=not uncertain and not frame.scene_change,
                assigned_reid=False,
            )
            if state:
                memory_jobs.append((track, detection, state))
            matched_det.add(det_index)
            matched_tracks.add(id(track))
            points.append(_point(track, frame, detection, score, "uncertain" if uncertain else "continuous", chunk))

        unmatched = [index for index in range(len(found)) if index not in matched_det]
        if active_encoder is not None and unmatched and any(track.status in {TrackStatus.LOST.value, TrackStatus.ARCHIVED.value} for track in tracks):
            try:
                vectors, spent = _encode(active_encoder, frame.image, [found[index] for index in unmatched], cache_dir)
                encoder_seconds += spent
            except EncoderUnavailable as exc:
                active_encoder = None
                encoder_status = "unavailable"
                encoder_error = str(exc)
                vectors = []
            if len(vectors):
                reid_tracks = [
                    track
                    for track in tracks
                    if track.status in {TrackStatus.LOST.value, TrackStatus.ARCHIVED.value} and track.trusted
                ]
                taken = _reid(
                    reid_tracks,
                    unmatched,
                    found,
                    vectors,
                    frame,
                    config,
                    hypotheses,
                )
                next_hyp = len(hypotheses) + 1
                for det_index, track, score, margin, cosine_value, motion in taken:
                    if margin < config.reid_min_margin:
                        ambiguous_n += 1
                        continue
                    reid_proposed += 1
                    detection = found[det_index]
                    _update_track(track, detection, frame, score, "reassociated", chunk)
                    track.uncertain = True
                    _store(track, vectors[unmatched.index(det_index)], frame, detection, "candidate", config, active_encoder)
                    matched_det.add(det_index)
                    points.append(_point(track, frame, detection, score, "reassociated", chunk, cosine_value, motion))

        for track in short_term:
            if id(track) in matched_tracks:
                continue
            if previous is not None and (track.human_initialized or track.status == TrackStatus.TRACKED.value):
                moved = flow_box(previous, frame.image, track.box)
                if moved is not None:
                    synthetic = Detection(
                        timestamp=frame.timestamp,
                        bbox=moved,
                        detector_id="optical_flow",
                        score=None,
                    )
                    _update_track(track, synthetic, frame, None, "continuous", chunk)
                    points.append(_point(track, frame, synthetic, None, "continuous", chunk))
                    matched_tracks.add(id(track))
                    continue
            _mark_missed(track, frame.timestamp, config)

        for index, detection in enumerate(found):
            if index in matched_det:
                continue
            track = _new_track(f"track_{next_id}", detection, frame, chunk)
            next_id += 1
            tracks.append(track)
            if detection.human:
                memory_jobs.append((track, detection, "trusted"))
            note = "New image-space track. Semantic identity is not established."
            if frame.scene_change and frame.index > 0:
                note = "Scene change opened a new track. Earlier tracks were kept and were not automatically continued."
            points.append(
                _point(track, frame, detection, detection.score, "new", chunk, note=note)
            )
        if active_encoder is not None and memory_jobs:
            try:
                vectors, spent = _encode(
                    active_encoder,
                    frame.image,
                    [detection for _track, detection, _state in memory_jobs],
                    cache_dir,
                )
                encoder_seconds += spent
                for (track, detection, state), vector in zip(memory_jobs, vectors, strict=True):
                    _store(track, vector, frame, detection, state, config, active_encoder)
            except EncoderUnavailable as exc:
                active_encoder = None
                encoder_status = "unavailable"
                encoder_error = str(exc)
        previous = frame.image

    elapsed = time.perf_counter() - started
    fps = (frame_count / elapsed) if elapsed > 0 else 0.0
    instances = [_instance(track, points) for track in tracks]
    for track in tracks:
        references.extend(_refs(track))
    diagnostics = {
        "elapsed_sec": round(elapsed, 4),
        "frames": frame_count,
        "processing_fps": round(fps, 3),
        "detections": detections_n,
        "tracks_created": next_id - 1,
        "active_at_end": sum(1 for track in tracks if track.status in {TrackStatus.TRACKED.value, TrackStatus.OCCLUDED.value, TrackStatus.TENTATIVE.value}),
        "ambiguous_matches": ambiguous_n,
        "reid_proposals": reid_proposed,
        "encoder_elapsed_sec": round(encoder_seconds, 4),
        "frames_segmented": segmented,
    }
    if encoder is None:
        encoder_status = "not_requested"
    elif encoder_status == "ready":
        encoder_status = "ready"
    perception = PerceptionRecord(
        strategy=config.strategy,
        detector_id=getattr(detector, "detector_id", config.detector),
        encoder_id=getattr(encoder, "encoder_id", None) if encoder is not None else None,
        encoder_version=getattr(encoder, "encoder_version", None) if encoder is not None else None,
        encoder_status=encoder_status,
        encoder_detail=encoder_error,
        segmenter_id="grabcut" if config.segment else None,
        tracking_fps=config.target_fps,
        frames_tracked=frame_count,
        world_coordinates_supported=False,
        used_for_identity=config.strategy != "vlm_only",
        scores_are_probabilities=False,
        diagnostics=diagnostics,
        instances=instances,
        hypotheses=hypotheses,
        appearance_references=references,
        notes=(
            "Instance ids are tracking hypotheses in the image frame. "
            "association_score and appearance_cosine are uncalibrated and are not probabilities. "
            "World coordinates were not estimated."
        ),
    )
    return TrackResult(points=points, perception=perception, compensated=compensated)
