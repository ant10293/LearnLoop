"""Tracking frames for the whole requested interval, decoded chunk by chunk.

Temporal coverage (how much of the interval was examined) and temporal
resolution (how densely) are reported separately. When the frame budget does
not fit the configured rate, a floor rate covers everything and the remaining
budget goes to stretches with motion or visual change. A floor too sparse for
the occlusion window is reported as insufficient continuity.

Processing chunks only bound disk and memory. The tracker runs continuously
across them, so a chunk edge never ends a physical identity.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.perception.types import FrameView
from app.video.decode import chunk_bounds, jpeg_pass
from app.video.frames import read_bgr
from app.video.sampling import MotionMeter, MotionSignal, _peak_threshold, window_frames
from app.video.timeline import coverage_gaps

FLOOR_FPS = 2.0
CHUNK_SEC = 30.0
# Dense tracking reaches this far each side of a change in the motion pass.
DENSE_PAD_SEC = 1.0


@dataclass
class TrackingPlan:
    start: float
    end: float
    peak_fps: float
    floor_fps: float
    dense: list[tuple[float, float]] = field(default_factory=list)
    continuity: str = "supported"
    planned_frames: int = 0
    notes: list[str] = field(default_factory=list)

    def fps_at(self, time: float) -> float:
        if self.floor_fps >= self.peak_fps:
            return self.peak_fps
        for left, right in self.dense:
            if left - 1e-6 <= time <= right + 1e-6:
                return self.peak_fps
        return self.floor_fps


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def plan_tracking(
    signals: list[MotionSignal],
    start: float,
    end: float,
    peak_fps: float,
    maximum_frames: int,
    occlusion_seconds: float,
) -> TrackingPlan:
    span = max(end - start, 0.0)
    peak = max(float(peak_fps), 0.1)
    cap = max(int(maximum_frames), 2)
    if span * peak + 1 <= cap:
        return TrackingPlan(start, end, peak, peak, [(start, end)], "supported", int(span * peak) + 1)
    floor = min(FLOOR_FPS, peak)
    notes: list[str] = []
    if span * floor + 1 > cap:
        floor = max((cap - 1) / max(span, 1e-6), 1e-3)
        gap = 1.0 / floor
        continuity = "insufficient" if gap > occlusion_seconds else "supported"
        notes.append(
            f"The frame budget allows {floor:.2f} FPS across {span:.1f}s. A {gap:.2f}s gap is longer than "
            f"the {occlusion_seconds:.2f}s occlusion window, so tracks across it are not continuous evidence."
            if continuity == "insufficient"
            else f"The frame budget allows {floor:.2f} FPS across {span:.1f}s."
        )
        return TrackingPlan(start, end, peak, floor, [], continuity, cap, notes)
    threshold = _peak_threshold(signals)
    ranked = sorted(
        (
            item
            for item in signals[1:]
            if item.boundary != "none" or (not math.isinf(threshold) and item.motion >= threshold)
        ),
        key=lambda item: ({"hard": 2, "soft": 1}.get(item.boundary, 0), item.motion, -item.timestamp),
        reverse=True,
    )
    remaining = cap - (span * floor + 1)
    dense: list[tuple[float, float]] = []
    extra = peak - floor
    for item in ranked:
        window = (max(start, item.timestamp - DENSE_PAD_SEC), min(end, item.timestamp + DENSE_PAD_SEC))
        trial = _merge([*dense, window])
        cost = sum(right - left for left, right in trial) * extra
        if cost > remaining:
            continue
        dense = trial
    continuity = "supported" if 1.0 / floor <= occlusion_seconds else "insufficient"
    if dense:
        notes.append(f"{len(dense)} stretches with change were tracked at {peak:g} FPS. The rest ran at {floor:g} FPS.")
    else:
        notes.append(f"No stretch with change fit the budget at {peak:g} FPS. The interval ran at {floor:g} FPS.")
    planned = int(span * floor + sum(right - left for left, right in dense) * extra) + 1
    return TrackingPlan(start, end, peak, floor, dense, continuity, planned, notes)


@dataclass
class StreamStats:
    frames: int = 0
    times: list[float] = field(default_factory=list)
    chunks: int = 0
    signals: list[MotionSignal] = field(default_factory=list)

    def summary(self, plan: TrackingPlan, occlusion_seconds: float) -> dict:
        largest, gaps = coverage_gaps(self.times, plan.start, plan.end, occlusion_seconds)
        return {
            "frame_source": "tracking_stream",
            "frames": self.frames,
            "span": [round(plan.start, 3), round(plan.end, 3)],
            "first_time": self.times[0] if self.times else None,
            "last_time": self.times[-1] if self.times else None,
            "peak_fps": round(plan.peak_fps, 3),
            "floor_fps": round(plan.floor_fps, 3),
            "dense_spans": [[round(a, 3), round(b, 3)] for a, b in plan.dense],
            "max_gap_sec": largest,
            "gaps_over_occlusion": gaps,
            "continuity": plan.continuity if not gaps or plan.continuity == "insufficient" else "partial",
            "processing_chunks": self.chunks,
            "time_basis": "presentation",
            "notes": plan.notes,
        }


def first_chunk(source: Path, plan: TrackingPlan, work: Path) -> list[tuple[float, Path]]:
    """Decode the opening chunk before tracking starts, so a decode failure is reported up front."""
    left, right = next(chunk_bounds(plan.start, plan.end, CHUNK_SEC))
    return _decode(source, plan, left, right, work / "chunk_0000")


def _decode(source: Path, plan: TrackingPlan, left: float, right: float, dest: Path) -> list[tuple[float, Path]]:
    densest = max(plan.fps_at(left), plan.fps_at(right), *(plan.peak_fps for a, b in plan.dense if a <= right and b >= left))
    return jpeg_pass(source, left, right, 1.0 / densest, dest)


def stream_views(
    source: Path,
    plan: TrackingPlan,
    work: Path,
    stats: StreamStats,
    opening: list[tuple[float, Path]] | None = None,
) -> Iterator[FrameView]:
    """Frames on the source clock, with motion and boundary signals, one chunk on disk at a time."""
    meter = MotionMeter(window_frames([0.0, 1.0 / plan.peak_fps]))
    last_kept: float | None = None
    index = 0

    def emit(pairs) -> Iterator[FrameView]:
        nonlocal index
        for (timestamp, image), signal in pairs:
            stats.signals.append(signal)
            stats.times.append(timestamp)
            stats.frames += 1
            view = FrameView(
                timestamp=timestamp,
                image=image,
                scene_change=signal.scene_change,
                global_motion=signal.global_motion,
                motion_score=signal.motion,
                index=index,
            )
            index += 1
            yield view

    try:
        for number, (left, right) in enumerate(chunk_bounds(plan.start, plan.end, CHUNK_SEC)):
            folder = work / f"chunk_{number:04d}"
            rows = opening if number == 0 and opening is not None else _decode(source, plan, left, right, folder)
            stats.chunks += 1
            for timestamp, path in rows:
                if last_kept is not None and timestamp <= last_kept + 1e-6:
                    continue
                spacing = 1.0 / plan.fps_at(timestamp)
                if last_kept is not None and timestamp - last_kept < spacing * 0.95:
                    continue
                last_kept = timestamp
                image = read_bgr(path)
                yield from emit(meter.push(timestamp, image, (timestamp, image)))
            shutil.rmtree(folder, ignore_errors=True)
        yield from emit(meter.flush())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def scale_global_motion(signals: list[MotionSignal], step: float, reference_step: float) -> list[MotionSignal]:
    """Express motion-pass flow per tracking frame so mode thresholds keep their meaning."""
    if step <= 0 or reference_step <= 0:
        return signals
    factor = reference_step / step
    return [
        MotionSignal(
            timestamp=item.timestamp,
            motion=item.motion,
            scene_change=item.scene_change,
            global_motion=float(np.float64(item.global_motion) * factor),
            boundary=item.boundary,
            boundary_reason=item.boundary_reason,
        )
        for item in signals
    ]
