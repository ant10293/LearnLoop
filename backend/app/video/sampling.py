"""Deterministic frame sampling on the source clock, including budgeted adaptive selection."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.schemas.enums import PRESET_FPS, SamplingPreset
from app.video.frames import read_bgr, sharpness_score
from app.video.probe import snap_timestamps


@dataclass
class SampledFrame:
    timestamp: float
    reason: str
    motion_score: float = 0
    sharpness: float = 0
    scene_change: bool = False


@dataclass
class MotionSignal:
    """Change between this frame and the previous analyzed frame.

    `boundary` is ``hard`` only for a sustained appearance break that a
    lighting change, a brief occlusion, or unstable motion does not explain.
    ``soft`` marks a change that was seen and not confirmed as a cut.
    `scene_change` is true only for ``hard``.
    """

    timestamp: float
    motion: float
    scene_change: bool
    global_motion: float
    boundary: str = "none"
    boundary_reason: str = ""


def uniform_targets(duration: float, fps: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    if fps <= 0:
        return [0.0]
    step = 1.0 / fps
    times: list[float] = []
    cursor = 0.0
    while cursor < duration - 1e-3:
        times.append(round(cursor, 3))
        cursor += step
    if not times:
        times.append(0.0)
    return times


def preset_fps(preset: str, custom_fps: float | None) -> float:
    if preset == SamplingPreset.CUSTOM.value:
        if not custom_fps or custom_fps <= 0:
            raise ValueError("custom sampling requires a positive fps")
        return float(custom_fps)
    if preset == SamplingPreset.ADAPTIVE.value:
        return 1.0
    return PRESET_FPS.get(preset, 1.0)


def _closing_time(last: float, end: float) -> float | None:
    """Return the end of the range when the grid leaves a real tail unseen.

    A 1 fps grid on a 21.952 s clip otherwise stops at 21 s. The final state
    is usually in that partial second, so it gets its own sample. A tail
    shorter than about one frame is left to the nearest grid point.
    """
    if end - last <= 0.04:
        return None
    closing = math.floor((end + 1e-9) * 1000) / 1000
    if closing - last <= 0.04:
        return None
    return closing


def plan_uniform(duration: float, fps: float, pts: list[float]) -> list[SampledFrame]:
    targets = uniform_targets(duration, fps)
    if targets:
        closing = _closing_time(targets[-1], duration)
        if closing is not None:
            targets = [*targets, closing]
    snapped = snap_timestamps(targets, pts)
    return [SampledFrame(timestamp=item, reason="uniform") for item in snapped]


def interval_targets(start: float, end: float, fps: float) -> list[float]:
    """Grid from `start`, not from zero, with a closing sample at `end` when a real tail is left."""
    if end <= start or fps <= 0:
        return [round(start, 3)]
    offsets = uniform_targets(end - start, fps)
    targets = [round(start + item, 3) for item in offsets]
    closing = _closing_time(targets[-1], end)
    if closing is not None:
        targets.append(closing)
    return targets


def plan_interval(timeline, fps: float, reason: str = "uniform") -> list[SampledFrame]:
    """Uniform samples over the timeline's requested interval, snapped to real frames."""
    snapped = timeline.snap_many(interval_targets(timeline.start, timeline.end, fps))
    return [SampledFrame(timestamp=item, reason=reason) for item in snapped]


# Two timestamps this close are one picture. A 60 fps frame is about 0.017 s
# apart, so a 2 ms pair is the same moment stored twice.
SAME_MOMENT_SEC = 0.02


def distinct_moments(frames: list, gap: float = SAME_MOMENT_SEC) -> list:
    """Drop a frame that repeats the previous one within `gap` seconds."""
    ordered = sorted(frames, key=lambda item: float(item.timestamp))
    kept: list = []
    for frame in ordered:
        if kept and float(frame.timestamp) - float(kept[-1].timestamp) < gap:
            continue
        kept.append(frame)
    return kept


def budget_indices(count: int, limit: int) -> list[int]:
    """Pick up to `limit` evenly spaced indexes, always keeping the first and last."""
    if count <= 0:
        return []
    if limit <= 0 or count <= limit:
        return list(range(count))
    if limit == 1:
        return [count - 1]
    step = (count - 1) / (limit - 1)
    chosen: list[int] = []
    seen: set[int] = set()
    for i in range(limit):
        index = min(count - 1, int(round(i * step)))
        if index in seen:
            continue
        seen.add(index)
        chosen.append(index)
    return chosen


# Histogram correlation below this is an appearance break worth checking.
HIST_BREAK = 0.72
# Correlation of mean- and variance-normalized images. A lighting or exposure
# change keeps the structure, so it stays above this. A different view does not.
STRUCTURE_KEPT = 0.5
# A break must last this long, without the earlier view returning, to be a cut.
BOUNDARY_WINDOW_SEC = 1.0
# A frame this flat is a covered or blacked-out lens, not a new scene.
BLANK_STD = 8.0


def _small(image: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] == size[0] and gray.shape[0] == size[1]:
        return gray
    return cv2.resize(gray, size)


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0], None, [32], [0, 256])
    hb = cv2.calcHist([b], [0], None, [32], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _structure(a: np.ndarray, b: np.ndarray) -> float:
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    fa = (fa - fa.mean()) / (fa.std() + 1e-6)
    fb = (fb - fb.mean()) / (fb.std() + 1e-6)
    return float(np.mean(fa * fb))


def _same_view(a: np.ndarray, b: np.ndarray) -> bool:
    return _hist_corr(a, b) >= HIST_BREAK or _structure(a, b) >= STRUCTURE_KEPT


@dataclass
class _Pending:
    timestamp: float
    small: np.ndarray
    motion: float
    global_motion: float
    broken: bool
    lit: bool
    payload: object = None


class MotionMeter:
    """Streaming motion and boundary signals with a short lookahead.

    A break is held until `window` later frames arrive. It becomes ``hard``
    only if the earlier view does not return and the new view holds. A break
    that keeps the image structure is a lighting change. A break the earlier
    view returns from is a transient occlusion. A break followed by more
    breaks is unstable motion. All of those are ``soft`` and do not end tracks.
    """

    def __init__(self, window: int = 4):
        self.window = max(1, int(window))
        self.queue: deque[_Pending] = deque()
        self.previous: np.ndarray | None = None
        self.anchor: np.ndarray | None = None

    def push(self, timestamp: float, image: np.ndarray, payload=None) -> list[tuple[object, MotionSignal]]:
        small = _small(image)
        if self.previous is None:
            item = _Pending(timestamp, small, 0.0, 0.0, False, False, payload)
        else:
            diff = float(np.mean(cv2.absdiff(self.previous, small)))
            broken = _hist_corr(self.previous, small) < HIST_BREAK
            lit = broken and _structure(self.previous, small) >= STRUCTURE_KEPT
            flow = cv2.calcOpticalFlowFarneback(self.previous, small, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            global_motion = float(np.median(np.linalg.norm(flow, axis=2)))
            item = _Pending(timestamp, small, diff, global_motion, broken, lit, payload)
        self.previous = small
        self.queue.append(item)
        out = []
        while len(self.queue) > self.window:
            out.append(self._emit(final=False))
        return out

    def flush(self) -> list[tuple[object, MotionSignal]]:
        out = []
        while self.queue:
            out.append(self._emit(final=True))
        return out

    def _emit(self, final: bool) -> tuple[object, MotionSignal]:
        item = self.queue.popleft()
        boundary, reason = "none", ""
        if item.broken:
            boundary, reason = self._classify(item, list(self.queue), final)
        if boundary == "hard" or self.anchor is None or not item.broken:
            self.anchor = item.small
        signal = MotionSignal(
            timestamp=item.timestamp,
            motion=item.motion,
            scene_change=boundary == "hard",
            global_motion=item.global_motion,
            boundary=boundary,
            boundary_reason=reason,
        )
        return item.payload, signal

    def _classify(self, item: _Pending, after: list[_Pending], final: bool) -> tuple[str, str]:
        if item.lit:
            return "soft", "lighting"
        if float(item.small.std()) < BLANK_STD:
            return "soft", "camera_covered"
        if self.anchor is not None and any(_same_view(self.anchor, later.small) for later in after):
            return "soft", "transient"
        if not after:
            return "soft", "unconfirmed" if final else "transient"
        if any(later.broken and not later.lit for later in after):
            return "soft", "unstable"
        return "hard", "sustained"


def motion_signals(images, timestamps: list[float], window: int | None = None) -> list[MotionSignal]:
    """Signals for an ordered list of images or small grayscale arrays."""
    if window is None:
        window = window_frames(timestamps)
    meter = MotionMeter(window)
    signals: list[MotionSignal] = []
    for image, timestamp in zip(images, timestamps, strict=True):
        signals.extend(signal for _payload, signal in meter.push(timestamp, image))
    signals.extend(signal for _payload, signal in meter.flush())
    return signals


def window_frames(timestamps: list[float], seconds: float = BOUNDARY_WINDOW_SEC) -> int:
    if len(timestamps) < 2:
        return 1
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a]
    step = float(np.median(gaps)) if gaps else 1.0
    return max(1, int(math.ceil(seconds / max(step, 1e-3))))


def measure_motion(paths: list[Path], timestamps: list[float]) -> list[MotionSignal]:
    return motion_signals((read_bgr(path) for path in paths), timestamps)


@dataclass
class Selection:
    frames: list[SampledFrame]
    candidate_peaks: list[float] = field(default_factory=list)
    selected_peaks: list[float] = field(default_factory=list)
    budget_exclusions: int = 0
    boundaries: list[dict] = field(default_factory=list)


def _peak_threshold(signals: list[MotionSignal]) -> float:
    motions = [item.motion for item in signals[1:]] or [0.0]
    median = float(np.median(motions))
    mad = float(np.median([abs(item - median) for item in motions]))
    if mad <= 1e-6:
        return math.inf if max(motions) - median <= 1e-6 else median + 1e-6
    return median + 1.5 * mad


def _rank(signal: MotionSignal, threshold: float) -> tuple[int, float]:
    tier = {"hard": 2, "soft": 1}.get(signal.boundary, 0)
    return tier, signal.motion - (0 if math.isinf(threshold) else threshold)


def _reason(signal: MotionSignal) -> str:
    if signal.boundary == "hard":
        return "scene_change"
    if signal.boundary == "soft":
        return "visual_change"
    return "motion_peak"


def select_frames(signals: list[MotionSignal], start: float, end: float, budget: int) -> Selection:
    """Pick at most `budget` analyzed frames across the whole interval.

    A uniform floor of ``budget // 2 + 1`` frames comes first. It keeps every
    gap at or under twice the spacing the budget allows. Remaining slots go to
    change peaks, one per floor segment per round, so a busy stretch cannot use
    the whole budget. Slots still left fill the widest gaps.
    """
    ordered = sorted(signals, key=lambda item: item.timestamp)
    by_time = {round(item.timestamp, 3): item for item in ordered}
    threshold = _peak_threshold(ordered)
    peaks = [
        item
        for item in ordered[1:]
        if item.boundary != "none" or (not math.isinf(threshold) and item.motion >= threshold)
    ]
    boundaries = [
        {"time": round(item.timestamp, 3), "kind": item.boundary, "reason": item.boundary_reason}
        for item in ordered
        if item.boundary != "none"
    ]
    peak_ids = {id(item) for item in peaks}
    if not ordered:
        return Selection(frames=[], boundaries=boundaries)
    budget = max(1, int(budget))
    if len(ordered) <= budget:
        frames = [_frame(item, "uniform") for item in ordered]
        for frame, item in zip(frames, ordered, strict=True):
            if id(item) in peak_ids:
                frame.reason = _reason(item)
        peak_times = [round(item.timestamp, 3) for item in peaks]
        return Selection(frames, peak_times, peak_times, 0, boundaries)

    times = [item.timestamp for item in ordered]
    floor_n = min(budget, budget // 2 + 1)
    chosen: dict[float, str] = {}
    for index in range(floor_n):
        target = start if floor_n == 1 else start + (end - start) * index / (floor_n - 1)
        nearest = min(ordered, key=lambda item, target=target: (abs(item.timestamp - target), item.timestamp))
        chosen.setdefault(round(nearest.timestamp, 3), "uniform")
    floor_times = sorted(chosen)

    def segment(time: float) -> int:
        for position in range(len(floor_times) - 1):
            if floor_times[position] <= time < floor_times[position + 1]:
                return position
        return max(len(floor_times) - 2, 0)

    queues: dict[int, list[MotionSignal]] = {}
    for item in sorted(peaks, key=lambda item: (_rank(item, threshold), -item.timestamp), reverse=True):
        queues.setdefault(segment(item.timestamp), []).append(item)
    selected_peaks: list[float] = []
    while len(chosen) < budget and any(queues.values()):
        heads = sorted(
            ((items[0], key) for key, items in queues.items() if items),
            key=lambda pair: (_rank(pair[0], threshold), -pair[0].timestamp),
            reverse=True,
        )
        for item, key in heads:
            if len(chosen) >= budget:
                break
            queues[key].pop(0)
            stamp = round(item.timestamp, 3)
            if stamp in chosen:
                if chosen[stamp] == "uniform":
                    chosen[stamp] = _reason(item)
                    selected_peaks.append(stamp)
                continue
            chosen[stamp] = _reason(item)
            selected_peaks.append(stamp)
    while len(chosen) < budget:
        marks = sorted(chosen)
        edges = list(zip(marks, marks[1:], strict=False))
        if not edges:
            break
        left, right = max(edges, key=lambda pair: (pair[1] - pair[0], -pair[0]))
        middle = (left + right) / 2
        inner = [time for time in times if left < time < right]
        if not inner:
            break
        pick = round(min(inner, key=lambda time: (abs(time - middle), time)), 3)
        chosen[pick] = "gap_fill"
    for stamp, reason in chosen.items():
        item = by_time.get(stamp)
        if item is not None and id(item) in peak_ids and reason == "uniform":
            chosen[stamp] = _reason(item)
            selected_peaks.append(stamp)
    frames = []
    for stamp in sorted(chosen):
        item = by_time.get(stamp)
        frames.append(_frame(item, chosen[stamp]) if item else SampledFrame(stamp, chosen[stamp]))
    peak_times = [round(item.timestamp, 3) for item in peaks]
    kept = set(chosen)
    excluded = sum(1 for stamp in peak_times if stamp not in kept)
    return Selection(frames, peak_times, sorted(set(selected_peaks)), excluded, boundaries)


def _frame(signal: MotionSignal, reason: str) -> SampledFrame:
    return SampledFrame(
        timestamp=signal.timestamp,
        reason=reason,
        motion_score=signal.motion,
        scene_change=signal.scene_change,
    )


def annotate_sharpness(frames: list[SampledFrame], paths: dict[float, Path]) -> None:
    for frame in frames:
        path = paths.get(round(frame.timestamp, 3))
        if path and path.exists():
            frame.sharpness = sharpness_score(read_bgr(path))
