"""One source clock for every stage.

Stored times are presentation seconds measured from the first video frame.
That is the clock `ffmpeg -ss` seeks on and the clock browser playback shows.
A subclip or chunk file starts at its own zero; its offset converts it back.
Boxes are normalized to the displayed frame, after rotation, origin top-left.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from app.video.probe import VideoError, presentation_timestamps

# Two timestamps this close are one picture. A 60 fps frame is about 0.017 s apart.
SAME_FRAME_SEC = 0.02


@dataclass
class SourceTimeline:
    """Presentation times of the source video and the requested interval.

    `basis` is ``presentation`` when ffprobe returned frame times and
    ``container`` when only a nominal frame rate was known. In the second case
    frame times are estimates and coverage says so.
    """

    pts: list[float]
    start: float
    end: float
    basis: str = "presentation"
    origin: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def span(self) -> float:
        return max(self.end - self.start, 0.0)

    def inside(self) -> list[float]:
        lo = bisect.bisect_left(self.pts, self.start - 1e-6)
        hi = bisect.bisect_right(self.pts, self.end + 1e-6)
        return self.pts[lo:hi]

    def snap(self, target: float) -> float:
        """Nearest real frame time inside the requested interval."""
        window = self.inside()
        if not window:
            return round(min(max(target, self.start), self.end), 6)
        index = bisect.bisect_left(window, target)
        options = [window[i] for i in (index - 1, index) if 0 <= i < len(window)]
        return round(min(options, key=lambda item: abs(item - target)), 6)

    def snap_many(self, targets: list[float]) -> list[float]:
        """Snap and drop targets that land on the same source frame. Order is by time."""
        seen: set[float] = set()
        kept: list[float] = []
        for target in sorted(targets):
            snapped = self.snap(target)
            key = round(snapped, 3)
            if key in seen:
                continue
            if kept and abs(snapped - kept[-1]) < SAME_FRAME_SEC:
                continue
            seen.add(key)
            kept.append(snapped)
        return kept

    def frame_after(self, chunk_start: float, index: int) -> float | None:
        """Source time of the `index`-th frame at or after `chunk_start`.

        A cut file that keeps every source frame maps its frame `index` here.
        Returns None when the index runs past the known frames.
        """
        if self.basis != "presentation" or index < 0:
            return None
        first = bisect.bisect_left(self.pts, chunk_start - 1e-6)
        position = first + index
        if position >= len(self.pts):
            return None
        return round(self.pts[position], 6)


def requested_interval(config: dict | None, duration: float) -> tuple[float, float]:
    """Clamp `time_start` and `time_end` to the recording."""
    config = config or {}
    total = max(float(duration or 0), 0.0)
    start = float(config.get("time_start") or 0)
    raw_end = config.get("time_end")
    end = float(raw_end) if raw_end not in (None, "") else total
    if total > 0:
        start = min(max(start, 0.0), total)
        end = min(max(end, 0.0), total)
    if end <= start:
        end = total if total > start else start
    return start, end


def build_timeline(raw_pts: list[float], start: float, end: float, *, fps: float = 0.0) -> SourceTimeline:
    """Shift raw presentation times so the first frame is zero, then attach the interval."""
    if raw_pts:
        ordered = sorted(raw_pts)
        origin = ordered[0]
        pts = [round(item - origin, 6) for item in ordered]
        notes = []
        if abs(origin) > 1e-3:
            notes.append(f"The first video frame is stamped {origin:.3f}s. Times here are measured from that frame.")
        return SourceTimeline(pts=pts, start=start, end=end, basis="presentation", origin=origin, notes=notes)
    if fps > 0:
        count = int(end * fps) + 2
        pts = [round(index / fps, 6) for index in range(count)]
        return SourceTimeline(
            pts=pts,
            start=start,
            end=end,
            basis="container",
            notes=["Frame times were not readable. They are estimated from the nominal frame rate."],
        )
    return SourceTimeline(pts=[], start=start, end=end, basis="container", notes=["No frame times are known."])


def load_timeline(video, config: dict | None, cache) -> SourceTimeline:
    """Source timeline for a stored video and a run's requested interval."""
    from app.services.storage import resolve_stored

    start, end = requested_interval(config, float(video.duration_sec or 0))
    fps = float(video.avg_fps or video.fps or 0)
    try:
        raw = presentation_timestamps(resolve_stored(video.stored_path), cache)
    except (VideoError, OSError):
        raw = []
    return build_timeline(raw, start, end, fps=fps)


def coverage_gaps(times: list[float], start: float, end: float, limit: float) -> tuple[float, list[list[float]]]:
    """Largest gap between analyzed times, and every gap longer than `limit`.

    The opening and closing stretches count. A gap is an interval that was not
    examined, not evidence that nothing happened in it.
    """
    points = sorted(time for time in times if start - 1e-6 <= time <= end + 1e-6)
    if not points:
        return max(end - start, 0.0), ([[round(start, 3), round(end, 3)]] if end > start else [])
    edges = [start, *points, end]
    largest = 0.0
    gaps: list[list[float]] = []
    for left, right in zip(edges, edges[1:], strict=False):
        width = right - left
        largest = max(largest, width)
        if limit > 0 and width > limit + 1e-6:
            gaps.append([round(left, 3), round(right, 3)])
    return round(largest, 3), gaps
