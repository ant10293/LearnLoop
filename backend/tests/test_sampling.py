"""Sampling covers the whole requested interval. These cases are synthetic signals, not real video."""

import numpy as np
from app.perception.stream import plan_tracking
from app.video.sampling import (
    MotionMeter,
    MotionSignal,
    budget_indices,
    interval_targets,
    select_frames,
)
from app.video.timeline import build_timeline, coverage_gaps, requested_interval


def _signals(duration: int, busy: set[int], step: float = 1.0) -> list[MotionSignal]:
    count = int(duration / step) + 1
    return [
        MotionSignal(
            timestamp=round(index * step, 3),
            motion=8.0 if round(index * step) in busy else 0.5,
            scene_change=False,
            global_motion=0.0,
        )
        for index in range(count)
    ]


def test_activity_after_twenty_seconds_is_sampled_and_the_end_is_kept():
    signals = _signals(40, busy={25, 26, 27})
    picked = select_frames(signals, 0.0, 40.0, budget=10)
    times = [frame.timestamp for frame in picked.frames]
    assert len(times) == 10
    assert any(25 <= time <= 27 for time in times)
    assert times[0] == 0.0 and times[-1] == 40.0
    largest, _gaps = coverage_gaps(times, 0.0, 40.0, limit=0)
    # The floor is budget // 2 + 1 evenly spaced frames, so no gap exceeds two floor steps.
    assert largest <= 2 * 40.0 / (10 // 2)


def test_a_busy_opening_cannot_use_the_whole_budget():
    signals = _signals(60, busy=set(range(0, 15)))
    times = [frame.timestamp for frame in select_frames(signals, 0.0, 60.0, budget=12).frames]
    assert max(times) == 60.0
    assert sum(1 for time in times if time > 30) >= 3


def test_a_truncated_tail_gets_its_own_closing_sample():
    targets = interval_targets(0.0, 21.952, 1.0)
    assert targets[-1] == 21.952
    assert targets[-2] == 21.0


def test_an_interval_grid_starts_at_the_interval():
    assert interval_targets(12.5, 15.0, 1.0)[:3] == [12.5, 13.5, 14.5]


def test_a_budget_cap_keeps_the_first_and_last_index():
    chosen = budget_indices(200, 7)
    assert chosen[0] == 0 and chosen[-1] == 199
    assert len(chosen) == 7


def test_the_requested_interval_is_clamped_and_snapped_to_real_frames():
    start, end = requested_interval({"time_start": 2, "time_end": 999}, 10.0)
    assert (start, end) == (2.0, 10.0)
    timeline = build_timeline([0.5 + index / 30 for index in range(301)], start, end)
    assert timeline.pts[0] == 0.0
    assert timeline.notes
    snapped = timeline.snap_many([2.0, 2.01, 5.0, 10.0])
    assert snapped[0] == 2.0
    assert len(snapped) == 3
    assert all(start <= time <= end for time in snapped)


def test_capped_tracking_keeps_a_floor_across_the_interval():
    signals = _signals(120, busy={40, 90}, step=0.5)
    plan = plan_tracking(signals, 0.0, 120.0, peak_fps=8.0, maximum_frames=400, occlusion_seconds=1.0)
    assert plan.floor_fps == 2.0
    assert plan.continuity == "supported"
    assert plan.planned_frames <= 400
    assert any(left <= 40 <= right for left, right in plan.dense)
    assert plan.fps_at(110.0) == 2.0


def test_a_budget_too_small_for_the_occlusion_window_is_reported():
    plan = plan_tracking(_signals(300, busy=set()), 0.0, 300.0, peak_fps=8.0, maximum_frames=100, occlusion_seconds=1.0)
    assert plan.continuity == "insufficient"
    assert plan.floor_fps < 1.0
    assert "occlusion window" in plan.notes[0]


def _scene(seed: int) -> np.ndarray:
    """Dark gradient views for odd seeds and bright textured views for even ones, so histograms differ."""
    rng = np.random.default_rng(seed)
    if seed % 2:
        image = np.tile(np.linspace(10, 110, 160, dtype=np.float32), (90, 1))
        for _ in range(6):
            x, y = int(rng.integers(0, 140)), int(rng.integers(0, 70))
            image[y : y + 20, x : x + 20] = float(rng.integers(20, 120))
        return image.astype(np.uint8)
    return rng.integers(150, 255, size=(90, 160), dtype=np.uint8)


def _boundaries(images: list[np.ndarray]) -> list[str]:
    meter = MotionMeter(window=3)
    out = []
    for index, image in enumerate(images):
        out.extend(signal.boundary for _payload, signal in meter.push(float(index), image))
    out.extend(signal.boundary for _payload, signal in meter.flush())
    return out


def test_a_sustained_new_view_is_a_hard_cut():
    first, second = _scene(1), _scene(2)
    boundaries = _boundaries([first] * 4 + [second] * 6)
    assert boundaries.count("hard") == 1
    assert boundaries[4] == "hard"


def test_a_brief_occlusion_is_soft():
    view, hand = _scene(1), _scene(8)
    boundaries = _boundaries([view] * 4 + [hand] + [view] * 5)
    assert "hard" not in boundaries
    assert boundaries[4] == "soft"


def test_a_lighting_change_is_soft():
    view = _scene(1)
    brighter = np.clip(view.astype(np.int16) + 70, 0, 255).astype(np.uint8)
    boundaries = _boundaries([view] * 4 + [brighter] * 6)
    assert "hard" not in boundaries
