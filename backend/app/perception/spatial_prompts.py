"""Build SAM box and point prompts from known locations.

The normal pipeline and targeted recovery both build their prompts here. Every
token is serialized by `render_spatial_prompt`. A seed that cannot be placed on
one frame of the submitted media fails with a reason and is never moved to fit.

SAM ids name a physical instance only inside one response. A prompt says which
region to follow. It does not promise that later frames still show that object.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field

from app.perception.sam_fusion import SamRequestPlan, plan_prompts
from app.providers.sam import MAX_PROMPTED_OBJECTS, SpatialMark, VideoChunk, render_spatial_prompt
from app.schemas.analysis import AnalysisDocument

PROMPT_STRATEGIES = ("concept", "trusted_spatial", "hybrid")
# The documented billing floor for one video request.
MIN_BILLED_FRAMES = 50
# The source clock's definition of one picture.
SAME_FRAME_SEC = 0.02
_SPATIAL_SESSION = re.compile(r"^(?:c\d+p\d+|rec[0-9a-f]+r\d+):")


class SeedMappingError(ValueError):
    """A seed could not be placed on a frame of the submitted media."""


@dataclass
class MediaMap:
    """The file SAM receives. Frame `k` in it shows source time `frame_times[k]`."""

    width: int
    height: int
    frame_times: list[float]
    time_basis: str = "presentation"
    transform: dict = field(default_factory=dict)

    def frame_for(self, timestamp: float, tolerance: float | None = None) -> int:
        """The frame showing `timestamp`. A time between frames beyond the tolerance is refused."""
        if self.time_basis != "presentation":
            raise SeedMappingError(
                "Frame times are estimated from a nominal rate, so a seed cannot be placed on an exact frame."
            )
        times = self.frame_times
        if not times:
            raise SeedMappingError("The submitted media has no known frame times.")
        index = bisect.bisect_left(times, timestamp)
        options = [item for item in (index - 1, index) if 0 <= item < len(times)]
        nearest = min(options, key=lambda item: abs(times[item] - timestamp))
        allowed = tolerance if tolerance is not None else self._default_tolerance(nearest)
        gap = abs(times[nearest] - timestamp)
        if gap > allowed + 1e-6:
            raise SeedMappingError(
                f"The seed at {timestamp:.3f}s is {gap:.3f}s from the nearest submitted frame at "
                f"{times[nearest]:.3f}s. The limit is {allowed:.3f}s."
            )
        return nearest

    def _default_tolerance(self, index: int) -> float:
        """A stored time names the frame it was measured on. Half a frame is the most rounding can explain."""
        times = self.frame_times
        steps = [right - left for left, right in zip(times[max(index - 1, 0) : index + 2], times[max(index - 1, 0) + 1 : index + 2], strict=False)]
        half = max(min(steps, default=2 * SAME_FRAME_SEC) / 2, 1e-3)
        return min(half, SAME_FRAME_SEC)


def is_spatial_track(track_ref: str | None) -> bool:
    """True for a track grown from a location prompt. Its agreement with that location proves nothing."""
    return bool(track_ref and _SPATIAL_SESSION.match(track_ref))


def box_mark(bbox: dict, frame_index: int, object_id: int, width: int, height: int) -> SpatialMark:
    """A normalized box as inclusive pixel corners in the submitted media.

    A pixel is included when the box reaches into it, so `x2` is the last column
    covered and a box that touches the right edge ends at `width - 1`.
    """
    if width <= 0 or height <= 0:
        raise SeedMappingError("The media width and height are unknown.")
    x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
    if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > 1 + 1e-6 or y + h > 1 + 1e-6:
        raise SeedMappingError(f"The seed box ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}) lies outside the frame.")
    x1 = min(int(math.floor(x * width + 1e-9)), width - 1)
    y1 = min(int(math.floor(y * height + 1e-9)), height - 1)
    x2 = min(max(int(math.ceil((x + w) * width - 1e-9)) - 1, x1), width - 1)
    y2 = min(max(int(math.ceil((y + h) * height - 1e-9)) - 1, y1), height - 1)
    return SpatialMark(
        frame_index=frame_index,
        object_id=object_id,
        kind="box",
        width=width,
        height=height,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def point_mark(point: dict, frame_index: int, object_id: int, width: int, height: int) -> SpatialMark:
    x, y = float(point["x"]), float(point["y"])
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise SeedMappingError(f"The seed point ({x:.3f}, {y:.3f}) lies outside the frame.")
    return SpatialMark(
        frame_index=frame_index,
        object_id=object_id,
        kind="point",
        width=width,
        height=height,
        x=min(int(math.floor(x * width)), width - 1),
        y=min(int(math.floor(y * height)), height - 1),
        negative=bool(point.get("negative")),
    )


def seed_marks(seed, media: MediaMap, object_id: int, tolerance: float | None = None) -> list[SpatialMark]:
    """Every token one seed contributes. Negative points alone do not name an object."""
    frame = media.frame_for(seed.timestamp, tolerance)
    marks: list[SpatialMark] = []
    if seed.bbox:
        marks.append(box_mark(seed.bbox, frame, object_id, media.width, media.height))
    for point in seed.points:
        marks.append(point_mark(point, frame, object_id, media.width, media.height))
    if not any(mark.kind == "box" or not mark.negative for mark in marks):
        raise SeedMappingError("A seed needs a box or at least one positive point.")
    return marks


def choose_direction(seed_time: float, start: float, end: float) -> str:
    """Track toward the interval from the seed."""
    if seed_time <= start:
        return "forward"
    if seed_time >= end:
        return "backward"
    return "both"


def billed_frames(earliest_frame: int, frame_count: int, direction: str) -> int:
    """Frames one request tracks. Tracking runs from the earliest prompt frame. Fifty is the floor."""
    if direction == "forward":
        tracked = frame_count - earliest_frame
    elif direction == "backward":
        tracked = earliest_frame + 1
    else:
        tracked = frame_count
    return max(tracked, MIN_BILLED_FRAMES)


def build_spatial_plan(
    seeded: list[tuple[int, object]],
    media: MediaMap,
    direction: str,
    session_id: str,
    tolerance: float | None = None,
) -> SamRequestPlan:
    """One request for (SAM object id, seed) pairs. Distinct instances keep distinct ids."""
    marks: list[SpatialMark] = []
    bindings: dict[int, str] = {}
    placed: dict[tuple[int, int], list[SpatialMark]] = {}
    ids: dict[str, int] = {}
    for object_id, seed in seeded:
        owner = bindings.setdefault(object_id, seed.object_local_id)
        if owner != seed.object_local_id:
            raise SeedMappingError(f"SAM object {object_id} cannot stand for both {owner} and {seed.object_local_id}.")
        if ids.setdefault(seed.object_local_id, object_id) != object_id:
            raise SeedMappingError(f"{seed.object_local_id} cannot use two SAM ids in one request.")
        own = seed_marks(seed, media, object_id, tolerance)
        frame_key = (own[0].frame_index, object_id)
        if frame_key in placed:
            if placed[frame_key] == own:
                continue
            raise SeedMappingError(f"{seed.object_local_id} has two different prompts on frame {frame_key[0]}.")
        placed[frame_key] = own
        marks.extend(own)
    if len(bindings) > MAX_PROMPTED_OBJECTS:
        raise SeedMappingError(f"{len(bindings)} objects exceed the limit of {MAX_PROMPTED_OBJECTS} in one request.")
    frames = {mark.frame_index for mark in marks}
    if direction == "backward" and min(frames) == 0:
        raise SeedMappingError("Backward tracking needs a prompt after the first submitted frame.")
    # Tracking runs from the earliest prompt frame, so a later prompt could not start a backward pass.
    if direction == "backward" and len(frames) > 1:
        raise SeedMappingError(
            "Backward tracking starts at the earliest prompt, so a later prompt would not reach the frames before it. "
            "Send the later seed in its own request."
        )
    marks.sort(key=lambda mark: (mark.frame_index, mark.object_id))
    return SamRequestPlan(
        kind="spatial",
        prompt=render_spatial_prompt(marks),
        session_id=session_id,
        bindings=bindings,
        propagation=direction,
        labels=[],
        targets=list(bindings.values()),
    )


@dataclass
class MultiSeedPolicy:
    max_prompts_per_object: int = 3
    max_prompts_per_request: int = 8
    # Prompts closer than this show nearly the same picture.
    min_gap_sec: float = 1.0
    # A wider gap between usable seeds may hold an occlusion, so both sides are worth prompting.
    coverage_gap_sec: float = 3.0
    # An area change past this ratio, or this much aspect change, is a new view of the part.
    appearance_change: float = 1.5


def _appearance_changed(seed, others: list, policy: MultiSeedPolicy) -> bool:
    if not seed.bbox:
        return False
    for other in others:
        if not other.bbox:
            continue
        area = (seed.bbox["w"] * seed.bbox["h"]) / max(other.bbox["w"] * other.bbox["h"], 1e-9)
        aspect = (seed.bbox["w"] / max(seed.bbox["h"], 1e-9)) / max(other.bbox["w"] / max(other.bbox["h"], 1e-9), 1e-9)
        if max(area, 1 / max(area, 1e-9), aspect, 1 / max(aspect, 1e-9)) >= policy.appearance_change:
            return True
    return False


def select_prompt_seeds(seeds: list, policy: MultiSeedPolicy | None = None) -> list:
    """Trusted seeds of one object that each add coverage or a new view, in time order.

    Only trusted seeds qualify, and a prompted SAM track cannot vouch for itself. A person's
    seed comes first, then seeds on the far side of the widest uncovered time. Confidence is unused.
    """
    policy = policy or MultiSeedPolicy()
    usable = sorted(
        (seed for seed in seeds if seed.automatic and (seed.source != "sam_spatial" or seed.manual)),
        key=lambda seed: (seed.trust != "human_verified", seed.timestamp, seed.seed_id),
    )
    if not usable:
        return []
    chosen = [usable[0]]
    while len(chosen) < policy.max_prompts_per_object:
        best = None
        best_score = -1.0
        for seed in usable:
            gap = min(abs(seed.timestamp - item.timestamp) for item in chosen)
            if gap < policy.min_gap_sec:
                continue
            changed = _appearance_changed(seed, chosen, policy)
            if gap < policy.coverage_gap_sec and not changed:
                continue
            score = gap + (policy.coverage_gap_sec if changed else 0.0)
            if score > best_score:
                best, best_score = seed, score
        if best is None:
            break
        chosen.append(best)
    return sorted(chosen, key=lambda seed: seed.timestamp)


def plan_for_strategy(
    document: AnalysisDocument,
    chunk: VideoChunk,
    width: int,
    height: int,
    fps: float,
    strategy: str,
    seeds: list,
    *,
    multi_seed: bool = False,
    multi_policy: MultiSeedPolicy | None = None,
) -> tuple[list[SamRequestPlan], list[dict]]:
    """Requests for one chunk under one prompting strategy, with a record of each object's outcome.

    `concept` is the existing behavior. `trusted_spatial` follows only seeds that
    land on this chunk. `hybrid` asks by noun for the objects left over. Concept
    and spatial prompts stay in separate requests, and an object gets one or the other.
    `multi_seed` lets one object be prompted on several frames under one SAM id.
    """
    if strategy == "concept":
        return plan_prompts(document, chunk, width, height, fps), []
    media = MediaMap(
        width=width,
        height=height,
        frame_times=list(chunk.frame_times),
        time_basis="presentation" if chunk.frame_times else "container",
    )
    multi_policy = multi_policy or MultiSeedPolicy()
    outcomes: list[dict] = []
    placeable: dict[str, list] = {}
    for seed in seeds:
        if not seed.automatic or not chunk.time_start - 1e-6 <= seed.timestamp < chunk.time_end + 1e-6:
            continue
        if not multi_seed and seed.object_local_id in placeable:
            continue
        try:
            seed_marks(seed, media, 1)
        except SeedMappingError as exc:
            outcomes.append(_outcome(seed.object_local_id, chunk, "seed_rejected", str(exc)))
            continue
        placeable.setdefault(seed.object_local_id, []).append(seed)
    plans: list[SamRequestPlan] = []
    if placeable:
        pairs: list[tuple[int, object]] = []
        remaining = multi_policy.max_prompts_per_request if multi_seed else MAX_PROMPTED_OBJECTS
        for object_id, local_id in enumerate(list(placeable)[:MAX_PROMPTED_OBJECTS], start=1):
            selected = select_prompt_seeds(placeable[local_id], multi_policy) if multi_seed else placeable[local_id]
            # The first prompt of every object goes in before any object gets a second.
            pairs.extend((object_id, seed) for seed in selected[:1])
        remaining -= len(pairs)
        for object_id, local_id in enumerate(list(placeable)[:MAX_PROMPTED_OBJECTS], start=1):
            if not multi_seed or remaining <= 0:
                break
            extra = select_prompt_seeds(placeable[local_id], multi_policy)[1:][:remaining]
            pairs.extend((object_id, seed) for seed in extra)
            remaining -= len(extra)
        plans.append(build_spatial_plan(pairs, media, "both", f"{chunk.session_id}p0"))
        for _object_id, seed in pairs:
            outcomes.append(_outcome(seed.object_local_id, chunk, "spatial", seed.explanation, trust=seed.trust))
    spatial_targets = set(plans[0].targets) if plans else set()
    leftover = [obj for obj in document.objects if obj.local_id not in spatial_targets]
    if strategy == "hybrid" and leftover:
        rest = document.model_copy(update={"objects": leftover})
        plans.extend(plan_prompts(rest, chunk, width, height, fps))
        outcomes.extend(_outcome(obj.local_id, chunk, "concept", "No usable seed on this chunk.") for obj in leftover)
    elif strategy == "trusted_spatial":
        outcomes.extend(
            _outcome(obj.local_id, chunk, "unobserved", "No usable seed on this chunk, and noun prompts are off.")
            for obj in leftover
        )
    return plans, outcomes


def _outcome(local_id: str, chunk: VideoChunk, prompt: str, reason: str, trust: str = "") -> dict:
    return {
        "object_local_id": local_id,
        "chunk": chunk.index,
        "prompt": prompt,
        "reason": reason,
        **({"trust": trust} if trust else {}),
    }
