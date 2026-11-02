"""Decide which observed boxes may start a SAM track.

A label, a model confidence, or a box that keeps repeating does not make a box
trustworthy. A person can vouch for one. Otherwise a measurement has to agree
with a producer that found the object another way. Every check is kept on the
seed so the decision can be read later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.perception.assignment import iou
from app.perception.resolve import (
    AGREEMENT_IOU,
    SAM_ID,
    SAME_FRAME_SEC,
    TRACKER_SOURCE,
    tracker_rows,
)
from app.perception.spatial_prompts import is_spatial_track
from app.schemas.analysis import AnalysisDocument, BoxDecision, ObservationPrediction
from app.schemas.enums import Provenance, Visibility

HUMAN_VERIFIED = "human_verified"
INDEPENDENTLY_SUPPORTED = "independently_supported"
CANDIDATE = "candidate"
REJECTED = "rejected"
AUTOMATIC_TRUST = (HUMAN_VERIFIED, INDEPENDENTLY_SUPPORTED)
_TRUST_ORDER = {HUMAN_VERIFIED: 0, INDEPENDENTLY_SUPPORTED: 1, CANDIDATE: 2, REJECTED: 3}
_UNMEASURED = "vlm_video_unmeasured"


@dataclass
class SeedPolicy:
    # A box narrower than this cannot hold a visible part at phone resolution.
    min_side: float = 0.004
    # A box over most of the frame is the scene, not one part.
    max_area: float = 0.8
    # Reuses the resolver's definition of two measurements agreeing.
    agree_iou: float = AGREEMENT_IOU
    # Half of the box shared with another object means it may be on the wrong one.
    neighbor_iou: float = 0.5
    occlusion_limit: float = 0.5
    # Matches the SAM fusion window around a hard cut.
    boundary_window: float = 0.45
    same_frame_sec: float = SAME_FRAME_SEC
    # Seeds within this many seconds of each other count as equally near.
    distance_band: float = 1.0


@dataclass
class Seed:
    object_local_id: str
    timestamp: float
    bbox: dict | None
    trust: str
    source: str
    instance_id: str | None = None
    source_ref: str = ""
    points: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    explanation: str = ""
    confidence: float | None = None
    sharpness: float | None = None
    occlusion: float | None = None
    manual: bool = False

    @property
    def seed_id(self) -> str:
        box = self.bbox or {}
        text = f"{self.object_local_id}|{self.timestamp:.3f}|{self.source}|{sorted(box.items())}|{self.points}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @property
    def automatic(self) -> bool:
        return self.trust in AUTOMATIC_TRUST

    def as_dict(self) -> dict:
        return {
            "seed_id": self.seed_id,
            "object_local_id": self.object_local_id,
            "instance_id": self.instance_id,
            "timestamp": self.timestamp,
            "bbox": self.bbox,
            "points": self.points,
            "trust": self.trust,
            "source": self.source,
            "source_ref": self.source_ref,
            "checks": self.checks,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "sharpness": self.sharpness,
            "occlusion": self.occlusion,
            "manual": self.manual,
        }


def row_family(row: ObservationPrediction) -> str:
    """Which producer made this box. Two rows of one family are one opinion."""
    if row.geometry_authority == "human" or row.evidence.provenance in {
        Provenance.HUMAN_ANNOTATION,
        Provenance.HUMAN_CORRECTION,
    }:
        return "human"
    if row.geometry_origin == "optical_flow":
        return "flow"
    if row.source_model == SAM_ID:
        return "sam_spatial" if is_spatial_track(row.track_ref) else "sam"
    if row.source_model == TRACKER_SOURCE or (not row.source_model and row.track_ref and row.geometry_origin == "detected"):
        return TRACKER_SOURCE
    return "vlm"


def _measured(row: ObservationPrediction) -> bool:
    family = row_family(row)
    if family == "sam":
        return (row.association_support or row.association_status) == "associated"
    if family == TRACKER_SOURCE:
        return (
            row.association_support == "associated"
            and row.detection_support in {"detected", "mask"}
            and row.temporal_support != "single"
        )
    return False


def _tuple(box) -> tuple[float, float, float, float]:
    return (box.x, box.y, box.w, box.h)


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": passed, "detail": detail}


def _pool(document: AnalysisDocument, spatial: AnalysisDocument | None) -> list[ObservationPrediction]:
    rows = [*document.observations, *document.raw_observations]
    if spatial is not None:
        rows = [row for row in rows if row_family(row) != TRACKER_SOURCE] + tracker_rows(document, spatial)
    seen: set[tuple] = set()
    unique: list[ObservationPrediction] = []
    for row in rows:
        if row.bbox is None:
            continue
        key = (row.object_local_id, round(row.timestamp, 3), row.source_model, row.track_ref, _tuple(row.bbox))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _flagged(decisions: list[BoxDecision], row: ObservationPrediction, same_frame: float) -> BoxDecision | None:
    for item in decisions:
        if item.outcome == "kept" or item.bbox is None or item.timestamp is None:
            continue
        if item.object_local_id == row.object_local_id and abs(item.timestamp - row.timestamp) <= same_frame:
            if iou(_tuple(item.bbox), _tuple(row.bbox)) >= 0.9:
                return item
    return None


def classify_row(
    row: ObservationPrediction,
    pool: list[ObservationPrediction],
    *,
    decisions: list[BoxDecision] | None = None,
    boundaries: list[float] | None = None,
    policy: SeedPolicy | None = None,
) -> tuple[str, list[dict], str]:
    """Trust level, checks, and a one-sentence reason for one stored box."""
    policy = policy or SeedPolicy()
    family = row_family(row)
    box = row.bbox
    checks = [
        _check("box_size", box.w >= policy.min_side and box.h >= policy.min_side, f"{box.w:.3f} by {box.h:.3f} of the frame"),
        _check("box_extent", box.w * box.h <= policy.max_area, f"covers {box.w * box.h:.2f} of the frame"),
        _check(
            "visible",
            row.visibility == Visibility.VISIBLE,
            f"stored as {row.visibility.value}",
        ),
        _check(
            "unoccluded",
            row.occlusion is None or row.occlusion < policy.occlusion_limit,
            "occlusion not measured" if row.occlusion is None else f"occlusion {row.occlusion:.2f}",
        ),
        _check("measured_on_a_frame", row.geometry_authority != _UNMEASURED, "a video-file box names no exact frame"
               if row.geometry_authority == _UNMEASURED else "tied to a timestamp"),
        _check("not_flow", family != "flow", "a flow hold is not a detection"),
        _check("not_conflicting", row.association_status not in {"conflict", "ambiguous"}, row.association_status or "no conflict recorded"),
    ]
    near_cut = next((item for item in boundaries or [] if abs(item - row.timestamp) <= policy.boundary_window), None)
    checks.append(_check("away_from_cut", near_cut is None, "no hard cut nearby" if near_cut is None else f"hard cut at {near_cut:.2f}s"))
    decision = _flagged(decisions or [], row, policy.same_frame_sec)
    checks.append(
        _check(
            "not_removed",
            decision is None,
            "no pipeline step removed this box" if decision is None else f"{decision.step} recorded {decision.reason_code}",
        )
    )
    frame_rows = [item for item in pool if abs(item.timestamp - row.timestamp) <= policy.same_frame_sec and item is not row]
    neighbor = next(
        (
            item
            for item in frame_rows
            if item.object_local_id != row.object_local_id and iou(_tuple(item.bbox), _tuple(box)) >= policy.neighbor_iou
        ),
        None,
    )
    checks.append(
        _check(
            "clear_of_neighbors",
            neighbor is None,
            "no other object shares this box" if neighbor is None else f"overlaps {neighbor.object_local_id}",
        )
    )
    failed = [item for item in checks if not item["passed"]]
    if failed:
        names = ", ".join(humanize(item["check"]) for item in failed)
        return REJECTED, checks, f"Not usable as a seed: {names}."
    if family == "human":
        return HUMAN_VERIFIED, checks, "A person drew or corrected this box."
    if family == "sam_spatial":
        checks.append(_check("independent_support", False, "grown from a location prompt, so it cannot vouch for itself"))
        return CANDIDATE, checks, "This box came from an earlier prompted track. Confirm it to use it."
    supporters = [
        item
        for item in frame_rows
        if item.object_local_id == row.object_local_id
        and row_family(item) not in {family, "flow", "sam_spatial"}
        and iou(_tuple(item.bbox), _tuple(box)) >= policy.agree_iou
        and (_measured(row) or _measured(item))
    ]
    if supporters:
        names = sorted({row_family(item) for item in supporters})
        checks.append(_check("independent_support", True, f"agrees with {', '.join(names)} on the same frame"))
        return INDEPENDENTLY_SUPPORTED, checks, f"The {family} box agrees with {' and '.join(names)} on the same frame."
    checks.append(_check("independent_support", False, "no other producer agrees on this frame"))
    return CANDIDATE, checks, f"Only the {family} producer places the object here. Confirm it to use it."


def humanize(name: str) -> str:
    return name.replace("_", " ")


def collect_seeds(
    document: AnalysisDocument,
    spatial: AnalysisDocument | None = None,
    *,
    manual: list[dict] | None = None,
    confirmed: set[str] | None = None,
    boundaries: list[float] | None = None,
    policy: SeedPolicy | None = None,
) -> list[Seed]:
    """Every stored box as a seed, plus manual ones. Order is deterministic."""
    policy = policy or SeedPolicy()
    instances = {obj.local_id: obj.instance_id for obj in document.objects}
    known = set(instances)
    pool = _pool(document, spatial)
    seeds: list[Seed] = []
    for row in pool:
        if row.object_local_id not in known:
            continue
        trust, checks, reason = classify_row(
            row, pool, decisions=document.box_decisions, boundaries=boundaries, policy=policy
        )
        seed = Seed(
            object_local_id=row.object_local_id,
            timestamp=row.timestamp,
            bbox=row.bbox.model_dump(),
            trust=trust,
            source=row_family(row),
            instance_id=instances.get(row.object_local_id),
            source_ref=f"{row.source_model or row.geometry_authority or 'model'}:{row.track_ref or ''}@{row.timestamp:.3f}",
            checks=checks,
            explanation=reason,
            confidence=row.confidence,
            sharpness=row.sharpness,
            occlusion=row.occlusion,
        )
        if confirmed and seed.seed_id in confirmed and trust != REJECTED:
            seed.trust = HUMAN_VERIFIED
            seed.explanation = f"A person confirmed this {seed.source} box. {reason}"
        seeds.append(seed)
    for item in manual or []:
        seeds.append(manual_seed(item, instances.get(str(item.get("object_local_id")))))
    seeds.sort(key=lambda seed: (seed.object_local_id, _TRUST_ORDER[seed.trust], seed.timestamp, seed.seed_id))
    return seeds


def manual_seed(item: dict, instance_id: str | None = None) -> Seed:
    return Seed(
        object_local_id=str(item["object_local_id"]),
        timestamp=float(item["timestamp"]),
        bbox=dict(item["bbox"]) if item.get("bbox") else None,
        points=[dict(point) for point in item.get("points") or []],
        trust=HUMAN_VERIFIED,
        source="human",
        instance_id=instance_id,
        source_ref="manual",
        checks=[_check("human_entered", True, "drawn or entered by the researcher")],
        explanation="A person placed this seed.",
        manual=True,
    )


def seed_from_dict(item: dict) -> Seed:
    return Seed(
        object_local_id=item["object_local_id"],
        timestamp=float(item["timestamp"]),
        bbox=item.get("bbox"),
        trust=item["trust"],
        source=item["source"],
        instance_id=item.get("instance_id"),
        source_ref=item.get("source_ref", ""),
        points=list(item.get("points") or []),
        checks=list(item.get("checks") or []),
        explanation=item.get("explanation", ""),
        confidence=item.get("confidence"),
        sharpness=item.get("sharpness"),
        occlusion=item.get("occlusion"),
        manual=bool(item.get("manual")),
    )


def _distance(seed: Seed, start: float, end: float) -> float:
    if seed.timestamp < start:
        return start - seed.timestamp
    if seed.timestamp > end:
        return seed.timestamp - end
    return 0.0


def rank_for_interval(seeds: list[Seed], start: float, end: float, policy: SeedPolicy | None = None) -> list[Seed]:
    """Nearest usable seeds first. Within a band of distance, trust and then visibility decide.

    The appearance nearest the interval is the one SAM can follow, so a far
    stronger seed does not beat a near one by trust alone. Model confidence is not used.
    """
    policy = policy or SeedPolicy()

    def key(seed: Seed) -> tuple:
        distance = _distance(seed, start, end)
        return (
            int(distance // policy.distance_band),
            _TRUST_ORDER[seed.trust],
            seed.occlusion if seed.occlusion is not None else 0.25,
            -(seed.sharpness or 0.0),
            distance,
            seed.seed_id,
        )

    return sorted((seed for seed in seeds if seed.automatic), key=key)


def seeds_around(seeds: list[Seed], local_id: str, start: float, end: float, policy: SeedPolicy | None = None) -> dict:
    """The best usable seed on each side of an interval. Either side may be empty, and that is reported."""
    owned = [seed for seed in seeds if seed.object_local_id == local_id]
    usable = rank_for_interval(owned, start, end, policy)
    before = next((seed for seed in usable if seed.timestamp <= start), None)
    after = next((seed for seed in usable if seed.timestamp >= end), None)
    inside = next((seed for seed in usable if start < seed.timestamp < end), None)
    best = next(iter(usable), None)
    return {"before": before, "after": after, "inside": inside, "best": best, "usable": usable}
