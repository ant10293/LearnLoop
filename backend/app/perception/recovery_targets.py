"""Find object and time intervals where localization looks missing or unreliable.

A missing box does not mean a failed detection. The object may be hidden, out of
view, removed, or attached behind another part. A recorded explanation is kept
and that interval is not suggested for recovery. Without evidence either way the
interval stays unresolved. Nothing here asserts absence.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field

from app.perception.assignment import iou
from app.perception.resolve import SAME_FRAME_SEC, TRACKER_SOURCE, is_estimate, tracker_rows
from app.perception.seeds import Seed, row_family, seeds_around
from app.schemas.analysis import AnalysisDocument, ObservationPrediction
from app.schemas.enums import AmbiguityKind, RelationKind, Visibility

_HIDING_RELATIONS = {RelationKind.ATTACHED, RelationKind.INSIDE, RelationKind.OCCLUDED_BY, RelationKind.AROUND}
_TIME_AMBIGUITIES = {
    AmbiguityKind.CONFLICTING_EVIDENCE: "unresolved_conflict",
    AmbiguityKind.UNCERTAIN_IDENTITY: "identity_conflict",
    AmbiguityKind.UNOBSERVED_INTERVAL: "tracking_gap",
    AmbiguityKind.MISSING_EVIDENCE: "tracking_gap",
}


@dataclass
class TargetPolicy:
    # A gap has to be longer than this and longer than the object's usual spacing.
    min_gap_sec: float = 1.0
    spacing_factor: float = 2.5
    # A step is a jump past this many box sizes and this multiple of the object's usual step.
    jump_sizes: float = 2.0
    jump_factor: float = 4.0
    # Area change between neighboring boxes. A turned part can double, so one change this size only suggests a look.
    size_ratio: float = 2.5
    agree_conflict_iou: float = 0.2
    swap_iou: float = 0.5
    edge_margin: float = 0.02
    merge_slack_sec: float = 0.5
    # A stored ambiguity longer than this describes the recording, not an interval to recover.
    max_recorded_sec: float = 8.0
    max_targets: int = 40


@dataclass
class _Finding:
    start: float
    end: float
    reason: str
    detail: str
    refs: list[str] = field(default_factory=list)
    state: str = "questionable"


def _box(row: ObservationPrediction) -> tuple[float, float, float, float]:
    return (row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h)


def _center(row: ObservationPrediction) -> tuple[float, float]:
    return (row.bbox.x + row.bbox.w / 2, row.bbox.y + row.bbox.h / 2)


def _ref(row: ObservationPrediction) -> str:
    return f"{row.object_local_id}@{row.timestamp:.3f}:{row.source_model or row.geometry_authority or 'model'}"


def _rows_by_object(document: AnalysisDocument, spatial: AnalysisDocument | None) -> dict[str, list[ObservationPrediction]]:
    rows = [*document.observations, *document.raw_observations]
    if spatial is not None:
        rows = [row for row in rows if row_family(row) != TRACKER_SOURCE] + tracker_rows(document, spatial)
    grouped: dict[str, list[ObservationPrediction]] = {}
    for row in rows:
        grouped.setdefault(row.object_local_id, []).append(row)
    for items in grouped.values():
        items.sort(key=lambda item: item.timestamp)
    return grouped


def _located(rows: list[ObservationPrediction]) -> list[ObservationPrediction]:
    """One box per moment, preferring canonical rows. Flow holds and conflicts do not locate an object."""
    best: dict[float, ObservationPrediction] = {}
    for row in rows:
        if row.bbox is None or row.visibility == Visibility.ABSENT or row_family(row) == "flow":
            continue
        if row.association_status == "conflict":
            continue
        key = round(row.timestamp, 2)
        current = best.get(key)
        if current is None or (row.observation_role == "canonical" and current.observation_role != "canonical"):
            best[key] = row
    return [best[key] for key in sorted(best)]


def _at_edge(row: ObservationPrediction, margin: float) -> bool:
    box = row.bbox
    return box.x <= margin or box.y <= margin or box.x + box.w >= 1 - margin or box.y + box.h >= 1 - margin


def _gap_findings(document, local_id, rows, located, policy) -> list[_Finding]:
    if len(located) < 2:
        return []
    steps = [right.timestamp - left.timestamp for left, right in zip(located, located[1:], strict=False)]
    limit = max(policy.min_gap_sec, policy.spacing_factor * statistics.median(steps))
    found: list[_Finding] = []
    for left, right in zip(located, located[1:], strict=False):
        if right.timestamp - left.timestamp <= limit:
            continue
        start, end = left.timestamp, right.timestamp
        inside = [row for row in rows if start < row.timestamp < end]
        state, detail = "unresolved", "No reliable box in this stretch, and nothing explains why."
        if any(row.visibility == Visibility.ABSENT for row in inside):
            state, detail = "recorded_absent", "A stored observation says the object was absent here."
        elif any(row.visibility == Visibility.OCCLUDED for row in inside):
            state, detail = "recorded_occluded", "A stored observation says the object was fully occluded here."
        elif _hiding_relation(document, local_id, start, end):
            state, detail = "possibly_hidden", "A stored relationship suggests another part may hide it here."
        elif _at_edge(left, policy.edge_margin) and _at_edge(right, policy.edge_margin):
            state, detail = "possibly_out_of_view", "The last and next boxes touch the frame edge, so it may have left the view."
        found.append(_Finding(start, end, "tracking_gap", detail, [_ref(left), _ref(right)], state))
    return found


def _hiding_relation(document: AnalysisDocument, local_id: str, start: float, end: float) -> bool:
    return any(
        item.subject_local_id == local_id
        and item.kind in _HIDING_RELATIONS
        and item.time_start < end
        and item.time_end > start
        for item in document.relationships
    )


def _jump_findings(located: list[ObservationPrediction], policy: TargetPolicy) -> list[_Finding]:
    """Sudden moves or size changes among model estimates. A tracker row is measured motion and is not flagged."""
    estimates = [row for row in located if is_estimate(row)]
    if len(estimates) < 3:
        return []
    pairs = list(zip(estimates, estimates[1:], strict=False))
    moves = [_distance(left, right) for left, right in pairs]
    usual = statistics.median(moves)
    found: list[_Finding] = []
    for (left, right), move in zip(pairs, moves, strict=False):
        size = max((left.bbox.w * left.bbox.h) ** 0.5, 1e-3)
        if move > policy.jump_sizes * size and move > policy.jump_factor * max(usual, 1e-3):
            found.append(
                _Finding(
                    left.timestamp,
                    right.timestamp,
                    "implausible_jump",
                    f"The box moved {move / size:.1f} box sizes with no tracker support.",
                    [_ref(left), _ref(right)],
                )
            )
        ratio = (right.bbox.w * right.bbox.h) / max(left.bbox.w * left.bbox.h, 1e-6)
        if max(ratio, 1 / max(ratio, 1e-6)) >= policy.size_ratio:
            found.append(
                _Finding(
                    left.timestamp,
                    right.timestamp,
                    "sudden_size_change",
                    f"The box area changed by a factor of {max(ratio, 1 / max(ratio, 1e-6)):.1f} with no tracker support.",
                    [_ref(left), _ref(right)],
                )
            )
    return found


def _distance(left: ObservationPrediction, right: ObservationPrediction) -> float:
    (lx, ly), (rx, ry) = _center(left), _center(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _disagreement_findings(rows: list[ObservationPrediction], policy: TargetPolicy) -> list[_Finding]:
    found: list[_Finding] = []
    boxed = [row for row in rows if row.bbox is not None]
    trackers = [row for row in boxed if row_family(row) == TRACKER_SOURCE and row.association_support == "associated"]
    for row in boxed:
        family = row_family(row)
        if family == "vlm":
            for other in trackers:
                if abs(other.timestamp - row.timestamp) <= SAME_FRAME_SEC and iou(_box(row), _box(other)) < policy.agree_conflict_iou:
                    found.append(
                        _Finding(
                            row.timestamp,
                            row.timestamp,
                            "vlm_disagrees_with_tracking",
                            "The model box does not overlap the tracked instance on this frame.",
                            [_ref(row), _ref(other)],
                        )
                    )
                    break
        elif row.association_status == "conflict":
            found.append(_Finding(row.timestamp, row.timestamp, "identity_conflict", "A stored row conflicts on identity.", [_ref(row)]))
    return found


def _swap_findings(document: AnalysisDocument, located: dict[str, list[ObservationPrediction]], policy: TargetPolicy) -> dict[str, list[_Finding]]:
    """Two objects whose boxes trade places between neighboring times."""
    found: dict[str, list[_Finding]] = {}
    ids = [obj.local_id for obj in document.objects]
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            times = {round(row.timestamp, 2): row for row in located.get(second, [])}
            shared = [(row, times[round(row.timestamp, 2)]) for row in located.get(first, []) if round(row.timestamp, 2) in times]
            for (a1, b1), (a2, b2) in zip(shared, shared[1:], strict=False):
                crossed = iou(_box(a1), _box(b2)) >= policy.swap_iou and iou(_box(b1), _box(a2)) >= policy.swap_iou
                if crossed and iou(_box(a1), _box(a2)) < policy.swap_iou:
                    detail = f"{first} and {second} appear to exchange places."
                    found.setdefault(first, []).append(_Finding(a1.timestamp, a2.timestamp, "identity_swap", detail, [_ref(a1), _ref(a2)]))
                    found.setdefault(second, []).append(_Finding(b1.timestamp, b2.timestamp, "identity_swap", detail, [_ref(b1), _ref(b2)]))
    return found


def _recorded_findings(
    document: AnalysisDocument, local_id: str, located: list[ObservationPrediction], policy: TargetPolicy
) -> list[_Finding]:
    found: list[_Finding] = []
    for item in document.ambiguities:
        reason = _TIME_AMBIGUITIES.get(item.kind)
        if reason and local_id in item.object_local_ids and item.time_start is not None:
            end = item.time_end if item.time_end is not None else item.time_start
            if end - item.time_start > policy.max_recorded_sec:
                continue
            found.append(_Finding(item.time_start, end, reason, item.description[:200], [item.local_id]))
    measured = [row.timestamp for row in located if not is_estimate(row)]
    for row in document.box_decisions:
        if row.object_local_id != local_id or row.timestamp is None or row.outcome == "kept" or row.bbox is None:
            continue
        if any(abs(time - row.timestamp) <= 0.45 for time in measured):
            continue
        found.append(
            _Finding(
                row.timestamp,
                row.timestamp,
                "box_removed_in_grounding",
                f"{row.step} recorded {row.reason_code}. {row.detail}".strip()[:200],
                [f"{local_id}@{row.timestamp:.3f}:{row.step}"],
            )
        )
    return found


def _merge(findings: list[_Finding], slack: float) -> list[list[_Finding]]:
    groups: list[list[_Finding]] = []
    end = None
    for item in sorted(findings, key=lambda row: (row.start, row.end)):
        if groups and end is not None and item.start <= end + slack:
            groups[-1].append(item)
            end = max(end, item.end)
        else:
            groups.append([item])
            end = item.end
    return groups


_NOT_SUGGESTED = {"recorded_absent", "recorded_occluded"}


def find_targets(
    document: AnalysisDocument,
    spatial: AnalysisDocument | None,
    seeds: list[Seed],
    *,
    policy: TargetPolicy | None = None,
) -> list[dict]:
    """Structured candidates. An explained interval is returned, marked as not recoverable."""
    policy = policy or TargetPolicy()
    grouped = _rows_by_object(document, spatial)
    located = {local_id: _located(rows) for local_id, rows in grouped.items()}
    swaps = _swap_findings(document, located, policy)
    targets: list[dict] = []
    for obj in document.objects:
        rows = grouped.get(obj.local_id, [])
        here = located.get(obj.local_id, [])
        findings = [
            *_gap_findings(document, obj.local_id, rows, here, policy),
            *_jump_findings(here, policy),
            *_disagreement_findings(rows, policy),
            *swaps.get(obj.local_id, []),
            *_recorded_findings(document, obj.local_id, here, policy),
        ]
        for group in _merge(findings, policy.merge_slack_sec):
            targets.append(_target(obj, group, seeds, located, document))
    targets.sort(key=lambda item: (not item["recoverable"], item["time_start"], item["object_local_id"]))
    return targets[: policy.max_targets]


def _target(obj, group: list[_Finding], seeds: list[Seed], located, document: AnalysisDocument) -> dict:
    start = min(item.start for item in group)
    end = max(item.end for item in group)
    states = {item.state for item in group}
    explained = states & _NOT_SUGGESTED
    state = next(iter(explained)) if explained else next(
        (name for name in ("possibly_hidden", "possibly_out_of_view", "unresolved") if name in states), "questionable"
    )
    around = seeds_around(seeds, obj.local_id, start, end)
    usable = around["usable"]
    nearest = around["best"]
    distance = max(start - nearest.timestamp, nearest.timestamp - end, 0.0) if nearest else None
    others = sum(
        1
        for local_id, rows in located.items()
        if local_id != obj.local_id and any(start <= row.timestamp <= end for row in rows)
    )
    difficulty = "easy"
    if end - start > 10 or state in {"possibly_hidden", "possibly_out_of_view"} or (distance or 0) > 8:
        difficulty = "hard"
    elif end - start > 3 or (distance or 0) > 3 or others > 2:
        difficulty = "moderate"
    if explained:
        blocker = "The stored evidence explains this interval, so it is not suggested. A person can still target it."
    elif nearest is None:
        blocker = "No trustworthy seed exists for this object. Confirm a candidate box or draw one."
    else:
        blocker = ""
    refs = sorted({ref for item in group for ref in item.refs})
    rows = located.get(obj.local_id, [])
    inside = [row for row in rows if start <= row.timestamp <= end]
    return {
        "target_id": hashlib.sha256(f"{obj.local_id}|{start:.3f}|{end:.3f}".encode()).hexdigest()[:16],
        "object_local_id": obj.local_id,
        "instance_id": obj.instance_id,
        "label": obj.label,
        "time_start": round(start, 6),
        "time_end": round(end, 6),
        "source": "automatic",
        "state": state,
        "reasons": sorted({item.reason for item in group}),
        "details": list(dict.fromkeys(item.detail for item in group))[:6],
        "source_observations": refs[:24],
        "seeds": [seed.as_dict() for seed in usable[:5]],
        "seed_before": around["before"].seed_id if around["before"] else None,
        "seed_after": around["after"].seed_id if around["after"] else None,
        "support": {
            "observations_in_interval": len(inside),
            "measured_in_interval": sum(not is_estimate(row) for row in inside),
            "usable_seeds": len(usable),
            "association_status": obj.association_status,
            "uncertain_identity": obj.uncertain_identity,
            "confidence_note": "Model and SAM scores are not calibrated probabilities and do not affect this list.",
        },
        "difficulty": difficulty,
        "recoverable": not blocker,
        "blocker": blocker,
    }
