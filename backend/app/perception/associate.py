"""One-to-one temporal association between two sets of box tracks.

A subject (a semantic object, or a SAM track) is linked to a candidate (a
tracked instance, or a semantic object) only by boxes that overlap at shared
times. Labels, confidence, and the nearest leftover center are not evidence.
A close second keeps both open instead of picking one.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from app.perception.assignment import iou, linear_assignment

ACCEPT_IOU = 0.35
MARGIN = 0.12
# A model-named instance whose boxes overlap less than this on repeated shared
# times is contradicted by the measurement.
CONTRADICT_IOU = 0.2
TIME_WINDOW = 0.45

Box = tuple[float, float, float, float]


@dataclass
class PairStats:
    shared: int
    score: float


@dataclass
class Link:
    subject: str
    candidate: str | None
    # "associated" (two or more shared times), "weak" (one), "ambiguous"
    # (a rival is within the margin), or "unmatched".
    status: str
    score: float
    shared: int
    margin: float | None
    rivals: list[str] = field(default_factory=list)


def pair_stats(subject: list[tuple[float, Box]], candidate: list[tuple[float, Box]], window: float = TIME_WINDOW) -> PairStats:
    """Mean IoU over the subject times that have a candidate box within `window`."""
    if not subject or not candidate:
        return PairStats(0, 0.0)
    ordered = sorted(candidate, key=lambda item: item[0])
    times = [item[0] for item in ordered]
    overlaps: list[float] = []
    for timestamp, box in subject:
        index = bisect.bisect_left(times, timestamp)
        nearby = [ordered[i] for i in (index - 1, index) if 0 <= i < len(ordered)]
        nearby = [item for item in nearby if abs(item[0] - timestamp) <= window]
        if not nearby:
            continue
        closest = min(nearby, key=lambda item: abs(item[0] - timestamp))
        overlaps.append(iou(box, closest[1]))
    if not overlaps:
        return PairStats(0, 0.0)
    return PairStats(len(overlaps), sum(overlaps) / len(overlaps))


def associate(
    subjects: dict[str, list[tuple[float, Box]]],
    candidates: dict[str, list[tuple[float, Box]]],
    *,
    window: float = TIME_WINDOW,
    accept: float = ACCEPT_IOU,
    margin: float = MARGIN,
) -> tuple[list[Link], dict[tuple[str, str], PairStats]]:
    """One-to-one links. Returns one link per subject and every scored pair."""
    subject_ids = list(subjects)
    candidate_ids = list(candidates)
    stats: dict[tuple[str, str], PairStats] = {}
    for s in subject_ids:
        for c in candidate_ids:
            pair = pair_stats(subjects[s], candidates[c], window)
            if pair.shared:
                stats[(s, c)] = pair
    links: list[Link] = []
    if not subject_ids:
        return links, stats
    cost = np.full((len(subject_ids), max(len(candidate_ids), 1)), 1.0)
    for (s, c), pair in stats.items():
        if pair.score >= accept:
            cost[subject_ids.index(s), candidate_ids.index(c)] = 1.0 - pair.score
    assigned = {
        subject_ids[i]: candidate_ids[j]
        for i, j in linear_assignment(cost)
        if candidate_ids and cost[i, j] < 1.0
    }
    for s in subject_ids:
        ranked = sorted(
            ((c, pair) for (subject, c), pair in stats.items() if subject == s),
            key=lambda item: item[1].score,
            reverse=True,
        )
        rivals = [c for c, _pair in ranked[:3]]
        chosen = assigned.get(s)
        if chosen is None:
            best = ranked[0][1] if ranked else PairStats(0, 0.0)
            links.append(Link(s, None, "unmatched", best.score, best.shared, None, rivals))
            continue
        pair = stats[(s, chosen)]
        # A track that never coexists with the chosen one is a continuation candidate, not a rival.
        # The caller records it as a re-identification hypothesis.
        span = _span(candidates[chosen])
        other_candidates = [p.score for c, p in ranked if c != chosen and _overlap(span, _span(candidates[c]))]
        other_subjects = [p.score for (subject, c), p in stats.items() if c == chosen and subject != s]
        second = max([*other_candidates, *other_subjects], default=0.0)
        gap = pair.score - second
        if gap < margin and second > 0:
            links.append(Link(s, None, "ambiguous", pair.score, pair.shared, gap, rivals))
        elif pair.shared < 2:
            links.append(Link(s, None, "weak", pair.score, pair.shared, gap, rivals))
        else:
            links.append(Link(s, chosen, "associated", pair.score, pair.shared, gap, rivals))
    return links, stats


def _span(rows: list[tuple[float, Box]]) -> tuple[float, float]:
    times = [item[0] for item in rows]
    return (min(times), max(times)) if times else (0.0, 0.0)


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]
