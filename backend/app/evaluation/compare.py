"""Compare two runs without assuming their object ids mean the same thing."""

from __future__ import annotations

from app.evaluation.metrics import greedy_match, interval_iou, prf
from app.schemas.analysis import AnalysisDocument


def compare_documents(
    left: AnalysisDocument,
    right: AnalysisDocument,
    manual_matches: list[dict] | None = None,
) -> dict:
    manual = {(item["left"], item["right"]) for item in (manual_matches or [])}
    pairs = []
    for i, a in enumerate(left.objects):
        for j, b in enumerate(right.objects):
            if (a.local_id, b.local_id) in manual:
                pairs.append((1.0, i, j))
                continue
            score = _label(a.label, b.label)
            pairs.append((score, i, j))
    threshold = 0.34 if not manual else 0.34
    matches = greedy_match(pairs, threshold)
    forced = []
    used_l = {i for i, _j, _s in matches}
    used_r = {j for _i, j, _s in matches}
    for i, a in enumerate(left.objects):
        for j, b in enumerate(right.objects):
            if (a.local_id, b.local_id) in manual and i not in used_l and j not in used_r:
                forced.append((i, j, 1.0))
                used_l.add(i)
                used_r.add(j)
    matches = matches + forced
    action_pairs = []
    for i, a in enumerate(left.actions):
        for j, b in enumerate(right.actions):
            score = interval_iou(a.start, a.end, b.start, b.end)
            if a.action_type != b.action_type:
                score *= 0.25
            if a.outcome != b.outcome:
                score *= 0.8
            action_pairs.append((score, i, j))
    action_matches = greedy_match(action_pairs, 0.3)
    disagreements = []
    for i, j, score in action_matches:
        a, b = left.actions[i], right.actions[j]
        if a.action_type != b.action_type or a.outcome != b.outcome or abs(a.start - b.start) > 0.4:
            disagreements.append(
                {
                    "left": a.local_id,
                    "right": b.local_id,
                    "overlap": round(score, 3),
                    "left_type": a.action_type,
                    "right_type": b.action_type,
                    "left_outcome": a.outcome.value,
                    "right_outcome": b.outcome.value,
                    "start_delta": round(a.start - b.start, 3),
                    "note": "Disagreement is not evidence that either run is wrong.",
                }
            )
    left_only = [
        item.local_id
        for index, item in enumerate(left.actions)
        if index not in {i for i, _j, _s in action_matches}
    ]
    right_only = [
        item.local_id
        for index, item in enumerate(right.actions)
        if index not in {j for _i, j, _s in action_matches}
    ]
    return {
        "object_matches": [
            {
                "left": left.objects[i].local_id,
                "right": right.objects[j].local_id,
                "left_label": left.objects[i].label,
                "right_label": right.objects[j].label,
                "score": round(score, 3),
                "method": "manual" if (left.objects[i].local_id, right.objects[j].local_id) in manual else "assisted_label",
            }
            for i, j, score in matches
        ],
        "unmatched_left_objects": [
            item.local_id for index, item in enumerate(left.objects) if index not in used_l
        ],
        "unmatched_right_objects": [
            item.local_id for index, item in enumerate(right.objects) if index not in used_r
        ],
        "action_overlap": prf(
            len(action_matches),
            len(left_only),
            len(right_only),
        ),
        "left_only_actions": left_only,
        "right_only_actions": right_only,
        "disagreements": disagreements,
        "counts": {
            "left_objects": len(left.objects),
            "right_objects": len(right.objects),
            "left_actions": len(left.actions),
            "right_actions": len(right.actions),
            "left_steps": len(left.workflow_steps),
            "right_steps": len(right.workflow_steps),
            "left_ambiguities": len(left.ambiguities),
            "right_ambiguities": len(right.ambiguities),
        },
        "note": "Object ids are not equivalent across runs. Assisted matches use labels only.",
    }


def _label(left: str, right: str) -> float:
    a = set(left.lower().split())
    b = set(right.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
