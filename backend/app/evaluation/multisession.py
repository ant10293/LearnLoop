"""Project-level scores against human ground truth. Confidence is reported apart from accuracy."""

from __future__ import annotations


def evaluate_project(prediction: dict, truth: dict, link_scores: list[float] | None = None) -> dict:
    order = _order_accuracy(prediction.get("order") or {}, truth.get("order") or [])
    identity = _identity_accuracy(prediction.get("links") or [], truth.get("identity_groups") or [])
    transitions = _transition_accuracy(prediction.get("transitions") or [], truth.get("transitions") or [])
    dependencies = _dependency_accuracy(
        (prediction.get("workflow") or {}).get("procedural_dependencies") or [],
        truth.get("dependencies") or [],
    )
    sessions = _session_quality(prediction.get("sessions") or [], truth.get("sessions") or [])
    workflow = prediction.get("workflow") or {}
    unresolved = [
        step for step in workflow.get("steps") or [] if step.get("kind") in {"unobserved_interval", "unresolved"}
    ]
    return {
        "accuracy_source": "human_ground_truth",
        "is_accuracy": True,
        "clip_ordering": order,
        "session_boundaries": sessions,
        "cross_session_identity": identity["all"],
        "reidentification_after_appearance_change": identity["appearance_change"],
        "inferred_transitions": transitions,
        "unresolved_intervals": {"count": len(unresolved), "is_accuracy": False},
        "workflow_dependencies": dependencies,
        "missing_clip_sensitivity": [
            missing_clip_sensitivity(prediction, video_id) for video_id in _video_ids(prediction)
        ],
        "confidence": {
            "kind": "model_confidence",
            "is_accuracy": False,
            "mean_link_score": round(sum(link_scores) / len(link_scores), 4) if link_scores else None,
            "note": "Link scores are model or matcher confidence. They are not accuracy.",
        },
    }


def revision_effect(previous: dict, current: dict) -> dict:
    prev_clips = set((previous or {}).get("clip_run_ids") or {})
    next_clips = set((current or {}).get("clip_run_ids") or {})
    prev_objects = set((previous or {}).get("canonical_ids") or [])
    next_objects = set((current or {}).get("canonical_ids") or [])
    return {
        "clips_added": sorted(next_clips - prev_clips),
        "clips_removed": sorted(prev_clips - next_clips),
        "canonical_ids_added": sorted(next_objects - prev_objects),
        "canonical_ids_removed": sorted(prev_objects - next_objects),
        "order_changed": (previous or {}).get("order") != (current or {}).get("order"),
        "previous_revision_retained": True,
        "cached_runs_unchanged": (previous or {}).get("clip_run_ids") == (current or {}).get("clip_run_ids")
        or not (prev_clips - next_clips),
    }


def missing_clip_sensitivity(prediction: dict, video_id: str) -> dict:
    steps = (prediction.get("workflow") or {}).get("steps") or []
    affected = [step.get("id") for step in steps if video_id in (step.get("video_ids") or [])]
    links = [
        link
        for link in prediction.get("links") or []
        if link.get("video_id") == video_id
    ]
    return {
        "video_id": video_id,
        "workflow_steps_touching_clip": affected,
        "identity_links_touching_clip": len(links),
        "note": "Removing this clip drops these project-level claims. Cached analyses of the other clips stay.",
    }


def _order_accuracy(order: dict, truth_order: list) -> dict:
    truth_pairs = _strict_pairs(_flatten_truth(truth_order))
    predicted_pairs = _ranked_pairs(order.get("ranked") or [])
    if not truth_pairs:
        return {"comparable_pairs": 0, "correct": 0, "accuracy": None}
    comparable = 0
    correct = 0
    for left, right in truth_pairs:
        forward = (left, right) in predicted_pairs
        backward = (right, left) in predicted_pairs
        if not forward and not backward:
            continue
        comparable += 1
        correct += int(forward)
    return {
        "comparable_pairs": comparable,
        "correct": correct,
        "accuracy": round(correct / comparable, 4) if comparable else None,
        "unordered_in_prediction": order.get("unordered") or [],
    }


def _identity_accuracy(links: list[dict], groups: list) -> dict:
    predicted = _groups_from_links(links)
    truth_pairs = _group_pairs(groups)
    predicted_pairs = _group_pairs(predicted["groups"])
    changed = {
        pair
        for pair in truth_pairs
        if _states_differ(pair, links)
    }
    predicted_changed = {pair for pair in predicted_pairs if _states_differ(pair, links)}
    return {
        "all": _pair_scores(predicted_pairs, truth_pairs),
        "appearance_change": _pair_scores(predicted_changed, changed),
    }


def _transition_accuracy(predicted: list[dict], truth: list[dict]) -> dict:
    pred = {(row.get("from_video_id"), row.get("to_video_id")): row for row in predicted}
    tp = fp = fn = 0
    class_mismatch = 0
    for row in truth:
        key = (row.get("from_video_id"), row.get("to_video_id"))
        found = pred.get(key)
        if found is None:
            fn += 1
            continue
        tp += 1
        if row.get("evidence_class") and found.get("evidence_class") != row.get("evidence_class"):
            class_mismatch += 1
    truth_keys = {(row.get("from_video_id"), row.get("to_video_id")) for row in truth}
    fp = len([key for key in pred if key not in truth_keys])
    scores = _prf(tp, fp, fn)
    scores["evidence_class_mismatches"] = class_mismatch
    scores["directly_observed_predictions"] = sum(
        1 for row in predicted if row.get("evidence_class") == "DIRECTLY_OBSERVED"
    )
    return scores


def _dependency_accuracy(predicted: list[dict], truth: list[dict]) -> dict:
    pred = {(row.get("before"), row.get("after")) for row in predicted}
    gold = {(row.get("before"), row.get("after")) for row in truth}
    tp = len(pred & gold)
    return _prf(tp, len(pred - gold), len(gold - pred))


def _session_quality(predicted: list[dict], truth: list[dict]) -> dict:
    if not truth:
        return {"mean_jaccard": None, "sessions": 0}
    scores = []
    for gold in truth:
        gold_ids = set(gold.get("video_ids") or [])
        best = 0.0
        for row in predicted:
            ids = set(row.get("video_ids") or [])
            union = gold_ids | ids
            best = max(best, (len(gold_ids & ids) / len(union)) if union else 1.0)
        scores.append(best)
    return {"mean_jaccard": round(sum(scores) / len(scores), 4), "sessions": len(scores)}


def _groups_from_links(links: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for link in links:
        if link.get("status") == "rejected":
            continue
        groups.setdefault(link.get("canonical_id") or "", []).append(
            {"video_id": link.get("video_id"), "local_id": link.get("object_local_id")}
        )
    return {"groups": list(groups.values())}


def _group_pairs(groups: list) -> set[tuple]:
    pairs = set()
    for group in groups:
        members = [_member_key(item) for item in group]
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                pairs.add(tuple(sorted((left, right))))
    return pairs


def _member_key(item) -> tuple:
    if isinstance(item, dict):
        return (item.get("video_id"), item.get("local_id") or item.get("object_local_id"))
    return tuple(item)


def _states_differ(pair: tuple, links: list[dict]) -> bool:
    left = _state_key(pair[0], links)
    right = _state_key(pair[1], links)
    return bool(left and right and left != right)


def _state_key(member: tuple, links: list[dict]) -> str:
    for link in links:
        if (link.get("video_id"), link.get("object_local_id")) == member:
            return f"{link.get('state_before', '')}|{link.get('state_after', '')}"
    return ""


def _pair_scores(predicted: set, truth: set) -> dict:
    tp = len(predicted & truth)
    return _prf(tp, len(predicted - truth), len(truth - predicted))


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _flatten_truth(order: list) -> list[str]:
    flat = []
    for item in order:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _strict_pairs(ids: list[str]) -> set[tuple[str, str]]:
    return {(left, right) for index, left in enumerate(ids) for right in ids[index + 1 :]}


def _ranked_pairs(groups: list) -> set[tuple[str, str]]:
    pairs = set()
    for index, group in enumerate(groups):
        for later in groups[index + 1 :]:
            for left in group:
                for right in later:
                    pairs.add((left, right))
    return pairs


def _video_ids(prediction: dict) -> list[str]:
    return list((prediction.get("clip_run_ids") or {}).keys())
