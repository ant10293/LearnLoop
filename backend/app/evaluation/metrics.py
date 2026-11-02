"""Interval matching. Scores are not model confidence and not inter-model agreement."""

from __future__ import annotations


def interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 1.0 if abs(start_a - start_b) < 1e-6 else 0.0
    return inter / union


def greedy_match(pairs: list[tuple[float, int, int]], threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one matching, highest overlap first. Not list-index comparison."""
    ordered = sorted((score, left, right) for score, left, right in pairs if score >= threshold)
    ordered.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = []
    for score, left, right in ordered:
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        matches.append((left, right, score))
    return matches


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_documents(predicted: dict, truth: dict, tolerances: dict | None = None) -> dict:
    tolerances = tolerances or {}
    action_iou = float(tolerances.get("action_iou") or 0.3)
    boundary = float(tolerances.get("boundary_sec") or 0.5)
    pred_actions = predicted.get("actions") or []
    true_actions = truth.get("actions") or []
    pairs = []
    for i, pred in enumerate(pred_actions):
        for j, gold in enumerate(true_actions):
            score = interval_iou(pred["start"], pred["end"], gold["start"], gold["end"])
            if pred.get("action_type") != gold.get("action_type"):
                score *= 0.25
            pairs.append((score, i, j))
    matches = greedy_match(pairs, action_iou)
    matched_p = {i for i, _j, _s in matches}
    matched_g = {j for _i, j, _s in matches}
    action_errors = []
    boundary_errors = []
    for i, j, score in matches:
        pred, gold = pred_actions[i], true_actions[j]
        start_err = abs(pred["start"] - gold["start"])
        end_err = abs(pred["end"] - gold["end"])
        boundary_errors.append(start_err + end_err)
        if pred.get("action_type") != gold.get("action_type") or start_err > boundary or end_err > boundary:
            action_errors.append(
                {
                    "kind": "boundary_or_type",
                    "predicted": pred.get("local_id"),
                    "truth": gold.get("local_id"),
                    "time_start": pred.get("start"),
                    "time_end": pred.get("end"),
                    "detail": (
                        f"Matched on overlap {score:.2f}. "
                        f"Type {pred.get('action_type')} vs {gold.get('action_type')}. "
                        f"Boundary error {start_err + end_err:.2f}s."
                    ),
                }
            )
    for i, pred in enumerate(pred_actions):
        if i not in matched_p:
            action_errors.append(
                {
                    "kind": "false_positive",
                    "predicted": pred.get("local_id"),
                    "time_start": pred.get("start"),
                    "time_end": pred.get("end"),
                    "detail": pred.get("description") or pred.get("action_type"),
                }
            )
    for j, gold in enumerate(true_actions):
        if j not in matched_g:
            action_errors.append(
                {
                    "kind": "false_negative",
                    "truth": gold.get("local_id"),
                    "time_start": gold.get("start"),
                    "time_end": gold.get("end"),
                    "detail": gold.get("description") or gold.get("action_type"),
                }
            )
    pred_objects = predicted.get("objects") or []
    true_objects = truth.get("objects") or []
    object_pairs = []
    for i, pred in enumerate(pred_objects):
        for j, gold in enumerate(true_objects):
            object_pairs.append((_label_score(pred.get("label", ""), gold.get("label", "")), i, j))
    object_matches = greedy_match(object_pairs, float(tolerances.get("label_score") or 0.5))
    obj_p = {i for i, _j, _s in object_matches}
    obj_g = {j for _i, j, _s in object_matches}
    dep_metrics = _dependencies(predicted, truth, matches, pred_actions, true_actions)
    return {
        "definition": (
            "Actions match one-to-one by interval IoU, down-weighted when the verb differs. "
            "Objects match by normalized label overlap and are not treated as identity ground truth "
            "unless a researcher confirmed the mapping. Model confidence is not used."
        ),
        "actions": prf(len(matches), len(pred_actions) - len(matched_p), len(true_actions) - len(matched_g)),
        "objects": prf(
            len(object_matches),
            len(pred_objects) - len(obj_p),
            len(true_objects) - len(obj_g),
        ),
        "mean_boundary_error_sec": round(sum(boundary_errors) / len(boundary_errors), 4)
        if boundary_errors
        else None,
        "dependencies": dep_metrics,
        "counts": {
            "predicted_objects": len(pred_objects),
            "truth_objects": len(true_objects),
            "predicted_actions": len(pred_actions),
            "truth_actions": len(true_actions),
            "predicted_steps": len(predicted.get("workflow_steps") or []),
            "truth_steps": len(truth.get("workflow_steps") or []),
            "predicted_ambiguities": len(predicted.get("ambiguities") or []),
        },
        "errors": action_errors,
        "identity": _identity(predicted, truth, tolerances),
        "object_matches": [
            {
                "predicted": pred_objects[i].get("local_id"),
                "truth": true_objects[j].get("local_id"),
                "score": round(score, 3),
                "method": "assisted_label",
            }
            for i, j, score in object_matches
        ],
    }


def _identity(predicted: dict, truth: dict, tolerances: dict | None) -> dict:
    from app.perception.identity_eval import evaluate_identity

    return evaluate_identity(predicted, truth, tolerances)


def _label_score(left: str, right: str) -> float:
    a = set(left.lower().split())
    b = set(right.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dependencies(predicted, truth, action_matches, pred_actions, true_actions) -> dict:
    pred_steps = predicted.get("workflow_steps") or []
    true_steps = truth.get("workflow_steps") or []
    pred_action_to_step = {}
    for step in pred_steps:
        for action_id in step.get("action_local_ids") or []:
            pred_action_to_step[action_id] = step.get("local_id")
    true_action_to_step = {}
    for step in true_steps:
        for action_id in step.get("action_local_ids") or []:
            true_action_to_step[action_id] = step.get("local_id")
    step_map = {}
    for i, j, _score in action_matches:
        pred_step = pred_action_to_step.get(pred_actions[i].get("local_id"))
        true_step = true_action_to_step.get(true_actions[j].get("local_id"))
        if pred_step and true_step:
            step_map[pred_step] = true_step
    def edges(document, mapping=None):
        found = set()
        for dep in document.get("dependencies") or []:
            src = dep.get("from_step")
            dst = dep.get("to_step")
            if mapping is not None:
                src = mapping.get(src)
                dst = mapping.get(dst)
            if src and dst:
                found.add((src, dst, dep.get("category")))
        return found
    pred_edges = edges(predicted, step_map)
    true_edges = edges(truth)
    tp = len(pred_edges & true_edges)
    return prf(tp, len(pred_edges) - tp, len(true_edges) - tp)
