"""Instance-identity metrics. Label overlap is not identity accuracy.

Observations match one-to-one by timestamp and, when both sides have a box,
by box IoU. Instance ids are then compared. The id strings do not have to be
the same vocabulary: predicted ids are mapped to ground-truth ids by a
one-to-one assignment that maximizes observation overlap. IDF1 uses that
mapping.

- IDTP: matched observations whose mapped ids agree.
- IDFP: predicted observations that are not IDTP.
- IDFN: ground-truth observations that are not IDTP.
- IDF1: the F1 of IDTP, IDFP, and IDFN.
- Identity switch: a ground-truth instance whose matched predicted id changes
  between consecutive matched observations.
- False merge: one predicted instance is matched to observations of two or
  more ground-truth instances.
- Tracking coverage: matched observations divided by annotated observations,
  averaged over ground-truth instances.
- Re-identification: a ground-truth instance with a gap of at least
  `reid_gap_sec`. Recovered when the predicted id is the same on both sides
  of the gap.
- Abstention: predicted observations with continuity `uncertain` or status
  `ambiguous`, divided by predicted observations.
- Hypothesis resolution: counts of proposed, rejected, ambiguous, and
  confirmed decisions. Those counts are not accuracy.

If the ground truth has no instance ids, no identity score is reported.
"""

from __future__ import annotations

import numpy as np

from app.perception.assignment import iou, linear_assignment


def evaluate_identity(predicted: dict, truth: dict, tolerances: dict | None = None) -> dict:
    tolerances = tolerances or {}
    time_tolerance = float(tolerances.get("identity_time_sec") or 0.2)
    box_threshold = float(tolerances.get("identity_box_iou") or 0.3)
    gap_sec = float(tolerances.get("reid_gap_sec") or 0.5)
    truth_obs = _observed(truth)
    pred_obs = _observed(predicted)
    if not truth_obs or not any(item.get("instance_id") for item in truth_obs):
        return {
            "evaluated": False,
            "reason": "Ground truth has no instance ids. Semantic labels are not identity accuracy.",
            "scores_are_probabilities": False,
        }
    if not any(item.get("instance_id") for item in pred_obs):
        return {
            "evaluated": False,
            "reason": "The prediction has no instance ids, so identity accuracy was not computed.",
            "scores_are_probabilities": False,
        }
    pairs = []
    for i, pred in enumerate(pred_obs):
        for j, gold in enumerate(truth_obs):
            if abs(float(pred["timestamp"]) - float(gold["timestamp"])) > time_tolerance:
                continue
            if pred.get("bbox") and gold.get("bbox"):
                score = iou(_box(pred["bbox"]), _box(gold["bbox"]))
                if score < box_threshold:
                    continue
            else:
                score = 1.0
            pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j))
    pred_ids = sorted({item["instance_id"] for item in pred_obs})
    gold_ids = sorted({item["instance_id"] for item in truth_obs})
    mapping = _map_ids(pred_obs, truth_obs, matches, pred_ids, gold_ids)
    idtp = sum(
        1
        for i, j in matches
        if mapping.get(pred_obs[i]["instance_id"]) == truth_obs[j]["instance_id"]
    )
    idfp = len(pred_obs) - idtp
    idfn = len(truth_obs) - idtp
    precision = idtp / (idtp + idfp) if idtp + idfp else 0.0
    recall = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    switches, false_merges, coverage, reid = _events(pred_obs, truth_obs, matches, gap_sec)
    abstained = sum(
        1
        for item in pred_obs
        if item.get("continuity") == "uncertain" or item.get("status") == "ambiguous"
    )
    resolution = {key: 0 for key in ("proposed", "rejected", "ambiguous", "confirmed")}
    for item in (predicted.get("perception") or {}).get("hypotheses") or []:
        decision = item.get("decision")
        if decision in resolution:
            resolution[decision] += 1
    failures = [{"kind": "identity_switch", **row} for row in switches]
    failures.extend({"kind": "false_merge", **row} for row in false_merges)
    failures.extend({"kind": "missed_reid", **row} for row in reid["misses"])
    return {
        "evaluated": True,
        "definition": (
            "Observations match by time and box. Predicted instance ids are mapped "
            "one-to-one onto ground-truth ids by maximum overlap. IDF1 counts a "
            "true positive only after that mapping. Label text is not used. "
            "These figures are counts, not probabilities."
        ),
        "scores_are_probabilities": False,
        "identity_precision": round(precision, 4),
        "identity_recall": round(recall, 4),
        "idf1": round(idf1, 4),
        "identity_switches": len(switches),
        "false_merges": len(false_merges),
        "tracking_coverage": coverage,
        "reid_opportunities": reid["opportunities"],
        "reid_recovered": reid["recovered"],
        "abstention_rate": round(abstained / len(pred_obs), 4) if pred_obs else 0.0,
        "hypothesis_resolution": resolution,
        "id_mapping": mapping,
        "failures": failures,
        "counts": {
            "predicted_observations": len(pred_obs),
            "truth_observations": len(truth_obs),
            "identity_tp": idtp,
        },
    }


def _map_ids(pred_obs, truth_obs, matches, pred_ids, gold_ids) -> dict[str, str]:
    if not pred_ids or not gold_ids:
        return {}
    cost = np.zeros((len(pred_ids), len(gold_ids)))
    for i, j in matches:
        cost[pred_ids.index(pred_obs[i]["instance_id"]), gold_ids.index(truth_obs[j]["instance_id"])] -= 1
    return {pred_ids[i]: gold_ids[j] for i, j in linear_assignment(cost)}


def _events(pred_obs, truth_obs, matches, gap_sec: float):
    linked: dict[tuple[str, float], str] = {}
    pred_golds: dict[str, set[str]] = {}
    for i, j in matches:
        gold_id = truth_obs[j]["instance_id"]
        pred_id = pred_obs[i]["instance_id"]
        linked[(gold_id, float(truth_obs[j]["timestamp"]))] = pred_id
        pred_golds.setdefault(pred_id, set()).add(gold_id)
    groups: dict[str, list[float]] = {}
    totals: dict[str, int] = {}
    for gold in truth_obs:
        groups.setdefault(gold["instance_id"], []).append(float(gold["timestamp"]))
        totals[gold["instance_id"]] = totals.get(gold["instance_id"], 0) + 1
    switches = []
    covered = []
    reid = {"opportunities": 0, "recovered": 0, "misses": []}
    for instance, times in groups.items():
        ordered = sorted(set(times))
        matched_times = [item for item in ordered if (instance, item) in linked]
        covered.append(len(matched_times) / max(totals[instance], 1))
        previous = None
        for timestamp in matched_times:
            current = linked[(instance, timestamp)]
            if previous is not None and current != previous:
                switches.append(
                    {
                        "truth_instance": instance,
                        "from_id": previous,
                        "to_id": current,
                        "timestamp": timestamp,
                    }
                )
            previous = current
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later - earlier < gap_sec:
                continue
            reid["opportunities"] += 1
            before = linked.get((instance, earlier))
            after = linked.get((instance, later))
            if before is not None and before == after:
                reid["recovered"] += 1
            else:
                reid["misses"].append(
                    {
                        "truth_instance": instance,
                        "timestamp": later,
                        "from_id": before,
                        "to_id": after,
                    }
                )
    first_time: dict[str, float] = {}
    for i, _j in matches:
        pred_id = pred_obs[i]["instance_id"]
        stamp = float(pred_obs[i]["timestamp"])
        first_time[pred_id] = min(stamp, first_time.get(pred_id, stamp))
    false_merges = [
        {
            "predicted_instance": pred_id,
            "truth_instances": sorted(golds),
            "timestamp": first_time.get(pred_id),
        }
        for pred_id, golds in pred_golds.items()
        if len(golds) > 1
    ]
    coverage = round(sum(covered) / len(covered), 4) if covered else None
    return switches, false_merges, coverage, reid


def _observed(document: dict) -> list[dict]:
    """The display set only. Raw rows are kept for inspection and would double-count an instance."""
    rows = []
    for item in document.get("observations") or []:
        instance = item.get("instance_id") or item.get("track_ref")
        if not instance or item.get("timestamp") is None:
            continue
        rows.append({**item, "instance_id": instance})
    return rows


def _box(bbox: dict) -> tuple[float, float, float, float]:
    return (float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"]))
