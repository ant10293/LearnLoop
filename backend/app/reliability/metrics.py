"""Reliability figures from normalized repetition records. Pure functions, computed on every read,
so a changed ground-truth label recalculates accuracy without running anything again.

Every rate carries its numerator and denominator. A figure that cannot be computed from what
was recorded is None (shown as N/A), never a guess. Consistency is agreement between runs and
says nothing about correctness; accuracy needs a label.
"""

from __future__ import annotations

import statistics
from collections import Counter

from app.reliability.normalize import CANCELED, EXECUTION_ERROR, INVALID, VERDICT

VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE")
STEP_VERDICTS = ("PASS", "FAIL", "CANNOT_VERIFY", "NOT_APPLICABLE")
TERMINAL = {"completed", "failed", "canceled"}


def rate(numerator: int, denominator: int) -> dict:
    return {"value": round(numerator / denominator, 4) if denominator else None, "numerator": numerator, "denominator": denominator}


def outcome_of(run: dict) -> str | None:
    """verdict, execution_error, canceled, invalid_result, or None while it has not finished."""
    status = run.get("status")
    if status == "canceled":
        return CANCELED
    if status == "failed":
        return INVALID if run.get("error_kind") == "invalid_result" else EXECUTION_ERROR
    if status == "completed":
        return (run.get("result") or {}).get("outcome") or INVALID
    return None


def counts(runs: list[dict]) -> dict:
    by_status = Counter(run.get("status") for run in runs)
    outcomes = Counter(outcome_of(run) for run in runs)
    return {
        "total": len(runs),
        "queued": by_status.get("queued", 0),
        "running": by_status.get("running", 0),
        "completed": by_status.get("completed", 0),
        "failed": by_status.get("failed", 0),
        "canceled": by_status.get("canceled", 0),
        "finished": sum(by_status.get(status, 0) for status in TERMINAL),
        "valid": outcomes.get(VERDICT, 0),
        "execution_errors": outcomes.get(EXECUTION_ERROR, 0),
        "invalid_results": outcomes.get(INVALID, 0),
        "error_kinds": dict(Counter(run.get("error_kind") for run in runs if outcome_of(run) in {EXECUTION_ERROR, INVALID})),
    }


def _valid(runs: list[dict]) -> list[dict]:
    return [run for run in runs if outcome_of(run) == VERDICT]


def consistency(runs: list[dict]) -> dict:
    """Distribution of overall verdicts among runs that produced one, and the share that agree with the most common."""
    valid = [run["result"]["verdict"] for run in _valid(runs)]
    distribution = {verdict: valid.count(verdict) for verdict in VERDICTS}
    top = max(distribution.values()) if valid else 0
    modes = [verdict for verdict, count in distribution.items() if count == top and top > 0]
    return {
        "valid_runs": len(valid),
        "distribution": distribution,
        "modal_verdicts": modes,
        "modal_agreement": rate(top, len(valid)),
        "distinct_verdicts": sum(count > 0 for count in distribution.values()),
    }


def accuracy(runs: list[dict], ground_truth: dict) -> dict:
    """Accuracy against the batch's expected verdict. Unlabeled batches get no accuracy figures.

    exact_match         runs whose verdict equals the label, over all valid runs. INCONCLUSIVE is a miss.
    decisive_accuracy   correct among runs that said PASS or FAIL, shown with decisive_coverage.
    decisive_coverage   runs that said PASS or FAIL, over all valid runs.
    inconclusive_rate   INCONCLUSIVE runs over all valid runs.
    false_pass_rate     on a recording labeled FAIL: PASS runs over all valid runs of it. N/A on a PASS label.
    false_fail_rate     on a recording labeled PASS: FAIL runs over all valid runs of it. N/A on a FAIL label.
    """
    expected = (ground_truth or {}).get("expected_verdict")
    if expected not in {"PASS", "FAIL"}:
        return {"labeled": False, "expected": None}
    verdicts = [run["result"]["verdict"] for run in _valid(runs)]
    total = len(verdicts)
    decisive = [verdict for verdict in verdicts if verdict in {"PASS", "FAIL"}]
    correct = sum(verdict == expected for verdict in verdicts)
    passes = verdicts.count("PASS")
    fails = verdicts.count("FAIL")
    na = rate(0, 0) | {"applicable": False}
    return {
        "labeled": True,
        "expected": expected,
        "exact_match": rate(correct, total),
        "decisive_accuracy": rate(sum(verdict == expected for verdict in decisive), len(decisive)),
        "decisive_coverage": rate(len(decisive), total),
        "inconclusive_rate": rate(verdicts.count("INCONCLUSIVE"), total),
        "false_pass": rate(passes, total) | {"applicable": True} if expected == "FAIL" else na,
        "false_fail": rate(fails, total) | {"applicable": True} if expected == "PASS" else na,
    }


def per_step(runs: list[dict], step_order: list[dict], ground_truth: dict) -> list[dict]:
    """Each procedure step across runs. Steps are the frozen procedure's own ids, so they line up exactly."""
    labels = (ground_truth or {}).get("step_labels") or {}
    valid = _valid(runs)
    seen_ids = [item["step_id"] for item in step_order]
    for run in valid:
        for step in run["result"].get("steps") or []:
            if step["step_id"] not in seen_ids:
                seen_ids.append(step["step_id"])
    names = {item["step_id"]: item.get("name") or item["step_id"] for item in step_order}
    out = []
    for step_id in seen_ids:
        present = []
        for run in valid:
            match = next((step for step in run["result"].get("steps") or [] if step["step_id"] == step_id), None)
            if match is not None:
                present.append(match)
                names.setdefault(step_id, match.get("name") or step_id)
        verdicts = [step.get("verdict") or "UNRECOGNIZED" for step in present]
        distribution = {verdict: verdicts.count(verdict) for verdict in STEP_VERDICTS}
        unrecognized = verdicts.count("UNRECOGNIZED")
        top = max(Counter(verdicts).values()) if verdicts else 0
        label = labels.get(step_id)
        entry = {
            "step_id": step_id,
            "name": names.get(step_id, step_id),
            "required": any(step.get("required") for step in present),
            "runs_with_step": len(present),
            "runs_missing_step": len(valid) - len(present),
            "distribution": distribution,
            "unrecognized_verdicts": unrecognized,
            "agreement": rate(top, len(present)),
            "unstable": len(set(verdicts)) > 1,
            "missing_evidence": rate(sum(step.get("missing_evidence", False) for step in present), len(present)),
            "with_warnings": rate(sum(bool(step.get("warnings")) for step in present), len(present)),
            "reinspected": sum(bool(step.get("reinspected")) for step in present),
            "mean_evidence": round(statistics.fmean([step.get("evidence_count", 0) for step in present]), 2) if present else None,
            "error_categories": dict(Counter(step.get("error_category") for step in present if step.get("error_category"))),
            "label": label,
            "accuracy": rate(sum(step.get("verdict") == label for step in present), len(present)) if label else None,
        }
        out.append(entry)
    return out


def detection(runs: list[dict]) -> dict:
    """Object and evidence diagnostics that the pipeline recorded, across valid runs. Nothing here is inferred."""
    valid = _valid(runs)
    role_status: dict[str, Counter] = {}
    role_labels: dict[str, str] = {}
    no_boxes: Counter = Counter()
    relocated: Counter = Counter()
    interrupted: Counter = Counter()
    labels: dict[str, str] = {}
    views_made = 0
    views_errors = 0
    appearance_used = 0
    recovered = []
    for run in valid:
        diagnostics = run["result"].get("diagnostics") or {}
        for role in diagnostics.get("roles") or []:
            role_status.setdefault(role["role_id"], Counter())[role.get("status") or "unknown"] += 1
            role_labels[role["role_id"]] = role.get("label") or role["role_id"]
        objects = diagnostics.get("objects") or {}
        if not objects.get("available"):
            continue
        views_made += 1
        views_errors += bool(objects.get("error"))
        appearance_used += bool(objects.get("appearance_used"))
        recovered.append(objects.get("recovered_sightings") or 0)
        for local_id in objects.get("objects_without_boxes") or []:
            no_boxes[local_id] += 1
        for local_id in objects.get("relocated") or []:
            relocated[local_id] += 1
        for local_id, item in (objects.get("objects") or {}).items():
            labels.setdefault(local_id, item.get("label") or local_id)
            if (item.get("track_segments") or 0) > 1:
                interrupted[local_id] += 1
    n = len(valid)
    return {
        "valid_runs": n,
        "roles": [
            {
                "role_id": role_id,
                "label": role_labels.get(role_id, role_id),
                "statuses": dict(tally),
                "unresolved": rate(tally.get("unresolved", 0), sum(tally.values())),
                "ambiguous": rate(tally.get("ambiguous", 0), sum(tally.values())),
            }
            for role_id, tally in sorted(role_status.items())
        ],
        "object_views": {
            "runs_with_views": views_made,
            "runs_with_view_errors": views_errors,
            "runs_with_appearance_check": appearance_used,
            "mean_recovered_sightings": round(statistics.fmean(recovered), 2) if recovered else None,
            # Object ids are the attempt analysis's own and can differ between runs; labels are shown beside them.
            "objects_without_boxes": [{"object_id": key, "label": labels.get(key, key), "runs": count} for key, count in no_boxes.most_common()],
            "relocated_after_loss": [{"object_id": key, "label": labels.get(key, key), "runs": count} for key, count in relocated.most_common()],
            "interrupted_tracks": [{"object_id": key, "label": labels.get(key, key), "runs": count} for key, count in interrupted.most_common()],
        },
        "warnings": dict(Counter(warning for run in valid for warning in (run["result"].get("diagnostics") or {}).get("warnings") or [])),
    }


def _seconds(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def performance(runs: list[dict], batch_seconds: float | None) -> dict:
    """Per-repetition runtime (each run's own launch to finish) apart from the batch's elapsed wall clock."""
    finished = [run for run in runs if run.get("status") in TERMINAL and run.get("started_at")]
    durations = [run["duration_seconds"] for run in finished if run.get("duration_seconds") is not None and run.get("status") == "completed"]
    usages = [((run.get("result") or {}).get("execution_metadata") or {}).get("usage") or {} for run in runs]
    costs = [item.get("cost_usd") for item in usages if item]
    terminal = [run for run in runs if run.get("status") in TERMINAL and run.get("status") != "canceled"]
    errors = [run for run in terminal if outcome_of(run) in {EXECUTION_ERROR, INVALID}]
    return {
        "runtime_seconds": _seconds(durations),
        "total_run_seconds": round(sum(durations), 2) if durations else None,
        "batch_wall_clock_seconds": round(batch_seconds, 2) if batch_seconds is not None else None,
        "execution_error_rate": rate(len(errors), len(terminal)),
        "requests": sum(item.get("requests", 0) for item in usages),
        "failed_requests": sum(item.get("failed_requests", 0) for item in usages),
        "input_tokens": sum(item.get("input_tokens", 0) for item in usages),
        "output_tokens": sum(item.get("output_tokens", 0) for item in usages),
        # Only when every recorded response had a known price.
        "cost_usd": round(sum(costs), 6) if costs and all(cost is not None for cost in costs) else None,
    }


def independence(runs: list[dict]) -> dict:
    """How many finished runs recorded fully fresh inference, read from each run's own bookkeeping."""
    finished = [run for run in runs if run.get("result")]
    fresh = [run for run in finished if (((run["result"].get("execution_metadata") or {}).get("inference") or {}).get("fully_fresh"))]
    replayed = sum((((run["result"].get("execution_metadata") or {}).get("inference") or {}).get("replayed_answers") or 0) for run in finished)
    return {"runs_with_results": len(finished), "fully_fresh_runs": len(fresh), "replayed_answers": replayed}


def verification_metrics(runs: list[dict], step_order: list[dict], ground_truth: dict, batch_seconds: float | None) -> dict:
    return {
        "counts": counts(runs),
        "consistency": consistency(runs),
        "accuracy": accuracy(runs, ground_truth),
        "steps": per_step(runs, step_order, ground_truth),
        "detection": detection(runs),
        "performance": performance(runs, batch_seconds),
        "independence": independence(runs),
    }
