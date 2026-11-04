"""Compare procedures learned independently from one demonstration.

Step ids and order mean nothing across runs, so nothing is lined up by position. Object
labels are normalized with fixed text rules, and a step is aligned with steps of other
runs only when its checkable conditions, written in those normalized labels, are exactly
the same. Everything that does not align that way is listed for a person to read. No
similarity score and no model decides that two things agree.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from itertools import combinations

from app.reliability.metrics import counts, outcome_of, performance, rate
from app.reliability.normalize import VERDICT

_ARTICLES = {"a", "an", "the"}


def normalize_label(text: str) -> str:
    """Lowercase words without articles or punctuation, each trailing plural s dropped. "The Red Bricks" -> "red brick"."""
    words = []
    for word in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if word in _ARTICLES:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def _role_labels(learned: dict) -> dict[str, str]:
    return {role["role_id"]: normalize_label(role.get("label") or role.get("original_label") or role["role_id"]) for role in learned.get("roles") or []}


def condition_signature(condition: dict, labels: dict[str, str]) -> str:
    subject = labels.get(condition.get("subject_role") or "", condition.get("subject_role") or "")
    target = labels.get(condition.get("target_role") or "", condition.get("target_role") or "")
    if condition.get("relation"):
        return f"{subject} {condition['relation']} {target}".strip()
    state = condition.get("to_state") or condition.get("state_label") or ""
    return f"{subject} is {state}".strip()


def step_signature(step: dict, labels: dict[str, str]) -> str:
    """The step's conditions in normalized labels, sorted. Empty for a step with no checkable condition."""
    return " & ".join(sorted({condition_signature(item, labels) for item in step.get("completion_conditions") or []}))


def _steps(learned: dict) -> list[dict]:
    return [step for step in learned.get("steps") or [] if not step.get("removed")]


def _spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "range": round(max(values) - min(values), 2),
        "stdev": round(statistics.pstdev(values), 3),
    }


def learning_metrics(runs: list[dict], reference: dict | None, batch_seconds: float | None) -> dict:
    valid = [run for run in runs if outcome_of(run) == VERDICT and run.get("learned")]
    n = len(valid)
    step_counts = Counter(run["result"]["step_count"] for run in valid)
    object_counts = Counter(run["result"]["object_count"] for run in valid)
    role_counts = Counter(run["result"]["role_count"] for run in valid)

    label_runs: Counter = Counter()
    run_label_sets = []
    signatures: dict[str, dict] = {}
    unaligned_steps: list[dict] = []
    for run in valid:
        learned = run["learned"]
        labels = _role_labels(learned)
        label_set = set(labels.values())
        run_label_sets.append(label_set)
        label_runs.update(label_set)
        for position, step in enumerate(_steps(learned), start=1):
            signature = step_signature(step, labels)
            if not signature:
                unaligned_steps.append({"run": run["index"], "position": position, "name": step.get("name") or "", "reason": "no checkable condition"})
                continue
            entry = signatures.setdefault(signature, {"signature": signature, "runs": set(), "names": Counter(), "positions": [], "starts": [], "ends": [], "required": 0, "occurrences": 0})
            entry["runs"].add(run["index"])
            entry["occurrences"] += 1
            entry["names"][step.get("name") or ""] += 1
            entry["positions"].append(position)
            entry["required"] += bool(step.get("required"))
            if step.get("demo_time_start") is not None:
                entry["starts"].append(float(step["demo_time_start"]))
            if step.get("demo_time_end") is not None:
                entry["ends"].append(float(step["demo_time_end"]))

    aligned = []
    for entry in sorted(signatures.values(), key=lambda item: (-len(item["runs"]), statistics.fmean(item["positions"]))):
        in_runs = sorted(entry["runs"])
        # A signature that occurs twice in one run cannot be lined up one to one, so its times are not compared.
        one_to_one = entry["occurrences"] == len(in_runs)
        aligned.append(
            {
                "signature": entry["signature"],
                "runs": in_runs,
                "share": rate(len(in_runs), n),
                "in_every_run": len(in_runs) == n and n > 0,
                "names": [name for name, _count in entry["names"].most_common()],
                "positions": sorted(set(entry["positions"])),
                "required_in": entry["required"],
                "one_to_one": one_to_one,
                "start_seconds": _spread(entry["starts"]) if one_to_one else None,
                "end_seconds": _spread(entry["ends"]) if one_to_one else None,
            }
        )

    pairwise = [len(a & b) / len(a | b) for a, b in combinations(run_label_sets, 2) if a | b]
    out = {
        "counts": counts(runs),
        "valid_runs": n,
        "step_count_distribution": dict(sorted(step_counts.items())),
        "object_count_distribution": dict(sorted(object_counts.items())),
        "role_count_distribution": dict(sorted(role_counts.items())),
        "labels": [{"label": label, "runs": count, "share": rate(count, n)} for label, count in label_runs.most_common()],
        "labels_in_every_run": sorted(label for label, count in label_runs.items() if count == n and n > 0),
        # Share of normalized labels two runs have in common, averaged over every pair of runs. Exact text after normalization.
        "mean_pairwise_label_overlap": round(statistics.fmean(pairwise), 4) if pairwise else None,
        "aligned_steps": aligned,
        "steps_in_every_run": sum(item["in_every_run"] for item in aligned),
        "unaligned_steps": unaligned_steps,
        "reviewable": rate(sum(bool(run["result"].get("reviewable")) for run in valid), n),
        "validation_errors": rate(sum(bool(run["result"]["validation"].get("errors")) for run in valid), n),
        "runs_with_ambiguities": rate(sum(bool(run["result"].get("ambiguities")) for run in valid), n),
        "performance": performance(runs, batch_seconds),
        "reference": None,
    }
    if reference:
        out["reference"] = reference_match(valid, reference)
    return out


def reference_match(valid: list[dict], reference: dict) -> dict:
    """How many of a reference procedure's condition signatures each run produced, by exact normalized text.
    Structural agreement only: it is not a judgement that a run's procedure is correct."""
    labels = _role_labels(reference)
    wanted = {step_signature(step, labels) for step in _steps(reference)} - {""}
    per_run = []
    for run in valid:
        learned = run["learned"]
        mine = {step_signature(step, _role_labels(learned)) for step in _steps(learned)} - {""}
        per_run.append({"run": run["index"], "matched": len(wanted & mine), "extra": len(mine - wanted), "missing": sorted(wanted - mine)})
    return {"reference_signatures": sorted(wanted), "per_run": per_run}
