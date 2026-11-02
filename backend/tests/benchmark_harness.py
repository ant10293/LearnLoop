"""Score analysis documents against ground truth with the existing evaluators.

A case is a folder with `truth.json` (an `AnalysisDocument` in the existing
shape, with instance ids on observations), `predicted.json` (a run's
document), and an optional `meta.json`:

    {"source": "human" | "synthetic", "latency_sec": 41.2, "model_calls": 2,
     "semantic_calls": 2, "note": "..."}

Only a case whose meta says ``"source": "human"`` is real-world evidence.
Synthetic cases check that the harness and the pipeline agree with
themselves. Their scores are not accuracy.

Run against a folder of cases:

    cd backend && .venv/bin/python -m tests.benchmark_harness ../data/benchmark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.evaluation.metrics import evaluate_documents, greedy_match, interval_iou, prf
from app.perception.assignment import iou
from app.perception.identity_eval import evaluate_identity

BOX_IOU = 0.5
TIME_SEC = 0.2
SHORT_EVENT_SEC = 1.5
STEP_IOU = 0.3


def _box(bbox: dict) -> tuple[float, float, float, float]:
    return (float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"]))


def _boxed(document: dict) -> list[dict]:
    """The display set. Raw rows are inspection data and are never scored."""
    return [
        row
        for row in document.get("observations") or []
        if row.get("bbox") and row.get("visibility", "visible") != "absent"
    ]


def detection(predicted: dict, truth: dict) -> dict:
    """Box recall and mean IoU, one-to-one per time. Identity is not used here."""
    pred = _boxed(predicted)
    gold = _boxed(truth)
    pairs = []
    for i, left in enumerate(pred):
        for j, right in enumerate(gold):
            if abs(float(left["timestamp"]) - float(right["timestamp"])) > TIME_SEC:
                continue
            pairs.append((iou(_box(left["bbox"]), _box(right["bbox"])), i, j))
    matches = greedy_match(pairs, BOX_IOU)
    scores = [score for _i, _j, score in matches]
    return {
        **prf(len(matches), len(pred) - len(matches), len(gold) - len(matches)),
        "mean_box_iou": round(sum(scores) / len(scores), 4) if scores else None,
    }


def short_events(predicted: dict, truth: dict) -> dict:
    """Recall of ground-truth actions no longer than `SHORT_EVENT_SEC`."""
    short = [item for item in truth.get("actions") or [] if float(item["end"]) - float(item["start"]) <= SHORT_EVENT_SEC]
    found = 0
    for gold in short:
        if any(
            interval_iou(float(item["start"]), float(item["end"]), float(gold["start"]), float(gold["end"])) > 0
            and item.get("action_type") == gold.get("action_type")
            for item in predicted.get("actions") or []
        ):
            found += 1
    return {"truth": len(short), "found": found, "recall": round(found / len(short), 4) if short else None}


def workflow(predicted: dict, truth: dict) -> dict:
    pred = predicted.get("workflow_steps") or []
    gold = truth.get("workflow_steps") or []
    pairs = [
        (interval_iou(float(a["time_start"]), float(a["time_end"]), float(b["time_start"]), float(b["time_end"])), i, j)
        for i, a in enumerate(pred)
        for j, b in enumerate(gold)
    ]
    matches = greedy_match(pairs, STEP_IOU)
    return prf(len(matches), len(pred) - len(matches), len(gold) - len(matches))


def _overlapped_instance(rows: list[dict], gold: list[dict]) -> str | None:
    """The annotated instance these boxes overlap most often."""
    votes: dict[str, int] = {}
    for row in rows:
        for other in gold:
            if abs(float(row["timestamp"]) - float(other["timestamp"])) > TIME_SEC or not other.get("instance_id"):
                continue
            if iou(_box(row["bbox"]), _box(other["bbox"])) >= BOX_IOU:
                votes[other["instance_id"]] = votes.get(other["instance_id"], 0) + 1
    return max(votes, key=lambda key: (votes[key], key)) if votes else None


def grounding(predicted: dict, truth: dict, mapping: dict[str, str]) -> dict:
    """Bound objects whose instance maps to the annotated instance their own boxes overlap.

    Objects are paired through their observations, so a label is never the evidence.
    """
    gold = _boxed(truth)
    rows = _boxed(predicted)
    bound = [obj for obj in predicted.get("objects") or [] if obj.get("instance_id")]
    correct = 0
    for obj in bound:
        expected = _overlapped_instance([row for row in rows if row["object_local_id"] == obj["local_id"]], gold)
        if expected is not None and mapping.get(obj["instance_id"]) == expected:
            correct += 1
    return {
        "bound_objects": len(bound),
        "correct": correct,
        "precision": round(correct / len(bound), 4) if bound else None,
        "truth_instances": len({row["instance_id"] for row in gold if row.get("instance_id")}),
    }


def score_case(predicted: dict, truth: dict, meta: dict | None = None) -> dict:
    meta = meta or {}
    documents = evaluate_documents(predicted, truth)
    identity = documents["identity"] if documents.get("identity") else evaluate_identity(predicted, truth)
    mapping = identity.get("id_mapping", {}) if identity.get("evaluated") else {}
    source = meta.get("source", "synthetic")
    return {
        "source": source,
        "is_real_world_evidence": source == "human",
        "detection": detection(predicted, truth),
        "identity": {
            key: identity.get(key)
            for key in ("evaluated", "reason", "idf1", "identity_switches", "false_merges", "tracking_coverage")
            if key in identity
        },
        "occlusion_recovery": {
            "opportunities": identity.get("reid_opportunities"),
            "recovered": identity.get("reid_recovered"),
        },
        "short_events": short_events(predicted, truth),
        "grounding": grounding(predicted, truth, mapping),
        "workflow": workflow(predicted, truth),
        "actions": documents["actions"],
        "latency_sec": meta.get("latency_sec"),
        "model_calls": meta.get("model_calls"),
        "semantic_calls": meta.get("semantic_calls"),
    }


def discover(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "truth.json").is_file() and (path / "predicted.json").is_file())


def run(root: Path) -> dict:
    cases = {}
    for folder in discover(root):
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        cases[folder.name] = score_case(
            json.loads((folder / "predicted.json").read_text()),
            json.loads((folder / "truth.json").read_text()),
            meta,
        )
    human = [name for name, item in cases.items() if item["is_real_world_evidence"]]
    return {
        "cases": cases,
        "human_annotated_cases": human,
        "note": (
            "No human-annotated case was found. Nothing here measures real-world accuracy."
            if not human
            else "Only the human-annotated cases are real-world evidence."
        ),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/benchmark")
    print(json.dumps(run(target), indent=2))
