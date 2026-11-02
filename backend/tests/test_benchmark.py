"""The benchmark harness on synthetic cases. These scores check the harness. They are not accuracy."""

import json
from pathlib import Path

from app.demo.scenarios import SCENARIOS, build_document

from tests.benchmark_harness import run, score_case


def _truth() -> dict:
    document = build_document(SCENARIOS["fixed_camera"]()).model_dump(mode="json")
    for row in document["observations"]:
        row["instance_id"] = f"gt-{row['object_local_id']}"
    return document


def _as_predicted(truth: dict, prefix: str = "inst") -> dict:
    predicted = json.loads(json.dumps(truth))
    for row in predicted["observations"]:
        row["instance_id"] = f"{prefix}-{row['object_local_id']}"
    for obj in predicted["objects"]:
        obj["instance_id"] = f"{prefix}-{obj['local_id']}"
    return predicted


def test_a_perfect_synthetic_prediction_scores_one_and_is_labeled_synthetic():
    truth = _truth()
    report = score_case(_as_predicted(truth), truth, {"source": "synthetic", "model_calls": 1, "semantic_calls": 1})
    assert report["is_real_world_evidence"] is False
    assert report["detection"]["recall"] == 1.0
    assert report["detection"]["mean_box_iou"] == 1.0
    assert report["identity"]["idf1"] == 1.0
    assert report["identity"]["identity_switches"] == 0
    assert report["grounding"]["precision"] == 1.0
    assert report["workflow"]["recall"] == 1.0
    assert report["semantic_calls"] == 1


def test_a_switched_id_and_a_dropped_box_lower_the_scores():
    truth = _truth()
    predicted = _as_predicted(truth)
    boxed = [row for row in predicted["observations"] if row.get("bbox")]
    target = boxed[len(boxed) // 2]["object_local_id"]
    rows = [row for row in predicted["observations"] if row["object_local_id"] == target and row.get("bbox")]
    rows[-1]["instance_id"] = "inst-intruder"
    predicted["observations"].remove(boxed[0])
    report = score_case(predicted, truth)
    assert report["detection"]["recall"] < 1.0
    assert report["identity"]["idf1"] < 1.0
    assert report["identity"]["identity_switches"] >= 1


def test_an_object_bound_to_the_wrong_instance_is_not_grounded():
    truth = _truth()
    predicted = _as_predicted(truth)
    first, second = predicted["objects"][0], predicted["objects"][1]
    first["instance_id"], second["instance_id"] = second["instance_id"], first["instance_id"]
    report = score_case(predicted, truth)
    assert report["grounding"]["correct"] == report["grounding"]["bound_objects"] - 2


def test_without_human_annotation_the_run_says_so(tmp_path: Path):
    truth = _truth()
    case = tmp_path / "synthetic_fixed"
    case.mkdir()
    (case / "truth.json").write_text(json.dumps(truth))
    (case / "predicted.json").write_text(json.dumps(_as_predicted(truth)))
    (case / "meta.json").write_text(json.dumps({"source": "synthetic"}))
    report = run(tmp_path)
    assert list(report["cases"]) == ["synthetic_fixed"]
    assert report["human_annotated_cases"] == []
    assert "No human-annotated case" in report["note"]
