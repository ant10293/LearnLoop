"""Metrics use interval overlap, not list position."""

from app.evaluation.metrics import evaluate_documents, interval_iou


def test_interval_iou_overlap():
    assert interval_iou(0, 2, 1, 3) == pytest_approx(1 / 3)


def test_swapped_lists_still_match():
    truth = {
        "actions": [
            {"local_id": "A", "action_type": "pick_up", "start": 0.0, "end": 1.0, "description": "lift"},
            {"local_id": "B", "action_type": "put_down", "start": 2.0, "end": 3.0, "description": "set"},
        ],
        "objects": [{"local_id": "Object_1", "label": "red block"}],
        "workflow_steps": [],
        "dependencies": [],
    }
    predicted = {
        "actions": [
            {"local_id": "X", "action_type": "put_down", "start": 2.1, "end": 3.1, "description": "set down"},
            {"local_id": "Y", "action_type": "pick_up", "start": 0.1, "end": 0.9, "description": "lifted"},
        ],
        "objects": [{"local_id": "Obj", "label": "red block"}],
        "workflow_steps": [],
        "dependencies": [],
    }
    report = evaluate_documents(predicted, truth, {"action_iou": 0.3})
    assert report["actions"]["tp"] == 2
    assert report["actions"]["fp"] == 0
    assert report["actions"]["fn"] == 0


def pytest_approx(value: float):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < 1e-6

    return _Approx()
