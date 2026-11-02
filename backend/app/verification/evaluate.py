"""Score verification reports against human labels.

Agreement with another model is not a metric. The reference is a ground-truth document.
"""

from __future__ import annotations

STRATEGIES = ("vlm_only", "vlm_align", "tracked", "hybrid")


def evaluate_report(report: dict, truth: dict) -> dict:
    gold = {item["step_id"]: item for item in truth.get("steps") or []}
    predicted = {item["step_id"]: item for item in report.get("steps") or []}
    ids = sorted(set(gold) & set(predicted)) if gold else []
    if not ids:
        ids = sorted(set(gold) | set(predicted))
    verdict_tp = verdict_fp = verdict_fn = 0
    error_tp = error_fp = error_fn = 0
    incorrect_pass = incorrect_fail = 0
    abstentions = 0
    timestamp_errors = []
    confusion: dict[str, dict[str, int]] = {}
    observable_correct = observable_total = 0
    hidden_abstain = hidden_total = 0
    hidden_wrong = 0
    for step_id in ids:
        pred = predicted.get(step_id) or {}
        ref = gold.get(step_id) or {}
        pred_verdict = pred.get("verdict") or "CANNOT_VERIFY"
        true_verdict = ref.get("verdict") or ""
        if not true_verdict:
            continue
        if pred_verdict == "CANNOT_VERIFY":
            abstentions += 1
        if pred_verdict == true_verdict:
            verdict_tp += 1
        else:
            if pred_verdict not in {"CANNOT_VERIFY", "NOT_APPLICABLE"}:
                verdict_fp += 1
            verdict_fn += 1
        pred_fail = pred_verdict == "FAIL"
        true_fail = true_verdict == "FAIL"
        if pred_fail and true_fail:
            error_tp += 1
        elif pred_fail and not true_fail:
            error_fp += 1
        elif true_fail and not pred_fail:
            error_fn += 1
        if pred_verdict == "PASS" and true_verdict in {"FAIL", "CANNOT_VERIFY"}:
            incorrect_pass += 1
        if pred_verdict == "FAIL" and true_verdict in {"PASS", "CANNOT_VERIFY"}:
            incorrect_fail += 1
        true_error = ref.get("error_type") or ref.get("error_category") or ""
        pred_error = pred.get("error_category") or ""
        if true_error or pred_error:
            confusion.setdefault(true_error or "(none)", {})
            confusion[true_error or "(none)"][pred_error or "(none)"] = (
                confusion[true_error or "(none)"].get(pred_error or "(none)", 0) + 1
            )
        if ref.get("timestamp") is not None and pred.get("execution_time") is not None:
            timestamp_errors.append(abs(float(pred["execution_time"]) - float(ref["timestamp"])))
        observable = ref.get("observable")
        if observable is True:
            observable_total += 1
            if pred_verdict == true_verdict:
                observable_correct += 1
        elif observable is False:
            hidden_total += 1
            if pred_verdict == "CANNOT_VERIFY":
                hidden_abstain += 1
            elif pred_verdict in {"PASS", "FAIL"} and pred_verdict != true_verdict:
                hidden_wrong += 1
    labeled = verdict_tp + verdict_fn
    return {
        "is_accuracy": True,
        "reference": "human_ground_truth",
        "model_agreement_is_accuracy": False,
        "steps_scored": labeled,
        "verdict": _prf(verdict_tp, verdict_fp, verdict_fn),
        "error_detection": _prf(error_tp, error_fp, error_fn),
        "false_positive_rate": _rate(error_fp, error_fp + sum(1 for step_id in ids if (gold.get(step_id) or {}).get("verdict") != "FAIL")),
        "false_negative_rate": _rate(error_fn, error_fn + error_tp),
        "incorrect_pass": incorrect_pass,
        "incorrect_fail": incorrect_fail,
        "incorrect_pass_rate": _rate(incorrect_pass, labeled),
        "incorrect_fail_rate": _rate(incorrect_fail, labeled),
        "abstention_rate": _rate(abstentions, labeled),
        "timestamp_mean_abs_error": round(sum(timestamp_errors) / len(timestamp_errors), 4) if timestamp_errors else None,
        "error_confusion": confusion,
        "observable_accuracy": _rate(observable_correct, observable_total),
        "unobservable_abstention_rate": _rate(hidden_abstain, hidden_total),
        "unobservable_incorrect": hidden_wrong,
    }


def compare_strategies(template, execution, truth: dict, *, proposals: dict | None = None, **kwargs) -> dict:
    from app.verification.engine import verify_execution

    reports = {}
    for strategy in STRATEGIES:
        report = verify_execution(template, execution, proposals=proposals, strategy=strategy, **kwargs)
        reports[strategy] = {
            "overall": report["overall"],
            "metrics": evaluate_report(report, truth),
        }
    return {
        "is_accuracy": True,
        "reference": "human_ground_truth",
        "model_agreement_is_accuracy": False,
        "note": "Strategies are compared on the same human labels. They are not scored against each other.",
        "strategies": reports,
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
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


def _rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 4)
