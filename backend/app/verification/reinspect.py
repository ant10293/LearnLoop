"""Plan a bounded second look at an unresolved step.

This does not repeat the full analysis and does not call a provider by itself.
Findings are applied only when they carry directly observed evidence.
"""

from __future__ import annotations


def plan_reinspections(report: dict, *, max_steps: int = 4, step_id: str = "") -> list[dict]:
    existing = report.get("reinspections") or []
    budget = max(0, max_steps - len(existing))
    if budget == 0:
        return [
            {
                "step_id": step_id,
                "status": "budget_exhausted",
                "question": "",
                "frame_timestamps": [],
                "reason": "The reinspection budget for this verification is exhausted.",
            }
        ]
    chosen = []
    for step in report.get("steps") or []:
        if step_id and step.get("step_id") != step_id:
            continue
        if step.get("verdict") != "CANNOT_VERIFY" or not step.get("required"):
            continue
        stamps = _timestamps(step)
        chosen.append(
            {
                "step_id": step["step_id"],
                "status": "not_sent",
                "question": _question(step),
                "frame_timestamps": stamps[:8],
                "modality": "sampled_frames",
                "reason": (
                    "No vision-language provider was invoked. "
                    "Submit structured findings, or inspect these timestamps as a short frame sequence."
                ),
                "previous_verdict": step.get("verdict"),
            }
        )
        if len(chosen) >= budget:
            break
    return chosen


def apply_finding(report: dict, finding: dict) -> dict:
    """Return an updated report. The previous step verdict stays in history."""
    updated = {
        **report,
        "steps": [dict(step) for step in report.get("steps") or []],
        "reinspections": list(report.get("reinspections") or []),
    }
    step_id = finding.get("step_id") or ""
    record = {
        "step_id": step_id,
        "status": "recorded",
        "question": finding.get("question") or "",
        "frame_timestamps": list(finding.get("timestamps") or []),
        "output": {
            "visibility": finding.get("visibility") or "",
            "relation_supported": finding.get("relation_supported"),
            "evidence_class": finding.get("evidence_class") or "UNKNOWN",
            "note": finding.get("note") or "",
        },
        "previous_verdict": "",
    }
    for step in updated["steps"]:
        if step.get("step_id") != step_id:
            continue
        record["previous_verdict"] = step.get("verdict")
        history = list(step.get("history") or [])
        history.append(
            {
                "verdict": step.get("verdict"),
                "error_category": step.get("error_category"),
                "provenance": "DETERMINISTIC",
                "note": "Verdict before reinspection.",
            }
        )
        step["history"] = history
        verdict, category, note = _interpret(finding)
        if verdict is None:
            step["explanation"] = (
                (step.get("explanation") or "")
                + " Reinspection did not add directly observed evidence, so the verdict is unchanged."
            ).strip()
            step["review_required"] = True
            break
        step["verdict"] = verdict
        step["error_category"] = category
        step["observed"] = note
        step["explanation"] = note
        step["review_required"] = verdict == "CANNOT_VERIFY"
        stamps = finding.get("timestamps") or []
        if stamps:
            step["execution_time"] = stamps[0]
            evidence = list(step.get("evidence") or [])
            evidence.append(
                {
                    "video_id": report.get("execution_video_id") or "",
                    "timestamp": stamps[0],
                    "observation": note,
                    "evidence_class": "DIRECTLY_OBSERVED",
                    "provenance": "AI_PREDICTION",
                    "side": "execution",
                    "object_local_ids": [],
                    "bbox": None,
                }
            )
            step["evidence"] = evidence
        break
    updated["reinspections"] = list(updated.get("reinspections") or []) + [record]
    updated["overall"] = _overall(updated["steps"])
    updated["score"] = _score(updated["steps"])
    return updated


def _interpret(finding: dict) -> tuple[str | None, str, str]:
    if finding.get("evidence_class") != "DIRECTLY_OBSERVED":
        return None, "", ""
    visibility = finding.get("visibility") or ""
    if visibility in {"occluded", "insufficient", "partial", ""}:
        return "CANNOT_VERIFY", "UNVERIFIABLE_OPERATION", "The reinspected frames do not show the condition clearly enough."
    supported = finding.get("relation_supported")
    if supported is True and visibility == "visible":
        return "PASS", "", "Reinspection directly observed the expected condition."
    if supported is False and visibility == "visible":
        return (
            "FAIL",
            finding.get("error_category") or "MISSING_COMPONENT",
            "Reinspection directly observed that the expected condition does not hold.",
        )
    return "CANNOT_VERIFY", "UNVERIFIABLE_OPERATION", "Reinspection did not establish the condition."


def _question(step: dict) -> str:
    expected = step.get("expected") or step.get("name") or "the learned condition"
    return (
        f"Determine whether this condition is visible: {expected}. "
        "Use only the supplied frames. Identify evidence for it, evidence against it, or insufficient visibility."
    )


def _timestamps(step: dict) -> list[float]:
    stamps = []
    for evidence in step.get("evidence") or []:
        if evidence.get("timestamp") is not None:
            stamps.append(float(evidence["timestamp"]))
    if step.get("execution_time") is not None:
        stamps.append(float(step["execution_time"]))
    if step.get("demo_time_start") is not None:
        stamps.append(float(step["demo_time_start"]))
    if step.get("demo_time_end") is not None:
        stamps.append(float(step["demo_time_end"]))
    unique = []
    for stamp in stamps:
        if stamp not in unique:
            unique.append(stamp)
    return unique


def _overall(steps: list[dict]) -> str:
    required = [step for step in steps if step.get("required") and step.get("verdict") != "NOT_APPLICABLE"]
    if any(step.get("verdict") == "FAIL" for step in required):
        return "FAIL"
    if any(step.get("verdict") == "CANNOT_VERIFY" for step in required):
        return "INCONCLUSIVE"
    if required and all(step.get("verdict") == "PASS" for step in required):
        return "PASS"
    return "INCONCLUSIVE"


def _score(steps: list[dict]) -> dict:
    required = [step for step in steps if step.get("required") and step.get("verdict") != "NOT_APPLICABLE"]
    verified = sum(step.get("verdict") == "PASS" for step in required)
    failed = sum(step.get("verdict") == "FAIL" for step in required)
    unresolved = sum(step.get("verdict") == "CANNOT_VERIFY" for step in required)
    denominator = verified + failed + unresolved
    return {
        "verified": verified,
        "failed": failed,
        "unresolved": unresolved,
        "not_applicable": sum(step.get("verdict") == "NOT_APPLICABLE" for step in steps),
        "value": round(verified / denominator, 4) if denominator else None,
        "explanation": (
            "Value is verified required steps divided by verified, failed, and unresolved required steps. "
            "CANNOT_VERIFY is unresolved and is not counted as success."
        ),
    }
