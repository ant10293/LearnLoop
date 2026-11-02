"""Compact tracked-object context for a language model. No embedding vectors."""

from __future__ import annotations

from collections import defaultdict

from app.pipeline.prompts import TRACKED_OBJECTS
from app.schemas.analysis import AnalysisDocument


def _track_facts(document: AnalysisDocument) -> dict[str, dict]:
    facts: dict[str, dict] = defaultdict(lambda: {"detected": 0, "flow": 0, "times": [], "support": ""})
    for row in document.observations:
        if not row.track_ref:
            continue
        item = facts[row.track_ref]
        if row.geometry_origin == "optical_flow":
            item["flow"] += 1
        else:
            item["detected"] += 1
            item["support"] = item["support"] or row.detection_support
        item["times"].append(row.timestamp)
    return facts


def _largest_gap(times: list[float]) -> float:
    ordered = sorted(times)
    return max((right - left for left, right in zip(ordered, ordered[1:], strict=False)), default=0.0)


def evidence_text(document: AnalysisDocument, limit: int = 24) -> str:
    perception = document.perception
    if perception is None or not perception.used_for_identity:
        return ""
    facts = _track_facts(document)
    lines = [TRACKED_OBJECTS, "Tracked instances (image-space measurements; none of them has a name yet):"]
    for instance in perception.instances[:limit]:
        fact = facts.get(instance.instance_id)
        detail = ""
        if fact:
            support = fact["support"] or "unknown"
            detail = (
                f", {fact['detected']} detector hits ({support})"
                f", {fact['flow']} optical-flow holds"
                f", largest unobserved gap {_largest_gap(fact['times']):.2f}s"
            )
        lines.append(
            f"- {instance.instance_id}: status {instance.status}, "
            f"visible {instance.first_seen:.2f}s–{instance.last_seen:.2f}s, "
            f"observations {instance.observation_count}{detail}, "
            f"uncertain {str(instance.uncertain).lower()}."
        )
    if len(perception.instances) > limit:
        lines.append(f"{len(perception.instances) - limit} further instances were omitted for length.")
    continuity = perception.diagnostics.get("tracking_continuity")
    if continuity and continuity != "supported":
        lines.append(
            f"Tracking continuity is {continuity}. Some gaps are longer than an occlusion, "
            "so identity across them is a hypothesis."
        )
    lines.append(
        "A detector hit marked proposal is a candidate region, not a recognized object. "
        "An optical-flow hold moves the last box; it is not a new detection. "
        "A gap is unobserved time, not absence."
    )
    open_hypotheses = [
        item
        for item in perception.hypotheses
        if item.decision in {"proposed", "ambiguous"}
    ][:8]
    if open_hypotheses:
        lines.append("Unconfirmed identity hypotheses (scores are not probabilities):")
        for item in open_hypotheses:
            cosine = "n/a" if item.appearance_cosine is None else f"{item.appearance_cosine:.2f}"
            lines.append(
                f"- {item.hypothesis_id} {item.decision}: {item.subject_instance_id} "
                f"versus {item.candidate_instance_id}, appearance cosine {cosine}, "
                f"association score {item.association_score:.2f}."
            )
    lines.append(
        "Use these instance ids when you refer to a tracked object. "
        "If you describe something that is not in the list, set identity_relation to provisional "
        "and do not reuse an existing instance id."
    )
    return "\n".join(lines)
