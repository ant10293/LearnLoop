"""Human identity decisions. They fork a revision and do not erase the previous one."""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument, InstanceRecord
from app.services.corrections import revise


def decide_hypothesis(session, run, hypothesis_id: str, decision: str):
    if decision not in {"confirmed", "rejected"}:
        raise ValueError("A hypothesis can be confirmed or rejected.")

    def mutate(doc: AnalysisDocument) -> None:
        if doc.perception is None:
            raise ValueError("This run has no perception record to correct.")
        found = next((item for item in doc.perception.hypotheses if item.hypothesis_id == hypothesis_id), None)
        if found is None:
            raise ValueError("hypothesis not found")
        previous = found.decision
        hypotheses = [
            item.model_copy(
                update={
                    "decision": decision,
                    "note": item.note + " A person recorded this decision on a new revision.",
                }
            )
            if item.hypothesis_id == hypothesis_id
            else item
            for item in doc.perception.hypotheses
        ]
        corrections = list(doc.perception.manual_corrections)
        corrections.append(
            {
                "hypothesis_id": hypothesis_id,
                "previous_decision": previous,
                "decision": decision,
            }
        )
        references = list(doc.perception.appearance_references)
        instances = list(doc.perception.instances)
        if decision == "confirmed":
            references, instances = _promote_reference(found, references, instances)
        doc.perception = doc.perception.model_copy(
            update={
                "hypotheses": hypotheses,
                "manual_corrections": corrections,
                "appearance_references": references,
                "instances": instances,
            }
        )

    return revise(session, run, f"{decision} {hypothesis_id}", "identity", hypothesis_id, mutate)


def merge_instances(session, run, keep_id: str, drop_id: str):
    if keep_id == drop_id:
        raise ValueError("Choose two different instances.")

    def mutate(doc: AnalysisDocument) -> None:
        _require_instance(doc, keep_id)
        _require_instance(doc, drop_id)
        for obj in doc.objects:
            if obj.instance_id == drop_id:
                obj.instance_id = keep_id
        for obs in doc.observations:
            if obs.track_ref == drop_id:
                obs.track_ref = keep_id
                obs.continuity = "uncertain"
        instances = []
        for item in doc.perception.instances:
            if item.instance_id == drop_id:
                instances.append(item.model_copy(update={"status": "archived", "uncertain": True}))
            else:
                instances.append(item)
        corrections = list(doc.perception.manual_corrections)
        corrections.append({"action": "merge", "keep": keep_id, "drop": drop_id})
        doc.perception = doc.perception.model_copy(
            update={"instances": instances, "manual_corrections": corrections}
        )

    return revise(session, run, f"merged instance {drop_id} into {keep_id}", "identity", keep_id, mutate)


def split_instance(session, run, instance_id: str, timestamps: list[float]):
    def mutate(doc: AnalysisDocument) -> None:
        _require_instance(doc, instance_id)
        new_id = _next_instance(doc)
        wanted = {round(float(item), 2) for item in timestamps}
        moved = 0
        for obs in doc.observations:
            if obs.track_ref == instance_id and round(obs.timestamp, 2) in wanted:
                obs.track_ref = new_id
                obs.continuity = "uncertain"
                moved += 1
        if moved == 0:
            raise ValueError("None of those timestamps belong to the instance.")
        source = next(item for item in doc.perception.instances if item.instance_id == instance_id)
        instances = list(doc.perception.instances)
        instances.append(
            InstanceRecord(
                instance_id=new_id,
                status="tracked",
                first_seen=min(timestamps),
                last_seen=max(timestamps),
                observation_count=moved,
                uncertain=True,
                human_initialized=True,
                detector_id=source.detector_id,
            )
        )
        corrections = list(doc.perception.manual_corrections)
        corrections.append({"action": "split", "from": instance_id, "to": new_id, "timestamps": timestamps})
        doc.perception = doc.perception.model_copy(
            update={"instances": instances, "manual_corrections": corrections}
        )

    return revise(session, run, f"split observations from {instance_id}", "identity", instance_id, mutate)


def _promote_reference(hypothesis, references, instances):
    """A confirmed hypothesis marks that observation trusted. Vectors stay in the embedding cache."""
    promoted = 0
    updated = []
    for item in references:
        same_time = abs(item.timestamp - hypothesis.timestamp) < 0.051
        same_instance = item.instance_id == hypothesis.candidate_instance_id
        if item.state == "candidate" and same_time and same_instance:
            updated.append(item.model_copy(update={"state": "trusted", "confirmed": True}))
            promoted += 1
        else:
            updated.append(item)
    if not promoted:
        return updated, instances
    rows = []
    for item in instances:
        if item.instance_id != hypothesis.candidate_instance_id:
            rows.append(item)
            continue
        rows.append(
            item.model_copy(
                update={
                    "trusted_references": item.trusted_references + promoted,
                    "candidate_references": max(0, item.candidate_references - promoted),
                }
            )
        )
    return updated, rows


def _require_instance(doc: AnalysisDocument, instance_id: str) -> None:
    if doc.perception is None:
        raise ValueError("This run has no perception record to correct.")
    if not any(item.instance_id == instance_id for item in doc.perception.instances):
        raise ValueError(f"Unknown instance {instance_id}.")


def _next_instance(doc: AnalysisDocument) -> str:
    numbers = []
    for item in doc.perception.instances:
        if item.instance_id.startswith("track_"):
            try:
                numbers.append(int(item.instance_id.split("_", 1)[1]))
            except ValueError:
                continue
    return f"track_{(max(numbers) if numbers else 0) + 1}"
