"""Human corrections fork a new revision. Raw and validated documents stay put."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import AnnotationRow
from app.models.media import FrameAsset
from app.models.runs import AnalysisRevision, AnalysisRun
from app.pipeline.validate import validate_document
from app.schemas.analysis import (
    ActionPrediction,
    AnalysisDocument,
    ObjectPrediction,
    WorkflowStepPrediction,
)
from app.services.materialize import materialize


def revise(session: Session, run: AnalysisRun, note: str, target: str, local_id: str, mutate: Callable) -> AnalysisRevision:
    source = _active_document(session, run)
    mutate(source)
    video_duration = 0
    if run.video_id:
        from app.models.media import VideoAsset

        video = session.get(VideoAsset, run.video_id)
        video_duration = video.duration_sec if video else 0
    validate_document(source, video_duration)
    parent = run.active_revision_id
    versions = list(
        session.scalars(
            select(AnalysisRevision.version).where(
                AnalysisRevision.run_id == run.id,
                AnalysisRevision.layer == "human",
            )
        )
    )
    revision = AnalysisRevision(
        run_id=run.id,
        layer="human",
        version=(max(versions) if versions else 0) + 1,
        parent_id=parent,
        document_json=source.model_dump(mode="json"),
        note=note,
    )
    session.add(revision)
    session.flush()
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.video_id == run.video_id)))
    materialize(session, run.id, revision.id, source, frames)
    run.active_revision_id = revision.id
    session.add(
        AnnotationRow(
            run_id=run.id,
            revision_id=revision.id,
            target_type=target,
            target_local_id=local_id,
            note=note,
            patch_json={"note": note},
        )
    )
    session.commit()
    return revision


def _active_document(session: Session, run: AnalysisRun) -> AnalysisDocument:
    if not run.active_revision_id:
        raise ValueError("this run has no analysis to edit")
    revision = session.get(AnalysisRevision, run.active_revision_id)
    if revision is None:
        raise ValueError("active revision is missing")
    return AnalysisDocument.model_validate(revision.document_json)


def update_object(session, run, local_id: str, fields: dict) -> AnalysisRevision:
    allowed = {"label", "description", "visual_characteristics", "initial_state", "final_state", "uncertain_identity"}

    def mutate(doc: AnalysisDocument) -> None:
        obj = _find(doc.objects, local_id)
        for key, value in fields.items():
            if key in allowed and value is not None:
                setattr(obj, key, value)

    return revise(session, run, "corrected object", "object", local_id, mutate)


def merge_objects(session, run, keep_id: str, drop_id: str) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        _find(doc.objects, keep_id)
        _find(doc.objects, drop_id)
        doc.objects = [item for item in doc.objects if item.local_id != drop_id]
        for collection in (
            doc.observations,
            doc.states,
            doc.transitions,
        ):
            for item in collection:
                if item.object_local_id == drop_id:
                    item.object_local_id = keep_id
        for item in doc.relationships:
            if item.subject_local_id == drop_id:
                item.subject_local_id = keep_id
            if item.other_local_id == drop_id:
                item.other_local_id = keep_id
        for action in doc.actions:
            action.object_local_ids = [keep_id if item == drop_id else item for item in action.object_local_ids]
        for step in doc.workflow_steps:
            step.required_objects = [keep_id if item == drop_id else item for item in step.required_objects]
        for item in doc.ambiguities:
            item.object_local_ids = [keep_id if value == drop_id else value for value in item.object_local_ids]

    return revise(session, run, f"merged {drop_id} into {keep_id}", "object", keep_id, mutate)


def split_object(session, run, local_id: str, timestamps: list[float], label: str) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        source = _find(doc.objects, local_id)
        new_id = _next_id(doc.objects, "Object_")
        doc.objects.append(
            ObjectPrediction(
                local_id=new_id,
                label=label or f"{source.label} (split)",
                description=source.description,
                visual_characteristics=source.visual_characteristics,
                initial_state="",
                final_state="",
                uncertain_identity=True,
                manipulated=source.manipulated,
            )
        )
        wanted = {round(item, 2) for item in timestamps}
        for obs in doc.observations:
            if obs.object_local_id == local_id and round(obs.timestamp, 2) in wanted:
                obs.object_local_id = new_id
                obs.continuity = "uncertain"

    return revise(session, run, f"split observations from {local_id}", "object", local_id, mutate)


def add_object(session, run, label: str, description: str) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        doc.objects.append(
            ObjectPrediction(
                local_id=_next_id(doc.objects, "Object_"),
                label=label,
                description=description,
                uncertain_identity=True,
            )
        )

    return revise(session, run, "added missed object", "object", label, mutate)


def update_action(session, run, local_id: str, fields: dict) -> AnalysisRevision:
    allowed = {"action_type", "description", "start", "end", "outcome", "initial_state", "final_state", "uncertainty"}

    def mutate(doc: AnalysisDocument) -> None:
        action = _find(doc.actions, local_id)
        for key, value in fields.items():
            if key in allowed and value is not None:
                setattr(action, key, value)

    return revise(session, run, "corrected action", "action", local_id, mutate)


def add_action(session, run, fields: dict) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        doc.actions.append(
            ActionPrediction(
                local_id=_next_id(doc.actions, "Action_"),
                action_type=fields.get("action_type") or "inspect",
                description=fields.get("description") or "",
                start=float(fields.get("start") or 0),
                end=float(fields.get("end") or 0),
                object_local_ids=fields.get("object_local_ids") or [],
                outcome=fields.get("outcome") or "unknown",
                initial_state=fields.get("initial_state") or "",
                final_state=fields.get("final_state") or "",
                uncertainty=float(fields.get("uncertainty") or 0.5),
                evidence=_manual_evidence(float(fields.get("start") or 0), float(fields.get("end") or 0)),
            )
        )

    return revise(session, run, "added missed action", "action", "", mutate)


def update_step(session, run, local_id: str, fields: dict) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        step = _find(doc.workflow_steps, local_id)
        for key in ("name", "completion_condition", "observed_result", "expected_action", "uncertainty"):
            if fields.get(key) is not None:
                setattr(step, key, fields[key])

    return revise(session, run, "corrected workflow step", "step", local_id, mutate)


def add_step(session, run, name: str) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        doc.workflow_steps.append(
            WorkflowStepPrediction(
                local_id=_next_id(doc.workflow_steps, "Step_"),
                name=name,
                evidence=_manual_evidence(0, 0),
            )
        )

    return revise(session, run, "added workflow step", "step", name, mutate)


def add_dependency(session, run, src: str, dst: str, category: str, rationale: str) -> AnalysisRevision:
    from app.schemas.analysis import DependencyPrediction
    from app.schemas.enums import DependencyCategory

    def mutate(doc: AnalysisDocument) -> None:
        doc.dependencies.append(
            DependencyPrediction(
                from_step=src,
                to_step=dst,
                category=DependencyCategory(category),
                rationale=rationale,
                evidence=_manual_evidence(0, 0),
            )
        )

    return revise(session, run, "added dependency", "dependency", f"{src}->{dst}", mutate)


def remove_dependency(session, run, src: str, dst: str) -> AnalysisRevision:
    def mutate(doc: AnalysisDocument) -> None:
        doc.dependencies = [
            item for item in doc.dependencies if not (item.from_step == src and item.to_step == dst)
        ]

    return revise(session, run, "removed dependency", "dependency", f"{src}->{dst}", mutate)


def _find(items, local_id: str):
    for item in items:
        if item.local_id == local_id:
            return item
    raise ValueError(f"unknown id {local_id}")


def _next_id(items, prefix: str) -> str:
    numbers = []
    for item in items:
        if item.local_id.startswith(prefix):
            tail = item.local_id[len(prefix) :]
            if tail.isdigit():
                numbers.append(int(tail))
    return f"{prefix}{(max(numbers) if numbers else 0) + 1}"


def _manual_evidence(start: float, end: float):
    from app.schemas.analysis import Evidence
    from app.schemas.enums import EvidenceClass, Provenance

    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.HUMAN_CORRECTION,
        note="Entered by the researcher.",
        time_start=start,
        time_end=end,
        frame_timestamps=[start] if start or end else [],
    )
