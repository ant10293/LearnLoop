"""Qualitative difference between a visual-only run and a transcript-assisted run."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.metrics import evaluate_documents
from app.models.domain import GroundTruthDataset
from app.models.runs import AnalysisRevision, AnalysisRun
from app.models.util import elapsed_seconds
from app.schemas.analysis import AnalysisDocument


def compare_audio_runs(session: Session, visual: AnalysisRun, assisted: AnalysisRun) -> dict:
    left = _document(session, visual)
    right = _document(session, assisted)
    report = {
        "accuracy": None,
        "note": "No human ground truth was scored. The lists below are qualitative differences, not accuracy.",
        "actions": _diff(_labels(left.actions), _labels(right.actions)),
        "objects": _diff(_labels(left.objects), _labels(right.objects)),
        "steps": _diff(_labels(left.workflow_steps), _labels(right.workflow_steps)),
        "ambiguities": {"visual": len(left.ambiguities), "audio_assisted": len(right.ambiguities)},
        "latency_sec": {
            "visual": _elapsed(visual),
            "audio_assisted": _elapsed(assisted),
        },
        "cost_usd": {"visual": visual.cost_usd, "audio_assisted": assisted.cost_usd},
        "speech_is_not_completion": True,
        "speech_classifications": _speech_counts(session, assisted.id),
        "workflow_document_changed": False,
    }
    report["workflow_document_changed"] = any(
        report[key]["only_visual"] or report[key]["only_audio_assisted"] for key in ("actions", "objects", "steps")
    )
    truth = session.scalar(select(GroundTruthDataset).where(GroundTruthDataset.run_id == visual.id))
    if truth is not None:
        report["ground_truth"] = {
            "source": "human_ground_truth",
            "visual": evaluate_documents(_raw(session, visual), truth.document_json),
            "audio_assisted": evaluate_documents(_raw(session, assisted), truth.document_json),
        }
        report["note"] = "Ground-truth scores are separate from the qualitative differences."
    return report


def _raw(session: Session, run: AnalysisRun) -> dict:
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    if revision is None:
        return {}
    return revision.document_json or {}


def _speech_counts(session: Session, run_id: str) -> dict:
    from app.models.audio import AudioAlignment

    counts: dict[str, int] = {}
    for row in session.scalars(select(AudioAlignment).where(AudioAlignment.run_id == run_id)):
        counts[row.classification] = counts.get(row.classification, 0) + 1
    return counts


def _document(session: Session, run: AnalysisRun) -> AnalysisDocument:
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    if revision is None:
        return AnalysisDocument()
    return AnalysisDocument.model_validate(revision.document_json)


def _labels(rows) -> set[str]:
    values = set()
    for row in rows:
        text = getattr(row, "description", "") or getattr(row, "label", "") or getattr(row, "action_type", "")
        values.add(f"{row.local_id}:{text}")
    return values


def _diff(left: set[str], right: set[str]) -> dict:
    return {
        "only_visual": sorted(left - right),
        "only_audio_assisted": sorted(right - left),
        "shared": len(left & right),
    }


def _elapsed(run: AnalysisRun) -> float | None:
    if run.started_at and run.finished_at:
        return round(elapsed_seconds(run.started_at, run.finished_at), 3)
    return None
