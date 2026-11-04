"""One repetition's outputs as a comparable record.

Verdict names come from the verifier unchanged, and this is the one place they are
mapped for comparison. An execution failure, a cancelled run, and a report with no
usable verdict are kept apart from every verdict: none of them is INCONCLUSIVE.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.runs import (
    AnalysisRevision,
    AnalysisRun,
    ModelRequestRow,
    ModelResponseRow,
    PipelineStageRun,
)
from app.models.verification import VerificationReinspection, VerificationRunRow

# The report's overall verdict, as compared across runs. INCOMPLETE and NOT_EVALUATED are not verdicts:
# the verifier did not get a finished analysis, so they are counted as invalid results.
OVERALL = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE", "CANNOT_VERIFY": "INCONCLUSIVE"}
# A step's verdict. CANNOT_VERIFY is the step-level form of an inconclusive result.
STEP = {"PASS": "PASS", "FAIL": "FAIL", "CANNOT_VERIFY": "CANNOT_VERIFY", "NOT_APPLICABLE": "NOT_APPLICABLE"}

# What a finished repetition contributes. Only "verdict" outcomes enter verdict statistics.
VERDICT, EXECUTION_ERROR, CANCELED, INVALID = "verdict", "execution_error", "canceled", "invalid_result"


def overall_verdict(raw: str | None) -> str | None:
    return OVERALL.get(str(raw or "").upper())


def step_verdict(raw: str | None) -> str | None:
    return STEP.get(str(raw or "").upper())


def classify_error(text: str) -> str:
    """A coarse, honest label for why a run stopped. Unknown failures stay pipeline_error."""
    lowered = (text or "").lower()
    if "429" in lowered or "rate limit" in lowered or "rate_limit" in lowered or "too many requests" in lowered:
        return "rate_limited"
    if "out of memory" in lowered or "memoryerror" in lowered or "cannot allocate" in lowered or "no space left" in lowered:
        return "resource_exhausted"
    if "database is locked" in lowered or "database is busy" in lowered:
        return "database_busy"
    if "no longer at" in lowered or "no such file or directory" in lowered:
        return "missing_recording"
    if "worker stopped" in lowered or "process restart" in lowered:
        return "interrupted"
    return "pipeline_error"


def inference_sources(session: Session, run: AnalysisRun, report: dict | None = None) -> dict:
    """Where each model answer in this repetition came from, read from what the run recorded, not assumed."""
    from app.pipeline.object_views import read_manifest

    config = run.config_json or {}
    manifest = read_manifest(run.id) or {}
    recovery = manifest.get("recovery") or {}
    sent = {
        stage: count
        for stage, count in session.execute(
            select(ModelRequestRow.stage, func.count()).where(ModelRequestRow.run_id == run.id).group_by(ModelRequestRow.stage)
        )
    }
    reinspected = []
    for item in (report or {}).get("reinspection", {}).get("outcomes") or []:
        reinspected.append({"step_id": item.get("step_id"), "status": item.get("status"), "source": item.get("source") or ""})
    replayed = (
        int(config.get("inference_source") == "cached")
        + int(manifest.get("replayed_requests") or 0)
        + int(manifest.get("locate_replayed") or 0)
        + sum(1 for item in reinspected if item["source"] == "cached")
    )
    return {
        "cache_policy": config.get("cache_policy") or "shared",
        "fresh": bool(config.get("fresh")),
        "guided_call": config.get("inference_source") or "",
        "object_views_live_requests": int(manifest.get("requests") or 0),
        "object_views_replayed_requests": int(manifest.get("replayed_requests") or 0),
        "locate_live_requests": int(manifest.get("locate_requests") or 0),
        "locate_replayed_requests": int(manifest.get("locate_replayed") or 0),
        "recovery_live_requests": int(recovery.get("requests") or 0),
        "reinspection": reinspected,
        "requests_sent_by_stage": sent,
        "replayed_answers": replayed,
        # True only when this run recorded a live guided answer and replayed no stored answer anywhere.
        "fully_fresh": config.get("inference_source") == "live" and replayed == 0,
    }


def usage(session: Session, run_id: str) -> dict:
    rows = session.execute(
        select(ModelResponseRow.input_tokens, ModelResponseRow.output_tokens, ModelResponseRow.cost_usd, ModelResponseRow.price_known).where(
            ModelResponseRow.run_id == run_id
        )
    ).all()
    requests = session.scalar(select(func.count()).select_from(ModelRequestRow).where(ModelRequestRow.run_id == run_id)) or 0
    failed = session.scalar(
        select(func.count()).select_from(ModelRequestRow).where(ModelRequestRow.run_id == run_id, ModelRequestRow.status == "failed")
    ) or 0
    known = bool(rows) and all(row.price_known for row in rows)
    return {
        "requests": int(requests),
        "failed_requests": int(failed),
        "input_tokens": int(sum(row.input_tokens or 0 for row in rows)),
        "output_tokens": int(sum(row.output_tokens or 0 for row in rows)),
        # A cost is given only when every response had a known price. Otherwise it stays unknown.
        "cost_usd": round(sum(row.cost_usd or 0 for row in rows), 6) if known else None,
    }


def stage_timings(session: Session, run_id: str) -> dict:
    from app.models.util import elapsed_seconds

    out = {}
    for stage in session.scalars(select(PipelineStageRun).where(PipelineStageRun.run_id == run_id)):
        seconds = elapsed_seconds(stage.started_at, stage.finished_at) if stage.started_at and stage.finished_at else None
        out[stage.stage] = {"status": stage.status, "seconds": round(seconds, 3) if seconds is not None else None}
    return out


def object_diagnostics(session: Session, run: AnalysisRun) -> dict:
    """What SAM and the appearance check recorded for this run's objects. Only recorded values are reported."""
    from app.pipeline.object_views import read_manifest

    manifest = read_manifest(run.id)
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    analysed = {item.get("local_id"): item.get("label") or "" for item in ((revision.document_json if revision else {}) or {}).get("objects") or []}
    if manifest is None:
        return {"available": False, "objects": {}, "analysis_objects": len(analysed), "note": "No object views were made for this run."}
    appearance = manifest.get("appearance") or {}
    tallies = appearance.get("by_object") or {}
    segments = appearance.get("segments") or {}
    objects = {}
    for local_id, item in (manifest.get("objects") or {}).items():
        track = item.get("track") or []
        tally = tallies.get(local_id) or {}
        objects[local_id] = {
            "label": analysed.get(local_id, ""),
            "boxes": len(track),
            "seen_cells": item.get("seen_cells", 0),
            "cells": len(item.get("cells") or []),
            "first_seen": item.get("first_seen"),
            "appearance": tally,
            # Separate pieces of one track, as the appearance check counted them. More than one is an interruption.
            "track_segments": segments.get(local_id),
        }
    return {
        "available": True,
        "error": manifest.get("error") or "",
        "analysis_objects": len(analysed),
        "objects": objects,
        "objects_without_boxes": sorted(local_id for local_id, item in objects.items() if item["boxes"] == 0),
        "located_later": manifest.get("located_later") or [],
        "relocated": manifest.get("relocated") or [],
        "appearance_used": bool(appearance.get("used")),
        "appearance_error": appearance.get("error") or "",
        "recovered_sightings": int((manifest.get("recovery") or {}).get("recovered_sightings") or 0),
    }


def _step(step: dict) -> dict:
    evidence = [item for item in step.get("evidence") or [] if item.get("side", "execution") == "execution"]
    stamps = sorted({round(float(item["timestamp"]), 2) for item in evidence if item.get("timestamp") is not None})
    raw = step.get("verdict")
    warnings = []
    if step.get("review_required"):
        warnings.append("review_required")
    if step.get("error_category") == "UNVERIFIABLE_OPERATION":
        warnings.append("unverifiable_operation")
    if step.get("uncertainty_notes"):
        warnings.append("uncertainty_noted")
    return {
        "step_id": step.get("step_id") or "",
        "name": step.get("name") or "",
        "required": bool(step.get("required")),
        "raw_verdict": raw,
        "verdict": step_verdict(raw),
        "basis": step.get("basis") or "",
        "error_category": step.get("error_category") or "",
        "execution_state": step.get("execution_state") or "",
        "evidence_count": len(evidence),
        "evidence_timestamps": stamps,
        "missing_evidence": len(evidence) == 0,
        "observed_objects": list(step.get("matched_object_ids") or []),
        "matched_actions": list(step.get("matched_action_ids") or []),
        "alignment": step.get("alignment") or "",
        "warnings": warnings,
        "reason": step.get("explanation") or "",
        "observed": step.get("observed") or "",
        "expected": step.get("expected") or "",
        "reinspected": any(item.get("reinspected") for item in step.get("predicates") or []),
    }


def verification_result(session: Session, run: AnalysisRun, row: VerificationRunRow | None) -> dict:
    """The comparable record of a finished verification repetition."""
    report = (row.report_json if row else None) or {}
    raw = row.overall_result if row else None
    verdict = overall_verdict(raw)
    steps = [_step(step) for step in report.get("steps") or []]
    outcome = VERDICT if verdict else INVALID
    roles = [
        {"role_id": item.get("role_id"), "label": item.get("label") or "", "status": item.get("status") or "", "objects": item.get("object_local_ids") or []}
        for item in report.get("roles") or []
    ]
    reinspection = report.get("reinspection") or {}
    reinspections = []
    if row is not None:
        for item in session.scalars(select(VerificationReinspection).where(VerificationReinspection.verification_id == row.id)):
            reinspections.append({"step_id": item.step_id, "status": item.status})
    return {
        "outcome": outcome,
        "verdict": verdict,
        "raw_overall": raw,
        "invalid_reason": "" if verdict else f"The report has no usable overall verdict ({raw or 'missing'}).",
        "score": report.get("score") or {},
        "steps": steps,
        "evidence_count": sum(step["evidence_count"] for step in steps),
        "diagnostics": {
            "warnings": list(report.get("warnings") or []),
            "roles": roles,
            "unresolved_roles": [item["role_id"] for item in roles if item["status"] == "unresolved"],
            "ambiguous_roles": [item["role_id"] for item in roles if item["status"] == "ambiguous"],
            "removed_steps": list(report.get("removed_steps") or []),
            "appearance": {key: (report.get("appearance") or {}).get(key) for key in ("used", "error", "seen")},
            "reinspection": {
                "status": reinspection.get("status") or "",
                "reason": reinspection.get("reason") or "",
                "changed": reinspection.get("changed") or [],
                "requests": reinspections,
            },
            "objects": object_diagnostics(session, run),
        },
        "execution_metadata": _metadata(session, run, report, row),
    }


def failure_result(session: Session, run: AnalysisRun) -> dict:
    """What a repetition that did not finish still recorded: its stages, their errors, and what it sent."""
    stages = {
        item.stage: {"status": item.status, "error": item.error}
        for item in session.scalars(select(PipelineStageRun).where(PipelineStageRun.run_id == run.id))
        if item.status in {"failed", "canceled"} or item.error
    }
    return {
        "outcome": CANCELED if run.status == "canceled" else EXECUTION_ERROR,
        "verdict": None,
        "steps": [],
        "diagnostics": {"failed_stages": stages, "error": run.error},
        "execution_metadata": _metadata(session, run, {}, None),
    }


def learning_result(session: Session, run: AnalysisRun, learned: dict) -> dict:
    """The comparable record of a finished learning repetition."""
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    document = (revision.document_json if revision else None) or {}
    validation = session.scalar(select(PipelineStageRun).where(PipelineStageRun.run_id == run.id, PipelineStageRun.stage == "validation"))
    checkpoint = (validation.checkpoint_json if validation else None) or {}
    live_steps = [step for step in learned.get("steps") or [] if not step.get("removed")]
    checkable = [step for step in live_steps if step.get("completion_conditions")]
    errors = checkpoint.get("errors")
    return {
        "outcome": VERDICT,
        "verdict": None,
        "step_count": len(live_steps),
        "required_steps": sum(bool(step.get("required")) for step in live_steps),
        "checkable_steps": len(checkable),
        "role_count": len(learned.get("roles") or []),
        "object_count": len(document.get("objects") or []),
        "action_count": len(document.get("actions") or []),
        "ambiguities": [item.get("description") or "" for item in document.get("ambiguities") or []],
        "notes": list(learned.get("notes") or []),
        "validation": {"errors": errors, "warnings": checkpoint.get("warnings"), "status": validation.status if validation else ""},
        # The existing rules: validation passed, and at least one step has a condition the verifier can check.
        "reviewable": errors == 0 and len(checkable) > 0,
        "diagnostics": {"objects": object_diagnostics(session, run)},
        "execution_metadata": _metadata(session, run, {}, None),
    }


def _metadata(session: Session, run: AnalysisRun, report: dict, row: VerificationRunRow | None) -> dict:
    config = run.config_json or {}
    return {
        "analysis_run_id": run.id,
        "analysis_status": run.status,
        "verification_id": row.id if row else "",
        "template_id": row.template_id if row else "",
        "template_version": row.template_version if row else None,
        "execution_revision_id": row.execution_revision_id if row else "",
        "model_id": run.model_id,
        "segmentation_model_id": config.get("segmentation_model_id") or "",
        "git_revision": run.git_revision,
        "app_version": run.app_version,
        "inference": inference_sources(session, run, report),
        "usage": usage(session, run.id),
        "stages": stage_timings(session, run.id),
        "analysis_started_at": run.started_at.isoformat() if run.started_at else None,
        "analysis_finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
