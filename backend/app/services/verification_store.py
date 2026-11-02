"""Persist learned templates and verification reports.

Reports reference an immutable template id and an analysis revision.
They do not replace either document.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.media import Project, VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun
from app.models.util import utcnow
from app.models.verification import (
    LearnedWorkflowTemplate,
    StepVerification,
    VerificationEvidence,
    VerificationGroundTruth,
    VerificationReinspection,
    VerificationRoleMapping,
    VerificationRunRow,
)
from app.verification.engine import verify_execution
from app.verification.evaluate import compare_strategies, evaluate_report
from app.verification.learn import learn_template
from app.verification.reinspect import apply_finding, plan_reinspections
from app.verification.schema import LearnedTemplate


def create_template(
    session: Session,
    project_id: str,
    run_id: str,
    *,
    name: str = "",
    promote_supported: bool = False,
    accept: bool = False,
) -> dict:
    project = _project(session, project_id)
    run, revision = _completed_run(session, run_id, project.id)
    video = session.get(VideoAsset, run.video_id)
    template = learn_template(
        revision.document_json,
        name=name,
        promote_supported=promote_supported,
        accept=accept,
        source_run_id=run.id,
        source_revision_id=revision.id,
        source_video_id=run.video_id,
    )
    row = LearnedWorkflowTemplate(
        project_id=project.id,
        family_id="",
        version=1,
        source_run_id=run.id,
        source_revision_id=revision.id,
        source_video_id=run.video_id,
        name=template.name,
        acceptance_status=template.acceptance_status,
        document_json=template.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()
    row.family_id = row.id
    session.commit()
    return template_payload(row, video)


def list_templates(session: Session, project_id: str) -> list[dict]:
    _project(session, project_id)
    rows = session.scalars(
        select(LearnedWorkflowTemplate)
        .where(LearnedWorkflowTemplate.project_id == project_id)
        .order_by(LearnedWorkflowTemplate.created_at)
    ).all()
    return [template_payload(row) for row in rows]


def get_template(session: Session, template_id: str) -> dict:
    return template_payload(_template(session, template_id))


def revise_template(session: Session, template_id: str, patch: dict) -> dict:
    current = _template(session, template_id)
    document = LearnedTemplate.model_validate(current.document_json)
    if patch.get("name"):
        document.name = patch["name"]
    if patch.get("acceptance_status") in {"draft", "accepted"}:
        document.acceptance_status = patch["acceptance_status"]
    steps = {step.step_id: step for step in document.steps}
    for change in patch.get("steps") or []:
        step = steps.get(change.get("step_id") or "")
        if step is None:
            continue
        if "required" in change:
            step.required = bool(change["required"])
        if change.get("requirement_status") in {"observed", "required", "optional", "uncertain"}:
            step.requirement_status = change["requirement_status"]
            step.required = step.requirement_status == "required"
        if "uncertainty_notes" in change:
            step.uncertainty_notes = change.get("uncertainty_notes") or ""
        if "name" in change and change["name"]:
            step.name = change["name"]
    roles = {role.role_id: role for role in document.roles}
    for change in patch.get("roles") or []:
        role = roles.get(change.get("role_id") or "")
        if role is None:
            continue
        if change.get("cardinality") in {"one", "group"}:
            role.cardinality = change["cardinality"]
        if "label" in change and change["label"]:
            role.label = change["label"]
    version = _next_version(session, current.family_id)
    row = LearnedWorkflowTemplate(
        project_id=current.project_id,
        family_id=current.family_id,
        version=version,
        parent_id=current.id,
        source_run_id=current.source_run_id,
        source_revision_id=current.source_revision_id,
        source_video_id=current.source_video_id,
        name=document.name,
        acceptance_status=document.acceptance_status,
        document_json=document.model_dump(mode="json"),
    )
    session.add(row)
    session.commit()
    return template_payload(row)


def start_verification(session: Session, template_id: str, payload: dict) -> dict:
    template = _template(session, template_id)
    run = session.get(AnalysisRun, payload["execution_run_id"])
    if run is None or run.project_id != template.project_id:
        raise LookupError("execution run not found")
    strategy = payload.get("strategy") or "hybrid"
    if strategy not in {"hybrid", "tracked", "vlm_only", "vlm_align"}:
        raise ValueError("unknown verification strategy")
    config = {
        "strategy": strategy,
        "provider_id": run.provider_id,
        "model_id": run.model_id,
        "perception": (run.config_json or {}).get("perception"),
        "modality": run.modality,
        "reinspect": bool(payload.get("reinspect")),
        "max_reinspections": int(payload.get("max_reinspections") or 4),
        "proposals": payload.get("proposals") or {},
        "role_overrides": {},
    }
    row = VerificationRunRow(
        project_id=template.project_id,
        template_id=template.id,
        template_version=template.version,
        execution_video_id=run.video_id,
        execution_run_id=run.id,
        execution_revision_id=run.active_revision_id or "",
        status="waiting",
        overall_result="NOT_EVALUATED",
        config_json=config,
        report_json={"status": "waiting", "overall": "NOT_EVALUATED", "steps": []},
    )
    session.add(row)
    session.flush()
    if run.status == "completed" and run.active_revision_id:
        _fill(session, row, run)
    elif run.status in {"failed", "canceled"}:
        row.status = "failed"
        row.overall_result = "INCOMPLETE"
        row.error = run.error or f"execution analysis {run.status}"
        row.finished_at = utcnow()
    session.commit()
    return verification_payload(session, row)


def settle_waiting(session: Session, run_id: str) -> int:
    rows = list(
        session.scalars(
            select(VerificationRunRow).where(
                VerificationRunRow.execution_run_id == run_id,
                VerificationRunRow.status == "waiting",
            )
        )
    )
    run = session.get(AnalysisRun, run_id)
    settled = 0
    for row in rows:
        if run is None or run.status != "completed" or not run.active_revision_id:
            if run is not None and run.status in {"failed", "canceled"}:
                row.status = "failed"
                row.overall_result = "INCOMPLETE"
                row.error = run.error or f"execution analysis {run.status}"
                row.finished_at = utcnow()
            continue
        _fill(session, row, run)
        settled += 1
    if rows:
        session.commit()
    return settled


def list_verifications(session: Session, project_id: str) -> list[dict]:
    _project(session, project_id)
    rows = session.scalars(
        select(VerificationRunRow)
        .where(VerificationRunRow.project_id == project_id)
        .order_by(VerificationRunRow.created_at.desc())
    ).all()
    return [verification_payload(session, row, brief=True) for row in rows]


def get_verification(session: Session, verification_id: str) -> dict:
    return verification_payload(session, _verification(session, verification_id))


def list_evidence(session: Session, verification_id: str) -> list[dict]:
    _verification(session, verification_id)
    rows = session.scalars(
        select(VerificationEvidence).where(VerificationEvidence.verification_id == verification_id)
    ).all()
    return [_evidence_payload(row) for row in rows]


def export_verification(session: Session, verification_id: str) -> dict:
    row = _verification(session, verification_id)
    template = _template(session, row.template_id)
    return {
        "schema": "verification_report_1",
        "template_id": template.id,
        "template_version": template.version,
        "template": template.document_json,
        "verification": verification_payload(session, row),
        "evidence": list_evidence(session, row.id),
        "machine_report": row.machine_report_json,
    }


def request_reinspection(session: Session, verification_id: str, payload: dict) -> dict:
    row = _verification(session, verification_id)
    report = dict(row.report_json or {})
    report["reinspections"] = [
        {"step_id": item.step_id, "status": item.status}
        for item in session.scalars(
            select(VerificationReinspection).where(VerificationReinspection.verification_id == row.id)
        )
    ]
    plans = plan_reinspections(
        report,
        max_steps=int((row.config_json or {}).get("max_reinspections") or 4),
        step_id=payload.get("step_id") or "",
    )
    for plan in plans:
        if plan.get("status") == "budget_exhausted":
            continue
        session.add(
            VerificationReinspection(
                verification_id=row.id,
                step_id=plan.get("step_id") or "",
                question=plan.get("question") or "",
                frame_timestamps=plan.get("frame_timestamps") or [],
                status=plan.get("status") or "not_sent",
                output_json={"reason": plan.get("reason") or "", "modality": plan.get("modality") or ""},
                previous_verdict=plan.get("previous_verdict") or "",
            )
        )
    session.commit()
    return {"reinspections": plans, "provider_invoked": False}


def submit_finding(session: Session, verification_id: str, finding: dict) -> dict:
    row = _verification(session, verification_id)
    report = apply_finding(dict(row.report_json or {}), finding)
    report["verification_id"] = row.id
    report["template_id"] = row.template_id
    report["template_version"] = row.template_version
    row.report_json = report
    row.overall_result = report.get("overall") or row.overall_result
    _replace_details(session, row, report)
    session.add(
        VerificationReinspection(
            verification_id=row.id,
            step_id=finding.get("step_id") or "",
            question=finding.get("question") or "",
            frame_timestamps=list(finding.get("timestamps") or []),
            status="recorded",
            output_json={
                "visibility": finding.get("visibility"),
                "relation_supported": finding.get("relation_supported"),
                "evidence_class": finding.get("evidence_class"),
                "note": finding.get("note") or "",
            },
            previous_verdict=(report.get("reinspections") or [{}])[-1].get("previous_verdict") or "",
        )
    )
    session.commit()
    return verification_payload(session, row)


def correct_step(session: Session, verification_id: str, step_id: str, verdict: str, note: str) -> dict:
    if verdict not in {"PASS", "FAIL", "CANNOT_VERIFY", "NOT_APPLICABLE"}:
        raise ValueError("unsupported verdict")
    row = _verification(session, verification_id)
    report = dict(row.report_json or {})
    found = False
    steps = []
    for step in report.get("steps") or []:
        item = dict(step)
        if item.get("step_id") == step_id:
            found = True
            history = list(item.get("history") or [])
            history.append(
                {
                    "verdict": item.get("verdict"),
                    "error_category": item.get("error_category"),
                    "provenance": "DETERMINISTIC",
                    "note": "Verdict before human correction.",
                }
            )
            item["history"] = history
            item["verdict"] = verdict
            item["explanation"] = note or item.get("explanation") or ""
            item["error_category"] = "" if verdict != "FAIL" else (item.get("error_category") or "INCORRECT_PHYSICAL_STATE")
            item["review_required"] = False
        steps.append(item)
    if not found:
        raise LookupError("step not found")
    from app.verification.reinspect import _overall, _score

    report["steps"] = steps
    report["overall"] = _overall(steps)
    report["score"] = _score(steps)
    row.report_json = report
    row.overall_result = report["overall"]
    _replace_details(session, row, report)
    session.commit()
    return verification_payload(session, row)


def correct_role(session: Session, verification_id: str, role_id: str, object_local_id: str) -> dict:
    row = _verification(session, verification_id)
    run = session.get(AnalysisRun, row.execution_run_id)
    revision = session.get(AnalysisRevision, row.execution_revision_id)
    template = _template(session, row.template_id)
    if run is None or revision is None:
        raise LookupError("execution analysis not found")
    config = dict(row.config_json or {})
    overrides = dict(config.get("role_overrides") or {})
    overrides[role_id] = object_local_id
    config["role_overrides"] = overrides
    previous = {step["step_id"]: step for step in (row.report_json or {}).get("steps") or []}
    report = _run_engine(template, revision, run, config)
    for step in report["steps"]:
        old = previous.get(step["step_id"])
        if not old:
            continue
        step["history"] = list(old.get("history") or []) + [
            {
                "verdict": old.get("verdict"),
                "error_category": old.get("error_category"),
                "provenance": "DETERMINISTIC",
                "note": "Verdict before a human role correction.",
            }
        ]
    report["verification_id"] = row.id
    report["template_id"] = template.id
    report["template_version"] = template.version
    row.config_json = config
    row.report_json = report
    row.overall_result = report["overall"]
    _replace_details(session, row, report)
    session.commit()
    return verification_payload(session, row)


def save_ground_truth(session: Session, verification_id: str, payload: dict) -> dict:
    row = _verification(session, verification_id)
    truth = VerificationGroundTruth(
        project_id=row.project_id,
        verification_id=row.id,
        name=payload.get("name") or "Verification labels",
        document_json={"steps": payload.get("steps") or []},
    )
    session.add(truth)
    session.commit()
    return {"id": truth.id, "name": truth.name, "document": truth.document_json}


def evaluate_verification(session: Session, verification_id: str, ground_truth_id: str, strategies: bool) -> dict:
    row = _verification(session, verification_id)
    truth = session.get(VerificationGroundTruth, ground_truth_id)
    if truth is None or truth.verification_id not in {"", row.id}:
        raise LookupError("ground truth not found")
    metrics = evaluate_report(row.report_json or {}, truth.document_json or {})
    compared = None
    if strategies:
        template = _template(session, row.template_id)
        revision = session.get(AnalysisRevision, row.execution_revision_id)
        if revision is None:
            raise LookupError("execution revision not found")
        config = row.config_json or {}
        compared = compare_strategies(
            template.document_json,
            revision.document_json,
            truth.document_json or {},
            proposals=config.get("proposals") or {},
            role_overrides=config.get("role_overrides") or {},
            execution_video_id=row.execution_video_id,
            demonstration_video_id=template.source_video_id,
        )
    return {"metrics": metrics, "strategies": compared}


def template_payload(row: LearnedWorkflowTemplate, video: VideoAsset | None = None) -> dict:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "family_id": row.family_id,
        "version": row.version,
        "parent_id": row.parent_id,
        "name": row.name,
        "acceptance_status": row.acceptance_status,
        "source_run_id": row.source_run_id,
        "source_revision_id": row.source_revision_id,
        "source_video_id": row.source_video_id,
        "source_filename": video.original_filename if video else "",
        "created_at": row.created_at.isoformat(),
        "document": row.document_json,
    }


def verification_payload(session: Session, row: VerificationRunRow, brief: bool = False) -> dict:
    payload = {
        "id": row.id,
        "project_id": row.project_id,
        "template_id": row.template_id,
        "template_version": row.template_version,
        "execution_video_id": row.execution_video_id,
        "execution_run_id": row.execution_run_id,
        "execution_revision_id": row.execution_revision_id,
        "status": row.status,
        "overall": row.overall_result,
        "config": _public_config(row.config_json or {}),
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }
    if brief:
        report = row.report_json or {}
        payload["score"] = report.get("score")
        payload["steps"] = [
            {
                "step_id": step.get("step_id"),
                "name": step.get("name"),
                "verdict": step.get("verdict"),
                "error_category": step.get("error_category"),
            }
            for step in report.get("steps") or []
        ]
        return payload
    payload["report"] = row.report_json
    return payload


def _fill(session: Session, row: VerificationRunRow, run: AnalysisRun) -> None:
    template = _template(session, row.template_id)
    revision = session.get(AnalysisRevision, run.active_revision_id)
    if revision is None:
        row.status = "failed"
        row.error = "execution revision missing"
        row.overall_result = "INCOMPLETE"
        row.finished_at = utcnow()
        return
    row.execution_revision_id = revision.id
    row.execution_video_id = run.video_id
    row.status = "running"
    row.started_at = row.started_at or utcnow()
    report = _run_engine(template, revision, run, row.config_json or {})
    report["verification_id"] = row.id
    report["template_id"] = template.id
    report["template_version"] = template.version
    report["execution_run_id"] = run.id
    report["execution_revision_id"] = revision.id
    if (row.config_json or {}).get("reinspect"):
        plans = plan_reinspections(report, max_steps=int((row.config_json or {}).get("max_reinspections") or 4))
        report["reinspections"] = plans
        for plan in plans:
            session.add(
                VerificationReinspection(
                    verification_id=row.id,
                    step_id=plan.get("step_id") or "",
                    question=plan.get("question") or "",
                    frame_timestamps=plan.get("frame_timestamps") or [],
                    status=plan.get("status") or "not_sent",
                    output_json={"reason": plan.get("reason") or ""},
                    previous_verdict=plan.get("previous_verdict") or "",
                )
            )
    row.report_json = report
    if not row.machine_report_json:
        row.machine_report_json = report
    row.overall_result = report.get("overall") or "INCOMPLETE"
    row.status = "completed"
    row.finished_at = utcnow()
    row.error = ""
    _replace_details(session, row, report)


def _run_engine(template: LearnedWorkflowTemplate, revision: AnalysisRevision, run: AnalysisRun, config: dict) -> dict:
    return verify_execution(
        template.document_json,
        revision.document_json,
        proposals=config.get("proposals") or {},
        strategy=config.get("strategy") or "hybrid",
        role_overrides=config.get("role_overrides") or {},
        execution_video_id=run.video_id,
        demonstration_video_id=template.source_video_id,
        analysis_complete=True,
    )


def _replace_details(session: Session, row: VerificationRunRow, report: dict) -> None:
    for model in (StepVerification, VerificationEvidence, VerificationRoleMapping):
        session.execute(delete(model).where(model.verification_id == row.id))
    for step in report.get("steps") or []:
        session.add(
            StepVerification(
                verification_id=row.id,
                step_id=step.get("step_id") or "",
                execution_state=step.get("execution_state") or "unknown",
                verdict=step.get("verdict") or "CANNOT_VERIFY",
                error_category=step.get("error_category") or "",
                matched_action_ids=step.get("matched_action_ids") or [],
                matched_object_ids=step.get("matched_object_ids") or [],
                explanation=step.get("explanation") or "",
                uncertainty=step.get("uncertainty_notes") or "",
                evidence_json=step.get("evidence") or [],
                history_json=step.get("history") or [],
                review_required=bool(step.get("review_required")),
            )
        )
        for evidence in list(step.get("evidence") or []) + list(step.get("demo_evidence") or []):
            session.add(_evidence_row(row.id, step.get("step_id") or "", evidence))
    for role in report.get("roles") or []:
        session.add(
            VerificationRoleMapping(
                verification_id=row.id,
                role_id=role.get("role_id") or "",
                status=role.get("status") or "unresolved",
                object_local_ids=role.get("object_local_ids") or [],
                score=float(role.get("score") or 0),
                hypotheses_json=role.get("hypotheses") or [],
                note=role.get("note") or "",
            )
        )


def _evidence_row(verification_id: str, step_id: str, evidence: dict) -> VerificationEvidence:
    return VerificationEvidence(
        verification_id=verification_id,
        step_id=step_id,
        video_id=evidence.get("video_id") or "",
        timestamp=evidence.get("timestamp"),
        frame_id=evidence.get("frame_id"),
        object_local_ids=evidence.get("object_local_ids") or [],
        evidence_class=evidence.get("evidence_class") or "UNKNOWN",
        provenance=evidence.get("provenance") or "DETERMINISTIC",
        explanation=evidence.get("observation") or "",
        side=evidence.get("side") or "execution",
        bbox_json=evidence.get("bbox"),
    )


def _evidence_payload(row: VerificationEvidence) -> dict:
    return {
        "id": row.id,
        "verification_id": row.verification_id,
        "step_id": row.step_id,
        "video_id": row.video_id,
        "timestamp": row.timestamp,
        "frame_id": row.frame_id,
        "object_local_ids": row.object_local_ids,
        "evidence_class": row.evidence_class,
        "provenance": row.provenance,
        "explanation": row.explanation,
        "side": row.side,
        "bbox": row.bbox_json,
    }


def _public_config(config: dict) -> dict:
    return {
        "strategy": config.get("strategy") or "hybrid",
        "provider_id": config.get("provider_id") or "",
        "model_id": config.get("model_id") or "",
        "perception": config.get("perception"),
        "modality": config.get("modality") or "",
        "reinspect": bool(config.get("reinspect")),
        "max_reinspections": config.get("max_reinspections") or 4,
        "role_overrides": config.get("role_overrides") or {},
    }


def _project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise LookupError("project not found")
    return project


def _template(session: Session, template_id: str) -> LearnedWorkflowTemplate:
    row = session.get(LearnedWorkflowTemplate, template_id)
    if row is None:
        raise LookupError("workflow template not found")
    return row


def _verification(session: Session, verification_id: str) -> VerificationRunRow:
    row = session.get(VerificationRunRow, verification_id)
    if row is None:
        raise LookupError("verification not found")
    return row


def _completed_run(session: Session, run_id: str, project_id: str) -> tuple[AnalysisRun, AnalysisRevision]:
    run = session.get(AnalysisRun, run_id)
    if run is None or run.project_id != project_id:
        raise LookupError("run not found")
    if run.status != "completed" or not run.active_revision_id:
        raise LookupError("run has no completed analysis")
    revision = session.get(AnalysisRevision, run.active_revision_id)
    if revision is None:
        raise LookupError("analysis revision not found")
    return run, revision


def _next_version(session: Session, family_id: str) -> int:
    current = session.scalar(
        select(LearnedWorkflowTemplate.version)
        .where(LearnedWorkflowTemplate.family_id == family_id)
        .order_by(LearnedWorkflowTemplate.version.desc())
    )
    return int(current or 0) + 1
