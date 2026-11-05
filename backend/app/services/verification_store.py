"""Persist learned templates and verification reports.

Reports reference an immutable template id and an analysis revision.
They do not replace either document.
"""

from __future__ import annotations

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from app.models.media import Project, VideoAsset
from app.models.product import HumanObservation, ProcedureRecording
from app.models.reliability import ReliabilityRun
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
from app.perception import appearance as app_
from app.services.appearance import ensure_gallery, gallery_status
from app.verification.engine import verify_execution
from app.verification.learn import learn_template
from app.verification.reinspect import plan_reinspections
from app.verification.schema import (
    CompletionPredicate,
    LearnedTemplate,
    RoleLookAlike,
    TemplateDependency,
)


def create_template(
    session: Session,
    project_id: str,
    run_id: str,
    *,
    name: str = "",
    promote_supported: bool = False,
    accept: bool = False,
    results_first: bool = False,
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
        results_first=results_first,
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
    # References are learned with the procedure, with nothing more for a person to do. A failure leaves the procedure.
    try:
        ensure_gallery(session, run.id)
    except Exception:
        session.rollback()
    return template_payload(row, video, session)


def list_templates(session: Session, project_id: str) -> list[dict]:
    _project(session, project_id)
    rows = session.scalars(
        select(LearnedWorkflowTemplate)
        .where(LearnedWorkflowTemplate.project_id == project_id)
        .order_by(LearnedWorkflowTemplate.created_at)
    ).all()
    return [template_payload(row, session.get(VideoAsset, row.source_video_id), session) for row in rows]


def get_template(session: Session, template_id: str) -> dict:
    return template_payload(_template(session, template_id), None, session)


def delete_template_family(session: Session, template_id: str) -> LearnedWorkflowTemplate:
    """Remove every version of a learned procedure and the reports made against them."""
    current = _template(session, template_id)
    family = session.scalars(select(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.family_id == current.family_id)).all()
    reports = list(
        session.scalars(select(VerificationRunRow.id).where(VerificationRunRow.template_id.in_([row.id for row in family])))
    )
    _delete_reports(session, reports)
    for row in family:
        session.delete(row)
    session.commit()
    return current


def delete_procedure_records(session: Session, project_id: str) -> None:
    """Remove a procedure's learned versions, reports, and recording links. Its videos and runs are left to the caller."""
    _project(session, project_id)
    _delete_reports(session, list(session.scalars(select(VerificationRunRow.id).where(VerificationRunRow.project_id == project_id))))
    session.execute(delete(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.project_id == project_id))
    session.execute(delete(ProcedureRecording).where(ProcedureRecording.project_id == project_id))
    session.commit()


def _delete_reports(session: Session, reports: list[str]) -> None:
    if not reports:
        return
    for model in (
        StepVerification,
        VerificationEvidence,
        VerificationRoleMapping,
        VerificationReinspection,
        VerificationGroundTruth,
        HumanObservation,
    ):
        session.execute(delete(model).where(model.verification_id.in_(reports)))
    session.execute(delete(VerificationRunRow).where(VerificationRunRow.id.in_(reports)))


CONTENT_FIELDS = ("name", "expected_action", "completion_text", "removed", "completion_conditions", "required")


def revise_template(session: Session, template_id: str, patch: dict) -> dict:
    """Every edit is a new version. The model's version stays as it was."""
    current = _template(session, template_id)
    document = LearnedTemplate.model_validate(current.document_json)
    log: list[dict] = []
    if patch.get("name") and patch["name"] != document.name:
        log.append(_edit("name", "", document.name, patch["name"]))
        document.name = patch["name"]
    steps = {step.step_id: step for step in document.steps}
    roles = {role.role_id: role for role in document.roles}
    for change in patch.get("steps") or []:
        step = steps.get(change.get("step_id") or "")
        if step is None:
            continue
        _revise_step(step, change, roles, log)
    order_changes = [change for change in patch.get("steps") or [] if change.get("order_after") is not None and change.get("step_id") in steps]
    for change in order_changes:
        _revise_order(steps[change["step_id"]], change["order_after"], steps, log)
    if order_changes:
        _reject_order_cycle(document.steps)
    for change in patch.get("roles") or []:
        role = roles.get(change.get("role_id") or "")
        if role is None:
            continue
        if change.get("cardinality") in {"one", "group"} and change["cardinality"] != role.cardinality:
            log.append(_edit("cardinality", role.role_id, role.cardinality, change["cardinality"]))
            role.cardinality = change["cardinality"]
        if change.get("label") and change["label"] != role.label:
            log.append(_edit("label", role.role_id, role.label, change["label"]))
            if change.get("confirm_identity"):
                _swap_look_alike(role, change["label"])
            role.original_label = role.original_label or role.label
            role.label = change["label"]
        if change.get("confirm_identity") and not role.identity_confirmed:
            log.append(_edit("identity_confirmed", role.role_id, False, True))
            role.identity_confirmed = True
            role.uncertain = False
    if log:
        # Changed content has not been accepted, whatever the parent's status was.
        document.acceptance_status = "draft"
    if patch.get("acceptance_status") in {"draft", "accepted"}:
        if patch["acceptance_status"] != document.acceptance_status:
            log.append(_edit("acceptance_status", "", document.acceptance_status, patch["acceptance_status"]))
        document.acceptance_status = patch["acceptance_status"]
    document.review_log = [*document.review_log, *[{**item, "from_version": current.version} for item in log]]
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


def _swap_look_alike(role, label: str) -> None:
    """The reviewer picked a look-alike: its distinguishing feature now describes the role, and the old label becomes the look-alike."""
    picked = next((item for item in role.look_alikes if item.label == label), None)
    if picked is None:
        return
    role.look_alikes = [
        *[item for item in role.look_alikes if item is not picked],
        RoleLookAlike(label=role.label, difference=role.visual_characteristics),
    ]
    if picked.difference:
        role.visual_characteristics = picked.difference


def _edit(field: str, target: str, before, after) -> dict:
    return {"field": field, "target": target, "before": before, "after": after, "provenance": "HUMAN_CORRECTION"}


def _revise_step(step, change: dict, roles: dict, log: list[dict]) -> None:
    def record(field: str, after) -> None:
        log.append(_edit(field, step.step_id, getattr(step, field), after))
        setattr(step, field, after)
        if field not in step.human_edited:
            step.human_edited.append(field)

    if change.get("requirement_status") in {"observed", "required", "optional", "uncertain"}:
        if change["requirement_status"] != step.requirement_status:
            record("requirement_status", change["requirement_status"])
        step.required = step.requirement_status == "required"
    elif change.get("required") is not None and bool(change["required"]) != step.required:
        record("required", bool(change["required"]))
    if "uncertainty_notes" in change and change["uncertainty_notes"] is not None:
        step.uncertainty_notes = change["uncertainty_notes"]
    for field in ("name", "expected_action", "completion_text"):
        if change.get(field) and change[field] != getattr(step, field):
            record(field, change[field])
    if change.get("removed") is not None and bool(change["removed"]) != step.removed:
        record("removed", bool(change["removed"]))
    if change.get("completion_conditions") is not None:
        conditions = []
        for index, item in enumerate(change["completion_conditions"]):
            if item.get("subject_role") not in roles or (item.get("target_role") and item["target_role"] not in roles):
                raise ValueError("A completion condition names an object that is not in this procedure.")
            if not item.get("relation") and not item.get("state_label"):
                raise ValueError("A completion condition needs a relation or a state.")
            conditions.append(
                CompletionPredicate(
                    id=item.get("id") or f"human:{index}",
                    type="physical_relationship" if item.get("relation") else "final_state",
                    subject_role=item["subject_role"],
                    target_role=item.get("target_role") or "",
                    relation=item.get("relation") or "",
                    state_label=item.get("state_label") or "",
                    to_state=item.get("state_label") or "",
                    evidence_requirement="required",
                    source_evidence_class="HUMAN_SPECIFIED",
                    context_relations=item.get("context_relations") or [],
                    origin="human",
                )
            )
        log.append(
            _edit(
                "completion_conditions",
                step.step_id,
                [item.model_dump() for item in step.completion_conditions],
                [item.model_dump() for item in conditions],
            )
        )
        step.completion_conditions = conditions
        if "completion_conditions" not in step.human_edited:
            step.human_edited.append("completion_conditions")


def _revise_order(step, order_after: list[str], steps: dict, log: list[dict]) -> None:
    """order_after is the whole set of steps this one must follow. Other steps may fall in any order."""
    wanted = list(dict.fromkeys(order_after))
    if step.step_id in wanted:
        raise ValueError("A step cannot be required to come after itself.")
    if any(step_id not in steps for step_id in wanted):
        raise ValueError("A step order names a step that is not in this procedure.")
    before = [dep.step_id for dep in step.dependencies if dep.constrains_order]
    if set(before) == set(wanted):
        return
    known = {dep.step_id for dep in step.dependencies}
    for dep in step.dependencies:
        enforced = dep.step_id in wanted
        if dep.constrains_order != enforced:
            dep.constrains_order = enforced
            dep.origin = "human"
    step.dependencies.extend(
        TemplateDependency(
            step_id=step_id,
            category="human_specified",
            constrains_order=True,
            evidence_class="HUMAN_SPECIFIED",
            origin="human",
        )
        for step_id in wanted
        if step_id not in known
    )
    log.append(_edit("order_after", step.step_id, before, wanted))
    if "order_after" not in step.human_edited:
        step.human_edited.append("order_after")


def _reject_order_cycle(steps: list) -> None:
    after = {step.step_id: [dep.step_id for dep in step.dependencies if dep.constrains_order] for step in steps}
    settled: set[str] = set()

    def visit(step_id: str, path: set[str]) -> None:
        if step_id in path:
            raise ValueError("These step orders form a loop, so no recording could satisfy them.")
        if step_id in settled:
            return
        for prior in after.get(step_id, []):
            visit(prior, path | {step_id})
        settled.add(step_id)

    for step_id in after:
        visit(step_id, set())


def review_state(document: dict) -> str:
    """ai_draft, human_reviewed, or accepted. Only a reviewer's action moves it off ai_draft."""
    if document.get("acceptance_status") == "accepted":
        return "accepted"
    return "human_reviewed" if document.get("review_log") else "ai_draft"


def start_verification(session: Session, template_id: str, payload: dict) -> dict:
    template = _template(session, template_id)
    if payload.get("require_accepted") and template.acceptance_status != "accepted":
        raise ValueError("Accept the procedure before checking a recording against it.")
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
    # Reliability Lab reports are kept with their experiment and never listed as the procedure's own attempts.
    rows = session.scalars(
        select(VerificationRunRow)
        .where(
            VerificationRunRow.project_id == project_id,
            ~exists().where(ReliabilityRun.analysis_run_id == VerificationRunRow.execution_run_id),
        )
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
        "human_observations": _observations(session, row.id),
        "machine_report": row.machine_report_json,
    }


ANSWERS = {"installed", "not_installed", "cannot_determine"}


def record_observation(session: Session, verification_id: str, step_id: str, payload: dict) -> dict:
    """Store what a reviewer saw and let the verifier read it. The button does not set a verdict."""
    if payload.get("answer") not in ANSWERS:
        raise ValueError("unsupported answer")
    row = _verification(session, verification_id)
    if not any(step.get("step_id") == step_id for step in (row.report_json or {}).get("steps") or []):
        raise LookupError("step not found")
    run = session.get(AnalysisRun, row.execution_run_id)
    revision = session.get(AnalysisRevision, row.execution_revision_id)
    template = _template(session, row.template_id)
    if run is None or revision is None:
        raise LookupError("execution analysis not found")
    session.add(
        HumanObservation(
            verification_id=row.id,
            step_id=step_id,
            predicate_id=payload.get("predicate_id") or "",
            answer=payload["answer"],
            timestamp=float(payload["timestamp"]),
            video_id=row.execution_video_id,
            revision_id=row.execution_revision_id,
            reviewer=(payload.get("reviewer") or "").strip(),
            note=payload.get("note") or "",
        )
    )
    session.flush()
    previous = {step["step_id"]: step for step in (row.report_json or {}).get("steps") or []}
    report = _run_engine(template, revision, run, row.config_json or {}, _observations(session, row.id), _findings(session, row.id), _seen_when_made(row))
    for step in report["steps"]:
        old = previous.get(step["step_id"]) or {}
        history = list(old.get("history") or [])
        if old.get("verdict") != step["verdict"]:
            history.append(
                {
                    "verdict": old.get("verdict"),
                    "error_category": old.get("error_category"),
                    "provenance": "DETERMINISTIC",
                    "note": "Verdict before a reviewer observation.",
                }
            )
        step["history"] = history
    report["verification_id"] = row.id
    report["template_id"] = template.id
    report["template_version"] = template.version
    report["execution_run_id"] = run.id
    report["execution_revision_id"] = revision.id
    _carry_checks(row.report_json or {}, report)
    row.report_json = report
    row.overall_result = report["overall"]
    _replace_details(session, row, report)
    session.commit()
    return verification_payload(session, row)


def _carry_checks(previous: dict, report: dict) -> None:
    """Keep what the automatic checks recorded when a reviewer's action makes the report again. The engine does not
    make these, so without this a pending second look would vanish from the page mid-run."""
    for key in ("appearance", "reinspection", "reinspections"):
        if key in previous:
            report[key] = previous[key]


def _observations(session: Session, verification_id: str) -> list[dict]:
    rows = session.scalars(
        select(HumanObservation)
        .where(HumanObservation.verification_id == verification_id)
        .order_by(HumanObservation.created_at)
    ).all()
    return [
        {
            "id": row.id,
            "step_id": row.step_id,
            "predicate_id": row.predicate_id,
            "answer": row.answer,
            "timestamp": row.timestamp,
            "video_id": row.video_id,
            "revision_id": row.revision_id,
            "reviewer": row.reviewer,
            "note": row.note,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def template_payload(row: LearnedWorkflowTemplate, video: VideoAsset | None = None, session: Session | None = None) -> dict:
    appearance = gallery_status(session, row.source_run_id) if session is not None else app_.gallery_summary(app_.read_gallery(row.source_run_id)[0])
    return {
        "appearance": appearance,
        "id": row.id,
        "project_id": row.project_id,
        "family_id": row.family_id,
        "version": row.version,
        "parent_id": row.parent_id,
        "name": row.name,
        "acceptance_status": row.acceptance_status,
        "review_state": review_state(row.document_json or {}),
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
    payload["human_observations"] = _observations(session, row.id)
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
    seen = _appearance_seen(run.id)
    report = _run_engine(template, revision, run, row.config_json or {}, _observations(session, row.id), seen=seen)
    report["verification_id"] = row.id
    report["template_id"] = template.id
    report["template_version"] = template.version
    report["execution_run_id"] = run.id
    report["execution_revision_id"] = revision.id
    report["appearance"] = {**_appearance_evidence(session, template, run), "seen": sorted(seen)}
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
    if not row.machine_report_json:
        row.machine_report_json = report
    from app.services.reinspection import schedule

    # The first report is kept as made. A second look, when it runs, writes a revision of report_json.
    report = {**report, "reinspection": schedule(session, row, report)}
    row.report_json = report
    row.overall_result = report.get("overall") or "INCOMPLETE"
    row.status = "completed"
    row.finished_at = utcnow()
    row.error = ""
    _replace_details(session, row, report)


def apply_reinspections(session: Session, row: VerificationRunRow, outcome: dict) -> None:
    """Make the report again with every answered second look, keeping verdicts before it in each step's history.
    A reviewer's observation or override still comes after it."""
    template = _template(session, row.template_id)
    run = session.get(AnalysisRun, row.execution_run_id)
    revision = session.get(AnalysisRevision, row.execution_revision_id)
    previous = dict(row.report_json or {})
    block = {
        **(previous.get("reinspection") or {}),
        "status": outcome.get("status") or "completed",
        "reason": outcome.get("reason") or "",
        "outcomes": outcome.get("outcomes") or [],
        "finished_at": utcnow().isoformat(),
    }
    if run is None or revision is None:
        row.report_json = {**previous, "reinspection": {**block, "status": "failed", "reason": "The attempt's analysis is missing."}}
        session.commit()
        return
    overridden = {step["step_id"]: step for step in previous.get("steps") or [] if step.get("basis") == "human_override"}
    report = _run_engine(template, revision, run, row.config_json or {}, _observations(session, row.id), _findings(session, row.id), _seen_when_made(row))
    old = {step["step_id"]: step for step in previous.get("steps") or []}
    changed = []
    steps = []
    for step in report["steps"]:
        before = old.get(step["step_id"]) or {}
        if step["step_id"] in overridden:
            # A reviewer set this verdict. A second look does not replace it.
            steps.append(overridden[step["step_id"]])
            continue
        history = list(before.get("history") or [])
        if before.get("verdict") != step["verdict"]:
            history.append(
                {
                    "verdict": before.get("verdict"),
                    "error_category": before.get("error_category"),
                    "provenance": "DETERMINISTIC",
                    "note": "Verdict before an automatic second look.",
                }
            )
            changed.append({"step_id": step["step_id"], "before": before.get("verdict"), "after": step["verdict"]})
        step["history"] = history
        steps.append(step)
    from app.verification.reinspect import _overall, _score

    report["steps"] = steps
    report["overall"] = _overall(steps)
    report["score"] = _score(steps)
    for key in ("verification_id", "template_id", "template_version", "execution_run_id", "execution_revision_id", "appearance"):
        if key in previous:
            report[key] = previous[key]
    report["reinspection"] = {**block, "changed": changed, "overall_before": previous.get("overall")}
    row.report_json = report
    row.overall_result = report["overall"]
    _replace_details(session, row, report)
    session.commit()


def _findings(session: Session, verification_id: str) -> list[dict]:
    rows = session.scalars(
        select(VerificationReinspection)
        .where(VerificationReinspection.verification_id == verification_id, VerificationReinspection.status == "answered")
        .order_by(VerificationReinspection.created_at)
    ).all()
    return [finding for item in rows for finding in (item.output_json or {}).get("findings") or []]


def _appearance_evidence(session: Session, template: LearnedWorkflowTemplate, run: AnalysisRun) -> dict:
    """What the appearance check of this attempt found, by role. It is identity evidence only: it cannot pass or fail a step."""
    from app.pipeline.object_views import read_manifest

    gallery = gallery_status(session, template.source_run_id)
    manifest = read_manifest(run.id) or {}
    checked = manifest.get("appearance") or {}
    roles = {}
    if checked.get("used"):
        for local_id, tally in (checked.get("by_object") or {}).items():
            total = sum(tally.values()) or 1
            roles[local_id] = {
                "validated_share": round((tally.get("validated", 0) + tally.get("recovered", 0)) / total, 3),
                "rejected_share": round(tally.get("rejected", 0) / total, 3),
                "ambiguous_share": round(tally.get("ambiguous", 0) / total, 3),
                "sightings": total,
            }
    return {
        "gallery": {key: gallery.get(key) for key in ("status", "gallery_id", "objects_learned", "objects", "reason") if key in gallery},
        "used": bool(checked.get("used")),
        "error": checked.get("error") or "",
        "recovery": manifest.get("recovery") or {},
        "objects": roles,
        "note": "Appearance matches say which box is which part. They never pass or fail a step.",
    }


def _run_engine(
    template: LearnedWorkflowTemplate,
    revision: AnalysisRevision,
    run: AnalysisRun,
    config: dict,
    human: list[dict] | None = None,
    findings: list[dict] | None = None,
    seen: set[str] | None = None,
) -> dict:
    return verify_execution(
        template.document_json,
        revision.document_json,
        proposals=config.get("proposals") or {},
        strategy=config.get("strategy") or "hybrid",
        role_overrides=config.get("role_overrides") or {},
        execution_video_id=run.video_id,
        demonstration_video_id=template.source_video_id,
        analysis_complete=True,
        human_observations=human or [],
        reinspection_findings=findings or [],
        appearance_seen=seen or set(),
    )


def _seen_when_made(row: VerificationRunRow) -> set[str]:
    """The appearance matches the report was first made with. Views made again later do not change a verdict
    when a reviewer acts; a report from before appearance checks had none. A report made before matches were kept
    with it reads the attempt's views, as it did then."""
    appearance = (row.report_json or {}).get("appearance")
    if appearance is None:
        return set()
    if "seen" not in appearance:
        return _appearance_seen(row.execution_run_id)
    return set(appearance["seen"])


def _appearance_seen(run_id: str) -> set[str]:
    """Objects of an attempt with at least one box that matched the procedure's learned appearance."""
    from app.pipeline.object_views import read_manifest

    objects = (read_manifest(run_id) or {}).get("objects") or {}
    return {
        local_id for local_id, item in objects.items() if any(row.get("status") in {"validated", "recovered"} for row in item.get("track") or [])
    }


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
