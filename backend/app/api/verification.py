"""Workflow template and execution verification routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.services import verification_store as store

router = APIRouter(tags=["verification"])


def db() -> Session:
    yield from get_session()


class LearnRequest(BaseModel):
    run_id: str
    name: str = ""
    promote_supported: bool = False
    accept: bool = False


class StepEdit(BaseModel):
    step_id: str
    required: bool | None = None
    requirement_status: Literal["observed", "required", "optional", "uncertain"] | None = None
    uncertainty_notes: str | None = None
    name: str | None = None


class RoleEdit(BaseModel):
    role_id: str
    cardinality: Literal["one", "group"] | None = None
    label: str | None = None


class TemplateEdit(BaseModel):
    name: str | None = None
    acceptance_status: Literal["draft", "accepted"] | None = None
    steps: list[StepEdit] = Field(default_factory=list)
    roles: list[RoleEdit] = Field(default_factory=list)


class VerifyRequest(BaseModel):
    execution_run_id: str
    strategy: Literal["hybrid", "tracked", "vlm_only", "vlm_align"] = "hybrid"
    reinspect: bool = False
    max_reinspections: int = 4
    proposals: dict = Field(default_factory=dict)


class ReinspectRequest(BaseModel):
    step_id: str = ""


class FindingRequest(BaseModel):
    step_id: str
    visibility: str = ""
    relation_supported: bool | None = None
    evidence_class: str = "UNKNOWN"
    timestamps: list[float] = Field(default_factory=list)
    note: str = ""
    error_category: str = ""
    question: str = ""


class StepCorrection(BaseModel):
    verdict: Literal["PASS", "FAIL", "CANNOT_VERIFY", "NOT_APPLICABLE"]
    note: str = ""


class RoleCorrection(BaseModel):
    object_local_id: str


class GroundTruthBody(BaseModel):
    name: str = "Verification labels"
    steps: list[dict] = Field(default_factory=list)


class EvaluateBody(BaseModel):
    ground_truth_id: str
    strategies: bool = False


def _call(action):
    try:
        return action()
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/api/tasks/{project_id}/workflow-templates")
def create_template(project_id: str, payload: LearnRequest, session: Session = Depends(db)) -> dict:
    return _call(
        lambda: store.create_template(
            session,
            project_id,
            payload.run_id,
            name=payload.name,
            promote_supported=payload.promote_supported,
            accept=payload.accept,
        )
    )


@router.get("/api/tasks/{project_id}/workflow-templates")
def list_templates(project_id: str, session: Session = Depends(db)) -> list[dict]:
    return _call(lambda: store.list_templates(session, project_id))


@router.get("/api/tasks/{project_id}/verifications")
def list_verifications(project_id: str, session: Session = Depends(db)) -> list[dict]:
    return _call(lambda: store.list_verifications(session, project_id))


@router.get("/api/workflow-templates/{template_id}")
def get_template(template_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.get_template(session, template_id))


@router.patch("/api/workflow-templates/{template_id}")
def patch_template(template_id: str, payload: TemplateEdit, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.revise_template(session, template_id, payload.model_dump()))


@router.post("/api/workflow-templates/{template_id}/verifications")
def start_verification(template_id: str, payload: VerifyRequest, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.start_verification(session, template_id, payload.model_dump()))


@router.get("/api/verifications/{verification_id}")
def get_verification(verification_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.get_verification(session, verification_id))


@router.get("/api/verifications/{verification_id}/evidence")
def get_evidence(verification_id: str, session: Session = Depends(db)) -> list[dict]:
    return _call(lambda: store.list_evidence(session, verification_id))


@router.get("/api/verifications/{verification_id}/export")
def export_verification(verification_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.export_verification(session, verification_id))


@router.post("/api/verifications/{verification_id}/reinspect")
def reinspect(verification_id: str, payload: ReinspectRequest, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.request_reinspection(session, verification_id, payload.model_dump()))


@router.post("/api/verifications/{verification_id}/findings")
def findings(verification_id: str, payload: FindingRequest, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.submit_finding(session, verification_id, payload.model_dump()))


@router.patch("/api/verifications/{verification_id}/steps/{step_id}")
def correct_step(
    verification_id: str,
    step_id: str,
    payload: StepCorrection,
    session: Session = Depends(db),
) -> dict:
    return _call(lambda: store.correct_step(session, verification_id, step_id, payload.verdict, payload.note))


@router.patch("/api/verifications/{verification_id}/roles/{role_id}")
def correct_role(
    verification_id: str,
    role_id: str,
    payload: RoleCorrection,
    session: Session = Depends(db),
) -> dict:
    return _call(lambda: store.correct_role(session, verification_id, role_id, payload.object_local_id))


@router.post("/api/verifications/{verification_id}/ground-truth")
def ground_truth(verification_id: str, payload: GroundTruthBody, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.save_ground_truth(session, verification_id, payload.model_dump()))


@router.post("/api/verifications/{verification_id}/evaluate")
def evaluate(verification_id: str, payload: EvaluateBody, session: Session = Depends(db)) -> dict:
    return _call(
        lambda: store.evaluate_verification(session, verification_id, payload.ground_truth_id, payload.strategies)
    )
