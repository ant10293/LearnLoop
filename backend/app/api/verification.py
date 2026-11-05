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


class StepEdit(BaseModel):
    step_id: str
    required: bool | None = None
    requirement_status: Literal["observed", "required", "optional", "uncertain"] | None = None
    uncertainty_notes: str | None = None
    name: str | None = None
    expected_action: str | None = None
    completion_text: str | None = None
    removed: bool | None = None
    completion_conditions: list[dict] | None = None
    order_after: list[str] | None = None


class RoleEdit(BaseModel):
    role_id: str
    cardinality: Literal["one", "group"] | None = None
    label: str | None = None
    # The reviewer answered which part this is. With a label from the role's look-alikes, the role becomes that part.
    confirm_identity: bool | None = None


class TemplateEdit(BaseModel):
    name: str | None = None
    acceptance_status: Literal["draft", "accepted"] | None = None
    steps: list[StepEdit] = Field(default_factory=list)
    roles: list[RoleEdit] = Field(default_factory=list)


class HumanObservationBody(BaseModel):
    answer: Literal["installed", "not_installed", "cannot_determine"]
    timestamp: float
    reviewer: str = ""
    predicate_id: str = ""
    note: str = ""


def _call(action):
    try:
        return action()
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/api/workflow-templates/{template_id}")
def get_template(template_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.get_template(session, template_id))


@router.patch("/api/workflow-templates/{template_id}")
def patch_template(template_id: str, payload: TemplateEdit, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.revise_template(session, template_id, payload.model_dump()))


@router.get("/api/verifications/{verification_id}")
def get_verification(verification_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.get_verification(session, verification_id))


@router.get("/api/verifications/{verification_id}/export")
def export_verification(verification_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: store.export_verification(session, verification_id))


@router.post("/api/verifications/{verification_id}/steps/{step_id}/observations")
def human_observation(
    verification_id: str,
    step_id: str,
    payload: HumanObservationBody,
    session: Session = Depends(db),
) -> dict:
    return _call(lambda: store.record_observation(session, verification_id, step_id, payload.model_dump()))


