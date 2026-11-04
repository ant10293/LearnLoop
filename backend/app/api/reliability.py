"""Reliability Lab routes: repeated, independent runs of one input, and their comparison."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.providers.types import UnsupportedMode
from app.reliability import service

router = APIRouter(prefix="/api/reliability", tags=["reliability"])


def db() -> Session:
    yield from get_session()


class GroundTruthBody(BaseModel):
    expected_verdict: Literal["PASS", "FAIL"] | None = None
    step_labels: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class BatchBody(BaseModel):
    mode: Literal["verification", "learning"] = "verification"
    project_id: str
    video_id: str
    template_id: str = ""
    reference_template_id: str = ""
    repetitions: int = 5
    max_concurrency: int = 2
    independent_inference: bool = True
    # Each independent repetition sends its own Meta requests. They are sent only when this is true.
    allow_live: bool = False
    model_id: str = ""
    segmentation_model_id: str | None = None
    ground_truth: GroundTruthBody | None = None
    name: str = ""


def _call(action):
    try:
        return action()
    except service.LiveCallRequired as exc:
        raise HTTPException(409, {"code": "live_call_required", "plan": exc.plan}) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, UnsupportedMode) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/options")
def options(session: Session = Depends(db)) -> dict:
    return service.options(session)


@router.get("/batches")
def list_batches(session: Session = Depends(db)) -> list[dict]:
    return service.list_batches(session)


@router.post("/batches")
def create_batch(payload: BatchBody, session: Session = Depends(db)) -> dict:
    def make() -> dict:
        body = payload.model_dump()
        body["ground_truth"] = payload.ground_truth.model_dump() if payload.ground_truth else {}
        batch = service.create_batch(session, body)
        return service.get_batch(session, batch.id)

    return _call(make)


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: service.get_batch(session, batch_id))


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: service.cancel_batch(session, batch_id))


@router.post("/batches/{batch_id}/runs/{run_id}/cancel")
def cancel_run(batch_id: str, run_id: str, session: Session = Depends(db)) -> dict:
    return _call(lambda: service.cancel_run(session, batch_id, run_id))


@router.put("/batches/{batch_id}/ground-truth")
def set_ground_truth(batch_id: str, payload: GroundTruthBody, session: Session = Depends(db)) -> dict:
    return _call(lambda: service.set_ground_truth(session, batch_id, payload.model_dump()))


@router.get("/batches/{batch_id}/export")
def export(batch_id: str, format: Literal["json", "csv"] = "json", session: Session = Depends(db)):
    if format == "csv":
        text = _call(lambda: service.export_csv(session, batch_id))
        return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="reliability-{batch_id[:8]}.csv"'})
    return _call(lambda: service.get_batch(session, batch_id))
