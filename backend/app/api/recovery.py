"""Targeted object recovery endpoints. None of them touches a run's stages or jobs."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models.recovery import RecoveryExperiment
from app.models.runs import AnalysisRun
from app.pipeline.lineage import load_lineage
from app.services import recovery, recovery_diagnostics, recovery_reacquisition

router = APIRouter(prefix="/api")


def _run(session: Session, run_id: str) -> AnalysisRun:
    run = session.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


def _experiment(session: Session, experiment_id: str) -> RecoveryExperiment:
    row = session.get(RecoveryExperiment, experiment_id)
    if row is None:
        raise HTTPException(404, "recovery experiment not found")
    return row


def _guard(work):
    try:
        return work()
    except recovery.RecoveryConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs/{run_id}/recovery/candidates")
def candidates(run_id: str, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    return _guard(lambda: recovery.list_candidates(session, run))


@router.post("/runs/{run_id}/recovery/dismissals/{target_id}")
def dismiss(run_id: str, target_id: str, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    return _guard(lambda: recovery.dismiss_target(session, run, target_id))


@router.delete("/runs/{run_id}/recovery/dismissals/{target_id}")
def restore(run_id: str, target_id: str, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    return _guard(lambda: recovery.restore_target(session, run, target_id))


@router.get("/runs/{run_id}/recovery/reacquisition")
def reacquisition(run_id: str, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    return _guard(lambda: recovery_reacquisition.reacquisition(session, run))


@router.post("/runs/{run_id}/recovery/reacquisition/plan")
def plan_reacquisition(run_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    """Makes a dry-run experiment from a reviewed step. Nothing is sent, and sending needs the plan's own approval."""
    run = _run(session, run_id)
    row = _guard(lambda: recovery_reacquisition.plan_attempt(session, run, payload))
    return recovery.results(session, row)


@router.get("/runs/{run_id}/recovery/experiments")
def experiments(run_id: str, session: Session = Depends(get_session)) -> list[dict]:
    _run(session, run_id)
    rows = session.scalars(
        select(RecoveryExperiment)
        .where(RecoveryExperiment.run_id == run_id)
        .order_by(RecoveryExperiment.created_at.desc())
    ).all()
    return [recovery.experiment_dict(session, row) for row in rows]


@router.post("/runs/{run_id}/recovery/experiments")
def create(run_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    row = _guard(lambda: recovery.create_experiment(session, run, payload))
    return recovery.results(session, row)


@router.get("/recovery/experiments/{experiment_id}")
def read(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    return recovery.results(session, _experiment(session, experiment_id))


@router.get("/recovery/experiments/{experiment_id}/diagnostics")
def diagnostics(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    return _guard(lambda: recovery_diagnostics.diagnostics(session, _experiment(session, experiment_id)))


@router.patch("/recovery/experiments/{experiment_id}/seeds")
def seeds(experiment_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    row = _experiment(session, experiment_id)
    _guard(lambda: recovery.update_seeds(session, row, payload))
    return recovery.results(session, row)


@router.post("/recovery/experiments/{experiment_id}/dry-run")
def dry_run(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    row = _experiment(session, experiment_id)
    _guard(lambda: recovery.dry_run(session, row))
    return recovery.results(session, row)


@router.post("/recovery/experiments/{experiment_id}/replay")
def replay(experiment_id: str, payload: dict | None = Body(default=None), session: Session = Depends(get_session)) -> dict:
    """Validates stored responses again, with optional `validation` thresholds. It never contacts a provider."""
    row = _experiment(session, experiment_id)
    return _guard(lambda: recovery.replay(session, row, (payload or {}).get("validation")))


@router.post("/recovery/experiments/{experiment_id}/live")
def live(experiment_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    """Needs `confirm: true` and the `plan_hash` of the plan the person approved."""
    row = _experiment(session, experiment_id)
    _guard(lambda: recovery.authorize_live(session, row, payload))
    recovery.start_live(row.id)
    session.expire_all()
    return recovery.results(session, _experiment(session, experiment_id))


@router.post("/recovery/experiments/{experiment_id}/cancel")
def cancel(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    row = _experiment(session, experiment_id)
    recovery.cancel(session, row)
    return recovery.results(session, row)


@router.post("/recovery/experiments/{experiment_id}/proposals/decide")
def decide(experiment_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    row = _experiment(session, experiment_id)
    return _guard(lambda: recovery.decide(session, row, payload))


@router.get("/runs/{run_id}/geometry-lineage")
def geometry_lineage(run_id: str, session: Session = Depends(get_session)) -> dict:
    run = _run(session, run_id)
    return {"run_id": run.id, "segmentation_mode": run.segmentation_mode, "stages": load_lineage(run.id)}
