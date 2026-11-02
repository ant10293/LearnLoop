"""The product surface: procedures, recordings, and one analysis call at a time."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.serialize import video_dict
from app.config import settings
from app.db import get_session
from app.models.media import Project, VideoAsset
from app.models.product import ProcedureRecording
from app.models.runs import AnalysisRun, ModelRequestRow, PipelineStageRun
from app.models.util import new_id
from app.models.verification import LearnedWorkflowTemplate, VerificationRunRow
from app.pipeline import guided, object_views
from app.providers.types import UnsupportedMode
from app.services import verification_store as store
from app.services.ingest import accept_reference, accept_upload, ingest_file, playback_path, referenced_file, schedule_ingest
from app.services.storage import data_path, safe_filename
from app.video.frames import extract_frame
from app.video.probe import VideoError

router = APIRouter(prefix="/api/learnloop", tags=["learnloop"])


def db() -> Session:
    yield from get_session()


class ProcedureBody(BaseModel):
    name: str


class ReferenceBody(BaseModel):
    path: str
    role: str


class AnalyzeBody(BaseModel):
    # False means: use a stored answer or say what a live call would involve.
    allow_live: bool = False
    # Empty uses the default inference model.
    model_id: str = ""
    # None uses the default segmentation model. Empty means no segmentation.
    segmentation_model_id: str | None = None


class LearnBody(BaseModel):
    run_id: str


class VerifyBody(BaseModel):
    run_id: str


@router.get("/status")
def status(session: Session = Depends(db)) -> dict:
    from app.services.credentials import load_secret

    secret, _source = load_secret(session, "meta")
    return {
        "profile": guided.PROFILE,
        "model_id": guided.DEFAULT_MODEL,
        "modality": guided.MODALITY,
        "max_clip_seconds": guided.MAX_CLIP_SECONDS,
        "live_enabled": settings.live_inference,
        "meta_key_configured": bool(secret),
        "providers": ["meta"],
    }


@router.get("/models")
def list_models(session: Session = Depends(db)) -> dict:
    return guided.model_choices(session)


@router.get("/procedures")
def list_procedures(session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(Project).order_by(Project.created_at.desc())).all()
    return [_summary(session, row) for row in rows if _is_procedure(session, row)]


@router.post("/procedures")
def create_procedure(payload: ProcedureBody, session: Session = Depends(db)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Name the procedure.")
    row = Project(name=name[:200], description="learnloop_procedure", is_task=False)
    session.add(row)
    session.commit()
    return _summary(session, row)


@router.get("/procedures/{project_id}")
def get_procedure(project_id: str, session: Session = Depends(db)) -> dict:
    project = _project(session, project_id)
    recordings = _recordings(session, project_id)
    templates = store.list_templates(session, project_id)
    verifications = store.list_verifications(session, project_id)
    return {
        **_summary(session, project),
        "recordings": [_recording(session, item) for item in recordings],
        "templates": templates,
        "template": templates[-1] if templates else None,
        "verifications": verifications,
    }


@router.post("/procedures/{project_id}/recordings")
async def upload_recording(
    project_id: str,
    role: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(db),
) -> dict:
    _project(session, project_id)
    if role not in {"demonstration", "execution"}:
        raise HTTPException(400, "A recording is a demonstration or an execution.")
    temp = data_path("imports", f"upload_{new_id()}_{safe_filename(file.filename or 'video')}")
    temp.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with temp.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "This file is larger than the upload limit.")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(400, "This file is empty.")
        if settings.ingest_inline:
            video = ingest_file(session, project_id, file.filename or "video.mp4", temp)
        else:
            video = accept_upload(session, project_id, file.filename or "video.mp4", temp)
    except VideoError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if temp.exists():
            temp.unlink()
    existing = session.get(ProcedureRecording, video.id)
    if existing is None:
        session.add(ProcedureRecording(video_id=video.id, project_id=project_id, role=role))
    else:
        existing.role = role
    session.commit()
    if not settings.ingest_inline:
        schedule_ingest(video.id)
    return _recording(session, session.get(ProcedureRecording, video.id))


@router.post("/procedures/{project_id}/recordings/reference")
def reference_recording(project_id: str, payload: ReferenceBody, session: Session = Depends(db)) -> dict:
    """Use a file where it already is. Nothing is copied."""
    _project(session, project_id)
    if payload.role not in {"demonstration", "execution"}:
        raise HTTPException(400, "A recording is a demonstration or an execution.")
    try:
        path = referenced_file(payload.path)
        if settings.ingest_inline:
            video = ingest_file(session, project_id, path.name, path, reference=True)
        else:
            video = accept_reference(session, project_id, path)
    except VideoError as exc:
        raise HTTPException(400, str(exc)) from exc
    session.merge(ProcedureRecording(video_id=video.id, project_id=project_id, role=payload.role))
    session.commit()
    if not settings.ingest_inline:
        schedule_ingest(video.id)
    return _recording(session, session.get(ProcedureRecording, video.id))


@router.delete("/recordings/{video_id}")
def delete_recording(video_id: str, session: Session = Depends(db)) -> dict:
    """Remove a recording, its analyses, and reports on it. A procedure's source recording stays."""
    from app.api.routes import delete_run, delete_video

    row = session.get(ProcedureRecording, video_id)
    if row is None:
        raise HTTPException(404, "recording not found")
    learned = session.scalar(select(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.source_video_id == video_id))
    if learned is not None:
        raise HTTPException(409, "The procedure was learned from this recording, so it is kept. Upload a new demonstration to replace it.")
    for report in session.scalars(select(VerificationRunRow).where(VerificationRunRow.execution_video_id == video_id)).all():
        session.delete(report)
    for run in session.scalars(select(AnalysisRun).where(AnalysisRun.video_id == video_id)).all():
        delete_run(run.id, session)
    session.delete(row)
    session.commit()
    return delete_video(video_id, session)


@router.get("/recordings/{video_id}")
def get_recording(video_id: str, session: Session = Depends(db)) -> dict:
    row = session.get(ProcedureRecording, video_id)
    if row is None:
        raise HTTPException(404, "recording not found")
    return _recording(session, row)


@router.post("/recordings/{video_id}/analyze")
def analyze(video_id: str, payload: AnalyzeBody, session: Session = Depends(db)) -> dict:
    """Replay a stored answer, or send one call that the person approved. Never both, never twice."""
    recording = session.get(ProcedureRecording, video_id)
    video = session.get(VideoAsset, video_id)
    if recording is None or video is None:
        raise HTTPException(404, "recording not found")
    if (video.ingest_status or "ready") != "ready":
        raise HTTPException(409, "This recording is still being prepared.")
    if float(video.duration_sec or 0) > guided.MAX_CLIP_SECONDS:
        raise HTTPException(
            400,
            f"This recording is {video.duration_sec:.0f} seconds. LearnLoop supports up to {guided.MAX_CLIP_SECONDS:.0f} seconds.",
        )
    template_row = None
    if recording.role == "execution":
        template_row = _accepted_template(session, recording.project_id)
        if template_row is None:
            raise HTTPException(409, "Accept the learned procedure before analyzing a new attempt.")
    document = template_row.document_json if template_row else None
    try:
        model_id, segmentation_model_id = guided.choose_models(session, payload.model_id, payload.segmentation_model_id)
    except UnsupportedMode as exc:
        raise HTTPException(400, str(exc)) from exc
    plan = guided.plan(video, template=document, model_id=model_id, segmentation_model_id=segmentation_model_id)
    if not plan["cached"] and not (payload.allow_live and settings.live_inference):
        raise HTTPException(409, {"code": "live_call_required", "plan": plan})
    try:
        run = guided.create_guided_run(
            session,
            video,
            template=document,
            template_id=template_row.id if template_row else "",
            allow_live=payload.allow_live,
            model_id=model_id,
            segmentation_model_id=segmentation_model_id,
        )
    except (ValueError, UnsupportedMode) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _run_payload(session, run)


@router.get("/runs/{run_id}")
def get_run(run_id: str, session: Session = Depends(db)) -> dict:
    run = session.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(404, "analysis not found")
    return _run_payload(session, run)


@router.get("/runs/{run_id}/objects")
def run_objects(run_id: str) -> dict:
    """Thumbnails and timelines, keyed by the analysis's object id. Empty until they exist."""
    manifest = object_views.read_manifest(run_id) or {"objects": {}}
    base = f"/api/learnloop/runs/{run_id}/objects"
    for item in manifest["objects"].values():
        item["thumbnail_url"] = f"{base}/{item['thumbnail']}" if item.get("thumbnail") else None
        item["cells"] = [
            {"time": cell["time"], "url": f"{base}/{cell['image']}" if cell["image"] else None} for cell in item["cells"]
        ]
    return manifest


@router.get("/runs/{run_id}/objects/{name}")
def run_object_image(run_id: str, name: str) -> FileResponse:
    path = object_views.views_dir(run_id) / name
    if not re.fullmatch(r"o\d+_(c\d+|thumb)\.(jpg|png)", name) or not path.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@router.post("/procedures/{project_id}/learn")
def learn(project_id: str, payload: LearnBody, session: Session = Depends(db)) -> dict:
    # Teach and Procedure can both ask. One analysis gives one learned procedure.
    existing = session.scalar(
        select(LearnedWorkflowTemplate)
        .where(LearnedWorkflowTemplate.project_id == project_id, LearnedWorkflowTemplate.source_run_id == payload.run_id)
        .order_by(LearnedWorkflowTemplate.version)
    )
    if existing is not None:
        return store.template_payload(existing)
    try:
        return store.create_template(session, project_id, payload.run_id, promote_supported=True, results_first=True)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/templates/{template_id}")
def delete_template(template_id: str, session: Session = Depends(db)) -> dict:
    """Remove a learned procedure and its reports. Its analysis goes too, or the page would learn it again."""
    from app.api.routes import delete_run

    try:
        row = store.delete_template_family(session, template_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    run_id, video_id = row.source_run_id, row.source_video_id
    shared = session.scalar(select(LearnedWorkflowTemplate).where(LearnedWorkflowTemplate.source_video_id == video_id))
    if shared is None and session.get(ProcedureRecording, video_id) is not None:
        delete_recording(video_id, session)
    elif session.get(AnalysisRun, run_id) is not None:
        delete_run(run_id, session)
    return {"deleted": template_id}


@router.post("/templates/{template_id}/verify")
def verify(template_id: str, payload: VerifyBody, session: Session = Depends(db)) -> dict:
    try:
        return store.start_verification(
            session,
            template_id,
            {"execution_run_id": payload.run_id, "strategy": "tracked", "require_accepted": True},
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/videos/{video_id}/frame")
def frame(video_id: str, t: float = 0, session: Session = Depends(db)) -> FileResponse:
    """The recorded picture at a timestamp, for evidence snapshots."""
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise HTTPException(404, "recording not found")
    stamp = min(max(t, 0), max(float(video.duration_sec or 0) - 0.05, 0))
    dest = data_path("evidence_frames", video_id, f"{stamp:.2f}.jpg")
    if not dest.exists():
        try:
            extract_frame(playback_path(video), stamp, dest)
        except Exception as exc:
            raise HTTPException(500, f"The frame could not be read: {exc}") from exc
    return FileResponse(dest, media_type="image/jpeg")


def _project(session: Session, project_id: str) -> Project:
    row = session.get(Project, project_id)
    if row is None:
        raise HTTPException(404, "procedure not found")
    return row


def _is_procedure(session: Session, row: Project) -> bool:
    return row.description == "learnloop_procedure"


def _recordings(session: Session, project_id: str) -> list[ProcedureRecording]:
    return list(
        session.scalars(
            select(ProcedureRecording)
            .where(ProcedureRecording.project_id == project_id)
            .order_by(ProcedureRecording.created_at)
        )
    )


def _latest_run(session: Session, video_id: str) -> AnalysisRun | None:
    return session.scalar(
        select(AnalysisRun).where(AnalysisRun.video_id == video_id).order_by(AnalysisRun.created_at.desc())
    )


def _recording(session: Session, row: ProcedureRecording) -> dict:
    video = session.get(VideoAsset, row.video_id)
    run = _latest_run(session, row.video_id)
    return {
        "role": row.role,
        "video": video_dict(video) if video else None,
        "run": _run_payload(session, run) if run else None,
    }


def _accepted_template(session: Session, project_id: str) -> LearnedWorkflowTemplate | None:
    return session.scalar(
        select(LearnedWorkflowTemplate)
        .where(
            LearnedWorkflowTemplate.project_id == project_id,
            LearnedWorkflowTemplate.acceptance_status == "accepted",
        )
        .order_by(LearnedWorkflowTemplate.created_at.desc())
    )


def _run_payload(session: Session, run: AnalysisRun) -> dict:
    config = run.config_json or {}
    stages = session.scalars(select(PipelineStageRun).where(PipelineStageRun.run_id == run.id)).all()
    sent = session.query(ModelRequestRow).filter(ModelRequestRow.run_id == run.id).count()
    return {
        "id": run.id,
        "video_id": run.video_id,
        "status": run.status,
        "error": run.error,
        "provider_id": run.provider_id,
        "model_id": run.model_id,
        "modality": run.modality,
        "profile": config.get("analysis_profile") or "",
        "source": config.get("inference_source") or "",
        "cached_at": config.get("cached_at") or "",
        "provider_requests": sent,
        "reference_template_id": config.get("reference_template_id") or "",
        "stages": {stage.stage: stage.status for stage in stages},
        "created_at": run.created_at.isoformat(),
    }


def _summary(session: Session, project: Project) -> dict:
    recordings = _recordings(session, project.id)
    demo = next((item for item in recordings if item.role == "demonstration"), None)
    video = session.get(VideoAsset, demo.video_id) if demo else None
    template = session.scalar(
        select(LearnedWorkflowTemplate)
        .where(LearnedWorkflowTemplate.project_id == project.id)
        .order_by(LearnedWorkflowTemplate.created_at.desc())
    )
    latest = session.scalar(
        select(VerificationRunRow)
        .where(VerificationRunRow.project_id == project.id)
        .order_by(VerificationRunRow.created_at.desc())
    )
    steps = [step for step in (template.document_json.get("steps") or []) if not step.get("removed")] if template else []
    return {
        "id": project.id,
        "name": project.name,
        "created_at": project.created_at.isoformat(),
        "thumbnail_url": f"/api/media/videos/{video.id}/thumbnail" if video and video.thumbnail_path else "",
        "step_count": len(steps),
        "review_state": store.review_state(template.document_json) if template else "",
        "execution_count": sum(item.role == "execution" for item in recordings),
        "latest_outcome": latest.overall_result if latest else "",
    }
