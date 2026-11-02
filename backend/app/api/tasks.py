"""Task projects: several recordings, one reconciliation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_session
from app.evaluation.multisession import evaluate_project, revision_effect
from app.models.media import Project, VideoAsset
from app.models.runs import Job
from app.models.sessions import (
    CanonicalObject,
    IdentityLink,
    InterSessionTransition,
    ProjectGroundTruth,
    ProjectRevision,
    RecordingSession,
)
from app.models.util import new_id
from app.services.ingest import ingest_file, referenced_file
from app.services.reconcile import manual_order, move_clip, new_session, reconcile
from app.services.storage import data_path
from app.services.task_export import export_task
from app.services.tasks import task_bundle
from app.video.capture_time import stamp_from_tags

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def db() -> Session:
    yield from get_session()


@router.get("")
def list_tasks(session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(Project).where(Project.is_task.is_(True)).order_by(Project.created_at.desc())).all()
    return [{"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at.isoformat()} for row in rows]


@router.post("")
def create_task(payload: dict, session: Session = Depends(db)) -> dict:
    row = Project(name=payload.get("name") or "Untitled task", description=payload.get("description") or "", is_task=True)
    session.add(row)
    session.flush()
    new_session(session, row.id, "Session 1")
    session.commit()
    return task_bundle(session, row.id)


@router.get("/{project_id}")
def get_task(project_id: str, session: Session = Depends(db)) -> dict:
    bundle = task_bundle(session, project_id)
    if bundle is None:
        raise HTTPException(404, "task not found")
    return bundle


@router.post("/{project_id}/clips")
async def upload_clips(
    project_id: str,
    session_id: str = Form(""),
    research_mode: str = Form(""),
    files: list[UploadFile] = File(...),
    session: Session = Depends(db),
) -> dict:
    from app.config import settings
    from app.video.probe import VideoError

    project = session.get(Project, project_id)
    if project is None or not project.is_task:
        raise HTTPException(404, "task not found")
    for upload in files:
        raw = await upload.read()
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(413, "file exceeds the upload limit")
        if not raw:
            continue
        temp = data_path("imports", f"upload_{upload.filename or 'video'}")
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(raw)
        try:
            ingest_file(
                session,
                project_id,
                upload.filename or "video.mp4",
                temp,
                research_mode=research_mode,
                session_id=session_id or None,
            )
        except VideoError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            if temp.exists():
                temp.unlink()
    before_jobs = session.scalar(select(func.count()).select_from(Job)) or 0
    reconcile(session, project_id, "clip upload")
    after_jobs = session.scalar(select(func.count()).select_from(Job)) or 0
    bundle = task_bundle(session, project_id)
    bundle["jobs_created"] = after_jobs - before_jobs
    return bundle


@router.post("/{project_id}/clips/reference")
def reference_clips(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    from app.video.probe import VideoError

    project = session.get(Project, project_id)
    if project is None or not project.is_task:
        raise HTTPException(404, "task not found")
    raw_paths = payload.get("paths") or []
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if not raw_paths:
        raise HTTPException(400, "add at least one path")
    research_mode = str(payload.get("research_mode") or "")
    session_id = str(payload.get("session_id") or "") or None
    for raw in raw_paths:
        try:
            path = referenced_file(str(raw))
            ingest_file(
                session,
                project_id,
                path.name,
                path,
                research_mode=research_mode,
                session_id=session_id,
                reference=True,
            )
        except VideoError as exc:
            raise HTTPException(400, str(exc)) from exc
    before_jobs = session.scalar(select(func.count()).select_from(Job)) or 0
    reconcile(session, project_id, "clip reference")
    after_jobs = session.scalar(select(func.count()).select_from(Job)) or 0
    bundle = task_bundle(session, project_id)
    bundle["jobs_created"] = after_jobs - before_jobs
    return bundle


@router.post("/{project_id}/sessions")
def create_session(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    _require(session, project_id)
    new_session(session, project_id, payload.get("name") or "Session")
    session.commit()
    return task_bundle(session, project_id)


@router.patch("/{project_id}/sessions/{session_id}")
def patch_session(project_id: str, session_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    row = session.get(RecordingSession, session_id)
    if row is None or row.project_id != project_id:
        raise HTTPException(404, "session not found")
    if "name" in payload:
        row.name = payload["name"]
    if "notes" in payload:
        row.notes = payload["notes"]
    if "locked" in payload:
        row.locked = bool(payload["locked"])
        clips = list(session.scalars(select(VideoAsset).where(VideoAsset.session_id == row.id)))
        clips.sort(key=lambda video: (video.order_index is None, video.order_index or 0))
        for index, video in enumerate(clips):
            video.order_locked = row.locked
            if row.locked:
                video.order_method = "manual"
                if video.order_index is None:
                    video.order_index = float(index)
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/sessions/{session_id}/split")
def split_session(project_id: str, session_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    source = session.get(RecordingSession, session_id)
    if source is None or source.project_id != project_id:
        raise HTTPException(404, "session not found")
    created = new_session(session, project_id, payload.get("name") or f"{source.name} split")
    for video_id in payload.get("video_ids") or []:
        video = session.get(VideoAsset, video_id)
        if video and video.project_id == project_id:
            move_clip(session, video, created.id)
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/order")
def set_order(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    _require(session, project_id)
    manual_order(session, project_id, payload.get("video_ids") or [])
    before = session.scalar(select(func.count()).select_from(Job)) or 0
    reconcile(session, project_id, "manual order")
    after = session.scalar(select(func.count()).select_from(Job)) or 0
    bundle = task_bundle(session, project_id)
    bundle["jobs_created"] = after - before
    return bundle


@router.patch("/{project_id}/clips/{video_id}")
def patch_clip(project_id: str, video_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    video = session.get(VideoAsset, video_id)
    if video is None or video.project_id != project_id:
        raise HTTPException(404, "clip not found")
    if "session_id" in payload:
        move_clip(session, video, payload["session_id"] or None)
    if "research_mode" in payload:
        video.research_mode = payload["research_mode"] or ""
    if "order_locked" in payload:
        video.order_locked = bool(payload["order_locked"])
        if video.order_locked:
            video.order_method = "manual"
    if payload.get("capture_time_raw"):
        stamp = stamp_from_tags({"creation_time": payload["capture_time_raw"]}, video.file_mtime or "")
        video.capture_time_raw = payload["capture_time_raw"]
        video.capture_time_utc = stamp.utc_iso
        video.capture_timezone = stamp.timezone_name
        video.timestamp_source = "manual"
        video.timestamp_reliability = stamp.reliability
        video.timestamp_note = "Researcher-supplied capture time. " + stamp.note
        if not video.order_locked:
            video.order_ambiguous = stamp.reliability != "reliable"
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/reconcile")
def reconcile_task(project_id: str, session: Session = Depends(db)) -> dict:
    _require(session, project_id)
    before = session.scalar(select(func.count()).select_from(Job)) or 0
    reconcile(session, project_id, "reconcile")
    after = session.scalar(select(func.count()).select_from(Job)) or 0
    bundle = task_bundle(session, project_id)
    bundle["jobs_created"] = after - before
    return bundle


@router.post("/{project_id}/identity/confirm")
def confirm_link(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    link = _link(session, project_id, payload.get("link_id"))
    link.status = "confirmed"
    link.uncertain = False
    link.method = "manual"
    link.provenance = "HUMAN_CORRECTION"
    _refresh_canonical(session, link.canonical_id)
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/identity/reject")
def reject_link(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    link = _link(session, project_id, payload.get("link_id"))
    link.status = "rejected"
    link.uncertain = True
    link.method = "manual"
    link.provenance = "HUMAN_CORRECTION"
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/identity/merge")
def merge_identity(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    keep = session.get(CanonicalObject, payload.get("keep_id"))
    drop = session.get(CanonicalObject, payload.get("drop_id"))
    if keep is None or drop is None or keep.project_id != project_id or drop.project_id != project_id:
        raise HTTPException(404, "canonical object not found")
    for link in session.scalars(select(IdentityLink).where(IdentityLink.canonical_id == drop.id)):
        link.canonical_id = keep.id
        link.status = "confirmed"
        link.uncertain = False
        link.method = "manual"
        link.provenance = "HUMAN_CORRECTION"
    session.delete(drop)
    _refresh_canonical(session, keep.id)
    session.commit()
    return task_bundle(session, project_id)


@router.post("/{project_id}/identity/split")
def split_identity(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    link = _link(session, project_id, payload.get("link_id"))
    count = session.scalar(select(func.count()).select_from(CanonicalObject).where(CanonicalObject.project_id == project_id)) or 0
    created = CanonicalObject(
        id=new_id(),
        project_id=project_id,
        local_id=f"Object_{count + 1}",
        label=link.label,
        description="Split from a proposed match by the researcher.",
        uncertain=False,
    )
    session.add(created)
    session.flush()
    link.canonical_id = created.id
    link.status = "confirmed"
    link.uncertain = False
    link.method = "manual"
    link.provenance = "HUMAN_CORRECTION"
    session.commit()
    return task_bundle(session, project_id)


@router.patch("/{project_id}/transitions/{transition_id}")
def patch_transition(project_id: str, transition_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    row = session.get(InterSessionTransition, transition_id)
    if row is None or row.project_id != project_id:
        raise HTTPException(404, "transition not found")
    if "observed_change" in payload:
        row.observed_change = payload["observed_change"]
    if "hypothesis" in payload:
        row.hypothesis = payload["hypothesis"]
    if "evidence_class" in payload:
        row.evidence_class = payload["evidence_class"]
    if "verified" in payload:
        row.verified = bool(payload["verified"])
    row.provenance = "HUMAN_ANNOTATION"
    session.commit()
    return task_bundle(session, project_id)


@router.get("/{project_id}/revisions/{revision_id}")
def get_revision(project_id: str, revision_id: str, session: Session = Depends(db)) -> dict:
    row = session.get(ProjectRevision, revision_id)
    if row is None or row.project_id != project_id:
        raise HTTPException(404, "revision not found")
    parent = session.get(ProjectRevision, row.parent_id) if row.parent_id else None
    return {
        "id": row.id,
        "version": row.version,
        "change_summary": row.change_summary,
        "document": row.document_json,
        "effect": revision_effect(parent.document_json if parent else {}, row.document_json),
    }


@router.post("/{project_id}/evaluate")
def evaluate_task(project_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    bundle = task_bundle(session, project_id)
    if bundle is None:
        raise HTTPException(404, "task not found")
    truth = payload.get("ground_truth") or payload
    session.add(ProjectGroundTruth(project_id=project_id, document_json=truth))
    session.commit()
    prediction = {
        "order": bundle["order"],
        "links": bundle["links"],
        "transitions": bundle["transitions"],
        "sessions": bundle["sessions"],
        "workflow": bundle["workflow"],
        "clip_run_ids": (bundle["revision"] or {}).get("document", {}).get("clip_run_ids", {}),
    }
    scores = evaluate_project(prediction, truth, [link["score"] for link in bundle["links"]])
    return {"metrics": scores, "ground_truth_is_accuracy": True, "confidence_is_accuracy": False}


@router.get("/{project_id}/export")
def download_task(project_id: str, session: Session = Depends(db)):
    dest = export_task(session, project_id)
    if dest is None:
        raise HTTPException(404, "task not found")
    return FileResponse(dest, filename=f"{project_id}-task.zip", media_type="application/zip")


def _require(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None or not project.is_task:
        raise HTTPException(404, "task not found")
    return project


def _link(session: Session, project_id: str, link_id: str | None) -> IdentityLink:
    link = session.get(IdentityLink, link_id) if link_id else None
    if link is None or link.project_id != project_id:
        raise HTTPException(404, "identity link not found")
    return link


def _refresh_canonical(session: Session, canonical_id: str) -> None:
    row = session.get(CanonicalObject, canonical_id)
    if row is None:
        return
    links = list(session.scalars(select(IdentityLink).where(IdentityLink.canonical_id == canonical_id, IdentityLink.status != "rejected")))
    row.uncertain = any(link.uncertain or link.status != "confirmed" for link in links)
