"""HTTP routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from app import __version__
from app.db import get_session
from app.models.audio import AudioAlignment, TranscriptSegment
from app.models.domain import (
    AnnotationRow,
    Artifact,
    ComparisonRow,
    DetectedObject,
    EvaluationResult,
    GroundTruthDataset,
    ObjectImage,
    SceneChunk,
    SegmentationMask,
    SpatialTrackPoint,
)
from app.models.media import FrameAsset, Project, VideoAsset
from app.models.runs import (
    AnalysisRun,
)
from app.models.sessions import (
    CanonicalObject,
    IdentityLink,
    InterSessionTransition,
    OrderHypothesis,
    ProjectGroundTruth,
    ProjectRevision,
    RecordingSession,
)
from app.services.ingest import (
    cancel_ingest,
)
from app.services.lab_settings import load_settings, save_settings
from app.services.locate import locate_dropped_video
from app.services.storage import data_path, is_external, resolve_stored

router = APIRouter(prefix="/api")


def db() -> Session:
    yield from get_session()


def _run(session: Session, run_id: str) -> AnalysisRun:
    run = session.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "name": "LearnLoop"}


def _unlink_stored(stored: str) -> None:
    if not stored or is_external(stored):
        return
    try:
        path = resolve_stored(stored)
    except ValueError:
        return
    if path.is_file():
        path.unlink()


def _drop_pts_if_unused(session: Session, checksum: str) -> None:
    prefix = (checksum or "")[:12]
    if len(prefix) < 12:
        return
    still = session.scalar(
        select(func.count()).select_from(VideoAsset).where(VideoAsset.checksum_sha256.startswith(prefix))
    ) or 0
    if still:
        return
    pts = data_path("originals", f"{prefix}.pts.json")
    if pts.is_file():
        pts.unlink()


def _remove_tree(folder: str, name: str) -> None:
    path = data_path(folder, name)
    if path.is_dir():
        shutil.rmtree(path)


def _delete_where(session: Session, model, column, ids: list[str]) -> None:
    if ids:
        session.execute(delete(model).where(column.in_(ids)))


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, session: Session = Depends(db)) -> dict:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    videos = list(session.scalars(select(VideoAsset).where(VideoAsset.project_id == project.id)))
    runs = list(session.scalars(select(AnalysisRun).where(AnalysisRun.project_id == project.id)))
    video_ids = [video.id for video in videos]
    checksums = [video.checksum_sha256 for video in videos if video.checksum_sha256]
    for video_id in video_ids:
        cancel_ingest(video_id)
    run_ids = [run.id for run in runs]
    paths: list[str] = []
    for video in videos:
        paths.extend([video.stored_path, video.proxy_path, video.thumbnail_path, video.contact_sheet_path])
    if video_ids:
        paths.extend(session.scalars(select(FrameAsset.path).where(FrameAsset.video_id.in_(video_ids))))
        paths.extend(session.scalars(select(Artifact.path).where(Artifact.video_id.in_(video_ids))))
    if run_ids:
        paths.extend(session.scalars(select(Artifact.path).where(Artifact.run_id.in_(run_ids))))
        paths.extend(session.scalars(select(SegmentationMask.path).where(SegmentationMask.run_id.in_(run_ids))))
        paths.extend(
            session.scalars(
                select(ObjectImage.path)
                .join(DetectedObject, ObjectImage.object_id == DetectedObject.id)
                .where(DetectedObject.run_id.in_(run_ids))
            )
        )
    for model in (
        RecordingSession,
        CanonicalObject,
        IdentityLink,
        InterSessionTransition,
        ProjectRevision,
        ProjectGroundTruth,
        OrderHypothesis,
    ):
        session.execute(delete(model).where(model.project_id == project.id))
    _delete_where(session, TranscriptSegment, TranscriptSegment.video_id, video_ids)
    _delete_where(session, TranscriptSegment, TranscriptSegment.run_id, run_ids)
    _delete_where(session, AudioAlignment, AudioAlignment.video_id, video_ids)
    _delete_where(session, AudioAlignment, AudioAlignment.run_id, run_ids)
    _delete_where(session, Artifact, Artifact.video_id, video_ids)
    _delete_where(session, Artifact, Artifact.run_id, run_ids)
    _delete_where(session, SegmentationMask, SegmentationMask.run_id, run_ids)
    _delete_where(session, SpatialTrackPoint, SpatialTrackPoint.run_id, run_ids)
    _delete_where(session, SceneChunk, SceneChunk.run_id, run_ids)
    _delete_where(session, GroundTruthDataset, GroundTruthDataset.run_id, run_ids)
    _delete_where(session, EvaluationResult, EvaluationResult.run_id, run_ids)
    _delete_where(session, AnnotationRow, AnnotationRow.run_id, run_ids)
    if run_ids:
        session.execute(
            delete(ComparisonRow).where(or_(ComparisonRow.left_run_id.in_(run_ids), ComparisonRow.right_run_id.in_(run_ids)))
        )
    session.execute(delete(Project).where(Project.id == project.id))
    session.commit()
    for stored in {path for path in paths if path}:
        remaining = session.scalar(
            select(func.count()).select_from(VideoAsset).where(
                or_(VideoAsset.stored_path == stored, VideoAsset.proxy_path == stored)
            )
        ) or 0
        if remaining:
            continue
        _unlink_stored(stored)
        if stored.startswith("originals/"):
            prefix = Path(stored).name.split("_", 1)[0]
            still = session.scalar(
                select(func.count()).select_from(VideoAsset).where(VideoAsset.stored_path.like(f"originals/{prefix}\\_%", escape="\\"))
            ) or 0
            if not still:
                pts = data_path("originals", f"{prefix}.pts.json")
                if pts.is_file():
                    pts.unlink()
    for checksum in checksums:
        _drop_pts_if_unused(session, checksum)
    for video_id in video_ids:
        _remove_tree("frames", video_id)
    for run_id in run_ids:
        _remove_tree("runs", run_id)
        run_export = data_path("exports", f"{run_id}.zip")
        if run_export.is_file():
            run_export.unlink()
    export = data_path("exports", f"{project_id}-task.zip")
    if export.is_file():
        export.unlink()
    return {"deleted": project_id}


@router.get("/settings")
def get_lab_settings() -> dict:
    return load_settings()


@router.put("/settings")
def put_lab_settings(payload: dict) -> dict:
    try:
        return save_settings(str(payload.get("video_storage") or ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/videos/locate")
def locate_video(payload: dict) -> dict:
    from app.video.probe import VideoError

    name = str(payload.get("name") or "")
    try:
        size = int(payload["size"])
        mtime_ms = int(payload["mtime_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "size and modification time are required") from exc
    try:
        path = locate_dropped_video(name, size, mtime_ms)
    except VideoError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"path": str(path)}


@router.delete("/videos/{video_id}")
def delete_video(video_id: str, session: Session = Depends(db)) -> dict:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    used = session.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.video_id == video.id))
    if used:
        raise HTTPException(409, "delete the runs that use this video first")
    cancel_ingest(video_id)
    checksum = video.checksum_sha256
    stored_paths = [video.stored_path, video.proxy_path, video.thumbnail_path, video.contact_sheet_path]
    session.delete(video)
    session.commit()
    # The same file uploaded twice shares its original and proxy, named by checksum. Keep any another video still uses.
    for stored in {path for path in stored_paths if path and not is_external(path)}:
        remaining = session.scalar(
            select(func.count()).select_from(VideoAsset).where(
                or_(VideoAsset.stored_path == stored, VideoAsset.proxy_path == stored)
            )
        ) or 0
        if remaining:
            continue
        path = resolve_stored(stored)
        if path.exists() and path.is_file():
            path.unlink()
    _remove_tree("frames", video_id)
    _drop_pts_if_unused(session, checksum)
    return {"deleted": video_id}


@router.get("/media/videos/{video_id}/{kind}")
def media_video(video_id: str, kind: str, session: Session = Depends(db)) -> FileResponse:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    stored = {
        "playback": video.proxy_path or video.stored_path,
        "original": video.stored_path,
        "thumbnail": video.thumbnail_path,
        "contact-sheet": video.contact_sheet_path,
    }.get(kind)
    if not stored:
        raise HTTPException(404, "media not found")
    path = resolve_stored(stored)
    if not path.exists():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(path)


@router.get("/media/frames/{frame_id}")
def media_frame(frame_id: str, session: Session = Depends(db)) -> FileResponse:
    frame = session.get(FrameAsset, frame_id)
    if frame is None:
        raise HTTPException(404, "frame not found")
    path = resolve_stored(frame.path)
    return FileResponse(path)


@router.get("/media/masks/{mask_id}")
def media_mask(mask_id: str, session: Session = Depends(db)) -> FileResponse:
    row = session.get(SegmentationMask, mask_id)
    if row is None or not row.path:
        raise HTTPException(404, "mask not found")
    path = resolve_stored(row.path)
    if not path.exists():
        raise HTTPException(404, "mask file missing")
    return FileResponse(path)


@router.get("/media/object-images/{image_id}")
def media_object_image(image_id: str, session: Session = Depends(db)) -> FileResponse:
    image = session.get(ObjectImage, image_id)
    if image is None or not image.path:
        raise HTTPException(404, "image not found")
    return FileResponse(resolve_stored(image.path))


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    folder = data_path("runs", run.id)
    session.delete(run)
    session.commit()
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    return {"deleted": run_id}


