"""HTTP routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import __version__
from app.api.serialize import bundle, run_summary, video_dict
from app.db import get_session
from app.evaluation.compare import compare_documents
from app.evaluation.metrics import evaluate_documents
from app.models.audio import AudioAlignment, TranscriptSegment
from app.models.domain import (
    ActionRow,
    AmbiguityRow,
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
    WorkflowRow,
)
from app.models.media import FrameAsset, Project, VideoAsset
from app.models.runs import (
    AnalysisRevision,
    AnalysisRun,
    Job,
    ManualModel,
    ModelRequestRow,
    ModelResponseRow,
    PipelineStageRun,
    PromptVersion,
    ProviderStatus,
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
from app.models.util import utcnow
from app.perception.corrections import decide_hypothesis, merge_instances, split_instance
from app.perception.types import TrackingCancelled
from app.pipeline.perception_stage import run_spatial_stage
from app.providers.catalog import lookup, models_for
from app.providers.claude_code import (
    begin_login,
    load_claude_code,
    probe,
    save_claude_code,
    submit_login_code,
)
from app.providers.codex_cli import begin_login as begin_codex_login
from app.providers.codex_cli import load_codex, save_codex
from app.providers.codex_cli import probe as probe_codex
from app.providers.cursor_cli import begin_login as begin_cursor_login
from app.providers.cursor_cli import load_cursor_cli, save_cursor_cli
from app.providers.cursor_cli import probe as probe_cursor
from app.providers.model_list import options_for
from app.providers.registry import get_provider, is_local, provider_ids
from app.providers.types import ProviderError, UnsupportedMode
from app.schemas.analysis import AnalysisDocument
from app.services.corrections import (
    add_action,
    add_dependency,
    add_object,
    add_step,
    merge_objects,
    remove_dependency,
    split_object,
    update_action,
    update_object,
    update_step,
)
from app.services.costs import aggregate_spend, project_costs, uses_api_key
from app.services.credentials import (
    delete_secret,
    load_secret,
    provider_status,
    record_validation,
    secret_hint,
    store_secret,
)
from app.services.exchange import export_run, import_run
from app.services.experiments import create_run, duplicate_run, retry_from, save_prompt
from app.services.ingest import (
    accept_reference,
    accept_upload,
    cancel_ingest,
    ingest_file,
    referenced_file,
    schedule_ingest,
)
from app.services.lab_settings import load_settings, save_settings
from app.services.locate import locate_dropped_video
from app.services.materialize import cached_observation_crop
from app.services.storage import data_path, is_external, resolve_stored, safe_filename

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
    return {"ok": True, "version": __version__, "name": "LearnLoop Research Lab"}


@router.get("/dashboard")
def dashboard(session: Session = Depends(db)) -> dict:
    videos = session.scalar(select(func.count()).select_from(VideoAsset)) or 0
    runs = session.scalars(select(AnalysisRun)).all()
    completed = sum(1 for run in runs if run.status == "completed")
    active = sum(1 for run in runs if run.status in {"pending", "running"})
    revision_ids = [run.active_revision_id for run in runs if run.active_revision_id]
    objects = actions = ambiguities = workflows = 0
    if revision_ids:
        objects = session.scalar(
            select(func.count()).select_from(DetectedObject).where(DetectedObject.revision_id.in_(revision_ids))
        ) or 0
        actions = session.scalar(
            select(func.count()).select_from(ActionRow).where(ActionRow.revision_id.in_(revision_ids))
        ) or 0
        ambiguities = session.scalar(
            select(func.count()).select_from(AmbiguityRow).where(AmbiguityRow.revision_id.in_(revision_ids))
        ) or 0
        workflows = session.scalar(
            select(func.count()).select_from(WorkflowRow).where(WorkflowRow.revision_id.in_(revision_ids))
        ) or 0
    connected = []
    for provider_id in provider_ids():
        status = provider_status(session, provider_id)
        if status and status.last_ok:
            connected.append(provider_id)
    recent = [run_summary(session, run) for run in sorted(runs, key=lambda item: item.created_at, reverse=True)[:12]]
    spend = aggregate_spend(session, list(runs))
    return {
        "videos": videos,
        "runs_completed": completed,
        "runs_active": active,
        "objects": objects,
        "actions": actions,
        "workflows": workflows,
        "ambiguities": ambiguities,
        "providers_connected": connected,
        "input_tokens": spend["input_tokens"],
        "output_tokens": spend["output_tokens"],
        "cost_usd": spend["cost_usd"],
        "unpriced_calls": spend["unpriced_calls"],
        "unitemized_usd": spend["unitemized_usd"],
        "cost_by_provider": spend["cost_by_provider"],
        "recent_runs": recent,
    }


def _project_payload(row: Project) -> dict:
    created = row.created_at.isoformat() if row.created_at else ""
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "is_task": bool(row.is_task),
        "created_at": created,
    }


@router.get("/projects")
def list_projects(session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(Project).order_by(Project.created_at.desc())).all()
    return [_project_payload(row) for row in rows]


@router.post("/projects")
def create_project(payload: dict, session: Session = Depends(db)) -> dict:
    from app.services.reconcile import new_session

    spans_days = bool(payload.get("is_task"))
    row = Project(
        name=payload.get("name") or "Untitled study",
        description=payload.get("description") or "",
        is_task=spans_days,
    )
    session.add(row)
    session.flush()
    if spans_days:
        new_session(session, row.id, "Session 1")
    session.commit()
    session.refresh(row)
    return _project_payload(row)


@router.get("/projects/{project_id}/costs")
def project_cost_report(project_id: str, session: Session = Depends(db)) -> dict:
    report = project_costs(session, project_id)
    if report is None:
        raise HTTPException(404, "project not found")
    return report


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


@router.post("/videos")
async def upload_video(
    project_id: str = Form(""),
    file: UploadFile = File(...),
    session: Session = Depends(db),
) -> dict:
    from app.config import settings
    from app.models.util import new_id
    from app.video.probe import VideoError

    temp = data_path("imports", f"upload_{new_id()}_{safe_filename(file.filename or 'video')}")
    temp.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with temp.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "file exceeds the upload limit")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty upload")
        target = project_id or None
        if settings.ingest_inline:
            video = ingest_file(session, target, file.filename or "video.mp4", temp)
        else:
            video = accept_upload(session, target, file.filename or "video.mp4", temp)
    except VideoError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if temp.exists():
            temp.unlink()
    if not settings.ingest_inline:
        schedule_ingest(video.id)
    return video_dict(video)


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


@router.post("/videos/reference")
def reference_video(payload: dict, session: Session = Depends(db)) -> dict:
    from app.config import settings
    from app.video.probe import VideoError

    project_id = str(payload.get("project_id") or "") or None
    raw = str(payload.get("path") or "").strip()
    if not raw:
        raise HTTPException(400, "path is required")
    try:
        path = referenced_file(raw)
        if settings.ingest_inline:
            video = ingest_file(session, project_id, path.name, path, reference=True)
        else:
            video = accept_reference(session, project_id, path)
    except VideoError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not settings.ingest_inline:
        schedule_ingest(video.id)
    return video_dict(video)


def _videos_in_display_order(project_id: str):
    """Custom order first. Unordered videos stay newest-first behind that sequence."""
    return (
        select(VideoAsset)
        .where(VideoAsset.project_id == project_id)
        .order_by(
            case((VideoAsset.order_index.is_(None), 1), else_=0),
            VideoAsset.order_index.asc(),
            VideoAsset.created_at.desc(),
        )
    )


@router.get("/videos")
def list_videos(
    project_id: str | None = None,
    unassigned: bool = False,
    session: Session = Depends(db),
) -> list[dict]:
    if unassigned and project_id:
        raise HTTPException(400, "choose a project or the library, not both")
    if unassigned:
        query = select(VideoAsset).where(VideoAsset.project_id.is_(None)).order_by(VideoAsset.created_at.desc())
    elif project_id:
        query = _videos_in_display_order(project_id)
    else:
        query = select(VideoAsset).order_by(VideoAsset.created_at.desc())
    return [video_dict(video) for video in session.scalars(query)]


@router.post("/projects/{project_id}/videos/attach")
def attach_project_videos(project_id: str, payload: dict, session: Session = Depends(db)) -> list[dict]:
    from app.services.membership import MembershipError, attach_videos

    requested = payload.get("video_ids")
    if not isinstance(requested, list) or any(not isinstance(video_id, str) or not video_id for video_id in requested):
        raise HTTPException(400, "video_ids must be a list of ids")
    try:
        videos = attach_videos(session, project_id, requested)
    except MembershipError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return [video_dict(video) for video in videos]


@router.post("/videos/{video_id}/detach")
def detach_project_video(video_id: str, session: Session = Depends(db)) -> dict:
    from app.services.membership import MembershipError, detach_video

    try:
        video = detach_video(session, video_id)
    except MembershipError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return video_dict(video)


@router.post("/projects/{project_id}/videos/order")
def order_project_videos(project_id: str, payload: dict, session: Session = Depends(db)) -> list[dict]:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    if project.is_task:
        raise HTTPException(400, "reorder task clips from the task chronology")
    videos = {
        video.id: video
        for video in session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id))
    }
    requested = payload.get("video_ids") or []
    if not isinstance(requested, list) or any(not isinstance(video_id, str) for video_id in requested):
        raise HTTPException(400, "video_ids must be a list of ids")
    if any(video_id not in videos for video_id in requested):
        raise HTTPException(400, "video list includes a clip from another project")
    placed = set(requested)
    rest = [
        video.id
        for video in sorted(
            (video for video in videos.values() if video.id not in placed),
            key=lambda video: (
                video.order_index is None,
                video.order_index if video.order_index is not None else 0,
                -(video.created_at.timestamp() if video.created_at else 0),
            ),
        )
    ]
    for index, video_id in enumerate([*requested, *rest]):
        videos[video_id].order_index = float(index)
    session.commit()
    return [video_dict(video) for video in session.scalars(_videos_in_display_order(project_id))]


@router.get("/videos/{video_id}")
def get_video(video_id: str, session: Session = Depends(db)) -> dict:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    frames = session.scalars(
        select(FrameAsset).where(FrameAsset.video_id == video.id, FrameAsset.run_id.is_(None)).order_by(FrameAsset.timestamp)
    ).all()
    payload = video_dict(video, len(frames))
    payload["frames"] = [
        {"id": frame.id, "timestamp": frame.timestamp, "reason": frame.reason, "url": f"/api/media/frames/{frame.id}"}
        for frame in frames
    ]
    return payload


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
    for stored in (video.stored_path, video.proxy_path, video.thumbnail_path, video.contact_sheet_path):
        if not stored or is_external(stored):
            continue
        path = resolve_stored(stored)
        if path.exists() and path.is_file():
            path.unlink()
    session.delete(video)
    session.commit()
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


@router.get("/runs/{run_id}/sam-masks")
def sam_masks(run_id: str, t: float = 0, session: Session = Depends(db)) -> list[dict]:
    """Boxes and mask URLs near one timestamp. Full rasters are not inlined."""
    rows = session.scalars(
        select(SegmentationMask).where(
            SegmentationMask.run_id == run_id,
            SegmentationMask.source == "sam-3.1",
        )
    ).all()
    if not rows:
        return []
    nearest = min(rows, key=lambda row: abs(row.timestamp - t))
    if abs(nearest.timestamp - t) > 0.5:
        return []
    chosen = [row for row in rows if abs(row.timestamp - nearest.timestamp) < 1e-3]
    objects = {
        obj.id: obj.local_id
        for obj in session.scalars(select(DetectedObject).where(DetectedObject.run_id == run_id))
    }
    payload = []
    for row in chosen:
        meta = row.rle_json or {}
        payload.append(
            {
                "id": row.id,
                "timestamp": row.timestamp,
                "bbox": row.bbox_json,
                "track_ref": row.track_ref,
                "confidence": row.confidence,
                "association_status": meta.get("association_status") or "",
                "object_local_id": objects.get(row.object_id or "", "") or meta.get("object_local_id") or "",
                "identity_uncertain": row.identity_uncertain,
                "decode": meta.get("decode") or "",
                "url": f"/api/media/masks/{row.id}" if row.path else "",
            }
        )
    return payload


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


@router.get("/media/observations/{observation_id}/crop")
def media_observation_crop(observation_id: str, session: Session = Depends(db)) -> FileResponse:
    try:
        path = cached_observation_crop(session, observation_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@router.get("/prompts")
def list_prompts(session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(PromptVersion).order_by(PromptVersion.key, PromptVersion.version.desc())).all()
    return [
        {
            "id": row.id,
            "key": row.key,
            "name": row.name,
            "version": row.version,
            "body": row.body,
            "foundational": row.foundational,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@router.post("/prompts")
def create_prompt(payload: dict, session: Session = Depends(db)) -> dict:
    row = save_prompt(session, payload["key"], payload.get("name") or payload["key"], payload["body"])
    return {"id": row.id, "key": row.key, "version": row.version}


@router.get("/providers/ids")
def list_provider_ids() -> list[dict]:
    return [{"id": provider_id, "local": is_local(provider_id)} for provider_id in provider_ids()]


@router.get("/providers")
def list_providers(session: Session = Depends(db)) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    manual = session.scalars(select(ManualModel)).all()
    added: dict[str, list] = {}
    for row in manual:
        added.setdefault(row.provider_id, []).append(row.model_id)
    prepared = []
    for provider_id in provider_ids():
        local = is_local(provider_id)
        configured, source, hint = (True, "local", "") if local else secret_hint(session, provider_id)
        status = provider_status(session, provider_id)
        secret = "" if local else load_secret(session, provider_id)[0]
        prepared.append(
            {
                "id": provider_id,
                "configured": configured,
                "source": source,
                "hint": hint,
                "status": status,
                "secret": secret,
                "base_url": status.extra_base_url if status else "",
                "catalog": [caps.as_dict() for caps in models_for(provider_id)],
            }
        )

    def listed(item: dict) -> list[dict]:
        return options_for(item["id"], item["catalog"], item["secret"], item["base_url"])

    with ThreadPoolExecutor(max_workers=len(prepared) or 1) as pool:
        model_lists = list(pool.map(listed, prepared))
    payload = []
    for item, models in zip(prepared, model_lists, strict=True):
        provider_id = item["id"]
        status = item["status"]
        models = with_manual_models(provider_id, models, added.get(provider_id, []))
        entry = {
                "id": provider_id,
                "configured": item["configured"],
                "secret_source": item["source"],
                "hint": item["hint"],
                "validated": bool(status and status.last_ok),
                "last_error": "" if not status else status.last_error,
                "last_validated_at": status.last_validated_at.isoformat() if status and status.last_validated_at else None,
                "base_url": status.extra_base_url if status else "",
                "local": is_local(provider_id),
                "models": models,
                **({"codex": load_codex()} if provider_id == "openai" else {}),
                **({"claude_code": load_claude_code()} if provider_id == "anthropic" else {}),
                **({"cursor_cli": load_cursor_cli()} if provider_id == "cursor" else {}),
        }
        if provider_id == "meta":
            from app.providers.sam import sam_capabilities

            entry["sam"] = sam_capabilities()
        payload.append(entry)
    return payload


@router.get("/providers/openai/codex")
async def codex_status() -> dict:
    return await probe_codex()


@router.put("/providers/openai/codex")
def put_codex(payload: dict) -> dict:
    try:
        return save_codex(payload.get("binary") or "codex", payload.get("transport") or "api")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/providers/openai/codex/link")
async def link_codex(payload: dict | None = Body(default=None)) -> dict:
    force = bool((payload or {}).get("force"))
    try:
        return await begin_codex_login(force=force)
    except ProviderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/providers/anthropic/claude-code")
async def claude_code_status() -> dict:
    return await probe()


@router.put("/providers/anthropic/claude-code")
def put_claude_code(payload: dict) -> dict:
    try:
        return save_claude_code(payload.get("binary") or "claude", payload.get("transport") or "api")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/providers/anthropic/claude-code/link")
async def link_claude_code(payload: dict | None = Body(default=None)) -> dict:
    force = bool((payload or {}).get("force"))
    try:
        return await begin_login(force=force)
    except ProviderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/providers/anthropic/claude-code/link/code")
async def claude_code_login_code(payload: dict | None = Body(default=None)) -> dict:
    try:
        return await submit_login_code((payload or {}).get("code") or "")
    except ProviderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/providers/cursor/cli")
async def cursor_cli_status(session: Session = Depends(db)) -> dict:
    secret, _source = load_secret(session, "cursor")
    return await probe_cursor(secret)


@router.put("/providers/cursor/cli")
def put_cursor_cli(payload: dict) -> dict:
    try:
        return save_cursor_cli(payload.get("binary") or "agent", payload.get("transport") or "login")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/providers/cursor/cli/link")
async def link_cursor_cli(payload: dict | None = Body(default=None)) -> dict:
    force = bool((payload or {}).get("force"))
    try:
        return await begin_cursor_login(force=force)
    except ProviderError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/providers/{provider_id}/credentials")
def put_credentials(provider_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    if is_local(provider_id):
        raise HTTPException(400, "this provider does not use an API key")
    secret = payload.get("api_key") or ""
    if not secret:
        raise HTTPException(400, "api_key is required")
    store_secret(session, provider_id, secret)
    if payload.get("base_url"):
        status = session.get(ProviderStatus, provider_id) or ProviderStatus(provider_id=provider_id)
        status.extra_base_url = payload["base_url"]
        session.merge(status)
        session.commit()
    return {"configured": True, "provider_id": provider_id}


@router.delete("/providers/{provider_id}/credentials")
def remove_credentials(provider_id: str, session: Session = Depends(db)) -> dict:
    delete_secret(session, provider_id)
    return {"configured": False}


@router.post("/providers/{provider_id}/validate")
async def validate_provider(provider_id: str, session: Session = Depends(db)) -> dict:
    if is_local(provider_id):
        record_validation(session, provider_id, True, "")
        return {"ok": True, "provider_id": provider_id, "detail": "local provider"}
    secret, source = load_secret(session, provider_id)
    cursor_login = provider_id == "cursor" and load_cursor_cli()["transport"] == "login"
    codex_login = provider_id == "openai" and load_codex()["transport"] == "codex"
    if cursor_login or codex_login:
        secret, source = "", "cli"
    elif not secret:
        record_validation(session, provider_id, False, "no credentials configured")
        raise HTTPException(400, "no credentials configured")
    try:
        await get_provider(provider_id, session).validate(secret)
    except Exception as exc:
        record_validation(session, provider_id, False, str(exc))
        raise HTTPException(502, f"validation failed: {exc}") from exc
    record_validation(session, provider_id, True, "")
    return {"ok": True, "provider_id": provider_id, "source": source}


@router.get("/providers/{provider_id}/models")
async def discover_models(provider_id: str, refresh: bool = False, session: Session = Depends(db)) -> dict:
    catalog = [caps.as_dict() for caps in models_for(provider_id)]
    secret, _source = load_secret(session, provider_id)
    status = provider_status(session, provider_id)
    base_url = status.extra_base_url if status else ""
    error = ""
    try:
        models = options_for(provider_id, catalog, secret, base_url, refresh=refresh)
    except Exception as exc:
        models = catalog
        error = str(exc)
    listed = models is not catalog
    manual = session.scalars(select(ManualModel.model_id).where(ManualModel.provider_id == provider_id)).all()
    models = with_manual_models(provider_id, models, list(manual))
    discovered = [item["model_id"] for item in models] if listed else []
    return {"catalog": models, "discovered": discovered, "error": error}


def with_manual_models(provider_id: str, models: list[dict], manual_ids: list[str]) -> list[dict]:
    """Append researcher-added ids the provider did not list. Each model id appears once."""
    merged: list[dict] = []
    seen: set[str] = set()
    for model in [*models, *(lookup(provider_id, model_id).as_dict() for model_id in manual_ids)]:
        if model["model_id"] not in seen:
            seen.add(model["model_id"])
            merged.append(model)
    return merged


@router.post("/providers/{provider_id}/models")
def add_model(provider_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(400, "model_id is required")
    exists = session.scalar(
        select(ManualModel.id).where(ManualModel.provider_id == provider_id, ManualModel.model_id == model_id)
    )
    if not exists:
        session.add(ManualModel(provider_id=provider_id, model_id=model_id, notes=payload.get("notes") or ""))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
    return lookup(provider_id, model_id).as_dict()


@router.post("/runs")
def post_run(payload: dict, session: Session = Depends(db)) -> dict:
    try:
        run = create_run(session, payload)
    except (UnsupportedMode, ValueError, ProviderError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return run_summary(session, run)


@router.get("/runs")
def list_runs(
    project_id: str | None = None,
    status: str | None = None,
    provider_id: str | None = None,
    session: Session = Depends(db),
) -> list[dict]:
    query = select(AnalysisRun).order_by(AnalysisRun.created_at.desc())
    if project_id:
        query = query.where(AnalysisRun.project_id == project_id)
    if status:
        query = query.where(AnalysisRun.status == status)
    if provider_id:
        query = query.where(AnalysisRun.provider_id == provider_id)
    return [run_summary(session, run) for run in session.scalars(query)]


@router.get("/runs/{run_id}")
def get_run(run_id: str, session: Session = Depends(db)) -> dict:
    return bundle(session, _run(session, run_id))


@router.patch("/runs/{run_id}/starred")
def set_starred(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    starred = payload.get("starred")
    if not isinstance(starred, bool):
        raise HTTPException(422, "starred must be true or false")
    run = _run(session, run_id)
    run.starred = starred
    session.commit()
    return run_summary(session, run)


@router.post("/runs/{run_id}/duplicate")
def duplicate(run_id: str, payload: dict | None = None, session: Session = Depends(db)) -> dict:
    run = duplicate_run(session, _run(session, run_id), (payload or {}).get("name"))
    return run_summary(session, run)


@router.post("/runs/{run_id}/cancel")
def cancel(run_id: str, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    run.status = "canceled"
    run.finished_at = utcnow()
    for job in session.scalars(select(Job).where(Job.run_id == run.id, Job.status.in_(["pending", "running"]))):
        job.status = "canceled"
        job.finished_at = utcnow()
    for stage in session.scalars(
        select(PipelineStageRun).where(
            PipelineStageRun.run_id == run.id,
            PipelineStageRun.status.in_(["pending", "running"]),
        )
    ):
        stage.status = "canceled"
        stage.finished_at = utcnow()
    session.commit()
    return run_summary(session, run)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    folder = data_path("runs", run.id)
    session.delete(run)
    session.commit()
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    return {"deleted": run_id}


@router.post("/runs/{run_id}/retry/{stage}")
def retry(run_id: str, stage: str, session: Session = Depends(db)) -> dict:
    try:
        retry_from(session, _run(session, run_id), stage)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return run_summary(session, _run(session, run_id))


@router.get("/runs/{run_id}/requests")
def run_requests(run_id: str, session: Session = Depends(db)) -> list[dict]:
    _run(session, run_id)
    rows = session.scalars(select(ModelRequestRow).where(ModelRequestRow.run_id == run_id)).all()
    payload = []
    for row in rows:
        response = session.scalar(select(ModelResponseRow).where(ModelResponseRow.request_id == row.id))
        payload.append(
            {
                "id": row.id,
                "stage": row.stage,
                "provider_id": row.provider_id,
                "model_id": row.model_id,
                "modality": row.modality,
                "parameters": row.parameters_json,
                "frame_timestamps": row.frame_timestamps,
                "provider_request_id": row.provider_request_id,
                "status": row.status,
                "error": row.error,
                "latency_ms": row.latency_ms,
                "response_text": response.text if response else "",
                "usage": {
                    "input_tokens": response.input_tokens if response else 0,
                    "output_tokens": response.output_tokens if response else 0,
                    "cost_usd": response.cost_usd if response else 0,
                    "price_known": bool(response.price_known) if response else False,
                    "api_key": uses_api_key(row.provider_id, row.parameters_json),
                },
            }
        )
    return payload


@router.get("/runs/{run_id}/scene")
def scene_at(run_id: str, t: float, session: Session = Depends(db)) -> dict:
    data = bundle(session, _run(session, run_id))
    states = [
        item
        for item in data["states"]
        if item["time_start"] - 1e-6 <= t <= item["time_end"] + 1e-6
    ]
    relationships = [
        item
        for item in data["relationships"]
        if item["time_start"] - 1e-6 <= t <= item["time_end"] + 1e-6
    ]
    actions = [item for item in data["actions"] if item["start"] - 1e-6 <= t <= item["end"] + 1e-6]
    ambiguities = [
        item
        for item in data["ambiguities"]
        if item["time_start"] is not None
        and item["time_start"] - 1e-6 <= t <= (item["time_end"] if item["time_end"] is not None else item["time_start"]) + 1e-6
    ]
    observations = []
    for obj in data["objects"]:
        samples = [item for item in data["observations"] if item["object_local_id"] == obj["local_id"]]
        if not samples:
            continue
        nearest = min(samples, key=lambda item: abs(item["timestamp"] - t))
        gap = abs(nearest["timestamp"] - t)
        if gap > 0.75:
            observations.append(
                {
                    "object_local_id": obj["local_id"],
                    "coverage": "gap",
                    "nearest_timestamp": nearest["timestamp"],
                    "note": "No observation covers this instant. The last box was not extrapolated.",
                }
            )
        else:
            observations.append({**nearest, "coverage": "sample" if gap < 0.05 else "nearest_sample", "gap_sec": gap})
    return {
        "t": t,
        "states": states,
        "relationships": relationships,
        "actions": actions,
        "ambiguities": ambiguities,
        "observations": observations,
    }


def _edit(session: Session, run_id: str, fn):
    try:
        fn(session, _run(session, run_id))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return bundle(session, _run(session, run_id))


@router.patch("/runs/{run_id}/objects/{local_id}")
def patch_object(run_id: str, local_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: update_object(db, run, local_id, payload))


@router.post("/runs/{run_id}/objects")
def post_object(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: add_object(db, run, payload.get("label") or "Untitled object", payload.get("description") or ""))


@router.post("/runs/{run_id}/perception/seed")
def seed_perception(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    video = session.get(VideoAsset, run.video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    try:
        box = {
            "timestamp": float(payload["timestamp"]),
            "x": float(payload["x"]),
            "y": float(payload["y"]),
            "w": float(payload["w"]),
            "h": float(payload["h"]),
            "provenance": "manual",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "A seed needs timestamp, x, y, w, and h.") from exc
    if min(box["x"], box["y"], box["w"], box["h"]) < 0 or box["x"] + box["w"] > 1.001 or box["y"] + box["h"] > 1.001:
        raise HTTPException(400, "The seed box must lie inside the frame.")
    config = dict(run.config_json or {})
    perception = dict(config.get("perception") or {})
    perception["manual_regions"] = list(perception.get("manual_regions") or []) + [box]
    if perception.get("strategy") in {None, "", "legacy_motion", "vlm_only"}:
        perception["strategy"] = "temporal"
        perception["detector"] = "manual" if not perception.get("detector") or perception.get("detector") == "motion" else perception["detector"]
    config["perception"] = perception
    run.config_json = config
    stage = session.scalar(
        select(PipelineStageRun).where(
            PipelineStageRun.run_id == run.id,
            PipelineStageRun.stage == "spatial_tracking",
        )
    )
    if stage is None:
        raise HTTPException(400, "This run has no tracking stage.")
    try:
        run_spatial_stage(session, run, video, stage)
    except (TrackingCancelled, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    session.commit()
    return bundle(session, run)


@router.post("/runs/{run_id}/perception/hypotheses/{hypothesis_id}")
def post_hypothesis(run_id: str, hypothesis_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(
        session,
        run_id,
        lambda db, run: decide_hypothesis(db, run, hypothesis_id, payload.get("decision") or ""),
    )


@router.post("/runs/{run_id}/perception/merge")
def post_instance_merge(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(
        session,
        run_id,
        lambda db, run: merge_instances(db, run, payload["keep_id"], payload["drop_id"]),
    )


@router.post("/runs/{run_id}/perception/split")
def post_instance_split(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(
        session,
        run_id,
        lambda db, run: split_instance(db, run, payload["instance_id"], payload.get("timestamps") or []),
    )


@router.post("/runs/{run_id}/objects/merge")
def post_merge(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: merge_objects(db, run, payload["keep_id"], payload["drop_id"]))


@router.post("/runs/{run_id}/objects/{local_id}/split")
def post_split(run_id: str, local_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(
        session,
        run_id,
        lambda db, run: split_object(db, run, local_id, payload.get("timestamps") or [], payload.get("label") or ""),
    )


@router.patch("/runs/{run_id}/actions/{local_id}")
def patch_action(run_id: str, local_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: update_action(db, run, local_id, payload))


@router.post("/runs/{run_id}/actions")
def post_action(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: add_action(db, run, payload))


@router.patch("/runs/{run_id}/steps/{local_id}")
def patch_step(run_id: str, local_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: update_step(db, run, local_id, payload))


@router.post("/runs/{run_id}/steps")
def post_step(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: add_step(db, run, payload.get("name") or "Untitled step"))


@router.post("/runs/{run_id}/dependencies")
def post_dependency(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    return _edit(
        session,
        run_id,
        lambda db, run: add_dependency(
            db, run, payload["from_step"], payload["to_step"], payload.get("category") or "unknown", payload.get("rationale") or ""
        ),
    )


@router.delete("/runs/{run_id}/dependencies")
def delete_dependency(run_id: str, from_step: str, to_step: str, session: Session = Depends(db)) -> dict:
    return _edit(session, run_id, lambda db, run: remove_dependency(db, run, from_step, to_step))


@router.post("/runs/{run_id}/ground-truth")
def promote_ground_truth(run_id: str, payload: dict | None = None, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    if revision is None:
        raise HTTPException(400, "nothing to promote")
    row = GroundTruthDataset(
        run_id=run.id,
        revision_id=revision.id,
        name=(payload or {}).get("name") or f"Ground truth · {run.name}",
        annotator=(payload or {}).get("annotator") or "researcher",
        research_mode=run.research_mode,
        document_json=revision.document_json,
    )
    session.add(row)
    session.commit()
    return {"id": row.id, "name": row.name, "revision_id": revision.id}


@router.get("/runs/{run_id}/ground-truth")
def list_ground_truth(run_id: str, session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(GroundTruthDataset).where(GroundTruthDataset.run_id == run_id)).all()
    return [{"id": row.id, "name": row.name, "annotator": row.annotator, "created_at": row.created_at.isoformat()} for row in rows]


@router.post("/runs/{run_id}/evaluations")
def evaluate(run_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    run = _run(session, run_id)
    truth = session.get(GroundTruthDataset, payload["ground_truth_id"])
    if truth is None:
        raise HTTPException(404, "ground truth not found")
    revision = session.get(AnalysisRevision, payload.get("revision_id") or run.active_revision_id)
    if revision is None:
        raise HTTPException(400, "no predicted revision")
    metrics = evaluate_documents(revision.document_json, truth.document_json, payload.get("tolerances"))
    row = EvaluationResult(
        run_id=run.id,
        ground_truth_id=truth.id,
        predicted_revision_id=revision.id,
        tolerances_json=payload.get("tolerances") or {},
        metrics_json=metrics,
        errors_json=metrics.get("errors") or [],
    )
    session.add(row)
    session.commit()
    return {"id": row.id, "metrics": metrics}


@router.get("/runs/{run_id}/evaluations")
def list_evaluations(run_id: str, session: Session = Depends(db)) -> list[dict]:
    rows = session.scalars(select(EvaluationResult).where(EvaluationResult.run_id == run_id)).all()
    return [{"id": row.id, "metrics": row.metrics_json, "created_at": row.created_at.isoformat()} for row in rows]


@router.post("/comparisons")
def create_comparison(payload: dict, session: Session = Depends(db)) -> dict:
    left = _run(session, payload["left_run_id"])
    right = _run(session, payload["right_run_id"])
    left_rev = session.get(AnalysisRevision, left.active_revision_id) if left.active_revision_id else None
    right_rev = session.get(AnalysisRevision, right.active_revision_id) if right.active_revision_id else None
    if left_rev is None or right_rev is None:
        raise HTTPException(400, "both runs need an analysis")
    report = compare_documents(
        AnalysisDocument.model_validate(left_rev.document_json),
        AnalysisDocument.model_validate(right_rev.document_json),
        payload.get("object_matches"),
    )
    row = ComparisonRow(
        name=payload.get("name") or "Comparison",
        left_run_id=left.id,
        right_run_id=right.id,
        object_matches_json=report["object_matches"],
        report_json=report,
    )
    session.add(row)
    session.commit()
    return {"id": row.id, "report": report}


@router.get("/comparisons/{comparison_id}")
def get_comparison(comparison_id: str, session: Session = Depends(db)) -> dict:
    row = session.get(ComparisonRow, comparison_id)
    if row is None:
        raise HTTPException(404, "comparison not found")
    return {
        "id": row.id,
        "name": row.name,
        "left_run_id": row.left_run_id,
        "right_run_id": row.right_run_id,
        "report": row.report_json,
    }


@router.post("/comparisons/{comparison_id}/matches")
def set_matches(comparison_id: str, payload: dict, session: Session = Depends(db)) -> dict:
    row = session.get(ComparisonRow, comparison_id)
    if row is None:
        raise HTTPException(404, "comparison not found")
    left = _run(session, row.left_run_id)
    right = _run(session, row.right_run_id)
    report = compare_documents(
        AnalysisDocument.model_validate(session.get(AnalysisRevision, left.active_revision_id).document_json),
        AnalysisDocument.model_validate(session.get(AnalysisRevision, right.active_revision_id).document_json),
        payload.get("object_matches") or [],
    )
    row.object_matches_json = report["object_matches"]
    row.report_json = report
    session.commit()
    return {"id": row.id, "report": report}


@router.get("/runs/{run_id}/export")
def download_export(run_id: str, session: Session = Depends(db)) -> FileResponse:
    run = _run(session, run_id)
    dest = data_path("exports", f"{run.id}.zip")
    export_run(session, run, dest)
    return FileResponse(dest, filename=f"learnloop-{run.id}.zip")


@router.post("/imports")
async def import_archive(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(db),
) -> dict:
    dest = data_path("imports", file.filename or "experiment.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    try:
        run = import_run(session, project_id, dest)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return run_summary(session, run)


@router.get("/runs/{run_id}/analysis.json")
def analysis_json(run_id: str, session: Session = Depends(db)) -> Response:
    run = _run(session, run_id)
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    if revision is None:
        raise HTTPException(404, "no analysis")
    return Response(content=json_dumps(revision.document_json), media_type="application/json")


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)
