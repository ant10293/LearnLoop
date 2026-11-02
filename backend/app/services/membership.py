"""Move a video between the library and one project."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.media import Project, VideoAsset
from app.services.ingest import ensure_default_session


class MembershipError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def attach_videos(session: Session, project_id: str, video_ids: list[str]) -> list[VideoAsset]:
    project = session.get(Project, project_id)
    if project is None:
        raise MembershipError("project not found", 404)
    ordered = _unique(video_ids)
    if not ordered:
        raise MembershipError("choose at least one video", 400)
    videos: list[VideoAsset] = []
    for video_id in ordered:
        video = session.get(VideoAsset, video_id)
        if video is None:
            raise MembershipError("video not found", 404)
        if video.project_id not in (None, project.id):
            raise MembershipError(f"{video.original_filename} is already in a project", 409)
        videos.append(video)
    moving = [video for video in videos if video.project_id != project.id]
    _reject_duplicate_files(session, project.id, moving)
    next_order = _next_order(session, project.id) if not project.is_task else None
    for video in moving:
        video.project_id = project.id
        if project.is_task:
            video.session_id = ensure_default_session(session, project)
            video.order_index = None
            video.order_locked = False
            video.order_method = ""
            video.order_ambiguous = False
            continue
        video.session_id = None
        if video.order_index is None and next_order is not None:
            video.order_index = next_order
            next_order += 1
    session.commit()
    for video in videos:
        session.refresh(video)
    return videos


def detach_video(session: Session, video_id: str) -> VideoAsset:
    video = session.get(VideoAsset, video_id)
    if video is None:
        raise MembershipError("video not found", 404)
    if video.project_id is None:
        return video
    video.project_id = None
    video.session_id = None
    video.order_index = None
    video.order_locked = False
    video.order_method = ""
    video.order_ambiguous = False
    session.commit()
    session.refresh(video)
    return video


def _next_order(session: Session, project_id: str) -> float | None:
    peak = session.scalar(select(func.max(VideoAsset.order_index)).where(VideoAsset.project_id == project_id))
    if peak is None:
        return None
    return float(peak) + 1


def _unique(video_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    for video_id in video_ids:
        if video_id not in ordered:
            ordered.append(video_id)
    return ordered


def _reject_duplicate_files(session: Session, project_id: str, moving: list[VideoAsset]) -> None:
    taken = {
        video.checksum_sha256
        for video in session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id))
        if video.checksum_sha256
    }
    for video in moving:
        checksum = video.checksum_sha256
        if not checksum:
            continue
        if checksum in taken:
            raise MembershipError(f"{video.original_filename} is already in this project", 409)
        taken.add(checksum)
