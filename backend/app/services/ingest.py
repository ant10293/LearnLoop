"""Video upload, probing, proxying, and preview frames."""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.media import FrameAsset, Project, VideoAsset
from app.models.sessions import RecordingSession
from app.models.util import new_id
from app.services.storage import (
    data_path,
    ensure_data_dirs,
    external_ref,
    is_external,
    relative,
    resolve_stored,
    safe_filename,
)
from app.video.capture_time import read_capture
from app.video.frames import contact_sheet, extract_frame, make_proxy, thumbnail
from app.video.probe import (
    VideoError,
    browser_compatible,
    presentation_timestamps,
    probe,
    sha256_file,
    snap_timestamps,
)
from app.video.sampling import SAME_MOMENT_SEC, plan_uniform

log = logging.getLogger("learnloop.ingest")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
Progress = Callable[[float, str], None]

_lock = threading.Lock()
_inflight: set[str] = set()
_cancel: set[str] = set()


class IngestCancelled(Exception):
    """The video row was removed while analysis was still running."""


def ingest_file(
    session: Session,
    project_id: str | None,
    filename: str,
    source: Path,
    research_mode: str = "",
    session_id: str | None = None,
    video: VideoAsset | None = None,
    on_progress: Progress | None = None,
    produced: list[Path] | None = None,
    reference: bool = False,
) -> VideoAsset:
    ensure_data_dirs()
    project = _project_or_library(session, project_id)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise VideoError(f"unsupported type {suffix or 'unknown'}; use mp4, mov, or webm")
    if video is not None:
        _drop_preview_frames(session, video.id)
    _step(on_progress, 8, "Hashing")
    hashed = 8.0

    def on_hashed(read: int, total: int) -> None:
        nonlocal hashed
        if total <= 0:
            return
        mark = 8 + (read / total) * 8
        if read < total and mark < hashed + 1:
            return
        hashed = mark
        _step(on_progress, mark, "Hashing")

    checksum = sha256_file(source, on_hashed)
    existing = session.scalar(
        select(VideoAsset).where(
            _same_project(project_id),
            VideoAsset.checksum_sha256 == checksum,
            VideoAsset.id != (video.id if video is not None else ""),
        )
    )
    if existing is not None:
        if video is not None:
            raise VideoError("this file is already in the project" if project_id else "this file is already in the library")
        if existing.timestamp_source in {"", "absent"} and not existing.capture_time_raw:
            _apply_capture(existing, source)
            if project is not None and project.is_task and not existing.session_id:
                existing.session_id = session_id or _default_session(session, project)
            session.commit()
        return existing
    _step(on_progress, 18, "Probing")
    meta = probe(source)
    if meta.duration_sec <= 0 or meta.width <= 0:
        raise VideoError("the file has no usable video duration or dimensions")
    _step(on_progress, 26, "Storing")
    referencing = reference or (video is not None and is_external(video.stored_path))
    if referencing:
        original = source.resolve()
        prior = None
    else:
        stored_name = f"{checksum[:12]}_{safe_filename(filename)}"
        if not stored_name.lower().endswith(suffix):
            stored_name += suffix
        original = data_path("originals", stored_name)
        if not original.exists():
            shutil.copy2(source, original)
            _keep(produced, original)
        prior = resolve_stored(video.stored_path) if video is not None and video.stored_path else None
    proxy_rel = ""
    if not browser_compatible(meta):
        _step(on_progress, 32, "Preparing playback")
        proxy = data_path("proxies", f"{checksum[:12]}.mp4")
        if not proxy.exists():
            make_proxy(original, proxy)
            _keep(produced, proxy)
        proxy_rel = relative(proxy)
    _step(on_progress, 34, "Reading frame times")
    pts_cache = data_path("originals", f"{checksum[:12]}.pts.json")
    pts = presentation_timestamps(original, pts_cache)
    stamp = read_capture(original)
    if video is None:
        video = VideoAsset(id=new_id(), project_id=project_id, original_filename=safe_filename(filename))
        session.add(video)
    video.original_filename = safe_filename(filename)
    video.stored_path = external_ref(original) if referencing else relative(original)
    video.proxy_path = proxy_rel
    video.checksum_sha256 = checksum
    video.mime_type = meta.mime_hint
    video.container = meta.container
    video.codec = meta.codec
    video.duration_sec = meta.duration_sec
    video.width = meta.width
    video.height = meta.height
    video.display_width = meta.display_width
    video.display_height = meta.display_height
    video.fps = meta.fps
    video.avg_fps = meta.avg_fps
    video.time_base = meta.time_base
    video.rotation_deg = meta.rotation_deg
    video.variable_frame_rate = meta.variable_frame_rate
    video.frame_count = meta.frame_count or len(pts)
    video.size_bytes = original.stat().st_size
    if project is None:
        video.session_id = session_id or video.session_id
    else:
        video.session_id = session_id or video.session_id or _default_session(session, project)
    video.research_mode = research_mode or video.research_mode
    video.capture_time_raw = stamp.raw
    video.capture_time_utc = stamp.utc_iso
    video.capture_timezone = stamp.timezone_name
    video.timestamp_source = stamp.source
    video.timestamp_reliability = stamp.reliability
    video.timestamp_note = stamp.note
    video.file_mtime = stamp.file_mtime
    session.flush()
    if on_progress is not None:
        if not _still_there(session, video.id):
            raise IngestCancelled()
        session.commit()
        if prior is not None and prior.is_file() and prior.resolve() != original.resolve():
            prior.unlink()
            prior = None
    preview = plan_uniform(meta.duration_sec, 1.0, pts)[:40]
    frame_paths: list[Path] = []
    total = max(len(preview), 1)
    for ordinal, sample in enumerate(preview):
        _step(on_progress, 36 + (54 * ordinal / total), f"Extracting frames ({ordinal + 1}/{len(preview)})")
        dest = data_path("frames", video.id, f"{ordinal:04d}_{sample.timestamp:.3f}.jpg")
        extract_frame(original, sample.timestamp, dest, accurate=True)
        _keep(produced, dest)
        frame_paths.append(dest)
        session.add(
            FrameAsset(
                video_id=video.id,
                run_id=None,
                timestamp=sample.timestamp,
                path=relative(dest),
                width=meta.display_width,
                height=meta.display_height,
                reason="upload_preview",
                strategy="uniform",
                ordinal=ordinal,
            )
        )
    if preview:
        _step(on_progress, 92, "Building preview")
        thumb = data_path("thumbs", f"{video.id}.jpg")
        thumbnail(original, preview[min(1, len(preview) - 1)].timestamp, thumb)
        _keep(produced, thumb)
        video.thumbnail_path = relative(thumb)
        sheet = data_path("sheets", f"{video.id}.jpg")
        contact_sheet(frame_paths, sheet)
        _keep(produced, sheet)
        video.contact_sheet_path = relative(sheet)
    if project is not None and not project.is_task and video.order_index is None:
        peak = session.scalar(
            select(func.max(VideoAsset.order_index)).where(
                VideoAsset.project_id == project_id,
                VideoAsset.id != video.id,
            )
        )
        if peak is not None:
            video.order_index = float(peak) + 1
    if on_progress is not None and not _still_there(session, video.id):
        raise IngestCancelled()
    video.ingest_status = "ready"
    video.ingest_progress = 100
    video.ingest_detail = ""
    video.ingest_error = ""
    session.commit()
    if prior is not None and prior.is_file() and prior.resolve() != original.resolve():
        prior.unlink()
    session.refresh(video)
    return video


def accept_upload(session: Session, project_id: str | None, filename: str, source: Path) -> VideoAsset:
    """Store the bytes and return before probing or frame extraction."""
    ensure_data_dirs()
    _project_or_library(session, project_id)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise VideoError(f"unsupported type {suffix or 'unknown'}; use mp4, mov, or webm")
    if not source.is_file() or source.stat().st_size <= 0:
        raise VideoError("empty upload")
    video_id = new_id()
    dest = data_path("imports", f"{video_id}_{safe_filename(filename)}")
    if source.resolve() != dest.resolve():
        shutil.move(str(source), dest)
    video = VideoAsset(
        id=video_id,
        project_id=project_id,
        original_filename=safe_filename(filename),
        stored_path=relative(dest),
        checksum_sha256="",
        size_bytes=dest.stat().st_size,
        ingest_status="processing",
        ingest_progress=0,
        ingest_detail="Queued",
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    return video


def referenced_file(raw: str) -> Path:
    text = raw.strip().strip("'\"")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise VideoError("use the full path to the video")
    path = path.resolve()
    if not path.is_file():
        raise VideoError("that path is not a video file")
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise VideoError(f"unsupported type {suffix or 'unknown'}; use mp4, mov, or webm")
    if path.stat().st_size <= 0:
        raise VideoError("empty file")
    return path


def accept_reference(session: Session, project_id: str | None, source: Path) -> VideoAsset:
    """Remember a file where it already is. Nothing is copied."""
    ensure_data_dirs()
    _project_or_library(session, project_id)
    path = referenced_file(str(source))
    video = VideoAsset(
        id=new_id(),
        project_id=project_id,
        original_filename=safe_filename(path.name),
        stored_path=external_ref(path),
        checksum_sha256="",
        size_bytes=path.stat().st_size,
        ingest_status="processing",
        ingest_progress=0,
        ingest_detail="Queued",
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    try:
        video.thumbnail_path = _preview_still(path, video.id)
        session.commit()
    except VideoError as exc:
        log.warning("preview still for %s failed: %s", video.id, exc)
    return video


def _preview_still(source: Path, video_id: str) -> str:
    """One frame for the progress row. The full ingest replaces this file later."""
    dest = data_path("thumbs", f"{video_id}.jpg")
    try:
        thumbnail(source, 1.0, dest)
    except VideoError:
        thumbnail(source, 0.0, dest)
    return relative(dest)


def repair_preview_paths() -> int:
    """Fill thumbnail and contact-sheet paths when the files exist but the row lost them."""
    from app.db import SessionLocal

    session = SessionLocal()
    fixed = 0
    try:
        for video in session.scalars(select(VideoAsset)):
            changed = False
            thumb = data_path("thumbs", f"{video.id}.jpg")
            sheet = data_path("sheets", f"{video.id}.jpg")
            if not video.thumbnail_path and thumb.is_file():
                video.thumbnail_path = relative(thumb)
                changed = True
            if not video.contact_sheet_path and sheet.is_file():
                video.contact_sheet_path = relative(sheet)
                changed = True
            if changed:
                fixed += 1
        if fixed:
            session.commit()
    finally:
        session.close()
    return fixed


def schedule_ingest(video_id: str) -> None:
    with _lock:
        if video_id in _inflight:
            return
        _inflight.add(video_id)

    def run() -> None:
        try:
            process_accepted(video_id)
        finally:
            with _lock:
                _inflight.discard(video_id)
                _cancel.discard(video_id)

    threading.Thread(target=run, name=f"ingest-{video_id[:8]}", daemon=True).start()


def cancel_ingest(video_id: str) -> None:
    with _lock:
        _cancel.add(video_id)


def ingest_cancelled(video_id: str) -> bool:
    with _lock:
        return video_id in _cancel


def resume_processing() -> int:
    """Restart analysis that was still running when the process stopped."""
    from app.db import SessionLocal

    session = SessionLocal()
    try:
        ids = list(session.scalars(select(VideoAsset.id).where(VideoAsset.ingest_status == "processing")))
    finally:
        session.close()
    for video_id in ids:
        schedule_ingest(video_id)
    return len(ids)


def process_accepted(video_id: str) -> None:
    from app.db import SessionLocal

    session = SessionLocal()
    produced: list[Path] = []
    try:
        video = session.get(VideoAsset, video_id)
        if video is None or video.ingest_status != "processing":
            return
        if ingest_cancelled(video_id):
            raise IngestCancelled()
        source = resolve_stored(video.stored_path)

        def on_progress(progress: float, detail: str) -> None:
            if not _still_there(session, video_id):
                raise IngestCancelled()
            current = session.get(VideoAsset, video_id)
            if current is None:
                raise IngestCancelled()
            current.ingest_status = "processing"
            current.ingest_progress = progress
            current.ingest_detail = detail
            session.commit()

        ingest_file(
            session,
            video.project_id,
            video.original_filename,
            source,
            video.research_mode,
            video.session_id,
            video=video,
            on_progress=on_progress,
            produced=produced,
        )
    except IngestCancelled:
        session.rollback()
        _cleanup_if_gone(video_id, produced)
    except Exception as exc:
        session.rollback()
        log.warning("ingest %s failed: %s", video_id, exc)
        _mark_failed(video_id, exc)
        _cleanup_if_gone(video_id, produced)
    finally:
        session.close()


def _step(on_progress: Progress | None, progress: float, detail: str) -> None:
    if on_progress is not None:
        on_progress(progress, detail)


def _keep(produced: list[Path] | None, path: Path) -> None:
    if produced is not None:
        produced.append(path)


def _still_there(session: Session, video_id: str) -> bool:
    """True when this analysis should keep going.

    Only the id is read. Reloading the whole row would discard thumbnail,
    contact sheet, and order changes that have not been committed yet.
    """
    if ingest_cancelled(video_id):
        return False
    return session.scalar(select(VideoAsset.id).where(VideoAsset.id == video_id)) is not None


def _drop_preview_frames(session: Session, video_id: str) -> None:
    rows = list(
        session.scalars(
            select(FrameAsset).where(FrameAsset.video_id == video_id, FrameAsset.run_id.is_(None), FrameAsset.reason == "upload_preview")
        )
    )
    for frame in rows:
        try:
            path = resolve_stored(frame.path)
        except ValueError:
            path = None
        if path is not None and path.is_file():
            path.unlink()
        session.delete(frame)
    if rows:
        session.flush()


def _mark_failed(video_id: str, exc: Exception) -> None:
    from app.db import SessionLocal

    if ingest_cancelled(video_id):
        return
    session = SessionLocal()
    try:
        video = session.get(VideoAsset, video_id)
        if video is None or video.ingest_status == "ready":
            return
        video.ingest_status = "failed"
        video.ingest_error = str(exc)
        video.ingest_detail = "Failed"
        session.commit()
    finally:
        session.close()


def _cleanup_if_gone(video_id: str, produced: list[Path]) -> None:
    from app.db import SessionLocal

    session = SessionLocal()
    try:
        if session.get(VideoAsset, video_id) is not None:
            return
    finally:
        session.close()
    for path in produced:
        if path.is_file():
            path.unlink()
    frames = data_path("frames", video_id)
    if frames.is_dir():
        shutil.rmtree(frames)


def _apply_capture(video: VideoAsset, source: Path) -> None:
    stamp = read_capture(source)
    video.capture_time_raw = stamp.raw
    video.capture_time_utc = stamp.utc_iso
    video.capture_timezone = stamp.timezone_name
    video.timestamp_source = stamp.source
    video.timestamp_reliability = stamp.reliability
    video.timestamp_note = stamp.note
    video.file_mtime = stamp.file_mtime


def ensure_default_session(session: Session, project: Project) -> str | None:
    """The session a clip joins. A single sitting has none."""
    return _default_session(session, project)


def _project_or_library(session: Session, project_id: str | None) -> Project | None:
    if not project_id:
        return None
    project = session.get(Project, project_id)
    if project is None:
        raise VideoError("project not found")
    return project


def _same_project(project_id: str | None):
    if project_id:
        return VideoAsset.project_id == project_id
    return VideoAsset.project_id.is_(None)


def _default_session(session: Session, project: Project) -> str | None:
    if not project.is_task:
        return None
    row = session.scalar(
        select(RecordingSession).where(RecordingSession.project_id == project.id).order_by(RecordingSession.ordinal)
    )
    if row is None:
        row = RecordingSession(project_id=project.id, name="Session 1", ordinal=0)
        session.add(row)
        session.flush()
    return row.id


def playback_path(video: VideoAsset) -> Path:
    if video.proxy_path:
        return resolve_stored(video.proxy_path)
    return resolve_stored(video.stored_path)


def sample_run_frames(
    session: Session,
    video: VideoAsset,
    run_id: str,
    samples,
) -> list[FrameAsset]:
    original = resolve_stored(video.stored_path)
    rows: list[FrameAsset] = []
    existing = list(
        session.scalars(select(FrameAsset).where(FrameAsset.video_id == video.id, FrameAsset.run_id == run_id))
    )
    used: set[str] = set()
    for ordinal, sample in enumerate(samples):
        match = _closest_frame(existing, sample.timestamp, used)
        if match is not None:
            match.ordinal = ordinal
            used.add(match.id)
            rows.append(match)
            continue
        dest = data_path("frames", video.id, run_id, f"{ordinal:04d}_{sample.timestamp:.3f}.jpg")
        if not dest.exists():
            snapped = snap_timestamps([sample.timestamp], [])[0]
            extract_frame(original, snapped, dest, accurate=True)
        row = FrameAsset(
            video_id=video.id,
            run_id=run_id,
            timestamp=sample.timestamp,
            path=relative(dest),
            width=video.display_width,
            height=video.display_height,
            reason=sample.reason,
            strategy=sample.reason,
            motion_score=sample.motion_score,
            sharpness=sample.sharpness,
            scene_change=sample.scene_change,
            ordinal=ordinal,
        )
        session.add(row)
        used.add(row.id)
        rows.append(row)
    for frame in existing:
        if frame.id not in used:
            session.delete(frame)
    session.flush()
    return rows


def _closest_frame(frames: list[FrameAsset], timestamp: float, used: set[str]) -> FrameAsset | None:
    """A frame a few milliseconds away is the same picture, not a second sample."""
    best: FrameAsset | None = None
    best_gap = SAME_MOMENT_SEC
    for frame in frames:
        if frame.id in used:
            continue
        gap = abs(float(frame.timestamp) - timestamp)
        if gap < best_gap:
            best = frame
            best_gap = gap
    return best
