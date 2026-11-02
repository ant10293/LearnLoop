"""Read a task project for the API and the export archive."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.serialize import video_dict
from app.models.domain import ActionRow
from app.models.media import Project, VideoAsset
from app.models.runs import AnalysisRun
from app.models.sessions import (
    CanonicalObject,
    IdentityLink,
    InterSessionTransition,
    OrderHypothesis,
    ProjectRevision,
    RecordingSession,
)


def task_bundle(session: Session, project_id: str) -> dict | None:
    project = session.get(Project, project_id)
    if project is None or not project.is_task:
        return None
    sessions = list(
        session.scalars(
            select(RecordingSession).where(RecordingSession.project_id == project_id).order_by(RecordingSession.ordinal)
        )
    )
    videos = list(session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id)))
    runs = list(
        session.scalars(
            select(AnalysisRun).where(AnalysisRun.project_id == project_id, AnalysisRun.status == "completed")
        )
    )
    latest: dict[str, AnalysisRun] = {}
    for run in runs:
        current = latest.get(run.video_id)
        if current is None or run.created_at >= current.created_at:
            latest[run.video_id] = run
    clips = []
    for video in videos:
        run = latest.get(video.id)
        actions = []
        if run and run.active_revision_id:
            for action in session.scalars(select(ActionRow).where(ActionRow.revision_id == run.active_revision_id)):
                actions.append(
                    {
                        "local_id": action.local_id,
                        "action_type": action.action_type,
                        "description": action.description,
                        "start": action.start,
                        "end": action.end,
                        "evidence_class": action.evidence_class,
                        "run_id": run.id,
                    }
                )
        payload = video_dict(video)
        payload.update(
            {
                "session_id": video.session_id,
                "research_mode": video.research_mode,
                "capture_time_raw": video.capture_time_raw,
                "capture_time_utc": video.capture_time_utc,
                "capture_timezone": video.capture_timezone,
                "timestamp_source": video.timestamp_source,
                "timestamp_reliability": video.timestamp_reliability,
                "timestamp_note": video.timestamp_note,
                "file_mtime": video.file_mtime,
                "uploaded_at": video.created_at.isoformat(),
                "order_index": video.order_index,
                "order_locked": video.order_locked,
                "order_method": video.order_method,
                "order_ambiguous": video.order_ambiguous,
                "run_id": run.id if run else "",
                "actions": actions,
            }
        )
        clips.append(payload)
    revision = session.scalar(
        select(ProjectRevision).where(ProjectRevision.project_id == project_id).order_by(ProjectRevision.version.desc())
    )
    revisions = list(
        session.scalars(
            select(ProjectRevision).where(ProjectRevision.project_id == project_id).order_by(ProjectRevision.version)
        )
    )
    hypotheses = list(
        session.scalars(select(OrderHypothesis).where(OrderHypothesis.project_id == project_id).order_by(OrderHypothesis.created_at.desc()))
    )
    return {
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "is_task": True,
            "created_at": project.created_at.isoformat(),
        },
        "sessions": [
            {
                "id": row.id,
                "name": row.name,
                "ordinal": row.ordinal,
                "locked": row.locked,
                "notes": row.notes,
                "video_ids": [clip["id"] for clip in clips if clip["session_id"] == row.id],
            }
            for row in sessions
        ],
        "clips": clips,
        "order": (revision.document_json.get("order") if revision else {"ranked": [], "unordered": [], "ambiguous_ids": [], "fact": False, "notes": []}),
        "context_hypothesis": (revision.document_json.get("context_hypothesis") if revision else {"fact": False, "pairs": [], "evidence": []}),
        "canonical_objects": [_canonical(session, row) for row in session.scalars(select(CanonicalObject).where(CanonicalObject.project_id == project_id))],
        "links": [_link(row) for row in session.scalars(select(IdentityLink).where(IdentityLink.project_id == project_id))],
        "transitions": [
            _transition(row)
            for row in session.scalars(select(InterSessionTransition).where(InterSessionTransition.project_id == project_id))
        ],
        "workflow": revision.document_json.get("workflow") if revision else {"steps": [], "continuous_sequence": False},
        "revision": _revision(revision) if revision else None,
        "revisions": [_revision(row) for row in revisions],
        "hypotheses": [
            {
                "id": row.id,
                "method": row.method,
                "pairs": row.ranked_json,
                "evidence": row.evidence_json,
                "accepted": row.accepted,
                "fact": False,
            }
            for row in hypotheses[:8]
        ],
        "time_basis": {
            "project_time": "capture timestamps, when the container provided a timezone",
            "clip_time": "seconds from the start of that recording",
            "file_mtime_is_capture_time": False,
            "upload_time_is_capture_time": False,
        },
    }


def _canonical(session: Session, row: CanonicalObject) -> dict:
    links = list(session.scalars(select(IdentityLink).where(IdentityLink.canonical_id == row.id)))
    return {
        "id": row.id,
        "local_id": row.local_id,
        "label": row.label,
        "description": row.description,
        "color": row.color,
        "uncertain": row.uncertain,
        "history": [
            {
                "video_id": link.video_id,
                "object_local_id": link.object_local_id,
                "label": link.label,
                "state_before": link.state_before,
                "state_after": link.state_after,
                "image_url": link.image_url,
                "status": link.status,
                "uncertain": link.uncertain,
            }
            for link in links
            if link.status != "rejected"
        ],
    }


def _link(row: IdentityLink) -> dict:
    return {
        "id": row.id,
        "canonical_id": row.canonical_id,
        "video_id": row.video_id,
        "run_id": row.run_id,
        "object_local_id": row.object_local_id,
        "label": row.label,
        "state_before": row.state_before,
        "state_after": row.state_after,
        "image_url": row.image_url,
        "method": row.method,
        "score": row.score,
        "status": row.status,
        "uncertain": row.uncertain,
        "note": row.note,
        "evidence_class": row.evidence_class,
        "provenance": row.provenance,
    }


def _transition(row: InterSessionTransition) -> dict:
    return {
        "id": row.id,
        "canonical_id": row.canonical_id,
        "from_video_id": row.from_video_id,
        "to_video_id": row.to_video_id,
        "observed_change": row.observed_change,
        "hypothesis": row.hypothesis,
        "from_state": row.from_state,
        "to_state": row.to_state,
        "evidence_class": row.evidence_class,
        "provenance": row.provenance,
        "verified": row.verified,
        "gap_note": row.gap_note,
    }


def _revision(row: ProjectRevision) -> dict:
    return {
        "id": row.id,
        "version": row.version,
        "parent_id": row.parent_id,
        "trigger": row.trigger,
        "change_summary": row.change_summary,
        "created_at": row.created_at.isoformat(),
        "document": row.document_json,
    }
