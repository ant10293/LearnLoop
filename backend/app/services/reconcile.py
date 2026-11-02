"""Project reconciliation. Clip analyses stay cached; only the project reading is rewritten."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import DetectedObject, ObjectImage
from app.models.media import VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun
from app.models.sessions import (
    CanonicalObject,
    IdentityLink,
    InterSessionTransition,
    OrderHypothesis,
    ProjectRevision,
    RecordingSession,
)
from app.models.util import new_id
from app.services.identity import observed_transition, propose_links
from app.services.ordering import context_hypothesis, metadata_order


def reconcile(session: Session, project_id: str, trigger: str) -> ProjectRevision:
    videos = list(session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id)))
    clip_rows = [_clip_payload(session, video) for video in videos]
    order = metadata_order(clip_rows)
    hypothesis = context_hypothesis(clip_rows, {row["id"]: row["objects"] for row in clip_rows})
    _write_order_flags(videos, order)
    session.add(
        OrderHypothesis(
            project_id=project_id,
            method="context",
            ranked_json=hypothesis["pairs"],
            evidence_json=hypothesis["evidence"],
            accepted=False,
        )
    )
    kept = _preserve_adjudication(session, project_id)
    proposals = propose_links(
        [{"video_id": row["id"], "run_id": row["run_id"], "objects": row["objects"]} for row in clip_rows]
    )
    canonicals, links = _materialize_identity(session, project_id, proposals, kept)
    transitions = _materialize_transitions(session, project_id, canonicals, links, order)
    workflow = _workflow(session, project_id, videos, order, transitions, hypothesis)
    previous = session.scalar(
        select(ProjectRevision).where(ProjectRevision.project_id == project_id).order_by(ProjectRevision.version.desc())
    )
    document = {
        "order": order,
        "context_hypothesis": hypothesis,
        "canonical_ids": [row.local_id for row in canonicals],
        "link_count": len(links),
        "transition_ids": [row.id for row in transitions],
        "workflow": workflow,
        "clip_run_ids": {row["id"]: row["run_id"] for row in clip_rows},
    }
    revision = ProjectRevision(
        project_id=project_id,
        version=(previous.version + 1) if previous else 1,
        parent_id=previous.id if previous else None,
        trigger=trigger,
        change_summary=_diff(previous.document_json if previous else {}, document),
        document_json=document,
    )
    session.add(revision)
    session.commit()
    return revision


def _clip_payload(session: Session, video: VideoAsset) -> dict:
    run = session.scalar(
        select(AnalysisRun)
        .where(AnalysisRun.video_id == video.id, AnalysisRun.status == "completed")
        .order_by(AnalysisRun.created_at.desc())
    )
    objects = []
    if run:
        revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
        if revision:
            for obj in session.scalars(select(DetectedObject).where(DetectedObject.revision_id == revision.id)):
                image = session.scalar(
                    select(ObjectImage).where(ObjectImage.object_id == obj.id, ObjectImage.kind == "primary")
                )
                objects.append(
                    {
                        "local_id": obj.local_id,
                        "label": obj.label,
                        "characteristics": obj.visual_characteristics,
                        "initial_state": obj.initial_state,
                        "final_state": obj.final_state,
                        "image_url": f"/api/media/object-images/{image.id}" if image else "",
                    }
                )
    return {
        "id": video.id,
        "filename": video.original_filename,
        "capture_time_utc": video.capture_time_utc,
        "timestamp_reliability": video.timestamp_reliability,
        "order_locked": video.order_locked,
        "order_index": video.order_index,
        "run_id": run.id if run else "",
        "objects": objects,
        "duration": video.duration_sec,
        "session_id": video.session_id,
    }


def _write_order_flags(videos: list[VideoAsset], order: dict) -> None:
    placed = {video_id: index for index, group in enumerate(order["ranked"]) for video_id in group}
    by_id = {video.id: video for video in videos}
    for video_id, index in placed.items():
        video = by_id[video_id]
        if video.order_locked:
            continue
        video.order_index = float(index)
        video.order_method = order["method"]
        video.order_ambiguous = video_id in order["ambiguous_ids"]
    for video in videos:
        if video.order_locked or video.id in placed:
            continue
        video.order_index = None
        video.order_method = ""
        video.order_ambiguous = True


def _preserve_adjudication(session: Session, project_id: str) -> dict:
    links = list(session.scalars(select(IdentityLink).where(IdentityLink.project_id == project_id)))
    confirmed = [link for link in links if link.status == "confirmed"]
    rejected = {(link.video_id, link.object_local_id) for link in links if link.status == "rejected"}
    for link in links:
        if link.status == "proposed":
            session.delete(link)
    transitions = list(
        session.scalars(select(InterSessionTransition).where(InterSessionTransition.project_id == project_id))
    )
    for row in transitions:
        if not row.verified:
            session.delete(row)
    session.flush()
    return {"confirmed": confirmed, "rejected": rejected}


def _materialize_identity(session: Session, project_id: str, proposals: list[dict], kept: dict) -> tuple[list, list]:
    confirmed = list(kept["confirmed"])
    rejected = kept["rejected"]
    by_member = {(link.video_id, link.object_local_id): link.canonical_id for link in confirmed}
    canonicals = {
        row.id: row for row in session.scalars(select(CanonicalObject).where(CanonicalObject.project_id == project_id))
    }
    links = list(confirmed)
    next_index = len(canonicals) + 1
    for proposal in proposals:
        members = [member for member in proposal["members"] if (member["video_id"], member["local_id"]) not in rejected]
        unique: list[dict] = []
        seen: set[str] = set()
        for member in members:
            if member["video_id"] in seen:
                continue
            seen.add(member["video_id"])
            unique.append(member)
        attached = [member for member in unique if (member["video_id"], member["local_id"]) in by_member]
        if len(unique) < 2 and not attached:
            continue
        canonical_id = by_member[(attached[0]["video_id"], attached[0]["local_id"])] if attached else None
        if canonical_id is None:
            canonical = CanonicalObject(
                project_id=project_id,
                local_id=f"Object_{next_index}",
                label=proposal["label"],
                description=proposal["note"],
                color=proposal["color"],
                uncertain=True,
            )
            next_index += 1
            session.add(canonical)
            session.flush()
            canonicals[canonical.id] = canonical
            canonical_id = canonical.id
        for member in unique:
            key = (member["video_id"], member["local_id"])
            if key in by_member:
                continue
            link = IdentityLink(
                project_id=project_id,
                canonical_id=canonical_id,
                video_id=member["video_id"],
                run_id=member.get("run_id", ""),
                object_local_id=member["local_id"],
                label=member.get("label", ""),
                state_before=member.get("initial_state", ""),
                state_after=member.get("final_state", ""),
                image_url=member.get("image_url", ""),
                method="context",
                score=float(member.get("score") or 0),
                status="proposed",
                uncertain=True,
                note=proposal["note"],
            )
            session.add(link)
            links.append(link)
            by_member[key] = canonical_id
    session.flush()
    linked_ids = {link.canonical_id for link in links}
    for canonical in list(canonicals.values()):
        if canonical.id not in linked_ids:
            session.delete(canonical)
            canonicals.pop(canonical.id, None)
    session.flush()
    return list(canonicals.values()), links


def _materialize_transitions(session, project_id, canonicals, links, order) -> list:
    sequence = [video_id for group in order["ranked"] for video_id in group]
    rank = {video_id: index for index, video_id in enumerate(sequence)}
    verified = list(
        session.scalars(
            select(InterSessionTransition).where(
                InterSessionTransition.project_id == project_id, InterSessionTransition.verified.is_(True)
            )
        )
    )
    rows = list(verified)
    by_canonical: dict[str, list[IdentityLink]] = {}
    for link in links:
        if link.status == "rejected":
            continue
        by_canonical.setdefault(link.canonical_id, []).append(link)
    for canonical in canonicals:
        members = sorted(
            by_canonical.get(canonical.id, []),
            key=lambda link: rank.get(link.video_id, 10_000),
        )
        ordered = [link for link in members if link.video_id in rank]
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if any(row.from_video_id == earlier.video_id and row.to_video_id == later.video_id and row.verified for row in verified):
                continue
            payload = observed_transition(
                {"final_state": earlier.state_after, "label": earlier.label},
                {"initial_state": later.state_before, "label": later.label},
            )
            row = InterSessionTransition(
                project_id=project_id,
                canonical_id=canonical.id,
                from_video_id=earlier.video_id,
                to_video_id=later.video_id,
                **payload,
            )
            session.add(row)
            rows.append(row)
    session.flush()
    return rows


def _workflow(session, project_id, videos, order, transitions, hypothesis) -> dict:
    sessions = list(session.scalars(select(RecordingSession).where(RecordingSession.project_id == project_id)))
    by_session: dict[str, list[VideoAsset]] = {}
    for video in videos:
        by_session.setdefault(video.session_id or "", []).append(video)
    steps = []
    for recording in sessions:
        members = by_session.get(recording.id, [])
        steps.append(
            {
                "id": f"session-{recording.id}",
                "kind": "session",
                "label": recording.name,
                "video_ids": [video.id for video in members],
                "evidence_class": "DIRECTLY_OBSERVED",
                "note": "Clips inside this session keep their own action timelines.",
            }
        )
    previous = None
    for group in order["ranked"]:
        if len(group) > 1:
            steps.append(
                {
                    "id": f"parallel-{group[0]}",
                    "kind": "parallel",
                    "label": "Unordered within a timestamp tie",
                    "video_ids": group,
                    "evidence_class": "UNKNOWN",
                    "parallel_group": group[0],
                }
            )
        for video_id in group:
            steps.append(
                {
                    "id": f"clip-{video_id}",
                    "kind": "clip",
                    "label": "Recording",
                    "video_ids": [video_id],
                    "evidence_class": "DIRECTLY_OBSERVED",
                }
            )
            if previous and previous not in group:
                steps.append(
                    {
                        "id": f"gap-{previous}-{video_id}",
                        "kind": "unobserved_interval",
                        "label": "Unobserved interval",
                        "video_ids": [previous, video_id],
                        "evidence_class": "UNKNOWN",
                        "note": "Nothing in the archive shows what happened between these recordings.",
                    }
                )
            previous = video_id
    for row in transitions:
        steps.append(
            {
                "id": row.id,
                "kind": "state_change",
                "label": "Inter-session state change",
                "observed_change": row.observed_change,
                "hypothesis": row.hypothesis,
                "evidence_class": row.evidence_class,
                "video_ids": [row.from_video_id, row.to_video_id],
            }
        )
    if order["unordered"]:
        steps.append(
            {
                "id": "unordered",
                "kind": "unresolved",
                "label": "Clips without an established position",
                "video_ids": order["unordered"],
                "evidence_class": "UNKNOWN",
            }
        )
    recording_order = []
    previous_group = None
    for group in order["ranked"]:
        if previous_group and len(previous_group) == 1 and len(group) == 1:
            recording_order.append(
                {
                    "before": previous_group[0],
                    "after": group[0],
                    "category": "recording_order",
                    "evidence_class": "DIRECTLY_OBSERVED" if order["method"] == "metadata" else "UNKNOWN",
                    "note": "This edge orders the recordings. It is not a claim that a procedure required that sequence.",
                }
            )
        previous_group = group
    procedural = [
        {
            "before": pair[0],
            "after": pair[1],
            "category": "procedural",
            "evidence_class": "INFERRED",
            "accepted": False,
            "note": "Context hypothesis. Not a verified dependency.",
        }
        for pair in hypothesis.get("pairs", [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    return {
        "project_id": project_id,
        "steps": steps,
        "recording_order": recording_order,
        "procedural_dependencies": procedural,
        "alternative_orderings": hypothesis.get("pairs", []),
        "continuous_sequence": False,
    }


def _diff(previous: dict, current: dict) -> str:
    if not previous:
        return "Initial project reading."
    added = set(current.get("clip_run_ids", {})) - set(previous.get("clip_run_ids", {}))
    parts = []
    if added:
        parts.append(f"Added {len(added)} clip(s). Previous clip analyses were kept.")
    if previous.get("canonical_ids") != current.get("canonical_ids"):
        parts.append("Canonical object proposals changed.")
    if previous.get("order") != current.get("order"):
        parts.append("Ordering reading changed.")
    return " ".join(parts) or "Reconciliation ran. Cached clip analyses were not rerun."


def manual_order(session: Session, project_id: str, video_ids: list[str]) -> None:
    videos = {video.id: video for video in session.scalars(select(VideoAsset).where(VideoAsset.project_id == project_id))}
    for index, video_id in enumerate(video_ids):
        video = videos.get(video_id)
        if video is None:
            continue
        video.order_index = float(index)
        video.order_locked = True
        video.order_method = "manual"
        video.order_ambiguous = False
    session.flush()


def move_clip(session: Session, video: VideoAsset, session_id: str | None) -> None:
    video.session_id = session_id


def new_session(session: Session, project_id: str, name: str) -> RecordingSession:
    count = len(list(session.scalars(select(RecordingSession).where(RecordingSession.project_id == project_id))))
    row = RecordingSession(id=new_id(), project_id=project_id, name=name, ordinal=count)
    session.add(row)
    session.flush()
    return row
