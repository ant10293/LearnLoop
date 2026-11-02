"""Run SAM image or video segmentation and persist its geometry.

A failed SAM call leaves the primary analysis in place. Repeated execution
replaces this run's SAM masks, cutouts, and track points instead of appending.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.domain import (
    DetectedObject,
    ObjectImage,
    ObjectObservation,
    SegmentationMask,
    SpatialTrackPoint,
)
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRun, Job
from app.perception.sam_fusion import (
    SamRequestPlan,
    TrackSample,
    associate_tracks,
    fuse_geometry,
    render_summary,
    write_sam_summary,
)
from app.perception.sam_masks import materialize_record
from app.perception.spatial_prompts import PROMPT_STRATEGIES, plan_for_strategy
from app.pipeline.infer import load_working, save_working
from app.pipeline.sampling_stage import timeline_for
from app.providers.catalog import lookup
from app.providers.sam import (
    DEFAULT_CHUNK_FRAMES,
    SamAdapter,
    SamRecord,
    frame_timestamp,
    normalized_box,
    parse_sam_output,
    plan_video_chunks,
    sam_capabilities,
    sam_video_accepted,
    scoped_track_id,
)
from app.providers.types import ProviderError, ProviderResponse, Usage, estimate_cost
from app.schemas.analysis import AnalysisDocument, BBox
from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
from app.services.credentials import load_secret
from app.services.storage import data_path, relative, resolve_stored
from app.video.frames import extract_frame, read_bgr
from app.video.probe import probe


async def run_image_segmentation(session: Session, run: AnalysisRun, stage) -> dict:
    secret, _source = load_secret(session, "meta")
    adapter = SamAdapter(secret, timeout_s=_timeout(run))
    _clear_sam_rows(session, run)
    objects = list(
        session.scalars(select(DetectedObject).where(DetectedObject.revision_id == run.active_revision_id))
    )
    stored = 0
    calls = 0
    for obj in objects:
        if calls >= _max_requests(run):
            break
        observation = _clearest_observation(session, obj.id)
        if observation is None or not observation.frame_id or not observation.bbox_json:
            continue
        frame = session.get(FrameAsset, observation.frame_id)
        if frame is None:
            continue
        image_path = data_path(*Path(frame.path).parts)
        box = BBox.model_validate(observation.bbox_json)
        image = read_bgr(image_path)
        height, width = image.shape[:2]
        prompt, kind = _image_prompt(obj.local_id, obj.label, box, width, height)
        call = open_provider_call(
            session,
            run,
            stage="segmentation",
            provider_id="meta",
            model_id="sam-3.1",
            modality="sampled_frames",
            parameters={"prompt_kind": kind, "object": obj.local_id, "frame_id": frame.id},
            frame_timestamps=[observation.timestamp],
        )
        session.commit()
        calls += 1
        try:
            result = await adapter.segment_image_prompt(image_path, prompt, prompt_kind=kind)
        except Exception as exc:
            fail_provider_call(call, exc)
            session.commit()
            raise
        _finish(session, run, call, result)
        session.commit()
        stored += _store_image_records(
            session,
            run,
            obj,
            frame,
            observation.timestamp,
            result,
            image,
            kind,
        )
    session.commit()
    return {
        "source": "sam-3.1",
        "mode": "image",
        "masks": stored,
        "requests": calls,
        "capabilities": sam_capabilities(),
        "note": "Image segmentation used a point at the center of a sighting, otherwise a noun.",
    }


async def run_video_segmentation(session: Session, run: AnalysisRun, video: VideoAsset, stage) -> dict:
    secret, _source = load_secret(session, "meta")
    adapter = SamAdapter(secret, timeout_s=_timeout(run))
    _clear_sam_rows(session, run)
    resuming = _resuming_after_restart(session, run)
    if not resuming:
        shutil.rmtree(_responses_dir(run.id), ignore_errors=True)
    document = load_working(run.id)
    source = resolve_stored(video.stored_path)
    prepared, transform = _prepare_media(source, run.id, int(video.rotation_deg or 0), reuse=resuming)
    metadata = probe(prepared)
    fps = float(metadata.avg_fps or metadata.fps or video.avg_fps or video.fps or 0)
    timeline = timeline_for(video, run.config_json)
    interval_times = timeline.inside() if timeline.basis == "presentation" else []
    whole = timeline.start <= 1e-3 and timeline.end >= float(video.duration_sec or 0) - 1e-3
    frame_count = len(interval_times) or int(metadata.frame_count or video.frame_count or 0)
    max_frames = int((run.config_json or {}).get("sam_max_frames") or 1800)
    chunk_frames = int((run.config_json or {}).get("sam_chunk_frames") or DEFAULT_CHUNK_FRAMES)
    if 0 < frame_count <= max_frames:
        chunk_frames = frame_count
    chunks = plan_video_chunks(
        frame_count,
        fps,
        chunk_frames=chunk_frames,
        duration_sec=timeline.span or float(metadata.duration_sec or video.duration_sec or 0),
        frame_times=interval_times,
    )
    if not interval_times and timeline.start > 0:
        for chunk in chunks:
            chunk.time_start += timeline.start
            chunk.time_end += timeline.start
    width = int(metadata.display_width or video.display_width or metadata.width)
    height = int(metadata.display_height or video.display_height or metadata.height)
    samples: list[TrackSample] = []
    calls = 0
    submitted_frames = 0
    skipped: list[str] = []
    skipped_spans: list[dict] = []
    failure: Exception | None = None
    max_requests = _max_requests(run)
    file_ids: dict[str, str] = {}
    resumed = 0
    sent_kinds: dict[str, int] = {}
    strategy = prompt_strategy(run)
    seeds = _pipeline_seeds(session, run, document, strategy)
    planned = []
    prompt_outcomes: list[dict] = []
    for chunk in chunks:
        plans, outcomes = plan_for_strategy(
            document, chunk, width, height, fps, strategy, seeds, multi_seed=bool((run.config_json or {}).get("sam_multi_seed"))
        )
        planned.append(plans)
        prompt_outcomes.extend(outcomes)
    # One unit per SAM request and one for saving, which ends when the stage completes.
    stage.progress_total = min(sum(len(plans) for plans in planned), max_requests) + 1
    stage.progress_current = 0
    session.commit()
    for chunk, plans in zip(chunks, planned):
        if not plans:
            skipped_spans.append(_skip(chunk, "no_concepts"))
            continue
        if calls and submitted_frames + chunk.frame_count > max_frames:
            skipped.extend(plan.session_id for plan in plans)
            skipped_spans.append(_skip(chunk, "frame_budget"))
            continue
        media: Path | None = None
        counted = False
        for plan in plans:
            if calls >= max_requests:
                skipped.append(plan.session_id)
                skipped_spans.append(_skip(chunk, "request_budget", plan.session_id))
                continue
            if not counted:
                submitted_frames += chunk.frame_count
                counted = True
            sent_kinds[plan.kind] = sent_kinds.get(plan.kind, 0) + 1
            saved = _load_response(run.id, plan, chunk) if resuming else None
            if saved is not None:
                calls += 1
                resumed += 1
                samples.extend(
                    _samples_from_result(
                        parse_sam_output(saved["text"]), saved["request_id"], plan, chunk, fps
                    )
                )
                stage.progress_current = calls
                session.commit()
                continue
            if media is None:
                media = _chunk_media(prepared, chunk, chunks, run.id, fps, whole=whole, reuse=resuming)
            call = open_provider_call(
                session,
                run,
                stage="segmentation",
                provider_id="meta",
                model_id="sam-3.1",
                modality="native_video",
                parameters={
                    "prompt_kind": plan.kind,
                    "session_id": plan.session_id,
                    "propagation_direction": plan.propagation or "",
                    "chunk_start": chunk.time_start,
                    "chunk_end": chunk.time_end,
                    "frames": chunk.frame_count,
                    "bindings": {str(key): value for key, value in plan.bindings.items()},
                    "targets": plan.targets,
                    "transform": transform,
                },
                frame_timestamps=[chunk.time_start, chunk.time_end],
            )
            session.commit()
            calls += 1
            try:
                file_id = file_ids.get(str(media))
                result = await adapter.segment_video(
                    media,
                    plan.prompt,
                    prompt_kind=plan.kind,
                    file_id=file_id or "",
                    propagation_direction=plan.propagation,
                )
            except Exception as exc:
                fail_provider_call(call, exc)
                session.commit()
                failure = exc
                break
            file_ids[str(media)] = result.file_id or file_ids.get(str(media), "")
            _finish(session, run, call, result)
            _save_response(run.id, plan, chunk, result)
            stage.progress_current = calls
            session.commit()
            samples.extend(
                _samples_from_result(result.records, result.request_id, plan, chunk, fps)
            )
        if failure is not None:
            break
    if not samples and failure is not None:
        raise failure
    boundaries = _scene_boundaries(session, run.id)
    from app.pipeline.perception_stage import load_spatial_document

    associations = associate_tracks(
        document,
        samples,
        scene_boundaries=boundaries,
        spatial=load_spatial_document(run.id),
        measured_bindings=strategy != "concept",
    )
    fused = fuse_geometry(
        document,
        samples,
        associations,
        scene_boundaries=boundaries,
        coverage_note=_coverage_note(chunks, skipped, submitted_frames, transform),
    )
    if fused.coverage is not None:
        fused.coverage.segmentation = {
            "interval": [round(timeline.start, 3), round(timeline.end, 3)],
            "time_basis": timeline.basis,
            "chunks": len(chunks),
            "frames_submitted": submitted_frames,
            "skipped_chunks": skipped_spans,
            "orientation": transform.get("orientation", "displayed"),
            "transformed": bool(transform.get("transformed")),
        }
    summary = render_summary(fused, samples, associations, _coverage_note(chunks, skipped, submitted_frames, transform))
    write_sam_summary(run.id, summary)
    _persist_tracks_and_masks(session, run, video, samples, associations, fused)
    save_working(run.id, fused)
    session.commit()
    if failure is not None:
        raise failure
    return {
        "document": fused,
        "source": "sam-3.1",
        "mode": "video",
        "masks": len(samples),
        "requests": calls,
        "resumed_requests": resumed,
        "frames_submitted": submitted_frames,
        "chunks": len(chunks),
        "interval": [round(timeline.start, 3), round(timeline.end, 3)],
        "time_basis": timeline.basis,
        "skipped_sessions": skipped,
        "skipped_chunks": skipped_spans,
        "coverage_incomplete": bool(skipped),
        "prompt_strategy": strategy,
        "multi_seed": bool((run.config_json or {}).get("sam_multi_seed")),
        "prompt_kinds_sent": sent_kinds,
        "prompt_outcomes": prompt_outcomes,
        "transform": transform,
        "capabilities": sam_capabilities(),
        "note": (
            f"Video tracking used the {strategy} prompt strategy on the video file. "
            "Object ids are scoped to each request. Chunk boundaries do not continue a raw id. "
            "Live API validation is separate from this recorded call."
        ),
    }


def prompt_strategy(run: AnalysisRun) -> str:
    """A run stored before this setting existed stays on noun prompts."""
    value = str((run.config_json or {}).get("sam_prompt_strategy") or "concept")
    return value if value in PROMPT_STRATEGIES else "concept"


def _pipeline_seeds(session: Session, run: AnalysisRun, document: AnalysisDocument, strategy: str) -> list:
    """Seeds known when segmentation begins. Earlier SAM rows are left out, so SAM cannot vouch for itself."""
    if strategy == "concept":
        return []
    from app.perception.seeds import collect_seeds, rank_for_interval
    from app.pipeline.perception_stage import load_spatial_document

    earlier = document.model_copy(
        update={
            "observations": [row for row in document.observations if row.source_model != "sam-3.1"],
            "raw_observations": [row for row in document.raw_observations if row.source_model != "sam-3.1"],
        }
    )
    seeds = collect_seeds(
        earlier, load_spatial_document(run.id), boundaries=_scene_boundaries(session, run.id)
    )
    return rank_for_interval(seeds, 0.0, float("inf"))


def _image_prompt(local_id: str, label: str, box: BBox, width: int, height: int) -> tuple[str, str]:
    from app.perception.sam_fusion import _mark
    from app.providers.sam import render_spatial_prompt

    mark = _mark(0, 1, box, width, height)
    return render_spatial_prompt([mark]), "spatial"


def _samples_from_result(records: list[SamRecord], request_id: str, plan: SamRequestPlan, chunk, fps: float):
    samples = []
    for record in records:
        timestamp = frame_timestamp(record.frame_index, fps, chunk)
        box = normalized_box(record)
        if timestamp is None or box is None:
            continue
        raw = record.track_id
        samples.append(
            TrackSample(
                session_id=plan.session_id,
                raw_track_id=raw,
                timestamp=timestamp,
                bbox=BBox.model_validate(box),
                confidence=record.confidence,
                frame_index=record.frame_index,
                request_id=request_id,
                bound_object_id=plan.bindings.get(int(raw), "") if raw.isdigit() else "",
                prompt_kind=plan.kind,
                candidates=tuple(plan.targets),
                mask_data=record.mask_data,
                mask_payload=record.mask_payload,
                mask_encoding=record.mask_encoding,
                mask_height=record.mask_height,
                mask_width=record.mask_width,
                source_width=record.width,
                source_height=record.height,
            )
        )
    # Drop object ids beyond the documented 16 per response by first-seen order.
    allowed: list[str] = []
    kept = []
    for sample in samples:
        if sample.raw_track_id not in allowed:
            if len(allowed) >= 16:
                continue
            allowed.append(sample.raw_track_id)
        kept.append(sample)
    return kept


def _store_image_records(session, run, obj, frame, timestamp, result, image, kind: str) -> int:
    from app.perception.assignment import iou

    target = None
    observation = _clearest_observation(session, obj.id)
    if observation and observation.bbox_json:
        target = BBox.model_validate(observation.bbox_json)
    stored = 0
    directory = data_path("runs", run.id, "sam", "image", obj.local_id)
    for index, record in enumerate(result.records):
        box = normalized_box(record)
        if box is None:
            continue
        overlap = 0.0
        if target is not None:
            overlap = iou((target.x, target.y, target.w, target.h), (box["x"], box["y"], box["w"], box["h"]))
        status = "associated" if overlap >= 0.35 else "ambiguous" if overlap >= 0.1 else "unassigned"
        files = materialize_record(
            record,
            directory,
            f"{index}",
            image_bgr=image if status == "associated" else None,
        )
        session.add(
            SegmentationMask(
                object_id=obj.id if status == "associated" else None,
                revision_id=run.active_revision_id,
                run_id=run.id,
                frame_id=frame.id,
                timestamp=timestamp,
                track_ref=scoped_track_id(f"image-{obj.local_id}", record.track_id),
                encoding="png" if files.decoded else record.mask_encoding,
                path=relative(files.png_path) if files.png_path else "",
                rle_json=_mask_meta(record, files, status, result.request_id, "image"),
                bbox_json=box,
                confidence=record.confidence,
                source="sam-3.1",
                width=record.width,
                height=record.height,
                identity_uncertain=status != "associated",
            )
        )
        if files.cutout_path is not None:
            written = cv2.imread(str(files.cutout_path), cv2.IMREAD_UNCHANGED)
            session.add(
                ObjectImage(
                    object_id=obj.id,
                    revision_id=run.active_revision_id,
                    kind="cutout",
                    path=relative(files.cutout_path),
                    frame_id=frame.id,
                    timestamp=timestamp,
                    width=int(written.shape[1]) if written is not None else 0,
                    height=int(written.shape[0]) if written is not None else 0,
                    score=overlap,
                )
            )
        stored += 1
    return stored


def _persist_tracks_and_masks(session, run, video, samples, associations, document: AnalysisDocument) -> None:
    lookup = {(item.session_id, item.raw_track_id): item for item in associations}
    best: dict[str, TrackSample] = {}
    for sample in samples:
        association = lookup.get((sample.session_id, sample.raw_track_id))
        if association is None or association.status != "associated" or not association.object_local_id:
            continue
        current = best.get(association.object_local_id)
        if current is None or (sample.confidence or 0) > (current.confidence or 0):
            best[association.object_local_id] = sample
    directory = data_path("runs", run.id, "sam", "video")
    source = resolve_stored(video.stored_path)
    for index, sample in enumerate(samples):
        association = lookup.get((sample.session_id, sample.raw_track_id))
        status = association.status if association else "unassigned"
        local_id = association.object_local_id if association and status == "associated" else ""
        scoped = scoped_track_id(sample.session_id, sample.raw_track_id)
        session.add(
            SpatialTrackPoint(
                run_id=run.id,
                video_id=video.id,
                track_ref=scoped[:80],
                timestamp=sample.timestamp,
                bbox_json=sample.bbox.model_dump(),
                continuity="uncertain" if status != "associated" else "continuous",
                track_status="ambiguous" if status in {"ambiguous", "conflict", "unassigned"} else "tracked",
                association_score=association.score if association else None,
                detector_id="sam-3.1",
                identity_note=(
                    f"object={local_id};association={status};session={sample.session_id};"
                    f"raw={sample.raw_track_id};request={sample.request_id}"
                ),
            )
        )
        record = _record_for_sample(sample)
        image = None
        if local_id and best.get(local_id) is sample:
            frame_path = directory / f"frame_{index}.jpg"
            try:
                extract_frame(source, sample.timestamp, frame_path)
                image = read_bgr(frame_path)
            except Exception:
                image = None
        files = (
            materialize_record(record, directory, f"{index:05d}", image_bgr=image, decode=image is not None)
            if record
            else None
        )
        mask_path = ""
        meta = {
            "session_id": sample.session_id,
            "request_id": sample.request_id,
            "association_status": status,
            "object_local_id": local_id,
            "decode": "skipped",
        }
        if files is not None:
            mask_path = relative(files.png_path) if files.png_path else ""
            meta = _mask_meta(record, files, status, sample.request_id, sample.session_id)
            if files.cutout_path is not None and local_id:
                _add_cutout(session, run, local_id, files.cutout_path, sample)
        session.add(
            SegmentationMask(
                object_id=_object_row_id(session, run, local_id) if local_id else None,
                revision_id=run.active_revision_id,
                run_id=run.id,
                timestamp=sample.timestamp,
                track_ref=scoped[:80],
                encoding="png" if files and files.decoded else "raw",
                path=mask_path,
                rle_json=meta,
                bbox_json=sample.bbox.model_dump(),
                confidence=sample.confidence,
                source="sam-3.1",
                width=record.width if record else 0,
                height=record.height if record else 0,
                identity_uncertain=status != "associated",
            )
        )


def _record_for_sample(sample: TrackSample) -> SamRecord | None:
    if not sample.mask_data and not sample.mask_payload:
        return None
    width = sample.source_width or 1
    height = sample.source_height or 1
    return SamRecord(
        frame_index=sample.frame_index,
        track_id=sample.raw_track_id,
        x1=round(sample.bbox.x * width),
        y1=round(sample.bbox.y * height),
        x2=round((sample.bbox.x + sample.bbox.w) * width),
        y2=round((sample.bbox.y + sample.bbox.h) * height),
        width=width,
        height=height,
        confidence=sample.confidence,
        mask_payload=sample.mask_payload,
        mask_encoding=sample.mask_encoding or "unknown",
        mask_height=sample.mask_height,
        mask_width=sample.mask_width,
        mask_data=sample.mask_data,
    )


def _add_cutout(session, run, local_id: str, path: Path, sample: TrackSample) -> None:
    object_id = _object_row_id(session, run, local_id)
    if object_id is None:
        return
    image = read_bgr(path)
    session.add(
        ObjectImage(
            object_id=object_id,
            revision_id=run.active_revision_id,
            kind="cutout",
            path=relative(path),
            timestamp=sample.timestamp,
            width=image.shape[1],
            height=image.shape[0],
            score=sample.confidence or 0,
        )
    )


def _object_row_id(session, run, local_id: str) -> str | None:
    row = session.scalar(
        select(DetectedObject).where(
            DetectedObject.revision_id == run.active_revision_id,
            DetectedObject.local_id == local_id,
        )
    )
    return row.id if row else None


def _mask_meta(record: SamRecord, files, status: str, request_id: str, session_id: str) -> dict:
    return {
        "payload_path": relative(files.raw_path) if files.raw_path else "",
        "encoding": record.mask_encoding,
        "mask_height": record.mask_height,
        "mask_width": record.mask_width,
        "source_width": record.width,
        "source_height": record.height,
        "session_id": session_id,
        "request_id": request_id,
        "association_status": status,
        "decode": "ok" if files.decoded else "failed" if files.error else "deferred",
        "decode_error": files.error,
        "frame_index": record.frame_index,
        "transform": "normalized by the width and height SAM reported on the box",
    }


def _clear_sam_rows(session: Session, run: AnalysisRun) -> None:
    session.execute(
        delete(SegmentationMask).where(
            SegmentationMask.run_id == run.id,
            SegmentationMask.source == "sam-3.1",
        )
    )
    session.execute(
        delete(SpatialTrackPoint).where(
            SpatialTrackPoint.run_id == run.id,
            SpatialTrackPoint.detector_id == "sam-3.1",
        )
    )
    if run.active_revision_id:
        session.execute(
            delete(ObjectImage).where(
                ObjectImage.revision_id == run.active_revision_id,
                ObjectImage.kind == "cutout",
                ObjectImage.path.like(f"%/runs/{run.id}/sam/%"),
            )
        )
    session.flush()


def _clearest_observation(session, object_id: str):
    return session.scalar(
        select(ObjectObservation)
        .where(
            ObjectObservation.object_id == object_id,
            ObjectObservation.bbox_json.is_not(None),
            ObjectObservation.visibility == "visible",
        )
        .order_by(ObjectObservation.sharpness.desc())
    )


def _finish(session, run, call, result) -> None:
    cost, known = estimate_cost(lookup("meta", "sam-3.1"), result.input_tokens, result.output_tokens)
    finish_provider_call(
        session,
        run,
        call,
        ProviderResponse(
            text=result.text,
            raw={
                "id": result.request_id,
                "model": "sam-3.1",
                "file_id": result.file_id,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            },
            usage=Usage(result.input_tokens, result.output_tokens, cost, known),
            request_id=result.request_id,
            latency_ms=result.latency_ms,
            model_id="sam-3.1",
        ),
    )


def _skip(chunk, reason: str, session_id: str = "") -> dict:
    return {
        "start": round(chunk.time_start, 3),
        "end": round(chunk.time_end, 3),
        "reason": reason,
        "session_id": session_id or chunk.session_id,
    }


def _resuming_after_restart(session: Session, run: AnalysisRun) -> bool:
    """A retry of the same job continues. A new job, such as a rerun of this stage, starts fresh."""
    job = session.scalar(
        select(Job).where(Job.run_id == run.id, Job.status == "running").order_by(Job.created_at.desc())
    )
    return job is not None and job.attempt > 1


def _responses_dir(run_id: str) -> Path:
    return data_path("runs", run_id, "sam", "responses")


def _response_path(run_id: str, plan: SamRequestPlan, chunk) -> Path:
    identity = json.dumps(
        [
            plan.session_id,
            plan.kind,
            plan.prompt,
            plan.propagation or "",
            chunk.time_start,
            chunk.time_end,
            chunk.frame_count,
        ]
    )
    return _responses_dir(run_id) / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"


def _save_response(run_id: str, plan: SamRequestPlan, chunk, result) -> None:
    """Keep the billed response, written whole so a kill cannot leave a partial file."""
    dest = _response_path(run_id, plan, chunk)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps({"text": result.text, "request_id": result.request_id}))
    tmp.replace(dest)


def _load_response(run_id: str, plan: SamRequestPlan, chunk) -> dict | None:
    try:
        return json.loads(_response_path(run_id, plan, chunk).read_text())
    except (OSError, ValueError):
        return None


def _encode(command: list[str], dest: Path) -> subprocess.CompletedProcess:
    """Run ffmpeg into a temporary name, so `dest` only ever holds a finished encode."""
    tmp = dest.with_name(f".{dest.stem}.tmp{dest.suffix}")
    completed = subprocess.run([*command, str(tmp)], capture_output=True, text=True, check=False)
    if completed.returncode == 0 and tmp.exists():
        tmp.replace(dest)
    else:
        tmp.unlink(missing_ok=True)
    return completed


def _prepare_media(source: Path, run_id: str, rotation: int = 0, *, reuse: bool = False) -> tuple[Path, dict]:
    """SAM always receives displayed-orientation media, so its boxes are displayed-frame boxes."""
    metadata = probe(source)
    if sam_video_accepted(metadata.codec, metadata.container) and rotation % 360 == 0:
        return source, {
            "transformed": False,
            "codec": metadata.codec,
            "container": metadata.container,
            "width": metadata.display_width,
            "height": metadata.display_height,
            "orientation": "displayed",
        }
    dest = data_path("runs", run_id, "sam", "prepared.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
    ]
    if not (reuse and dest.exists()):
        dest.unlink(missing_ok=True)
        completed = _encode(command, dest)
        if completed.returncode != 0 or not dest.exists():
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise ProviderError(
                f"SAM 3.1 accepts MP4 or MOV with H.264, H.265, or MPEG-4. "
                f"This file is {metadata.codec} in {metadata.container}, and converting it failed. {detail}"
            )
    converted = probe(dest)
    return dest, {
        "transformed": True,
        "from_codec": metadata.codec,
        "from_container": metadata.container,
        "to_codec": converted.codec,
        "width": converted.display_width,
        "height": converted.display_height,
        "rotation_applied": rotation % 360,
        "orientation": "displayed",
        "note": (
            "The file was re-encoded to H.264 in its displayed orientation so SAM could read it. "
            "Boxes are normalized to the displayed frame."
        ),
    }


def _chunk_media(
    source: Path, chunk, chunks, run_id: str, fps: float, *, whole: bool = True, reuse: bool = False
) -> Path:
    """The file SAM receives for one chunk. Frame `k` in it is source frame `chunk.frame_times[k]`.

    A cut is re-encoded with accurate seeking and without dropping or
    duplicating frames. A stream copy would start at a keyframe and shift
    every frame index.
    """
    if len(chunks) == 1 and whole:
        return source
    dest = data_path("runs", run_id, "sam", f"chunk_{chunk.index}.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if reuse and dest.exists():
        return dest
    half_frame = 0.5 / max(fps, 1)
    duration = max(chunk.time_end - chunk.time_start - half_frame, 1 / max(fps, 1))
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{chunk.time_start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
    ]
    dest.unlink(missing_ok=True)
    completed = _encode(command, dest)
    if completed.returncode != 0 or not dest.exists():
        raise ProviderError((completed.stderr or "Could not cut a SAM video chunk.")[-500:])
    return dest


def _scene_boundaries(session, run_id: str) -> list[float]:
    from app.models.domain import SceneChunk

    rows = session.scalars(select(SceneChunk).where(SceneChunk.run_id == run_id)).all()
    return [row.time_start for row in rows if row.scene_change and row.time_start > 0]


def _coverage_note(chunks, skipped, submitted_frames, transform: dict) -> str:
    parts = [
        f"SAM chunks {len(chunks)}. Frames submitted {submitted_frames}.",
        "A chunk boundary starts a new SAM session. Raw object ids are not continued across it.",
    ]
    if skipped:
        parts.append(
            f"Sessions not sent because the request budget was reached: {', '.join(skipped)}. "
            "Those intervals are unobserved."
        )
    if transform.get("transformed"):
        parts.append(str(transform.get("note") or "The video was re-encoded before SAM."))
    return " ".join(parts)


def _timeout(run: AnalysisRun) -> float:
    return float((run.config_json or {}).get("timeout_s") or 180)


def _max_requests(run: AnalysisRun) -> int:
    return max(int((run.config_json or {}).get("sam_max_requests") or 8), 1)
