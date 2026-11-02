"""Durable stage runner. Completed stages are skipped. Failures retry from the failed stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Artifact, SegmentationMask
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun, Job, PipelineStageRun
from app.models.util import utcnow
from app.perception.assembly import apply_assembly
from app.perception.continuity import reconcile_identity
from app.perception.ground import apply_perception
from app.perception.resolve import fence_repair
from app.perception.roster import drop_contained_objects, omit_workspace_and_hands
from app.perception.types import TrackingCancelled
from app.pipeline.actions import collapse_repeated_actions
from app.pipeline.contracts import StructuredOutputError
from app.pipeline.infer import (
    SEMANTIC_LISTS,
    BudgetExceeded,
    StageSkipped,
    _require_video_length,
    _video_file,
    load_working,
    model_for_call,
    produce_document,
    save_semantic_raw,
    save_working,
    semantic_calls,
    semantic_mode,
)
from app.pipeline.lineage import record_lineage
from app.pipeline.locate_pass import locate_missing
from app.pipeline.merge import merge_interpretation, retain_prior_records
from app.pipeline.perception_stage import load_spatial_document, run_spatial_stage
from app.pipeline.prompts import compose
from app.pipeline.sampling_stage import sample_stage
from app.pipeline.validate import validate_document
from app.providers.types import UnsupportedMode
from app.schemas.analysis import AnalysisDocument, Coverage, ReinspectionProposal
from app.schemas.enums import PIPELINE_STAGES
from app.services.experiments import sends_video_file
from app.services.ingest import playback_path
from app.services.materialize import grabcut_cutout, materialize
from app.services.storage import data_path, relative, resolve_stored
from app.video.frames import contact_sheet, extract_frame, read_bgr
from app.video.sampling import distinct_moments

log = logging.getLogger("learnloop.runner")


async def execute_job(session: Session, job: Job) -> None:
    run = session.get(AnalysisRun, job.run_id)
    if run is None or run.status == "canceled":
        job.status = "canceled"
        job.finished_at = utcnow()
        session.commit()
        return
    run_id = run.id
    video = session.get(VideoAsset, run.video_id)
    if video is None:
        _fail_job(session, job, run, "video missing")
        return
    run.status = "running"
    run.started_at = run.started_at or utcnow()
    run.error = ""
    session.commit()
    started = job.from_stage == ""
    try:
        for name in PIPELINE_STAGES:
            session.refresh(run)
            if run.status == "canceled":
                _cancel_remaining(session, run, name)
                job.status = "canceled"
                job.finished_at = utcnow()
                session.commit()
                return
            if not started:
                if name != job.from_stage:
                    continue
                started = True
            stage = _stage(session, run.id, name)
            if stage.status == "completed":
                continue
            stage.status = "running"
            stage.attempt += 1
            stage.started_at = utcnow()
            stage.error = ""
            stage.progress_current = 0
            stage.progress_total = 1
            job.heartbeat_at = utcnow()
            session.commit()
            try:
                await _dispatch(session, run, video, stage)
            except TrackingCancelled:
                run.status = "canceled"
                _cancel_remaining(session, run, name)
                job.status = "canceled"
                job.finished_at = utcnow()
                session.commit()
                return
            except StageSkipped as skipped:
                stage.status = "skipped"
                stage.checkpoint_json = {"reason": skipped.reason, **(stage.checkpoint_json or {})}
                stage.finished_at = utcnow()
                session.commit()
                continue
            except Exception as exc:
                stage.status = "failed"
                stage.error = _error_text(exc)
                stage.finished_at = utcnow()
                if name in {"segmentation", "evidence_fusion", "reinspection", "transcription", "audio_context"}:
                    session.commit()
                    continue
                raise
            stage.status = "completed"
            stage.progress_current = max(stage.progress_total, 1)
            stage.finished_at = utcnow()
            session.commit()
        run.status = "completed"
        run.finished_at = utcnow()
        job.status = "completed"
        job.finished_at = utcnow()
        session.commit()
        try:
            from app.services.verification_store import settle_waiting

            settle_waiting(session, run.id)
        except Exception:
            session.rollback()
            log.exception("verification waiting on run %s was not settled", run.id)
    except Exception as exc:
        session.rollback()
        if session.get(AnalysisRun, run_id) is None:
            # The run was deleted while it ran, and its job row went with it.
            log.info("run %s was deleted during execution; stopped its job", run_id)
            return
        run.status = "failed"
        run.error = _error_text(exc)
        run.finished_at = utcnow()
        if job.attempt < job.max_attempts and not isinstance(
            exc, (UnsupportedMode, BudgetExceeded, StructuredOutputError)
        ):
            job.status = "pending"
            job.error = _error_text(exc)
            job.worker_id = ""
        else:
            job.status = "failed"
            job.error = _error_text(exc)
            job.finished_at = utcnow()
        session.commit()


def _error_text(exc: BaseException, limit: int = 2000) -> str:
    """Keep the end of a long failure. ffmpeg puts the cause after its banner."""
    text = str(exc).strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _fail_job(session, job, run, message: str) -> None:
    job.status = "failed"
    job.error = message
    job.finished_at = utcnow()
    run.status = "failed"
    run.error = message
    session.commit()


def _cancel_remaining(session, run, current: str) -> None:
    seen = False
    for name in PIPELINE_STAGES:
        if name == current:
            seen = True
        if not seen:
            continue
        stage = _stage(session, run.id, name)
        if stage.status in {"pending", "running"}:
            stage.status = "canceled"
            stage.finished_at = utcnow()


def _stage(session, run_id: str, name: str) -> PipelineStageRun:
    stage = session.scalar(
        select(PipelineStageRun).where(PipelineStageRun.run_id == run_id, PipelineStageRun.stage == name)
    )
    if stage is None:
        stage = PipelineStageRun(run_id=run_id, stage=name, status="pending", checkpoint_json={})
        session.add(stage)
        session.flush()
    return stage


async def _dispatch(session, run: AnalysisRun, video: VideoAsset, stage: PipelineStageRun) -> None:
    name = stage.stage
    if name == "metadata":
        stage.checkpoint_json = {
            "duration_sec": video.duration_sec,
            "fps": video.fps,
            "avg_fps": video.avg_fps,
            "variable_frame_rate": video.variable_frame_rate,
            "width": video.display_width,
            "height": video.display_height,
            "checksum": video.checksum_sha256,
            "rotation_deg": video.rotation_deg,
        }
        stage.progress_total = 1
        stage.progress_current = 1
        return
    if name == "preprocess":
        path = playback_path(video)
        stage.checkpoint_json = {
            "playback": relative(path),
            "proxy": bool(video.proxy_path),
            **_native_subclip(run, video),
        }
        stage.progress_total = 1
        stage.progress_current = 1
        return
    if name == "sample_frames":
        _sample(session, run, video, stage)
        return
    if name == "spatial_tracking":
        _spatial(session, run, video, stage)
        return
    if name in {"scene_analysis", "action_analysis", "state_tracking", "workflow", "ambiguity"}:
        await _model_stage(session, run, video, stage)
        return
    if name == "object_extraction":
        _publish_current(session, run)
        stage.checkpoint_json = {"note": "Crops were taken from stored frames during materialization."}
        return
    if name == "segmentation":
        await _segment(session, run, video, stage)
        return
    if name == "evidence_fusion":
        await _evidence_fusion(session, run, video, stage)
        return
    if name == "validation":
        _validate(session, run, video, stage)
        return
    if name == "reinspection":
        await _reinspect(session, run, video, stage)
        return
    if name == "transcription":
        await _transcription(session, run, video, stage)
        return
    if name == "audio_context":
        await _audio_context(session, run, video, stage)
        return
    if name == "artifacts":
        _artifacts(session, run, stage)
        return
    stage.status = "skipped"
    stage.checkpoint_json = {"reason": "unknown stage"}


def _native_subclip(run, video) -> dict:
    """Cut the requested interval for a native-video model. A failed cut fails the stage."""
    from app.pipeline.clip_time import save_transform, subclip_path
    from app.pipeline.sampling_stage import timeline_for
    from app.video.decode import cut_clip

    if not sends_video_file(run):
        return {}
    timeline = timeline_for(video, run.config_json)
    duration = float(video.duration_sec or 0)
    if timeline.start <= 1e-3 and timeline.end >= duration - 1e-3:
        return {}
    dest = subclip_path(run.id)
    transform = cut_clip(resolve_stored(video.stored_path), timeline.start, timeline.end, dest)
    save_transform(run.id, transform)
    return {"subclip": relative(dest), "media_transform": transform}


async def _transcription(session, run, video, stage) -> None:
    config = run.config_json or {}
    if not config.get("transcribe"):
        raise StageSkipped("Local transcription is off. Whisper was not loaded.")
    from app.audio.service import transcribe_clip

    source = resolve_stored(video.stored_path)

    def stop() -> bool:
        session.refresh(run)
        return run.status == "canceled"

    stage.progress_total = 3
    stage.progress_current = 1
    session.commit()
    stage.checkpoint_json = transcribe_clip(session, run, video, source, stop)
    stage.progress_current = 3


async def _audio_context(session, run, video, stage) -> None:
    config = run.config_json or {}
    if not config.get("use_transcript"):
        raise StageSkipped("Transcript text was not sent for workflow analysis.")
    from app.audio.service import (
        LOCAL_PROVIDERS,
        align_clip,
        apply_provider_classes,
        context_request,
        visual_events,
    )
    from app.models.audio import TranscriptSegment
    from app.providers.registry import get_provider, needs_secret, run_transport
    from app.providers.types import FrameInput, GenerationRequest
    from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
    from app.services.credentials import load_secret

    result = align_clip(session, run, video)
    if run.provider_id in LOCAL_PROVIDERS:
        result["note"] = "The selected provider is local. Transcript text stayed on this machine."
        stage.checkpoint_json = result
        return
    segments = list(session.scalars(select(TranscriptSegment).where(TranscriptSegment.run_id == run.id)))
    if not segments:
        result["note"] = "There was no speech to send."
        stage.checkpoint_json = result
        return
    request_body = context_request(run, segments, visual_events(session, run, video.id))
    if request_body is None:
        result["note"] = "No provider key is saved. Local classification was kept and the transcript was not sent."
        stage.checkpoint_json = result
        return
    transport = run_transport(run.provider_id, config)
    secret = ""
    if needs_secret(run.provider_id, transport):
        secret, _source = load_secret(session, run.provider_id)
        if not secret:
            result["note"] = "No provider key is saved. Local classification was kept and the transcript was not sent."
            stage.checkpoint_json = result
            return
    frames = session.scalars(
        select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp)
    ).all()[:4]
    frame_inputs = [
        FrameInput(timestamp=frame.timestamp, path=data_path(*Path(frame.path).parts))
        for frame in frames
        if data_path(*Path(frame.path).parts).exists()
    ]
    send_video = sends_video_file(run)
    model_id, requested = model_for_call(session, run)
    video_path = None
    if send_video:
        video_path = _video_file(video)
        _require_video_length(run, video)
        frame_inputs = []
    generation = GenerationRequest(
        model_id=model_id,
        system_prompt=request_body["system_prompt"],
        user_prompt=request_body["user_prompt"],
        frames=frame_inputs,
        video_path=video_path,
        modality="native_video" if send_video else "sampled_frames",
        temperature=config.get("temperature"),
        timeout_s=float(config.get("timeout_s") or 180),
        image_detail=config.get("image_detail") or "low",
        metadata={"transport": transport} if run.provider_id in {"openai", "cursor"} else {},
    )
    call = open_provider_call(
        session,
        run,
        stage="audio_context",
        provider_id=run.provider_id,
        model_id=model_id,
        modality="native_video" if send_video else "sampled_frames",
        prompt_version_id=run.prompt_version_id,
        parameters={
            "temperature": generation.temperature,
            "image_detail": generation.image_detail,
            "purpose": "transcript_context",
            **({"requested_model_id": requested} if requested != model_id else {}),
        },
        frame_timestamps=[frame.timestamp for frame in frame_inputs],
    )
    session.commit()
    try:
        response = await get_provider(run.provider_id, session).generate(secret, generation)
    except Exception as exc:
        fail_provider_call(call, exc)
        session.commit()
        raise
    finish_provider_call(session, run, call, response)
    session.commit()
    parsed = _json_object(response.text)
    if parsed:
        result["provider_updates"] = apply_provider_classes(session, run.id, parsed)
    result["transcript_text_sent_to_provider"] = True
    result["audio_bytes_sent"] = False
    result["video_file_sent"] = send_video
    result["note"] = (
        "Transcript text and the video file were sent. The audio track was not."
        if send_video
        else "Transcript text, not the audio track, was sent to the selected model."
    )
    stage.checkpoint_json = result


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sample(session, run, video, stage) -> None:
    sample_stage(session, run, video, stage)


def _spatial(session, run, video, stage) -> None:
    run_spatial_stage(session, run, video, stage)


def _record_semantic(session, run, coverage: Coverage) -> Coverage:
    coverage.semantic_mode = semantic_mode(run)
    coverage.semantic_calls = semantic_calls(session, run.id)
    return coverage


async def _model_stage(session, run, video, stage) -> None:
    config = run.config_json or {}
    grounded = semantic_mode(run) == "grounded"
    if grounded and stage.stage in {"action_analysis", "state_tracking", "ambiguity"}:
        raise StageSkipped(
            "Grounded mode interprets actions, states, and ambiguities once, in workflow, after evidence fusion. "
            "This stage made no model call."
        )
    if grounded and stage.stage == "workflow":
        await _interpret(session, run, video, stage)
        return
    single = config.get("analysis_pass", "single") == "single"
    if single and stage.stage != "scene_analysis":
        document = load_working(run.id)
        stage.checkpoint_json = {
            "source": "single_pass",
            "note": "This stage did not issue another model call. It records the slice of the single response.",
        }
        _remember(session, run, document, stage.stage)
        return
    if stage.stage != "scene_analysis" and run.provider_id in {"deterministic", "fixture"}:
        document = load_working(run.id)
        stage.checkpoint_json = {"source": run.provider_id}
        _remember(session, run, document, stage.stage)
        return
    frames = distinct_moments(
        list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp)))
    )
    coverage = Coverage.model_validate(run.coverage_json or {})
    spatial = load_spatial_document(run.id)
    caps_window = 16
    from app.providers.catalog import lookup

    caps_window = lookup(run.provider_id, run.model_id).max_images
    config = dict(config)
    config["max_images_per_call"] = caps_window
    run.config_json = config
    discovery = grounded and stage.stage == "scene_analysis"
    document = await produce_document(
        session, run, video, frames, stage.stage, coverage, spatial, purpose="discovery" if discovery else "full"
    )
    if discovery:
        document = document.model_copy(update={name: [] for name in SEMANTIC_LISTS})
    if stage.stage == "scene_analysis":
        save_semantic_raw(run.id, document)
    # Sampling already counted the frames. The text-only merge must not replace that
    # record with a note about how it combined the windows.
    document = document.model_copy(update={"coverage": coverage})
    document, locate_info = await _ground(session, run, stage, document, frames)
    if document.perception is None and spatial is not None and spatial.perception is not None:
        document = apply_perception(document, spatial)
    if run.provider_id == "fixture":
        run.fixture = True
        run.fixture_label = document.fixture_label
    if document.camera and run.research_mode == "auto":
        run.viewpoint = document.camera.viewpoint
        run.task_constraint = document.camera.task_constraint
    _record_semantic(session, run, coverage)
    document = document.model_copy(update={"coverage": coverage})
    run.coverage_json = coverage.model_dump(mode="json")
    save_working(run.id, document)
    _remember(session, run, document, stage.stage)
    stage.checkpoint_json = {
        "provider": run.provider_id,
        "model": run.model_id,
        "objects": len(document.objects),
        "actions": len(document.actions),
        "modality": run.modality,
        **({"locate": locate_info} if locate_info else {}),
    }


async def _ground(session, run, stage, document: AnalysisDocument, frames: list[FrameAsset]):
    """Keep ids on the part they belong to, find that part again, then read insertions."""
    if stage.stage != "scene_analysis":
        document = retain_prior_records(document, load_working(run.id))
    locate_info = None
    if document.observations:
        document = omit_workspace_and_hands(document)
        document = fence_repair(document, reconcile_identity)
        document = drop_contained_objects(document)
        if stage.stage == "scene_analysis":
            document, locate_info = await locate_missing(session, run, document, frames)
            document = fence_repair(document, reconcile_identity)
            document = drop_contained_objects(document)
        document = apply_assembly(document)
    document = collapse_repeated_actions(document)
    return document, locate_info


def _remember(session, run, document: AnalysisDocument, note: str) -> None:
    record_lineage(run, note, document)
    revision = session.scalar(
        select(AnalysisRevision).where(
            AnalysisRevision.run_id == run.id,
            AnalysisRevision.layer == "parsed",
        )
    )
    payload = document.model_dump(mode="json")
    if revision is None:
        revision = AnalysisRevision(
            run_id=run.id,
            layer="parsed",
            version=1,
            document_json=payload,
            note=note,
        )
        session.add(revision)
        session.flush()
    else:
        revision.document_json = payload
        revision.note = note
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id)))
    if not frames:
        frames = list(
            session.scalars(select(FrameAsset).where(FrameAsset.video_id == run.video_id, FrameAsset.run_id.is_(None)))
        )
    materialize(session, run.id, revision.id, document, frames)
    if not _human_active(session, run):
        run.active_revision_id = revision.id
    session.commit()


def _publish_current(session, run) -> None:
    document = load_working(run.id)
    if document.objects or document.actions:
        _remember(session, run, document, "object_extraction")


async def _segment(session, run, video, stage) -> None:
    mode = run.segmentation_mode
    if mode in {"disabled", "boxes_only"}:
        raise StageSkipped(
            "Segmentation is off. Object images use boxes and rectangular crops from the analysis."
        )
    if not run.active_revision_id:
        raise StageSkipped("No analysis revision is available to segment.")
    if mode in {"sam", "sam_video"}:
        if not _has_meta_key(session):
            raise UnsupportedMode(
                "SAM 3.1 was selected, but no Meta Model API key is configured. "
                "Local segmentation was not substituted. Add MODEL_API_KEY or choose a different segmentation mode."
            )
        from app.pipeline.sam_stage import run_image_segmentation, run_video_segmentation

        if mode == "sam_video":
            try:
                result = await run_video_segmentation(session, run, video, stage)
            finally:
                document = load_working(run.id)
                if any(item.source_model == "sam-3.1" for item in document.observations):
                    _remember(session, run, document, "segmentation")
            stage.checkpoint_json = {key: value for key, value in result.items() if key != "document"}
            return
        stage.checkpoint_json = await run_image_segmentation(session, run, stage)
        return
    if mode == "local_grabcut":
        _run_grabcut(session, run, stage)
        return
    raise StageSkipped(f"Unknown segmentation mode {mode}.")


def _has_meta_key(session) -> bool:
    from app.services.credentials import load_secret

    secret, _source = load_secret(session, "meta")
    return bool(secret)


async def _evidence_fusion(session, run, video, stage) -> None:
    """Resolve canonical geometry and identity. This stage never calls a language model.

    It is the only writer of canonical rows. It rebuilds them from the
    tracker document, the SAM rows already fused into the working document,
    and the model's raw estimates, then replaces the working document, so a
    retry does not append.
    """
    from app.perception.resolve import resolve_document

    if run.provider_id == "fixture":
        raise StageSkipped("Fixture documents are authored. Their geometry was not resolved.")
    document = load_working(run.id)
    if not document.objects and not document.observations:
        raise StageSkipped("There is no analysis to resolve.")
    notes: list[str] = []
    segmentation = _stage(session, run.id, "segmentation")
    if segmentation.status == "failed":
        notes.append("SAM segmentation failed. Geometry was resolved from tracker and model evidence only.")
    elif segmentation.status == "skipped":
        notes.append("Segmentation was skipped. Geometry was resolved from tracker and model evidence only.")
    spatial = load_spatial_document(run.id)
    usable = spatial is not None and spatial.perception is not None and spatial.perception.used_for_identity
    if usable:
        document = apply_perception(document, spatial)
    elif spatial is not None:
        notes.append("Tracking was not used for identity, so tracker rows did not bind to objects.")
    resolved = resolve_document(document, spatial if usable else None)
    if semantic_mode(run) == "combined":
        # Measured rows exist only from here on. Grounded runs read them in the interpretation stage instead.
        resolved = collapse_repeated_actions(apply_assembly(resolved))
    coverage = _record_semantic(session, run, Coverage.model_validate(run.coverage_json or {}))
    resolved = resolved.model_copy(update={"coverage": coverage})
    run.coverage_json = coverage.model_dump(mode="json")
    save_working(run.id, resolved)
    _remember(session, run, resolved, "evidence_fusion")
    diagnostics = resolved.perception.diagnostics if resolved.perception else {}
    stage.checkpoint_json = {
        "source": "resolver",
        "model_calls": 0,
        "resolution": diagnostics.get("resolution", {}),
        "grounding": diagnostics.get("grounding", {}),
        "notes": notes,
    }


async def _interpret(session, run, video, stage) -> None:
    """Grounded mode: one semantic call that reads resolved geometry.

    Objects, canonical rows, and the display set from evidence fusion are
    kept. The call replaces actions, states, relationships, workflow, and
    ambiguities. Boxes it returns are stored as raw estimates only.
    """
    document = load_working(run.id)
    if run.provider_id in {"deterministic", "fixture", "manual_json"}:
        stage.checkpoint_json = {"source": run.provider_id, "model_calls": 0}
        _remember(session, run, document, stage.stage)
        return
    frames = distinct_moments(
        list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp)))
    )
    coverage = Coverage.model_validate(run.coverage_json or {})
    spatial = load_spatial_document(run.id)
    from app.providers.catalog import lookup

    config = dict(run.config_json or {})
    config["max_images_per_call"] = lookup(run.provider_id, run.model_id).max_images
    run.config_json = config
    proposed = await produce_document(
        session, run, video, frames, "workflow", coverage, spatial, purpose="interpretation"
    )
    merged = merge_interpretation(document, proposed)
    merged = collapse_repeated_actions(apply_assembly(merged))
    coverage = _record_semantic(session, run, coverage)
    merged = merged.model_copy(update={"coverage": coverage})
    run.coverage_json = coverage.model_dump(mode="json")
    save_working(run.id, merged)
    _remember(session, run, merged, stage.stage)
    stage.checkpoint_json = {
        "source": "interpretation",
        "provider": run.provider_id,
        "model": run.model_id,
        "actions": len(merged.actions),
        "workflow_steps": len(merged.workflow_steps),
        "semantic_calls": coverage.semantic_calls,
    }


def _run_grabcut(session, run, stage) -> None:
    from app.models.domain import DetectedObject, ObjectImage, ObjectObservation

    objects = list(
        session.scalars(select(DetectedObject).where(DetectedObject.revision_id == run.active_revision_id))
    )
    written = 0
    for obj in objects:
        obs = session.scalar(
            select(ObjectObservation)
            .where(
                ObjectObservation.object_id == obj.id,
                ObjectObservation.bbox_json.is_not(None),
                ObjectObservation.visibility != "absent",
            )
            .order_by(ObjectObservation.timestamp)
        )
        if obs is None or not obs.frame_id:
            continue
        frame = session.get(FrameAsset, obs.frame_id)
        if frame is None:
            continue
        image = read_bgr(data_path(*Path(frame.path).parts))
        from app.schemas.analysis import BBox

        cutout = grabcut_cutout(image, BBox.model_validate(obs.bbox_json))
        if cutout is None:
            continue
        dest = data_path("runs", run.id, "objects", obj.local_id, "grabcut.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), cutout)
        session.add(
            ObjectImage(
                object_id=obj.id,
                revision_id=run.active_revision_id,
                kind="cutout",
                path=relative(dest),
                frame_id=frame.id,
                timestamp=obs.timestamp,
                width=cutout.shape[1],
                height=cutout.shape[0],
                score=0,
            )
        )
        session.add(
            SegmentationMask(
                object_id=obj.id,
                revision_id=run.active_revision_id,
                run_id=run.id,
                frame_id=frame.id,
                timestamp=obs.timestamp,
                encoding="png",
                path=relative(dest),
                bbox_json=obs.bbox_json,
                source="grabcut",
                width=cutout.shape[1],
                height=cutout.shape[0],
                identity_uncertain=True,
            )
        )
        written += 1
    stage.checkpoint_json = {
        "source": "grabcut",
        "cutouts": written,
        "note": "GrabCut was initialized from stored boxes. These masks are not SAM output.",
    }
    stage.progress_total = max(len(objects), 1)
    stage.progress_current = written


def _validate(session, run, video, stage) -> None:
    document = load_working(run.id)
    issues = validate_document(document, video.duration_sec)
    errors = [issue for issue in issues if issue.severity == "error"]
    revision = AnalysisRevision(
        run_id=run.id,
        layer="validated",
        version=_next_version(session, run.id, "validated"),
        parent_id=run.active_revision_id,
        document_json=document.model_dump(mode="json"),
        note=f"{len(errors)} validation errors, {len(issues) - len(errors)} warnings",
    )
    session.add(revision)
    session.flush()
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id)))
    materialize(session, run.id, revision.id, document, frames)
    if not _human_active(session, run):
        run.active_revision_id = revision.id
    stage.checkpoint_json = {"errors": len(errors), "warnings": len(issues) - len(errors)}
    stage.progress_total = max(len(issues), 1)
    stage.progress_current = max(len(issues), 1)


async def _reinspect(session, run, video, stage) -> None:
    if (run.config_json or {}).get("analysis_pass") != "multi_reinspect":
        raise StageSkipped("Targeted reinspection is off for this experiment.")
    if run.provider_id in {"deterministic", "fixture", "manual_json"}:
        raise StageSkipped("Reinspection needs a vision-language provider.")
    document = load_working(run.id)
    notes = []
    for ambiguity in document.ambiguities[:4]:
        if ambiguity.time_start is None:
            continue
        end = ambiguity.time_end if ambiguity.time_end is not None else ambiguity.time_start
        original = resolve_stored(video.stored_path)
        extra_rows = []
        cursor = ambiguity.time_start
        while cursor <= end and len(extra_rows) < 4:
            dest = data_path("frames", video.id, run.id, f"reinspect_{cursor:.3f}.jpg")
            extract_frame(original, cursor, dest)
            row = FrameAsset(
                video_id=video.id,
                run_id=run.id,
                timestamp=cursor,
                path=relative(dest),
                width=video.display_width,
                height=video.display_height,
                reason="reinspection",
                strategy="uncertainty_window",
                ordinal=1000 + len(extra_rows),
            )
            session.add(row)
            extra_rows.append(row)
            cursor += max((end - ambiguity.time_start) / 3, 0.2)
        session.flush()
        question = (
            f"What is visible between {ambiguity.time_start:.2f}s and {end:.2f}s? "
            "Compare the state before and after. Do not infer attachment from proximity. "
            f"Existing ambiguity: {ambiguity.description}"
        )
        try:
            refined = await produce_document(
                session,
                run,
                video,
                extra_rows,
                "reinspection",
                Coverage.model_validate(run.coverage_json or {}),
                None,
                user_extra=question,
            )
            proposal_note = (
                "Reinspection is a proposal. It was not written over the original conclusion "
                "and its certainty was not increased."
            )
            document.reinspection_proposals.append(
                ReinspectionProposal(
                    ambiguity_local_id=ambiguity.local_id,
                    summary=refined.summary,
                    evidence_classification="UNKNOWN",
                    changed=False,
                    applied=False,
                    notes=proposal_note,
                    frame_timestamps=[row.timestamp for row in extra_rows],
                )
            )
            notes.append(
                {
                    "ambiguity": ambiguity.local_id,
                    "question": question,
                    "original": ambiguity.description,
                    "refined_summary": refined.summary,
                    "frames": [row.timestamp for row in extra_rows],
                    "applied": False,
                }
            )
        except Exception as exc:
            notes.append({"ambiguity": ambiguity.local_id, "error": str(exc)})
    document.reinspection_notes = [json.dumps(item) for item in notes]
    save_working(run.id, document)
    stage.checkpoint_json = {"reinspections": len(notes)}
    stage.progress_total = max(len(document.ambiguities), 1)
    stage.progress_current = len(notes)


def _artifacts(session, run, stage) -> None:
    document = load_working(run.id)
    folder = data_path("runs", run.id)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "analysis": document.model_dump(mode="json"),
        "objects": [item.model_dump(mode="json") for item in document.objects],
        "actions": [item.model_dump(mode="json") for item in document.actions],
        "workflow": {
            "steps": [item.model_dump(mode="json") for item in document.workflow_steps],
            "dependencies": [item.model_dump(mode="json") for item in document.dependencies],
        },
    }
    for kind, payload in files.items():
        path = folder / f"{kind}.json"
        path.write_text(json.dumps(payload, indent=2))
        session.add(Artifact(run_id=run.id, kind=kind, path=relative(path), label=kind))
    frames = list(
        session.scalars(select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp))
    )
    if frames:
        sheet = folder / "run_contact_sheet.jpg"
        contact_sheet([data_path(*Path(frame.path).parts) for frame in frames[:30]], sheet)
        session.add(Artifact(run_id=run.id, kind="contact_sheet", path=relative(sheet), label="Run contact sheet"))
    stage.checkpoint_json = {"files": list(files)}
    stage.progress_total = len(files)
    stage.progress_current = len(files)


def _human_active(session, run) -> bool:
    if not run.active_revision_id:
        return False
    revision = session.get(AnalysisRevision, run.active_revision_id)
    return bool(revision and revision.layer == "human")


def _next_version(session, run_id: str, layer: str) -> int:
    versions = session.scalars(
        select(AnalysisRevision.version).where(
            AnalysisRevision.run_id == run_id,
            AnalysisRevision.layer == layer,
        )
    )
    current = list(versions)
    return (max(current) if current else 0) + 1


def ensure_prompt_snapshot(run: AnalysisRun) -> None:
    if run.config_json.get("prompt_snapshot"):
        return
    config = dict(run.config_json)
    config["prompt_snapshot"] = compose("scene", run.research_mode)
    run.config_json = config
