"""Export and reimport an experiment without credentials."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import SCHEMA_VERSION, __version__
from app.models.audio import AudioAlignment, TranscriptSegment
from app.models.domain import ObjectImage
from app.models.media import FrameAsset, VideoAsset
from app.models.runs import AnalysisRevision, AnalysisRun, ModelRequestRow, ModelResponseRow
from app.models.util import new_id
from app.schemas.analysis import AnalysisDocument
from app.services.experiments import create_run
from app.services.materialize import materialize
from app.services.storage import data_path, resolve_stored


def export_run(session: Session, run: AnalysisRun, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    revision = session.get(AnalysisRevision, run.active_revision_id) if run.active_revision_id else None
    video = session.get(VideoAsset, run.video_id)
    config = dict(run.config_json or {})
    config.pop("credentials", None)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "run_id": run.id,
        "name": run.name,
        "provider_id": run.provider_id,
        "model_id": run.model_id,
        "modality": run.modality,
        "research_mode": run.research_mode,
        "sampling_preset": run.sampling_preset,
        "fixture": run.fixture,
        "fixture_label": run.fixture_label,
        "video_checksum": video.checksum_sha256 if video else "",
        "video_filename": video.original_filename if video else "",
        "coverage": run.coverage_json,
        "config": config,
        "prompt_snapshot": config.get("prompt_snapshot", ""),
        "git_revision": run.git_revision,
    }
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        if revision:
            archive.writestr("analysis.json", json.dumps(revision.document_json, indent=2))
        archive.writestr("transcripts.json", json.dumps(_transcript_export(session, run.id), indent=2))
        archive.writestr("alignments.json", json.dumps(_alignment_export(session, run.id), indent=2))
        requests = session.scalars(select(ModelRequestRow).where(ModelRequestRow.run_id == run.id)).all()
        archive.writestr(
            "requests.json",
            json.dumps(
                [
                    {
                        "id": item.id,
                        "stage": item.stage,
                        "model_id": item.model_id,
                        "modality": item.modality,
                        "parameters": item.parameters_json,
                        "frame_timestamps": item.frame_timestamps,
                        "provider_request_id": item.provider_request_id,
                        "status": item.status,
                        "error": item.error,
                    }
                    for item in requests
                ],
                indent=2,
            ),
        )
        responses = session.scalars(select(ModelResponseRow).where(ModelResponseRow.run_id == run.id)).all()
        for item in responses:
            archive.writestr(f"responses/{item.id}.json", json.dumps(item.raw_json, indent=2))
            archive.writestr(f"responses/{item.id}.txt", item.text or "")
        if revision:
            images = session.scalars(select(ObjectImage).where(ObjectImage.revision_id == revision.id)).all()
            for image in images:
                if not image.path:
                    continue
                path = data_path(*Path(image.path).parts)
                if path.exists():
                    archive.write(path, f"object_images/{image.kind}/{path.name}")
        if video:
            frames = session.scalars(
                select(FrameAsset).where(FrameAsset.run_id == run.id).order_by(FrameAsset.timestamp)
            ).all()
            archive.writestr(
                "frames.json",
                json.dumps(
                    [
                        {"timestamp": frame.timestamp, "reason": frame.reason, "path": frame.path}
                        for frame in frames
                    ],
                    indent=2,
                ),
            )
            original = resolve_stored(video.stored_path)
            if original.exists() and original.stat().st_size <= 80 * 1024 * 1024:
                archive.write(original, f"media/{video.original_filename}")
    return dest


def import_run(session: Session, project_id: str, archive_path: Path) -> AnalysisRun:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"export schema {manifest.get('schema_version')} is not {SCHEMA_VERSION}. "
                "Add a migration before importing it."
            )
        analysis = json.loads(archive.read("analysis.json"))
        document = AnalysisDocument.model_validate(analysis)
        video = session.scalar(
            select(VideoAsset).where(
                VideoAsset.project_id == project_id,
                VideoAsset.checksum_sha256 == manifest.get("video_checksum"),
            )
        )
        if video is None:
            media_name = next((name for name in archive.namelist() if name.startswith("media/")), None)
            if media_name is None:
                raise ValueError("archive has no video and the checksum is not in this project")
            dest = data_path("imports", Path(media_name).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(media_name))
            from app.services.ingest import ingest_file

            video = ingest_file(session, project_id, Path(media_name).name, dest)
    run = create_run(
        session,
        {
            "name": f"Imported · {manifest.get('name')}",
            "video_id": video.id,
            "provider_id": "manual_json",
            "model_id": manifest.get("model_id") or "imported-json",
            "modality": "sampled_frames",
            "research_mode": manifest.get("research_mode") or "fixed_camera",
            "sampling_preset": manifest.get("sampling_preset") or "standard",
            "analysis_pass": "single",
            "segmentation_mode": "disabled",
        },
        enqueue=False,
    )
    run.fixture = bool(document.fixture or manifest.get("fixture"))
    run.fixture_label = document.fixture_label or manifest.get("fixture_label") or ""
    run.status = "completed"
    run.coverage_json = manifest.get("coverage") or {}
    config = dict(run.config_json)
    config["imported_from"] = manifest.get("run_id")
    config["imported_provider"] = manifest.get("provider_id")
    config["imported_model"] = manifest.get("model_id")
    config["prompt_snapshot"] = manifest.get("prompt_snapshot") or config.get("prompt_snapshot")
    run.config_json = config
    revision = AnalysisRevision(
        id=new_id(),
        run_id=run.id,
        layer="validated",
        version=1,
        document_json=document.model_dump(mode="json"),
        note="imported archive",
    )
    session.add(revision)
    session.flush()
    frames = list(session.scalars(select(FrameAsset).where(FrameAsset.video_id == video.id)))
    materialize(session, run.id, revision.id, document, frames)
    run.active_revision_id = revision.id
    for stage in session.scalars(select_stages(run.id)):
        stage.status = "completed" if stage.stage != "segmentation" else "skipped"
    session.commit()
    return run


def select_stages(run_id: str):
    return select(PipelineStageRun).where(PipelineStageRun.run_id == run_id)


def _transcript_export(session: Session, run_id: str) -> list[dict]:
    rows = session.scalars(select(TranscriptSegment).where(TranscriptSegment.run_id == run_id)).all()
    return [
        {
            "id": row.id,
            "video_id": row.video_id,
            "local_id": row.local_id,
            "start": row.start,
            "end": row.end,
            "text": row.text,
            "text_model": row.text_model,
            "language": row.language,
            "confidence": row.confidence,
            "words": row.words_json or [],
            "model_name": row.model_name,
            "device": row.device,
            "corrected": row.corrected,
            "provenance": row.provenance,
        }
        for row in rows
    ]


def _alignment_export(session: Session, run_id: str) -> list[dict]:
    rows = session.scalars(select(AudioAlignment).where(AudioAlignment.run_id == run_id)).all()
    return [
        {
            "id": row.id,
            "segment_id": row.segment_id,
            "video_id": row.video_id,
            "classification": row.classification,
            "local_classification": row.local_classification,
            "explanation": row.explanation,
            "action_local_ids": row.action_local_ids or [],
            "object_local_ids": row.object_local_ids or [],
            "step_local_ids": row.step_local_ids or [],
            "frame_timestamps": row.frame_timestamps or [],
            "time_offset_sec": row.time_offset_sec,
            "confidence": row.confidence,
            "evidence_class": row.evidence_class,
            "provenance": row.provenance,
            "corrected": row.corrected,
            "speech_is_not_completion": row.speech_is_not_completion,
        }
        for row in rows
    ]


from app.models.runs import PipelineStageRun  # noqa: E402
