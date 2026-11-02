"""Transcript inspection. Whisper stays local; this router never receives audio bytes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.serialize import bundle
from app.audio.engine import WHISPER_MODELS, acceleration, model_cache_status
from app.audio.service import correct_alignment, correct_segment
from app.db import get_session
from app.evaluation.audio_compare import compare_audio_runs
from app.models.audio import AudioAlignment, TranscriptSegment
from app.models.runs import AnalysisRun
from app.services.experiments import duplicate_visual_only

router = APIRouter(prefix="/api")

CLASSES = {"visually_supported", "contextual", "contradictory", "irrelevant", "uncertain"}


@router.get("/audio/models")
def audio_models() -> dict:
    installed = True
    detail = ""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        installed = False
        detail = "Install with backend/.venv/bin/pip install faster-whisper. Weights download on first transcription."
    return {
        "installed": installed,
        "detail": detail,
        "models": model_cache_status(),
        "acceleration": acceleration(),
        "supported_models": list(WHISPER_MODELS),
        "default_model": "base",
        "audio_bytes_leave_machine": False,
    }


@router.patch("/runs/{run_id}/transcript/{segment_id}")
def patch_transcript(run_id: str, segment_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    row = session.get(TranscriptSegment, segment_id)
    if row is None or row.run_id != run_id:
        raise HTTPException(404, "transcript segment not found")
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "transcript text is empty")
    correct_segment(row, text)
    session.commit()
    run = session.get(AnalysisRun, run_id)
    return bundle(session, run)


@router.patch("/runs/{run_id}/alignments/{alignment_id}")
def patch_alignment(run_id: str, alignment_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    row = session.get(AudioAlignment, alignment_id)
    if row is None or row.run_id != run_id:
        raise HTTPException(404, "alignment not found")
    label = payload.get("classification") or row.classification
    if label not in CLASSES:
        raise HTTPException(400, "unknown classification")
    correct_alignment(row, label, payload.get("explanation") or "")
    session.commit()
    run = session.get(AnalysisRun, run_id)
    return bundle(session, run)


@router.post("/runs/{run_id}/visual-only")
def visual_only(run_id: str, session: Session = Depends(get_session)) -> dict:
    run = session.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    copy = duplicate_visual_only(session, run)
    return {"id": copy.id, "name": copy.name, "status": copy.status}


@router.post("/runs/{run_id}/audio-comparison")
def audio_comparison(run_id: str, payload: dict, session: Session = Depends(get_session)) -> dict:
    assisted = session.get(AnalysisRun, payload.get("assisted_run_id") or run_id)
    visual = session.get(AnalysisRun, payload.get("visual_run_id") or "")
    if assisted is None or visual is None:
        raise HTTPException(404, "both runs are required")
    if assisted.video_id != visual.video_id:
        raise HTTPException(400, "the two runs must use the same recording")
    report = compare_audio_runs(session, visual, assisted)
    report["visual_run_id"] = visual.id
    report["assisted_run_id"] = assisted.id
    return report
