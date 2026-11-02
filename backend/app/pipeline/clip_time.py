"""Native-video subclips and their clock.

A model that receives `runs/{id}/subclip.mp4` sees time zero at the start of
the requested interval. Its answer is shifted by the recorded offset before
anything else reads it, so every stored time stays on the source clock.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.analysis import AnalysisDocument, Evidence
from app.services.storage import data_path


def subclip_path(run_id: str) -> Path:
    return data_path("runs", run_id, "subclip.mp4")


def transform_path(run_id: str) -> Path:
    return data_path("runs", run_id, "subclip.json")


def save_transform(run_id: str, transform: dict) -> None:
    path = transform_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transform, indent=2))


def load_transform(run_id: str) -> dict | None:
    path = transform_path(run_id)
    if not path.exists() or not subclip_path(run_id).exists():
        return None
    return json.loads(path.read_text())


def _evidence(evidence: Evidence, offset: float) -> None:
    evidence.frame_timestamps = [round(item + offset, 6) for item in evidence.frame_timestamps]
    if evidence.time_start is not None:
        evidence.time_start = round(evidence.time_start + offset, 6)
    if evidence.time_end is not None:
        evidence.time_end = round(evidence.time_end + offset, 6)


def shift_to_source(document: AnalysisDocument, offset: float) -> AnalysisDocument:
    """Add `offset` to every model-reported time. Coverage and perception are already on the source clock."""
    if abs(offset) < 1e-9:
        return document
    for item in document.observations:
        item.timestamp = round(item.timestamp + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.states:
        item.time_start = round(item.time_start + offset, 6)
        item.time_end = round(item.time_end + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.transitions:
        item.timestamp = round(item.timestamp + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.relationships:
        item.time_start = round(item.time_start + offset, 6)
        item.time_end = round(item.time_end + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.actions:
        item.start = round(item.start + offset, 6)
        item.end = round(item.end + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.workflow_steps:
        if item.time_start is not None:
            item.time_start = round(item.time_start + offset, 6)
        if item.time_end is not None:
            item.time_end = round(item.time_end + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.dependencies:
        _evidence(item.evidence, offset)
    for item in document.ambiguities:
        if item.time_start is not None:
            item.time_start = round(item.time_start + offset, 6)
        if item.time_end is not None:
            item.time_end = round(item.time_end + offset, 6)
        _evidence(item.evidence, offset)
    for item in document.objects:
        if getattr(item, "evidence", None) is not None:
            _evidence(item.evidence, offset)
    return document
