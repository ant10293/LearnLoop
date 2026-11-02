"""Record what the geometry looked like after each stage that saved it.

The parsed revision is rewritten in place during a run, so the database keeps no
stage snapshots. This small log answers one question without a paid call: did the
stored boxes change, and at which stage?
"""

from __future__ import annotations

import hashlib
import json

from app.schemas.analysis import AnalysisDocument
from app.services.storage import data_path

# Settings that only the segmentation stage reads.
SEGMENTATION_KEYS = {
    "segmentation_mode",
    "segmentation_model_id",
    "sam_max_requests",
    "sam_max_frames",
    "sam_chunk_frames",
    "sam_prompt_strategy",
    "sam_multi_seed",
}


def geometry_fingerprint(document: AnalysisDocument) -> dict:
    rows = sorted(
        (
            row.object_local_id,
            round(row.timestamp, 3),
            row.source_model,
            row.geometry_origin,
            row.geometry_authority,
            [round(value, 5) for value in (row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h)] if row.bbox else [],
        )
        for row in [*document.observations, *document.raw_observations]
    )
    sources: dict[str, int] = {}
    for row in rows:
        name = row[2] or row[4] or "model"
        sources[name] = sources.get(name, 0) + 1
    return {
        "digest": hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16],
        "boxes": sum(1 for row in rows if row[5]),
        "sources": sources,
    }


def upstream_config_digest(config: dict | None) -> str:
    """Hash of the settings that may affect stages before segmentation."""
    kept = {key: value for key, value in (config or {}).items() if key not in SEGMENTATION_KEYS}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()[:16]


def record_lineage(run, stage: str, document: AnalysisDocument) -> None:
    path = data_path("runs", run.id, "geometry_lineage.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "stage": stage,
        **geometry_fingerprint(document),
        "segmentation_mode": run.segmentation_mode,
        "upstream_config": upstream_config_digest(run.config_json),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def load_lineage(run_id: str) -> list[dict]:
    path = data_path("runs", run_id, "geometry_lineage.jsonl")
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
