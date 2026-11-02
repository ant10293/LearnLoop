"""Build the three local fixtures and run the fixture provider on each."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.demo.encode import write_clip
from app.demo.scenarios import SCENARIOS, render_scenario
from app.models.media import Project
from app.pipeline.infer import register_fixture
from app.services.experiments import create_run
from app.services.ingest import ingest_file
from app.services.storage import data_path

log = logging.getLogger("learnloop.seed")


def seed_if_empty(session: Session) -> None:
    existing = session.scalar(select(Project).limit(1))
    if existing is not None:
        return
    project = Project(
        name="Fixture laboratory",
        description=(
            "Three authored recordings for stationary, constrained head-mounted, and unconstrained "
            "head-mounted research. The analyses are fixtures, not model output."
        ),
    )
    session.add(project)
    session.commit()
    for key, factory in SCENARIOS.items():
        scenario = factory()
        clip = data_path("imports", f"{key}.mp4")
        write_clip(render_scenario(scenario), clip)
        video = ingest_file(session, project.id, f"{key}.mp4", clip)
        register_fixture(video.checksum_sha256, key)
        create_run(
            session,
            {
                "name": scenario.title,
                "video_id": video.id,
                "provider_id": "fixture",
                "model_id": "authored-fixture",
                "modality": "sampled_frames",
                "research_mode": key,
                "sampling_preset": "detailed",
                "analysis_pass": "single",
                "segmentation_mode": "boxes_only",
                "max_frames": 24,
            },
        )
        log.info("queued fixture %s", key)
