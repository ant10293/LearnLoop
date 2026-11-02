"""Locate calls overlap, then boxes are applied in target order."""

import asyncio
import json

import cv2
import numpy as np
import pytest
from app.providers.types import ProviderResponse, Usage
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
)
from app.schemas.enums import EvidenceClass, Provenance, Visibility


def _evidence() -> Evidence:
    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.AI_PREDICTION,
    )


def _seen(object_id: str, timestamp: float, x: float = 0.2) -> ObservationPrediction:
    return ObservationPrediction(
        object_local_id=object_id,
        timestamp=timestamp,
        bbox=BBox(x=x, y=0.2, w=0.1, h=0.1),
        evidence=_evidence(),
    )


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNLOOP_DATABASE_URL", f"sqlite:///{tmp_path}/lab.db")
    monkeypatch.setenv("LEARNLOOP_WORKER_ENABLED", "0")
    monkeypatch.setenv("LEARNLOOP_AUTO_SEED", "0")
    from app.config import settings
    from app.db import rebind_engine

    settings.data_dir = tmp_path
    settings.database_url = f"sqlite:///{tmp_path}/lab.db"
    settings.worker_enabled = False
    settings.auto_seed = False
    rebind_engine(settings.sqlalchemy_url)
    from app.main import init_database

    init_database()
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _flat_frames(session):
    from app.models.media import FrameAsset, VideoAsset
    from app.models.runs import AnalysisRun
    from app.services.storage import data_path, relative

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    dest = data_path("frames", "flat.jpg")
    cv2.imwrite(str(dest), image)
    video = VideoAsset(
        original_filename="clip.mp4",
        stored_path="originals/clip.mp4",
        checksum_sha256="locate",
        duration_sec=4,
        display_width=100,
        display_height=100,
    )
    session.add(video)
    session.flush()
    run = AnalysisRun(
        video_id=video.id,
        name="locate",
        research_mode="fixed_camera",
        viewpoint="fixed",
        task_constraint="constrained",
        provider_id="openai",
        model_id="gpt-4.1",
        modality="sampled_frames",
        sampling_preset="standard",
        config_json={"max_requests": 12},
    )
    session.add(run)
    session.flush()
    frames = []
    for timestamp in (0, 1, 2, 3):
        frame = FrameAsset(
            video_id=video.id,
            run_id=run.id,
            timestamp=float(timestamp),
            path=relative(dest),
            width=100,
            height=100,
        )
        session.add(frame)
        frames.append(frame)
    session.commit()
    return run, frames


def _document(*object_ids: str) -> AnalysisDocument:
    objects = [
        ObjectPrediction(
            local_id=object_id,
            label=object_id,
            manipulated=object_id == "Object_1",
        )
        for object_id in object_ids
    ]
    observations = []
    for object_id in object_ids:
        observations.extend(_seen(object_id, float(timestamp)) for timestamp in (0, 1, 2))
    observations.append(
        ObservationPrediction(
            object_local_id=object_ids[0],
            timestamp=3,
            bbox=None,
            visibility=Visibility.ABSENT,
            evidence=_evidence(),
        )
    )
    return AnalysisDocument(objects=objects, observations=observations)


class _Provider:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.current = 0
        self.peak = 0
        self.seen: list[str] = []

    async def generate(self, _secret, request):
        object_id = next(part for part in request.user_prompt.split() if part.startswith("Object_"))
        object_id = object_id.rstrip(").,")
        self.seen.append(object_id)
        self.current += 1
        self.peak = max(self.peak, self.current)
        await asyncio.sleep(0.05)
        self.current -= 1
        if object_id in self.fail:
            raise RuntimeError("boom")
        body = {
            "observations": [
                {
                    "timestamp": 3,
                    "visibility": "visible",
                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
                    "confidence": 0.9,
                }
            ]
        }
        return ProviderResponse(text=json.dumps(body), raw={}, usage=Usage(), latency_ms=1)


def _install(monkeypatch, provider: _Provider) -> None:
    from app.pipeline import locate_pass

    monkeypatch.setattr(locate_pass, "needs_secret", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(locate_pass, "get_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(locate_pass, "model_for_call", lambda _session, run: (run.model_id, run.model_id))


def test_locate_calls_overlap_and_a_later_box_yields(session, monkeypatch):
    from app.pipeline.locate_pass import locate_missing

    run, frames = _flat_frames(session)
    provider = _Provider()
    _install(monkeypatch, provider)
    document, report = asyncio.run(locate_missing(session, run, _document("Object_1", "Object_2"), frames))
    assert provider.peak == 2
    assert sorted(provider.seen) == ["Object_1", "Object_2"]
    assert report["located"] == ["Object_1", "Object_2"]
    assert report["errors"] == []
    boxed = [item for item in document.observations if item.timestamp == 3 and item.bbox is not None]
    assert [item.object_local_id for item in boxed] == ["Object_1"]


def test_a_failed_locate_frees_its_budget_slot(session, monkeypatch):
    from app.pipeline.locate_pass import locate_missing

    run, frames = _flat_frames(session)
    run.config_json = {"max_requests": 1}
    session.commit()
    provider = _Provider(fail={"Object_1"})
    _install(monkeypatch, provider)
    document, report = asyncio.run(
        locate_missing(session, run, _document("Object_1", "Object_2", "Object_3"), frames)
    )
    assert provider.peak == 1
    assert provider.seen == ["Object_1", "Object_2"]
    assert report["located"] == ["Object_2"]
    assert report["errors"] == ["Object_1: boom", "Object_3: request budget reached"]
    boxed = [item for item in document.observations if item.timestamp == 3 and item.bbox is not None]
    assert [item.object_local_id for item in boxed] == ["Object_2"]


def test_a_broken_contract_fails_the_run_instead_of_losing_boxes(session, monkeypatch):
    from app.pipeline.contracts import SchemaContractError
    from app.pipeline.locate_pass import locate_missing

    class Broken(_Provider):
        async def generate(self, secret, request):
            raise SchemaContractError("#.observations[] must require every property.")

    run, frames = _flat_frames(session)
    _install(monkeypatch, Broken())
    with pytest.raises(SchemaContractError, match="must require every property"):
        asyncio.run(locate_missing(session, run, _document("Object_1"), frames))


def test_a_full_budget_does_not_start_a_locate_call(session, monkeypatch):
    from app.models.runs import ModelRequestRow
    from app.pipeline.locate_pass import locate_missing

    run, frames = _flat_frames(session)
    run.config_json = {"max_requests": 1}
    session.add(
        ModelRequestRow(
            run_id=run.id,
            stage="scene_analysis",
            provider_id="openai",
            model_id="gpt-4.1",
            modality="sampled_frames",
        )
    )
    session.commit()
    provider = _Provider()
    _install(monkeypatch, provider)
    _document_out, report = asyncio.run(locate_missing(session, run, _document("Object_1", "Object_2"), frames))
    assert provider.seen == []
    assert report == {"located": [], "errors": ["Object_1: request budget reached"]}
