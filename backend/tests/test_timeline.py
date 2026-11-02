"""FFmpeg alignment on generated clips. Every frame carries its own index in its pixels.

These are generated media, not recordings. They check the clock and the
coordinate frame, not perception accuracy.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.demo.encode import write_clip
from app.demo.scenarios import FPS, HEIGHT, WIDTH
from app.perception import stream
from app.perception.stream import StreamStats, plan_tracking, stream_views
from app.pipeline.clip_time import shift_to_source
from app.schemas.analysis import (
    AnalysisDocument,
    BBox,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
)
from app.schemas.enums import EvidenceClass, Provenance
from app.video.decode import cut_clip, jpeg_pass
from app.video.probe import probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for alignment tests")

SECONDS = 8
FRAME = 1.0 / FPS


def _frame(index: int) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    # Orientation marker: a white bar in the top-left corner of the stored picture.
    image[10:50, 10:130] = 255
    # The index, as two gray patches with sixteen levels each.
    image[0:64, 560:624] = (index % 16) * 16
    image[80:144, 560:624] = (index // 16) * 16
    # A moving square so the clip is not static.
    x = 150 + (index * 4) % 300
    image[200:240, x : x + 40] = (40, 200, 40)
    return image


def _index(image: np.ndarray) -> int:
    low = int(round(float(image[8:56, 568:616].mean()) / 16))
    high = int(round(float(image[88:136, 568:616].mean()) / 16))
    return high * 16 + low


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("clips") / "indexed.mp4"
    write_clip([_frame(index) for index in range(SECONDS * FPS)], dest)
    return dest


def test_extracted_frames_carry_their_source_time(clip: Path, tmp_path: Path):
    rows = jpeg_pass(clip, 3.0, 6.0, 0.5, tmp_path / "frames")
    assert len(rows) >= 6
    for timestamp, path in rows:
        assert 3.0 - 1e-6 <= timestamp <= 6.0 + FRAME
        index = _index(cv2.imread(str(path)))
        # Accurate seek can land one frame off. A larger miss fails.
        assert abs(index * FRAME - timestamp) <= FRAME + 1e-6


def test_a_subclip_keeps_its_bounds_and_maps_back_to_source(clip: Path, tmp_path: Path):
    dest = tmp_path / "sub.mp4"
    transform = cut_clip(clip, 2.0, 5.0, dest)
    assert transform["offset"] == 2.0 and transform["duration"] == 3.0
    meta = probe(dest)
    assert abs(meta.duration_sec - 3.0) <= 2 * FRAME
    rows = jpeg_pass(dest, 0.0, meta.duration_sec, 0, tmp_path / "sub_frames")
    first_index = _index(cv2.imread(str(rows[0][1])))
    last_index = _index(cv2.imread(str(rows[-1][1])))
    assert abs(first_index - 20) <= 1
    assert abs(last_index - 49) <= 1
    for clip_time, path in rows[::7]:
        source_time = clip_time + transform["offset"]
        assert abs(_index(cv2.imread(str(path))) * FRAME - source_time) <= FRAME + 1e-6

    document = AnalysisDocument(
        objects=[ObjectPrediction(local_id="Object_1", label="green square")],
        observations=[
            ObservationPrediction(
                object_local_id="Object_1",
                timestamp=1.0,
                bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1),
                evidence=Evidence(
                    classification=EvidenceClass.DIRECTLY_OBSERVED,
                    provenance=Provenance.AI_PREDICTION,
                    frame_timestamps=[1.0],
                    time_start=0.5,
                    time_end=1.5,
                ),
            )
        ],
    )
    shifted = shift_to_source(document, transform["offset"])
    row = shifted.observations[0]
    assert row.timestamp == pytest.approx(3.0)
    assert row.evidence.frame_timestamps == [pytest.approx(3.0)]
    assert row.evidence.time_start == pytest.approx(2.5)


def test_capped_tracking_reaches_the_end_across_processing_chunks(clip: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(stream, "CHUNK_SEC", 2.0)
    duration = probe(clip).duration_sec
    plan = plan_tracking([], 0.0, duration, peak_fps=8.0, maximum_frames=20, occlusion_seconds=1.0)
    assert plan.floor_fps == 2.0
    stats = StreamStats()
    views = list(stream_views(clip, plan, tmp_path / "work", stats))
    assert stats.chunks == 4
    assert stats.times[-1] >= duration - 0.5 - FRAME
    assert all(signal.boundary != "hard" for signal in stats.signals)
    for view in views:
        assert abs(_index(view.image) * FRAME - view.timestamp) <= FRAME + 1e-6
    summary = stats.summary(plan, 1.0)
    assert summary["continuity"] == "supported"
    assert summary["processing_chunks"] == 4


def test_rotated_media_reaches_segmentation_in_the_displayed_frame(clip: Path, tmp_path: Path, monkeypatch):
    from app.config import settings
    from app.pipeline.sam_stage import _prepare_media

    rotated = tmp_path / "rotated.mp4"
    done = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(clip), "-c", "copy", str(rotated)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"this ffmpeg cannot write display rotation: {done.stderr[-200:]}")
    meta = probe(rotated)
    assert meta.rotation_deg % 180 == 90
    assert (meta.display_width, meta.display_height) == (HEIGHT, WIDTH)

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    prepared, info = _prepare_media(rotated, "run-rotation", meta.rotation_deg)
    assert info["orientation"] == "displayed"
    assert (info["width"], info["height"]) == (HEIGHT, WIDTH)

    shown = cv2.imread(str(jpeg_pass(rotated, 0.0, 0.05, 0, tmp_path / "shown")[0][1]))
    sent = cv2.imread(str(jpeg_pass(prepared, 0.0, 0.05, 0, tmp_path / "sent")[0][1]))
    assert shown.shape == sent.shape == (WIDTH, HEIGHT, 3)
    # The tracker and segmentation see the same picture, so a normalized box means the same place in both.
    assert float(np.mean(cv2.absdiff(shown, sent))) < 6.0
    stored = _frame(0)
    turned = [np.rot90(stored, 1), np.rot90(stored, -1)]
    assert min(float(np.mean(cv2.absdiff(shown, item))) for item in turned) < 6.0
