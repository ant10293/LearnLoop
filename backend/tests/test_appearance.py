"""Appearance galleries and SAM track checks. No test loads DINOv2 weights or reaches a provider."""

import json
from types import SimpleNamespace

import numpy as np
from app.perception import appearance as A
from app.perception.encoder import preprocess

from tests.test_learnloop_flow import _analyze, _upload, lab  # noqa: F401


def _vec(*values):
    vector = np.zeros(8, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def _sample(local_id, t, vector, box=None, segment=0, **quality):
    sample = A.Sample(local_id, t, int(round(t * 12)), box or {"x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1}, 0.9, segment=segment)
    sample.vector = vector
    sample.crop = np.zeros((20, 20, 3), np.uint8)
    sample.quality.update({"relative_sharpness": 1.0, "mask_coverage": 0.6, **quality})
    return sample


RED, BLUE, GREEN = _vec(1, 0.1, 0), _vec(0, 0.1, 1), _vec(0, 1, 0.1)
OPTIONS = A.Settings()


def _view(base, t):
    """The same part from a slightly different angle at each time: still a match, not a duplicate."""
    turn = np.zeros(8, dtype=np.float32)
    turn[3 + int(t * 3) % 5] = 0.35
    vector = base + turn
    return vector / np.linalg.norm(vector)


def test_a_gap_or_a_jump_starts_a_new_segment_and_steady_motion_does_not():
    box = lambda x, w=0.05: {"x": x, "y": 0.5, "w": w, "h": 0.05}  # noqa: E731
    times = [0.0, 0.083, 0.167, 1.5, 1.583, 1.667]
    boxes = [box(0.1), box(0.11), box(0.12), box(0.13), box(0.8), box(0.81)]
    assert A.split_segments(times, boxes) == [0, 0, 0, 1, 2, 2]
    grown = [box(0.1), box(0.1, 0.3)]
    assert A.split_segments([0.0, 0.083], grown) == [0, 1]


def test_quality_rejects_small_blurry_clipped_and_overlapping_crops():
    sharp = np.zeros((40, 40, 3), np.uint8)
    sharp[::4] = 255
    blurry = np.full((40, 40, 3), 120, np.uint8)
    items = [
        A.Sample("a", 0.0, 0, {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}, 0.9, crop=sharp),
        A.Sample("a", 0.5, 6, {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}, 0.9, crop=sharp),
        A.Sample("a", 1.0, 12, {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}, 0.9, crop=blurry),
        A.Sample("a", 1.5, 18, {"x": 0.3, "y": 0.3, "w": 0.005, "h": 0.005}, 0.9, crop=sharp),
        A.Sample("a", 2.0, 24, {"x": 0.0, "y": 0.3, "w": 0.1, "h": 0.1}, 0.9, crop=sharp),
        A.Sample("a", 2.5, 30, {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}, 0.9, crop=sharp),
    ]
    # A small part sitting inside a bigger box: IoU is low, containment is not.
    others = {2.5: [("b", {"x": 0.25, "y": 0.25, "w": 0.3, "h": 0.3})]}
    A.score_quality(items, others, (480, 640), OPTIONS)
    assert items[0].rejections == [] and items[1].rejections == []
    assert "blurry" in items[2].rejections
    assert "too_small" in items[3].rejections
    assert "clipped" in items[4].rejections
    assert "overlaps:b" in items[5].rejections


def test_references_come_from_the_seeded_stretch_and_stop_where_it_drifts():
    # a is seeded at 0. SAM follows it to 3s, then slides onto something blue in the same stretch.
    a = [_sample("a", t, _view(RED, t)) for t in np.arange(0, 3.0, 0.34)] + [_sample("a", t, BLUE) for t in np.arange(3.0, 6.0, 0.34)]
    a += [_sample("a", 6.5, RED)]  # after the drift, even a red crop is not trusted in that stretch
    b = [_sample("b", t, BLUE, box={"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}) for t in np.arange(0, 6.0, 0.5)]
    chosen = A.select_references({"a": a, "b": b}, {"a": [0.0], "b": [0.0]}, {"a": "red block", "b": "blue block"}, OPTIONS)
    refs = chosen["a"]["references"]
    assert chosen["a"]["status"] == "ready"
    assert all(item.timestamp < 3.0 for item in refs)
    assert 3 <= len(refs) <= OPTIONS.max_references
    assert chosen["a"]["rejection_reasons"].get("drifted") == 1 and chosen["a"]["rejection_reasons"].get("after_drift", 0) >= 8
    # Temporal diversity: no two references closer than the minimum gap.
    gaps = np.diff([item.timestamp for item in refs])
    assert (gaps >= A.MIN_REFERENCE_GAP_SEC - 1e-6).all()


def test_selection_is_deterministic_and_keeps_few_views_as_limited():
    a = [_sample("a", 0.0, RED), _sample("a", 0.4, RED)]
    first = A.select_references({"a": a}, {"a": [0.0]}, {"a": "red"}, OPTIONS)
    second = A.select_references({"a": a}, {"a": [0.0]}, {"a": "red"}, OPTIONS)
    assert [s.timestamp for s in first["a"]["references"]] == [s.timestamp for s in second["a"]["references"]] == [0.0]
    assert first["a"]["status"] == "limited"


def test_no_seed_means_no_reference_rather_than_an_invented_one():
    a = [_sample("a", t, RED) for t in (0.0, 1.0, 2.0)]
    chosen = A.select_references({"a": a}, {}, {"a": "red"}, OPTIONS)
    assert chosen["a"]["status"] == "unavailable" and chosen["a"]["references"] == []


def test_blurry_crops_never_become_references():
    a = [_sample("a", 0.0, RED)] + [_sample("a", t, RED) for t in (1.0, 2.0, 3.0)]
    for item in a[1:]:
        item.rejections.append("blurry")
    chosen = A.select_references({"a": a}, {"a": [0.0]}, {"a": "red"}, OPTIONS)
    assert [s.timestamp for s in chosen["a"]["references"]] == [0.0]
    assert chosen["a"]["rejection_reasons"]["blurry"] == 3


def _rows(own_vectors, galleries, labels=None, boxes=None):
    items = [_sample("a", index * 0.33, vector, box=(boxes or {}).get(index)) for index, vector in enumerate(own_vectors)]
    labels = labels or {"a": "red block", "b": "blue block"}
    return A.validate_samples({"a": items}, galleries, labels, {"a": {0}}, OPTIONS)["a"]


def test_two_frames_that_look_like_another_object_are_rejected_and_one_is_not():
    galleries = {"a": np.stack([RED]), "b": np.stack([BLUE])}
    rows = _rows([RED, RED, BLUE, BLUE, RED, BLUE, RED], galleries)
    assert [row["status"] for row in rows] == ["validated", "validated", "rejected", "rejected", "validated", "provisional", "validated"]
    assert rows[2]["reason"] == "looks_like:b"


def test_identical_parts_stay_ambiguous_instead_of_swapping():
    galleries = {"a": np.stack([RED]), "b": np.stack([RED])}
    rows = _rows([RED, RED], galleries, {"a": "red block", "b": "red block"})
    assert {row["status"] for row in rows} == {"ambiguous"}


def test_an_unfamiliar_view_is_provisional_not_rejected():
    # A hand over the part, or a new angle: unlike the references, and nothing else matches either.
    galleries = {"a": np.stack([RED]), "b": np.stack([BLUE])}
    rows = _rows([RED, GREEN, GREEN, RED], galleries)
    assert [row["status"] for row in rows] == ["validated", "provisional", "provisional", "validated"]


def test_a_box_that_grows_past_the_object_is_rejected():
    galleries = {"a": np.stack([RED]), "b": np.stack([BLUE])}
    big = {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
    rows = _rows([RED, RED, GREEN, GREEN], galleries, boxes={2: big, 3: big})
    assert [row["status"] for row in rows][2:] == ["rejected", "rejected"]
    assert rows[2]["reason"] == "box_grew_beyond_object"


def test_a_crop_overlapping_another_object_is_not_called_a_swap():
    galleries = {"a": np.stack([RED]), "b": np.stack([BLUE])}
    items = [_sample("a", t, BLUE) for t in (0.0, 0.33)]
    for item in items:
        item.rejections.append("overlaps:b")
    rows = A.validate_samples({"a": items}, galleries, {"a": "red", "b": "blue"}, {"a": {0}}, OPTIONS)["a"]
    assert {row["status"] for row in rows} == {"provisional"} and rows[0]["reason"] == "overlaps_other_object"


def test_no_reference_means_provisional_for_every_box():
    rows = _rows([RED, BLUE], {"b": np.stack([BLUE])})
    assert {row["status"] for row in rows} == {"provisional"}


def test_rejected_never_appears_among_drawn_statuses():
    assert "rejected" not in A.SHOWN


def test_masked_preprocessing_differs_and_narrow_parts_are_padded_not_stretched():
    crop = np.zeros((100, 20, 3), np.uint8)
    crop[:, :, 2] = 255
    crop[:10] = (0, 255, 0)
    mask = np.zeros((100, 20), np.uint8)
    mask[10:] = 1
    plain, masked = preprocess(crop), preprocess(crop, mask)
    assert plain.shape == masked.shape == (3, 224, 224)
    assert not np.allclose(plain, masked)
    # The 100x20 crop becomes 224x45 in the middle; the sides are the zero padding value after normalization.
    assert np.allclose(plain[:, :, :80], 0, atol=1e-5) and not np.allclose(plain[:, :, 100:124], 0)


def test_cached_encode_passes_masks_and_keys_on_them(tmp_path):
    from app.perception.encoder import cached_encode

    seen = []

    class Recording:
        encoder_id, encoder_version, preprocess_version, dimension = "fake", "1", "p", 3

        def encode(self, crops, masks=None):
            seen.append(masks)
            return np.tile(np.array([[1.0, 0, 0]], np.float32), (len(crops), 1))

    crop = np.full((8, 8, 3), 9, np.uint8)
    mask = np.ones((8, 8), np.uint8)
    cached_encode(Recording(), [crop], [mask], tmp_path)
    cached_encode(Recording(), [crop], [None], tmp_path)
    assert seen[0][0] is mask and len(seen) == 2


def test_a_gallery_round_trips_and_another_version_is_never_read(lab):  # noqa: F811
    vectors = {"a": np.stack([RED, GREEN])}
    manifest = {
        "schema": A.GALLERY_SCHEMA, "status": "ready", "encoder_id": "dinov2_vits14",
        "encoder_version": A.DINOV2_VERSION, "preprocessing_version": A.PREPROCESS_VERSION,
        "selection_version": A.SELECTION_VERSION, "objects": {"a": {"status": "ready", "references": [{}, {}]}},
    }
    written = A.write_gallery("run-x", manifest, vectors, {"a": [np.zeros((4, 4, 3), np.uint8)]})
    read, loaded = A.read_gallery("run-x")
    assert read["gallery_id"] == written["gallery_id"]
    assert np.allclose(loaded["a"], vectors["a"]) and abs(np.linalg.norm(loaded["a"][0]) - 1) < 1e-5
    template = {"roles": [{"role_id": "role:a", "source_object_ids": ["a"]}, {"role_id": "role:z", "source_object_ids": ["z"]}]}
    _, by_role = A.role_galleries(template, "run-x")
    assert list(by_role) == ["role:a"]
    stale = json.loads((A.gallery_dir("run-x") / "manifest.json").read_text())
    stale["preprocessing_version"] = "old"
    (A.gallery_dir("run-x") / "manifest.json").write_text(json.dumps(stale))
    assert A.read_gallery("run-x") == (None, {})


class ColorEncoder:
    """Mean colour of the masked pixels. Deterministic, so the flow test needs no weights."""

    encoder_id, encoder_version, preprocess_version, dimension, device = "dinov2_vits14", "test", "test", 8, "cpu"

    def encode(self, crops, masks=None):
        rows = []
        for index, crop in enumerate(crops):
            mask = masks[index] if masks else None
            pixels = crop[mask > 0] if mask is not None and mask.any() else crop.reshape(-1, 3)
            b, g, r = pixels.reshape(-1, 3).mean(axis=0) / 255.0
            rows.append(_vec(b, g, r, 0.05))
        return np.stack(rows)


def _two_colour_video(path, frames=24):
    """A red square on the left and a blue one on the right of a grey 160x120 frame, at 12 fps."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
    for _ in range(frames):
        frame = np.full((120, 160, 3), 128, np.uint8)
        frame[10:60, 10:60] = (0, 0, 255)
        frame[10:60, 100:150] = (255, 0, 0)
        writer.write(frame)
    writer.release()
    return path


def _sam_answer(frames, drift_from=None):
    """Track 0 follows the red square; from drift_from on it slides onto the blue one. Track 1 follows blue."""
    lines = []
    for frame in range(frames):
        left = 100 if drift_from is not None and frame >= drift_from else 10
        lines.append(
            f"<{frame}f>0<|box;x1={left};y1=10;x2={left + 50};y2=60;w=160;h=120;c=0.9|><|mask;data=|>"
            f"1<|box;x1=100;y1=10;x2=150;y2=60;w=160;h=120;c=0.9|><|mask;data=|>"
        )
    return "\n".join(lines)


def test_learning_builds_a_gallery_automatically_and_drift_is_not_drawn(lab, monkeypatch):  # noqa: F811
    from app.config import settings
    from app.pipeline import appearance_stage, object_views

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    monkeypatch.setattr(appearance_stage, "shared_encoder", lambda *args: ColorEncoder())
    monkeypatch.setattr(object_views, "_low_rate_copy", lambda source, dest: source)
    answer = _sam_answer(24, drift_from=14)

    class FakeSam:
        def __init__(self, *args, **kwargs):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            return SimpleNamespace(text=answer, file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    ids = [item["local_id"] for item in json.loads(stub.text)["objects"]]
    stub.locate_text = json.dumps(
        {"objects": [{"local_id": ids[0], "bbox": {"x": 0.06, "y": 0.08, "w": 0.31, "h": 0.42}}, {"local_id": ids[1], "bbox": {"x": 0.62, "y": 0.08, "w": 0.31, "h": 0.42}}]}
    )
    procedure = client.post("/api/learnloop/procedures", json={"name": "Appearance"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _two_colour_video(tmp / "demo.mp4"), "demonstration"))
    assert run["status"] == "completed", run["error"]

    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert views["appearance"]["used"] is True
    track = views["objects"][ids[0]]["track"]
    # The drifted boxes (on the blue square, x ~ 0.62) are kept in raw_tracks.json and never drawn.
    assert track and all(row["x"] < 0.5 for row in track)
    assert {row["status"] for row in track} <= A.SHOWN
    raw = json.loads((object_views.views_dir(run["id"]) / "raw_tracks.json").read_text())["sightings"][ids[0]]
    assert len(raw) == 24 and any(row["status"] == "rejected" and row["box"]["x"] > 0.5 for row in raw)

    template = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    assert template["appearance"]["status"] in {"ready", "partial"}
    manifest, vectors = A.read_gallery(run["id"])
    references = manifest["objects"][ids[0]]["references"]
    assert references and all(ref["box"]["x"] < 0.5 for ref in references)
    assert vectors[ids[0]].shape[1] == 8

    from app.db import SessionLocal
    from app.models.runs import Job
    from app.services.appearance import APPEARANCE_JOB

    again = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    assert again["id"] == template["id"]
    with SessionLocal() as session:
        # The gallery was already made with the views, so learning queued no rebuild, twice.
        assert session.query(Job).filter(Job.from_stage == APPEARANCE_JOB).count() == 0


def test_a_missing_gallery_is_rebuilt_once_from_stored_answers_and_the_procedure_survives(lab, monkeypatch):  # noqa: F811
    from app.config import settings
    from app.db import SessionLocal
    from app.models.runs import Job
    from app.pipeline import appearance_stage, object_views
    from app.services.appearance import APPEARANCE_JOB

    from tests.test_learnloop_flow import _run_job

    client, stub, tmp = lab
    monkeypatch.setattr(settings, "object_views", True)
    monkeypatch.setattr(object_views, "_low_rate_copy", lambda source, dest: source)
    sam_calls = []

    class FakeSam:
        def __init__(self, *args, **kwargs):
            pass

        async def segment_video(self, media, prompt, **kwargs):
            sam_calls.append(prompt)
            return SimpleNamespace(text=_sam_answer(24), file_id="f", request_id="r", input_tokens=0, output_tokens=0, latency_ms=1)

    monkeypatch.setattr(object_views, "SamAdapter", FakeSam)
    ids = [item["local_id"] for item in json.loads(stub.text)["objects"]]
    stub.locate_text = json.dumps({"objects": [{"local_id": ids[0], "bbox": {"x": 0.06, "y": 0.08, "w": 0.31, "h": 0.42}}]})

    def unavailable(*args):
        from app.perception.types import EncoderUnavailable

        raise EncoderUnavailable("PyTorch is not installed.")

    # DINOv2 missing while the demonstration is analyzed: the analysis and the procedure are unaffected.
    monkeypatch.setattr(appearance_stage, "shared_encoder", unavailable)
    procedure = client.post("/api/learnloop/procedures", json={"name": "Rebuild"}).json()["id"]
    run = _analyze(client, _upload(client, procedure, _two_colour_video(tmp / "demo.mp4"), "demonstration"))
    assert run["status"] == "completed"
    views = client.get(f"/api/learnloop/runs/{run['id']}/objects").json()
    assert views["appearance"]["used"] is False and "PyTorch" in views["appearance"]["error"]
    assert {row["status"] for row in views["objects"][ids[0]]["track"]} == {"provisional"}

    template = client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]}).json()
    assert template["id"] and template["appearance"]["status"] == "building"
    client.post(f"/api/learnloop/procedures/{procedure}/learn", json={"run_id": run["id"]})
    with SessionLocal() as session:
        assert session.query(Job).filter(Job.from_stage == APPEARANCE_JOB).count() == 1

    # The encoder is back. The job replays the stored SAM answer: no new request is sent.
    monkeypatch.setattr(appearance_stage, "shared_encoder", lambda *args: ColorEncoder())
    sent = len(sam_calls)
    _run_job(run["id"])
    assert len(sam_calls) == sent
    page = client.get(f"/api/learnloop/procedures/{procedure}").json()
    assert page["template"]["appearance"]["status"] in {"ready", "partial"}
