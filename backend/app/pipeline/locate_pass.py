"""A second look for one part, using a picture of that part.

The scene pass names every object at once. It drops a part whose pose changed,
or it draws that part's id on a neighbor. Extent matching removes the neighbor's
box. This pass asks only where the missing part is, and only on the frames
where its own box is absent. A box that comes back larger than the part is rejected.
A box the model scores under 70% is rejected. A box with no score is still eligible.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.models.media import FrameAsset
from app.models.runs import AnalysisRun, ModelRequestRow
from app.perception.locate import apply_located, choose_locate_times, reference_time, same_surface
from app.pipeline.contracts import SchemaContractError
from app.pipeline.infer import model_for_call
from app.pipeline.parsing import ParseFailure, extract_json_object
from app.pipeline.prompts import LOCATE_PART
from app.providers.registry import get_provider, needs_secret, run_transport
from app.providers.types import DEFAULT_MAX_REQUESTS, FrameInput, GenerationRequest
from app.schemas.analysis import AnalysisDocument, BBox
from app.services.costs import fail_provider_call, finish_provider_call, open_provider_call
from app.services.credentials import load_secret
from app.services.storage import data_path

logger = logging.getLogger(__name__)

MAX_OBJECTS = 8


@dataclass(frozen=True)
class _Still:
    """A frame's path and size, copied before any locate call commits."""

    timestamp: float
    path: Path
    width: int
    height: int
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "timestamp": {"type": "number"},
                    "visibility": {"type": "string"},
                    "bbox": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "w": {"type": "number"},
                                    "h": {"type": "number"},
                                },
                                "required": ["x", "y", "w", "h"],
                            },
                            {"type": "null"},
                        ]
                    },
                    "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": ["timestamp", "visibility", "bbox", "confidence"],
            },
        }
    },
    "required": ["observations"],
}


async def locate_missing(
    session: Session,
    run: AnalysisRun,
    document: AnalysisDocument,
    frames: list[FrameAsset],
) -> tuple[AnalysisDocument, dict]:
    """Fill missing boxes for parts that already have a stable extent. Failures stay in the report.

    The model calls overlap. Each one reads this document and does not see another
    call's boxes. The boxes are applied in target order, so a later part still
    yields when its box lands on an earlier one. A call that fails does not spend
    a budget slot. A call that returns does, including one that adds no box.
    """
    if run.provider_id in {"fixture", "deterministic", "manual_json"} or not frames:
        return document, {"skipped": "no vision provider"}
    transport = run_transport(run.provider_id, run.config_json)
    secret = ""
    if needs_secret(run.provider_id, transport):
        secret, _source = load_secret(session, run.provider_id)
        if not secret:
            return document, {"skipped": "no API key"}
    config = run.config_json or {}
    sent = session.query(ModelRequestRow).filter(ModelRequestRow.run_id == run.id).count()
    budget = int(config.get("max_requests") or DEFAULT_MAX_REQUESTS)
    targets = _targets(document)[:MAX_OBJECTS]
    stills = _stills(frames)
    provider = get_provider(run.provider_id, session)
    gate = asyncio.Lock()
    found_boxes: dict[str, list] = {}
    failures: dict[str, BaseException] = {}
    pending = list(targets)
    in_flight: set[asyncio.Task] = set()
    run_id = run.id

    def start_ready() -> None:
        while pending and sent + len(in_flight) < budget:
            object_id, times = pending.pop(0)
            in_flight.add(
                asyncio.create_task(
                    _locate_object(
                        session,
                        gate,
                        provider,
                        run,
                        run_id,
                        document,
                        stills,
                        object_id,
                        times,
                        secret,
                        transport,
                    )
                )
            )

    try:
        while True:
            start_ready()
            if not in_flight:
                break
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                object_id, boxes, error = task.result()
                if error is None:
                    found_boxes[object_id] = boxes
                    sent += 1
                else:
                    logger.warning("locate failed for %s: %s", object_id, error)
                    failures[object_id] = error
    finally:
        for task in in_flight:
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)

    for failure in failures.values():
        # A broken contract is our defect, not one part the model failed to find. It must not pass as worse boxes.
        if isinstance(failure, SchemaContractError):
            raise failure
    errors = [f"{object_id}: {failures[object_id]}" for object_id, _times in targets if object_id in failures]
    if pending and sent >= budget:
        errors.append(f"{pending[0][0]}: request budget reached")
    for object_id, _times in targets:
        boxes = found_boxes.get(object_id)
        if boxes is None:
            continue
        document = apply_located(document, object_id, boxes)
    located = [object_id for object_id, _times in targets if object_id in found_boxes]
    return document, {"located": located, "errors": errors}


def _stills(frames: list[FrameAsset]) -> list[_Still]:
    return [
        _Still(
            timestamp=float(frame.timestamp),
            path=_stored(frame),
            width=int(frame.width or 0),
            height=int(frame.height or 0),
        )
        for frame in frames
    ]


def _targets(document: AnalysisDocument) -> list[tuple[str, list[float]]]:
    from app.perception.resolve import canonical_objects

    all_times = sorted({round(item.timestamp, 3) for item in document.observations})
    ranked: list[tuple[tuple, str, list[float]]] = []
    labels = {obj.local_id: obj for obj in document.objects}
    measured = canonical_objects(document)
    for object_id, obj in labels.items():
        if object_id in measured:
            # The tracker or SAM timeline already places this object. A locate call would only add an estimate.
            continue
        kept = [
            round(item.timestamp, 3)
            for item in document.observations
            if item.object_local_id == object_id and item.bbox is not None
        ]
        chosen = choose_locate_times(kept, all_times)
        if reference_time(document.observations, object_id) is None or not chosen:
            continue
        tail = sum(1 for timestamp in all_times if kept and timestamp > max(kept))
        priority = (0 if obj.manipulated else 1, -tail, object_id)
        ranked.append((priority, object_id, chosen))
    ranked.sort(key=lambda item: item[0])
    return [(object_id, times) for _priority, object_id, times in ranked]


async def _locate_object(
    session,
    gate: asyncio.Lock,
    provider,
    run,
    run_id: str,
    document,
    stills: list["_Still"],
    object_id: str,
    times: list[float],
    secret: str,
    transport: str,
) -> tuple[str, list, BaseException | None]:
    try:
        boxes = await _lookup(
            session, gate, provider, run, run_id, document, stills, object_id, times, secret, transport
        )
    except Exception as exc:
        return object_id, [], exc
    return object_id, boxes, None


async def _lookup(
    session,
    gate: asyncio.Lock,
    provider,
    run,
    run_id: str,
    document,
    stills: list["_Still"],
    object_id: str,
    times: list[float],
    secret: str,
    transport: str,
) -> list:
    """Ask for one part. The database is touched only while the gate is held."""
    reference = reference_time(document.observations, object_id)
    if reference is None or reference.bbox is None:
        return []
    source = _frame_at(stills, reference.timestamp)
    if source is None:
        raise RuntimeError(f"no stored frame near {reference.timestamp:.3f}s")
    image = cv2.imread(str(source.path))
    if image is None:
        raise RuntimeError("reference frame could not be read")
    crop = _crop(image, reference.bbox)
    if crop.size == 0:
        raise RuntimeError("reference crop was empty")
    dest = data_path("runs", run_id, "locate", f"{object_id}.jpg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), crop):
        raise RuntimeError("reference crop could not be stored")
    selected: list[FrameInput] = [FrameInput(reference.timestamp, dest)]
    search_times: list[float] = []
    for timestamp in times:
        frame = _frame_at(stills, timestamp)
        if frame is None:
            continue
        selected.append(FrameInput(frame.timestamp, frame.path))
        search_times.append(frame.timestamp)
    if len(selected) < 2:
        return []
    label = next((obj.label for obj in document.objects if obj.local_id == object_id), object_id)
    listed = ", ".join(f"{timestamp:.3f}s" for timestamp in search_times)
    user = (
        f"The part is {object_id} ({label}). "
        f"The first image is the reference crop from {reference.timestamp:.0f}s, not a frame to annotate. "
        f"Return a tight box for this part on the full frames at {listed}, or absent when it is fully hidden."
    )
    async with gate:
        config = run.config_json or {}
        model_id, requested = model_for_call(session, run)
        request = GenerationRequest(
            model_id=model_id,
            system_prompt=LOCATE_PART,
            user_prompt=user,
            frames=selected,
            modality="sampled_frames",
            json_schema=_SCHEMA,
            temperature=config.get("temperature"),
            reasoning_effort=config.get("reasoning_effort"),
            image_detail=config.get("image_detail") or "high",
            timeout_s=float(config.get("timeout_s") or 180),
            metadata={"transport": transport} if run.provider_id in {"openai", "anthropic", "cursor"} else {},
        )
        row = open_provider_call(
            session,
            run,
            stage="locate",
            provider_id=run.provider_id,
            model_id=model_id,
            modality="sampled_frames",
            prompt_version_id=run.prompt_version_id,
            parameters={
                "object_id": object_id,
                "reference_timestamp": reference.timestamp,
                **({"requested_model_id": requested} if requested != model_id else {}),
            },
            frame_timestamps=search_times,
        )
        session.commit()
    try:
        response = await provider.generate(secret, request)
    except Exception as exc:
        async with gate:
            fail_provider_call(row, exc)
            session.commit()
        raise
    async with gate:
        finish_provider_call(session, run, row, response)
        session.commit()
    if response.incomplete_reason:
        raise ParseFailure(f"locate call stopped before finishing ({response.incomplete_reason})", response.text)
    sizes = {
        round(float(frame.timestamp), 3): (int(frame.width), int(frame.height))
        for frame in stills
        if frame.width and frame.height
    }
    found = _parse(response.text, set(search_times), sizes)
    return _on_the_same_surface(crop, stills, found)


def _on_the_same_surface(
    reference, frames: list[_Still], found: list[tuple[float, BBox | None, str, float | None]]
):
    """Drop a follow-up box whose pixels are a different surface from the reference crop."""
    kept = []
    for timestamp, bbox, visibility, confidence in found:
        if bbox is None:
            kept.append((timestamp, bbox, visibility, None))
            continue
        frame = _frame_at(frames, timestamp)
        image = cv2.imread(str(frame.path)) if frame is not None else None
        candidate = _crop(image, bbox, pad=0.0) if image is not None else None
        if candidate is not None and candidate.size and same_surface(reference, candidate):
            kept.append((timestamp, bbox, visibility, confidence))
        else:
            # The refused box and score ride along so the refusal can be recorded.
            kept.append((timestamp, None, visibility, None, bbox, confidence))
    return kept


def _fraction_box(raw: dict, width: int, height: int) -> BBox | None:
    """Accept 0–1 fractions or pixel coordinates from the frame that was sent."""
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        w = float(raw["w"])
        h = float(raw["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if max(x, y, w, h) > 1.0 and width > 0 and height > 0:
        x, w = x / width, w / width
        y, h = y / height, h / height
    if w <= 0 or h <= 0:
        return None
    x = min(max(x, 0.0), 0.999)
    y = min(max(y, 0.0), 0.999)
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    if w <= 0 or h <= 0:
        return None
    try:
        return BBox(x=x, y=y, w=w, h=h)
    except ValidationError:
        return None


def _parse(
    text: str, allowed: set[float], sizes: dict[float, tuple[int, int]] | None = None
) -> list[tuple[float, BBox | None, str, float | None]]:
    try:
        payload = extract_json_object(text)
    except ParseFailure:
        return []
    rows = payload.get("observations") or []
    if not isinstance(rows, list):
        return []
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            timestamp = float(row.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if not any(abs(timestamp - item) < 0.05 for item in allowed):
            continue
        raw = row.get("bbox")
        bbox = None
        if isinstance(raw, dict):
            width, height = _frame_size(sizes or {}, timestamp)
            bbox = _fraction_box(raw, width, height)
        visibility = str(row.get("visibility") or "visible")
        found.append((timestamp, bbox, visibility, None if bbox is None else _score(row.get("confidence"))))
    return found


def _score(raw) -> float | None:
    """A model's 0–1 judgment. Anything else is left unset."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score != score:
        return None
    return score


def _frame_size(sizes: dict[float, tuple[int, int]], timestamp: float) -> tuple[int, int]:
    if not sizes:
        return (0, 0)
    nearest = min(sizes, key=lambda item: abs(item - timestamp))
    if abs(nearest - timestamp) > 0.05:
        return (0, 0)
    return sizes[nearest]


def _frame_at(frames: list[_Still], timestamp: float) -> _Still | None:
    best = None
    best_dt = 0.2
    for frame in frames:
        delta = abs(float(frame.timestamp) - timestamp)
        if delta < best_dt:
            best = frame
            best_dt = delta
    return best


def _stored(frame: FrameAsset) -> Path:
    return data_path(*Path(frame.path).parts)


def _crop(image, bbox: BBox, pad: float = 0.03):
    height, width = image.shape[:2]
    x0 = max(0, int((bbox.x - pad) * width))
    y0 = max(0, int((bbox.y - pad) * height))
    x1 = min(width, int((bbox.x + bbox.w + pad) * width))
    y1 = min(height, int((bbox.y + bbox.h + pad) * height))
    return image[y0:y1, x0:x1]
