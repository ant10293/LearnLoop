"""Replace video-file boxes with boxes measured on the displayed frames.

A video answer can report the right width and height and still place the
corner off the part. The extracted frames are the picture the player shows,
so a follow-up call on those frames is what puts the boxes on the parts.
The follow-up asks only for boxes, a few frames at a time. A rescale of the
video numbers does not.
"""

from __future__ import annotations

import copy
import json

from app.perception.decisions import box_decision, with_decisions
from app.pipeline.parsing import extract_json_object, parse_analysis
from app.schemas.analysis import AnalysisDocument, ObservationPrediction
from app.schemas.enums import EvidenceClass, Visibility

# A phone frame is about 1.78. Anything under this can honestly have square boxes.
_MIN_ASPECT = 1.35
_MIN_BOXES = 6
_NEAR_ONE = 0.5


def square_normalized(observations: list[ObservationPrediction], width: int, height: int) -> bool:
    """True when most boxes are square in normalized units on a non-square frame."""
    if width <= 0 or height <= 0:
        return False
    aspect = max(width, height) / min(width, height)
    if aspect < _MIN_ASPECT:
        return False
    ratios = [item.bbox.w / item.bbox.h for item in observations if item.bbox is not None and item.bbox.h > 0]
    if len(ratios) < _MIN_BOXES:
        return False
    ratios.sort()
    median = ratios[len(ratios) // 2]
    near = sum(1 for ratio in ratios if 0.9 <= ratio <= 1.1) / len(ratios)
    return 0.85 <= median <= 1.15 and near >= _NEAR_ONE


# A few stills per call. Sixteen portrait frames in one request came back as
# track points and boxes on the hands.
FRAME_BOX_WINDOW = 4
# A window can come back empty once and answer normally the next time.
FRAME_BOX_ATTEMPTS = 2
# The authority of a video-file box that no frame answer replaced.
UNMEASURED_AUTHORITY = "vlm_video_unmeasured"

FRAME_BOX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object_local_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "visibility": {"type": "string"},
                    "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
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
                },
                "required": ["object_local_id", "timestamp", "visibility", "bbox", "confidence"],
            },
        }
    },
    "required": ["observations"],
}


def frame_box_schema(object_ids: list[str]) -> dict:
    """The contract for one call. Only the ids on its roster can be written.

    Without this a model can answer with 1, 2, 3 or Obj1, and that whole window is
    unusable and keeps the video boxes it was meant to replace.
    """
    schema = copy.deepcopy(FRAME_BOX_SCHEMA)
    fields = schema["properties"]["observations"]["items"]["properties"]
    fields["object_local_id"] = {"type": "string", "enum": sorted(object_ids)}
    fields["visibility"] = {"type": "string", "enum": [item.value for item in Visibility]}
    return schema


def frame_boxes_usable(observations: list[ObservationPrediction], width: int, height: int) -> bool:
    """A follow-up answer is usable when its boxes are not still on a square scale."""
    ratios = [item.bbox.w / item.bbox.h for item in observations if item.bbox is not None and item.bbox.h > 0]
    if not ratios or square_normalized(observations, width, height):
        return False
    if width <= 0 or height <= 0:
        return True
    if max(width, height) / min(width, height) < _MIN_ASPECT:
        return True
    ratios.sort()
    median = ratios[len(ratios) // 2]
    return not 0.85 <= median <= 1.15


def frame_box_rejection(
    observations: list[ObservationPrediction],
    width: int,
    height: int,
    roster: set[str],
) -> str:
    """Why a follow-up answer must not replace the video boxes. Empty when it may."""
    if not any(item.bbox is not None for item in observations):
        if observations and all(_visibility(item) == "absent" for item in observations):
            return "the answer called every part absent"
        return "the answer contained no boxes"
    if not frame_boxes_usable(observations, width, height):
        return "boxes were not in displayed-frame coordinates"
    named = [item for item in observations if item.bbox is not None and item.object_local_id in roster]
    if len({item.object_local_id for item in named}) < min(2, len(roster)):
        return "boxes did not use the object ids"
    shared: dict[tuple, set[str]] = {}
    for item in named:
        box = item.bbox
        assert box is not None
        key = (
            round(item.timestamp, 3),
            round(box.x, 3),
            round(box.y, 3),
            round(box.w, 3),
            round(box.h, 3),
        )
        shared.setdefault(key, set()).add(item.object_local_id)
    if any(len(ids) >= 3 for ids in shared.values()):
        return "one box was reused for several objects"
    return ""


def parse_frame_boxes(text: str, width: int, height: int) -> AnalysisDocument:
    """Read a box answer, converting pixel corners into fractions of the frame."""
    payload = extract_json_object(text)
    for item in payload.get("observations") or []:
        if isinstance(item, dict):
            _scale_pixel_box(item, width, height)
    return parse_analysis(json.dumps(payload))


def _scale_pixel_box(item: dict, width: int, height: int) -> None:
    box = item.get("bbox") if isinstance(item.get("bbox"), dict) else item
    keys = ("x", "y", "w", "h")
    nums = [box.get(key) for key in keys]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in nums):
        return
    if max(nums) <= 1.5 or width <= 0 or height <= 0:
        return
    item["bbox"] = {
        "x": min(max(nums[0] / width, 0), 1),
        "y": min(max(nums[1] / height, 0), 1),
        "w": min(max(nums[2] / width, 0), 1),
        "h": min(max(nums[3] / height, 0), 1),
    }


def frame_box_prompt(width: int, height: int) -> str:
    ratio = height / width if width else 1
    return (
        "You are placing boxes on still frames from a recording. "
        f"Each image is {width} pixels wide and {height} pixels tall. "
        "x and w are fractions of that width. y and h are fractions of that height. "
        "The origin is the top-left pixel of the image, and the bottom-right pixel is x=1, y=1. "
        "Do not inset the box, and do not leave a margin for a border that is not in the image. "
        f"A circle on this image has w/h = {ratio:.3f}. "
        "Do not give w and h the same fraction unless the part has that shape in the image. "
        "The box touches the outermost visible pixels of that one part. "
        "Keep the given object ids. On every image, include one observation for each named object. "
        "When any of that part is visible, visibility is visible or partial, bbox is that part alone, "
        "and confidence is your score from 0 to 1 that the box is that part and only that part. "
        "When the part is fully hidden or not in the frame, visibility is absent, bbox is null, and confidence is null. "
        "Do not point its id at a different part, and do not give a score to a guess. "
        "Two parts never share one box. "
        "Do not rename the objects."
    )


def merge_frame_boxes(document: AnalysisDocument, fresh: AnalysisDocument) -> AnalysisDocument:
    """Copy boxes from a frame answer onto the matching object and timestamp.

    An absent answer clears the box. The score has to come from that same answer,
    because the video score was about a different corner. A moment the video
    answer left out is added, because this answer looked at that frame.
    """
    incoming: dict[tuple[str, float], ObservationPrediction] = {}
    for item in fresh.observations:
        if item.bbox is None and _visibility(item) != "absent":
            continue
        incoming[(item.object_local_id, round(item.timestamp, 3))] = item
    if not incoming:
        return document
    updated = []
    cleared = []
    matched = set()
    for item in document.observations:
        key = (item.object_local_id, round(item.timestamp, 3))
        match = incoming.get(key)
        matched.add(key)
        if match is None:
            updated.append(item)
            continue
        if match.bbox is None:
            if item.bbox is not None:
                cleared.append(
                    box_decision(
                        "frame_boxes",
                        "frame_answer_absent",
                        item,
                        detail="The answer measured on this frame called the part absent.",
                    )
                )
            updated.append(
                item.model_copy(update={"bbox": None, "visibility": match.visibility, "confidence": None})
            )
            continue
        updated.append(
            item.model_copy(
                update={
                    "bbox": match.bbox,
                    "visibility": match.visibility,
                    "confidence": match.confidence,
                }
            )
        )
    known = {obj.local_id for obj in document.objects}
    updated.extend(
        item.model_copy(
            update={
                "evidence": item.evidence.model_copy(
                    update={
                        "classification": EvidenceClass.DIRECTLY_OBSERVED,
                        "frame_timestamps": [item.timestamp],
                    }
                )
            }
        )
        for key, item in incoming.items()
        if key not in matched and item.object_local_id in known
    )
    return drop_copied_boxes(with_decisions(document.model_copy(update={"observations": updated}), cleared))


def answered_keys(fresh: AnalysisDocument) -> set[tuple[str, float]]:
    """The object and moment pairs a frame answer settled, with a box or by calling the part absent."""
    return {
        (item.object_local_id, round(item.timestamp, 3))
        for item in fresh.observations
        if item.bbox is not None or _visibility(item) == "absent"
    }


def mark_unmeasured(
    document: AnalysisDocument,
    roster: set[str],
    measured: set[tuple[str, float]],
    failed: list[tuple[float, float, str]],
) -> AnalysisDocument:
    """Tag each video-file box no frame answer replaced, so it cannot pass as measured.

    `failed` holds the start, end, and reason of each refused window.
    """
    updated = []
    kept = []
    for item in document.observations:
        key = (item.object_local_id, round(item.timestamp, 3))
        if item.bbox is None or item.object_local_id not in roster or key in measured:
            updated.append(item)
            continue
        reason = next((why for start, end, why in failed if start - 1e-3 <= item.timestamp <= end + 1e-3), "")
        kept.append(
            box_decision(
                "frame_boxes",
                "frame_box_unmeasured",
                item,
                outcome="kept",
                detail=f"This video-file box was kept unmeasured: {reason or 'no frame answer settled this moment'}.",
            )
        )
        updated.append(item.model_copy(update={"geometry_authority": UNMEASURED_AUTHORITY}))
    return with_decisions(document.model_copy(update={"observations": updated}), kept)


def drop_copied_boxes(document: AnalysisDocument) -> AnalysisDocument:
    """Clear the lower-scoring box when two ids were given the same rectangle."""
    grouped: dict[float, list[int]] = {}
    observations = list(document.observations)
    for index, item in enumerate(observations):
        if item.bbox is None:
            continue
        grouped.setdefault(round(item.timestamp, 3), []).append(index)
    clear: dict[int, int] = {}
    for indexes in grouped.values():
        ranked = sorted(indexes, key=lambda index: (observations[index].confidence or 0), reverse=True)
        kept: list[int] = []
        for index in ranked:
            box = observations[index].bbox
            assert box is not None
            winner = next((other for other in kept if _iou(box, observations[other].bbox) >= 0.9), None)
            if winner is not None:
                clear[index] = winner
                continue
            kept.append(index)
    if not clear:
        return document
    from app.schemas.enums import Visibility

    updated = []
    for index, item in enumerate(observations):
        if index not in clear:
            updated.append(item)
            continue
        updated.append(item.model_copy(update={"bbox": None, "visibility": Visibility.ABSENT, "confidence": None}))
    decisions = [
        box_decision(
            "frame_boxes",
            "duplicate_copied_box",
            observations[index],
            other_object_id=observations[winner].object_local_id,
            detail=f"The same rectangle was given to {observations[winner].object_local_id}, which scored higher or came first.",
        )
        for index, winner in clear.items()
    ]
    return with_decisions(document.model_copy(update={"observations": updated}), decisions)


def _visibility(item: ObservationPrediction) -> str:
    value = item.visibility
    return str(getattr(value, "value", value))


def _iou(left, right) -> float:
    if left is None or right is None:
        return 0
    ix = max(0, min(left.x + left.w, right.x + right.w) - max(left.x, right.x))
    iy = max(0, min(left.y + left.h, right.y + right.h) - max(left.y, right.y))
    inter = ix * iy
    union = left.w * left.h + right.w * right.h - inter
    return inter / union if union else 0
