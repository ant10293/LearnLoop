"""Associate speech with visual events in one clip. Speech is not evidence of completion."""

from __future__ import annotations

import re

CLASSIFICATIONS = (
    "visually_supported",
    "contextual",
    "contradictory",
    "irrelevant",
    "uncertain",
)

_STOP = {
    "the",
    "and",
    "into",
    "with",
    "that",
    "this",
    "from",
    "then",
    "just",
    "going",
    "because",
    "would",
    "using",
    "used",
    "have",
    "been",
    "they",
    "them",
    "are",
    "was",
    "were",
    "all",
    "for",
    "you",
    "i'm",
    "im",
}
_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_REASON = ("because", "so that", "in order to", "so the", "reason")


def align_segments(segments: list[dict], events: list[dict], *, video_id: str, window_sec: float = 4.0) -> list[dict]:
    """events need video_id, kind, local_id, text, start, end. Other clips are ignored."""
    local = [event for event in events if event.get("video_id") == video_id]
    rows = []
    for segment in segments:
        rows.append(_one(segment, local, window_sec))
    return rows


def _one(segment: dict, events: list[dict], window_sec: float) -> dict:
    speech = _tokens(segment.get("text") or "")
    start = float(segment.get("start") or 0)
    end = float(segment.get("end") or start)
    timed = [event for event in events if event.get("kind") in {"action", "step"}]
    ranked = []
    for event in timed:
        overlap = _overlap(speech, _tokens(event.get("text") or ""))
        if overlap <= 0:
            continue
        near = _near(start, end, float(event.get("start") or 0), float(event.get("end") or 0), window_sec)
        ranked.append((overlap, near, event))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    claim = _claimed_count(segment.get("text") or "")
    observed = _observed_count(claim[1] if claim else "", events) if claim else None
    reason = any(marker in (segment.get("text") or "").lower() for marker in _REASON)
    best = ranked[0] if ranked else None
    actions = [event["local_id"] for _score, near, event in ranked if event["kind"] == "action" and near and _score >= 0.34]
    steps = [event["local_id"] for _score, near, event in ranked if event["kind"] == "step" and near and _score >= 0.34]
    offset = None
    if best and best[2].get("kind") == "action":
        offset = round(start - float(best[2].get("start") or 0), 3)
    noun_known = bool(claim) and any(claim[1] in _tokens(event.get("text") or "") for event in events)
    contradictory = bool(claim) and noun_known and observed is not None and claim[0] != observed
    if contradictory:
        classification = "contradictory"
        explanation = (
            f"The speech mentions {claim[0]} {claim[1]}. The visual record of this clip shows {observed}. "
            "The camera may have missed an action, so this is a discrepancy, not a conclusion that the speaker is wrong. "
            "The spoken count does not complete a workflow step."
        )
    elif reason and not (best and best[1] and best[0] >= 0.5 and not _occluded(best[2])):
        classification = "contextual"
        explanation = (
            "The speech gives a reason or selection that the pictures do not establish. "
            "It is supplemental context, not visual proof."
        )
    elif best and best[1] and best[0] >= 0.34 and not _occluded(best[2]):
        classification = "visually_supported"
        timing = ""
        if offset is not None and offset < -0.4:
            timing = " The speech precedes the visible action."
        elif offset is not None and offset > 0.4:
            timing = " The speech follows the visible action."
        explanation = "The words match a visual event inside the search window." + timing + " The speech does not prove the action."
    elif best and (not best[1] or _occluded(best[2]) or best[0] < 0.34):
        classification = "uncertain"
        why = "outside the time window" if not best[1] else "weak or partly hidden"
        explanation = f"The words are {why}. No confident link was made, including to the nearest frame."
    else:
        classification = "irrelevant"
        explanation = "The words do not match the objects, actions, or steps recorded for this clip."
    frames = []
    if best and best[1]:
        frames = [round(float(best[2].get("start") or start), 3)]
    objects = [
        event["local_id"]
        for event in events
        if event.get("kind") == "object" and _overlap(speech, _tokens(event.get("text") or "")) >= 0.34
    ]
    return {
        "segment_local_id": segment.get("local_id") or "",
        "classification": classification,
        "explanation": explanation,
        "action_local_ids": _unique(actions),
        "object_local_ids": _unique(objects),
        "step_local_ids": _unique(steps),
        "frame_timestamps": frames,
        "time_offset_sec": offset,
        "confidence": round(best[0], 3) if best else 0.0,
        "evidence_class": "INFERRED",
        "speech_is_not_completion": True,
    }


def _claimed_count(text: str) -> tuple[int, str] | None:
    lowered = text.lower()
    match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-z]{3,})\b", lowered)
    if not match:
        return None
    raw, noun = match.group(1), match.group(2)
    count = int(raw) if raw.isdigit() else _NUMBERS[raw]
    return count, noun.rstrip("s")


def _observed_count(noun: str, events: list[dict]) -> int:
    if not noun:
        return 0
    return sum(1 for event in events if event.get("kind") == "action" and noun in _tokens(event.get("text") or ""))


def _near(speech_start: float, speech_end: float, event_start: float, event_end: float, window: float) -> bool:
    return speech_start <= event_end + window and speech_end >= event_start - window


def _occluded(event: dict) -> bool:
    return bool(event.get("occluded"))


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOP}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
