"""Cross-clip identity proposals. Similar appearance is not an instance match."""

from __future__ import annotations

import re

from app.schemas.enums import OBJECT_COLORS

PROCESS_WORDS = (
    "dry",
    "drying",
    "dried",
    "cure",
    "curing",
    "cured",
    "cool",
    "cooling",
    "ferment",
    "fermentation",
    "fire",
    "fired",
    "firing",
    "glaze",
    "glazed",
)


def propose_links(clips: list[dict], appearance: dict | None = None) -> list[dict]:
    """Each clip is {video_id, run_id, objects:[{local_id, label, characteristics, initial_state, final_state, image_url}]}."""
    proposals: list[dict] = []
    groups: list[list[dict]] = []
    flat = []
    for clip in clips:
        for obj in clip["objects"]:
            flat.append({**obj, "video_id": clip["video_id"], "run_id": clip.get("run_id", "")})
    for item in flat:
        rivals = [
            other
            for other in flat
            if other["video_id"] == item["video_id"] and other["local_id"] != item["local_id"] and _overlap(item, other) >= 0.5
        ]
        placed = False
        for group in groups:
            if any(member["video_id"] == item["video_id"] for member in group):
                continue
            if rivals:
                continue
            anchor = group[0]
            score = _overlap(anchor, item)
            if score >= 0.5 and _feature_overlap(anchor, item) >= 0.34:
                group.append({**item, "score": score})
                placed = True
                break
        if not placed:
            groups.append([{**item, "score": 1.0}])
    for index, group in enumerate(groups):
        videos = {item["video_id"] for item in group}
        if len(videos) < 2:
            continue
        appearance_shift = len({(item.get("initial_state"), item.get("final_state")) for item in group}) > 1
        relation = _relation(group, appearance)
        proposals.append(
            {
                "local_id": f"Object_{index + 1}",
                "label": group[0].get("label") or group[0]["local_id"],
                "uncertain": True,
                "relation": relation,
                "confirmed": False,
                "note": (
                    "Proposed from shared description across clips. Appearance differs between recordings, so this is not a confirmed instance."
                    if appearance_shift
                    else "Proposed from shared description. Similar objects are not assumed to be one physical instance until a person confirms."
                ),
                "color": OBJECT_COLORS[index % len(OBJECT_COLORS)],
                "members": group,
            }
        )
    return proposals


def observed_transition(before: dict, after: dict) -> dict:
    text = f"{before.get('final_state', '')} {after.get('initial_state', '')}".lower()
    words = [word for word in PROCESS_WORDS if re.search(rf"\b{re.escape(word)}\b", text)]
    hypothesis = ""
    if words:
        hypothesis = (
            "Hypothesis, not an observation: the endpoint descriptions mention "
            + ", ".join(words)
            + ". The interval between the recordings was not filmed."
        )
    changed = (before.get("final_state") or "").strip() != (after.get("initial_state") or "").strip()
    observed = (
        f"Earlier recording ends with: {before.get('final_state') or 'undescribed'}. "
        f"Later recording starts with: {after.get('initial_state') or 'undescribed'}."
    )
    return {
        "observed_change": observed,
        "hypothesis": hypothesis,
        "from_state": before.get("final_state") or "",
        "to_state": after.get("initial_state") or "",
        "evidence_class": "INFERRED" if changed else "UNKNOWN",
        "provenance": "DETERMINISTIC",
        "verified": False,
        "gap_note": "No footage covers the time between these recordings. The change is not marked directly observed.",
    }


def _tokens(text: str) -> set[str]:
    return {word for word in (text or "").lower().replace(",", " ").split() if len(word) > 2}


def _overlap(left: dict, right: dict) -> float:
    a = _tokens(f"{left.get('label', '')} {left.get('characteristics', '')}")
    b = _tokens(f"{right.get('label', '')} {right.get('characteristics', '')}")
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _relation(group: list[dict], appearance: dict | None) -> str:
    """Text overlap proposes a type. A cosine does not confirm the instance."""
    cosines = [item.get("appearance_cosine") for item in group if item.get("appearance_cosine") is not None]
    if appearance:
        for left in group:
            for right in group:
                if left["video_id"] == right["video_id"]:
                    continue
                key = (left["video_id"], left["local_id"], right["video_id"], right["local_id"])
                other = (right["video_id"], right["local_id"], left["video_id"], left["local_id"])
                value = appearance.get(key, appearance.get(other))
                if value is not None:
                    cosines.append(value)
    if cosines and max(cosines) >= 0.8:
        return "possibly_same_instance"
    if cosines and max(cosines) >= 0.6:
        return "visually_similar"
    return "same_semantic_type"


def _feature_overlap(left: dict, right: dict) -> float:
    a = _tokens(left.get("characteristics", ""))
    b = _tokens(right.get("characteristics", ""))
    if not a or not b:
        return _overlap(left, right)
    return len(a & b) / len(a | b)
