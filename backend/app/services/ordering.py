"""Partial clip order. Inferred context order is a hypothesis, not a sequence of record."""

from __future__ import annotations


def metadata_order(clips: list[dict]) -> dict:
    """clips need id, capture_time_utc, timestamp_reliability, order_locked, order_index."""
    locked = [clip for clip in clips if clip.get("order_locked")]
    if locked and len(locked) == len(clips):
        ranked = [[clip["id"]] for clip in sorted(locked, key=lambda item: item.get("order_index") or 0)]
        return _result("manual", ranked, [], ["Manual order is locked."], fact=False)
    reliable = [
        clip
        for clip in clips
        if clip.get("capture_time_utc") and clip.get("timestamp_reliability") == "reliable" and not clip.get("order_locked")
    ]
    reliable.sort(key=lambda item: item["capture_time_utc"])
    groups: list[list[str]] = []
    ambiguous: list[str] = []
    notes = []
    index = 0
    while index < len(reliable):
        tied = [reliable[index]]
        while index + 1 < len(reliable) and reliable[index + 1]["capture_time_utc"] == reliable[index]["capture_time_utc"]:
            index += 1
            tied.append(reliable[index])
        groups.append([item["id"] for item in tied])
        if len(tied) > 1:
            ambiguous.extend(item["id"] for item in tied)
            notes.append("Identical capture timestamps do not establish an order inside the tie.")
        index += 1
    for clip in clips:
        if clip.get("order_locked"):
            continue
        if clip["id"] in {item for group in groups for item in group}:
            continue
        ambiguous.append(clip["id"])
        reason = clip.get("timestamp_reliability") or "absent"
        label = clip.get("filename") or clip["id"]
        notes.append(f"{label} is unordered: timestamp reliability is {reason}.")
    if locked:
        head = [[clip["id"]] for clip in sorted(locked, key=lambda item: item.get("order_index") or 0)]
        groups = head + groups
        notes.append("Locked clips stay where the researcher placed them.")
    return _result("metadata", groups, ambiguous, notes, fact=False)


def context_hypothesis(clips: list[dict], objects_by_video: dict[str, list[dict]]) -> dict:
    """Suggest A before B when A's ending state matches B's starting state. Not a fact."""
    evidence = []
    pairs: list[tuple[str, str]] = []
    ids = [clip["id"] for clip in clips]
    for left in ids:
        for right in ids:
            if left == right:
                continue
            support = _state_bridge(objects_by_video.get(left, []), objects_by_video.get(right, []))
            if support:
                pairs.append([left, right])
                evidence.append({"before": left, "after": right, "support": support, "evidence_class": "INFERRED"})
    return {
        "method": "context",
        "fact": False,
        "pairs": pairs,
        "evidence": evidence,
        "note": "Visual context can suggest a relative order. It does not establish one.",
    }


def apply_context_if_unlocked(meta: dict, hypothesis: dict) -> dict:
    """Context never reorders a metadata or manual placement. It only fills a fully unordered set."""
    placed = [item for group in meta["ranked"] for item in group]
    if placed or meta["method"] == "manual":
        return meta
    if not hypothesis["pairs"]:
        return meta
    return meta


def _state_bridge(earlier: list[dict], later: list[dict]) -> str:
    for left in earlier:
        for right in later:
            end = _tokens(left.get("final_state", ""))
            start = _tokens(right.get("initial_state", ""))
            if end and start and len(end & start) >= 2 and _tokens(left.get("label", "")) & _tokens(right.get("label", "")):
                return f"{left.get('label')} ends '{left.get('final_state')}' and {right.get('label')} starts '{right.get('initial_state')}'."
    return ""


def _tokens(text: str) -> set[str]:
    return {word for word in text.lower().replace(",", " ").split() if len(word) > 2}


def _result(method: str, ranked: list[list[str]], ambiguous: list[str], notes: list[str], fact: bool) -> dict:
    return {
        "method": method,
        "fact": fact,
        "ranked": ranked,
        "unordered": [item for item in ambiguous if item not in {value for group in ranked for value in group}],
        "ambiguous_ids": list(dict.fromkeys(ambiguous)),
        "notes": notes,
    }
