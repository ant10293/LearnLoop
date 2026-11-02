"""What each SAM request returned, and where its boxes went. Reads stored responses and sends nothing.

A returned box shows that SAM produced output. It does not show that the box is on the right object.
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.recovery import RecoveryExperiment, RecoveryProposal, RecoveryRequest
from app.perception.sam_fusion import associate_tracks
from app.providers.sam import (
    FRAME_RE,
    MAX_TRACKED_OBJECTS,
    SamRecord,
    VideoChunk,
    frame_timestamp,
    normalized_box,
    parse_sam_output,
)
from app.services.recovery import _document, _spec_samples, _usable_spatial, load_response

SAMPLE_CHARS = 120
MAX_SAMPLES = 5


def scan_output(text: str) -> tuple[list[SamRecord], dict]:
    """Parse line by line with the pipeline's parser, counting everything it passes over."""
    records: list[SamRecord] = []
    frame_lines = 0
    empty_frames = 0
    box_tokens = 0
    dropped_tokens = 0
    unparsed_lines = 0
    samples: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = FRAME_RE.match(line)
        if not match:
            unparsed_lines += 1
            if len(samples) < MAX_SAMPLES:
                samples.append(line[:SAMPLE_CHARS])
            continue
        frame_lines += 1
        found = parse_sam_output(line)
        tokens = match.group("body").count("<|box;")
        box_tokens += tokens
        records.extend(found)
        if not tokens:
            empty_frames += 1
        if tokens > len(found) or (not tokens and match.group("body").strip()):
            dropped_tokens += max(tokens - len(found), 0)
            if len(samples) < MAX_SAMPLES:
                samples.append(line[:SAMPLE_CHARS])
    return records, {
        "text_chars": len(text),
        "frame_lines": frame_lines,
        "empty_frame_lines": empty_frames,
        "box_tokens": box_tokens,
        "dropped_tokens": dropped_tokens,
        "unparsed_lines": unparsed_lines,
        "unparsed_samples": samples,
    }


def raw_boxes(records: list[SamRecord], spec: dict, statuses: dict[str, str]) -> list[dict]:
    """Every parsed box, with why it was or was not usable. Masks stay in the stored response."""
    times = spec["frame_times"]
    chunk = VideoChunk(0, 0, len(times), times[0], times[-1], spec["session_id"], times)
    bindings = spec["bindings"]
    allowed: list[str] = []
    boxes = []
    for record in records:
        track = record.track_id
        if track not in allowed and len(allowed) < MAX_TRACKED_OBJECTS:
            allowed.append(track)
        timestamp = frame_timestamp(record.frame_index, 0.0, chunk)
        box = normalized_box(record)
        if timestamp is None:
            status = "frame_out_of_range"
        elif box is None:
            status = "invalid_box"
        elif track not in allowed:
            status = "over_track_limit"
        elif track not in bindings:
            status = "unassigned"
        else:
            status = statuses.get(track, "associated")
        boxes.append(
            {
                "request_key": spec["key"],
                "raw_track_id": track,
                "object_local_id": bindings.get(track, ""),
                "frame_index": record.frame_index,
                "timestamp": timestamp,
                "bbox": box,
                "pixels": [record.x1, record.y1, record.x2, record.y2],
                "confidence": record.confidence,
                "has_mask": bool(record.mask_data or record.mask_payload),
                "status": status,
            }
        )
    return boxes


def _expected_frames(spec: dict) -> range:
    """Frames the prompt could reach. Propagation starts at the earliest prompted frame."""
    count = len(spec["frame_times"])
    first = min((item["seed_frame_index"] for item in spec["objects"]), default=0)
    if spec["direction"] == "forward":
        return range(first, count)
    if spec["direction"] == "backward":
        return range(0, first + 1)
    return range(count)


def _missing_spans(spec: dict, seen: set[int]) -> list[dict]:
    times = spec["frame_times"]
    spans: list[dict] = []
    for frame in _expected_frames(spec):
        if frame in seen:
            continue
        if spans and spans[-1]["last_frame"] == frame - 1:
            spans[-1]["last_frame"] = frame
            spans[-1]["end"] = times[frame]
            spans[-1]["frames"] += 1
        else:
            spans.append({"first_frame": frame, "last_frame": frame, "start": times[frame], "end": times[frame], "frames": 1})
    return spans


def _tracks(boxes: list[dict], spec: dict, associations: dict, rows: list[dict]) -> list[dict]:
    prompted = {str(item["sam_object_id"]) for item in spec["objects"]}
    tracks = []
    for track in sorted({*prompted, *(box["raw_track_id"] for box in boxes)}, key=int):
        owned = sorted((box for box in boxes if box["raw_track_id"] == track), key=lambda item: item["frame_index"])
        scores = [box["confidence"] for box in owned if box["confidence"] is not None]
        link = associations.get(track)
        placed = [box["timestamp"] for box in owned if box["timestamp"] is not None]
        checks = sorted((row for row in rows if row.get("raw_track_id") == track), key=lambda row: row["timestamp"])
        broke = next(
            (
                {"timestamp": row["timestamp"], "detail": check["detail"]}
                for row in checks
                for check in row["checks"]
                if check["check"] == "track_intact" and not check["passed"]
            ),
            None,
        )
        tracks.append(
            {
                "raw_track_id": track,
                "prompted": track in prompted,
                "object_local_id": spec["bindings"].get(track, ""),
                "association": link.status if link else ("no_rows" if not owned else "unassigned"),
                "association_note": link.note if link else "",
                "boxes": len(owned),
                "first_time": min(placed) if placed else None,
                "last_time": max(placed) if placed else None,
                "first_confidence": scores[0] if scores else None,
                "mean_confidence": round(sum(scores) / len(scores), 4) if scores else None,
                "last_confidence": scores[-1] if scores else None,
                "broke": broke,
            }
        )
    return tracks


def _findings(parsed: dict, funnel: dict, tracks: list[dict], missing: list[dict]) -> list[dict]:
    found: list[dict] = []

    def add(code: str, severity: str, message: str) -> None:
        found.append({"code": code, "severity": severity, "message": message})

    if parsed["unparsed_lines"]:
        add("unparsed_lines", "warning", f"{parsed['unparsed_lines']} lines were not frame lines and were skipped.")
    if parsed["dropped_tokens"]:
        add("dropped_tokens", "warning", f"{parsed['dropped_tokens']} box tokens could not be parsed, such as a box with no mask.")
    if funnel["frame_out_of_range"]:
        add("frame_out_of_range", "warning", f"{funnel['frame_out_of_range']} boxes name a frame outside this clip, so they have no time.")
    if funnel["invalid_box"]:
        add("invalid_box", "warning", f"{funnel['invalid_box']} boxes have unusable coordinates.")
    if funnel["over_track_limit"]:
        add("over_track_limit", "warning", f"{funnel['over_track_limit']} boxes belong to tracks past the limit of {MAX_TRACKED_OBJECTS}.")
    for track in tracks:
        name = f"SAM track {track['raw_track_id']}"
        if track["prompted"] and not track["boxes"]:
            add("prompted_track_missing", "error", f"{name} was prompted and returned no boxes.")
        elif not track["prompted"]:
            add("unprompted_track", "warning", f"{name} was not prompted, so its {track['boxes']} boxes are unassigned and were not validated.")
        elif track["association"] in {"conflict", "ambiguous"}:
            add("association_loss", "error", f"{name} has association {track['association']}. {track['association_note']}".strip())
        if track["broke"]:
            add("track_broke", "warning", f"{name} broke at {track['broke']['timestamp']:.2f}s. {track['broke']['detail']}")
    if missing:
        frames = sum(item["frames"] for item in missing)
        add("missing_frames", "warning", f"{frames} reachable frames have no usable box.")
    return found


def _verdict(spec: dict, payload: dict | None, row: RecoveryRequest | None, parsed: dict, funnel: dict) -> tuple[str, str]:
    if payload is None:
        if row is not None and row.error:
            return "request_failed", f"No response is stored. {row.error}"
        if row is not None and row.submission in {"submitted", "uploaded"}:
            return "no_response", "The request was sent and no response is stored."
        return "not_run", "This planned request has not run."
    if not parsed["text_chars"]:
        return "empty_output", "SAM returned an empty response."
    if not parsed["box_tokens"]:
        if parsed["frame_lines"]:
            return "no_boxes", "SAM returned frame lines with no boxes, so it reported no detection."
        return "parse_failure", "The response has no frame lines the parser recognizes."
    if not funnel["parsed"]:
        return "parse_failure", "SAM returned box tokens and none could be parsed."
    if not funnel["usable"]:
        return "no_usable_boxes", "Boxes were parsed and none could be placed on a frame."
    if spec.get("prompt_kind") == "concept":
        return "concept_candidates", "SAM found possible locations. A person has to say which physical part each one is."
    if not funnel["assigned"]:
        return "no_assignment", "SAM returned usable boxes and none belong to a prompted object."
    return "boxes_returned", "SAM returned usable boxes for a prompted object. This does not show they are correct."


def _with_validation(boxes: list[dict], rows: list[dict]) -> None:
    """Say what validation decided, so a wrong detection can be told from a rejected correct one."""
    decided: dict[tuple, dict] = {}
    for row in rows:
        decided[(row.get("raw_track_id"), round(row["timestamp"], 3))] = row
    for box in boxes:
        row = decided.get((box["raw_track_id"], round(box["timestamp"], 3))) if box["timestamp"] is not None else None
        box["outcome"] = row["outcome"] if row else ""
        box["lifecycle"] = row.get("lifecycle", "") if row else ""
        box["segment_id"] = row.get("segment_id", "") if row else ""


def _diagnose(spec: dict, payload: dict | None, row: RecoveryRequest | None, document, spatial, rows: list[dict]) -> dict:
    records, parsed = scan_output((payload or {}).get("text") or "")
    associations: dict = {}
    if payload is not None:
        samples = _spec_samples(spec, payload)
        associations = {
            item.raw_track_id: item
            for item in associate_tracks(document, samples, spatial=spatial, measured_bindings=True)
        }
    statuses = {track: link.status for track, link in associations.items()}
    boxes = raw_boxes(records, spec, statuses)
    _with_validation(boxes, rows)
    by_status = Counter(box["status"] for box in boxes)
    unusable = by_status["frame_out_of_range"] + by_status["invalid_box"] + by_status["over_track_limit"]
    usable = len(boxes) - unusable
    outcomes = Counter(item["outcome"] for item in rows)
    funnel = {
        "box_tokens": parsed["box_tokens"],
        "parsed": len(boxes),
        "frame_out_of_range": by_status["frame_out_of_range"],
        "invalid_box": by_status["invalid_box"],
        "over_track_limit": by_status["over_track_limit"],
        "usable": usable,
        "unassigned": by_status["unassigned"],
        "assigned": usable - by_status["unassigned"],
        "supported": outcomes["supported"],
        "provisional": outcomes["provisional"],
        "conflicting": outcomes["conflicting"],
        "rejected": outcomes["rejected"],
    }
    seen = {box["frame_index"] for box in boxes if box["timestamp"] is not None and box["bbox"] is not None}
    missing = _missing_spans(spec, seen) if payload is not None else []
    tracks = _tracks(boxes, spec, associations, rows) if payload is not None else []
    verdict, message = _verdict(spec, payload, row, parsed, funnel)
    failed = Counter(
        check["check"] for item in rows for check in item["checks"] if not check["passed"]
    )
    return {
        "key": spec["key"],
        "verdict": verdict,
        "message": message,
        "boxes_returned": bool(boxes),
        "synthetic": bool((payload or {}).get("synthetic")),
        "request": {
            "status": row.status if row else "",
            "submission": row.submission if row else "",
            "source": row.source if row else "",
            "provider_request_id": (payload or {}).get("request_id", "") or (row.provider_request_id if row else ""),
            "latency_ms": int((payload or {}).get("latency_ms") or 0),
            "error": row.error if row else "",
        },
        "prompt": spec["prompt"],
        "direction": spec["direction"],
        "planned_frames": len(spec["frame_times"]),
        "reachable_frames": len(_expected_frames(spec)),
        "frames_with_boxes": len(seen),
        "parsed": parsed,
        "funnel": funnel,
        "tracks": tracks,
        "missing_frames": missing,
        "failed_checks": dict(failed.most_common()),
        "findings": _findings(parsed, funnel, tracks, missing) if payload is not None else [],
        "boxes": boxes,
    }


def diagnostics(session: Session, experiment: RecoveryExperiment) -> dict:
    specs = (experiment.plan_json or {}).get("requests") or []
    requests = {
        row.cache_key: row
        for row in session.scalars(select(RecoveryRequest).where(RecoveryRequest.experiment_id == experiment.id)).all()
    }
    proposals = session.scalars(
        select(RecoveryProposal).where(
            RecoveryProposal.experiment_id == experiment.id,
            RecoveryProposal.generation == experiment.generation,
        )
    ).all()
    rows_by_key: dict[str, list[dict]] = {}
    for proposal in proposals:
        for row in proposal.rows_json or []:
            rows_by_key.setdefault(row.get("request_key", ""), []).append(row)
    document = _document(session, experiment.source_revision_id)
    spatial = _usable_spatial(experiment.run_id)
    return {
        "experiment_id": experiment.id,
        "generation": experiment.generation,
        "note": "A returned box shows SAM produced output. It does not show the box is on the right object.",
        "requests": [
            _diagnose(
                spec,
                load_response(experiment.run_id, spec["key"]),
                requests.get(spec["key"]),
                document,
                spatial,
                rows_by_key.get(spec["key"], []),
            )
            for spec in specs
        ],
    }
