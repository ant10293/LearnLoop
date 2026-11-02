"""Capture timestamps from container metadata. File mtime is never a capture time."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CaptureStamp:
    raw: str
    utc_iso: str
    timezone_name: str
    source: str
    reliability: str
    file_mtime: str
    note: str


def read_capture(path: Path) -> CaptureStamp:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    tags: dict[str, str] = {}
    if proc.returncode == 0:
        try:
            tags = (json.loads(proc.stdout).get("format") or {}).get("tags") or {}
        except json.JSONDecodeError:
            tags = {}
    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return stamp_from_tags(tags, mtime)


def stamp_from_tags(tags: dict, file_mtime: str) -> CaptureStamp:
    quick = _first(tags, ("com.apple.quicktime.creationdate", "creationdate"))
    created = _first(tags, ("creation_time",))
    exif = _first(tags, ("DateTimeOriginal", "datetimeoriginal", "DateTime"))
    chosen = ""
    source = "absent"
    if quick:
        chosen, source = quick, "quicktime"
    elif created:
        chosen, source = created, "format_creation_time"
    elif exif:
        chosen, source = exif, "exif"
    parsed = _parse(chosen) if chosen else None
    conflicts = []
    for label, value in (("quicktime", quick), ("creation_time", created), ("exif", exif)):
        other = _parse(value) if value else None
        if parsed and other and abs((parsed[0] - other[0]).total_seconds()) > 120:
            conflicts.append(label)
    if not chosen:
        return CaptureStamp("", "", "", "absent", "absent", file_mtime, "No capture timestamp in container tags. Upload time and file modification time were not used.")
    if conflicts:
        return CaptureStamp(
            chosen,
            "",
            "",
            source,
            "conflicting",
            file_mtime,
            "Container tags disagree by more than two minutes. No capture order was taken from them.",
        )
    if parsed is None:
        return CaptureStamp(chosen, "", "", source, "weak", file_mtime, "The capture tag was not a recognizable timestamp.")
    moment, tz_name, aware = parsed
    if not aware:
        return CaptureStamp(
            chosen,
            "",
            "",
            source,
            "weak",
            file_mtime,
            "The capture timestamp has no timezone. It was preserved and was not normalized to UTC.",
        )
    return CaptureStamp(
        chosen,
        moment.astimezone(timezone.utc).isoformat(),
        tz_name,
        source,
        "reliable",
        file_mtime,
        "Normalized to UTC from a timestamp that included a timezone.",
    )


def _first(tags: dict, keys: tuple[str, ...]) -> str:
    folded = {str(key).lower(): value for key, value in tags.items()}
    for key in keys:
        value = folded.get(key.lower())
        if value:
            return str(value)
    return ""


def _parse(raw: str) -> tuple[datetime, str, bool] | None:
    text = raw.strip().replace("Z", "+00:00")
    if len(text) >= 19 and text[10] == " ":
        text = text[:10] + "T" + text[11:]
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment, "", False
    return moment, moment.tzinfo.tzname(moment) or "offset", True
