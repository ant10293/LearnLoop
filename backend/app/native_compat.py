"""Keep OpenCV and PyAV from registering the same AVFoundation classes."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Same length as the originals so the Mach-O cstring offsets stay put.
# OpenCV only uses these classes for AVFoundation camera capture. File
# decode and PyAV keep the names FFmpeg registered.
_RENAMES = (
    (b"AVFFrameReceiver", b"OcvFrameReceiver"),
    (b"AVFAudioReceiver", b"OcvAudioReceiver"),
)


def isolate_bundled_avfoundation() -> None:
    """Rename ObjC classes inside OpenCV's bundled libavdevice.

    opencv-python-headless and PyAV (pulled in by faster-whisper) each
    ship a libavdevice that implements AVFFrameReceiver and
    AVFAudioReceiver. macOS allows one implementation of a class. The
    duplicate prints the objc warning and can crash a process that
    loads both. Renaming OpenCV's copy leaves PyAV's device classes
    as the ones the runtime knows by the original names.
    """
    if sys.platform != "darwin":
        return
    for path in _opencv_avdevice_dylibs():
        try:
            _rename_classes(path)
        except Exception:
            log.warning("Could not rename AVFoundation classes in %s", path, exc_info=True)


def renamed_avfoundation_classes(data: bytes) -> bytes | None:
    """Return patched library bytes, or None when nothing has to change."""
    if not any(old in data for old, _new in _RENAMES):
        return None
    updated = data
    for old, new in _RENAMES:
        if len(old) != len(new):
            raise ValueError("class name replacement must stay the same length")
        count = updated.count(old)
        if count == 0:
            continue
        if count != 1:
            raise ValueError(f"{old.decode()} appears {count} times")
        updated = updated.replace(old, new)
    if updated == data:
        return None
    return updated


def _opencv_avdevice_dylibs() -> list[Path]:
    found: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        dylibs = Path(entry) / "cv2" / ".dylibs"
        if not dylibs.is_dir():
            continue
        found.extend(sorted(dylibs.glob("libavdevice*.dylib")))
    return found


def _rename_classes(path: Path) -> None:
    updated = renamed_avfoundation_classes(path.read_bytes())
    if updated is None:
        return
    temporary = path.with_name(f".{path.name}.rename")
    temporary.write_bytes(updated)
    try:
        subprocess.run(
            ["codesign", "-s", "-", "--force", os.fspath(temporary)],
            check=True,
            capture_output=True,
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    log.info("Renamed duplicate AVFoundation classes in %s", path)
