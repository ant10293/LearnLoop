"""Paths under the data directory. User-supplied names never become path segments."""

from __future__ import annotations

import re
from pathlib import Path

from app.config import settings


def ensure_data_dirs() -> Path:
    root = settings.resolved_data_dir
    for name in ("originals", "proxies", "frames", "thumbs", "sheets", "runs", "exports", "objects", "imports"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or "video")[:180]


def data_path(*parts: str) -> Path:
    root = settings.resolved_data_dir.resolve()
    path = root.joinpath(*parts).resolve()
    if path != root and root not in path.parents:
        raise ValueError("refusing a path outside the data directory")
    return path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(settings.resolved_data_dir.resolve()))


EXTERNAL_PREFIX = "external:"


def external_ref(path: Path) -> str:
    return f"{EXTERNAL_PREFIX}{path.expanduser().resolve()}"


def is_external(stored: str) -> bool:
    return stored.startswith(EXTERNAL_PREFIX)


def resolve_stored(stored: str) -> Path:
    if not stored:
        raise ValueError("empty stored path")
    if is_external(stored):
        path = Path(stored.removeprefix(EXTERNAL_PREFIX)).expanduser()
        if not path.is_absolute():
            raise ValueError("referenced path must be absolute")
        return path.resolve()
    return data_path(*Path(stored).parts)
