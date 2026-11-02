"""Task-project archive: workflow, imagery references, timestamps, and uncertainties."""

from __future__ import annotations

import json
import zipfile

from sqlalchemy.orm import Session

from app import SCHEMA_VERSION, __version__
from app.services.storage import data_path
from app.services.tasks import task_bundle


def export_task(session: Session, project_id: str):
    bundle = task_bundle(session, project_id)
    if bundle is None:
        return None
    dest = data_path("exports", f"{project_id}-task.zip")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_extension": "1.1.0",
                    "app_version": __version__,
                    "project_id": project_id,
                    "kind": "task_project",
                    "secrets_included": False,
                },
                indent=2,
            ),
        )
        archive.writestr("task.json", json.dumps(bundle, indent=2, default=str))
    return dest
