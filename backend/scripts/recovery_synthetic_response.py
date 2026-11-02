"""Write a labeled synthetic SAM response so the recovery UI can be exercised offline.

It follows each seed box unchanged through the planned frames. This checks the
plumbing and says nothing about accuracy. Results built on it are marked synthetic
in the API and the UI. It makes no network call.

    cd backend && .venv/bin/python -m scripts.recovery_synthetic_response <experiment_id>
"""

from __future__ import annotations

import sys

from app.db import SessionLocal
from app.models.recovery import RecoveryExperiment
from app.perception.spatial_prompts import box_mark
from app.services.recovery import load_response, store_response


def synthetic_text(spec: dict) -> str:
    """Frames run from the seed in the planned direction, in descending order for backward."""
    width, height = spec["width"], spec["height"]
    lines = []
    first = min(item["seed_frame_index"] for item in spec["objects"])
    count = len(spec["frame_times"])
    frames = {
        "forward": range(first, count),
        "backward": range(first, -1, -1),
    }.get(spec["direction"], range(count))
    for frame in frames:
        tokens = []
        seen: set[int] = set()
        for item in sorted(spec["objects"], key=lambda entry: abs(entry["seed_frame_index"] - frame)):
            box = item["seed"].get("bbox")
            if not box or item["sam_object_id"] in seen:
                continue
            seen.add(item["sam_object_id"])
            mark = box_mark(box, frame, item["sam_object_id"], width, height)
            tokens.append(
                f"{mark.object_id}<|box;x1={mark.x1};y1={mark.y1};x2={mark.x2};y2={mark.y2};w={width};h={height};c=0.5|>"
                f"<|mask;data=1,1,!A|>"
            )
        if tokens:
            lines.append(f"<{frame}f>" + ",".join(tokens))
    return "\n".join(lines)


def main(experiment_id: str) -> None:
    session = SessionLocal()
    try:
        experiment = session.get(RecoveryExperiment, experiment_id)
        if experiment is None or not experiment.plan_json:
            raise SystemExit("Unknown experiment, or it has no dry-run plan yet.")
        for spec in experiment.plan_json["requests"]:
            if load_response(experiment.run_id, spec["key"]) is not None:
                print(f"{spec['key'][:12]} already stored, left unchanged")
                continue
            store_response(
                experiment.run_id,
                spec["key"],
                {
                    "cache_key": spec["key"],
                    "text": synthetic_text(spec),
                    "request_id": "synthetic",
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model_id": "synthetic",
                    "synthetic": True,
                },
            )
            print(f"{spec['key'][:12]} wrote a synthetic response")
    finally:
        session.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
