"""One evidence-resolution policy for geometry and identity.

Raw outputs from the language model, the tracker, and segmentation are kept.
For each object at each analyzed time, at most one canonical row is resolved
from supported measurements. Detection, association, temporal, and semantic
support are separate fields and are never folded into one score.

`AnalysisDocument.observations` is the display set after resolution. Every
consumer reads that list. `AnalysisDocument.raw_observations` keeps the rows
the display set does not show, for inspection. Resolution is idempotent:
running it again on its own output gives the same partition.
"""

from __future__ import annotations

from collections import defaultdict

from app.perception.assignment import iou
from app.perception.detect import DETECTOR_EVIDENCE
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    Evidence,
    ObservationPrediction,
)
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance

SAM_ID = "sam-3.1"
TRACKER_SOURCE = "tracker"
# A box a person drew. It can be canonical without a detector, and a disagreeing measurement still conflicts.
HUMAN_AUTHORITY = "human"
# Two supported boxes for one object at one time must overlap this much to agree.
AGREEMENT_IOU = 0.5
# Rows this close in time are the same picture.
SAME_FRAME_SEC = 0.02
# A model estimate within this distance of a canonical row is superseded by it.
SUPERSEDE_SEC = 0.45
_PREFIX = "resolve_"


def detection_support(detector_id: str | None, has_mask: bool = False) -> str:
    kind = DETECTOR_EVIDENCE.get((detector_id or "").strip(), "proposal")
    if kind == "detected" and has_mask:
        return "mask"
    return kind


def temporal_support_by_track(points, minimum_hits: int = 2) -> dict[str, str]:
    """``repeated`` when a track has at least `minimum_hits` non-flow detections."""
    hits: dict[str, int] = defaultdict(int)
    flows: dict[str, int] = defaultdict(int)
    for point in points:
        ref = getattr(point, "track_ref", None)
        if not ref:
            continue
        if getattr(point, "detector_id", "") == "optical_flow":
            flows[ref] += 1
        else:
            hits[ref] += 1
    support: dict[str, str] = {}
    for ref in set(hits) | set(flows):
        if hits[ref] >= minimum_hits:
            support[ref] = "repeated"
        elif hits[ref] == 0:
            support[ref] = "flow_only"
        else:
            support[ref] = "single"
    return support


def is_tracker_row(row: ObservationPrediction) -> bool:
    if row.source_model == TRACKER_SOURCE:
        return True
    return not row.source_model and bool(row.track_ref) and row.geometry_origin in {"detected", "optical_flow"}


def is_estimate(row: ObservationPrediction) -> bool:
    return row.source_model not in {SAM_ID, TRACKER_SOURCE} and not is_tracker_row(row)


def mark_estimates(document: AnalysisDocument, authority: str = "vlm") -> AnalysisDocument:
    """Tag a language-model response's boxes as estimates at the moment they are parsed.

    The role stays empty. Only the resolver assigns a role, so a role on any row
    means evidence fusion has run.
    """
    rows = [
        row
        if row.geometry_origin or row.source_model
        else row.model_copy(
            update={
                "geometry_origin": "vlm_estimated" if row.bbox is not None else "unobserved",
                "geometry_authority": authority,
                "detection_support": "none",
                "semantic_support": "labeled",
            }
        )
        for row in document.observations
    ]
    return document.model_copy(update={"observations": rows})


def display_observations(document: AnalysisDocument) -> list[ObservationPrediction]:
    """The rows every overlay, crop, metric, and prompt reads."""
    return list(document.observations)


def inspection_observations(document: AnalysisDocument) -> list[ObservationPrediction]:
    """Every stored row, for the perception inspector. Not for scoring or overlays."""
    return [*document.observations, *document.raw_observations]


def has_canonical(document: AnalysisDocument, local_id: str) -> bool:
    return any(row.object_local_id == local_id and row.observation_role == "canonical" for row in document.observations)


def canonical_objects(document: AnalysisDocument) -> set[str]:
    return {row.object_local_id for row in document.observations if row.observation_role == "canonical"}


def _canonical_times(document: AnalysisDocument) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    for row in document.observations:
        if row.observation_role == "canonical":
            times[row.object_local_id].append(row.timestamp)
    return times


def fence_repair(document: AnalysisDocument, repair) -> AnalysisDocument:
    """Run a legacy box repair on model estimates only.

    The repair sees the estimates of each object at times where that object
    has no canonical row. Canonical rows, tracker rows, and SAM rows pass
    through untouched. An estimate the repair moves onto an object that is
    measured at that time is kept as a raw row instead of being shown.
    """
    canonical = _canonical_times(document)

    def covered(row: ObservationPrediction) -> bool:
        return _near(canonical.get(row.object_local_id, []), row.timestamp, SUPERSEDE_SEC)

    editable = [row for row in document.observations if is_estimate(row) and not covered(row)]
    protected = [row for row in document.observations if not (is_estimate(row) and not covered(row))]
    repaired = repair(document.model_copy(update={"observations": editable}))
    shown = [row for row in repaired.observations if not (is_estimate(row) and covered(row))]
    moved = [row for row in repaired.observations if is_estimate(row) and covered(row)]
    observations = sorted([*protected, *shown], key=lambda item: (item.timestamp, item.object_local_id, item.source_model))
    return repaired.model_copy(
        update={"observations": observations, "raw_observations": [*repaired.raw_observations, *moved]}
    )


def tracker_rows(document: AnalysisDocument, spatial: AnalysisDocument) -> list[ObservationPrediction]:
    """Tracker rows for every object bound to a tracked instance.

    An instance claimed by two objects binds to neither. A track no object
    names stays in the perception registry and adds no row here.
    """
    claims: dict[str, list] = defaultdict(list)
    for obj in document.objects:
        if obj.instance_id:
            claims[obj.instance_id].append(obj)
    rows: list[ObservationPrediction] = []
    for row in spatial.observations:
        if not row.track_ref:
            continue
        owners = claims.get(row.track_ref, [])
        if len(owners) != 1:
            continue
        owner = owners[0]
        status = owner.association_status or "referenced"
        rows.append(
            row.model_copy(
                update={
                    "object_local_id": owner.local_id,
                    "source_model": TRACKER_SOURCE,
                    "association_support": status,
                    "semantic_support": "labeled" if status == "associated" else "provisional",
                }
            )
        )
    return rows


def _fill(row: ObservationPrediction, sam_counts: dict[str, int]) -> ObservationPrediction:
    """Fill missing provenance and reset the resolver's own outputs."""
    update: dict = {"observation_role": "raw"}
    if row.source_model == SAM_ID:
        support = row.association_support or ("associated" if row.association_status in {"associated", "conflict"} else "unmatched")
        update.update(
            geometry_origin="detected",
            geometry_authority=SAM_ID,
            detection_support="mask" if row.mask_ref else "detected",
            association_support=support,
            association_status=support if support != "associated" else "associated",
            temporal_support=row.temporal_support
            or ("repeated" if sam_counts.get(row.track_ref or "", 0) >= 2 else "single"),
            semantic_support=row.semantic_support or "provisional",
        )
    elif is_tracker_row(row):
        flow = row.geometry_origin == "optical_flow"
        update.update(
            source_model=TRACKER_SOURCE,
            geometry_origin="optical_flow" if flow else "detected",
            detection_support="none" if flow else (row.detection_support or detection_support(row.geometry_authority)),
            temporal_support="flow_only" if flow and not row.temporal_support else row.temporal_support or "single",
            association_status="",
        )
    else:
        update.update(
            geometry_origin=row.geometry_origin or ("vlm_estimated" if row.bbox is not None else "unobserved"),
            geometry_authority=row.geometry_authority or "vlm",
            detection_support="none",
            semantic_support=row.semantic_support or "labeled",
        )
    return row.model_copy(update=update)


def _eligible(row: ObservationPrediction) -> bool:
    if row.bbox is None:
        return False
    if row.geometry_authority == HUMAN_AUTHORITY:
        return True
    if row.source_model == SAM_ID:
        return row.association_support == "associated" and row.detection_support in {"mask", "detected"}
    if row.source_model == TRACKER_SOURCE:
        return (
            row.geometry_origin == "detected"
            and row.detection_support in {"detected", "mask"}
            and row.temporal_support == "repeated"
        )
    return False


def _rank(row: ObservationPrediction) -> tuple:
    strength = {"mask": 0, "detected": 1, "proposal": 2, "none": 3}.get(row.detection_support, 4)
    if row.geometry_authority == HUMAN_AUTHORITY:
        strength = -1
    return (strength, 0 if row.source_model == SAM_ID else 1, row.timestamp)


def _clusters(rows: list[ObservationPrediction]) -> list[list[ObservationPrediction]]:
    ordered = sorted(rows, key=lambda item: item.timestamp)
    groups: list[list[ObservationPrediction]] = []
    for row in ordered:
        if groups and row.timestamp - groups[-1][0].timestamp <= SAME_FRAME_SEC:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _agree(rows: list[ObservationPrediction]) -> bool:
    boxes = [(row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h) for row in rows if row.bbox is not None]
    return all(iou(left, right) >= AGREEMENT_IOU for i, left in enumerate(boxes) for right in boxes[i + 1 :])


def _near(times: list[float], timestamp: float, window: float) -> bool:
    return any(abs(item - timestamp) <= window for item in times)


def resolve_document(document: AnalysisDocument, spatial: AnalysisDocument | None = None) -> AnalysisDocument:
    """Partition every stored row into the display set and the raw set.

    With `spatial`, tracker rows are rebuilt from it for objects bound to a
    tracked instance, so a retry does not append duplicates.
    """
    pool = [*document.observations, *document.raw_observations]
    if spatial is not None:
        pool = [row for row in pool if not is_tracker_row(row)]
        pool.extend(tracker_rows(document, spatial))
    sam_counts: dict[str, int] = defaultdict(int)
    for row in pool:
        if row.source_model == SAM_ID and row.track_ref:
            sam_counts[row.track_ref] += 1
    pool = [_fill(row, sam_counts) for row in pool]

    by_object: dict[str, list[ObservationPrediction]] = defaultdict(list)
    for row in pool:
        by_object[row.object_local_id].append(row)

    display: list[ObservationPrediction] = []
    hidden: list[ObservationPrediction] = []
    conflicts: dict[str, list[float]] = defaultdict(list)
    counts = defaultdict(int)
    for local_id, rows in by_object.items():
        canonical: list[ObservationPrediction] = []
        rest: list[ObservationPrediction] = []
        for cluster in _clusters([row for row in rows if _eligible(row)]):
            if _agree(cluster):
                best = min(cluster, key=_rank)
                canonical.append(
                    best.model_copy(
                        update={
                            "observation_role": "canonical",
                            "evidence_ref": f"{best.geometry_authority}:{best.track_ref or ''}@{best.timestamp:.3f}",
                        }
                    )
                )
                rest.extend(item for item in cluster if item is not best)
            else:
                conflicts[local_id].append(cluster[0].timestamp)
                rest.extend(item.model_copy(update={"association_status": "conflict"}) for item in cluster)
        rest.extend(row for row in rows if not _eligible(row))
        canonical_times = [row.timestamp for row in canonical]
        estimate_times = [
            row.timestamp
            for row in rest
            if is_estimate(row) and not _near(canonical_times, row.timestamp, SUPERSEDE_SEC)
        ]
        display.extend(canonical)
        counts["canonical"] += len(canonical)
        fallback: list[ObservationPrediction] = []
        for row in rest:
            if _near(canonical_times, row.timestamp, SUPERSEDE_SEC):
                hidden.append(row)
            elif is_estimate(row):
                display.append(row)
                counts["estimate"] += 1
            elif row.association_status == "conflict" or _near(estimate_times, row.timestamp, SUPERSEDE_SEC):
                hidden.append(row)
            else:
                fallback.append(row)
        for cluster in _clusters(fallback):
            best = min(cluster, key=_rank)
            display.append(best)
            counts["provisional"] += 1
            hidden.extend(item for item in cluster if item is not best)
    counts["hidden"] = len(hidden)

    ambiguities = [item for item in document.ambiguities if not item.local_id.startswith(_PREFIX)]
    for local_id, times in sorted(conflicts.items()):
        counts["conflicts"] += len(times)
        ambiguities.append(
            AmbiguityPrediction(
                local_id=f"{_PREFIX}conflict_{local_id}",
                kind=AmbiguityKind.CONFLICTING_EVIDENCE,
                description=(
                    f"Two measurements of {local_id} disagree at {len(times)} analyzed "
                    f"{'time' if len(times) == 1 else 'times'} between {min(times):.2f}s and {max(times):.2f}s. "
                    "Neither was made canonical. Both raw rows were kept."
                ),
                time_start=min(times),
                time_end=max(times),
                object_local_ids=[local_id],
                evidence=Evidence(
                    classification=EvidenceClass.UNKNOWN,
                    provenance=Provenance.DETERMINISTIC,
                    note="Supported boxes overlapped less than the agreement threshold.",
                    frame_timestamps=sorted(times)[:12],
                ),
            )
        )
    display.sort(key=lambda item: (item.timestamp, item.object_local_id, item.source_model))
    hidden.sort(key=lambda item: (item.timestamp, item.object_local_id, item.source_model))
    update: dict = {"observations": display, "raw_observations": hidden, "ambiguities": ambiguities}
    if document.perception is not None:
        diagnostics = dict(document.perception.diagnostics)
        diagnostics["resolution"] = {
            "canonical": counts["canonical"],
            "estimates_shown": counts["estimate"],
            "provisional_shown": counts["provisional"],
            "raw_hidden": counts["hidden"],
            "conflicts": counts["conflicts"],
        }
        update["perception"] = document.perception.model_copy(update={"diagnostics": diagnostics})
    return document.model_copy(update=update)
