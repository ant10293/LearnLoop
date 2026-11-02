"""Associate SAM tracks with LearnLoop objects and fuse geometry.

SAM supplies boxes, masks, and track identity inside one response. It does not
supply actions, relations, or completion verdicts. A frame SAM did not return
stays unobserved. It is not recorded as absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.providers.sam import (
    MAX_PROMPTED_OBJECTS,
    SpatialMark,
    VideoChunk,
    concept_text,
    scoped_track_id,
)
from app.providers.types import ProviderError
from app.schemas.analysis import (
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    HypothesisRecord,
    ObservationPrediction,
    PerceptionRecord,
)
from app.schemas.enums import (
    AmbiguityKind,
    Continuity,
    EvidenceClass,
    Provenance,
    Visibility,
)
from app.services.storage import data_path

SUMMARY_STAGES = {
    "workflow",
    "ambiguity",
    "evidence_fusion",
    "reinspection",
}
ACCEPT_IOU = 0.35
MARGIN = 0.12
CONFLICT_IOU = 0.2
TIME_WINDOW = 0.45
_TRAILING_INSTANCE = re.compile(
    r"(?:\s+(?:first|second|third|fourth|fifth|another|copy|\d+))+$",
    re.IGNORECASE,
)


@dataclass
class TrackSample:
    session_id: str
    raw_track_id: str
    timestamp: float
    bbox: BBox
    confidence: float | None
    frame_index: int
    request_id: str = ""
    bound_object_id: str = ""
    mask_ref: str = ""
    mask_data: str = ""
    mask_payload: str = ""
    mask_encoding: str = ""
    mask_height: int = 0
    mask_width: int = 0
    source_width: int = 0
    source_height: int = 0
    prompt_kind: str = ""
    candidates: tuple[str, ...] = ()


@dataclass
class TrackAssociation:
    session_id: str
    raw_track_id: str
    object_local_id: str | None
    status: str
    score: float
    margin: float | None
    rivals: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def scoped_id(self) -> str:
        return scoped_track_id(self.session_id, self.raw_track_id)


@dataclass
class SamRequestPlan:
    kind: str
    prompt: str
    session_id: str
    bindings: dict[int, str]
    propagation: str | None
    labels: list[str]
    targets: list[str] = field(default_factory=list)


def plan_prompts(
    document: AnalysisDocument,
    chunk: VideoChunk,
    width: int,
    height: int,
    fps: float,
) -> list[SamRequestPlan]:
    """One concept prompt per noun. The reasoning model's box is not the prompt.

    A point on a bad rectangle makes SAM segment that spot, so the outline
    inherits the wrong size and place. The noun lets SAM find the part.
    Two objects that share a noun share one request. Concept text never
    includes a box or a point.
    """
    del width, height, fps
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for obj in document.objects:
        try:
            noun = _concept_noun(obj.label)
        except ProviderError:
            continue
        if noun not in groups:
            order.append(noun)
            groups[noun] = []
        if obj.local_id not in groups[noun]:
            groups[noun].append(obj.local_id)
    plans: list[SamRequestPlan] = []
    for noun in order[:MAX_PROMPTED_OBJECTS]:
        targets = groups[noun]
        plans.append(
            SamRequestPlan(
                kind="concept",
                prompt=noun,
                session_id=f"{chunk.session_id}s{len(plans)}",
                bindings={},
                propagation=None,
                labels=[noun],
                targets=targets,
            )
        )
    return plans


def reference_boxes(
    document: AnalysisDocument,
    spatial: AnalysisDocument | None = None,
    *,
    include_estimates: bool = True,
) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    """Boxes that stand for each object when a SAM track is matched to it.

    Repeated tracker detections of the object's bound instance come first.
    The model's own estimates are used where no measurement exists, unless
    `include_estimates` is off. SAM rows and flow holds are never references.
    """
    from app.perception.resolve import is_estimate

    refs: dict[str, list] = {}
    measured: set[str] = set()
    if spatial is not None:
        owners = {obj.instance_id: obj.local_id for obj in document.objects if obj.instance_id}
        for row in spatial.observations:
            owner = owners.get(row.track_ref or "")
            if owner is None or row.bbox is None or row.geometry_origin == "optical_flow":
                continue
            refs.setdefault(owner, []).append((row.timestamp, _tuple(row.bbox)))
            measured.add(owner)
    if not include_estimates:
        return refs
    for row in [*document.observations, *document.raw_observations]:
        if row.bbox is None or not is_estimate(row) or row.object_local_id in measured:
            continue
        refs.setdefault(row.object_local_id, []).append((row.timestamp, _tuple(row.bbox)))
    return refs


def associate_tracks(
    document: AnalysisDocument,
    samples: list[TrackSample],
    *,
    scene_boundaries: list[float] | None = None,
    spatial: AnalysisDocument | None = None,
    measured_bindings: bool = False,
) -> list[TrackAssociation]:
    """Link each SAM track to at most one object by temporal box overlap.

    Matching runs inside one SAM response, so a raw id never crosses a
    session. A concept prompt limits the candidates to the objects that noun
    was asked for; the noun is not evidence of which one. A prompted point
    binds its object unless the track contradicts it. Confidence and the
    nearest leftover center are not used. A close rival leaves the track
    unassigned. With `measured_bindings`, a track prompted from a trusted seed
    can only be contradicted by a measurement, not by a model's box estimate.
    """
    from app.perception.associate import associate, pair_stats

    del scene_boundaries  # gaps are recorded in the summary; they do not force a match
    references = reference_boxes(document, spatial)
    binding_references = reference_boxes(document, spatial, include_estimates=False) if measured_bindings else references
    grouped: dict[tuple[str, str], list[TrackSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.session_id, sample.raw_track_id), []).append(sample)
    by_session: dict[str, list[tuple[str, list[TrackSample]]]] = {}
    for (session_id, raw_id), rows in grouped.items():
        by_session.setdefault(session_id, []).append((raw_id, rows))

    associations: list[TrackAssociation] = []
    for session_id, tracks in by_session.items():
        open_tracks: dict[str, list] = {}
        allowed: set[str] = set()
        bound_targets: set[str] = set()
        for raw_id, rows in tracks:
            boxes = [(row.timestamp, _tuple(row.bbox)) for row in rows]
            binding = next((row.bound_object_id for row in rows if row.bound_object_id), "")
            if binding:
                pair = pair_stats(boxes, binding_references.get(binding, []))
                if pair.shared and pair.score < CONFLICT_IOU:
                    associations.append(
                        TrackAssociation(
                            session_id,
                            raw_id,
                            binding,
                            "conflict",
                            pair.score,
                            None,
                            rivals=[binding],
                            note=(
                                "The prompted object did not overlap this track. "
                                "The track was kept as a conflict and was not forced onto the object."
                            ),
                        )
                    )
                else:
                    bound_targets.add(binding)
                    associations.append(
                        TrackAssociation(
                            session_id,
                            raw_id,
                            binding,
                            "associated",
                            pair.score,
                            None,
                            rivals=[binding],
                            note="SAM tracked the point it was prompted with for this object.",
                        )
                    )
                continue
            open_tracks[raw_id] = boxes
            for row in rows:
                allowed.update(row.candidates)
        candidates = {
            local_id: rows
            for local_id, rows in references.items()
            if local_id not in bound_targets and (not allowed or local_id in allowed)
        }
        links, _stats = associate(open_tracks, candidates, accept=ACCEPT_IOU, margin=MARGIN)
        for link in links:
            status = {"unmatched": "unassigned"}.get(link.status, link.status)
            associations.append(
                TrackAssociation(
                    session_id,
                    link.subject,
                    link.candidate,
                    status,
                    link.score,
                    link.margin,
                    rivals=link.rivals,
                    note=_link_note(link),
                )
            )
    return associations


def _link_note(link) -> str:
    if link.status == "associated":
        return f"Linked by box overlap {link.score:.2f} on {link.shared} shared times inside one SAM response."
    if link.status == "weak":
        return "Only one shared time overlapped. The track was not assigned on a single frame."
    if link.status == "ambiguous":
        return "Competing objects overlap this track. Neither was selected."
    if link.rivals:
        return "Overlap was too weak to assign this track. Labels and confidence were not used."
    return "No spatial overlap linked this track to a known object. The label was not used."


def fuse_geometry(
    document: AnalysisDocument,
    samples: list[TrackSample],
    associations: list[TrackAssociation],
    *,
    scene_boundaries: list[float] | None = None,
    coverage_note: str = "",
) -> AnalysisDocument:
    """Add SAM boxes beside existing evidence. Do not delete a stronger observation."""
    lookup = {(item.session_id, item.raw_track_id): item for item in associations}
    kept = [item for item in document.observations if item.source_model != "sam-3.1"]
    kept_raw = [item for item in document.raw_observations if item.source_model != "sam-3.1"]
    ambiguities = [item for item in document.ambiguities if not item.local_id.startswith("sam_")]
    added: list[ObservationPrediction] = []
    extra_ambiguities: list[AmbiguityPrediction] = []
    boundaries = list(scene_boundaries or [])
    for sample in samples:
        association = lookup.get((sample.session_id, sample.raw_track_id))
        if association is None or association.object_local_id is None:
            continue
        if association.status not in {"associated", "conflict"}:
            continue
        status, continuity, note = _placement(association, sample, boundaries)
        added.append(
            ObservationPrediction(
                object_local_id=association.object_local_id,
                timestamp=sample.timestamp,
                bbox=sample.bbox,
                visibility=Visibility.VISIBLE,
                description="SAM 3.1 segmentation",
                evidence=Evidence(
                    classification=EvidenceClass.DIRECTLY_OBSERVED,
                    provenance=Provenance.CV_DETECTION,
                    note=note,
                    frame_timestamps=[sample.timestamp],
                ),
                confidence=sample.confidence,
                continuity=continuity,
                track_ref=association.scoped_id,
                association_status=status,
                source_model="sam-3.1",
                request_id=sample.request_id,
                mask_ref=sample.mask_ref,
                observation_role="raw",
                geometry_origin="detected",
                geometry_authority="sam-3.1",
                detection_support="mask" if (sample.mask_ref or sample.mask_data or sample.mask_payload) else "detected",
                association_support="associated" if status == "associated" else "conflict",
                semantic_support="provisional",
                source_width=sample.source_width or None,
                source_height=sample.source_height or None,
            )
        )
        if status == "conflict":
            extra_ambiguities.append(
                AmbiguityPrediction(
                    local_id=f"sam_conflict_{association.scoped_id}_{sample.frame_index}",
                    kind=AmbiguityKind.CONFLICTING_EVIDENCE,
                    description=(
                        f"SAM and the reasoning model disagree on {association.object_local_id} "
                        f"at {sample.timestamp:.2f}s. Both boxes were kept."
                    ),
                    time_start=sample.timestamp,
                    time_end=sample.timestamp,
                    object_local_ids=[association.object_local_id],
                    evidence=Evidence(
                        classification=EvidenceClass.DIRECTLY_OBSERVED,
                        provenance=Provenance.CV_DETECTION,
                        note="Conflicting localizations were preserved.",
                    ),
                )
            )
    for association in associations:
        if association.status not in {"ambiguous", "unassigned"}:
            continue
        rows = [
            sample
            for sample in samples
            if sample.session_id == association.session_id and sample.raw_track_id == association.raw_track_id
        ]
        if not rows:
            continue
        start = min(sample.timestamp for sample in rows)
        end = max(sample.timestamp for sample in rows)
        kind = (
            AmbiguityKind.UNCERTAIN_IDENTITY
            if association.status == "ambiguous"
            else AmbiguityKind.MISSING_EVIDENCE
        )
        extra_ambiguities.append(
            AmbiguityPrediction(
                local_id=f"sam_{association.status}_{association.scoped_id}",
                kind=kind,
                description=association.note,
                time_start=start,
                time_end=end,
                object_local_ids=association.rivals[:2],
                evidence=Evidence(
                    classification=EvidenceClass.UNKNOWN,
                    provenance=Provenance.CV_DETECTION,
                    note="This track was not assigned to an object.",
                ),
            )
        )
    for start, end, local_id in _gap_intervals(samples, lookup):
        extra_ambiguities.append(
            AmbiguityPrediction(
                local_id=f"sam_gap_{local_id}_{start:.2f}",
                kind=AmbiguityKind.UNOBSERVED_INTERVAL,
                description=(
                    f"{local_id} has no SAM observation from {start:.2f}s to {end:.2f}s. "
                    "That gap is not evidence the object is absent."
                ),
                time_start=start,
                time_end=end,
                object_local_ids=[local_id],
                evidence=Evidence(
                    classification=EvidenceClass.UNKNOWN,
                    provenance=Provenance.CV_DETECTION,
                    note="Unobserved interval. Not confirmed absence.",
                ),
            )
        )
    summary = render_summary(document, samples, associations, coverage_note)
    perception = _perception(document, associations, summary)
    observations = [*kept, *added]
    observations.sort(key=lambda item: (item.timestamp, item.object_local_id, item.source_model))
    return document.model_copy(
        update={
            "observations": observations,
            "raw_observations": kept_raw,
            "ambiguities": ambiguities + extra_ambiguities,
            "perception": perception,
        }
    )


def render_summary(
    document: AnalysisDocument,
    samples: list[TrackSample],
    associations: list[TrackAssociation],
    coverage_note: str = "",
) -> str:
    """Compact text for a later reasoning call. Masks are not included."""
    lines = [
        "Roles: the primary model owns actions, relations, and workflow judgments. "
        "SAM 3.1 owns boxes, masks, and track identity inside one response.",
    ]
    if coverage_note:
        lines.append(coverage_note)
    lookup = {(item.session_id, item.raw_track_id): item for item in associations}
    labels = {obj.local_id: obj.label for obj in document.objects}
    grouped: dict[str, list[TrackSample]] = {}
    for sample in samples:
        association = lookup.get((sample.session_id, sample.raw_track_id))
        key = association.object_local_id if association and association.object_local_id else association.scoped_id if association else sample.raw_track_id
        grouped.setdefault(key, []).append(sample)
    for key, rows in grouped.items():
        rows.sort(key=lambda item: item.timestamp)
        association = lookup.get((rows[0].session_id, rows[0].raw_track_id))
        status = association.status if association else "unassigned"
        label = labels.get(key, "")
        name = f"{key} {label}".strip()
        start = rows[0].timestamp
        end = rows[-1].timestamp
        conf = [row.confidence for row in rows if row.confidence is not None]
        conf_text = f" sam_confidence {min(conf):.2f}-{max(conf):.2f}" if conf else ""
        lines.append(
            f"{name}: track {association.scoped_id if association else rows[0].raw_track_id} "
            f"{status} {start:.2f}-{end:.2f}s samples {len(rows)}{conf_text}"
        )
    if not samples:
        lines.append("SAM returned no tracks. That is an empty detection, not confirmed absence.")
    lines.append(
        "Intervals without a sample were not observed. Do not convert them into absence, "
        "and do not invent an action from a box."
    )
    return "\n".join(lines)


def write_sam_summary(run_id: str, text: str) -> None:
    path = data_path("runs", run_id, "sam_summary.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def prompt_prior(document: AnalysisDocument) -> str:
    """JSON for a later model call, without per-frame SAM rows.

    A tracked clip can hold thousands of boxes. Those stay in the stored
    document. The call gets the reasoning-model records plus the short SAM summary.
    """
    canonical_sam: dict[str, list[ObservationPrediction]] = {}
    for item in document.observations:
        if item.source_model == "sam-3.1" and item.observation_role == "canonical":
            canonical_sam.setdefault(item.object_local_id, []).append(item)
    sampled = [
        rows[index]
        for rows in canonical_sam.values()
        for index in sorted({round(i * (len(rows) - 1) / 7) for i in range(8)} if len(rows) > 8 else range(len(rows)))
    ]
    compact = document.model_copy(
        update={
            "raw_observations": [],
            "observations": [item for item in document.observations if item.source_model != "sam-3.1"] + sampled,
            "ambiguities": [item for item in document.ambiguities if not item.local_id.startswith("sam_")],
        }
    )
    text = compact.model_dump_json()
    if len(text) <= 120_000:
        return text
    shorter = compact.model_copy(update={"observations": compact.observations[:48]})
    return (
        shorter.model_dump_json()
        + "\n\nObservation list shortened to stay inside the model context window."
    )


def sam_evidence_block(run_id: str, stage: str) -> str:
    if stage not in SUMMARY_STAGES:
        return ""
    path = data_path("runs", run_id, "sam_summary.txt")
    if not path.exists():
        return ""
    body = path.read_text()[:6000]
    if not body.strip():
        return ""
    return (
        "SAM 3.1 geometric evidence from a separate model. "
        "Use it for location, continuity, and whether a step has visual support. "
        "Do not treat it as an action or completion authority. "
        "A missing detection is unobserved, not confirmed absence. "
        "Its confidence is not calibrated against yours.\n"
        + body
    )


def merge_reconciliation(base: AnalysisDocument, proposed: AnalysisDocument) -> AnalysisDocument:
    """Keep SAM geometry. Accept refined actions, states, and workflow steps."""
    sam_ambiguities = [item for item in base.ambiguities if item.local_id.startswith("sam_")]
    proposed_ambiguities = [
        item for item in (proposed.ambiguities or []) if not item.local_id.startswith("sam_")
    ]
    return base.model_copy(
        update={
            "summary": proposed.summary or base.summary,
            "actions": proposed.actions or base.actions,
            "states": proposed.states or base.states,
            "transitions": proposed.transitions or base.transitions,
            "relationships": proposed.relationships or base.relationships,
            "workflow_steps": proposed.workflow_steps or base.workflow_steps,
            "dependencies": proposed.dependencies or base.dependencies,
            "ambiguities": sam_ambiguities + (proposed_ambiguities or [
                item for item in base.ambiguities if not item.local_id.startswith("sam_")
            ]),
        }
    )


def _concept_noun(label: str) -> str:
    text = concept_text(label)
    noun = _TRAILING_INSTANCE.sub("", text).strip()
    return noun or text


def _mark(frame: int, object_id: int, box: BBox, width: int, height: int) -> SpatialMark:
    """Prompt with the center of the sighting, not the rectangle itself."""
    x = min(max(int(round((box.x + box.w / 2) * width)), 0), max(width - 1, 0))
    y = min(max(int(round((box.y + box.h / 2) * height)), 0), max(height - 1, 0))
    return SpatialMark(
        frame_index=frame,
        object_id=object_id,
        kind="point",
        width=width,
        height=height,
        x=x,
        y=y,
    )


def _placement(association: TrackAssociation, sample: TrackSample, boundaries: list[float]):
    """Continuity for one SAM sample. A model estimate never demotes a SAM row; the resolver decides."""
    continuity = Continuity.CONTINUOUS
    note = association.note
    if association.status == "conflict":
        continuity = Continuity.UNCERTAIN
    if _crosses(sample.timestamp, boundaries):
        continuity = Continuity.UNCERTAIN
        note += " A scene boundary is nearby, so this sighting was not treated as continuous."
    return association.status, continuity, note.strip()


def _crosses(timestamp: float, boundaries: list[float]) -> bool:
    return any(abs(timestamp - boundary) <= TIME_WINDOW for boundary in boundaries)


def _gap_intervals(samples: list[TrackSample], lookup: dict) -> list[tuple[float, float, str]]:
    grouped: dict[tuple[str, str], list[TrackSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.session_id, sample.raw_track_id), []).append(sample)
    gaps = []
    for key, rows in grouped.items():
        association = lookup.get(key)
        if association is None or not association.object_local_id or association.status != "associated":
            continue
        ordered = sorted(rows, key=lambda item: item.frame_index)
        for left, right in zip(ordered, ordered[1:]):
            if right.frame_index - left.frame_index <= 1:
                continue
            if right.timestamp - left.timestamp < 0.2:
                continue
            gaps.append((left.timestamp, right.timestamp, association.object_local_id))
    return gaps


def _perception(document: AnalysisDocument, associations: list[TrackAssociation], summary: str) -> PerceptionRecord:
    current = document.perception
    hypotheses = list(current.hypotheses) if current else []
    for association in associations:
        if association.status not in {"ambiguous", "conflict"}:
            continue
        candidate = association.rivals[0] if association.rivals else (association.object_local_id or "")
        runner = association.rivals[1] if len(association.rivals) > 1 else None
        hypotheses.append(
            HypothesisRecord(
                hypothesis_id=f"sam-{association.scoped_id}",
                timestamp=0,
                subject_instance_id=association.scoped_id,
                candidate_instance_id=candidate or association.scoped_id,
                runner_up_instance_id=runner,
                association_score=association.score,
                margin=association.margin,
                decision="ambiguous",
                note=association.note,
            )
        )
    diagnostics = dict(current.diagnostics) if current else {}
    diagnostics["sam_tracks"] = len(associations)
    diagnostics["sam_associated"] = sum(item.status == "associated" for item in associations)
    diagnostics["sam_ambiguous"] = sum(item.status == "ambiguous" for item in associations)
    diagnostics["sam_unassigned"] = sum(item.status == "unassigned" for item in associations)
    note = summary if current is None or not current.notes else current.notes + "\n" + summary
    if current is None:
        return PerceptionRecord(
            strategy="sam_fusion",
            detector_id="sam-3.1",
            segmenter_id="sam-3.1",
            scores_are_probabilities=False,
            diagnostics=diagnostics,
            hypotheses=hypotheses,
            notes=note,
        )
    return current.model_copy(
        update={
            "segmenter_id": "sam-3.1",
            "diagnostics": diagnostics,
            "hypotheses": hypotheses,
            "notes": note,
            "scores_are_probabilities": False,
        }
    )


def _tuple(box: BBox) -> tuple[float, float, float, float]:
    return (box.x, box.y, box.w, box.h)
