"""Turn spatial tracks into an explicit, non-semantic analysis document."""

from __future__ import annotations

from collections import defaultdict

from app.perception.relations import geometric_relationships
from app.schemas.analysis import (
    ActionPrediction,
    AmbiguityPrediction,
    AnalysisDocument,
    BBox,
    CameraAssessment,
    Coverage,
    DependencyPrediction,
    Evidence,
    ObjectPrediction,
    ObservationPrediction,
    StatePrediction,
    TransitionPrediction,
    WorkflowStepPrediction,
)
from app.schemas.enums import (
    ActionOutcome,
    AmbiguityKind,
    DependencyCategory,
    EvidenceClass,
    Provenance,
    Visibility,
)
from app.video.spatial import SpatialReport


def _ev(note: str, start: float, end: float | None = None) -> Evidence:
    return Evidence(
        classification=EvidenceClass.DIRECTLY_OBSERVED,
        provenance=Provenance.CV_DETECTION,
        note=note,
        time_start=start,
        time_end=end if end is not None else start,
        frame_timestamps=[start],
    )


def document_from_spatial(
    report: SpatialReport,
    coverage: Coverage,
    *,
    width: int | None = None,
    height: int | None = None,
) -> AnalysisDocument:
    from app.perception.resolve import detection_support, temporal_support_by_track

    grouped: dict[str, list] = defaultdict(list)
    repeated = temporal_support_by_track(report.tracks)
    for point in report.tracks:
        grouped[point.track_ref].append(point)
    objects = []
    observations = []
    states = []
    transitions = []
    actions = []
    steps = []
    ambiguities = []
    for index, (ref, points) in enumerate(sorted(grouped.items()), start=1):
        points = sorted(points, key=lambda item: item.timestamp)
        local_id = f"Object_{index}"
        uncertain = any(item.continuity in {"uncertain", "reassociated"} for item in points)
        legacy = report.perception is None or report.perception.strategy in {"legacy_motion", "vlm_only"}
        objects.append(
            ObjectPrediction(
                local_id=local_id,
                instance_id=ref,
                identity_relation="physical_instance",
                label=f"Motion region {index}" if legacy else f"Unidentified instance {index}",
                description=(
                    "Image-space region that moved relative to the chosen camera model. "
                    "No semantic class was assigned."
                    if legacy
                    else "A tracked physical instance without a semantic class. The id is not a recognition label."
                ),
                visual_characteristics="motion blob" if legacy else "tracked region",
                initial_state="visible motion region" if legacy else "visible",
                final_state="visible motion region" if points and legacy else ("visible" if points else "unknown"),
                uncertain_identity=uncertain or legacy,
                manipulated=False,
                confidence=None,
            )
        )
        for point in points:
            visibility = getattr(point, "visibility", "visible") or "visible"
            flow = point.detector_id == "optical_flow"
            observations.append(
                ObservationPrediction(
                    object_local_id=local_id,
                    timestamp=point.timestamp,
                    bbox=_box(point.x, point.y, point.w, point.h),
                    visibility=visibility if visibility in {item.value for item in Visibility} else Visibility.VISIBLE,
                    description=point.note or "Region detected in the image.",
                    evidence=_ev(point.note or "Computer-vision track observation.", point.timestamp),
                    continuity=point.continuity,
                    coordinate_frame="image",
                    camera_compensated=point.camera_compensated,
                    track_ref=ref,
                    observation_role="raw",
                    source_model="tracker",
                    geometry_origin="optical_flow" if flow else "detected",
                    geometry_authority=point.detector_id or (report.perception.detector_id if report.perception else ""),
                    detection_support="none" if flow else detection_support(point.detector_id),
                    temporal_support="flow_only" if flow else repeated.get(ref, "single"),
                    semantic_support="unlabeled",
                    association_support="associated",
                    source_width=width or None,
                    source_height=height or None,
                )
            )
        states.append(
            StatePrediction(
                object_local_id=local_id,
                time_start=points[0].timestamp,
                time_end=points[-1].timestamp,
                state_label="visible in image",
                description="The region is visible at the sampled times. Gaps between samples are not filled in.",
                evidence=_ev("Sampled detections only.", points[0].timestamp, points[-1].timestamp),
            )
        )
        if uncertain:
            ambiguities.append(
                AmbiguityPrediction(
                    local_id=f"Ambiguity_track_{index}",
                    kind=AmbiguityKind.UNCERTAIN_IDENTITY,
                    description=points[-1].note or "Tracking continuity for this region is not a semantic identity.",
                    time_start=points[0].timestamp,
                    time_end=points[-1].timestamp,
                    object_local_ids=[local_id],
                    evidence=Evidence(
                        classification=EvidenceClass.UNKNOWN,
                        provenance=Provenance.DETERMINISTIC,
                        note="The baseline does not reidentify objects by meaning.",
                        time_start=points[0].timestamp,
                        time_end=points[-1].timestamp,
                    ),
                )
            )
        intervals = _motion_intervals(points)
        for nth, (start, end) in enumerate(intervals, start=1):
            action_id = f"Action_{index}_{nth}"
            actions.append(
                ActionPrediction(
                    local_id=action_id,
                    action_type="motion",
                    description=(
                        "Image-space motion was measured in this interval. "
                        "The physical verb (pick up, insert, and so on) is unknown."
                    ),
                    start=start,
                    end=end,
                    actor=None,
                    object_local_ids=[local_id],
                    initial_state="visible",
                    final_state="visible",
                    outcome=ActionOutcome.UNKNOWN,
                    uncertainty=0.85,
                    evidence=_ev(
                        "Motion score exceeded the baseline threshold. This is not an action label.",
                        start,
                        end,
                    ),
                )
            )
            steps.append(
                WorkflowStepPrediction(
                    local_id=f"Step_{index}_{nth}",
                    name=f"Observed motion {index}.{nth}",
                    required_objects=[local_id],
                    preconditions=["region visible"],
                    expected_action="motion",
                    completion_condition="motion score falls; semantic completion is unknown",
                    observed_result="image-space displacement",
                    action_local_ids=[action_id],
                    time_start=start,
                    time_end=end,
                    uncertainty=0.9,
                    evidence=Evidence(
                        classification=EvidenceClass.INFERRED,
                        provenance=Provenance.DETERMINISTIC,
                        note="Grouping motion into a step is bookkeeping, not a recovered procedure.",
                        time_start=start,
                        time_end=end,
                    ),
                )
            )
            if len(points) >= 2:
                transitions.append(
                    TransitionPrediction(
                        object_local_id=local_id,
                        timestamp=end,
                        from_state="earlier image position",
                        to_state="later image position",
                        description="The box moved between samples. Physical displacement is not claimed when the camera moves.",
                        evidence=_ev("Box endpoints differ.", start, end),
                        certainty=0.3 if report.viewpoint == "head_mounted" else 0.6,
                    )
                )
    dependencies = []
    for left, right in zip(steps, steps[1:], strict=False):
        dependencies.append(
            DependencyPrediction(
                from_step=left.local_id,
                to_step=right.local_id,
                category=DependencyCategory.OBSERVED_TEMPORAL,
                rationale="These motion intervals occurred in this order. No physical necessity is claimed.",
                evidence=Evidence(
                    classification=EvidenceClass.INFERRED,
                    provenance=Provenance.DETERMINISTIC,
                    note="Temporal bookkeeping only.",
                    time_start=left.time_start,
                    time_end=right.time_end,
                ),
            )
        )
    ambiguities.append(
        AmbiguityPrediction(
            local_id="Ambiguity_semantics",
            kind=AmbiguityKind.MISSING_EVIDENCE,
            description=(
                "The deterministic baseline detects motion regions. It does not identify objects "
                "or decide whether an attempted action succeeded."
            ),
            evidence=Evidence(
                classification=EvidenceClass.UNKNOWN,
                provenance=Provenance.DETERMINISTIC,
                note="No vision-language model ran.",
            ),
        )
    )
    if report.scene_change_count:
        ambiguities.append(
            AmbiguityPrediction(
                local_id="Ambiguity_scenes",
                kind=AmbiguityKind.UNOBSERVED_INTERVAL,
                description=(
                    f"{report.scene_change_count} hard scene cuts were detected. "
                    "Object identity was not invented across them."
                ),
                evidence=Evidence(
                    classification=EvidenceClass.DIRECTLY_OBSERVED,
                    provenance=Provenance.CV_DETECTION,
                    note="The view changed and did not return within the confirmation window.",
                ),
            )
        )
    camera = CameraAssessment(
        viewpoint=report.viewpoint,
        task_constraint=report.task_constraint,
        research_mode=report.research_mode,
        global_motion=report.global_motion,
        scene_change_count=report.scene_change_count,
        coordinate_frame="image",
        world_coordinates_supported=False,
        camera_motion_compensated=report.camera_motion_compensated,
        compensation_model=report.compensation_model,
        characterization_source="auto" if report.notes.endswith("overridden.") or "automatically" in report.notes else "user",
        notes=report.notes,
    )
    track_to_object = {obj.instance_id: obj.local_id for obj in objects if obj.instance_id}
    relationships = []
    for relation in geometric_relationships(report.tracks):
        subject = track_to_object.get(relation.subject_local_id)
        other = track_to_object.get(relation.other_local_id or "")
        if not subject or not other:
            continue
        relationships.append(relation.model_copy(update={"subject_local_id": subject, "other_local_id": other}))
    summary = (
        "Deterministic motion baseline. Regions are not named objects, and motion intervals "
        "are not verified physical actions."
        if report.perception is None or report.perception.strategy in {"legacy_motion", "vlm_only"}
        else "Tracked physical instances. Labels are not recognition classes, and motion is not a completed action."
    )
    return AnalysisDocument(
        summary=summary,
        fixture=False,
        camera=camera,
        coverage=coverage,
        objects=objects,
        observations=observations,
        states=states,
        transitions=transitions,
        actions=actions,
        workflow_steps=steps,
        dependencies=dependencies,
        ambiguities=ambiguities,
        relationships=relationships,
        perception=report.perception,
    )


def _motion_intervals(points) -> list[tuple[float, float]]:
    hot = [point for point in points if point.motion_score >= 2.0]
    if not hot:
        if len(points) >= 2:
            return [(points[0].timestamp, points[-1].timestamp)]
        return []
    intervals = []
    start = hot[0].timestamp
    prev = hot[0].timestamp
    for point in hot[1:]:
        if point.timestamp - prev > 1.2:
            intervals.append((start, max(prev, start + 0.1)))
            start = point.timestamp
        prev = point.timestamp
    intervals.append((start, max(prev, start + 0.1)))
    return intervals


def _box(x: float, y: float, w: float, h: float) -> BBox:
    x = min(max(x, 0.0), 0.98)
    y = min(max(y, 0.0), 0.98)
    w = min(max(w, 0.01), 1 - x)
    h = min(max(h, 0.01), 1 - y)
    return BBox(x=x, y=y, w=w, h=h)
