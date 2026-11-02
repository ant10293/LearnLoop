"""Bind model output to tracked instances without letting labels merge them."""

from __future__ import annotations

from app.schemas.analysis import AmbiguityPrediction, AnalysisDocument, Evidence, HypothesisRecord
from app.schemas.enums import AmbiguityKind, EvidenceClass, Provenance


def object_id_map(partial: AnalysisDocument, merged: AnalysisDocument) -> dict[str, str]:
    """Map window ids onto the merged document.

    Identical labels do not merge two records that carry different instance ids.
    Records without an instance id still match by label, which is the VLM-only path.
    """
    merged_by_instance = {obj.instance_id: obj.local_id for obj in merged.objects if obj.instance_id}
    merged_labels = {obj.local_id: " ".join(obj.label.casefold().split()) for obj in merged.objects}
    labels = {
        " ".join(obj.label.casefold().split()): obj.local_id
        for obj in reversed(merged.objects)
        if not obj.instance_id
    }
    mapping: dict[str, str] = {}
    for obj in partial.objects:
        if obj.instance_id:
            mapping[obj.local_id] = merged_by_instance.get(obj.instance_id, obj.local_id)
            continue
        key = " ".join(obj.label.casefold().split())
        if merged_labels.get(obj.local_id) == key:
            mapping[obj.local_id] = obj.local_id
            continue
        matched = labels.get(key)
        if matched:
            mapping[obj.local_id] = matched
        # A reused id with a different label is a different part. Leaving it
        # unmapped stops that box from landing on the earlier part.
    return mapping


def _boxes(rows) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for row in rows:
        if row.bbox is None:
            continue
        grouped.setdefault(row.object_local_id, []).append(
            (row.timestamp, (row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h))
        )
    return grouped


def _track_boxes(spatial: AnalysisDocument) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for row in spatial.observations:
        if row.bbox is None or not row.track_ref or row.geometry_origin == "optical_flow":
            continue
        grouped.setdefault(row.track_ref, []).append((row.timestamp, (row.bbox.x, row.bbox.y, row.bbox.w, row.bbox.h)))
    return grouped


def _span(rows: list) -> tuple[float, float]:
    times = [item[0] for item in rows]
    return min(times), max(times)


def associate_objects(document: AnalysisDocument, spatial: AnalysisDocument) -> AnalysisDocument:
    """Bind semantic objects to tracked instances by temporal box overlap, one to one.

    A model-named instance id is kept as ``referenced`` unless repeated shared
    times contradict it. Other objects bind only with two or more shared times
    above the floor and a clear margin. One shared time is ``weak``. A close
    rival is ``ambiguous``. Neither sets an instance id. A strong match to a
    second, time-disjoint track is stored as a re-identification hypothesis
    and is not merged.
    """
    from app.perception.associate import CONTRADICT_IOU, associate, pair_stats
    from app.perception.resolve import is_estimate

    perception = spatial.perception
    tracks = _track_boxes(spatial)
    known = {item.instance_id for item in perception.instances} if perception else set(tracks)
    estimates = _boxes(
        row for row in [*document.observations, *document.raw_observations] if is_estimate(row)
    )
    objects = {obj.local_id: obj for obj in document.objects}
    updates: dict[str, dict] = {}
    ambiguities = [item for item in document.ambiguities if not item.local_id.startswith("ground_")]
    claimed: set[str] = set()
    named_counts: dict[str, int] = {}
    for obj in document.objects:
        if obj.instance_id and obj.instance_id in known:
            named_counts[obj.instance_id] = named_counts.get(obj.instance_id, 0) + 1
    for obj in document.objects:
        if not obj.instance_id or obj.instance_id not in known:
            continue
        pair = pair_stats(estimates.get(obj.local_id, []), tracks.get(obj.instance_id, []))
        if named_counts[obj.instance_id] > 1 or (pair.shared >= 2 and pair.score < CONTRADICT_IOU):
            reason = (
                "Two objects named the same instance."
                if named_counts[obj.instance_id] > 1
                else f"Its boxes overlap that track by {pair.score:.2f} on {pair.shared} shared times."
            )
            updates[obj.local_id] = {
                "instance_id": None,
                "association_status": "ambiguous",
                "association_score": round(pair.score, 3),
                "association_rivals": [obj.instance_id],
                "uncertain_identity": True,
            }
            ambiguities.append(
                AmbiguityPrediction(
                    local_id=f"ground_contradicted_{obj.local_id}",
                    kind=AmbiguityKind.UNCERTAIN_IDENTITY,
                    description=f"{obj.local_id} named {obj.instance_id}, but the binding was not kept. {reason}",
                    object_local_ids=[obj.local_id],
                    evidence=Evidence(
                        classification=EvidenceClass.UNKNOWN,
                        provenance=Provenance.DETERMINISTIC,
                        note="A model-named instance id was contradicted or shared.",
                    ),
                )
            )
            continue
        status = "associated" if pair.shared >= 2 and pair.score >= 0.35 else "referenced"
        updates[obj.local_id] = {"association_status": status, "association_score": round(pair.score, 3) if pair.shared else None}
        claimed.add(obj.instance_id)
    open_subjects = {
        local_id: rows
        for local_id, rows in estimates.items()
        if local_id in objects and local_id not in updates and not objects[local_id].instance_id
    }
    open_tracks = {ref: rows for ref, rows in tracks.items() if ref not in claimed}
    links, stats = associate(open_subjects, open_tracks)
    hypotheses = list(perception.hypotheses) if perception else []
    hypotheses = [item for item in hypotheses if not item.hypothesis_id.startswith("ground-")]
    bound = {link.candidate for link in links if link.candidate}
    for link in links:
        update = {
            "association_status": link.status,
            "association_score": round(link.score, 3) if link.shared else None,
            "association_rivals": link.rivals,
        }
        if link.candidate:
            update["instance_id"] = link.candidate
            update["identity_relation"] = "physical_instance"
            primary = _span(open_tracks[link.candidate])
            for (subject, ref), pair in stats.items():
                if subject != link.subject or ref == link.candidate or ref in bound or ref in claimed:
                    continue
                if pair.shared < 2 or pair.score < 0.35:
                    continue
                other = _span(open_tracks[ref])
                if other[0] <= primary[1] and primary[0] <= other[1]:
                    continue
                hypotheses.append(
                    HypothesisRecord(
                        hypothesis_id=f"ground-{link.subject}-{ref}",
                        timestamp=other[0],
                        subject_instance_id=link.candidate,
                        candidate_instance_id=ref,
                        association_score=round(pair.score, 3),
                        decision="proposed",
                        note=(
                            f"{link.subject} overlaps both tracks, which never coexist. "
                            "They may be one object across a cut or occlusion. They were not merged."
                        ),
                    )
                )
        elif link.status in {"ambiguous", "weak"}:
            update["uncertain_identity"] = True
        updates[link.subject] = update
    new_objects = [obj.model_copy(update=updates[obj.local_id]) if obj.local_id in updates else obj for obj in document.objects]
    update: dict = {"objects": new_objects, "ambiguities": ambiguities}
    if perception is not None:
        diagnostics = dict(perception.diagnostics)
        diagnostics["grounding"] = {
            status: sum(1 for item in updates.values() if item.get("association_status") == status)
            for status in ("associated", "referenced", "weak", "ambiguous", "unmatched")
        }
        update["perception"] = perception.model_copy(update={"hypotheses": hypotheses, "diagnostics": diagnostics})
    return document.model_copy(update=update)


def apply_perception(document: AnalysisDocument, spatial: AnalysisDocument | None) -> AnalysisDocument:
    """Copy tracked instances onto a model document, bind objects, and flag unknown ids.

    Object labels are not replaced and never bind identity. An id the tracker
    did not produce stays a provisional proposal. Geometric overlap is not
    copied in as attachment.
    """
    if spatial is None or spatial.perception is None:
        return document
    perception = spatial.perception
    if not perception.used_for_identity:
        return document.model_copy(update={"perception": perception})
    document = associate_objects(document.model_copy(update={"perception": perception}), spatial)
    perception = document.perception or perception
    known = {item.instance_id for item in perception.instances}
    objects = []
    ambiguities = list(document.ambiguities)
    for obj in document.objects:
        if obj.instance_id and obj.instance_id not in known:
            objects.append(obj.model_copy(update={"uncertain_identity": True, "identity_relation": "provisional"}))
            ambiguities.append(
                AmbiguityPrediction(
                    local_id=f"Ambiguity_untracked_{obj.local_id}",
                    kind=AmbiguityKind.UNCERTAIN_IDENTITY,
                    description=(
                        f"{obj.local_id} names instance {obj.instance_id}, which is not in the "
                        "tracked registry. It is a provisional proposal, not a confirmed object."
                    ),
                    object_local_ids=[obj.local_id],
                    evidence=Evidence(
                        classification=EvidenceClass.UNKNOWN,
                        provenance=Provenance.DETERMINISTIC,
                        note="The language model did not receive this id from the tracker.",
                    ),
                )
            )
        elif obj.instance_id:
            objects.append(obj.model_copy(update={"identity_relation": obj.identity_relation or "physical_instance"}))
        else:
            objects.append(obj.model_copy(update={"identity_relation": obj.identity_relation or "semantic_label"}))
    return document.model_copy(update={"perception": perception, "objects": objects, "ambiguities": ambiguities})
