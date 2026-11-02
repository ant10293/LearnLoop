"""Reject unsupported certainty. Geometry does not prove a physical connection."""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument, ValidationIssue
from app.schemas.enums import RelationKind


def audit_claims(document: AnalysisDocument) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for transition in document.transitions:
        classification = transition.evidence.classification.value
        if transition.certainty is not None and transition.certainty >= 0.8 and classification in {"UNKNOWN", "INFERRED"}:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unsupported_certainty",
                    message=(
                        f"Transition {transition.from_state} → {transition.to_state} states high certainty "
                        "without a directly observed interval. The certainty was not increased."
                    ),
                    entity_type="transition",
                    entity_id=transition.object_local_id,
                )
            )
    physical = {RelationKind.ATTACHED.value, RelationKind.INSIDE.value, RelationKind.HELD_BY.value}
    for relation in document.relationships:
        kind = relation.kind.value if hasattr(relation.kind, "value") else relation.kind
        if kind not in physical:
            continue
        if relation.evidence.classification.value == "DIRECTLY_OBSERVED" and "proximity" in relation.description.lower():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="proximity_is_not_attachment",
                    message=f"Relationship {kind} is marked directly observed while the description only supports proximity.",
                    entity_type="relationship",
                    entity_id=relation.subject_local_id,
                )
            )
    return issues
