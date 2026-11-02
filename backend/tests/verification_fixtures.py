"""Deterministic procedure documents for verification tests.

The part names belong to the fixture, not to the verifier.
"""

from __future__ import annotations

from app.schemas.analysis import AnalysisDocument


def evidence(kind: str = "DIRECTLY_OBSERVED", times: list[float] | None = None) -> dict:
    return {
        "classification": kind,
        "provenance": "HUMAN_ANNOTATION",
        "frame_timestamps": times or [],
    }


def _box(area: float) -> dict:
    side = max(area, 0.02) ** 0.5
    return {"x": 0.1, "y": 0.1, "w": side, "h": side}


def obj(local_id: str, label: str, description: str, visual: str) -> dict:
    return {
        "local_id": local_id,
        "label": label,
        "description": description,
        "visual_characteristics": visual,
        "manipulated": True,
    }


def obs(local_id: str, timestamp: float, visibility: str = "visible", area: float = 0.04) -> dict:
    return {
        "object_local_id": local_id,
        "timestamp": timestamp,
        "visibility": visibility,
        "bbox": _box(area),
        "evidence": evidence(times=[timestamp]),
    }


def rel(subject: str, other: str, kind: str, start: float, end: float, classification: str = "DIRECTLY_OBSERVED") -> dict:
    return {
        "subject_local_id": subject,
        "other_local_id": other,
        "kind": kind,
        "time_start": start,
        "time_end": end,
        "evidence": evidence(classification, [start]),
    }


def state(local_id: str, label: str, start: float, end: float) -> dict:
    return {
        "object_local_id": local_id,
        "time_start": start,
        "time_end": end,
        "state_label": label,
        "evidence": evidence(times=[start]),
    }


def transition(local_id: str, before: str, after: str, timestamp: float) -> dict:
    return {
        "object_local_id": local_id,
        "timestamp": timestamp,
        "from_state": before,
        "to_state": after,
        "evidence": evidence(times=[timestamp]),
    }


def action(local_id: str, action_type: str, start: float, end: float, objects: list[str], outcome: str = "completed") -> dict:
    return {
        "local_id": local_id,
        "action_type": action_type,
        "description": action_type,
        "start": start,
        "end": end,
        "object_local_ids": objects,
        "outcome": outcome,
        "evidence": evidence(times=[start, end]),
    }


def step(local_id: str, name: str, objects: list[str], actions: list[str], start: float, end: float, expected: str) -> dict:
    return {
        "local_id": local_id,
        "name": name,
        "required_objects": objects,
        "expected_action": expected,
        "completion_condition": expected,
        "observed_result": expected,
        "action_local_ids": actions,
        "time_start": start,
        "time_end": end,
        "evidence": evidence(times=[start]),
    }


def dep(source: str, dest: str, category: str) -> dict:
    classification = "DIRECTLY_OBSERVED" if category == "necessary_physical" else "INFERRED"
    return {
        "from_step": source,
        "to_step": dest,
        "category": category,
        "rationale": category,
        "evidence": evidence(classification),
    }


def document(**parts) -> AnalysisDocument:
    payload = {
        "summary": parts.pop("summary", "Fixture procedure"),
        "fixture": True,
        "objects": parts.pop("objects", []),
        "observations": parts.pop("observations", []),
        "states": parts.pop("states", []),
        "transitions": parts.pop("transitions", []),
        "relationships": parts.pop("relationships", []),
        "actions": parts.pop("actions", []),
        "workflow_steps": parts.pop("steps", []),
        "dependencies": parts.pop("dependencies", []),
        "ambiguities": parts.pop("ambiguities", []),
    }
    return AnalysisDocument.model_validate(payload)


def assembly_demo() -> AnalysisDocument:
    left, right, bolt, washer, nut = "D_left", "D_right", "D_bolt", "D_washer", "D_nut"
    return document(
        summary="Join two plates with a shaft, ring, and clamp",
        objects=[
            obj(left, "left bracket", "plate on the left", "flat plate with one hole"),
            obj(right, "right bracket", "plate on the right", "flat plate with one hole"),
            obj(bolt, "bolt", "threaded shaft", "long threaded rod"),
            obj(washer, "washer", "flat ring", "flat annular ring"),
            obj(nut, "wingnut", "clamping fastener", "fastener with two tabs"),
        ],
        observations=[
            obs(item, 30, area=0.08 if "bracket" in item else 0.03)
            for item in (left, right, bolt, washer, nut)
        ],
        states=[
            state(left, "positioned", 0, 40),
            state(right, "positioned", 0, 40),
            state(bolt, "inserted", 8, 40),
            state(washer, "seated", 12, 40),
            state(nut, "secured", 16, 40),
        ],
        transitions=[
            transition(bolt, "free", "inserted", 8),
            transition(washer, "loose", "seated", 12),
            transition(nut, "loose", "secured", 16),
        ],
        relationships=[
            rel(left, right, "touching", 0, 40),
            rel(bolt, left, "inside", 8, 40),
            rel(washer, bolt, "around", 12, 40),
            rel(nut, bolt, "attached", 16, 40),
        ],
        actions=[
            action("A_position", "align", 0, 5, [left, right]),
            action("A_insert", "insert", 5, 10, [bolt, left]),
            action("A_washer", "place", 10, 14, [washer, bolt]),
            action("A_nut", "attach", 14, 18, [nut, bolt]),
        ],
        steps=[
            step("step_position", "Position plates", [left, right], ["A_position"], 0, 5, "plates touching"),
            step("step_insert", "Insert shaft", [bolt, left], ["A_insert"], 5, 10, "shaft inside plate"),
            step("step_washer", "Seat ring", [washer, bolt], ["A_washer"], 10, 14, "ring around shaft"),
            step("step_nut", "Secure clamp", [nut, bolt], ["A_nut"], 14, 18, "clamp attached to shaft"),
        ],
        dependencies=[
            dep("step_position", "step_insert", "necessary_physical"),
            dep("step_insert", "step_washer", "necessary_physical"),
            dep("step_insert", "step_nut", "necessary_physical"),
            dep("step_washer", "step_nut", "observed_temporal"),
        ],
    )


def _execution_clock() -> dict[str, tuple[float, float, float]]:
    """name -> action start, action end, relation onset. Relations last until 50."""
    return {
        "position": (0, 8, 0),
        "insert": (10, 18, 14),
        "washer": (20, 28, 22),
        "nut": (30, 38, 32),
    }


def assembly_execution(
    *,
    include_washer: bool = True,
    bolt_visibility: str = "visible",
    extra_actions: list[dict] | None = None,
) -> AnalysisDocument:
    left, right, bolt, washer, nut = "E_left", "E_right", "E_bolt", "E_washer", "E_nut"
    clock = _execution_clock()
    objects = [
        obj(left, "left bracket", "plate on the left", "flat plate with one hole"),
        obj(right, "right bracket", "plate on the right", "flat plate with one hole"),
        obj(bolt, "bolt", "threaded shaft", "long threaded rod"),
    ]
    if include_washer:
        objects.append(obj(washer, "washer", "flat ring", "flat annular ring"))
    objects.append(obj(nut, "wingnut", "clamping fastener", "fastener with two tabs"))
    present = [item["local_id"] for item in objects]
    observations = []
    for local_id in present:
        visibility = bolt_visibility if local_id == bolt else "visible"
        observations.append(obs(local_id, 16, "visible", 0.04))
        observations.append(obs(local_id, 45, visibility, 0.04))
    states = [
        state(left, "positioned", 0, 50),
        state(right, "positioned", 0, 50),
        state(bolt, "inserted", clock["insert"][2], 50),
        state(nut, "secured", clock["nut"][2], 50),
    ]
    transitions = [
        transition(bolt, "free", "inserted", clock["insert"][2]),
        transition(nut, "loose", "secured", clock["nut"][2]),
    ]
    relationships = [
        rel(left, right, "touching", 0, 50),
        rel(bolt, left, "inside", clock["insert"][2], 50),
        rel(nut, bolt, "attached", clock["nut"][2], 50),
    ]
    actions = [
        action("X_position", "align", *clock["position"][:2], [left, right]),
        action("X_insert", "insert", *clock["insert"][:2], [bolt, left]),
        action("X_nut", "attach", *clock["nut"][:2], [nut, bolt]),
    ]
    if include_washer:
        states.append(state(washer, "seated", clock["washer"][2], 50))
        transitions.append(transition(washer, "loose", "seated", clock["washer"][2]))
        relationships.append(rel(washer, bolt, "around", clock["washer"][2], 50))
        actions.append(action("X_washer", "place", *clock["washer"][:2], [washer, bolt]))
    if extra_actions:
        actions.extend(extra_actions)
    return document(
        objects=objects,
        observations=observations,
        states=states,
        transitions=transitions,
        relationships=relationships,
        actions=actions,
    )
