"""Read joins and removals from a part's own track.

A free part that is carried to a host and then stays with that host has been
joined to it. The join is an insertion when the part's separate visible extent
shrinks or ends at the host. It is an attachment when the part keeps its size
against the host, as a handle or an applique does. Any portion that is still
visible keeps its own box either way.

A joined part that later separates from the host and stays apart has been
removed or detached. From then on it is not joined. One part can be joined and
removed more than once. Proximity alone is not a join: the part has to have
been separate, then meet the host, and then stay with it. A short touch that
lifts away again, the way a tool is pressed on and lifted, is not a join.

A part that stays boxed with its host is still in view, so a model action that
already says what happened there, such as a placement or an attempted
insertion, stands. A removal always ends the joined relation and state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from app.perception.assignment import center, iou
from app.schemas.analysis import (
    ActionPrediction,
    AnalysisDocument,
    BBox,
    Evidence,
    RelationshipPrediction,
    StatePrediction,
    TransitionPrediction,
)
from app.schemas.enums import ActionOutcome, EvidenceClass, Provenance, RelationKind

CONTACT_GAP = 0.045
# A join starts only when the boxes overlap or nearly abut. Staying with the host
# uses the looser contact gap, so a box that wobbles does not end the join.
JOIN_GAP = 0.01
MOVE = 0.05
MIN_SIGHTINGS = 3
MIN_ABSENT = 2
# A joined part needs this many sightings apart from its host before it counts
# as removed. One stray box is not a removal.
MIN_APART = 2
# A join that later separates needs this many sightings with the host. A shorter
# contact is a touch.
MIN_HELD = 3
# A joined part whose visible area falls to this share of its free area is inside the host.
SHRINK = 0.75
BORDER = 0.02
INSIDE = RelationKind.INSIDE.value
ATTACHED = RelationKind.ATTACHED.value
# Actions whose wording is allowed to become a join or a removal. A pickup or a
# rotation that merely overlaps the interval stays what it is.
_REWRITABLE = {"insert", "attach", "approach", "move", "hold", "unknown", "align", "adjust", "push"}
_REMOVAL_REWRITABLE = {"remove", "detach", "pull", "move", "hold", "unknown", "twist", "adjust"}
# Actions that do not describe how a join or a removal turned out.
_JOIN_IGNORED = {"pick_up", "inspect", "motion"}
_REMOVAL_IGNORED = {"inspect", "motion", "put_down", "place", "insert", "attach"}
# A model action of another kind already describes a part that stays boxed with its host.
_UNDESCRIBED = {"approach", "move", "hold", "unknown"}
_JOINED_LABELS = {"inserted", "attached"}
_PHYSICAL = {INSIDE, ATTACHED, RelationKind.AROUND.value, RelationKind.SUPPORTED_BY.value}

_WORDS = {
    INSIDE: {"join": "insert", "joined": "inserted", "release": "remove", "released": "removed", "prep": "into"},
    ATTACHED: {"join": "attach", "joined": "attached", "release": "detach", "released": "detached", "prep": "to"},
}


@dataclass(frozen=True)
class Join:
    guest_id: str
    host_ids: tuple[str, ...]
    kind: str
    contact_time: float
    last_seen: float
    joined_from: float
    until: float
    released: float | None = None
    free_until: float | None = None


def apply_assembly(document: AnalysisDocument) -> AnalysisDocument:
    """Rewrite join and removal actions, and the steps that cite them, from the tracks."""
    joins = find_joins(document)
    if not joins:
        return document
    actions = list(document.actions)
    steps = list(document.workflow_steps)
    relationships = list(document.relationships)
    states = list(document.states)
    transitions = list(document.transitions)
    objects = list(document.objects)
    labels = {obj.local_id: obj.label or obj.local_id for obj in objects}
    used: set[str] = set()
    last: dict[str, Join] = {}
    for join in joins:
        words = _WORDS[join.kind]
        guest = labels.get(join.guest_id, join.guest_id)
        host = _host_phrase(join.host_ids, labels)
        statement = _join_statement(guest, host, join)
        evidence = _join_evidence(join, statement)
        # A removal keeps the model's own wording for the join when it already has one.
        told = None
        if join.released is not None:
            told = _narrated(actions, join.guest_id, join.contact_time, join.joined_from, _REWRITABLE, _JOIN_IGNORED, used)
        if told is not None:
            used.add(told.local_id)
        else:
            action = _rewrite_join_action(actions, join, guest, host, statement, evidence, used)
            used.add(action.local_id)
            if _vanished(join):
                completion = (
                    f"The {guest}'s separate visible extent ends at {host} "
                    f"and the {guest} does not return as its own part"
                )
            else:
                completion = f"The {guest} meets {host} and stays with it. The visible portion keeps its own box."
            _rewrite_steps(
                steps,
                action.local_id,
                f"{words['join'].capitalize()} the {guest} {words['prep']} {host}",
                completion,
                statement,
                evidence,
            )
        _ensure_relationship(relationships, join, statement, evidence)
        _ensure_state(states, transitions, join, statement, evidence)
        if join.released is not None:
            removal = _removal_statement(guest, host, join)
            removal_evidence = _removal_evidence(join, removal)
            told = _narrated(
                actions, join.guest_id, join.last_seen, join.released, _REMOVAL_REWRITABLE, _REMOVAL_IGNORED, used
            )
            if told is not None:
                used.add(told.local_id)
            else:
                action = _rewrite_removal_action(actions, join, host, removal, removal_evidence, used)
                used.add(action.local_id)
                _rewrite_steps(
                    steps,
                    action.local_id,
                    f"{words['release'].capitalize()} the {guest} from {host}",
                    f"The {guest} separates from {host} and stays apart.",
                    removal,
                    removal_evidence,
                )
            _end_joined_relations(relationships, join)
            _end_joined_states(states, join)
            _ensure_released_state(states, transitions, join, removal, removal_evidence)
        last[join.guest_id] = join
    for join in last.values():
        _set_final_state(objects, join, labels.get(join.guest_id, join.guest_id), _host_phrase(join.host_ids, labels))
    return document.model_copy(
        update={
            "actions": actions,
            "workflow_steps": steps,
            "relationships": relationships,
            "states": states,
            "transitions": transitions,
            "objects": objects,
        }
    )


def find_joins(document: AnalysisDocument) -> list[Join]:
    times = sorted(
        {
            round(item.timestamp, 3)
            for item in document.observations
            if item.geometry_origin not in {"vlm_estimated", "unobserved"}
        }
    )
    boxes = _boxes(document)
    found: list[Join] = []
    for guest_id, series in boxes.items():
        if len(series) < MIN_SIGHTINGS:
            continue
        joins = _guest_joins(document, boxes, guest_id, series, times)
        for index, join in enumerate(joins):
            if join.released is None:
                continue
            following = joins[index + 1].contact_time if index + 1 < len(joins) else times[-1]
            joins[index] = replace(join, free_until=max(following, join.released))
        found.extend(joins)
    return found


def _guest_joins(
    document: AnalysisDocument,
    boxes: dict[str, list[tuple[float, tuple[float, float, float, float]]]],
    guest_id: str,
    series: list[tuple[float, tuple[float, float, float, float]]],
    times: list[float],
) -> list[Join]:
    ordered = [_hosts_at(boxes, guest_id, timestamp, box) for timestamp, box in series]
    touching = [set(hosts) for hosts in ordered]
    close = [set(_hosts_at(boxes, guest_id, timestamp, box, JOIN_GAP)) for timestamp, box in series]
    # A part whose box ends at its last contact may never show a clean overlap.
    later = [timestamp for timestamp in times if timestamp > series[-1][0] + 1e-6]
    if len(later) >= MIN_ABSENT:
        close[-1] = touching[-1]
    joins: list[Join] = []
    # Hosts the part already touches when it is first seen were never seen apart from it.
    blocked = set(touching[0])
    anchor = 0
    index = 1
    while index < len(series):
        blocked &= touching[index]
        hosts = [host for host in ordered[index] if host in close[index] and host not in blocked]
        if not hosts or _distance(series[anchor][1], series[index][1]) < MOVE:
            index += 1
            continue
        contact = index
        while contact - 1 > anchor and any(host in touching[contact - 1] for host in hosts):
            contact -= 1
        held: list[int] = []
        apart: list[int] = []
        release: int | None = None
        for cursor in range(contact, len(series)):
            if any(host in touching[cursor] for host in hosts):
                held.append(cursor)
                apart = []
                continue
            apart.append(cursor)
            if len(apart) >= MIN_APART:
                release = apart[0]
                break
        if release is None and apart:
            release = apart[0]
        if release is None:
            join = _open_join(document, guest_id, hosts, series, times, anchor, contact, held)
            if join is not None:
                joins.append(join)
            break
        if len(held) >= MIN_HELD:
            until = series[held[-1]][0]
            joins.append(
                Join(
                    guest_id=guest_id,
                    host_ids=tuple(hosts),
                    kind=_kind(document, guest_id, hosts, series, anchor, contact, held, False, until),
                    contact_time=series[contact][0],
                    last_seen=until,
                    joined_from=series[held[1]][0],
                    until=until,
                    released=series[release][0],
                )
            )
        anchor = release
        blocked = set(touching[release])
        index = release + 1
    return joins


def _open_join(document, guest_id, hosts, series, times, anchor, contact, held) -> Join | None:
    """A join that is still in place when the part's track ends."""
    last_time, last_box = series[-1]
    if _on_border(last_box):
        return None
    later = [timestamp for timestamp in times if timestamp > last_time + 1e-6]
    # A box that stops after the part meets the host, or a box that continues
    # while the part stays with that host. Either one is the same join.
    if len(later) >= MIN_ABSENT:
        joined_from, until, vanished = later[0], later[-1], True
    elif len(held) >= MIN_HELD:
        joined_from, until, vanished = series[held[1]][0], last_time, False
        # The part is still in view, so a model action that already says what
        # happened, such as a placement or an attempted insertion, stands.
        if _narrated(document.actions, guest_id, series[contact][0], until, _UNDESCRIBED, _JOIN_IGNORED, set()):
            return None
    else:
        return None
    return Join(
        guest_id=guest_id,
        host_ids=tuple(hosts),
        kind=_kind(document, guest_id, hosts, series, anchor, contact, held, vanished, until),
        contact_time=series[contact][0],
        last_seen=last_time,
        joined_from=joined_from,
        until=until,
    )


def _kind(document, guest_id, hosts, series, anchor, contact, held, vanished: bool, until: float) -> str:
    """Inside when the visible part shrinks or ends at the host, attached when it keeps its size.

    A relationship or action the model already gave this pair decides it first,
    since the frames show more than the box size does.
    """
    claimed = _claimed_kind(document, guest_id, hosts, series[contact][0], until)
    if claimed is not None:
        return claimed
    if vanished:
        return INSIDE
    free = [_area(box) for _timestamp, box in series[anchor:contact]] or [_area(series[contact][1])]
    joined = [_area(series[index][1]) for index in held[1:]] or [_area(series[held[0]][1])]
    return INSIDE if _median(joined) <= SHRINK * _median(free) else ATTACHED


def _claimed_kind(document: AnalysisDocument, guest_id: str, hosts: list[str], start: float, end: float) -> str | None:
    for relation in document.relationships:
        kind = _value(relation.kind)
        if kind not in (INSIDE, ATTACHED) or relation.subject_local_id != guest_id:
            continue
        if relation.other_local_id in hosts and _overlaps(relation.time_start, relation.time_end, start, end):
            return kind
    for action in document.actions:
        if action.action_type not in ("insert", "attach") or guest_id not in action.object_local_ids:
            continue
        if _overlaps(action.start, action.end, start, end):
            return INSIDE if action.action_type == "insert" else ATTACHED
    return None


def _boxes(document: AnalysisDocument) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    """Measured boxes only. A model estimate is not evidence of contact.

    Rows from older documents carry no origin and are read as they were stored.
    """
    grouped: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for observation in document.observations:
        if observation.bbox is None or observation.visibility.value == "absent":
            continue
        if observation.geometry_origin in {"vlm_estimated", "unobserved"}:
            continue
        grouped.setdefault(observation.object_local_id, []).append(
            (round(observation.timestamp, 3), _tuple(observation.bbox))
        )
    for series in grouped.values():
        series.sort()
    return grouped


def _hosts_at(
    boxes: dict[str, list[tuple[float, tuple[float, float, float, float]]]],
    guest_id: str,
    timestamp: float,
    guest_box: tuple[float, float, float, float],
    limit: float = CONTACT_GAP,
) -> list[str]:
    found: list[tuple[float, str]] = []
    for other_id, series in boxes.items():
        if other_id == guest_id:
            continue
        other = _at(series, timestamp)
        if other is None:
            continue
        gap = _gap(guest_box, other)
        if gap <= limit or iou(guest_box, other) > 0:
            found.append((gap, other_id))
    found.sort()
    return [object_id for _gap_value, object_id in found]


def _at(
    series: list[tuple[float, tuple[float, float, float, float]]], timestamp: float
) -> tuple[float, float, float, float] | None:
    for time, box in series:
        if abs(time - timestamp) < 1e-6:
            return box
    return None


def _rewrite_join_action(
    actions: list[ActionPrediction],
    join: Join,
    guest: str,
    host: str,
    statement: str,
    evidence: Evidence,
    used: set[str],
) -> ActionPrediction:
    words = _WORDS[join.kind]
    around = join.last_seen if join.released is None else join.joined_from
    chosen = _matching_action(
        actions, join.guest_id, join.contact_time, join.joined_from, around, _REWRITABLE, words["join"], used
    )
    if _vanished(join):
        kept = f"The separate visible extent ends at {join.joined_from:.0f}s and does not return."
    else:
        kept = f"The {guest} stays with {host}, and the portion that is still visible keeps its own box."
    final_state = f"{words['joined'].capitalize()} {words['prep']} {host}. {kept}"
    if chosen is None:
        chosen = ActionPrediction(
            local_id=_fresh_id(actions, words["join"]),
            action_type=words["join"],
            description=statement,
            start=join.contact_time,
            end=join.joined_from,
            actor=None,
            object_local_ids=[join.guest_id, *join.host_ids],
            initial_state=f"{guest} separate from {host}",
            final_state=final_state,
            # Sustained contact is inferred from boxes. It does not confirm a finished join.
            outcome=ActionOutcome.UNKNOWN,
            uncertainty=0.5,
            evidence=evidence,
        )
        actions.append(chosen)
        return chosen
    ids = list(dict.fromkeys([*chosen.object_local_ids, join.guest_id, *join.host_ids]))
    updated = chosen.model_copy(
        update={
            "action_type": words["join"],
            "description": statement,
            "initial_state": chosen.initial_state or f"{guest} separate from {host}",
            "final_state": final_state,
            "outcome": chosen.outcome,
            "uncertainty": max(chosen.uncertainty, 0.5),
            "object_local_ids": ids,
            "evidence": evidence,
            "end": max(chosen.end, join.joined_from),
        }
    )
    actions[actions.index(chosen)] = updated
    return updated


def _rewrite_removal_action(
    actions: list[ActionPrediction],
    join: Join,
    host: str,
    statement: str,
    evidence: Evidence,
    used: set[str],
) -> ActionPrediction:
    words = _WORDS[join.kind]
    released = join.released if join.released is not None else join.until
    chosen = _matching_action(
        actions, join.guest_id, join.last_seen, released, released, _REMOVAL_REWRITABLE, words["release"], used
    )
    initial_state = f"{words['joined'].capitalize()} {words['prep']} {host}"
    final_state = f"Separate from {host}"
    if chosen is None:
        chosen = ActionPrediction(
            local_id=_fresh_id(actions, words["release"]),
            action_type=words["release"],
            description=statement,
            start=join.last_seen,
            end=released,
            actor=None,
            object_local_ids=[join.guest_id, *join.host_ids],
            initial_state=initial_state,
            final_state=final_state,
            outcome=ActionOutcome.UNKNOWN,
            uncertainty=0.5,
            evidence=evidence,
        )
        actions.append(chosen)
        return chosen
    ids = list(dict.fromkeys([*chosen.object_local_ids, join.guest_id, *join.host_ids]))
    updated = chosen.model_copy(
        update={
            "action_type": words["release"],
            "description": statement,
            "initial_state": initial_state,
            "final_state": final_state,
            "outcome": chosen.outcome,
            "uncertainty": max(chosen.uncertainty, 0.5),
            "object_local_ids": ids,
            "evidence": evidence,
            "end": max(chosen.end, released),
        }
    )
    actions[actions.index(chosen)] = updated
    return updated


def _matching_action(
    actions: list[ActionPrediction],
    guest_id: str,
    start: float,
    end: float,
    around: float,
    rewritable: set[str],
    preferred: str,
    used: set[str],
) -> ActionPrediction | None:
    ranked: list[tuple[float, ActionPrediction]] = []
    for action in actions:
        if action.local_id in used or guest_id not in action.object_local_ids:
            continue
        if action.action_type not in rewritable:
            continue
        overlap = min(action.end, end) - max(action.start, start)
        if overlap < -0.2 and not (action.start <= around <= action.end):
            continue
        preference = 0 if action.action_type == preferred else 1
        ranked.append((preference - max(overlap, 0), action))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _narrated(
    actions: list[ActionPrediction],
    guest_id: str,
    start: float,
    end: float,
    rewritable: set[str],
    ignored: set[str],
    used: set[str],
) -> ActionPrediction | None:
    """A model action that already says what happened in this interval."""
    for action in actions:
        if action.local_id in used or guest_id not in action.object_local_ids:
            continue
        if action.action_type in rewritable or action.action_type in ignored:
            continue
        if _overlaps(action.start, action.end, start, end):
            return action
    return None


def _rewrite_steps(steps, action_id: str, name: str, completion: str, statement: str, evidence: Evidence) -> None:
    for index, step in enumerate(steps):
        if action_id not in step.action_local_ids:
            continue
        steps[index] = step.model_copy(
            update={
                "name": name,
                "expected_action": name,
                "completion_condition": completion,
                "observed_result": statement,
                "uncertainty": max(step.uncertainty, 0.5),
                "evidence": evidence,
            }
        )


def _ensure_relationship(
    relationships: list[RelationshipPrediction], join: Join, statement: str, evidence: Evidence
) -> None:
    host_id = join.host_ids[0]
    kind = RelationKind(join.kind)
    for index, relation in enumerate(relationships):
        if relation.subject_local_id != join.guest_id or relation.other_local_id != host_id:
            continue
        current = _value(relation.kind)
        if current not in _PHYSICAL:
            continue
        if not _overlaps(relation.time_start, relation.time_end, join.contact_time - 0.5, join.until + 0.5):
            continue
        if current not in (INSIDE, ATTACHED):
            # The model's own reading of the join, such as a washer around a shaft, stays.
            return
        relationships[index] = relation.model_copy(
            update={
                "kind": kind,
                "time_start": join.contact_time,
                "time_end": join.until,
                "description": statement,
                "evidence": evidence,
            }
        )
        return
    relationships.append(
        RelationshipPrediction(
            subject_local_id=join.guest_id,
            other_local_id=host_id,
            kind=kind,
            time_start=join.contact_time,
            time_end=join.until,
            description=statement,
            evidence=evidence,
        )
    )


def _ensure_state(
    states: list[StatePrediction],
    transitions: list[TransitionPrediction],
    join: Join,
    statement: str,
    evidence: Evidence,
) -> None:
    label = _WORDS[join.kind]["joined"]
    start = min(join.last_seen, join.joined_from)
    for index, state in enumerate(states):
        if state.object_local_id != join.guest_id:
            continue
        if state.time_end < start - 1e-6 or state.time_start > join.until + 1e-6:
            continue
        text = f"{state.state_label} {state.description}".lower()
        if state.state_label in _JOINED_LABELS or any(
            token in text for token in ("not clearly", "not_clear", "unverif", "unknown", "hidden")
        ):
            previous = state.state_label
            time_end = join.until if join.released is not None else max(state.time_end, join.until)
            states[index] = state.model_copy(
                update={
                    "state_label": label,
                    "description": statement,
                    "time_start": min(state.time_start, join.joined_from),
                    "time_end": time_end,
                    "evidence": evidence,
                }
            )
            if previous != label:
                transitions.append(
                    TransitionPrediction(
                        object_local_id=join.guest_id,
                        timestamp=join.joined_from,
                        from_state=previous,
                        to_state=label,
                        description=statement,
                        evidence=evidence,
                        certainty=0.55,
                    )
                )
            return
    states.append(
        StatePrediction(
            object_local_id=join.guest_id,
            time_start=join.joined_from,
            time_end=join.until,
            state_label=label,
            description=statement,
            evidence=evidence,
        )
    )


def _end_joined_relations(relationships: list[RelationshipPrediction], join: Join) -> None:
    """A physical relation with the host that runs past the removal ends at the last sighting with it."""
    released = join.released if join.released is not None else join.until
    for index, relation in enumerate(relationships):
        if relation.subject_local_id != join.guest_id or relation.other_local_id not in join.host_ids:
            continue
        if _value(relation.kind) not in _PHYSICAL:
            continue
        if relation.time_start <= join.until + 1e-6 and relation.time_end > released - 1e-6:
            relationships[index] = relation.model_copy(update={"time_end": join.until})


def _end_joined_states(states: list[StatePrediction], join: Join) -> None:
    """A joined state that runs past the removal ends at the last sighting with the host."""
    released = join.released if join.released is not None else join.until
    for index, state in enumerate(states):
        if state.object_local_id != join.guest_id or state.state_label not in _JOINED_LABELS:
            continue
        if state.time_start <= join.until + 1e-6 and state.time_end > released - 1e-6:
            states[index] = state.model_copy(update={"time_end": join.until})


def _ensure_released_state(
    states: list[StatePrediction],
    transitions: list[TransitionPrediction],
    join: Join,
    statement: str,
    evidence: Evidence,
) -> None:
    words = _WORDS[join.kind]
    released = join.released if join.released is not None else join.until
    free_until = join.free_until if join.free_until is not None else released
    for index, state in enumerate(states):
        if state.object_local_id != join.guest_id or state.state_label != words["released"]:
            continue
        if _overlaps(state.time_start, state.time_end, released, free_until):
            states[index] = state.model_copy(
                update={"time_start": released, "time_end": free_until, "description": statement, "evidence": evidence}
            )
            return
    states.append(
        StatePrediction(
            object_local_id=join.guest_id,
            time_start=released,
            time_end=free_until,
            state_label=words["released"],
            description=statement,
            evidence=evidence,
        )
    )
    transitions.append(
        TransitionPrediction(
            object_local_id=join.guest_id,
            timestamp=released,
            from_state=words["joined"],
            to_state=words["released"],
            description=statement,
            evidence=evidence,
            certainty=0.55,
        )
    )


def _set_final_state(objects, join: Join, guest: str, host: str) -> None:
    words = _WORDS[join.kind]
    if _vanished(join):
        text = (
            f"{words['joined'].capitalize()} {words['prep']} {host} by {join.joined_from:.0f}s. "
            f"The {guest} was separate, met {host}, and its own visible extent then ended."
        )
    elif join.released is None:
        text = (
            f"{words['joined'].capitalize()} {words['prep']} {host} by {join.joined_from:.0f}s. "
            f"The {guest} stays with {host}, and the portion that is still visible keeps its own box."
        )
    else:
        text = (
            f"{words['released'].capitalize()} from {host} at {join.released:.0f}s. "
            f"The {guest} was {words['joined']} {words['prep']} {host} from {join.contact_time:.0f}s "
            f"through {join.until:.0f}s and is separate after that."
        )
    for index, obj in enumerate(objects):
        if obj.local_id == join.guest_id:
            objects[index] = obj.model_copy(update={"final_state": text})
            return


def _join_statement(guest: str, host: str, join: Join) -> str:
    words = _WORDS[join.kind]
    if join.last_seen > join.contact_time + 1e-6:
        seen = (
            f"The {guest} is separate, then meets {host} at {join.contact_time:.0f}s "
            f"and is still its own visible part at {join.last_seen:.0f}s. "
        )
    else:
        seen = f"The {guest} is separate, then meets {host} at {join.contact_time:.0f}s. "
    joined = f"It is {words['joined']} {words['prep']} {host}. "
    if _vanished(join):
        kept = (
            f"From {join.joined_from:.0f}s through {join.until:.0f}s it has no separate visible extent "
            "and does not return."
        )
    else:
        kept = f"It stays with {host} through {join.last_seen:.0f}s, and the visible portion keeps its own box."
    return seen + joined + kept


def _removal_statement(guest: str, host: str, join: Join) -> str:
    words = _WORDS[join.kind]
    released = join.released if join.released is not None else join.until
    return (
        f"The {guest} is {words['joined']} {words['prep']} {host} through {join.until:.0f}s, "
        f"then is separate from it at {released:.0f}s and stays apart. "
        f"It is {words['released']} and no longer {words['joined']}."
    )


def _host_phrase(host_ids: tuple[str, ...], labels: dict[str, str]) -> str:
    if len(host_ids) >= 2:
        return "the assembly"
    return f"the {labels.get(host_ids[0], host_ids[0])}"


def _join_evidence(join: Join, statement: str) -> Evidence:
    return _evidence([join.contact_time, join.last_seen, join.joined_from], join.contact_time, join.until, statement)


def _removal_evidence(join: Join, statement: str) -> Evidence:
    released = join.released if join.released is not None else join.until
    return _evidence([join.last_seen, released], join.last_seen, released, statement)


def _evidence(frames: list[float], start: float, end: float, statement: str) -> Evidence:
    return Evidence(
        classification=EvidenceClass.INFERRED,
        provenance=Provenance.DETERMINISTIC,
        note=statement,
        frame_timestamps=list(dict.fromkeys(frames)),
        time_start=start,
        time_end=end,
    )


def _fresh_id(actions: list[ActionPrediction], verb: str) -> str:
    taken = {action.local_id for action in actions}
    number = 1
    while f"Action_{verb}_{number}" in taken:
        number += 1
    return f"Action_{verb}_{number}"


def _vanished(join: Join) -> bool:
    """The part's own box ended at the host rather than continuing with it."""
    return join.released is None and join.until > join.last_seen + 1e-6


def _value(kind) -> str:
    return kind.value if hasattr(kind, "value") else str(kind)


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a <= end_b + 1e-6 and start_b <= end_a + 1e-6


def _distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def _area(box: tuple[float, float, float, float]) -> float:
    return box[2] * box[3]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _tuple(bbox: BBox) -> tuple[float, float, float, float]:
    return (bbox.x, bbox.y, bbox.w, bbox.h)


def _gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
    dy = max(0.0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
    return math.hypot(dx, dy)


def _on_border(box: tuple[float, float, float, float]) -> bool:
    return box[0] <= BORDER or box[1] <= BORDER or box[0] + box[2] >= 1 - BORDER or box[1] + box[3] >= 1 - BORDER
