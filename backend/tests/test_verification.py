"""Learned-workflow verification. Fixtures are deterministic and do not call a provider."""

from __future__ import annotations

import os

from app.verification import learn_template, verify_execution

from tests.verification_fixtures import (
    action,
    assembly_demo,
    assembly_execution,
    dep,
    document,
    obj,
    obs,
    rel,
    step,
)

os.environ.setdefault("LEARNLOOP_WORKER_ENABLED", "0")
os.environ.setdefault("LEARNLOOP_AUTO_SEED", "0")


def _run(execution, proposals=None, strategy="hybrid", promote=True):
    template = learn_template(assembly_demo(), promote_supported=promote, accept=promote)
    report = verify_execution(
        template,
        execution,
        proposals=proposals,
        strategy=strategy,
        execution_video_id="execution-video",
        demonstration_video_id="demo-video",
    )
    return template, report


def _steps(report):
    return {item["step_id"]: item for item in report["steps"]}


def test_observed_order_is_not_promoted_to_a_requirement():
    template = learn_template(assembly_demo())
    assert all(step.required is False for step in template.steps)
    nut = next(step for step in template.steps if step.step_id == "step_nut")
    temporal = next(item for item in nut.dependencies if item.category == "observed_temporal")
    physical = next(item for item in nut.dependencies if item.category == "necessary_physical")
    assert temporal.constrains_order is False
    assert physical.constrains_order is True
    assert any("not prove" in note for note in template.notes)


def test_a_correct_execution_passes_with_different_instance_ids():
    template, report = _run(assembly_execution())
    steps = _steps(report)
    assert report["overall"] == "PASS", report["steps"]
    for name in ("step_position", "step_insert", "step_washer", "step_nut"):
        assert steps[name]["verdict"] == "PASS", (name, steps[name]["explanation"], steps[name]["predicates"])
        assert steps[name]["evidence"]
        assert all(item["evidence_class"] == "DIRECTLY_OBSERVED" for item in steps[name]["evidence"])
    sources = {object_id for role in template.roles for object_id in role.source_object_ids}
    mapped = {
        object_id
        for role in report["roles"]
        if role["status"] == "assigned"
        for object_id in role["object_local_ids"]
    }
    assert sources.isdisjoint(mapped)
    washer = next(role for role in report["roles"] if role["label"] == "washer")
    assert washer["object_local_ids"] == ["E_washer"]


def test_b_visible_omission_is_a_missing_component():
    _template, report = _run(assembly_execution(include_washer=False))
    washer = _steps(report)["step_washer"]
    assert report["overall"] == "FAIL", report["steps"]
    assert washer["verdict"] == "FAIL"
    assert washer["error_category"] == "MISSING_COMPONENT"
    assert washer["evidence"]
    assert washer["evidence"][0]["evidence_class"] == "DIRECTLY_OBSERVED"
    assert washer["evidence"][0]["timestamp"] is not None
    assert washer["review_required"] is False
    assert _steps(report)["step_nut"]["verdict"] == "PASS"


def test_c_occluded_anchor_cannot_verify_and_does_not_fail():
    _template, report = _run(assembly_execution(include_washer=False, bolt_visibility="occluded"))
    washer = _steps(report)["step_washer"]
    assert washer["verdict"] == "CANNOT_VERIFY", washer
    assert washer["error_category"] != "MISSING_COMPONENT"
    assert report["overall"] == "INCONCLUSIVE"
    assert all(step["verdict"] != "FAIL" for step in report["steps"])


def test_d_speed_change_does_not_fail_alignment():
    _template, report = _run(assembly_execution())
    washer = _steps(report)["step_washer"]
    assert report["overall"] == "PASS"
    assert washer["matched_action_ids"] == ["X_washer"]
    assert washer["execution_time"] != washer["demo_time_start"]
    assert all(step["error_category"] != "INCORRECT_ORDER" for step in report["steps"])
    assert "not errors" in report["timing_policy"].lower()


def test_e_independent_reorder_passes():
    demo = document(
        objects=[
            obj("D_red", "red block", "block on the left", "red cube"),
            obj("D_blue", "blue block", "block on the right", "blue cube"),
            obj("D_board", "board", "work surface", "flat board"),
        ],
        observations=[obs("D_red", 10), obs("D_blue", 10), obs("D_board", 10)],
        relationships=[
            rel("D_red", "D_board", "left_of", 1, 20),
            rel("D_blue", "D_board", "right_of", 5, 20),
        ],
        actions=[
            action("A_red", "place", 0, 2, ["D_red", "D_board"]),
            action("A_blue", "place", 4, 6, ["D_blue", "D_board"]),
        ],
        steps=[
            step("step_red", "Place red", ["D_red", "D_board"], ["A_red"], 0, 2, "red left of board"),
            step("step_blue", "Place blue", ["D_blue", "D_board"], ["A_blue"], 4, 6, "blue right of board"),
        ],
        dependencies=[dep("step_red", "step_blue", "observed_temporal")],
    )
    execution = document(
        objects=[
            obj("E_red", "red block", "block on the left", "red cube"),
            obj("E_blue", "blue block", "block on the right", "blue cube"),
            obj("E_board", "board", "work surface", "flat board"),
        ],
        observations=[obs("E_red", 12), obs("E_blue", 12), obs("E_board", 12)],
        relationships=[
            rel("E_blue", "E_board", "right_of", 1, 20),
            rel("E_red", "E_board", "left_of", 8, 20),
        ],
        actions=[
            action("X_blue", "place", 0, 2, ["E_blue", "E_board"]),
            action("X_red", "place", 6, 9, ["E_red", "E_board"]),
        ],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    report = verify_execution(template, execution)
    assert report["overall"] == "PASS", report["steps"]
    assert all(step["error_category"] != "INCORRECT_ORDER" for step in report["steps"])


def test_f_required_order_violation_fails():
    demo = document(
        objects=[
            obj("D_pin", "pin", "short shaft", "metal pin"),
            obj("D_block", "block", "base block", "square block"),
            obj("D_cap", "cap", "cover", "round cap"),
        ],
        observations=[obs("D_pin", 12), obs("D_block", 12), obs("D_cap", 12)],
        relationships=[
            rel("D_pin", "D_block", "inside", 4, 30),
            rel("D_cap", "D_pin", "attached", 8, 30),
        ],
        actions=[
            action("A_ready", "insert", 0, 5, ["D_pin", "D_block"]),
            action("A_lock", "attach", 5, 10, ["D_cap", "D_pin"]),
        ],
        steps=[
            step("step_ready", "Seat pin", ["D_pin", "D_block"], ["A_ready"], 0, 5, "pin inside block"),
            step("step_lock", "Lock cap", ["D_cap", "D_pin"], ["A_lock"], 5, 10, "cap attached to pin"),
        ],
        dependencies=[dep("step_ready", "step_lock", "necessary_physical")],
    )
    execution = document(
        objects=[
            obj("E_pin", "pin", "short shaft", "metal pin"),
            obj("E_block", "block", "base block", "square block"),
            obj("E_cap", "cap", "cover", "round cap"),
        ],
        observations=[obs("E_pin", 20), obs("E_block", 20), obs("E_cap", 20)],
        relationships=[
            rel("E_cap", "E_pin", "attached", 2, 30),
            rel("E_pin", "E_block", "inside", 12, 30),
        ],
        actions=[
            action("X_lock", "attach", 2, 6, ["E_cap", "E_pin"]),
            action("X_ready", "insert", 12, 18, ["E_pin", "E_block"]),
        ],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    report = verify_execution(template, execution)
    lock = _steps(report)["step_lock"]
    assert lock["verdict"] == "FAIL", lock
    assert lock["error_category"] == "INCORRECT_ORDER"
    assert report["overall"] == "FAIL"


def test_g_wrong_instance_in_the_relation_is_an_incorrect_component():
    demo = document(
        objects=[
            obj("D_post", "post", "vertical post", "round post"),
            obj("D_sleeve", "sleeve", "split tube", "split tube"),
        ],
        observations=[obs("D_post", 10), obs("D_sleeve", 10)],
        relationships=[rel("D_sleeve", "D_post", "around", 4, 20)],
        actions=[action("A_sleeve", "place", 2, 6, ["D_sleeve", "D_post"])],
        steps=[step("step_sleeve", "Seat sleeve", ["D_sleeve", "D_post"], ["A_sleeve"], 2, 6, "sleeve around post")],
    )
    execution = document(
        objects=[
            obj("E_post", "post", "vertical post", "round post"),
            obj("E_wedge", "wedge", "triangular block", "triangular block"),
        ],
        observations=[obs("E_post", 16), obs("E_wedge", 16)],
        relationships=[rel("E_wedge", "E_post", "around", 4, 20)],
        actions=[action("X_wedge", "place", 4, 10, ["E_wedge", "E_post"])],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    report = verify_execution(template, execution)
    sleeve = _steps(report)["step_sleeve"]
    assert sleeve["verdict"] == "FAIL", sleeve
    assert sleeve["error_category"] == "INCORRECT_COMPONENT"
    role = next(item for item in report["roles"] if item["label"] == "sleeve")
    assert role["status"] == "unresolved"


def test_a_reversed_directional_arrangement_is_a_misplacement_and_an_inverse_kind_is_support():
    demo = document(
        objects=[obj("a", "top block", "red", "red"), obj("b", "bottom block", "blue", "blue")],
        observations=[obs("a", 9), obs("b", 9)],
        relationships=[rel("a", "b", "above", 3, 9)],
        actions=[action("act", "attach", 2, 3, ["a", "b"])],
        steps=[step("s", "Stack", ["a", "b"], ["act"], 2, 3, "Stack the blocks")],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    objects = [obj("a", "top block", "red", "red"), obj("b", "bottom block", "blue", "blue")]
    seen = [obs("a", 9), obs("b", 9)]
    acts = [action("act", "attach", 2, 3, ["a", "b"])]
    swapped = document(objects=objects, observations=seen, relationships=[rel("b", "a", "above", 3, 9)], actions=acts)
    assert verify_execution(template, swapped)["steps"][0]["error_category"] == "INCORRECT_PLACEMENT"
    inverse = document(objects=objects, observations=seen, relationships=[rel("b", "a", "below", 3, 9)], actions=acts)
    assert verify_execution(template, inverse)["steps"][0]["verdict"] == "PASS"


def test_steps_done_in_another_order_are_matched_to_the_action_that_moved_their_part():
    # Seen on a real attempt: every action touching the shared base tied, so each step got another step's action.
    demo = document(
        objects=[obj("D_disc", "clear disc", "small clear disc", "clear"), obj("D_pin", "white pin", "short white pin", "white"), obj("D_body", "body", "white body", "white")],
        observations=[obs("D_disc", 10), obs("D_pin", 10), obs("D_body", 10)],
        relationships=[rel("D_disc", "D_body", "attached", 2, 20), rel("D_pin", "D_body", "attached", 6, 20)],
        actions=[action("A_disc", "place", 0, 2, ["D_disc", "D_body"]), action("A_pin", "place", 4, 6, ["D_pin", "D_body"])],
        steps=[
            step("step_disc", "Attach disc", ["D_disc", "D_body"], ["A_disc"], 0, 2, "disc attached to body"),
            step("step_pin", "Attach pin", ["D_pin", "D_body"], ["A_pin"], 4, 6, "pin attached to body"),
        ],
    )
    execution = document(
        objects=[obj("E_disc", "clear disc", "small clear disc", "clear"), obj("E_pin", "white pin", "short white pin", "white"), obj("E_body", "body", "white body", "white")],
        observations=[obs("E_disc", 12), obs("E_pin", 12), obs("E_body", 12)],
        relationships=[rel("E_pin", "E_body", "attached", 5, 20), rel("E_disc", "E_body", "attached", 9, 20)],
        actions=[
            action("X_body", "place", 0, 2, ["E_body"]),
            action("X_pin", "place", 3, 5, ["E_pin", "E_body"]),
            action("X_disc", "place", 7, 9, ["E_disc", "E_body"]),
        ],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    steps = _steps(verify_execution(template, execution))
    assert steps["step_disc"]["matched_action_ids"] == ["X_disc"]
    assert steps["step_pin"]["matched_action_ids"] == ["X_pin"]


def _never_seen_case():
    demo = document(
        objects=[obj("D_stud", "clear stud", "small clear stud", "clear"), obj("D_body", "body", "white body", "white")],
        observations=[obs("D_stud", 10), obs("D_body", 10)],
        relationships=[rel("D_stud", "D_body", "attached", 2, 20)],
        actions=[action("A_stud", "place", 0, 2, ["D_stud", "D_body"])],
        steps=[step("step_stud", "Attach stud", ["D_stud", "D_body"], ["A_stud"], 0, 2, "stud attached to body")],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    hidden = obs("E_stud", 12, visibility="occluded")
    execution = document(
        objects=[obj("E_stud", "clear stud", "small clear stud", "clear"), obj("E_body", "body", "white body", "white")],
        observations=[hidden, obs("E_body", 12)],
        relationships=[],
        actions=[action("X_body", "place", 0, 2, ["E_body"])],
    )
    return template, execution


def test_a_part_seen_nowhere_fails_unless_its_appearance_matched_or_the_work_was_not_shown():
    template, execution = _never_seen_case()
    step_result = _steps(verify_execution(template, execution))["step_stud"]
    assert step_result["verdict"] == "FAIL" and step_result["error_category"] == "MISSING_COMPONENT"
    # A box that matched the stud's learned appearance means it was there.
    assert _steps(verify_execution(template, execution, appearance_seen={"E_stud"}))["step_stud"]["verdict"] == "CANNOT_VERIFY"
    # If the part it goes on was never seen either, the recording did not show the work.
    execution.actions = []
    execution.observations = [item for item in execution.observations if item.object_local_id != "E_body"]
    assert _steps(verify_execution(template, execution))["step_stud"]["verdict"] == "CANNOT_VERIFY"


def test_both_parts_seen_nowhere_fail_as_missing_when_the_rest_of_the_attempt_was_shown():
    # Seen live: a figure's legs, torso, and head were never in the attempt, only a motorcycle, and every
    # "attached to" condition stayed undecided for lack of evidence.
    template, _ = _never_seen_case()
    execution = document(
        objects=[obj("E_stud", "clear stud", "small clear stud", "clear"), obj("E_body", "body", "white body", "white"), obj("E_bike", "motorcycle", "red motorcycle", "red")],
        observations=[obs("E_stud", 12, visibility="occluded"), obs("E_body", 12, visibility="occluded"), obs("E_bike", 12)],
        relationships=[],
        actions=[action("X_bike", "place", 0, 2, ["E_bike"])],
    )
    result = _steps(verify_execution(template, execution))["step_stud"]
    assert result["verdict"] == "FAIL" and result["error_category"] == "MISSING_COMPONENT" and result["basis"] == "not_observed"
    failed = next(item for item in result["predicates"] if item["status"] == "fail")
    assert "Neither the" in failed["note"] and failed["evidence"][0]["object_local_ids"] == ["E_body", "E_stud"]


def test_a_seen_part_cannot_end_up_against_a_part_seen_nowhere():
    # Seen live: the lock washer was never in the attempt, yet "nut touching lock washer" was left undecided and a
    # second Meta look reported a lock washer beside the nut.
    demo = document(
        objects=[obj("D_nut", "hex nut", "grey hex nut", "grey"), obj("D_lock", "split lock washer", "dark split lock washer ring", "dark")],
        observations=[obs("D_nut", 10), obs("D_lock", 10)],
        relationships=[rel("D_nut", "D_lock", "touching", 2, 20)],
        actions=[action("A_nut", "place", 0, 2, ["D_nut", "D_lock"])],
        steps=[step("step_nut", "Tighten nut", ["D_nut", "D_lock"], ["A_nut"], 0, 2, "nut touching lock washer")],
    )
    template = learn_template(demo, promote_supported=True, accept=True)
    execution = document(
        objects=[obj("E_nut", "hex nut", "grey hex nut", "grey"), obj("E_lock", "split lock washer", "dark split lock washer ring", "dark")],
        observations=[obs("E_nut", 12), obs("E_lock", 12, visibility="occluded")],
        relationships=[],
        actions=[action("X_nut", "place", 0, 2, ["E_nut"])],
    )
    result = _steps(verify_execution(template, execution))["step_nut"]
    assert result["verdict"] == "FAIL" and result["error_category"] == "MISSING_COMPONENT" and result["basis"] == "not_observed"
    failed = next(item for item in result["predicates"] if item["status"] == "fail")
    assert "was not seen at any point" in failed["note"] and failed["evidence"][0]["object_local_ids"] == ["E_lock"]
    # A box that matched the lock washer's learned appearance means it was there.
    assert _steps(verify_execution(template, execution, appearance_seen={"E_lock"}))["step_nut"]["verdict"] != "FAIL"


def test_a_single_part_seen_nowhere_fails_only_when_another_part_was_seen():
    from types import SimpleNamespace

    from app.verification.predicates import _never_seen

    predicate = SimpleNamespace(id="p1")
    stud = SimpleNamespace(object_local_ids=["E_stud"], label="clear stud")
    objects = [obj("E_stud", "clear stud", "small clear stud", "clear"), obj("E_body", "body", "white body", "white")]
    # Nothing in the recording was in view: it did not show the work, so the condition stays undecided.
    dark = document(objects=objects, observations=[obs("E_stud", 12, visibility="occluded"), obs("E_body", 12, visibility="absent")], relationships=[], actions=[])
    assert _never_seen(predicate, dark, stud, None, set(), "v") is None
    shown = document(objects=objects, observations=[obs("E_stud", 12, visibility="occluded"), obs("E_body", 12)], relationships=[], actions=[])
    assert _never_seen(predicate, shown, stud, None, set(), "v").status == "fail"
