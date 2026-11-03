"""Attempts that swap a part or change the order, shaped like recorded bolt-stack runs.

The attempt is labeled from the reference list, so a part the model judged to be
the same kind keeps its reference label and any other part gets its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.verification import learn_template, verify_execution
from app.verification.roles import same_kind

from tests.verification_fixtures import action, dep, document, obj, obs, rel, step

BOLT, FLAT, SPLIT, NUT = "gray threaded bolt", "silver flat washer", "dark split washer", "silver hex nut"


def _demo():
    return document(
        objects=[
            obj("D_bolt", BOLT, "bolt", "hexagonal head, threaded shaft"),
            obj("D_flat", FLAT, "flat ring", "silver flat ring"),
            obj("D_split", SPLIT, "split ring", "dark ring with a gap"),
            obj("D_nut", NUT, "nut", "six-sided nut"),
        ],
        observations=[obs(item, time) for item in ("D_bolt", "D_flat", "D_split", "D_nut") for time in (0, 15)],
        relationships=[
            rel("D_flat", "D_bolt", "around", 2, 15),
            rel("D_split", "D_bolt", "around", 4, 15),
            rel("D_split", "D_flat", "touching", 5, 15),
            rel("D_nut", "D_bolt", "attached", 6, 15),
            rel("D_nut", "D_split", "touching", 11, 15),
        ],
        actions=[
            action("A1", "place", 1, 3, ["D_flat", "D_bolt"]),
            action("A2", "place", 3, 5.2, ["D_split", "D_bolt"]),
            action("A3", "thread", 5.5, 13.5, ["D_nut", "D_bolt"]),
        ],
        steps=[
            step("step_1", "Place flat washer", ["D_flat", "D_bolt"], ["A1"], 1, 3, "flat ring on bolt"),
            step("step_2", "Place split washer", ["D_split", "D_bolt", "D_flat"], ["A2"], 3, 5.2, "split ring on flat ring"),
            step("step_3", "Thread nut", ["D_nut", "D_bolt", "D_split"], ["A3"], 5.5, 13.5, "nut on split ring"),
        ],
        dependencies=[dep("step_1", "step_2", "necessary_physical"), dep("step_2", "step_3", "necessary_physical")],
    )


def _template():
    return learn_template(_demo(), promote_supported=True, accept=True, results_first=True)


def _in_order(middle_label: str, middle_visual: str):
    """The demonstrated order, with the middle part labeled by the attempt's analysis. Hands leave parts partly visible at the end."""
    return document(
        objects=[
            obj("E_bolt", BOLT, "bolt", "hexagonal head, threaded shaft"),
            obj("E_flat", FLAT, "flat ring", "silver flat ring"),
            obj("E_mid", middle_label, "ring", middle_visual),
            obj("E_nut", NUT, "nut", "six-sided nut"),
        ],
        observations=[obs(item, 0) for item in ("E_bolt", "E_flat", "E_mid", "E_nut")]
        + [obs(item, 16, "partial") for item in ("E_bolt", "E_flat", "E_mid", "E_nut")],
        relationships=[
            rel("E_flat", "E_bolt", "around", 2, 16),
            rel("E_mid", "E_bolt", "around", 5, 16),
            rel("E_mid", "E_flat", "touching", 6, 16),
            rel("E_nut", "E_bolt", "attached", 8, 16),
            rel("E_nut", "E_mid", "touching", 12, 16),
        ],
        actions=[
            action("X1", "place", 1, 3, ["E_flat", "E_bolt"]),
            action("X2", "place", 4, 6, ["E_mid", "E_bolt"]),
            action("X3", "thread", 7, 14, ["E_nut", "E_bolt"]),
        ],
    )


def _steps(report):
    return {item["step_id"]: item for item in report["steps"]}


def test_a_part_out_of_order_is_still_that_part_and_the_order_fails():
    # Split washer, then nut, then flat washer last, as in the recorded run. The flat washer touches the nut, not the split washer.
    execution = document(
        objects=[
            obj("o1", BOLT, "bolt", "hexagonal head at lower left"),
            obj("o2", SPLIT, "split ring", "small circular split ring"),
            obj("o3", FLAT, "flat ring", "flat silver ring with central hole"),
            obj("o4", NUT, "nut", "six-sided metal nut"),
        ],
        observations=[obs(item, 0) for item in ("o1", "o2", "o3", "o4")]
        + [obs(item, 15.45, "partial") for item in ("o1", "o2", "o3", "o4")],
        relationships=[
            rel("o2", "o1", "around", 3.18, 15.45),
            rel("o4", "o1", "attached", 4.25, 15.45),
            rel("o4", "o2", "touching", 8.5, 15.45),
            rel("o3", "o1", "around", 11.7, 15.45),
            rel("o3", "o4", "touching", 13.83, 15.45, "INFERRED"),
        ],
        actions=[
            action("a1", "slide", 1.5, 3.2, ["o1", "o2"]),
            action("a2", "thread", 3.5, 8.5, ["o4", "o1", "o2"]),
            action("a3", "slide", 11.0, 11.7, ["o3", "o1"]),
            action("a4", "spin", 12.8, 14.83, ["o3", "o1", "o4"]),
        ],
    )
    report = verify_execution(_template(), execution)
    roles = {role["role_id"]: role for role in report["roles"]}
    assert all(role["status"] == "assigned" for role in roles.values()), roles
    assert roles["role:D_flat"]["object_local_ids"] == ["o3"]
    steps = _steps(report)
    assert steps["step_1"]["verdict"] == "PASS"
    assert steps["step_2"]["verdict"] == "FAIL" and steps["step_2"]["error_category"] == "INCORRECT_ORDER"
    assert report["overall"] == "FAIL"


def test_the_recorded_out_of_order_attempt_fails_on_order_instead_of_being_inconclusive():
    # A stored template and attempt. The flat washer kept its exact label but, sitting on the nut instead of the
    # split washer, scored under the threshold when relations counted toward identity, and every step after it went unverified.
    recorded = json.loads((Path(__file__).parent / "fixtures" / "bolt_stack_out_of_order.json").read_text())
    report = verify_execution(recorded["template"], recorded["execution"])
    roles = {role["role_id"]: role for role in report["roles"]}
    assert roles["role:obj_flat_washer"]["object_local_ids"] == ["o3"], roles["role:obj_flat_washer"]
    steps = _steps(report)
    assert steps["step_2"]["verdict"] == "FAIL" and steps["step_2"]["error_category"] == "INCORRECT_ORDER", steps["step_2"]
    assert report["overall"] == "FAIL"


def test_a_different_kind_of_part_in_its_place_is_an_incorrect_component():
    report = verify_execution(_template(), _in_order("dark flat washer", "dark flat ring"))
    roles = {role["role_id"]: role for role in report["roles"]}
    assert roles["role:D_split"]["status"] == "unresolved"
    assert "E_mid" not in [hypothesis["object_local_id"] for hypothesis in roles["role:D_split"]["hypotheses"]]
    step_2 = _steps(report)["step_2"]
    assert step_2["verdict"] == "FAIL" and step_2["error_category"] == "INCORRECT_COMPONENT", step_2
    assert report["overall"] == "FAIL"


def test_the_same_kind_of_part_in_another_color_passes():
    for label in (SPLIT, "silver split washer"):
        report = verify_execution(_template(), _in_order(label, "silver ring with a gap"))
        assert report["overall"] == "PASS", (label, _steps(report))


def test_a_word_left_out_is_the_same_part_and_a_word_replaced_is_not():
    assert same_kind("silver flat round washer", "silver flat washer")
    assert same_kind("dark split washer", "black split washer")
    assert same_kind("hex nut 2", "silver hex nut")
    assert not same_kind("dark split washer", "dark flat washer")
    assert not same_kind("hex nut", "wing nut")
