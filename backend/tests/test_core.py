"""Sampling, prompts, schema checks, and SAM grammar that do not need a database."""

import re

from app.demo.scenarios import SCENARIOS, build_document
from app.pipeline.infer import _merge_instruction, _window_note, _windows
from app.pipeline.prompts import ACTIONS, FOUNDATIONAL, SCENE, TEMPLATES, WORKFLOW
from app.pipeline.validate import validate_document
from app.providers.sam import parse_sam_output
from app.schemas.analysis import AnalysisDocument
from app.schemas.enums import RECOGNIZED_ACTIONS
from app.video.probe import parse_presentation_times, snap_timestamps
from app.video.sampling import budget_indices, distinct_moments, plan_uniform, uniform_targets


def test_uniform_targets_use_duration_not_frame_index():
    times = uniform_targets(2.0, 2.0)
    assert times == [0.0, 0.5, 1.0, 1.5]


def test_presentation_times_keep_the_first_csv_field():
    stdout = "0.000000,,,,\nN/A\n21.935000,,,,\nnot-a-time\n"
    assert parse_presentation_times(stdout) == [0.0, 21.935]


def test_timestamps_snap_to_presentation_times():
    snapped = snap_timestamps([0.1, 1.0], [0.0, 0.5, 1.0])
    assert snapped[0] == 0.0
    assert snapped[-1] == 1.0


def test_plan_uniform_records_reason():
    frames = plan_uniform(1.0, 1.0, [0.0, 0.5, 1.0])
    assert [frame.timestamp for frame in frames] == [0.0, 1.0]
    assert all(frame.reason == "uniform" for frame in frames)


def test_closing_sample_stays_on_the_last_picture():
    pts = [float(index) for index in range(22)] + [21.935]
    frames = plan_uniform(21.951667, 1.0, pts)
    assert frames[-1].timestamp == 21.935


def test_one_hertz_grid_includes_a_partial_final_second():
    pts = [float(index) for index in range(22)] + [21.918]
    frames = plan_uniform(21.952, 1.0, pts)
    stamps = [frame.timestamp for frame in frames]
    assert stamps[0] == 0.0
    assert 21.0 in stamps
    assert stamps[-1] == 21.918


def test_exact_grid_endpoint_is_sampled():
    frames = plan_uniform(2.0, 2.0, [0.0, 0.5, 1.0, 1.5, 1.967])
    assert [frame.timestamp for frame in frames] == [0.0, 0.5, 1.0, 1.5, 1.967]


def test_short_tail_does_not_add_another_sample():
    frames = plan_uniform(1.02, 1.0, [0.0, 1.0])
    assert [frame.timestamp for frame in frames] == [0.0, 1.0]


def test_frames_two_milliseconds_apart_are_one_moment():
    class Frame:
        def __init__(self, timestamp: float) -> None:
            self.timestamp = timestamp

    frames = [Frame(float(second)) for second in range(22)]
    frames += [Frame(second + 0.002) for second in range(10, 22)]
    frames.append(Frame(21.935))
    kept = [frame.timestamp for frame in distinct_moments(frames)]
    assert 14.0 in kept
    assert 14.002 not in kept
    assert kept[-1] == 21.935
    assert len(kept) == 23


def test_frame_budget_keeps_the_closing_sample():
    indexes = budget_indices(50, 48)
    assert indexes[0] == 0
    assert indexes[-1] == 49
    assert len(indexes) <= 48


def test_prompts_do_not_name_the_hardware_task():
    banned = ["bolt", "washer", "nut", "hexagonal", "lock washer"]
    blob = FOUNDATIONAL.lower() + "\n".join(body.lower() for _name, body in TEMPLATES.values())
    for word in banned:
        assert word not in blob


def test_fixture_documents_validate_for_every_mode():
    for key, factory in SCENARIOS.items():
        scenario = factory()
        document = build_document(scenario)
        issues = validate_document(document, scenario.duration)
        errors = [issue for issue in issues if issue.severity == "error"]
        assert errors == [], (key, errors)
        assert document.fixture is True
        assert document.camera is not None
        assert document.camera.world_coordinates_supported is False
        assert document.coverage is not None
        assert document.coverage.modality == "sampled_frames"


def test_sam_box_grammar():
    line = (
        "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334;c=0.5|>"
        "<|mask;x=0;y=0;c=0.5;data=27,60,~abc|>"
    )
    records = parse_sam_output(line)
    assert len(records) == 1
    assert records[0].frame_index == 0
    assert records[0].x1 == 211
    assert records[0].confidence == 0.5
    assert records[0].mask_encoding == "lossless"


class _Frame:
    def __init__(self, timestamp: float):
        self.timestamp = timestamp


def _evidence() -> dict:
    return {
        "classification": "DIRECTLY_OBSERVED",
        "provenance": "AI_PREDICTION",
        "frame_timestamps": [0.0],
    }


def test_windows_repeat_two_boundary_frames():
    frames = [_Frame(float(index)) for index in range(23)]
    windows = _windows(frames, 20)
    assert [len(window) for window in windows] == [20, 5]
    assert [frame.timestamp for frame in windows[1][:2]] == [18.0, 19.0]
    assert windows[0][-2:] == windows[1][:2]
    assert windows[-1][-1].timestamp == 22.0


def test_detailed_sampling_still_overlaps_every_boundary():
    frames = [_Frame(index / 2) for index in range(45)]
    windows = _windows(frames, 20)
    assert len(windows) == 3
    assert windows[0][-2:] == windows[1][:2]
    assert windows[1][-2:] == windows[2][:2]
    assert windows[-1][-1].timestamp == 22.0


def test_short_sequence_stays_one_window():
    frames = [_Frame(float(index)) for index in range(20)]
    assert _windows(frames, 20) == [frames]


def test_window_note_names_the_shared_timestamps():
    frames = [_Frame(float(index)) for index in range(23)]
    windows = _windows(frames, 20)
    opening = _window_note(1, windows)
    assert "The opening frames at 18 s and 19 s repeat" in opening
    assert "fewer pieces" in opening
    assert "The supplied frames are at 18 s, 19 s, 20 s, 21 s, and 22 s." in opening
    assert "tight box around that object alone" in opening
    assert "color cast" in SCENE
    assert "hands, fingers, and arms" in SCENE
    assert "Never list the workspace as an object" in SCENE
    assert "a hand picks it up or moves it" in SCENE
    # Localization rules are generic. No object shape is named in the prompts.
    for text in (opening, SCENE):
        lowered = text.lower()
        assert not re.search(r"\brings?\b", lowered)
        assert "facet" not in lowered
        assert "head or top" not in lowered
    merge = _merge_instruction(windows)
    assert "18 s and 19 s" in merge
    assert "keep the earlier ids" in merge
    assert "Do not duplicate actions" in merge
    assert "Keep every observation" in merge
    assert "one box per object" in merge


def test_unknown_action_is_in_the_vocabulary_and_approach_is_offered():
    assert "unknown" in RECOGNIZED_ACTIONS
    assert "approach" in ACTIONS
    assert "say unknown rather than forcing one" in ACTIONS
    assert "one start and one end is one action" in ACTIONS
    assert "inserted state" in ACTIONS
    assert "Do not paraphrase that action as handle, bring, or inspect." in WORKFLOW
    assert "Do not keep an id alive" in FOUNDATIONAL
    assert "rotated or partly covered part" in SCENE
    assert "Do not make a step depend on itself" in WORKFLOW


def test_action_dependency_is_dropped_and_unknown_type_is_kept():
    document = AnalysisDocument.model_validate(
        {
            "objects": [{"local_id": "Object_1", "label": "part"}],
            "actions": [
                {
                    "local_id": "Action_13",
                    "action_type": "unknown",
                    "start": 20,
                    "end": 21,
                    "evidence": _evidence(),
                },
                {
                    "local_id": "Action_14",
                    "action_type": "move",
                    "start": 21,
                    "end": 22,
                    "evidence": _evidence(),
                },
            ],
            "workflow_steps": [
                {
                    "local_id": "Step_1",
                    "name": "Reorient the assembly",
                    "action_local_ids": ["Action_13", "Action_14"],
                    "evidence": _evidence(),
                }
            ],
            "dependencies": [
                {
                    "from_step": "Step_1",
                    "to_step": "Step_2",
                    "category": "observed_temporal",
                    "evidence": _evidence(),
                },
                {
                    "from_step": "Action_13",
                    "to_step": "Action_14",
                    "category": "observed_temporal",
                    "evidence": _evidence(),
                },
            ],
        }
    )
    issues = validate_document(document, 22.127)
    codes = [issue.code for issue in issues]
    assert "unrecognized_action" not in codes
    assert "dependency_names_action" in codes
    assert "missing_step" in codes
    assert [(item.from_step, item.to_step) for item in document.dependencies] == [("Step_1", "Step_2")]
    dropped = next(issue for issue in issues if issue.code == "dependency_names_action")
    assert dropped.severity == "warning"
    missing = next(issue for issue in issues if issue.code == "missing_step")
    assert missing.severity == "error"


def test_an_observation_timestamp_is_its_frame_reference():
    document = AnalysisDocument.model_validate(
        {
            "objects": [{"local_id": "Object_1", "label": "part"}],
            "observations": [
                {
                    "object_local_id": "Object_1",
                    "timestamp": 1.25,
                    "evidence": {
                        "classification": "DIRECTLY_OBSERVED",
                        "provenance": "AI_PREDICTION",
                    },
                }
            ],
        }
    )
    issues = validate_document(document, 10)
    assert not any(issue.code == "observation_without_evidence" for issue in issues)
    assert document.observations[0].evidence.frame_timestamps == [1.25]


def test_a_step_that_depends_on_itself_is_dropped():
    document = AnalysisDocument.model_validate(
        {
            "objects": [{"local_id": "Object_1", "label": "part"}],
            "workflow_steps": [
                {
                    "local_id": "Step_1",
                    "name": "Reposition the part",
                    "evidence": _evidence(),
                }
            ],
            "dependencies": [
                {
                    "from_step": "Step_1",
                    "to_step": "Step_1",
                    "category": "observed_temporal",
                    "rationale": "Single continuous workflow step.",
                    "evidence": _evidence(),
                }
            ],
        }
    )
    issues = validate_document(document, 21.952)
    codes = [issue.code for issue in issues]
    assert codes.count("self_dependency") == 1
    assert "cyclic_dependency" not in codes
    assert document.dependencies == []
    dropped = next(issue for issue in issues if issue.code == "self_dependency")
    assert dropped.severity == "warning"


def test_prompt_family_stays_visible_and_secrets_stay_hidden():
    from app.api.serialize import _public_config

    public = _public_config({
        "prompt_key": "scene",
        "api_key": "secret",
        "credentials": {"token": "nope"},
        "sampling_preset": "standard",
        "perception": {"preset": "B_temporal"},
    })
    assert public["prompt_key"] == "scene"
    assert public["perception"]["preset"] == "B_temporal"
    assert "api_key" not in public
    assert "credentials" not in public


def test_attempted_outcome_is_distinct_in_fixture():
    document = build_document(SCENARIOS["fixed_camera"]())
    outcomes = {action.local_id: action.outcome.value for action in document.actions}
    assert "attempted" in outcomes.values()
    assert "completed" in outcomes.values()
