"""Tests never reach a provider unless a test opens the gate itself."""

import os

os.environ["LEARNLOOP_RESEARCH_ROUTES"] = "1"
os.environ["LEARNLOOP_LIVE_INFERENCE"] = "0"

import pytest  # noqa: E402

# These modules replace the Meta client with a stub and test what the adapter sends.
_STUBBED_META = {"test_meta_contracts", "test_recovery_execution", "test_sam_fusion", "test_providers"}


@pytest.fixture(autouse=True)
def _open_gate_for_stubbed_clients(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] in _STUBBED_META:
        from app.config import settings

        monkeypatch.setattr(settings, "live_inference", True)


# The bodies stay as they were in the source repository. They run there. Here the
# code they exercise is gone: non-Meta providers and Whisper are not in this build.
_REMOVED_RUNTIME = {
    "test_recovery_execution.py::test_selecting_sam_does_not_change_geometry_before_segmentation",
    "test_recovery_execution.py::test_concept_is_the_default_and_a_legacy_run_stays_on_it",
    "test_recovery_execution.py::test_trusted_spatial_with_no_seed_makes_no_concept_call_and_says_so",
    "test_recovery_execution.py::test_hybrid_records_each_prompt_kind_separately",
    "test_sam_fusion.py::test_sam_configuration_requires_a_meta_key_and_keeps_the_primary_model",
    "test_audio.py::test_no_audio_track_transcribes_nothing",
    "test_audio.py::test_transcript_alignment_does_not_change_action_outcomes",
    "test_costs.py::test_each_provider_call_is_priced_on_the_project_and_dashboard",
    "test_grounding.py::test_combined_adds_no_call_and_grounded_reports_its_second",
    "test_grounding.py::test_perception_does_not_depend_on_the_provider",
    "test_manual_models.py::test_adding_a_model_twice_keeps_one_entry",
    "test_pipeline.py::test_native_video_rejected_for_openai",
    "test_pipeline.py::test_video_model_defaults_to_the_file_and_honors_frames",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.removeprefix("tests/") in _REMOVED_RUNTIME:
            item.add_marker(pytest.mark.skip(reason="Non-Meta or Whisper runtime removed from the submission build."))
