"""Tests never reach a provider unless a test opens the gate itself."""

import os

os.environ["LEARNLOOP_LIVE_INFERENCE"] = "0"

import pytest  # noqa: E402

# These modules replace the Meta client with a stub and test what the adapter sends.
_STUBBED_META = {"test_meta_contracts", "test_providers"}


@pytest.fixture(autouse=True)
def _open_gate_for_stubbed_clients(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] in _STUBBED_META:
        from app.config import settings

        monkeypatch.setattr(settings, "live_inference", True)

