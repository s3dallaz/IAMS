"""
Shared pytest fixtures.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def _no_spoof_models_or_snapshots(monkeypatch):
    """
    Keep tests independent of the optional spoof-detection model files and
    never write attendance snapshots into data/. Tests that cover these
    features turn them back on with monkeypatch and inject fakes.
    """
    monkeypatch.setattr(config, "ANTI_SPOOF_ENABLED", False)
    monkeypatch.setattr(config, "SCREEN_DETECTION_ENABLED", False)
    monkeypatch.setattr(config, "ATTENDANCE_SNAPSHOTS_ENABLED", False)
