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


@pytest.fixture(autouse=True)
def _previous_face_models_by_default(monkeypatch):
    """
    Existing tests target the MediaPipe detector and the dlib embedder (they
    mock their internals). Tests of YuNet / SFace select them explicitly.
    """
    monkeypatch.setattr(config, "FACE_DETECTOR_BACKEND", "mediapipe")
    monkeypatch.setattr(config, "FACE_RECOGNITION_BACKEND", "dlib")


@pytest.fixture(autouse=True)
def _deterministic_liveness_test(monkeypatch):
    """
    One blink per test and an immediate liveness reset when the face leaves,
    so tests do not depend on random blink counts or wall-clock grace periods.
    Tests of those features set the values they need.
    """
    monkeypatch.setattr(config, "LIVENESS_MAX_BLINKS", config.LIVENESS_MIN_BLINKS)
    monkeypatch.setattr(config, "LIVENESS_FACE_LOST_GRACE", 0.0)
