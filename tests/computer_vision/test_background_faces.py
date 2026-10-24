"""
Tests for ignoring small background faces (config.BACKGROUND_FACE_MAX_RATIO).

The face in front is recognised, liveness-checked and marked while much
smaller faces behind it are ignored; two faces of similar size still block
marking, and the blink / head-turn landmarks must come from the face being
checked.
"""

from types import SimpleNamespace

import numpy as np

import config
from src.computer_vision.face_detector import DetectedFace, select_foreground_face
from src.computer_vision.liveness_detector import LivenessDetector


# ============================================================================
# select_foreground_face
# ============================================================================

def test_single_face_is_the_foreground():
    assert select_foreground_face([(0, 0, 50, 50)]) == 0


def test_no_faces():
    assert select_foreground_face([]) is None


def test_small_background_face_is_ignored():
    assert select_foreground_face([(0, 0, 40, 40), (100, 0, 200, 200)], 0.5) == 1


def test_similar_sized_faces_block():
    assert select_foreground_face([(0, 0, 150, 150), (200, 0, 200, 200)], 0.5) is None


def test_ratio_zero_turns_the_feature_off():
    assert select_foreground_face([(0, 0, 10, 10), (100, 0, 200, 200)], 0.0) is None


def test_default_ratio_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "BACKGROUND_FACE_MAX_RATIO", 0.0)
    assert select_foreground_face([(0, 0, 10, 10), (100, 0, 200, 200)]) is None


# ============================================================================
# Liveness landmarks: only the face being checked
# ============================================================================

def _face_landmarks(x0, y0, size, frame_w=1000, frame_h=1000):
    """A square of landmark points (normalised coordinates) covering one face."""
    return [SimpleNamespace(x=(x0 + dx) / frame_w, y=(y0 + dy) / frame_h)
            for dx in (0, size) for dy in (0, size)]


FRAME = np.zeros((1000, 1000, 3), dtype=np.uint8)
FRONT = _face_landmarks(400, 300, 300)      # big face in front
BEHIND = _face_landmarks(50, 50, 80)        # small face far behind
BESIDE = _face_landmarks(750, 300, 250)     # second person of similar size


def test_landmarks_of_the_front_face_are_used_despite_a_background_face():
    chosen = LivenessDetector._select_landmarks([BEHIND, FRONT], FRAME, (380, 280, 340, 340))
    assert chosen is FRONT


def test_two_people_of_similar_size_give_no_landmarks():
    assert LivenessDetector._select_landmarks([FRONT, BESIDE], FRAME, (380, 280, 340, 340)) is None


def test_landmarks_must_belong_to_the_face_being_checked():
    """The front face is not the one recognised (box elsewhere): its blink must not count."""
    assert LivenessDetector._select_landmarks([BEHIND, FRONT], FRAME, (0, 0, 200, 200)) is None


def test_single_face_landmarks_unchanged():
    assert LivenessDetector._select_landmarks([FRONT], FRAME, None) is FRONT
