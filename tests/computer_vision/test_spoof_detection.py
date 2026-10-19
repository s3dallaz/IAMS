"""
Tests for the spoof-detection helpers (anti-spoof model and screen detector).

The model-based smoke tests run only when the model files are present in the
project root; everything else uses synthetic frames and patched detections.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.computer_vision.spoof_detection import (
    ANTI_SPOOF_MODELS,
    SCREEN_MODEL_FILE,
    AntiSpoofDetector,
    ScreenDetector,
    crop_around_face,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class TestCropAroundFace:
    def test_crop_is_scaled_around_the_face_centre(self):
        crop = crop_around_face(FRAME, (270, 190, 100, 100), 2.0)
        assert crop.shape[:2] == (201, 201)

    def test_crop_at_the_frame_edge_is_shifted_inside_and_keeps_its_size(self):
        """Like Silent-Face's CropImage: the model always sees the full context."""
        crop = crop_around_face(FRAME, (0, 0, 100, 100), 2.7)
        assert crop.shape[:2] == (271, 271)

    def test_scale_shrinks_when_the_face_fills_the_frame(self):
        crop = crop_around_face(FRAME, (100, 40, 400, 400), 4.0)
        assert crop.shape[0] <= 480

    def test_empty_box_gives_no_crop(self):
        assert crop_around_face(FRAME, (10, 10, 0, 50), 2.7) is None


class TestAntiSpoofDetector:
    def test_missing_models_disable_the_check(self, tmp_path):
        detector = AntiSpoofDetector(model_dir=tmp_path)
        assert detector.available is False
        assert detector.real_score(FRAME, (270, 190, 100, 100)) is None

    @pytest.mark.skipif(
        not all((PROJECT_ROOT / name).exists() for name, _ in ANTI_SPOOF_MODELS),
        reason="anti-spoof model files not downloaded"
    )
    def test_real_models_return_a_probability(self):
        detector = AntiSpoofDetector()
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        score = detector.real_score(frame, (250, 150, 140, 170))
        assert detector.available is True
        assert 0.0 <= score <= 1.0


class TestScreenDetector:
    def test_missing_model_disables_the_check(self, tmp_path):
        detector = ScreenDetector(model_path=tmp_path / "missing.tflite")
        assert detector.available is False
        assert detector.detect_screens(FRAME) == []
        assert detector.screen_covering_face(FRAME, (10, 10, 50, 50)) is None

    @pytest.mark.parametrize("screen_box, expected", [
        ((0, 0, 200, 200), "cell phone"),   # face shown on the phone
        ((300, 0, 100, 200), None),         # phone held next to the face
        ((40, 10, 100, 100), None),         # covers only 40 % of the face
    ])
    def test_screen_must_cover_most_of_the_face(self, tmp_path, screen_box, expected):
        detector = ScreenDetector(model_path=tmp_path / "missing.tflite")
        with patch.object(detector, "detect_screens", return_value=[("cell phone", 0.8, screen_box)]):
            assert detector.screen_covering_face(FRAME, (10, 10, 50, 50)) == expected

    @pytest.mark.skipif(not (PROJECT_ROOT / SCREEN_MODEL_FILE).exists(),
                        reason="screen detector model not downloaded")
    def test_real_model_finds_no_screen_in_an_empty_frame(self):
        detector = ScreenDetector()
        assert detector.available is True
        assert detector.detect_screens(FRAME) == []
