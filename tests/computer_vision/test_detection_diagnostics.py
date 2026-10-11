"""
Tests for the Phase 2B face detection diagnostics.

Verifies the instrumentation itself: that it is inert when disabled, that it
records the measurements the audit needs, and that it cannot break the
detection pipeline.
"""
import csv

import numpy as np
import pytest

# No heavy-dependency mocking is needed here: detection_diagnostics only
# imports numpy/cv2/config, and DetectedFace comes from face_detector
# (MediaPipe, which is installed). Deliberately NOT stubbing `fpdf` or
# `face_recognition` in sys.modules, because those stubs leak into other test
# modules in the same pytest session.
from src.computer_vision.detection_diagnostics import (
    DetectionDiagnostics,
    get_diagnostics,
)
from src.computer_vision.face_detector import DetectedFace


def face(px=150, conf=0.9):
    """Build a DetectedFace with a square bounding box."""
    return DetectedFace((100, 100, px, px), conf)


class TestDisabledByDefault:
    """Diagnostics must not affect normal behaviour."""

    def test_config_default_is_off(self):
        import config
        assert config.FACE_DIAGNOSTICS_ENABLED is False

    def test_disabled_records_nothing(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.record((480, 640, 3), [face()])
        assert diag.summary()["frames"] == 0

    def test_disabled_overlay_returns_input_unchanged(self):
        import numpy as np
        diag = DetectionDiagnostics(enabled=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert diag.overlay(frame, [face()]) is frame

    def test_disabled_stage_recording_is_inert(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.record_stage("recognition", (1, 2, 100, 100), embedding_ok=True, recognized=True)
        assert diag.stage_counts == {}

    def test_disabled_close_is_safe(self):
        DetectionDiagnostics(enabled=False).close()


class TestRecording:
    """Measurements must reflect what was actually detected."""

    def test_counts_detected_frame(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True  # enable without env
        diag.record((480, 640, 3), [face()])
        s = diag.summary()
        assert s["frames"] == 1
        assert s["detected_frames"] == 1
        assert s["missed_frames"] == 0

    def test_counts_missed_frame(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [])
        s = diag.summary()
        assert s["frames"] == 1
        assert s["detected_frames"] == 0
        assert s["missed_frames"] == 1

    def test_records_face_size_and_confidence(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [face(px=120, conf=0.83)])
        s = diag.summary()
        assert s["avg_face_px"] == 120
        assert s["max_face_px"] == 120
        assert round(s["avg_confidence"], 2) == 0.83

    def test_detect_rate_over_many_frames(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        for _ in range(8):
            diag.record((480, 640, 3), [face()])
        for _ in range(2):
            diag.record((480, 640, 3), [])
        s = diag.summary()
        assert s["frames"] == 10
        assert round(s["detect_rate"], 2) == 0.8

    def test_multiple_faces_in_one_frame(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [face(px=100), face(px=200)])
        s = diag.summary()
        assert s["min_face_px"] == 100
        assert s["max_face_px"] == 200

    def test_record_never_raises_on_bad_input(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [DetectedFace((1, 2), 0.5)])  # malformed bbox
        assert diag.summary()["frames"] == 1


class TestStageMeasurement:
    """Per-stage counters separate detection from recognition from liveness."""

    def test_recognition_success_recorded(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record_stage("recognition", (0, 0, 100, 100),
                          embedding_ok=True, recognized=True, distance=0.42)
        r = diag.stage_report()
        assert r["recognition"]["processed"] == 1
        assert r["recognition"]["success_rate"] == 1.0
        assert round(r["avg_recognition_distance"], 2) == 0.42

    def test_recognition_failure_recorded(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record_stage("recognition", (0, 0, 100, 100),
                          embedding_ok=True, recognized=False)
        assert diag.stage_report()["recognition"]["success_rate"] == 0.0

    def test_embedding_failure_counted(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record_stage("recognition", (0, 0, 100, 100), embedding_ok=False)
        assert diag.stage_report()["embedding_failures"] == 1

    def test_ear_extraction_failure_counted(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record_stage("liveness", (0, 0, 100, 100), ear=None)
        assert diag.stage_report()["ear_extraction_failures"] == 1

    def test_ear_value_recorded(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record_stage("liveness", (0, 0, 100, 100), ear=0.25)
        r = diag.stage_report()
        assert r["ear_extraction_failures"] == 0
        assert r["avg_ear"] == 0.25


class TestReporting:
    def test_report_contains_key_numbers(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [face(px=140, conf=0.88)])
        text = diag.report()
        assert "detection rate" in text
        assert "face size" in text
        assert "Downstream stages" in text

    def test_log_line_format(self):
        diag = DetectionDiagnostics(enabled=False)
        diag.enabled = True
        diag.record((480, 640, 3), [face(px=140)])
        line = DetectionDiagnostics.format_line(
            diag.summary(), 480, 640, 1, "detection"
        )
        assert "[face-diag]" in line
        assert "frame_size=640x480" in line


class TestCsvExport:
    def test_csv_written_and_closed(self, tmp_path):
        csv_file = tmp_path / "diag.csv"
        diag = DetectionDiagnostics(enabled=True, csv_path=str(csv_file))
        diag.record((480, 640, 3), [face(px=150, conf=0.9)])
        diag.record((480, 640, 3), [])
        diag.close()

        rows = list(csv.DictReader(csv_file.open(encoding="utf-8")))
        assert len(rows) == 2
        assert rows[0]["bbox_w"] == "150"
        assert rows[1]["faces_detected"] == "0"

    def test_csv_appends_not_truncates(self, tmp_path):
        csv_file = tmp_path / "diag.csv"
        first = DetectionDiagnostics(enabled=True, csv_path=str(csv_file))
        first.record((480, 640, 3), [face()])
        first.close()

        second = DetectionDiagnostics(enabled=True, csv_path=str(csv_file))
        second.record((480, 640, 3), [face()])
        second.close()

        rows = list(csv.DictReader(csv_file.open(encoding="utf-8")))
        assert len(rows) == 2

    def test_bad_csv_path_does_not_raise(self):
        diag = DetectionDiagnostics(enabled=True, csv_path="Z:/nope/dir/diag.csv")
        diag.record((480, 640, 3), [face()])  # must not raise
        diag.close()


class TestOverlay:
    def test_overlay_annotates_when_enabled(self):
        import numpy as np
        diag = DetectionDiagnostics(enabled=True, draw_overlay=True, warn_face_px=100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = diag.overlay(frame, [face(px=150)])
        assert out is not frame
        assert out.shape == frame.shape
        assert out.any()  # something was drawn

    def test_overlay_skipped_when_draw_disabled(self):
        import numpy as np
        diag = DetectionDiagnostics(enabled=True, draw_overlay=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert diag.overlay(frame, [face()]) is frame

    def test_overlay_does_not_mutate_input(self):
        import numpy as np
        diag = DetectionDiagnostics(enabled=True, draw_overlay=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        diag.overlay(frame, [face()])
        assert not frame.any()  # input untouched


class TestSharedInstance:
    def test_get_diagnostics_is_singleton(self):
        assert get_diagnostics() is get_diagnostics()


class TestDetectionIntegration:
    """The detector must keep working, with and without diagnostics."""

    @pytest.fixture
    def detector(self):
        """Create a real detector and always close it.

        Closing matters: MediaPipe keeps native threads alive, and an
        unclosed detector can stop pytest from exiting.
        """
        from src.computer_vision.face_detector import FaceDetector
        det = FaceDetector()
        try:
            yield det
        finally:
            det.close()

    def test_detector_detect_faces_still_returns_list(self, detector):
        out = detector.detect_faces(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(out, list)

    def test_detector_handles_invalid_input(self, detector):
        assert detector.detect_faces(None) == []
        assert detector.detect_faces(np.array([])) == []

    def test_confidence_threshold_from_config(self, detector):
        import config
        assert detector.min_detection_confidence == config.FACE_DETECTION_CONFIDENCE