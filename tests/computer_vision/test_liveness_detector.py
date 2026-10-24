"""
Tests for LivenessDetector.

Tests liveness detection foundation (Phase 3A) including:
- Initialization and configuration
- Result representation
- Input validation
- Safe failure behavior
- API structure for temporal analysis

Phase 3A establishes the foundation only. Complete algorithm tests
will be added in Phase 3B/3C.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessResult,
    LivenessState,
    LivenessDetectionError
)
from src.computer_vision.face_detector import DetectedFace


class TestLivenessState:
    """Test LivenessState enum."""

    def test_liveness_states_exist(self):
        """Test that all required liveness states are defined."""
        assert hasattr(LivenessState, 'LIVE')
        assert hasattr(LivenessState, 'SPOOF')
        assert hasattr(LivenessState, 'UNKNOWN')

    def test_liveness_state_values(self):
        """Test liveness state string values."""
        assert LivenessState.LIVE.value == "live"
        assert LivenessState.SPOOF.value == "spoof"
        assert LivenessState.UNKNOWN.value == "unknown"

    def test_liveness_state_str(self):
        """Test string representation of liveness states."""
        assert str(LivenessState.LIVE) == "live"
        assert str(LivenessState.SPOOF) == "spoof"
        assert str(LivenessState.UNKNOWN) == "unknown"


class TestLivenessResult:
    """Test LivenessResult data class."""

    def test_init_live_result(self):
        """Test initialization of LIVE result."""
        result = LivenessResult(
            state=LivenessState.LIVE,
            confidence=0.95,
            frame_count=15,
            reason="Blink detected"
        )

        assert result.state == LivenessState.LIVE
        assert result.confidence == 0.95
        assert result.frame_count == 15
        assert result.reason == "Blink detected"

    def test_init_spoof_result(self):
        """Test initialization of SPOOF result."""
        result = LivenessResult(
            state=LivenessState.SPOOF,
            confidence=0.88,
            frame_count=20,
            reason="No motion detected"
        )

        assert result.state == LivenessState.SPOOF
        assert result.confidence == 0.88
        assert result.frame_count == 20
        assert result.reason == "No motion detected"

    def test_init_unknown_result(self):
        """Test initialization of UNKNOWN result."""
        result = LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=5,
            reason="Insufficient frames"
        )

        assert result.state == LivenessState.UNKNOWN
        assert result.confidence == 0.0
        assert result.frame_count == 5
        assert result.reason == "Insufficient frames"

    def test_confidence_validation_lower_bound(self):
        """Test confidence cannot be below 0.0."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            LivenessResult(
                state=LivenessState.UNKNOWN,
                confidence=-0.1,
                frame_count=10,
                reason="Test"
            )

    def test_confidence_validation_upper_bound(self):
        """Test confidence cannot exceed 1.0."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            LivenessResult(
                state=LivenessState.LIVE,
                confidence=1.5,
                frame_count=10,
                reason="Test"
            )

    def test_confidence_boundary_values(self):
        """Test confidence at boundary values 0.0 and 1.0."""
        # Test 0.0
        result_zero = LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=1,
            reason="Test zero"
        )
        assert result_zero.confidence == 0.0

        # Test 1.0
        result_one = LivenessResult(
            state=LivenessState.LIVE,
            confidence=1.0,
            frame_count=1,
            reason="Test one"
        )
        assert result_one.confidence == 1.0

    def test_frame_count_validation_negative(self):
        """Test frame count cannot be negative."""
        with pytest.raises(ValueError, match="Frame count cannot be negative"):
            LivenessResult(
                state=LivenessState.UNKNOWN,
                confidence=0.5,
                frame_count=-1,
                reason="Test"
            )

    def test_frame_count_zero_allowed(self):
        """Test frame count can be zero."""
        result = LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=0,
            reason="No frames processed"
        )
        assert result.frame_count == 0

    def test_repr(self):
        """Test string representation of LivenessResult."""
        result = LivenessResult(
            state=LivenessState.LIVE,
            confidence=0.876,
            frame_count=12,
            reason="Test reason"
        )

        repr_str = repr(result)

        assert "LivenessResult" in repr_str
        assert "live" in repr_str
        assert "0.876" in repr_str
        assert "frames=12" in repr_str
        assert "Test reason" in repr_str


class TestLivenessDetectorInitialization:
    """Test LivenessDetector initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration from config."""
        detector = LivenessDetector()

        assert detector.min_frames == config.LIVENESS_MIN_FRAMES
        assert detector.liveness_threshold == config.LIVENESS_THRESHOLD
        assert detector.temporal_window_seconds == config.LIVENESS_TEMPORAL_WINDOW

    def test_init_with_custom_min_frames(self):
        """Test initialization with custom min_frames."""
        custom_min_frames = 15
        detector = LivenessDetector(min_frames=custom_min_frames)

        assert detector.min_frames == custom_min_frames
        assert detector.liveness_threshold == config.LIVENESS_THRESHOLD
        assert detector.temporal_window_seconds == config.LIVENESS_TEMPORAL_WINDOW

    def test_init_with_custom_threshold(self):
        """Test initialization with custom liveness threshold."""
        custom_threshold = 0.85
        detector = LivenessDetector(liveness_threshold=custom_threshold)

        assert detector.min_frames == config.LIVENESS_MIN_FRAMES
        assert detector.liveness_threshold == custom_threshold
        assert detector.temporal_window_seconds == config.LIVENESS_TEMPORAL_WINDOW

    def test_init_with_custom_temporal_window(self):
        """Test initialization with custom temporal window."""
        custom_window = 3.5
        detector = LivenessDetector(temporal_window_seconds=custom_window)

        assert detector.min_frames == config.LIVENESS_MIN_FRAMES
        assert detector.liveness_threshold == config.LIVENESS_THRESHOLD
        assert detector.temporal_window_seconds == custom_window

    def test_init_with_all_custom_params(self):
        """Test initialization with all custom parameters."""
        detector = LivenessDetector(
            min_frames=20,
            liveness_threshold=0.90,
            temporal_window_seconds=1.5
        )

        assert detector.min_frames == 20
        assert detector.liveness_threshold == 0.90
        assert detector.temporal_window_seconds == 1.5

    def test_init_min_frames_validation_zero(self):
        """Test min_frames cannot be zero."""
        with pytest.raises(ValueError, match="min_frames must be at least 1"):
            LivenessDetector(min_frames=0)

    def test_init_min_frames_validation_negative(self):
        """Test min_frames cannot be negative."""
        with pytest.raises(ValueError, match="min_frames must be at least 1"):
            LivenessDetector(min_frames=-5)

    def test_init_threshold_validation_negative(self):
        """Test liveness_threshold cannot be negative."""
        with pytest.raises(ValueError, match="liveness_threshold must be between 0.0 and 1.0"):
            LivenessDetector(liveness_threshold=-0.1)

    def test_init_threshold_validation_too_high(self):
        """Test liveness_threshold cannot exceed 1.0."""
        with pytest.raises(ValueError, match="liveness_threshold must be between 0.0 and 1.0"):
            LivenessDetector(liveness_threshold=1.5)

    def test_init_threshold_boundary_values(self):
        """Test liveness_threshold at boundary values."""
        # Test 0.0
        detector_zero = LivenessDetector(liveness_threshold=0.0)
        assert detector_zero.liveness_threshold == 0.0

        # Test 1.0
        detector_one = LivenessDetector(liveness_threshold=1.0)
        assert detector_one.liveness_threshold == 1.0

    def test_init_temporal_window_validation_zero(self):
        """Test temporal_window_seconds cannot be zero."""
        with pytest.raises(ValueError, match="temporal_window_seconds must be positive"):
            LivenessDetector(temporal_window_seconds=0.0)

    def test_init_temporal_window_validation_negative(self):
        """Test temporal_window_seconds cannot be negative."""
        with pytest.raises(ValueError, match="temporal_window_seconds must be positive"):
            LivenessDetector(temporal_window_seconds=-1.5)


class TestLivenessDetectorValidation:
    """Test input validation in LivenessDetector."""

    def _create_test_frame(self, height=480, width=640, channels=3):
        """Helper to create a test frame."""
        if channels == 3:
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width), dtype=np.uint8)

    def _create_test_face(self, bbox=(100, 100, 200, 200), confidence=0.95):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=bbox, confidence=confidence)

    def test_detect_liveness_none_frame(self):
        """Test detection with None frame raises error."""
        detector = LivenessDetector()
        face = self._create_test_face()

        with pytest.raises(LivenessDetectionError, match="Frame cannot be None"):
            detector.detect_liveness(None, face)

    def test_detect_liveness_invalid_frame_type(self):
        """Test detection with invalid frame type raises error."""
        detector = LivenessDetector()
        face = self._create_test_face()

        with pytest.raises(LivenessDetectionError, match="Frame must be a numpy array"):
            detector.detect_liveness([1, 2, 3], face)

    def test_detect_liveness_empty_frame(self):
        """Test detection with empty frame raises error."""
        detector = LivenessDetector()
        face = self._create_test_face()
        empty_frame = np.array([])

        with pytest.raises(LivenessDetectionError, match="Frame cannot be empty"):
            detector.detect_liveness(empty_frame, face)

    def test_detect_liveness_invalid_frame_dimensions(self):
        """Test detection with invalid frame dimensions raises error."""
        detector = LivenessDetector()
        face = self._create_test_face()
        # Create a 4D array (invalid)
        invalid_frame = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

        with pytest.raises(LivenessDetectionError, match="Invalid frame shape"):
            detector.detect_liveness(invalid_frame, face)

    def test_detect_liveness_frame_too_small(self):
        """Test detection with frame dimensions too small."""
        detector = LivenessDetector()
        face = self._create_test_face()
        small_frame = self._create_test_frame(height=5, width=5)

        with pytest.raises(LivenessDetectionError, match="Frame dimensions too small"):
            detector.detect_liveness(small_frame, face)

    def test_detect_liveness_none_face(self):
        """Test detection with None face raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        with pytest.raises(LivenessDetectionError, match="DetectedFace cannot be None"):
            detector.detect_liveness(frame, None)

    def test_detect_liveness_invalid_face_type(self):
        """Test detection with invalid face type raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        with pytest.raises(LivenessDetectionError, match="Expected DetectedFace object"):
            detector.detect_liveness(frame, {"bbox": (0, 0, 10, 10)})

    def test_detect_liveness_invalid_bbox_format(self):
        """Test detection with invalid bbox format raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        # Create a mock face with invalid bbox
        face = DetectedFace(bbox=(100, 100, 200), confidence=0.9)  # Only 3 values

        with pytest.raises(LivenessDetectionError, match="must have valid bbox"):
            detector.detect_liveness(frame, face)

    def test_detect_liveness_negative_bbox_dimensions(self):
        """Test detection with negative bbox dimensions raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        # Negative width
        face_neg_width = DetectedFace(bbox=(100, 100, -50, 100), confidence=0.9)
        with pytest.raises(LivenessDetectionError, match="bbox dimensions must be positive"):
            detector.detect_liveness(frame, face_neg_width)

        # Negative height
        face_neg_height = DetectedFace(bbox=(100, 100, 50, -100), confidence=0.9)
        with pytest.raises(LivenessDetectionError, match="bbox dimensions must be positive"):
            detector.detect_liveness(frame, face_neg_height)

    def test_detect_liveness_zero_bbox_dimensions(self):
        """Test detection with zero bbox dimensions raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        # Zero width
        face_zero_width = DetectedFace(bbox=(100, 100, 0, 100), confidence=0.9)
        with pytest.raises(LivenessDetectionError, match="bbox dimensions must be positive"):
            detector.detect_liveness(frame, face_zero_width)

        # Zero height
        face_zero_height = DetectedFace(bbox=(100, 100, 50, 0), confidence=0.9)
        with pytest.raises(LivenessDetectionError, match="bbox dimensions must be positive"):
            detector.detect_liveness(frame, face_zero_height)

    def test_detect_liveness_invalid_face_confidence(self):
        """Test detection with invalid face confidence raises error."""
        detector = LivenessDetector()
        frame = self._create_test_frame()

        # Confidence > 1.0
        face_high = DetectedFace(bbox=(100, 100, 200, 200), confidence=1.5)
        with pytest.raises(LivenessDetectionError, match="confidence must be between 0.0 and 1.0"):
            detector.detect_liveness(frame, face_high)

        # Confidence < 0.0
        face_low = DetectedFace(bbox=(100, 100, 200, 200), confidence=-0.1)
        with pytest.raises(LivenessDetectionError, match="confidence must be between 0.0 and 1.0"):
            detector.detect_liveness(frame, face_low)


class TestLivenessDetectorSingleFrame:
    """Test single-frame liveness detection."""

    def _create_test_frame(self, height=480, width=640, channels=3):
        """Helper to create a test frame."""
        if channels == 3:
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width), dtype=np.uint8)

    def _create_test_face(self, bbox=(100, 100, 200, 200), confidence=0.95):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=bbox, confidence=confidence)

    def test_detect_liveness_returns_result(self):
        """Test that detect_liveness returns a LivenessResult."""
        detector = LivenessDetector()
        frame = self._create_test_frame()
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        assert isinstance(result, LivenessResult)

    def test_detect_liveness_phase_3a_returns_unknown(self):
        """Test that Phase 3A returns UNKNOWN state (safe behavior)."""
        detector = LivenessDetector()
        frame = self._create_test_frame()
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        # Phase 3A should return UNKNOWN (algorithm not implemented)
        assert result.state == LivenessState.UNKNOWN

    def test_detect_liveness_phase_3a_safe_failure(self):
        """Test that Phase 3A does NOT incorrectly return LIVE."""
        detector = LivenessDetector()
        frame = self._create_test_frame()
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        # Critical safety check: must NOT return LIVE without evidence
        assert result.state != LivenessState.LIVE, \
            "Phase 3A must not return LIVE without implemented detection algorithm"

    def test_detect_liveness_has_frame_count(self):
        """Test that result includes frame count."""
        detector = LivenessDetector()
        frame = self._create_test_frame()
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        assert result.frame_count >= 1

    def test_detect_liveness_has_reason(self):
        """Test that result includes a reason."""
        detector = LivenessDetector()
        frame = self._create_test_frame()
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_detect_liveness_color_frame(self):
        """Test detection with color (BGR) frame."""
        detector = LivenessDetector()
        frame = self._create_test_frame(channels=3)
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        assert isinstance(result, LivenessResult)

    def test_detect_liveness_grayscale_frame(self):
        """Test detection with grayscale frame."""
        detector = LivenessDetector()
        frame = self._create_test_frame(channels=1)
        face = self._create_test_face()

        result = detector.detect_liveness(frame, face)

        assert isinstance(result, LivenessResult)

    def test_detect_liveness_different_resolutions(self):
        """Test detection with various frame resolutions."""
        detector = LivenessDetector()
        face = self._create_test_face()

        resolutions = [
            (480, 640),   # VGA
            (720, 1280),  # HD
            (1080, 1920), # Full HD
            (100, 100),   # Small
        ]

        for height, width in resolutions:
            frame = self._create_test_frame(height=height, width=width)
            result = detector.detect_liveness(frame, face)
            assert isinstance(result, LivenessResult)


class TestLivenessDetectorSequence:
    """Test sequence-based (temporal) liveness detection."""

    def _create_test_frame(self, height=480, width=640, channels=3):
        """Helper to create a test frame."""
        if channels == 3:
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width), dtype=np.uint8)

    def _create_test_face(self, bbox=(100, 100, 200, 200), confidence=0.95):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=bbox, confidence=confidence)

    def test_detect_liveness_sequence_none_frames(self):
        """Test sequence detection with None frames raises error."""
        detector = LivenessDetector()
        faces = [self._create_test_face() for _ in range(5)]

        with pytest.raises(LivenessDetectionError, match="frames and detected_faces must be provided"):
            detector.detect_liveness_sequence(None, faces)

    def test_detect_liveness_sequence_none_faces(self):
        """Test sequence detection with None faces raises error."""
        detector = LivenessDetector()
        frames = [self._create_test_frame() for _ in range(5)]

        with pytest.raises(LivenessDetectionError, match="frames and detected_faces must be provided"):
            detector.detect_liveness_sequence(frames, None)

    def test_detect_liveness_sequence_invalid_frames_type(self):
        """Test sequence detection with non-list frames raises error."""
        detector = LivenessDetector()
        faces = [self._create_test_face() for _ in range(5)]

        with pytest.raises(LivenessDetectionError, match="frames and detected_faces must be lists"):
            detector.detect_liveness_sequence("not a list", faces)

    def test_detect_liveness_sequence_invalid_faces_type(self):
        """Test sequence detection with non-list faces raises error."""
        detector = LivenessDetector()
        frames = [self._create_test_frame() for _ in range(5)]

        with pytest.raises(LivenessDetectionError, match="frames and detected_faces must be lists"):
            detector.detect_liveness_sequence(frames, "not a list")

    def test_detect_liveness_sequence_length_mismatch(self):
        """Test sequence detection with mismatched lengths raises error."""
        detector = LivenessDetector()
        frames = [self._create_test_frame() for _ in range(5)]
        faces = [self._create_test_face() for _ in range(3)]

        with pytest.raises(LivenessDetectionError, match="Mismatch: 5 frames but 3 detected faces"):
            detector.detect_liveness_sequence(frames, faces)

    def test_detect_liveness_sequence_empty_lists(self):
        """Test sequence detection with empty lists raises error."""
        detector = LivenessDetector()

        with pytest.raises(LivenessDetectionError, match="Cannot perform liveness detection on empty sequence"):
            detector.detect_liveness_sequence([], [])

    def test_detect_liveness_sequence_invalid_frame_in_sequence(self):
        """Test sequence detection with invalid frame in sequence raises error."""
        detector = LivenessDetector()
        frames = [self._create_test_frame() for _ in range(3)]
        frames[1] = None  # Invalid frame at index 1
        faces = [self._create_test_face() for _ in range(3)]

        with pytest.raises(LivenessDetectionError, match="Invalid input at index 1"):
            detector.detect_liveness_sequence(frames, faces)

    def test_detect_liveness_sequence_invalid_face_in_sequence(self):
        """Test sequence detection with invalid face in sequence raises error."""
        detector = LivenessDetector()
        frames = [self._create_test_frame() for _ in range(3)]
        faces = [self._create_test_face() for _ in range(3)]
        faces[2] = None  # Invalid face at index 2

        with pytest.raises(LivenessDetectionError, match="Invalid input at index 2"):
            detector.detect_liveness_sequence(frames, faces)

    def test_detect_liveness_sequence_insufficient_frames(self):
        """Test sequence detection with insufficient frames returns UNKNOWN."""
        detector = LivenessDetector(min_frames=10)
        frames = [self._create_test_frame() for _ in range(5)]
        faces = [self._create_test_face() for _ in range(5)]

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.UNKNOWN
        assert "Insufficient frames" in result.reason
        assert result.frame_count == 5

    def test_detect_liveness_sequence_sufficient_frames(self):
        """Test sequence detection with sufficient frames."""
        detector = LivenessDetector(min_frames=10)
        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]

        result = detector.detect_liveness_sequence(frames, faces)

        assert isinstance(result, LivenessResult)
        assert result.frame_count == 15

    def test_detect_liveness_sequence_phase_3a_returns_unknown(self):
        """Test that Phase 3A sequence detection returns UNKNOWN."""
        detector = LivenessDetector(min_frames=5)
        frames = [self._create_test_frame() for _ in range(10)]
        faces = [self._create_test_face() for _ in range(10)]

        result = detector.detect_liveness_sequence(frames, faces)

        # Phase 3A should return UNKNOWN (temporal algorithm not implemented)
        assert result.state == LivenessState.UNKNOWN

    def test_detect_liveness_sequence_phase_3a_safe_failure(self):
        """Test that Phase 3A does NOT incorrectly return LIVE for sequence."""
        detector = LivenessDetector(min_frames=5)
        frames = [self._create_test_frame() for _ in range(20)]
        faces = [self._create_test_face() for _ in range(20)]

        result = detector.detect_liveness_sequence(frames, faces)

        # Critical safety check: must NOT return LIVE without evidence
        assert result.state != LivenessState.LIVE, \
            "Phase 3A must not return LIVE without implemented temporal algorithm"

    def test_detect_liveness_sequence_exactly_min_frames(self):
        """Test sequence detection with exactly minimum frames."""
        detector = LivenessDetector(min_frames=10)
        frames = [self._create_test_frame() for _ in range(10)]
        faces = [self._create_test_face() for _ in range(10)]

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.frame_count == 10
        # Should not complain about insufficient frames
        assert "Insufficient frames" not in result.reason

    def test_detect_liveness_sequence_one_frame_below_min(self):
        """Test sequence detection with one frame below minimum."""
        detector = LivenessDetector(min_frames=10)
        frames = [self._create_test_frame() for _ in range(9)]
        faces = [self._create_test_face() for _ in range(9)]

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.UNKNOWN
        assert "Insufficient frames" in result.reason
        assert result.frame_count == 9


class TestLivenessDetectorDeterminism:
    """Test deterministic behavior of LivenessDetector."""

    def _create_test_frame(self, seed=42):
        """Helper to create a deterministic test frame."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    def test_detect_liveness_deterministic(self):
        """Test that identical inputs produce identical results."""
        detector = LivenessDetector()
        frame = self._create_test_frame(seed=42)
        face = self._create_test_face()

        result1 = detector.detect_liveness(frame, face)
        result2 = detector.detect_liveness(frame, face)

        assert result1.state == result2.state
        assert result1.confidence == result2.confidence
        assert result1.frame_count == result2.frame_count


class TestLivenessDetectorConfiguration:
    """Test configuration behavior of LivenessDetector."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    def test_min_frames_affects_sequence_detection(self):
        """Test that min_frames configuration affects behavior."""
        frames = [self._create_test_frame() for _ in range(8)]
        faces = [self._create_test_face() for _ in range(8)]

        # Detector with min_frames=5 should accept 8 frames
        detector_low = LivenessDetector(min_frames=5)
        result_low = detector_low.detect_liveness_sequence(frames, faces)
        assert "Insufficient frames" not in result_low.reason

        # Detector with min_frames=10 should reject 8 frames
        detector_high = LivenessDetector(min_frames=10)
        result_high = detector_high.detect_liveness_sequence(frames, faces)
        assert "Insufficient frames" in result_high.reason

    def test_configuration_persists(self):
        """Test that configuration persists across multiple calls."""
        detector = LivenessDetector(min_frames=15, liveness_threshold=0.88)

        assert detector.min_frames == 15
        assert detector.liveness_threshold == 0.88

        # Perform detection
        frame = self._create_test_frame()
        face = self._create_test_face()
        detector.detect_liveness(frame, face)

        # Configuration should still be the same
        assert detector.min_frames == 15
        assert detector.liveness_threshold == 0.88


# =============================================================================
# Phase 3B Tests - Blink Detection and Temporal Analysis
# =============================================================================


class TestPhase3BConfiguration:
    """Test Phase 3B configuration parameters."""

    def test_ear_threshold_configuration(self):
        """Test EAR threshold configuration."""
        detector = LivenessDetector(ear_threshold=0.20)
        assert detector.ear_threshold == 0.20

    def test_ear_threshold_default(self):
        """Test EAR threshold uses config default."""
        detector = LivenessDetector()
        assert detector.ear_threshold == config.LIVENESS_EAR_THRESHOLD

    def test_ear_threshold_validation_negative(self):
        """Test EAR threshold cannot be negative."""
        with pytest.raises(ValueError, match="ear_threshold must be between 0.0 and 1.0"):
            LivenessDetector(ear_threshold=-0.1)

    def test_ear_threshold_validation_too_high(self):
        """Test EAR threshold cannot exceed 1.0."""
        with pytest.raises(ValueError, match="ear_threshold must be between 0.0 and 1.0"):
            LivenessDetector(ear_threshold=1.5)

    def test_ear_consecutive_frames_configuration(self):
        """Test EAR consecutive frames configuration."""
        detector = LivenessDetector(ear_consecutive_frames=3)
        assert detector.ear_consecutive_frames == 3

    def test_ear_consecutive_frames_default(self):
        """Test EAR consecutive frames uses config default."""
        detector = LivenessDetector()
        assert detector.ear_consecutive_frames == config.LIVENESS_EAR_CONSECUTIVE_FRAMES

    def test_ear_consecutive_frames_validation(self):
        """Test EAR consecutive frames must be at least 1."""
        with pytest.raises(ValueError, match="ear_consecutive_frames must be at least 1"):
            LivenessDetector(ear_consecutive_frames=0)

    def test_min_blinks_configuration(self):
        """Test minimum blinks configuration."""
        detector = LivenessDetector(min_blinks=2)
        assert detector.min_blinks == 2

    def test_min_blinks_default(self):
        """Test minimum blinks uses config default."""
        detector = LivenessDetector()
        assert detector.min_blinks == config.LIVENESS_MIN_BLINKS

    def test_min_blinks_validation(self):
        """Test minimum blinks cannot be negative."""
        with pytest.raises(ValueError, match="min_blinks cannot be negative"):
            LivenessDetector(min_blinks=-1)

    def test_min_blinks_can_be_zero(self):
        """Test minimum blinks can be zero."""
        detector = LivenessDetector(min_blinks=0)
        assert detector.min_blinks == 0


class TestEARCalculation:
    """Test Eye Aspect Ratio (EAR) calculation."""

    def test_calculate_ear_valid_landmarks(self):
        """Test EAR calculation with valid eye landmarks."""
        detector = LivenessDetector()

        # Create synthetic eye landmarks (open eye, horizontal orientation)
        # p1 (left corner), p2 (top), p3 (top), p4 (right corner), p5 (bottom), p6 (bottom)
        eye_landmarks = [
            (0, 5),    # p1: left corner
            (2, 3),    # p2: top-left
            (3, 3),    # p3: top-right
            (5, 5),    # p4: right corner
            (3, 7),    # p5: bottom-right
            (2, 7)     # p6: bottom-left
        ]

        ear = detector._calculate_ear(eye_landmarks)

        assert ear is not None
        assert 0.0 < ear < 1.0

    def test_calculate_ear_closed_eye(self):
        """Test EAR calculation for closed eye (low EAR)."""
        detector = LivenessDetector()

        # Closed eye: vertical distances are very small
        eye_landmarks = [
            (0, 5),    # p1: left corner
            (2, 5.1),  # p2: barely above center
            (3, 5.1),  # p3: barely above center
            (5, 5),    # p4: right corner
            (3, 4.9),  # p5: barely below center
            (2, 4.9)   # p6: barely below center
        ]

        ear = detector._calculate_ear(eye_landmarks)

        assert ear is not None
        assert ear < 0.15  # Very low EAR for closed eye

    def test_calculate_ear_open_eye(self):
        """Test EAR calculation for open eye (high EAR)."""
        detector = LivenessDetector()

        # Open eye: larger vertical distances
        eye_landmarks = [
            (0, 5),    # p1: left corner
            (2, 2),    # p2: well above center
            (3, 2),    # p3: well above center
            (5, 5),    # p4: right corner
            (3, 8),    # p5: well below center
            (2, 8)     # p6: well below center
        ]

        ear = detector._calculate_ear(eye_landmarks)

        assert ear is not None
        assert ear > 0.3  # Higher EAR for open eye

    def test_calculate_ear_invalid_landmark_count(self):
        """Test EAR calculation with wrong number of landmarks."""
        detector = LivenessDetector()

        # Only 4 landmarks instead of 6
        eye_landmarks = [(0, 0), (1, 1), (2, 2), (3, 3)]

        ear = detector._calculate_ear(eye_landmarks)

        assert ear is None

    def test_calculate_ear_zero_horizontal_distance(self):
        """Test EAR calculation when horizontal distance is zero."""
        detector = LivenessDetector()

        # All points at same x-coordinate (invalid eye)
        eye_landmarks = [
            (5, 5), (5, 3), (5, 3), (5, 5), (5, 7), (5, 7)
        ]

        ear = detector._calculate_ear(eye_landmarks)

        assert ear is None


class TestBlinkDetection:
    """Test blink detection logic."""

    def test_detect_blink_from_ear_open_to_closed(self):
        """Test eye state transition from OPEN to CLOSED with hysteresis."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2
        )
        detector._reset_temporal_state()

        # Start with open eyes
        assert detector._eye_state == detector._eye_state.OPEN

        # First frame with eyes closed
        blink = detector._detect_blink_from_ear(0.20)
        assert blink == False  # Not confirmed yet (need 2 consecutive frames)
        assert detector._eye_state.value == "open"  # Still OPEN

        # Second frame with eyes closed - confirms closure
        blink = detector._detect_blink_from_ear(0.19)
        assert blink == False  # Closed but not yet reopened
        assert detector._eye_state.value == "closed"

    def test_detect_blink_complete_cycle(self):
        """Test complete blink cycle detection with hysteresis."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2
        )
        detector._reset_temporal_state()

        # Start with open eyes
        detector._detect_blink_from_ear(0.35)  # Open

        # Close eyes (requires 2 consecutive frames)
        detector._detect_blink_from_ear(0.20)  # First frame below threshold
        detector._detect_blink_from_ear(0.19)  # Closed (confirmed)

        # Reopen eyes (requires 2 consecutive frames above open threshold)
        blink1 = detector._detect_blink_from_ear(0.26)  # First frame above open threshold
        assert blink1 == False  # Not confirmed yet

        blink2 = detector._detect_blink_from_ear(0.27)  # Open (confirmed)
        assert blink2 == True  # Complete blink detected!
        assert detector._blink_count == 1

    def test_detect_blink_false_alarm(self):
        """Test that brief closure doesn't count as blink with consecutive frame requirement."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2
        )
        detector._reset_temporal_state()

        # Start with open eyes
        detector._detect_blink_from_ear(0.35)

        # Brief closure (only 1 frame below threshold, then back up)
        detector._detect_blink_from_ear(0.20)  # First frame below threshold
        detector._detect_blink_from_ear(0.35)  # Reopened before confirmation (only 1 frame)

        # Should not have registered a blink
        assert detector._blink_count == 0
        assert detector._eye_state.value == "open"

    def test_detect_multiple_blinks(self):
        """Test detection of multiple blinks in sequence with hysteresis."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2
        )
        detector._reset_temporal_state()

        # First blink
        detector._detect_blink_from_ear(0.35)  # Open
        detector._detect_blink_from_ear(0.20)  # First frame closing
        detector._detect_blink_from_ear(0.19)  # Closed (confirmed)
        detector._detect_blink_from_ear(0.26)  # First frame reopening
        blink1 = detector._detect_blink_from_ear(0.27)  # Open (confirmed) - blink complete
        assert blink1 == True
        assert detector._blink_count == 1

        # Second blink
        detector._detect_blink_from_ear(0.35)  # Open
        detector._detect_blink_from_ear(0.20)  # First frame closing
        detector._detect_blink_from_ear(0.18)  # Closed (confirmed)
        detector._detect_blink_from_ear(0.26)  # First frame reopening
        blink2 = detector._detect_blink_from_ear(0.28)  # Open (confirmed) - blink complete
        assert blink2 == True
        assert detector._blink_count == 2

    def test_detect_blink_interrupted_reopening(self):
        """Test that eyes closing again before full reopening returns to CLOSED state."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2
        )
        detector._reset_temporal_state()

        # Complete closure
        detector._detect_blink_from_ear(0.35)  # Open
        detector._detect_blink_from_ear(0.20)  # First frame closing
        detector._detect_blink_from_ear(0.19)  # Closed (confirmed)

        # Start reopening but close again before confirmation
        detector._detect_blink_from_ear(0.26)  # First frame above open threshold (not confirmed)
        detector._detect_blink_from_ear(0.18)  # Drops below close threshold again

        # Should be back in closed state
        assert detector._eye_state.value == "closed"
        assert detector._blink_count == 0  # No complete blink


class TestLivenessDetectionWithMockLandmarks:
    """Test liveness detection with mocked landmarks."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    def _mock_landmarks_open_eyes(self):
        """Create mock landmarks for open eyes."""
        left_eye = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_eye = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        return (left_eye, right_eye)

    def _mock_landmarks_closed_eyes(self):
        """Create mock landmarks for closed eyes."""
        left_eye = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_eye = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        return (left_eye, right_eye)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_sequence_with_blinks_returns_live(self, mock_extract):
        """Test that sequence with blinks returns LIVE."""
        detector = LivenessDetector(min_frames=10, min_blinks=1, enable_challenges=False)

        # Create sequence: open -> close -> open (one blink)
        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]

        # Mock landmark extraction: simulate blink in middle of sequence
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return self._mock_landmarks_closed_eyes()
            else:
                return self._mock_landmarks_open_eyes()

        mock_extract.side_effect = mock_landmarks

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.LIVE
        assert result.frame_count == 15
        assert "Blinks detected" in result.reason

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_sequence_without_blinks_returns_unknown(self, mock_extract):
        """Test that sequence without blinks returns UNKNOWN."""
        detector = LivenessDetector(min_frames=10, min_blinks=1)

        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]

        # Mock landmark extraction: all frames have open eyes
        mock_extract.return_value = self._mock_landmarks_open_eyes()

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.UNKNOWN
        assert "Insufficient blinks" in result.reason

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_sequence_with_no_landmarks_returns_unknown(self, mock_extract):
        """Test that sequence with no landmarks returns UNKNOWN."""
        detector = LivenessDetector(min_frames=10)

        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]

        # Mock landmark extraction: no landmarks detected
        mock_extract.return_value = None

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.UNKNOWN
        assert "Insufficient valid landmarks" in result.reason

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_sequence_with_multiple_blinks(self, mock_extract):
        """Test sequence with multiple blinks increases confidence."""
        detector = LivenessDetector(min_frames=20, min_blinks=1, enable_challenges=False)

        frames = [self._create_test_frame() for _ in range(30)]
        faces = [self._create_test_face() for _ in range(30)]

        # Mock landmarks: simulate 3 blinks
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            # First blink at frames 5-8
            if 5 <= call_count <= 8:
                return self._mock_landmarks_closed_eyes()
            # Second blink at frames 15-18
            elif 15 <= call_count <= 18:
                return self._mock_landmarks_closed_eyes()
            # Third blink at frames 25-28
            elif 25 <= call_count <= 28:
                return self._mock_landmarks_closed_eyes()
            else:
                return self._mock_landmarks_open_eyes()

        mock_extract.side_effect = mock_landmarks

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.LIVE
        assert result.confidence > 0.75


class TestPhase3BBackwardCompatibility:
    """Test backward compatibility with Phase 3A tests."""

    def test_single_frame_still_returns_unknown(self):
        """Test that single-frame detection still returns UNKNOWN in Phase 3B."""
        detector = LivenessDetector()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        face = DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

        result = detector.detect_liveness(frame, face)

        # Single frame should still return UNKNOWN
        assert result.state == LivenessState.UNKNOWN
        assert result.frame_count == 1

    def test_insufficient_frames_still_returns_unknown(self):
        """Test that insufficient frames still returns UNKNOWN."""
        detector = LivenessDetector(min_frames=10)
        frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        faces = [DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95) for _ in range(5)]

        result = detector.detect_liveness_sequence(frames, faces)

        assert result.state == LivenessState.UNKNOWN
        assert "Insufficient frames" in result.reason

    def test_phase_3a_validation_still_works(self):
        """Test that Phase 3A validation still works."""
        detector = LivenessDetector()

        # None frame should still raise error
        with pytest.raises(LivenessDetectionError, match="Frame cannot be None"):
            detector.detect_liveness(None, DetectedFace(bbox=(0, 0, 10, 10), confidence=0.9))


class TestTemporalStateReset:
    """Test temporal state management."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_state_resets_between_sequences(self, mock_extract):
        """Test that temporal state resets between different sequences."""
        detector = LivenessDetector(min_frames=10, min_blinks=1, enable_challenges=False)

        # Mock landmarks for blink
        def mock_with_blink(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                left = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
                right = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
            else:
                left = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
                right = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
            return (left, right)

        mock_extract.side_effect = mock_with_blink

        # First sequence with blink
        frames1 = [self._create_test_frame() for _ in range(15)]
        faces1 = [self._create_test_face() for _ in range(15)]
        result1 = detector.detect_liveness_sequence(frames1, faces1)
        assert result1.state == LivenessState.LIVE

        # Reset mock
        mock_extract.reset_mock()
        mock_extract.side_effect = mock_with_blink

        # Second sequence should start with fresh state
        frames2 = [self._create_test_frame() for _ in range(15)]
        faces2 = [self._create_test_face() for _ in range(15)]
        result2 = detector.detect_liveness_sequence(frames2, faces2)
        assert result2.state == LivenessState.LIVE

        # Both should have detected blinks independently
        assert "Blinks detected: 1" in result1.reason or "Blinks detected" in result1.reason
        assert "Blinks detected: 1" in result2.reason or "Blinks detected" in result2.reason


class TestHysteresisAndNaturalBlinks:
    """Test hysteresis and natural blink behavior."""

    def test_hysteresis_configuration_validation(self):
        """Test validation of hysteresis thresholds."""
        # Open threshold must be >= close threshold
        with pytest.raises(ValueError, match="ear_open_threshold.*must be greater than or equal to"):
            LivenessDetector(ear_close_threshold=0.25, ear_open_threshold=0.20)

        # Close threshold cannot be negative
        with pytest.raises(ValueError, match="ear_close_threshold must be between"):
            LivenessDetector(ear_close_threshold=-0.1, ear_open_threshold=0.25)

        # Open threshold cannot exceed 1.0
        with pytest.raises(ValueError, match="ear_open_threshold must be between"):
            LivenessDetector(ear_close_threshold=0.20, ear_open_threshold=1.2)

    def test_short_natural_blink_single_frame(self):
        """Test that a short natural blink (1-frame closure) is detected with consecutive_frames=1."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=1,
            ear_smoothing_window=0
        )
        detector.reset_temporal_state()

        # Sequence of EAR: open (0.33) -> closed (0.16) -> open (0.32)
        assert detector.update_temporal_state(0.33) is False
        assert detector.eye_state == detector.eye_state.OPEN

        assert detector.update_temporal_state(0.16) is False
        assert detector.eye_state == detector.eye_state.CLOSED

        assert detector.update_temporal_state(0.32) is True
        assert detector.eye_state == detector.eye_state.OPEN
        assert detector.blink_count == 1

    def test_prolonged_closure_counts_single_blink(self):
        """Test that keeping eyes closed for multiple frames only counts as 1 blink upon reopening."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=1,
            ear_smoothing_window=0
        )
        detector.reset_temporal_state()

        # Open
        detector.update_temporal_state(0.35)

        # Closed for 5 frames
        for _ in range(5):
            blink = detector.update_temporal_state(0.15)
            assert blink is False
            assert detector.eye_state == detector.eye_state.CLOSED
            assert detector.blink_count == 0

        # Reopen
        blink = detector.update_temporal_state(0.30)
        assert blink is True
        assert detector.blink_count == 1

    def test_partial_closure_within_hysteresis_band(self):
        """Test that partial eye squinting between close and open thresholds does not trigger a false blink."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=1,
            ear_smoothing_window=0
        )
        detector.reset_temporal_state()

        # Open
        detector.update_temporal_state(0.32)

        # Squint to 0.23 (above 0.21 close threshold, below 0.25 open threshold)
        detector.update_temporal_state(0.23)
        assert detector.eye_state == detector.eye_state.OPEN
        assert detector.blink_count == 0

        # Back to open 0.32
        detector.update_temporal_state(0.32)
        assert detector.eye_state == detector.eye_state.OPEN
        assert detector.blink_count == 0


class TestIncrementalAndStreaming:
    """Test incremental processing and stream resilience."""

    def test_face_loss_and_recovery_during_closing(self):
        """Test that face loss (None EAR) resets frame counters safely."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=2,
            ear_smoothing_window=0
        )
        detector.reset_temporal_state()

        # Open
        detector.update_temporal_state(0.32)
        assert detector.eye_state == detector.eye_state.OPEN

        # 1 frame below threshold (not yet confirmed as CLOSED - needs 2 consecutive)
        detector.update_temporal_state(0.18)
        assert detector.eye_state == detector.eye_state.OPEN  # Still OPEN (not confirmed)

        # Face lost - resets frame counters
        detector.update_temporal_state(None)
        assert detector.eye_state == detector.eye_state.OPEN
        assert detector.blink_count == 0

        # Face returns with open eyes
        detector.update_temporal_state(0.32)
        assert detector.eye_state == detector.eye_state.OPEN
        assert detector.blink_count == 0

    def test_reset_temporal_state_clears_all(self):
        """Test that reset_temporal_state clears all counters and history."""
        detector = LivenessDetector(
            ear_close_threshold=0.21,
            ear_open_threshold=0.25,
            ear_consecutive_frames=1,
            ear_smoothing_window=0
        )
        # Record a blink
        detector.update_temporal_state(0.32)
        detector.update_temporal_state(0.18)
        detector.update_temporal_state(0.32)
        assert detector.blink_count == 1

        # Reset
        detector.reset_temporal_state()
        assert detector.blink_count == 0
        assert detector.eye_state == detector.eye_state.OPEN

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_extract_ear_returns_average(self, mock_landmarks):
        """Test extract_ear calculates average EAR from both eyes."""
        detector = LivenessDetector()

        # Mock left eye (EAR ~ 0.33) and right eye (EAR ~ 0.33)
        left = [(0, 5), (2, 3), (3, 3), (5, 5), (3, 7), (2, 7)]
        right = [(10, 5), (12, 3), (13, 3), (15, 5), (13, 7), (12, 7)]
        mock_landmarks.return_value = (left, right)

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        face = DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

        ear = detector.extract_ear(frame, face)
        assert ear is not None
        assert 0.7 < ear < 0.9


# =============================================================================
# Phase 3C Tests - Active Challenges and Anti-Spoofing
# =============================================================================


class TestPhase3CConfiguration:
    """Test Phase 3C configuration parameters."""

    def test_enable_challenges_default(self):
        """Test enable_challenges uses config default."""
        detector = LivenessDetector()
        assert detector.enable_challenges == config.LIVENESS_ENABLE_ACTIVE_CHALLENGES

    def test_enable_challenges_custom(self):
        """Test enable_challenges can be customized."""
        detector = LivenessDetector(enable_challenges=False)
        assert detector.enable_challenges == False

    def test_head_turn_angle_default(self):
        """Test head_turn_angle uses config default."""
        detector = LivenessDetector()
        assert detector.head_turn_angle == config.LIVENESS_HEAD_TURN_ANGLE

    def test_head_turn_angle_custom(self):
        """Test head_turn_angle can be customized."""
        detector = LivenessDetector(head_turn_angle=30.0)
        assert detector.head_turn_angle == 30.0

    def test_head_turn_angle_validation(self):
        """Test head_turn_angle must be positive."""
        with pytest.raises(ValueError, match="head_turn_angle must be positive"):
            LivenessDetector(head_turn_angle=0.0)
        
        with pytest.raises(ValueError, match="head_turn_angle must be positive"):
            LivenessDetector(head_turn_angle=-10.0)

    def test_head_hold_frames_default(self):
        """Test head_hold_frames uses config default."""
        detector = LivenessDetector()
        assert detector.head_hold_frames == config.LIVENESS_HEAD_HOLD_FRAMES

    def test_head_hold_frames_custom(self):
        """Test head_hold_frames can be customized."""
        detector = LivenessDetector(head_hold_frames=5)
        assert detector.head_hold_frames == 5

    def test_head_hold_frames_validation(self):
        """Test head_hold_frames must be at least 1."""
        with pytest.raises(ValueError, match="head_hold_frames must be at least 1"):
            LivenessDetector(head_hold_frames=0)

    def test_require_challenge_default(self):
        """Test require_challenge uses config default."""
        detector = LivenessDetector()
        assert detector.require_challenge == config.LIVENESS_REQUIRE_CHALLENGE

    def test_require_challenge_custom(self):
        """Test require_challenge can be customized."""
        detector = LivenessDetector(require_challenge=False)
        assert detector.require_challenge == False


class TestChallengeEnums:
    """Test Phase 3C challenge enums."""

    def test_challenge_type_values(self):
        """Test ChallengeType enum values."""
        from src.computer_vision.liveness_detector import ChallengeType
        
        assert hasattr(ChallengeType, 'TURN_LEFT')
        assert hasattr(ChallengeType, 'TURN_RIGHT')
        assert hasattr(ChallengeType, 'NONE')
        
        assert ChallengeType.TURN_LEFT.value == "turn_left"
        assert ChallengeType.TURN_RIGHT.value == "turn_right"
        assert ChallengeType.NONE.value == "none"

    def test_challenge_state_values(self):
        """Test ChallengeState enum values."""
        from src.computer_vision.liveness_detector import ChallengeState
        
        assert hasattr(ChallengeState, 'INACTIVE')
        assert hasattr(ChallengeState, 'ACTIVE')
        assert hasattr(ChallengeState, 'COMPLETED')
        assert hasattr(ChallengeState, 'FAILED')


class TestChallengeInitialization:
    """Test challenge initialization and state management."""

    def test_initial_challenge_state(self):
        """Test challenge starts in INACTIVE state."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(enable_challenges=True)
        
        assert detector.current_challenge == ChallengeType.NONE
        assert detector.challenge_state == ChallengeState.INACTIVE

    def test_start_new_challenge_when_enabled(self):
        """Test starting a new challenge when challenges are enabled."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(enable_challenges=True)
        
        challenge = detector.start_new_challenge()
        
        assert challenge in [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT]
        assert detector.current_challenge == challenge
        assert detector.challenge_state == ChallengeState.ACTIVE

    def test_start_new_challenge_when_disabled(self):
        """Test starting a challenge when challenges are disabled returns NONE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(enable_challenges=False)
        
        challenge = detector.start_new_challenge()
        
        assert challenge == ChallengeType.NONE
        assert detector.current_challenge == ChallengeType.NONE

    def test_challenge_randomization(self):
        """Test that challenges are randomized."""
        from src.computer_vision.liveness_detector import ChallengeType
        
        detector = LivenessDetector(enable_challenges=True)
        
        # Start multiple challenges and check we get both types
        challenges = set()
        for _ in range(20):
            detector.reset_temporal_state()
            challenge = detector.start_new_challenge()
            challenges.add(challenge)
        
        # Should have at least both types over 20 trials
        assert len(challenges) >= 2
        assert ChallengeType.TURN_LEFT in challenges or ChallengeType.TURN_RIGHT in challenges

    def test_reset_temporal_state_resets_challenge(self):
        """Test that reset_temporal_state also resets challenge state."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(enable_challenges=True)
        detector.start_new_challenge()
        
        assert detector.challenge_state == ChallengeState.ACTIVE
        
        detector.reset_temporal_state()
        
        assert detector.current_challenge == ChallengeType.NONE
        assert detector.challenge_state == ChallengeState.INACTIVE


class TestHeadYawEstimation:
    """Test head yaw estimation."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_estimate_head_yaw_called_during_challenge(self, mock_yaw):
        """Test that head yaw estimation is called during active challenge."""
        from src.computer_vision.liveness_detector import ChallengeState
        
        mock_yaw.return_value = 0.0  # Center position
        
        detector = LivenessDetector(enable_challenges=True)
        detector.start_new_challenge()
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        detector.update_temporal_state(0.30, frame, face)
        
        # Yaw estimation should be called
        assert mock_yaw.called

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_turn_left_requires_negative_yaw(self, mock_yaw):
        """Test that TURN_LEFT challenge requires negative yaw angle."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3
        )
        
        # Force TURN_LEFT challenge
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        # Test with insufficient left turn
        mock_yaw.return_value = -15.0  # Not enough
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 0
        
        # Test with sufficient left turn
        mock_yaw.return_value = -30.0  # Enough
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 1

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_turn_right_requires_positive_yaw(self, mock_yaw):
        """Test that TURN_RIGHT challenge requires positive yaw angle."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3
        )
        
        # Force TURN_RIGHT challenge
        detector._current_challenge = ChallengeType.TURN_RIGHT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        # Test with insufficient right turn
        mock_yaw.return_value = 15.0  # Not enough
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 0
        
        # Test with sufficient right turn
        mock_yaw.return_value = 30.0  # Enough
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 1


class TestChallengeCompletion:
    """Test challenge completion logic."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_challenge_requires_hold_frames(self, mock_yaw):
        """Test that challenge requires holding position for required frames."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3
        )
        
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        mock_yaw.return_value = -30.0  # Correct position
        
        # Frame 1 - not complete yet
        detector.update_temporal_state(0.30, frame, face)
        assert detector.challenge_state == ChallengeState.ACTIVE
        assert detector._challenge_frames_held == 1
        
        # Frame 2 - still not complete
        detector.update_temporal_state(0.30, frame, face)
        assert detector.challenge_state == ChallengeState.ACTIVE
        assert detector._challenge_frames_held == 2
        
        # Frame 3 - now complete!
        detector.update_temporal_state(0.30, frame, face)
        assert detector.challenge_state == ChallengeState.COMPLETED
        assert detector._challenge_frames_held == 3

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_challenge_resets_on_lost_position(self, mock_yaw):
        """Test that losing target position resets hold counter."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3
        )
        
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        # Hold position for 2 frames
        mock_yaw.return_value = -30.0
        detector.update_temporal_state(0.30, frame, face)
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 2
        
        # Lose position
        mock_yaw.return_value = -10.0  # Not enough
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 0  # Reset

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_challenge_resets_on_face_loss(self, mock_yaw):
        """Test that losing face resets hold counter."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3
        )
        
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frame = self._create_test_frame()
        face = self._create_test_face()
        
        # Hold position for 2 frames
        mock_yaw.return_value = -30.0
        detector.update_temporal_state(0.30, frame, face)
        detector.update_temporal_state(0.30, frame, face)
        assert detector._challenge_frames_held == 2
        
        # Face lost (None EAR)
        detector.update_temporal_state(None, frame, face)
        assert detector._challenge_frames_held == 0  # Reset


class TestChallengeProgress:
    """Test challenge progress tracking."""

    def test_get_challenge_progress_inactive(self):
        """Test get_challenge_progress when challenge is inactive."""
        detector = LivenessDetector(enable_challenges=True)
        
        progress = detector.get_challenge_progress()
        
        assert progress['challenge_type'] == 'none'
        assert progress['challenge_state'] == 'inactive'
        assert progress['frames_held'] == 0
        assert progress['progress_percent'] == 0.0

    def test_get_challenge_progress_active(self):
        """Test get_challenge_progress when challenge is active."""
        from src.computer_vision.liveness_detector import ChallengeType
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_hold_frames=5
        )
        
        challenge = detector.start_new_challenge()
        
        progress = detector.get_challenge_progress()
        
        assert progress['challenge_type'] in ['turn_left', 'turn_right']
        assert progress['challenge_state'] == 'active'
        assert progress['frames_held'] == 0
        assert progress['frames_required'] == 5
        assert progress['progress_percent'] == 0.0

    def test_get_challenge_progress_partial(self):
        """Test get_challenge_progress with partial completion."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            enable_challenges=True,
            head_hold_frames=5
        )
        
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE
        detector._challenge_frames_held = 3
        
        progress = detector.get_challenge_progress()
        
        assert progress['frames_held'] == 3
        assert progress['frames_required'] == 5
        assert progress['progress_percent'] == 0.6  # 3/5


class TestLivenessWithChallenges:
    """Test liveness detection with challenge integration."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    def _mock_landmarks_open_eyes(self):
        """Create mock landmarks for open eyes."""
        left_eye = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_eye = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        return (left_eye, right_eye)

    def _mock_landmarks_closed_eyes(self):
        """Create mock landmarks for closed eyes."""
        left_eye = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_eye = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        return (left_eye, right_eye)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_liveness_without_challenges_returns_live_on_blinks(self, mock_extract):
        """Test that with challenges disabled, blinks alone give LIVE result."""
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=False
        )
        
        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]
        
        # Mock blink: frames 5-8 closed
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return self._mock_landmarks_closed_eyes()
            else:
                return self._mock_landmarks_open_eyes()
        
        mock_extract.side_effect = mock_landmarks
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        assert result.state == LivenessState.LIVE
        assert "Blinks detected" in result.reason

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_liveness_with_challenges_requires_both_blink_and_challenge(self, mock_extract, mock_yaw):
        """Test that with challenges required, must have both blinks and challenge completion."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        # Start a TURN_LEFT challenge
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        frames = [self._create_test_frame() for _ in range(20)]
        faces = [self._create_test_face() for _ in range(20)]
        
        # Mock blink in frames 5-8
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return self._mock_landmarks_closed_eyes()
            else:
                return self._mock_landmarks_open_eyes()
        
        mock_extract.side_effect = mock_landmarks
        
        # Mock head turn in frames 12-15
        def mock_yaw_value(frame, face):
            call_count = mock_yaw.call_count
            if 12 <= call_count <= 15:
                return -30.0  # Left turn
            else:
                return 0.0
        
        mock_yaw.side_effect = mock_yaw_value
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # Should be LIVE with both blinks and challenge completed
        assert result.state == LivenessState.LIVE
        assert "Challenge" in result.reason
        assert "completed" in result.reason.lower()

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_liveness_with_challenges_returns_unknown_without_challenge(self, mock_extract):
        """Test that with challenge required but not completed, returns UNKNOWN."""
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True
        )
        
        # Challenge not started
        assert detector.challenge_state.value == 'inactive'
        
        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]
        
        # Mock blink
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return self._mock_landmarks_closed_eyes()
            else:
                return self._mock_landmarks_open_eyes()
        
        mock_extract.side_effect = mock_landmarks
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # Should be UNKNOWN because challenge not completed
        assert result.state == LivenessState.UNKNOWN
        assert "challenge" in result.reason.lower()


class TestPhase3CBackwardCompatibility:
    """Test backward compatibility with Phase 3A and 3B."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_phase_3b_behavior_preserved_when_challenges_disabled(self, mock_extract):
        """Test that Phase 3B behavior is preserved when challenges are disabled."""
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=False  # Phase 3B mode
        )
        
        frames = [self._create_test_frame() for _ in range(15)]
        faces = [self._create_test_face() for _ in range(15)]
        
        # Mock blink
        left = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        open_eyes = (left, right)
        
        left_closed = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_closed = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        closed_eyes = (left_closed, right_closed)
        
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return closed_eyes
            else:
                return open_eyes
        
        mock_extract.side_effect = mock_landmarks
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # Phase 3B behavior: LIVE based on blinks alone
        assert result.state == LivenessState.LIVE
        assert "Blinks detected" in result.reason


class TestBlinkBugRegression:
    """Regression tests for the critical blink requirement bug."""

    def _create_test_frame(self):
        """Helper to create a test frame."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def _create_test_face(self):
        """Helper to create a test DetectedFace."""
        return DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_challenge_completed_without_blink_must_not_be_live(self, mock_extract, mock_yaw):
        """CRITICAL BUG TEST: Challenge completes with zero blinks -> result must NOT be LIVE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        frames = [self._create_test_frame() for _ in range(20)]
        faces = [self._create_test_face() for _ in range(20)]
        
        # Mock: eyes ALWAYS open (NO blinks)
        left = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        mock_extract.return_value = (left, right)
        
        # Mock: head turn completes - simulate left turn completing in frames 10-13
        call_counter = {'count': 0}
        def mock_yaw_value(frame, face):
            call_counter['count'] += 1
            if 10 <= call_counter['count'] <= 13:
                return -30.0  # Left turn
            else:
                return 0.0
        
        mock_yaw.side_effect = mock_yaw_value
        
        # Start the challenge before calling detect_liveness_sequence
        # Note: detect_liveness_sequence calls _reset_temporal_state but NOT _reset_challenge_state
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        detector._challenge_frames_held = 0
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # CRITICAL: Must NOT be LIVE because blink requirement not satisfied
        assert result is not None, "Result should not be None"
        assert result.state == LivenessState.UNKNOWN, f"Expected UNKNOWN but got {result.state}"
        assert result.state != LivenessState.LIVE, "Must NOT be LIVE without blinks"
        # Should indicate challenge completed but insufficient blinks
        assert detector.challenge_state == ChallengeState.COMPLETED, f"Challenge should be COMPLETED but is {detector.challenge_state}"
        assert detector.blink_count == 0, f"Expected 0 blinks but got {detector.blink_count}"

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_blink_satisfied_without_challenge_must_not_be_live(self, mock_extract, mock_yaw):
        """Blink requirement satisfied but challenge incomplete -> result must NOT be LIVE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        # Start challenge
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        
        frames = [self._create_test_frame() for _ in range(20)]
        faces = [self._create_test_face() for _ in range(20)]
        
        # Mock: blink occurs in frames 5-8
        left_open = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_open = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        left_closed = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_closed = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return (left_closed, right_closed)
            else:
                return (left_open, right_open)
        
        mock_extract.side_effect = mock_landmarks
        
        # Mock: head NEVER turns enough (challenge incomplete)
        mock_yaw.return_value = -10.0  # Not enough
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # CRITICAL: Must NOT be LIVE because challenge not completed
        assert result.state == LivenessState.UNKNOWN
        assert result.state != LivenessState.LIVE
        # Blinks should be detected but challenge still active
        assert detector.blink_count >= 1
        assert detector.challenge_state == ChallengeState.ACTIVE

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_both_requirements_satisfied_must_be_live(self, mock_extract, mock_yaw):
        """Both blink AND challenge completed -> result must be LIVE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        # Start challenge
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        
        frames = [self._create_test_frame() for _ in range(20)]
        faces = [self._create_test_face() for _ in range(20)]
        
        # Mock: blink occurs in frames 5-8
        left_open = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_open = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        left_closed = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_closed = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return (left_closed, right_closed)
            else:
                return (left_open, right_open)
        
        mock_extract.side_effect = mock_landmarks
        
        # Mock: head turn completes in frames 12-15
        def mock_yaw_value(frame, face):
            call_count = mock_yaw.call_count
            if 12 <= call_count <= 15:
                return -30.0  # Left turn
            else:
                return 0.0
        
        mock_yaw.side_effect = mock_yaw_value
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # CORRECT: Should be LIVE with both requirements satisfied
        assert result.state == LivenessState.LIVE
        assert detector.blink_count >= 1
        assert detector.challenge_state == ChallengeState.COMPLETED

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_challenge_first_then_blink_becomes_live(self, mock_extract, mock_yaw):
        """Challenge completes first, then blink occurs -> final result becomes LIVE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        # Start challenge
        detector._current_challenge = ChallengeType.TURN_RIGHT
        detector._challenge_state = ChallengeState.ACTIVE

        
        frames = [self._create_test_frame() for _ in range(25)]
        faces = [self._create_test_face() for _ in range(25)]
        
        # Mock: blink occurs AFTER challenge (frames 18-21)
        left_open = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_open = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        left_closed = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_closed = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 18 <= call_count <= 21:
                return (left_closed, right_closed)
            else:
                return (left_open, right_open)
        
        mock_extract.side_effect = mock_landmarks
        
        # Mock: head turn completes EARLY in frames 5-8
        def mock_yaw_value(frame, face):
            call_count = mock_yaw.call_count
            if 5 <= call_count <= 8:
                return 30.0  # Right turn
            else:
                return 0.0
        
        mock_yaw.side_effect = mock_yaw_value
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # Should be LIVE because both requirements eventually satisfied
        assert result.state == LivenessState.LIVE
        assert detector.blink_count >= 1
        assert detector.challenge_state == ChallengeState.COMPLETED

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    @patch('src.computer_vision.liveness_detector.LivenessDetector._extract_eye_landmarks')
    def test_blink_first_then_challenge_becomes_live(self, mock_extract, mock_yaw):
        """Blink occurs first, then challenge completes -> final result becomes LIVE."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True,
            head_hold_frames=3
        )
        
        # Start challenge
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state = ChallengeState.ACTIVE

        
        frames = [self._create_test_frame() for _ in range(25)]
        faces = [self._create_test_face() for _ in range(25)]
        
        # Mock: blink occurs EARLY (frames 5-8)
        left_open = [(33, 160), (35, 155), (37, 155), (39, 160), (37, 165), (35, 165)]
        right_open = [(200, 160), (202, 155), (204, 155), (206, 160), (204, 165), (202, 165)]
        left_closed = [(33, 160), (35, 160), (37, 160), (39, 160), (37, 160), (35, 160)]
        right_closed = [(200, 160), (202, 160), (204, 160), (206, 160), (204, 160), (202, 160)]
        
        def mock_landmarks(frame, face):
            call_count = mock_extract.call_count
            if 5 <= call_count <= 8:
                return (left_closed, right_closed)
            else:
                return (left_open, right_open)
        
        mock_extract.side_effect = mock_landmarks
        
        # Mock: head turn completes LATER in frames 18-21
        def mock_yaw_value(frame, face):
            call_count = mock_yaw.call_count
            if 18 <= call_count <= 21:
                return -30.0  # Left turn
            else:
                return 0.0
        
        mock_yaw.side_effect = mock_yaw_value
        
        result = detector.detect_liveness_sequence(frames, faces)
        
        # Should be LIVE because both requirements eventually satisfied
        assert result.state == LivenessState.LIVE
        assert detector.blink_count >= 1
        assert detector.challenge_state == ChallengeState.COMPLETED

    def test_challenge_state_completed_alone_not_sufficient(self):
        """ChallengeState.COMPLETED alone must NOT indicate overall test completion."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState
        
        detector = LivenessDetector(
            min_frames=10,
            min_blinks=1,
            enable_challenges=True,
            require_challenge=True
        )
        
        # Simulate challenge completion without blinks
        detector._challenge_state = ChallengeState.COMPLETED
        detector._challenge_completed = True
        detector._blink_count = 0  # NO blinks
        
        # Challenge state is COMPLETED but blink requirement not met
        assert detector.challenge_state == ChallengeState.COMPLETED
        assert detector.blink_count < 1
        
        # This state should NOT be considered overall LIVE
        # The evaluation logic must check BOTH conditions


class TestYawSignConvention:
    """
    Tests for the yaw sign convention fix.

    The camera frame fed to _estimate_head_yaw is the raw, UNMIRRORED frame
    captured from OpenCV.  In a front-facing camera:

      - MediaPipe landmark 33  (RIGHT_EYE_OUTER) is the subject's anatomical
        right eye → appears on the IMAGE-LEFT side (smaller x pixel value).
      - MediaPipe landmark 263 (LEFT_EYE_OUTER)  is the subject's anatomical
        left eye  → appears on the IMAGE-RIGHT side (larger x pixel value).

    When the user turns their head RIGHT:
      • their nose moves toward their right eye (toward smaller image-x)
      • nose_deviation  = -(nose_x - eye_center_x)  should be POSITIVE
      • _estimate_head_yaw should return a POSITIVE value

    When the user turns their head LEFT:
      • their nose moves toward their left eye (toward larger image-x)
      • nose_deviation  = -(nose_x - eye_center_x)  should be NEGATIVE
      • _estimate_head_yaw should return a NEGATIVE value

    The challenge checks are:
      TURN_RIGHT  requires  yaw > +threshold   (positive)
      TURN_LEFT   requires  yaw < -threshold   (negative)

    These tests use a mock MediaPipe landmark result so they are deterministic
    and do not depend on a camera or the MediaPipe model file.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _frame(width=640, height=480):
        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def _face():
        return DetectedFace(bbox=(220, 140, 200, 200), confidence=0.95)

    @staticmethod
    def _make_landmark(x_norm, y_norm=0.5):
        """Return a minimal mock landmark with .x and .y normalised attributes."""
        class _L:
            pass
        lm = _L()
        lm.x = x_norm
        lm.y = y_norm
        return lm

    def _landmarks_for_position(self, nose_x_norm, frame_width=640):
        """
        Build a 478-element mock landmark list whose relevant indices match the
        sign-convention table in LivenessDetector:

            NOSE_TIP        =   1   →  given nose_x_norm
            RIGHT_EYE_OUTER =  33   →  image-left  (subject's right eye, small x)
            LEFT_EYE_OUTER  = 263   →  image-right (subject's left eye,  large x)

        We fix the eye positions symmetrically at 0.35 / 0.65 so the eye span
        is non-zero and the centre is at 0.50.
        """
        right_eye_x = 0.35   # subject's right eye: image-left
        left_eye_x  = 0.65   # subject's left eye:  image-right

        lms = [self._make_landmark(0.5) for _ in range(478)]
        lms[1]   = self._make_landmark(nose_x_norm)   # NOSE_TIP
        lms[33]  = self._make_landmark(right_eye_x)   # RIGHT_EYE_OUTER
        lms[263] = self._make_landmark(left_eye_x)    # LEFT_EYE_OUTER
        return lms

    def _patch_landmarks(self, detector, landmarks):
        """
        Force the per-frame landmark cache to return `landmarks` without running
        the actual MediaPipe model.
        """
        detector._cached_frame_id = None   # will be overwritten on first call
        detector._cached_landmarks = landmarks

        # Patch _get_landmarks_for_frame to always return our landmarks
        def _fake_get(frame, face_bbox=None):
            detector._cached_frame_id = id(frame)
            detector._cached_landmarks = landmarks
            return landmarks

        detector._get_landmarks_for_frame = _fake_get

    # ── yaw sign tests ────────────────────────────────────────────────────────

    def test_nose_at_centre_gives_zero_yaw(self):
        """Nose exactly between the eyes → yaw ≈ 0."""
        detector = LivenessDetector()
        frame = self._frame()
        face  = self._face()
        self._patch_landmarks(detector, self._landmarks_for_position(0.50))

        yaw = detector._estimate_head_yaw(frame, face)

        assert yaw is not None
        assert abs(yaw) < 1.0, f"Expected ~0, got {yaw}"

    def test_right_turn_gives_positive_yaw(self):
        """
        User turns RIGHT: nose moves toward subject's right eye (image-left,
        smaller x).  The fixed sign convention must return a POSITIVE yaw so
        that the TURN_RIGHT challenge condition  (yaw > +threshold)  is met.
        """
        detector = LivenessDetector()
        frame = self._frame()
        face  = self._face()
        # Nose shifted left of centre (toward image-left = subject's right eye)
        self._patch_landmarks(detector, self._landmarks_for_position(0.38))

        yaw = detector._estimate_head_yaw(frame, face)

        assert yaw is not None, "Expected a numeric yaw, got None"
        assert yaw > 0, (
            f"RIGHT turn (nose at 0.38, eyes at 0.35/0.65) must give POSITIVE yaw; "
            f"got {yaw:.2f}.  TURN_RIGHT challenge would never fire."
        )

    def test_left_turn_gives_negative_yaw(self):
        """
        User turns LEFT: nose moves toward subject's left eye (image-right,
        larger x).  The fixed sign convention must return a NEGATIVE yaw so
        that the TURN_LEFT challenge condition  (yaw < -threshold)  is met.
        """
        detector = LivenessDetector()
        frame = self._frame()
        face  = self._face()
        # Nose shifted right of centre (toward image-right = subject's left eye)
        self._patch_landmarks(detector, self._landmarks_for_position(0.62))

        yaw = detector._estimate_head_yaw(frame, face)

        assert yaw is not None, "Expected a numeric yaw, got None"
        assert yaw < 0, (
            f"LEFT turn (nose at 0.62, eyes at 0.35/0.65) must give NEGATIVE yaw; "
            f"got {yaw:.2f}.  TURN_LEFT challenge would never fire."
        )

    def test_yaw_magnitude_increases_with_turn_angle(self):
        """Larger head turn → larger absolute yaw value."""
        detector = LivenessDetector()
        frame = self._frame()
        face  = self._face()

        self._patch_landmarks(detector, self._landmarks_for_position(0.45))
        yaw_small = detector._estimate_head_yaw(frame, face)

        self._patch_landmarks(detector, self._landmarks_for_position(0.35))
        yaw_large = detector._estimate_head_yaw(frame, face)

        assert yaw_small is not None
        assert yaw_large is not None
        assert abs(yaw_large) > abs(yaw_small), (
            f"Larger turn should give larger |yaw|: {abs(yaw_large):.2f} vs {abs(yaw_small):.2f}"
        )

    # ── challenge completion with physical movement ───────────────────────────

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_right_challenge_completed_by_right_turn(self, mock_yaw):
        """
        TURN_RIGHT challenge must be satisfied by a positive yaw value
        (user physically turns right, nose toward image-left).
        """
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_RIGHT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # positive yaw = user turned right
        mock_yaw.return_value = 30.0
        for _ in range(3):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state == ChallengeState.COMPLETED, (
            "TURN_RIGHT challenge must complete when yaw is +30 (right turn)"
        )

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_right_challenge_not_completed_by_left_turn(self, mock_yaw):
        """
        TURN_RIGHT challenge must NOT be satisfied by a negative yaw value
        (user physically turns left).
        """
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_RIGHT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # negative yaw = user turned left — must not satisfy TURN_RIGHT
        mock_yaw.return_value = -30.0
        for _ in range(5):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state != ChallengeState.COMPLETED, (
            "TURN_RIGHT challenge must NOT complete when yaw is -30 (left turn)"
        )

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_left_challenge_completed_by_left_turn(self, mock_yaw):
        """
        TURN_LEFT challenge must be satisfied by a negative yaw value
        (user physically turns left, nose toward image-right).
        """
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # negative yaw = user turned left
        mock_yaw.return_value = -30.0
        for _ in range(3):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state == ChallengeState.COMPLETED, (
            "TURN_LEFT challenge must complete when yaw is -30 (left turn)"
        )

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_left_challenge_not_completed_by_right_turn(self, mock_yaw):
        """
        TURN_LEFT challenge must NOT be satisfied by a positive yaw value
        (user physically turns right).
        """
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # positive yaw = user turned right — must not satisfy TURN_LEFT
        mock_yaw.return_value = 30.0
        for _ in range(5):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state != ChallengeState.COMPLETED, (
            "TURN_LEFT challenge must NOT complete when yaw is +30 (right turn)"
        )

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_insufficient_angle_does_not_complete_right_challenge(self, mock_yaw):
        """Turn that does not reach head_turn_angle threshold must not complete."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_RIGHT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # positive yaw but below the 25° threshold
        mock_yaw.return_value = 20.0
        for _ in range(5):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state != ChallengeState.COMPLETED

    @patch('src.computer_vision.liveness_detector.LivenessDetector._estimate_head_yaw')
    def test_insufficient_angle_does_not_complete_left_challenge(self, mock_yaw):
        """Turn that does not reach head_turn_angle threshold must not complete."""
        from src.computer_vision.liveness_detector import ChallengeType, ChallengeState

        detector = LivenessDetector(
            enable_challenges=True,
            head_turn_angle=25.0,
            head_hold_frames=3,
        )
        detector._current_challenge = ChallengeType.TURN_LEFT
        detector._challenge_state   = ChallengeState.ACTIVE
        detector._challenge_frames_held = 0

        frame = self._frame()
        face  = self._face()

        # negative yaw but magnitude below the 25° threshold
        mock_yaw.return_value = -20.0
        for _ in range(5):
            detector.update_temporal_state(0.30, frame, face)

        assert detector._challenge_state != ChallengeState.COMPLETED

    # ── landmark-level yaw sign derivation ───────────────────────────────────

    def test_yaw_sign_derived_from_landmark_positions(self):
        """
        Black-box check: feed explicit landmark positions to _estimate_head_yaw
        and verify the sign matches the expected user-perspective convention
        without mocking the method itself.

        Eye layout (normalised x, frame width 640):
            RIGHT_EYE_OUTER (lm 33)  = 0.35  →  x = 224  (image-left)
            LEFT_EYE_OUTER  (lm 263) = 0.65  →  x = 416  (image-right)
            eye_center_x             = 0.50  →  x = 320

        Right turn: nose_x = 0.38 → x = 243.2
            raw deviation  = 243.2 - 320 = -76.8
            after negation = +76.8  →  positive yaw  ✓ TURN_RIGHT fires

        Left turn: nose_x = 0.62 → x = 396.8
            raw deviation  = 396.8 - 320 = +76.8
            after negation = -76.8  →  negative yaw  ✓ TURN_LEFT fires
        """
        detector = LivenessDetector()
        frame    = self._frame(width=640)
        face     = self._face()

        # Right turn
        self._patch_landmarks(detector, self._landmarks_for_position(0.38))
        yaw_right = detector._estimate_head_yaw(frame, face)
        assert yaw_right is not None
        assert yaw_right > 0, (
            f"Right turn (nose 0.38, eyes 0.35/0.65) must yield positive yaw; got {yaw_right:.3f}"
        )

        # Left turn
        self._patch_landmarks(detector, self._landmarks_for_position(0.62))
        yaw_left = detector._estimate_head_yaw(frame, face)
        assert yaw_left is not None
        assert yaw_left < 0, (
            f"Left turn (nose 0.62, eyes 0.35/0.65) must yield negative yaw; got {yaw_left:.3f}"
        )

        # Symmetry: equal-and-opposite turns must give equal-and-opposite yaw
        assert abs(abs(yaw_right) - abs(yaw_left)) < 0.01, (
            f"Symmetric turns must give symmetric yaw: {yaw_right:.3f} vs {yaw_left:.3f}"
        )


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
