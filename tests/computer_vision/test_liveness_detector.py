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
        detector = LivenessDetector(min_frames=10, min_blinks=1)

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
        detector = LivenessDetector(min_frames=20, min_blinks=1)

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
        detector = LivenessDetector(min_frames=10, min_blinks=1)

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


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
