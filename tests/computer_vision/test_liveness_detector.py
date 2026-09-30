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


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
