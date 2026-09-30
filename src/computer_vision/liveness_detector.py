"""
Liveness Detector for the Intelligent Attendance Monitoring System.

Provides liveness detection foundation to determine if a detected face belongs
to a live person. This is Phase 3A: establishes the API, data structures, and
validation framework. The complete liveness algorithm will be implemented in
Phase 3B and Phase 3C.

Phase 3A: Foundation & Architecture (current)
Phase 3B: Will implement blink detection and temporal analysis
Phase 3C: Will implement complete anti-spoofing decision logic
"""

from typing import Optional, List
from enum import Enum
import numpy as np

import config
from src.computer_vision.face_detector import DetectedFace
from src.utils.logger import get_logger


class LivenessState(Enum):
    """
    Represents the liveness state of a face.
    
    States:
        LIVE: Face has been determined to be a live person
        SPOOF: Face has been determined to be a spoofing attempt (photo/video)
        UNKNOWN: Insufficient evidence to determine liveness
    """
    LIVE = "live"
    SPOOF = "spoof"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class LivenessResult:
    """
    Represents the result of a liveness detection attempt.
    
    Attributes:
        state: The liveness state (LIVE, SPOOF, or UNKNOWN)
        confidence: Confidence score for the determination (0.0 to 1.0)
        frame_count: Number of frames/observations analyzed
        reason: Human-readable reason for the determination
    """
    
    def __init__(
        self,
        state: LivenessState,
        confidence: float,
        frame_count: int,
        reason: str
    ):
        """
        Initialize a liveness result.
        
        Args:
            state: The liveness state
            confidence: Confidence score (0.0 to 1.0)
            frame_count: Number of frames analyzed
            reason: Reason for the determination
            
        Raises:
            ValueError: If confidence is not in valid range or frame_count is negative
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        if frame_count < 0:
            raise ValueError(f"Frame count cannot be negative, got {frame_count}")
        
        self.state = state
        self.confidence = confidence
        self.frame_count = frame_count
        self.reason = reason
    
    def __repr__(self) -> str:
        return (
            f"LivenessResult(state={self.state}, "
            f"confidence={self.confidence:.3f}, "
            f"frames={self.frame_count}, "
            f"reason='{self.reason}')"
        )


class LivenessDetectionError(Exception):
    """Exception raised when liveness detection fails."""
    pass


class LivenessDetector:
    """
    Detects liveness in faces to prevent spoofing attacks.
    
    This is the foundation implementation (Phase 3A) that establishes the API
    and validation framework. The complete liveness detection algorithm will be
    implemented in subsequent phases.
    
    Phase 3A provides:
    - API structure for temporal liveness detection
    - Input validation
    - Result representation
    - Configuration framework
    - Safe failure behavior (unknown state when insufficient evidence)
    
    Phase 3B will add:
    - Blink detection using eye aspect ratio
    - Motion analysis
    - Temporal consistency checks
    
    Phase 3C will add:
    - Complete anti-spoofing decision logic
    - Challenge-response mechanisms
    - Advanced spoof detection
    
    Current Limitations (Phase 3A):
    - Does NOT yet provide anti-spoofing protection
    - Always returns UNKNOWN state as algorithm is not implemented
    - This is intentional: failing safe until full implementation
    """
    
    def __init__(
        self,
        min_frames: Optional[int] = None,
        liveness_threshold: Optional[float] = None,
        temporal_window_seconds: Optional[float] = None
    ):
        """
        Initialize the LivenessDetector.
        
        Args:
            min_frames: Minimum number of frames required for reliable detection.
                       If None, uses config.LIVENESS_MIN_FRAMES (default: 10)
            liveness_threshold: Confidence threshold for LIVE determination.
                               If None, uses config.LIVENESS_THRESHOLD (default: 0.75)
            temporal_window_seconds: Time window for temporal analysis in seconds.
                                    If None, uses config.LIVENESS_TEMPORAL_WINDOW (default: 2.0)
        
        Note: These parameters establish the configuration framework for Phase 3B/3C.
              Default values are placeholders and will be tuned during algorithm implementation.
        """
        self.min_frames = (
            min_frames 
            if min_frames is not None 
            else config.LIVENESS_MIN_FRAMES
        )
        self.liveness_threshold = (
            liveness_threshold 
            if liveness_threshold is not None 
            else config.LIVENESS_THRESHOLD
        )
        self.temporal_window_seconds = (
            temporal_window_seconds 
            if temporal_window_seconds is not None 
            else config.LIVENESS_TEMPORAL_WINDOW
        )
        
        # Validate configuration
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be at least 1, got {self.min_frames}")
        
        if not 0.0 <= self.liveness_threshold <= 1.0:
            raise ValueError(
                f"liveness_threshold must be between 0.0 and 1.0, got {self.liveness_threshold}"
            )
        
        if self.temporal_window_seconds <= 0:
            raise ValueError(
                f"temporal_window_seconds must be positive, got {self.temporal_window_seconds}"
            )
        
        self.logger = get_logger(__name__)
        self.logger.info(
            f"LivenessDetector initialized (Phase 3A - Foundation only): "
            f"min_frames={self.min_frames}, "
            f"threshold={self.liveness_threshold}, "
            f"temporal_window={self.temporal_window_seconds}s"
        )
        self.logger.warning(
            "Phase 3A: Liveness algorithm not yet implemented. "
            "Will return UNKNOWN state until Phase 3B/3C."
        )
    
    def detect_liveness(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> LivenessResult:
        """
        Detect liveness for a single frame observation.
        
        This is a foundation method (Phase 3A) that establishes the API structure.
        The actual liveness detection algorithm will be implemented in Phase 3B/3C.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            detected_face: DetectedFace object containing face location and confidence
        
        Returns:
            LivenessResult with state, confidence, and reasoning
        
        Raises:
            LivenessDetectionError: If input validation fails
        
        Current Behavior (Phase 3A):
            Always returns UNKNOWN state as the algorithm is not yet implemented.
            This is intentional safe behavior - we do not claim liveness without
            proper detection logic.
        """
        # Validate inputs
        self._validate_frame(frame)
        self._validate_detected_face(detected_face)
        
        # Phase 3A: Foundation only - algorithm not implemented
        # Returning UNKNOWN is safe behavior: we don't claim LIVE without evidence
        self.logger.debug(
            "Phase 3A: Liveness algorithm not implemented. Returning UNKNOWN state."
        )
        
        return LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=1,
            reason="Phase 3A: Liveness detection algorithm not yet implemented"
        )
    
    def detect_liveness_sequence(
        self,
        frames: List[np.ndarray],
        detected_faces: List[DetectedFace]
    ) -> LivenessResult:
        """
        Detect liveness across a sequence of frames (temporal analysis).
        
        This method supports temporal liveness detection by analyzing multiple
        consecutive frames. This is more robust than single-frame detection.
        
        Args:
            frames: List of consecutive frames as numpy arrays
            detected_faces: List of DetectedFace objects for each frame
        
        Returns:
            LivenessResult based on temporal analysis
        
        Raises:
            LivenessDetectionError: If inputs are invalid or mismatched
        
        Current Behavior (Phase 3A):
            Validates inputs and returns UNKNOWN state. Temporal analysis
            algorithm will be implemented in Phase 3B.
        """
        # Validate input sequences
        if frames is None or detected_faces is None:
            raise LivenessDetectionError(
                "Both frames and detected_faces must be provided"
            )
        
        if not isinstance(frames, list) or not isinstance(detected_faces, list):
            raise LivenessDetectionError(
                "frames and detected_faces must be lists"
            )
        
        if len(frames) != len(detected_faces):
            raise LivenessDetectionError(
                f"Mismatch: {len(frames)} frames but {len(detected_faces)} detected faces"
            )
        
        if len(frames) == 0:
            raise LivenessDetectionError(
                "Cannot perform liveness detection on empty sequence"
            )
        
        # Validate each frame and face
        for idx, (frame, face) in enumerate(zip(frames, detected_faces)):
            try:
                self._validate_frame(frame)
                self._validate_detected_face(face)
            except LivenessDetectionError as e:
                raise LivenessDetectionError(
                    f"Invalid input at index {idx}: {e}"
                )
        
        frame_count = len(frames)
        
        # Check if we have minimum required frames
        if frame_count < self.min_frames:
            self.logger.debug(
                f"Insufficient frames: {frame_count} < {self.min_frames} required"
            )
            return LivenessResult(
                state=LivenessState.UNKNOWN,
                confidence=0.0,
                frame_count=frame_count,
                reason=f"Insufficient frames: {frame_count} < {self.min_frames} required"
            )
        
        # Phase 3A: Temporal analysis algorithm not implemented
        self.logger.debug(
            f"Phase 3A: Temporal analysis not implemented. "
            f"Analyzed {frame_count} frames, returning UNKNOWN."
        )
        
        return LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=frame_count,
            reason="Phase 3A: Temporal liveness analysis not yet implemented"
        )
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """
        Validate input frame.
        
        Args:
            frame: Frame to validate
        
        Raises:
            LivenessDetectionError: If frame is invalid
        """
        if frame is None:
            raise LivenessDetectionError("Frame cannot be None")
        
        if not isinstance(frame, np.ndarray):
            raise LivenessDetectionError(
                f"Frame must be a numpy array, got {type(frame).__name__}"
            )
        
        if frame.size == 0:
            raise LivenessDetectionError("Frame cannot be empty")
        
        # Validate frame shape
        if len(frame.shape) not in [2, 3]:
            raise LivenessDetectionError(
                f"Invalid frame shape: {frame.shape}. "
                "Expected 2D (grayscale) or 3D (color) array"
            )
        
        # Validate minimum dimensions
        height, width = frame.shape[:2]
        if height < 10 or width < 10:
            raise LivenessDetectionError(
                f"Frame dimensions too small: {width}x{height}. "
                "Minimum 10x10 required"
            )
    
    def _validate_detected_face(self, detected_face: DetectedFace) -> None:
        """
        Validate DetectedFace object.
        
        Args:
            detected_face: DetectedFace to validate
        
        Raises:
            LivenessDetectionError: If detected_face is invalid
        """
        if detected_face is None:
            raise LivenessDetectionError("DetectedFace cannot be None")
        
        if not isinstance(detected_face, DetectedFace):
            raise LivenessDetectionError(
                f"Expected DetectedFace object, got {type(detected_face).__name__}"
            )
        
        # Validate bounding box
        if detected_face.bbox is None or len(detected_face.bbox) != 4:
            raise LivenessDetectionError(
                "DetectedFace must have valid bbox (x, y, width, height)"
            )
        
        x, y, w, h = detected_face.bbox
        if w <= 0 or h <= 0:
            raise LivenessDetectionError(
                f"DetectedFace bbox dimensions must be positive: width={w}, height={h}"
            )
        
        # Validate confidence
        if not 0.0 <= detected_face.confidence <= 1.0:
            raise LivenessDetectionError(
                f"DetectedFace confidence must be between 0.0 and 1.0, "
                f"got {detected_face.confidence}"
            )
