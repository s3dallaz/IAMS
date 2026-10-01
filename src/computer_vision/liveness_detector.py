"""
Liveness Detector for the Intelligent Attendance Monitoring System.

Provides liveness detection using passive blink detection, temporal analysis,
and active challenge-response to determine if a detected face belongs to a live person.

Phase 3A: Foundation & Architecture (complete)
Phase 3B: Blink detection and temporal analysis (complete)
Phase 3C: Active challenges and anti-spoofing (current)
"""

from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import time
import random
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


class EyeState(Enum):
    """
    Represents the state of eyes during blink detection.

    States:
        OPEN: Eyes are open (EAR above open threshold with hysteresis)
        CLOSED: Eyes are closed (EAR below close threshold with hysteresis)

    Note: Hysteresis prevents oscillation at threshold boundaries.
          Transition OPEN->CLOSED requires EAR < close_threshold
          Transition CLOSED->OPEN requires EAR >= open_threshold
    """
    OPEN = "open"
    CLOSING = "closing"  # Deprecated - kept for backward compatibility
    CLOSED = "closed"
    REOPENING = "reopening"  # Deprecated - kept for backward compatibility


class ChallengeType(Enum):
    """
    Types of active liveness challenges.

    Phase 3C: Challenge-response for anti-spoofing
    """
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NONE = "none"


class ChallengeState(Enum):
    """
    State of an active challenge.
    """
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class LivenessDetector:
    """
    Detects liveness in faces using passive blink detection and temporal analysis.

    Phase 3A provided:
    - API structure for temporal liveness detection
    - Input validation
    - Result representation
    - Configuration framework
    - Safe failure behavior (unknown state when insufficient evidence)

    Phase 3B provides:
    - MediaPipe Face Mesh integration for facial landmarks
    - Eye Aspect Ratio (EAR) calculation
    - Blink detection through temporal eye state tracking (hysteresis support)
    - Liveness classification based on detected blinks
    - Incremental per-frame processing and batch sequence processing

    Phase 3C provides (current):
    - Randomized head-turn challenges (left/right)
    - Challenge-response anti-spoofing
    - Combined blink + challenge verification
    - Prevention of static photo attacks

    Current Limitations (Phase 3C):
    - Head-turn challenges prevent static photos
    - Prerecorded videos with head movements may still pass
    - No texture/frequency analysis for video replay detection
    - This is an active challenge method suitable for preventing photo spoofing
    """

    # MediaPipe Face Mesh landmark indices for eyes
    # Left eye: 6 landmarks forming eye contour
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    # Right eye: 6 landmarks forming eye contour
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

    # MediaPipe Face Mesh landmark indices for head pose estimation
    # Nose tip
    NOSE_TIP = 1
    # Chin
    CHIN = 152
    # Left eye outer corner
    LEFT_EYE_OUTER = 263
    # Right eye outer corner
    RIGHT_EYE_OUTER = 33
    # Left ear
    LEFT_EAR_TRAGION = 234
    # Right ear
    RIGHT_EAR_TRAGION = 454

    def __init__(
        self,
        min_frames: Optional[int] = None,
        liveness_threshold: Optional[float] = None,
        temporal_window_seconds: Optional[float] = None,
        ear_threshold: Optional[float] = None,
        ear_close_threshold: Optional[float] = None,
        ear_open_threshold: Optional[float] = None,
        ear_consecutive_frames: Optional[int] = None,
        min_blinks: Optional[int] = None,
        ear_smoothing_window: Optional[int] = None,
        enable_challenges: Optional[bool] = None,
        challenge_timeout: Optional[float] = None,
        head_turn_angle: Optional[float] = None,
        head_center_tolerance: Optional[float] = None,
        head_hold_frames: Optional[int] = None,
        require_challenge: Optional[bool] = None
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
            ear_threshold: Single EAR threshold (for backward compatibility).
            ear_close_threshold: Eye Aspect Ratio threshold for detecting closed eyes.
                                If None, uses config.LIVENESS_EAR_CLOSE_THRESHOLD (default: 0.21)
            ear_open_threshold: Eye Aspect Ratio threshold for detecting open eyes (hysteresis).
                               If None, uses config.LIVENESS_EAR_OPEN_THRESHOLD (default: 0.25)
            ear_consecutive_frames: Consecutive frames below EAR threshold to confirm closure.
                                   If None, uses config.LIVENESS_EAR_CONSECUTIVE_FRAMES (default: 1)
            min_blinks: Minimum blinks required for LIVE classification.
                       If None, uses config.LIVENESS_MIN_BLINKS (default: 1)
            ear_smoothing_window: Number of frames to smooth EAR values.
                                 If None, uses config.LIVENESS_EAR_SMOOTHING_WINDOW (default: 0)
            enable_challenges: Enable active challenge-response detection.
                              If None, uses config.LIVENESS_ENABLE_ACTIVE_CHALLENGES (default: True)
            challenge_timeout: Challenge timeout in seconds.
                              If None, uses config.LIVENESS_CHALLENGE_TIMEOUT (default: 5.0)
            head_turn_angle: Required head turn angle in degrees for challenges.
                            If None, uses config.LIVENESS_HEAD_TURN_ANGLE (default: 25.0)
            head_center_tolerance: Maximum deviation from center in degrees.
                                  If None, uses config.LIVENESS_HEAD_CENTER_TOLERANCE (default: 15.0)
            head_hold_frames: Minimum frames to hold head in target position.
                             If None, uses config.LIVENESS_HEAD_HOLD_FRAMES (default: 3)
            require_challenge: Require challenge completion for LIVE state.
                              If None, uses config.LIVENESS_REQUIRE_CHALLENGE (default: True)
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

        # Handle ear_threshold for backward compatibility
        if ear_threshold is not None:
            if not 0.0 <= ear_threshold <= 1.0:
                raise ValueError(
                    f"ear_threshold must be between 0.0 and 1.0, got {ear_threshold}"
                )
            self.ear_close_threshold = (
                ear_close_threshold if ear_close_threshold is not None else ear_threshold
            )
            self.ear_open_threshold = (
                ear_open_threshold if ear_open_threshold is not None else ear_threshold
            )
        else:
            self.ear_close_threshold = (
                ear_close_threshold
                if ear_close_threshold is not None
                else getattr(config, "LIVENESS_EAR_CLOSE_THRESHOLD", getattr(config, "LIVENESS_EAR_THRESHOLD", 0.21))
            )
            self.ear_open_threshold = (
                ear_open_threshold
                if ear_open_threshold is not None
                else getattr(config, "LIVENESS_EAR_OPEN_THRESHOLD", getattr(config, "LIVENESS_EAR_THRESHOLD", 0.25))
            )

        self.ear_consecutive_frames = (
            ear_consecutive_frames
            if ear_consecutive_frames is not None
            else config.LIVENESS_EAR_CONSECUTIVE_FRAMES
        )
        self.min_blinks = (
            min_blinks
            if min_blinks is not None
            else config.LIVENESS_MIN_BLINKS
        )
        self.ear_smoothing_window = (
            ear_smoothing_window
            if ear_smoothing_window is not None
            else getattr(config, "LIVENESS_EAR_SMOOTHING_WINDOW", 0)
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

        if not 0.0 <= self.ear_close_threshold <= 1.0:
            raise ValueError(
                f"ear_close_threshold must be between 0.0 and 1.0, got {self.ear_close_threshold}"
            )

        if not 0.0 <= self.ear_open_threshold <= 1.0:
            raise ValueError(
                f"ear_open_threshold must be between 0.0 and 1.0, got {self.ear_open_threshold}"
            )

        if self.ear_open_threshold < self.ear_close_threshold:
            raise ValueError(
                f"ear_open_threshold ({self.ear_open_threshold}) must be greater than or equal to "
                f"ear_close_threshold ({self.ear_close_threshold})"
            )

        if self.ear_consecutive_frames < 1:
            raise ValueError(
                f"ear_consecutive_frames must be at least 1, got {self.ear_consecutive_frames}"
            )

        if self.min_blinks < 0:
            raise ValueError(
                f"min_blinks cannot be negative, got {self.min_blinks}"
            )

        if self.ear_smoothing_window < 0:
            raise ValueError(
                f"ear_smoothing_window cannot be negative, got {self.ear_smoothing_window}"
            )

        # Phase 3C: Challenge configuration
        self.enable_challenges = (
            enable_challenges
            if enable_challenges is not None
            else getattr(config, "LIVENESS_ENABLE_ACTIVE_CHALLENGES", True)
        )
        self.challenge_timeout = (
            challenge_timeout
            if challenge_timeout is not None
            else getattr(config, "LIVENESS_CHALLENGE_TIMEOUT", 5.0)
        )
        self.head_turn_angle = (
            head_turn_angle
            if head_turn_angle is not None
            else getattr(config, "LIVENESS_HEAD_TURN_ANGLE", 25.0)
        )
        self.head_center_tolerance = (
            head_center_tolerance
            if head_center_tolerance is not None
            else getattr(config, "LIVENESS_HEAD_CENTER_TOLERANCE", 15.0)
        )
        self.head_hold_frames = (
            head_hold_frames
            if head_hold_frames is not None
            else getattr(config, "LIVENESS_HEAD_HOLD_FRAMES", 3)
        )
        self.require_challenge = (
            require_challenge
            if require_challenge is not None
            else getattr(config, "LIVENESS_REQUIRE_CHALLENGE", True)
        )

        # Validate challenge configuration
        if self.challenge_timeout <= 0:
            raise ValueError(f"challenge_timeout must be positive, got {self.challenge_timeout}")
        if self.head_turn_angle <= 0:
            raise ValueError(f"head_turn_angle must be positive, got {self.head_turn_angle}")
        if self.head_center_tolerance < 0:
            raise ValueError(f"head_center_tolerance cannot be negative, got {self.head_center_tolerance}")
        if self.head_hold_frames < 1:
            raise ValueError(f"head_hold_frames must be at least 1, got {self.head_hold_frames}")

        self.logger = get_logger(__name__)

        # MediaPipe Face Landmarker (lazy initialization)
        self._landmarker = None

        # Temporal state for blink detection
        self._reset_temporal_state()

        # Phase 3C: Challenge state
        self._reset_challenge_state()

        self.logger.info(
            f"LivenessDetector initialized (Phase 3C - Active Challenges): "
            f"min_frames={self.min_frames}, "
            f"threshold={self.liveness_threshold}, "
            f"temporal_window={self.temporal_window_seconds}s, "
            f"ear_close={self.ear_close_threshold}, "
            f"ear_open={self.ear_open_threshold}, "
            f"ear_consecutive_frames={self.ear_consecutive_frames}, "
            f"ear_smoothing={self.ear_smoothing_window}, "
            f"min_blinks={self.min_blinks}, "
            f"challenges_enabled={self.enable_challenges}, "
            f"require_challenge={self.require_challenge}"
        )

    @property
    def ear_threshold(self) -> float:
        """EAR threshold property for backward compatibility."""
        return self.ear_close_threshold

    @property
    def blink_count(self) -> int:
        """Current number of detected blinks."""
        return self._blink_count

    @property
    def eye_state(self) -> EyeState:
        """Current eye state."""
        return self._eye_state

    def reset_temporal_state(self) -> None:
        """Public method to reset temporal state for blink detection."""
        self._reset_temporal_state()
        self._reset_challenge_state()

    def _reset_temporal_state(self) -> None:
        """Reset internal temporal state for blink detection."""
        self._blink_count = 0
        self._eye_state = EyeState.OPEN
        self._frames_below_threshold = 0
        self._frames_above_threshold = 0
        self._ear_history = []  # For smoothing

    def _reset_challenge_state(self) -> None:
        """Reset challenge state (Phase 3C)."""
        self._current_challenge = ChallengeType.NONE
        self._challenge_state = ChallengeState.INACTIVE
        self._challenge_start_time = None
        self._challenge_frames_held = 0
        self._challenge_completed = False

    @property
    def current_challenge(self) -> ChallengeType:
        """Current active challenge."""
        return self._current_challenge

    @property
    def challenge_state(self) -> ChallengeState:
        """Current challenge state."""
        return self._challenge_state

    def start_new_challenge(self) -> ChallengeType:
        """
        Start a new randomized challenge (Phase 3C).

        Returns:
            The challenge type assigned
        """
        if not self.enable_challenges:
            return ChallengeType.NONE

        # Randomly select a challenge
        challenges = [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT]
        self._current_challenge = random.choice(challenges)
        self._challenge_state = ChallengeState.ACTIVE
        self._challenge_start_time = time.time()
        self._challenge_frames_held = 0
        self._challenge_completed = False

        self.logger.info(f"New challenge started: {self._current_challenge.value}")
        return self._current_challenge

    def get_challenge_time_remaining(self) -> float:
        """Get remaining time for current challenge in seconds."""
        if self._challenge_start_time is None:
            return 0.0
        elapsed = time.time() - self._challenge_start_time
        return max(0.0, self.challenge_timeout - elapsed)

    def _initialize_landmarker(self) -> None:
        """
        Initialize MediaPipe Face Landmarker (lazy initialization).

        Called on first use to avoid unnecessary resource allocation.
        """
        if self._landmarker is not None:
            return

        try:
            # Configure MediaPipe Face Landmarker
            base_options = python.BaseOptions(
                model_asset_path=str(self._get_landmarker_model_path())
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,  # We only analyze one face at a time
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self.logger.info("MediaPipe Face Landmarker initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Face Landmarker: {e}")
            self._landmarker = None

    def _get_landmarker_model_path(self) -> str:
        """
        Get path to MediaPipe Face Landmarker model.

        Returns:
            Path to face_landmarker.task model file

        Raises:
            FileNotFoundError: If model file not found
        """
        from pathlib import Path

        # Check project root for model file
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "face_landmarker.task"

        if model_path.exists():
            return str(model_path)

        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model not found at {model_path}. "
            "Please download from: "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    def _extract_eye_landmarks(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """
        Extract eye landmarks from a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            detected_face: DetectedFace with bbox information

        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks) as lists of (x, y) tuples,
            or None if landmarks cannot be extracted
        """
        if self._landmarker is None:
            self._initialize_landmarker()

        if self._landmarker is None:
            self.logger.warning("Face Landmarker not initialized, cannot extract landmarks")
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = self._convert_to_rgb(frame)

            # Create MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

            # Detect landmarks
            detection_result = self._landmarker.detect(mp_image)

            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                self.logger.debug("No face landmarks detected")
                return None

            # Use first face's landmarks
            face_landmarks = detection_result.face_landmarks[0]

            # Extract eye landmarks in pixel coordinates
            height, width = frame.shape[:2]

            left_eye = []
            for idx in self.LEFT_EYE_LANDMARKS:
                landmark = face_landmarks[idx]
                x = landmark.x * width
                y = landmark.y * height
                left_eye.append((x, y))

            right_eye = []
            for idx in self.RIGHT_EYE_LANDMARKS:
                landmark = face_landmarks[idx]
                x = landmark.x * width
                y = landmark.y * height
                right_eye.append((x, y))

            return (left_eye, right_eye)

        except Exception as e:
            self.logger.error(f"Error extracting eye landmarks: {e}")
            return None

    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio (EAR) for an eye.

        EAR formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Where p1-p6 are the 6 eye landmarks in order.

        Args:
            eye_landmarks: List of 6 (x, y) tuples representing eye contour points

        Returns:
            EAR value (typically 0.15-0.4), or None if calculation fails
        """
        if eye_landmarks is None or len(eye_landmarks) != 6:
            return None

        try:
            p1, p2, p3, p4, p5, p6 = eye_landmarks

            def distance(pt1, pt2):
                return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

            vertical_1 = distance(p2, p6)
            vertical_2 = distance(p3, p5)
            horizontal = distance(p1, p4)

            if horizontal == 0:
                return None

            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return float(ear)

        except Exception as e:
            self.logger.debug(f"Error calculating EAR: {e}")
            return None

    def extract_ear(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> Optional[float]:
        """
        Extract eye landmarks and calculate average Eye Aspect Ratio (EAR) for a single frame.

        Args:
            frame: Input frame as numpy array (BGR format)
            detected_face: DetectedFace object containing face location and confidence

        Returns:
            Average EAR float for both eyes, or None if extraction fails
        """
        eye_data = self._extract_eye_landmarks(frame, detected_face)
        if eye_data is None:
            return None

        left_eye, right_eye = eye_data
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)

        if left_ear is None or right_ear is None:
            return None

        return (left_ear + right_ear) / 2.0

    def _estimate_head_yaw(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> Optional[float]:
        """
        Estimate head yaw angle (left/right rotation) from facial landmarks.

        Phase 3C: Used for head-turn challenge validation.

        Args:
            frame: Input frame as numpy array (BGR format)
            detected_face: DetectedFace object

        Returns:
            Yaw angle in degrees (negative=left, positive=right), or None if estimation fails
        """
        if self._landmarker is None:
            self._initialize_landmarker()

        if self._landmarker is None:
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = self._convert_to_rgb(frame)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

            # Detect landmarks
            detection_result = self._landmarker.detect(mp_image)

            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                return None

            # Use first face's landmarks
            face_landmarks = detection_result.face_landmarks[0]
            height, width = frame.shape[:2]

            # Get key landmarks for head pose estimation
            nose = face_landmarks[self.NOSE_TIP]
            left_eye_outer = face_landmarks[self.LEFT_EYE_OUTER]
            right_eye_outer = face_landmarks[self.RIGHT_EYE_OUTER]

            # Calculate horizontal positions
            nose_x = nose.x * width
            left_eye_x = left_eye_outer.x * width
            right_eye_x = right_eye_outer.x * width

            # Calculate eye center
            eye_center_x = (left_eye_x + right_eye_x) / 2.0

            # Calculate eye span
            eye_span = abs(right_eye_x - left_eye_x)

            if eye_span == 0:
                return None

            # Calculate nose deviation from eye center
            nose_deviation = nose_x - eye_center_x

            # Estimate yaw angle
            # Positive deviation = turned right, negative = turned left
            # Scale factor empirically determined for typical face geometry
            yaw_angle = (nose_deviation / eye_span) * 60.0  # Scale to degrees

            return float(yaw_angle)

        except Exception as e:
            self.logger.debug(f"Error estimating head yaw: {e}")
            return None

    def _check_challenge_progress(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> None:
        """
        Check progress on current active challenge (Phase 3C).

        Args:
            frame: Input frame
            detected_face: DetectedFace object
        """
        if not self.enable_challenges:
            return

        if self._challenge_state != ChallengeState.ACTIVE:
            return

        # Check timeout
        if self._challenge_start_time is not None:
            elapsed = time.time() - self._challenge_start_time
            if elapsed > self.challenge_timeout:
                self._challenge_state = ChallengeState.TIMEOUT
                self.logger.info(f"Challenge {self._current_challenge.value} timed out")
                return

        # Estimate head pose
        yaw = self._estimate_head_yaw(frame, detected_face)

        if yaw is None:
            # Can't estimate pose - reset hold counter
            self._challenge_frames_held = 0
            return

        # Check if head is in target position
        target_met = False

        if self._current_challenge == ChallengeType.TURN_LEFT:
            # Left turn requires negative yaw (beyond threshold)
            if yaw < -self.head_turn_angle:
                target_met = True
        elif self._current_challenge == ChallengeType.TURN_RIGHT:
            # Right turn requires positive yaw (beyond threshold)
            if yaw > self.head_turn_angle:
                target_met = True

        # Update hold counter
        if target_met:
            self._challenge_frames_held += 1

            # Check if held long enough
            if self._challenge_frames_held >= self.head_hold_frames:
                self._challenge_state = ChallengeState.COMPLETED
                self._challenge_completed = True
                self.logger.info(f"Challenge {self._current_challenge.value} completed!")
        else:
            # Reset if not in position
            self._challenge_frames_held = 0

    def _smooth_ear(self, ear: float) -> float:
        """
        Apply temporal smoothing to EAR value if configured.

        Args:
            ear: Current EAR value

        Returns:
            Smoothed EAR value
        """
        if self.ear_smoothing_window <= 1:
            return ear

        self._ear_history.append(ear)

        if len(self._ear_history) > self.ear_smoothing_window:
            self._ear_history.pop(0)

        return sum(self._ear_history) / len(self._ear_history)

    def update_temporal_state(self, ear: Optional[float], frame: Optional[np.ndarray] = None, detected_face: Optional[DetectedFace] = None) -> bool:
        """
        Update temporal eye state and detect if a complete blink occurred.
        Phase 3C: Also updates challenge progress if frame and face provided.

        Handles missing landmarks / face loss safely.

        Args:
            ear: Current Eye Aspect Ratio value, or None if face/landmarks lost
            frame: Optional frame for challenge validation (Phase 3C)
            detected_face: Optional DetectedFace for challenge validation (Phase 3C)

        Returns:
            True if a complete blink cycle just finished on this frame, False otherwise
        """
        if ear is None:
            # Face or landmarks temporarily lost
            self._frames_below_threshold = 0
            self._frames_above_threshold = 0
            if self._eye_state == EyeState.CLOSING:
                self._eye_state = EyeState.OPEN
            elif self._eye_state == EyeState.REOPENING:
                self._eye_state = EyeState.CLOSED

            # Phase 3C: Reset challenge on face loss
            if self._challenge_state == ChallengeState.ACTIVE:
                self._challenge_frames_held = 0

            return False

        # Update blink detection
        blink_result = self._detect_blink_from_ear(ear)

        # Phase 3C: Update challenge progress
        if self.enable_challenges and frame is not None and detected_face is not None:
            self._check_challenge_progress(frame, detected_face)

        return blink_result

    def _detect_blink_from_ear(self, ear: float) -> bool:
        """
        Update eye state and detect if a blink occurred based on EAR with hysteresis.

        A blink cycle is: OPEN -> CLOSED -> OPEN.
        Uses hysteresis to avoid oscillation at threshold boundaries.

        Args:
            ear: Current Eye Aspect Ratio value

        Returns:
            True if a complete blink was just detected, False otherwise
        """
        smoothed_ear = self._smooth_ear(ear)
        blink_detected = False

        if self._eye_state == EyeState.OPEN:
            # Eyes are currently open - check for closure
            if smoothed_ear < self.ear_close_threshold:
                self._frames_below_threshold += 1
                if self._frames_below_threshold >= self.ear_consecutive_frames:
                    # Confirmed closure - transition to CLOSED
                    self._eye_state = EyeState.CLOSED
                    self._frames_above_threshold = 0
                    self.logger.debug(f"Eyes CLOSED (EAR: {smoothed_ear:.3f})")
            else:
                # Still open
                self._frames_below_threshold = 0

        elif self._eye_state == EyeState.CLOSED:
            # Eyes are currently closed - check for reopening
            if smoothed_ear >= self.ear_open_threshold:
                self._frames_above_threshold += 1
                if self._frames_above_threshold >= self.ear_consecutive_frames:
                    # Confirmed reopening - BLINK COMPLETE
                    self._blink_count += 1
                    blink_detected = True
                    self._eye_state = EyeState.OPEN
                    self._frames_below_threshold = 0
                    self._frames_above_threshold = 0
                    self.logger.debug(f"BLINK #{self._blink_count} detected! (EAR: {smoothed_ear:.3f})")
            elif smoothed_ear < self.ear_close_threshold:
                # Still closed - reset reopening counter
                self._frames_above_threshold = 0
            # else: in hysteresis zone between thresholds - maintain CLOSED state

        return blink_detected

    def _convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB format if needed.

        Args:
            image: Input image (assumed BGR from OpenCV)

        Returns:
            RGB image
        """
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)

        if image.shape[2] == 3:
            return image[:, :, ::-1]  # BGR to RGB

        return image

    def detect_liveness(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace
    ) -> LivenessResult:
        """
        Detect liveness for a single frame observation.

        Phase 3B Note: Single-frame liveness detection has limited reliability.
        Use detect_liveness_sequence() or incremental tracking for robust temporal analysis.

        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            detected_face: DetectedFace object containing face location and confidence

        Returns:
            LivenessResult with state, confidence, and reasoning

        Raises:
            LivenessDetectionError: If input validation fails
        """
        self._validate_frame(frame)
        self._validate_detected_face(detected_face)

        self.logger.debug(
            "Phase 3B: Single-frame liveness detection not reliable. "
            "Use detect_liveness_sequence() for temporal analysis."
        )

        return LivenessResult(
            state=LivenessState.UNKNOWN,
            confidence=0.0,
            frame_count=1,
            reason="Single frame insufficient - use detect_liveness_sequence() for temporal analysis"
        )

    def detect_liveness_sequence(
        self,
        frames: List[np.ndarray],
        detected_faces: List[DetectedFace]
    ) -> LivenessResult:
        """
        Detect liveness across a sequence of frames (temporal analysis).

        Phase 3B implements blink detection through Eye Aspect Ratio (EAR) analysis.
        Tracks eye states across frames to detect complete blink cycles.

        Args:
            frames: List of consecutive frames as numpy arrays
            detected_faces: List of DetectedFace objects for each frame

        Returns:
            LivenessResult based on temporal blink analysis

        Raises:
            LivenessDetectionError: If inputs are invalid or mismatched
        """
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

        # Reset temporal state for evaluating new batch sequence
        self._reset_temporal_state()

        frames_with_landmarks = 0

        for idx, (frame, face) in enumerate(zip(frames, detected_faces)):
            ear = self.extract_ear(frame, face)

            if ear is None:
                self.logger.debug(f"Frame {idx}: No valid landmarks or EAR calculation failed")
                self.update_temporal_state(None)
                continue

            frames_with_landmarks += 1
            self._detect_blink_from_ear(ear)

        blinks_detected = self._blink_count

        if frames_with_landmarks < self.min_frames * 0.5:
            return LivenessResult(
                state=LivenessState.UNKNOWN,
                confidence=0.0,
                frame_count=frame_count,
                reason=f"Insufficient valid landmarks: {frames_with_landmarks}/{frame_count} frames"
            )

        if blinks_detected >= self.min_blinks:
            confidence = min(
                self.liveness_threshold + (blinks_detected - self.min_blinks) * 0.05,
                1.0
            )
            return LivenessResult(
                state=LivenessState.LIVE,
                confidence=confidence,
                frame_count=frame_count,
                reason=f"Blinks detected: {blinks_detected} (threshold: {self.min_blinks})"
            )
        else:
            return LivenessResult(
                state=LivenessState.UNKNOWN,
                confidence=0.0,
                frame_count=frame_count,
                reason=f"Insufficient blinks: {blinks_detected} < {self.min_blinks} required"
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

        if len(frame.shape) not in [2, 3]:
            raise LivenessDetectionError(
                f"Invalid frame shape: {frame.shape}. "
                "Expected 2D (grayscale) or 3D (color) array"
            )

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

        if detected_face.bbox is None or len(detected_face.bbox) != 4:
            raise LivenessDetectionError(
                "DetectedFace must have valid bbox (x, y, width, height)"
            )

        x, y, w, h = detected_face.bbox
        if w <= 0 or h <= 0:
            raise LivenessDetectionError(
                f"DetectedFace bbox dimensions must be positive: width={w}, height={h}"
            )

        if not 0.0 <= detected_face.confidence <= 1.0:
            raise LivenessDetectionError(
                f"DetectedFace confidence must be between 0.0 and 1.0, "
                f"got {detected_face.confidence}"
            )

    def close(self) -> None:
        """Release MediaPipe landmarker resources."""
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            try:
                landmarker.close()
                self.logger.info("Face Landmarker closed")
            except Exception as e:
                self.logger.error(f"Error closing Face Landmarker: {e}")
            finally:
                self._landmarker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Destructor."""
        self.close()
