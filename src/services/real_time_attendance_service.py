"""
Real-Time Attendance Service and Integration Layer.

Connects the Computer Vision pipeline (Face Detection, Face Recognition, Liveness Detection)
to the Attendance Engine (AttendanceService, SessionService, StudentService).
Provides real-time frame processing, caching/debouncing, lifecycle verification,
and anti-spoofing enforcement for live video attendance monitoring.
"""

import time
from typing import Optional, List, Tuple, Dict, Any, Set
from pathlib import Path
import numpy as np

from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding, FaceEmbeddingError
from src.computer_vision.face_recognizer import FaceRecognizer, RecognitionResult
from src.computer_vision.detection_diagnostics import get_diagnostics
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessResult,
    LivenessState,
    ChallengeState,
    ChallengeType,
    LivenessDetectionError
)
from src.services.attendance_service import (
    AttendanceService,
    AttendanceStatus,
    DuplicateAttendanceError,
    SessionNotAcceptingAttendanceError,
    InvalidAttendanceDataError
)
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService
from src.services.enrollment_service import EnrollmentService
from src.utils.logger import get_logger
from src.utils.performance_timer import time_operation

logger = get_logger(__name__)


# ============================================================================
# Data Classes & Exceptions
# ============================================================================

class RealTimeAttendanceError(Exception):
    """Base exception for real-time attendance errors."""
    pass


class FaceProcessingResult:
    """
    Result of processing a single detected face in a video frame.

    Attributes:
        bbox: Bounding box as (x, y, width, height) in pixels
        detection_confidence: Face detection confidence score
        identity: Recognized identity string (e.g. student_id) or "Unknown"
        is_recognized: True if face matched a gallery identity
        recognition_distance: Distance to matched embedding
        student_id: Student identifier string if recognized and in DB, else None
        student_db_id: Student database primary key if found, else None
        student_name: Student full name if found in database, else None
        liveness_state: LivenessState enum (LIVE, SPOOF, UNKNOWN)
        liveness_confidence: Confidence score of liveness determination
        blink_count: Number of blinks detected
        min_blinks_required: Minimum blinks required for verification
        challenge_type: Current challenge type (NONE, TURN_LEFT, TURN_RIGHT)
        challenge_state: Current challenge state (INACTIVE, ACTIVE, COMPLETED, FAILED)
        challenge_progress: Progress percentage (0.0 to 1.0)
        attendance_marked: True if attendance was newly marked in this step
        attendance_id: Database ID of created attendance record if marked
        attendance_status: Status assigned ('present', 'late') or None
        already_marked: True if student was already marked in this session
        reason: Human-readable status/explanation message
    """

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        detection_confidence: float,
        identity: str = "Unknown",
        is_recognized: bool = False,
        recognition_distance: float = float('inf'),
        student_id: Optional[str] = None,
        student_db_id: Optional[int] = None,
        student_name: Optional[str] = None,
        liveness_state: LivenessState = LivenessState.UNKNOWN,
        liveness_confidence: float = 0.0,
        blink_count: int = 0,
        min_blinks_required: int = 0,
        challenge_type: str = "none",
        challenge_state: str = "inactive",
        challenge_progress: float = 0.0,
        attendance_marked: bool = False,
        attendance_id: Optional[int] = None,
        attendance_status: Optional[str] = None,
        already_marked: bool = False,
        reason: str = ""
    ):
        self.bbox = bbox
        self.detection_confidence = detection_confidence
        self.identity = identity
        self.is_recognized = is_recognized
        self.recognition_distance = recognition_distance
        self.student_id = student_id
        self.student_db_id = student_db_id
        self.student_name = student_name
        self.liveness_state = liveness_state
        self.liveness_confidence = liveness_confidence
        self.blink_count = blink_count
        self.min_blinks_required = min_blinks_required
        self.challenge_type = challenge_type
        self.challenge_state = challenge_state
        self.challenge_progress = challenge_progress
        self.attendance_marked = attendance_marked
        self.attendance_id = attendance_id
        self.attendance_status = attendance_status
        self.already_marked = already_marked
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"FaceProcessingResult(identity='{self.identity}', "
            f"recognized={self.is_recognized}, "
            f"liveness={self.liveness_state.value}, "
            f"marked={self.attendance_marked}, "
            f"status={self.attendance_status}, "
            f"reason='{self.reason}')"
        )


class RealTimeAttendanceResult:
    """
    Result of processing an entire video frame for attendance.

    Attributes:
        session_id: Active session ID processed
        faces_detected: Number of faces found in the frame
        processed_faces: List of FaceProcessingResult for each detected face
        session_active: Whether the session was active during processing
        marked_count: Number of new attendance records created in this frame
        timestamp: Time when frame was processed
    """

    def __init__(
        self,
        session_id: Optional[int],
        faces_detected: int,
        processed_faces: List[FaceProcessingResult],
        session_active: bool,
        marked_count: int = 0,
        timestamp: Optional[float] = None
    ):
        self.session_id = session_id
        self.faces_detected = faces_detected
        self.processed_faces = processed_faces
        self.session_active = session_active
        self.marked_count = marked_count
        self.timestamp = timestamp if timestamp is not None else time.time()

    def __repr__(self) -> str:
        return (
            f"RealTimeAttendanceResult(session={self.session_id}, "
            f"faces={self.faces_detected}, "
            f"marked={self.marked_count}, "
            f"active={self.session_active})"
        )


# ============================================================================
# Real-Time Attendance Manager / Service
# ============================================================================

class RealTimeAttendanceManager:
    """
    Orchestrates real-time attendance marking from video streams.

    Integrates CameraManager / video frames with FaceDetector, FaceEmbedder,
    FaceRecognizer, LivenessDetector, and AttendanceService.

    Key features:
    - Real-time video frame processing
    - Multi-stage pipeline: Detection -> Embedding -> Recognition -> Liveness -> Attendance
    - Enforces session lifecycle (only active sessions accept attendance)
    - Anti-spoofing / liveness requirement enforcement
    - In-memory caching and real-time debounce to avoid redundant DB queries
    - Comprehensive error handling and clean status reporting
    """

    def __init__(
        self,
        session_id: Optional[int] = None,
        database_path: Optional[Path] = None,
        attendance_service: Optional[AttendanceService] = None,
        session_service: Optional[SessionService] = None,
        student_service: Optional[StudentService] = None,
        face_detector: Optional[FaceDetector] = None,
        face_embedder: Optional[FaceEmbedder] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        liveness_detector: Optional[LivenessDetector] = None,
        require_liveness: bool = True,
        cooldown_seconds: float = 3.0,
        marked_by: str = 'system',
        head_turn_angle: Optional[float] = None
    ):
        """
        Initialize the RealTimeAttendanceManager.

        Args:
            session_id: Target session database ID (optional on init, can be set later)
            database_path: Database path (if None, uses default config)
            attendance_service: Injected AttendanceService instance (optional)
            session_service: Injected SessionService instance (optional)
            student_service: Injected StudentService instance (optional)
            face_detector: Injected FaceDetector instance (optional)
            face_embedder: Injected FaceEmbedder instance (optional)
            face_recognizer: Injected FaceRecognizer instance (optional)
            liveness_detector: Injected LivenessDetector instance (optional)
            require_liveness: If True, only LIVE faces can be marked present
            cooldown_seconds: Real-time debounce cooldown in seconds per student
            marked_by: Method identifier passed to attendance records ('system', 'cv_live', etc.)
            head_turn_angle: Required head turn angle in degrees for liveness challenges (default: from config)
        """
        self.database_path = database_path

        # Initialize services
        self.attendance_service = attendance_service or AttendanceService(database_path)
        self.session_service = session_service or SessionService(database_path)
        self.student_service = student_service or StudentService(database_path)

        # Initialize CV components (lazy/injected)
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.face_recognizer = face_recognizer
        self.liveness_detector = liveness_detector

        self.require_liveness = require_liveness
        self.cooldown_seconds = cooldown_seconds
        self.marked_by = marked_by
        self.head_turn_angle = head_turn_angle

        # In-memory student gallery for face recognition: List[Tuple[str, FaceEmbedding]]
        self._gallery: List[Tuple[str, FaceEmbedding]] = []

        # Active session state & in-memory caches for fast frame processing
        self._session_id: Optional[int] = None
        self._marked_students: Set[int] = set()  # Set of student_db_id already marked for current session
        self._cooldown_tracker: Dict[int, float] = {}  # student_db_id -> last_processed_timestamp

        # Session status cache to avoid DB query on every CV frame (~6/sec)
        self._session_status_cache: Optional[Dict[str, Any]] = None
        self._session_status_cache_time: float = 0.0
        self._session_status_cache_ttl: float = 1.0  # Cache for 1 second

        if session_id is not None:
            self.set_session(session_id)

        logger.info(
            f"RealTimeAttendanceManager initialized: session={self._session_id}, "
            f"require_liveness={self.require_liveness}, cooldown={self.cooldown_seconds}s"
        )

    # ------------------------------------------------------------------------
    # Gallery Management
    # ------------------------------------------------------------------------

    def set_gallery(self, gallery: List[Tuple[str, FaceEmbedding]]) -> None:
        """
        Set the known student gallery for face recognition.

        Args:
            gallery: List of (identity_string, FaceEmbedding) tuples
        """
        if gallery is None:
            self._gallery = []
        else:
            self._gallery = list(gallery)
        logger.info(f"Gallery updated with {len(self._gallery)} identities")

    def get_gallery(self) -> List[Tuple[str, FaceEmbedding]]:
        """Get the current student gallery."""
        return self._gallery

    # ------------------------------------------------------------------------
    # Session Management & Cache Pre-loading
    # ------------------------------------------------------------------------

    def set_session(self, session_id: int) -> None:
        """
        Set the active session and pre-load existing attendance records into the cache.

        Args:
            session_id: Session database ID

        Raises:
            InvalidAttendanceDataError: If session_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")

        self._session_id = session_id
        self._marked_students.clear()
        self._cooldown_tracker.clear()

        # Pre-load existing attendance records for this session into in-memory cache
        # This prevents redundant database lookups on every video frame
        try:
            existing_records = self.attendance_service.get_session_attendance(session_id)
            for record in existing_records:
                self._marked_students.add(record['student_id'])
            logger.info(
                f"Session {session_id} set. Pre-loaded {len(self._marked_students)} "
                f"existing attendance records into cache."
            )
        except Exception as e:
            logger.warning(f"Could not pre-load attendance for session {session_id}: {e}")

    @property
    def current_session_id(self) -> Optional[int]:
        """Get the current session ID."""
        return self._session_id

    def get_marked_students(self) -> Set[int]:
        """Get the set of student database IDs marked in the current session."""
        return set(self._marked_students)

    def reset_tracking(self) -> None:
        """Reset temporal liveness state and cooldown tracker."""
        self._cooldown_tracker.clear()
        self._session_status_cache = None
        self._session_status_cache_time = 0.0
        if self.liveness_detector is not None:
            self.liveness_detector.reset_temporal_state()
        logger.debug("Tracking state reset")

    def _get_session_status_cached(self, session_id: int) -> bool:
        """
        Get session active status with 1-second cache to avoid DB queries on every CV frame.

        Returns:
            True if session is ACTIVE, False otherwise
        """
        current_time = time.time()

        # Check if cache is valid
        if (self._session_status_cache is not None and
            current_time - self._session_status_cache_time < self._session_status_cache_ttl and
            self._session_status_cache.get('session_id') == session_id):
            return self._session_status_cache.get('is_active', False)

        # Cache miss or expired - query database
        try:
            with time_operation("session_status_db_query"):
                session = self.session_service.get_session(session_id)
                is_active = session is not None and session['status'] == SessionStatus.ACTIVE

            # Update cache
            self._session_status_cache = {
                'session_id': session_id,
                'is_active': is_active
            }
            self._session_status_cache_time = current_time

            return is_active

        except Exception as e:
            logger.error(f"Error checking session {session_id} status: {e}")
            return False

    # ------------------------------------------------------------------------
    # Lazy Initialization Helpers
    # ------------------------------------------------------------------------

    def _get_face_detector(self) -> FaceDetector:
        if self.face_detector is None:
            self.face_detector = FaceDetector()
        return self.face_detector

    def _get_face_embedder(self) -> FaceEmbedder:
        if self.face_embedder is None:
            self.face_embedder = FaceEmbedder()
        return self.face_embedder

    def _get_face_recognizer(self) -> FaceRecognizer:
        if self.face_recognizer is None:
            self.face_recognizer = FaceRecognizer()
        return self.face_recognizer

    def _get_liveness_detector(self) -> LivenessDetector:
        if self.liveness_detector is None:
            # Pass head_turn_angle if provided by UI, otherwise use config default
            kwargs = {}
            if self.head_turn_angle is not None:
                kwargs['head_turn_angle'] = self.head_turn_angle
            self.liveness_detector = LivenessDetector(**kwargs)
        elif self.head_turn_angle is not None and self.liveness_detector.head_turn_angle != self.head_turn_angle:
            # Update existing detector with new angle if changed
            self.liveness_detector.head_turn_angle = self.head_turn_angle
        return self.liveness_detector

    # ------------------------------------------------------------------------
    # Real-Time Frame Processing Pipeline
    # ------------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        session_id: Optional[int] = None,
        marked_at: Optional[str] = None
    ) -> RealTimeAttendanceResult:
        """
        Process a single video frame for face detection, recognition, liveness,
        and attendance marking.

        Complete Flow:
        1. Validate frame & session
        2. Detect faces using FaceDetector
        3. For each detected face:
           a. Crop face and compute embedding via FaceEmbedder
           b. Match against gallery via FaceRecognizer
           c. Verify liveness via LivenessDetector (EAR / temporal state / challenge)
           d. If recognized as known student + liveness verified + not already marked:
              - Call AttendanceService.record_attendance()
              - Cache student in _marked_students
        4. Return structured RealTimeAttendanceResult

        Args:
            frame: Input image array (BGR format from OpenCV)
            session_id: Session ID to process (if None, uses current session)
            marked_at: Optional explicit ISO timestamp for attendance

        Returns:
            RealTimeAttendanceResult with details for each detected face

        Raises:
            RealTimeAttendanceError: If input frame is invalid
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RealTimeAttendanceError("Invalid frame: frame must be a non-empty numpy array")

        target_session = session_id if session_id is not None else self._session_id
        if target_session is None:
            raise RealTimeAttendanceError("No session_id specified or configured")

        if target_session != self._session_id:
            self.set_session(target_session)

        # Check session lifecycle status (must be ACTIVE to accept attendance)
        # Uses 1-second cache to avoid DB query on every CV frame (~6/sec)
        session_active = self._get_session_status_cached(target_session)

        # Step 1: Detect faces
        face_detector = self._get_face_detector()
        detected_faces = face_detector.detect_faces(frame)

        if not detected_faces:
            # Diagnostics: record the post-detection stages for a missed frame,
            # which separates "not detected" from "detected but unusable".
            get_diagnostics().record(frame.shape, [], stage="no_detection")
            return RealTimeAttendanceResult(
                session_id=target_session,
                faces_detected=0,
                processed_faces=[],
                session_active=session_active,
                marked_count=0
            )

        # Step 2: Process each detected face
        processed_faces: List[FaceProcessingResult] = []
        newly_marked_count = 0
        current_time = time.time()

        face_embedder = self._get_face_embedder()
        face_recognizer = self._get_face_recognizer()
        liveness_detector = self._get_liveness_detector()

        for detected_face in detected_faces:
            face_result = self._process_single_face(
                frame=frame,
                detected_face=detected_face,
                target_session=target_session,
                session_active=session_active,
                face_embedder=face_embedder,
                face_recognizer=face_recognizer,
                liveness_detector=liveness_detector,
                current_time=current_time,
                marked_at=marked_at
            )

            if face_result.attendance_marked:
                newly_marked_count += 1

            processed_faces.append(face_result)

        return RealTimeAttendanceResult(
            session_id=target_session,
            faces_detected=len(detected_faces),
            processed_faces=processed_faces,
            session_active=session_active,
            marked_count=newly_marked_count,
            timestamp=current_time
        )

    def _process_single_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
        target_session: int,
        session_active: bool,
        face_embedder: FaceEmbedder,
        face_recognizer: FaceRecognizer,
        liveness_detector: LivenessDetector,
        current_time: float,
        marked_at: Optional[str]
    ) -> FaceProcessingResult:
        """Process a single face through recognition, liveness, and attendance marking."""
        bbox = detected_face.bbox
        det_conf = float(detected_face.confidence)
        x, y, w, h = bbox

        # Safe face crop
        frame_h, frame_w = frame.shape[:2]
        crop_x = max(0, min(x, frame_w - 1))
        crop_y = max(0, min(y, frame_h - 1))
        crop_w = max(1, min(w, frame_w - crop_x))
        crop_h = max(1, min(h, frame_h - crop_y))
        face_crop = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Step A: Generate embedding & Recognize face
        identity = "Unknown"
        is_recognized = False
        distance = float('inf')
        embedding_ok = False

        if face_crop.size > 0 and len(self._gallery) > 0:
            try:
                # PERFORMANCE: Use the MediaPipe face crop for dlib encoding.
                # Benchmark confirmed: crop (11ms) >> full_frame+bbox (248ms) >> full_frame (138ms)
                # The crop approach is 22x faster than the "optimized" bbox approach.
                with time_operation("face_embedding"):
                    embedding = face_embedder.generate_embedding(face_crop)

                if embedding is not None:
                    embedding_ok = True
                    with time_operation("face_recognition"):
                        rec_result = face_recognizer.recognize(embedding, self._gallery)
                        if rec_result.is_match and rec_result.identity != "Unknown":
                            identity = rec_result.identity
                            is_recognized = True
                            distance = rec_result.distance
            except FaceEmbeddingError as e:
                logger.debug(f"Face embedding generation skipped: {e}")
            except Exception as e:
                logger.error(f"Face recognition error: {e}")

        # Step B: Liveness analysis
        # CRITICAL FIX: Start challenge if enabled and not yet started
        if (liveness_detector.enable_challenges and
            liveness_detector.require_challenge and
            liveness_detector.current_challenge == ChallengeType.NONE):
            liveness_detector.start_new_challenge()

        # Update temporal state for the face
        # PERFORMANCE FIX: MediaPipe landmarks are now cached per-frame to avoid double inference
        with time_operation("liveness_ear_extraction"):
            ear = liveness_detector.extract_ear(frame, detected_face)

        with time_operation("liveness_temporal_update"):
            liveness_detector.update_temporal_state(ear, frame, detected_face)

        # Diagnostics: which stage consumed the face. This distinguishes
        # detection loss from recognition failure from liveness failure.
        get_diagnostics().record_stage(
            stage="recognition",
            bbox=bbox,
            detection_confidence=det_conf,
            embedding_ok=embedding_ok,
            recognized=is_recognized,
            distance=distance if is_recognized else None,
        )
        if ear is None:
            get_diagnostics().record_stage(
                stage="liveness",
                bbox=bbox,
                detection_confidence=det_conf,
                ear=None,
            )

        # BUG FIX: Properly evaluate liveness state using both blinks AND challenges
        # Previous implementation only checked blink_count manually, bypassing challenge verification
        # Now we check both requirements according to the detector's configuration
        blinks_detected = liveness_detector.blink_count
        min_blinks = liveness_detector.min_blinks
        blink_passed = blinks_detected >= min_blinks

        # Get challenge progress for UI display
        challenge_progress = liveness_detector.get_challenge_progress()

        # Check if challenges are enabled and required
        if liveness_detector.enable_challenges and liveness_detector.require_challenge:
            # CRITICAL FIX: Must verify BOTH blinks AND challenge completion for LIVE state
            challenge_state = liveness_detector.challenge_state
            challenge_completed = (challenge_state == ChallengeState.COMPLETED)

            # Only LIVE if BOTH conditions are met
            if blink_passed and challenge_completed:
                liveness_state = LivenessState.LIVE
                liveness_confidence = max(0.85, liveness_detector.liveness_threshold)
            else:
                # Insufficient evidence - need both blinks and challenge
                liveness_state = LivenessState.UNKNOWN
                liveness_confidence = 0.0
        else:
            # Challenges disabled - only check blinks
            if blink_passed:
                liveness_state = LivenessState.LIVE
                liveness_confidence = max(0.85, liveness_detector.liveness_threshold)
            else:
                liveness_state = LivenessState.UNKNOWN
                liveness_confidence = 0.0

        # Step C: Resolve student identity from database
        student_db_id = None
        student_id_str = None
        student_name = None

        if is_recognized:
            try:
                with time_operation("student_db_lookup"):
                    student_record = self.student_service.get_student_by_student_id(identity)
                if student_record:
                    # Check if student is active
                    if not student_record.get('is_active', 1):
                        return FaceProcessingResult(
                            bbox=bbox,
                            detection_confidence=det_conf,
                            identity=identity,
                            is_recognized=True,
                            recognition_distance=distance,
                            student_id=identity,
                            student_db_id=student_record['id'],
                            student_name=student_record.get('name'),
                            liveness_state=liveness_state,
                            liveness_confidence=liveness_confidence,
                            blink_count=blinks_detected,
                            min_blinks_required=min_blinks,
                            challenge_type=challenge_progress['challenge_type'],
                            challenge_state=challenge_progress['challenge_state'],
                            challenge_progress=challenge_progress['progress_percent'],
                            attendance_marked=False,
                            reason="Student is deactivated in system"
                        )
                    student_db_id = student_record['id']
                    student_id_str = student_record['student_id']
                    student_name = student_record.get('name')
                else:
                    return FaceProcessingResult(
                        bbox=bbox,
                        detection_confidence=det_conf,
                        identity=identity,
                        is_recognized=True,
                        recognition_distance=distance,
                        liveness_state=liveness_state,
                        liveness_confidence=liveness_confidence,
                        blink_count=blinks_detected,
                        min_blinks_required=min_blinks,
                        challenge_type=challenge_progress['challenge_type'],
                        challenge_state=challenge_progress['challenge_state'],
                        challenge_progress=challenge_progress['progress_percent'],
                        attendance_marked=False,
                        reason=f"Identity '{identity}' not found in student database"
                    )
            except Exception as e:
                logger.error(f"Student lookup error for '{identity}': {e}")
                return FaceProcessingResult(
                    bbox=bbox,
                    detection_confidence=det_conf,
                    identity=identity,
                    is_recognized=True,
                    recognition_distance=distance,
                    liveness_state=liveness_state,
                    liveness_confidence=liveness_confidence,
                    blink_count=blinks_detected,
                    min_blinks_required=min_blinks,
                    challenge_type=challenge_progress['challenge_type'],
                    challenge_state=challenge_progress['challenge_state'],
                    challenge_progress=challenge_progress['progress_percent'],
                    attendance_marked=False,
                    reason=f"Error looking up student: {e}"
                )
        else:
            # Unknown identity
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity="Unknown",
                is_recognized=False,
                recognition_distance=distance,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason="Unknown person - not enrolled"
            )

        # Step D: Verify Liveness Requirement
        if self.require_liveness and liveness_state != LivenessState.LIVE:
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason="Liveness verification pending / not confirmed"
            )

        # Step E: Verify Session Status
        if not session_active:
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason="Session is not active (scheduled or closed)"
            )

        # Step F: Check In-Memory Duplicate Cache & Cooldown
        if student_db_id in self._marked_students:
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                already_marked=True,
                reason="Attendance already recorded for this session"
            )

        # Check cooldown debounce to avoid rapid repeated DB calls
        last_time = self._cooldown_tracker.get(student_db_id, 0.0)
        if current_time - last_time < self.cooldown_seconds:
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason="Cooldown in progress"
            )

        self._cooldown_tracker[student_db_id] = current_time

        # Step G: Mark Attendance via AttendanceService
        try:
            attendance_id = self.attendance_service.record_attendance(
                session_id=target_session,
                student_id=student_db_id,
                marked_at=marked_at,
                detection_confidence=det_conf,
                liveness_score=liveness_confidence,
                marked_by=self.marked_by,
                notes="Marked via Real-Time Computer Vision"
            )

            # Update cache
            self._marked_students.add(student_db_id)

            # Fetch recorded status
            record = self.attendance_service.get_attendance(attendance_id)
            status = record['status'] if record else AttendanceStatus.PRESENT

            logger.info(
                f"Real-time attendance recorded: Student {identity} (DB: {student_db_id}) "
                f"in session {target_session} -> {status} (ID: {attendance_id})"
            )

            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=True,
                attendance_id=attendance_id,
                attendance_status=status,
                already_marked=False,
                reason=f"Attendance marked successfully ({status})"
            )

        except DuplicateAttendanceError:
            self._marked_students.add(student_db_id)
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                already_marked=True,
                reason="Attendance already recorded for this session"
            )
        except SessionNotAcceptingAttendanceError as e:
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason=str(e)
            )
        except Exception as e:
            logger.error(f"Failed to record real-time attendance for {identity}: {e}")
            return FaceProcessingResult(
                bbox=bbox,
                detection_confidence=det_conf,
                identity=identity,
                is_recognized=True,
                recognition_distance=distance,
                student_id=student_id_str,
                student_db_id=student_db_id,
                student_name=student_name,
                liveness_state=liveness_state,
                liveness_confidence=liveness_confidence,
                blink_count=blinks_detected,
                min_blinks_required=min_blinks,
                challenge_type=challenge_progress['challenge_type'],
                challenge_state=challenge_progress['challenge_state'],
                challenge_progress=challenge_progress['progress_percent'],
                attendance_marked=False,
                reason=f"Failed to record attendance: {e}"
            )


# Alias for backward compatibility / API flexibility
RealTimeAttendanceService = RealTimeAttendanceManager
