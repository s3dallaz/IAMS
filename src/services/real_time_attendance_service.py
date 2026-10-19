"""
Real-Time Attendance Service and Integration Layer.

Connects the Computer Vision pipeline (Face Detection, Face Recognition, Liveness Detection)
to the Attendance Engine (AttendanceService, SessionService, StudentService).
Provides real-time frame processing, caching/debouncing, lifecycle verification,
and anti-spoofing enforcement for live video attendance monitoring.
"""

import random
import re
import time
import threading
from collections import deque
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, List, Tuple, Dict, Any, Set
from pathlib import Path

import config
import cv2
import numpy as np

from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding, FaceEmbeddingError
from src.computer_vision.face_recognizer import FaceRecognizer, RecognitionResult
from src.computer_vision.detection_diagnostics import get_diagnostics
from src.computer_vision.spoof_detection import AntiSpoofDetector, ScreenDetector, crop_around_face
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
        # Filled in by the manager for the face in view (None = not available)
        self.spoof_score: Optional[float] = None             # mean real-face probability
        self.challenge_seconds_left: Optional[float] = None  # time left for the challenge

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
        multiple_faces: True if more than one face was in view, so nobody was
            recognised or marked (one person at a time)
        message: Frame-level message for the UI (e.g. MULTIPLE_FACES_MESSAGE)
    """

    def __init__(
        self,
        session_id: Optional[int],
        faces_detected: int,
        processed_faces: List[FaceProcessingResult],
        session_active: bool,
        marked_count: int = 0,
        timestamp: Optional[float] = None,
        multiple_faces: bool = False,
        message: str = ""
    ):
        self.session_id = session_id
        self.faces_detected = faces_detected
        self.processed_faces = processed_faces
        self.session_active = session_active
        self.marked_count = marked_count
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.multiple_faces = multiple_faces
        self.message = message

    def __repr__(self) -> str:
        return (
            f"RealTimeAttendanceResult(session={self.session_id}, "
            f"faces={self.faces_detected}, "
            f"marked={self.marked_count}, "
            f"active={self.session_active})"
        )


# Shown when more than one face is in view. Attendance works like a kiosk:
# one person at a time, so one person's blink or head turn can never verify
# somebody else (for example a photo held up next to a live face).
MULTIPLE_FACES_MESSAGE = "Multiple faces detected — please step up one at a time"


def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes: 1.0 = identical, 0.0 = no overlap."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


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
        anti_spoof_detector: Optional[AntiSpoofDetector] = None,
        screen_detector: Optional[ScreenDetector] = None,
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
            anti_spoof_detector: Injected AntiSpoofDetector instance (optional)
            screen_detector: Injected ScreenDetector instance (optional)
            require_liveness: If True, only LIVE faces that pass the spoof checks can be marked present
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
        self.anti_spoof_detector = anti_spoof_detector
        self.screen_detector = screen_detector

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

        # Per-session student-record cache: identity_string -> student dict (or None if not found).
        # Avoids a SQLite round-trip on every CV frame for the same recognized identity.
        # Cleared whenever the session changes so stale data cannot persist across sessions.
        self._student_record_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        # Recognise once, then follow the face.
        # The dlib embedding costs ~230 ms on the CPU and blocks the preview, so
        # it is not recomputed for every processed frame. While exactly one face
        # stays in view and its box keeps overlapping the previous one, its
        # identity is reused. It is recomputed when the face is lost, a second
        # face appears, the box jumps, or track_reverify_seconds have passed.
        # SAFETY: attendance is never marked from a reused identity - the face
        # is recognised again on the marking frame and must still match.
        self._tracked_face: Optional[Dict[str, Any]] = None
        self.track_reverify_seconds: float = 2.0
        self.track_min_iou: float = 0.5

        # Liveness belongs to one person. The detector's blink count and
        # head-turn challenge are reset after a student is marked, when the
        # face leaves the view, when several faces are in view, and when the
        # identity in front of the camera changes. Without this, the first
        # student's blink + turn would verify everyone who came after them.
        # Holds the identity (including "Unknown") the current liveness
        # evidence was collected for; None = nobody yet.
        self._liveness_identity: Optional[str] = None

        # Spoof evidence for the same person, reset together with liveness.
        # Blinks and head turns can be replayed from a video, so the face must
        # also look real to the anti-spoof model, and no phone / laptop / TV
        # may be seen covering it (sticky until the person leaves the view).
        self._spoof_scores: deque = deque(maxlen=max(1, config.ANTI_SPOOF_WINDOW))
        self._screen_label: str = ""
        self._spoof_flagged: bool = False     # anti-spoof model saw a photo / screen (sticky)
        self._last_screen_check: float = 0.0# screen detection every SCREEN_CHECK_INTERVAL
        self._last_spoof_log: float = 0.0     # SPOOF CHECK diagnostics, once per second
# When the current random challenge started; it must be finished
        # within config.LIVENESS_CHALLENGE_TIMEOUT. None = no challenge yet.
        self._challenge_started_at: Optional[float] = None

        # Per-frame liveness. Blinks last 100-250 ms, so they are measured by
        # update_liveness() on EVERY camera frame, while recognition, spoof
        # checks and marking run less often (process_frame / the background
        # worker). The page sets fast_liveness = True when it calls
        # update_liveness(); process_frame() then only reads the evidence.
        # The lock guards the liveness detector state shared by both paths.
        self.fast_liveness: bool = False
        self._liveness_lock = threading.RLock()
        self._last_fast_ear: Optional[float] = None
        # The single face seen by the last process_frame(); a jump of its box
        # (IoU below person_change_iou) means a different person stepped in
        self._last_face: Optional[DetectedFace] = None
        self.person_change_iou: float = 0.1
        # When the face was last lost (grace period before liveness resets)
        self._face_missing_since: Optional[float] = None
        # Head-turn part of the liveness test: None = detector/config default,
        # True/False = set by the page ("Require head turn" switch)
        self.require_head_turn: Optional[bool] = None

        # Session status cache to avoid DB query on every CV frame (~6/sec)
        self._session_status_cache: Optional[Dict[str, Any]] = None
        self._session_status_cache_time: float = 0.0
        self._session_status_cache_ttl: float = 1.0  # Cache for 1 second

        # Async processing for camera smoothness
        # Allows frame capture/display to continue while CV processing runs in background
        self._bg_worker_enabled: bool = False
        self._bg_worker_thread: Optional[threading.Thread] = None
        self._bg_pending_frame: Optional[Tuple[np.ndarray, int, Optional[str]]] = None  # (frame, session_id, marked_at)
        self._bg_latest_result: Optional[RealTimeAttendanceResult] = None
        self._bg_lock = threading.Lock()
        self._bg_stop_flag = threading.Event()
        self._bg_frame_ready = threading.Event()  # wakes the worker when a frame is submitted
        # Shortest time between two background processing runs (seconds).
        # Recognition does not need every frame; a pause leaves CPU time and
        # the GIL to the camera and per-frame liveness threads.
        self.min_processing_interval: float = 0.0

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
        # Identities may have changed (enrolment / deactivation): re-recognise
        self._tracked_face = None
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
        self._student_record_cache.clear()
        self._tracked_face = None

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
        self._reset_liveness()
        logger.debug("Tracking state reset")

    def _reset_liveness(self) -> None:
        """Start liveness from scratch: no blinks, no challenge, no spoof evidence, no owner."""
        with self._liveness_lock:
            if self.liveness_detector is not None:
                self.liveness_detector.reset_temporal_state()
            self._liveness_identity = None
            self._spoof_scores.clear()
            self._screen_label = ""
            self._spoof_flagged = False
            self._last_screen_check = 0.0
            self._challenge_started_at = None

    def _update_spoof_evidence(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Add this frame's anti-spoof score and screen check for the person in view."""
        anti_spoof = self._get_anti_spoof_detector()
        if anti_spoof is not None:
            with time_operation("anti_spoof_score"):
                score = anti_spoof.real_score(frame, bbox)
            if score is not None:
                self._spoof_scores.append(score)
                bad = sum(s < config.ANTI_SPOOF_THRESHOLD for s in self._spoof_scores)
                if not self._spoof_flagged and bad >= config.ANTI_SPOOF_BAD_FRAMES:
                    self._spoof_flagged = True
                    logger.warning(f"Spoof suspected: {bad} of the last {len(self._spoof_scores)} frames "
                                   f"look like a photo or screen; marking blocked")

        screen_detector = self._get_screen_detector()
        now = time.time()
        if (screen_detector is not None and not self._screen_label
                and now - self._last_screen_check >= config.SCREEN_CHECK_INTERVAL):
            self._last_screen_check = now
            with time_operation("screen_detection"):
                label = screen_detector.screen_covering_face(frame, bbox)
            if label:
                self._screen_label = label
                logger.warning(f"Spoof suspected: face shown on a {label}; marking blocked")

    def _log_spoof_diagnostics(self, face_result: "FaceProcessingResult", now: float) -> None:
        """
        Once per second while a face is in view (config.SPOOF_DIAGNOSTICS):
        the anti-spoof scores and the screens seen, for calibrating the
        thresholds from data/logs/app.log.
        """
        if not config.SPOOF_DIAGNOSTICS or not self.require_liveness or now - self._last_spoof_log < 1.0:
            return
        self._last_spoof_log = now
        scores = list(self._spoof_scores)
        score_text = (f"n={len(scores)} last={scores[-1]:.2f} mean={sum(scores) / len(scores):.2f} "
                      f"min={min(scores):.2f}") if scores else "n=0"
        screens = getattr(self.screen_detector, "last_screens", None) or []
        screen_text = ", ".join(f"{label} {score:.2f} covers {overlap:.0%}" for label, score, overlap in screens) or "none"
        logger.info(f"SPOOF CHECK: identity={face_result.identity} scores {score_text} | "
                    f"screens: {screen_text} | blocked by screen: {self._screen_label or 'no'} | "
                    f"reason: {face_result.reason}")

    def _spoof_score(self) -> Optional[float]:
        """Mean real-face probability over the recent frames, or None if not measured."""
        return sum(self._spoof_scores) / len(self._spoof_scores) if self._spoof_scores else None

    def _spoof_rejection_reason(self) -> Optional[str]:
        """Reason the face looks like a spoof, or None if no spoof has been seen."""
        if self._screen_label:
            return f"Spoof suspected: face shown on a {self._screen_label} - step out of view and try again"
        if self._spoof_flagged:
            return "Spoof suspected: face looks like a photo or screen"
        return None

    def _spoof_check_pending(self) -> bool:
        """
        True until the last ANTI_SPOOF_MIN_FRAMES scored frames all look real
        (too few frames yet, or a recent frame below the threshold).
        """
        if self._get_anti_spoof_detector() is None:
            return False
        recent = list(self._spoof_scores)[-config.ANTI_SPOOF_MIN_FRAMES:]
        return (len(recent) < config.ANTI_SPOOF_MIN_FRAMES
                or any(s < config.ANTI_SPOOF_THRESHOLD for s in recent))

    def _challenge_seconds_left(self, current_time: float) -> Optional[float]:
        if self._challenge_started_at is None or config.LIVENESS_CHALLENGE_TIMEOUT <= 0:
            return None
        return max(0.0, config.LIVENESS_CHALLENGE_TIMEOUT - (current_time - self._challenge_started_at))

    def _save_snapshot(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        session_id: int,
        student_id: Optional[str],
        timestamp: float
    ) -> Optional[Path]:
        """
        Save a small photo of the face and its surroundings for audit review:
        data/attendance_snapshots/session_<id>/<student_id>_<YYYYmmdd_HHMMSS>.jpg.
        A failure is logged and never blocks the attendance mark.
        """
        if not config.ATTENDANCE_SNAPSHOTS_ENABLED:
            return None
        try:
            # 3x the face box shows hands and anything held up to the camera
            crop = crop_around_face(frame, bbox, 3.0)
            if crop is None or crop.size == 0:
                return None
            scale = 320 / max(crop.shape[:2])
            if scale < 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            folder = Path(config.ATTENDANCE_SNAPSHOTS_DIR) / f"session_{session_id}"
            folder.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(student_id))
            path = folder / f"{safe_id}_{datetime.fromtimestamp(timestamp):%Y%m%d_%H%M%S}.jpg"
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise IOError(f"cv2.imwrite failed for {path}")
            return path
        except Exception as e:
            logger.error(f"Could not save attendance snapshot: {e}")
            return None

    def _note_liveness_identity(self, identity: str) -> None:
        """
        Reset liveness if a different person is now in front of the camera.

        Called only for frames whose identity is backed by an embedding, so a
        frame where the embedding failed does not wipe a student's progress.
        - Student A -> "Unknown": progress is KEPT. Recognition briefly fails
          while a student turns their head for the challenge; resetting then
          would wipe the very progress the challenge asks for.
        - "Unknown" -> student, or student A -> student B: liveness restarts,
          so an unenrolled live person's blink + turn can never be handed
          over to a student (e.g. a photo swapped in at the same place).
        Losing the face, a second face and a jump of the face box also
        restart it (see _process_frame_sync).
        """
        with self._liveness_lock:
            owner = self._liveness_identity
            if identity == "Unknown":
                if owner is None:
                    self._liveness_identity = identity
                return
            if owner is not None and identity != owner:
                logger.info(f"Person changed ('{owner}' -> '{identity}'); liveness restarted")
                self._reset_liveness()
            self._liveness_identity = identity

    def update_liveness(self, frame: np.ndarray) -> None:
        """
        Per-frame liveness: call for EVERY camera frame (~10 ms).

        Measures the eyes and head pose of the single face in view and
        advances the blink counter and head-turn challenge. Recognition, spoof
        checks and marking stay in process_frame(), which reads this evidence
        (set fast_liveness = True). With no face or several faces the
        landmarker reports no eyes, which only pauses the counters;
        process_frame() resets liveness for those cases.
        """
        if not self.require_liveness or frame is None or frame.size == 0:
            return
        face = self._last_face or DetectedFace(bbox=(0, 0, frame.shape[1], frame.shape[0]), confidence=1.0)
        with self._liveness_lock:
            liveness_detector = self._get_liveness_detector()
            liveness_detector.enable_video_mode()  # consecutive camera frames: track the face
            with time_operation("liveness_fast_update"):
                ear = liveness_detector.extract_ear(frame, face)
                liveness_detector.update_temporal_state(ear, frame, face)
            self._last_fast_ear = ear

    def refresh_liveness_display(self, result: "RealTimeAttendanceResult") -> None:
        """
        Copy the live blink / challenge progress into the face of `result`
        (the latest recognition result), so the on-screen instructions react
        to each blink immediately instead of when the next recognition ends.
        """
        if not self.fast_liveness or result.multiple_faces or len(result.processed_faces) != 1:
            return
        face = result.processed_faces[0]
        if face.already_marked or face.attendance_marked or self.liveness_detector is None:
            return
        with self._liveness_lock:
            detector = self.liveness_detector
            progress = detector.get_challenge_progress()
            face.blink_count = detector.blink_count
            face.min_blinks_required = detector.min_blinks
            face.challenge_type = progress['challenge_type']
            face.challenge_state = progress['challenge_state']
            face.challenge_progress = progress['progress_percent']
            face.challenge_seconds_left = self._challenge_seconds_left(time.time())

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
            # Thresholds must match the model that produced the embeddings
            backend = getattr(self._get_face_embedder(), "backend", None)
            self.face_recognizer = FaceRecognizer(backend=backend)
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
        if self.require_head_turn is not None:
            self.liveness_detector.enable_challenges = self.require_head_turn
            self.liveness_detector.require_challenge = self.require_head_turn
        return self.liveness_detector

    def _get_anti_spoof_detector(self) -> Optional[AntiSpoofDetector]:
        """The anti-spoof model, or None if disabled in config or its model file is missing."""
        if not config.ANTI_SPOOF_ENABLED:
            return None
        if self.anti_spoof_detector is None:
            self.anti_spoof_detector = AntiSpoofDetector()
        return self.anti_spoof_detector if self.anti_spoof_detector.available else None

    def _get_screen_detector(self) -> Optional[ScreenDetector]:
        """The phone / screen detector, or None if disabled in config or its model file is missing."""
        if not config.SCREEN_DETECTION_ENABLED:
            return None
        if self.screen_detector is None:
            self.screen_detector = ScreenDetector()
        return self.screen_detector if self.screen_detector.available else None

    def _recognize_crop(
        self,
        face_crop: np.ndarray,
        face_embedder: FaceEmbedder,
        face_recognizer: FaceRecognizer,
        frame: Optional[np.ndarray] = None,
        detected_face: Optional[DetectedFace] = None
    ) -> Tuple[str, bool, float, bool]:
        """
        Compute the face embedding and match it against the gallery.

        With SFace and a face that has YuNet landmarks, the face is aligned
        straight from `frame`; otherwise `face_crop` is embedded.

        Returns:
            (identity, is_recognized, distance, embedding_ok); identity is
            "Unknown" unless the recognizer reports a confident match.
        """
        identity = "Unknown"
        is_recognized = False
        distance = float('inf')
        embedding_ok = False

        if face_crop.size == 0 or len(self._gallery) == 0:
            return identity, is_recognized, distance, embedding_ok

        try:
            # PERFORMANCE: Use the MediaPipe face crop for dlib encoding.
            # Benchmark confirmed: crop (11ms) >> full_frame+bbox (248ms) >> full_frame (138ms)
            # The crop approach is 22x faster than the "optimized" bbox approach.
            if (frame is not None and detected_face is not None
                    and getattr(face_embedder, "backend", None) == "sface"
                    and getattr(detected_face, "landmarks", None) is not None):
                embedding = face_embedder.generate_embedding_from_frame(frame, detected_face)
            else:
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

        return identity, is_recognized, distance, embedding_ok

    # ------------------------------------------------------------------------
    # Background Worker for Async Processing
    # ------------------------------------------------------------------------

    def enable_background_processing(self) -> None:
        """
        Enable background processing mode for camera smoothness.
        
        When enabled, process_frame_async() returns immediately with the latest
        cached result while processing the new frame in a background thread.
        This prevents CV processing from blocking frame capture/display.
        """
        if self._bg_worker_enabled:
            return
            
        self._bg_worker_enabled = True
        self._bg_stop_flag.clear()
        self._bg_worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._bg_worker_thread.start()
        logger.info("Background processing enabled for camera smoothness")

    def disable_background_processing(self) -> None:
        """Stop the background worker thread."""
        if not self._bg_worker_enabled:
            return
            
        self._bg_worker_enabled = False
        self._bg_stop_flag.set()
        self._bg_frame_ready.set()
        
        if self._bg_worker_thread is not None:
            self._bg_worker_thread.join(timeout=1.0)
            self._bg_worker_thread = None
            
        logger.info("Background processing disabled")

    def _background_worker(self) -> None:
        """Background thread that processes frames asynchronously."""
        while not self._bg_stop_flag.is_set():
            if not self._bg_frame_ready.wait(timeout=0.1):
                continue
            self._bg_frame_ready.clear()
            # Check for pending frame
            with self._bg_lock:
                pending = self._bg_pending_frame
                self._bg_pending_frame = None
            
            if pending is not None:
                frame, session_id, marked_at = pending
                run_started = time.perf_counter()

                # Process frame synchronously in background thread
                try:
                    with time_operation("cv_frame_processing"):
                        result = self._process_frame_sync(frame, session_id, marked_at)
                    
                    # Publish result
                    with self._bg_lock:
                        self._bg_latest_result = result
                        
                except Exception as e:
                    logger.error(f"Background worker error: {e}")

                pause = self.min_processing_interval - (time.perf_counter() - run_started)
                if pause > 0:
                    self._bg_stop_flag.wait(pause)

    def process_frame_async(
        self,
        frame: np.ndarray,
        session_id: Optional[int] = None,
        marked_at: Optional[str] = None
    ) -> RealTimeAttendanceResult:
        """
        Process frame asynchronously for camera smoothness.
        
        Returns immediately with the latest cached result while submitting
        the new frame for background processing. This prevents CV processing
        from blocking frame capture and display updates.
        
        Args:
            frame: Input image array (BGR format from OpenCV)
            session_id: Session ID to process (if None, uses current session)
            marked_at: Optional explicit ISO timestamp for attendance
            
        Returns:
            Latest available RealTimeAttendanceResult (may be from previous frame)
        """
        if not self._bg_worker_enabled:
            # Fallback to synchronous processing if background worker not enabled
            return self.process_frame(frame, session_id, marked_at)
        
        # Submit new frame for background processing (replaces any pending frame)
        target_session = session_id if session_id is not None else self._session_id
        with self._bg_lock:
            self._bg_pending_frame = (frame.copy(), target_session, marked_at)
            result = self._bg_latest_result
        self._bg_frame_ready.set()
        
        # Return latest available result or empty result
        if result is None:
            result = RealTimeAttendanceResult(
                session_id=target_session,
                faces_detected=0,
                processed_faces=[],
                session_active=True,
                marked_count=0
            )
        
        return result

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
        Process a single video frame synchronously (blocks until complete).
        
        For non-blocking processing that allows smooth camera preview,
        use enable_background_processing() and process_frame_async() instead.
        
        Args:
            frame: Input image array (BGR format from OpenCV)
            session_id: Session ID to process (if None, uses current session)
            marked_at: Optional explicit ISO timestamp for attendance
            
        Returns:
            RealTimeAttendanceResult with details for each detected face
        """
        return self._process_frame_sync(frame, session_id, marked_at)

    def _process_frame_sync(
        self,
        frame: np.ndarray,
        session_id: Optional[int] = None,
        marked_at: Optional[str] = None
    ) -> RealTimeAttendanceResult:
        """
        Internal synchronous frame processing implementation.
        
        Called by both process_frame() (synchronous) and the background worker thread.
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
            self._tracked_face = None  # face left the view: re-recognise next time
            self._last_face = None
            # The detector briefly loses a face that turns for the head-turn
            # challenge. Only a face gone for LIVENESS_FACE_LOST_GRACE seconds
            # counts as the person leaving; the next person then starts afresh.
            now = time.time()
            if self._face_missing_since is None:
                self._face_missing_since = now
            if now - self._face_missing_since >= config.LIVENESS_FACE_LOST_GRACE:
                self._reset_liveness()
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

        current_time = time.time()
        self._face_missing_since = None  # a face is back in view

        # Step 2: One person at a time. With several faces in view nobody is
        # recognised, liveness-checked or marked: the eye/head landmarks come
        # from a single face, so one person's blink and head turn would
        # otherwise verify every face in the frame. Skipping recognition here
        # also avoids a ~230 ms embedding per face on every CV frame.
        if len(detected_faces) > 1:
            self._tracked_face = None
            self._last_face = None
            self._reset_liveness()
            return RealTimeAttendanceResult(
                session_id=target_session,
                faces_detected=len(detected_faces),
                processed_faces=[
                    FaceProcessingResult(
                        bbox=face.bbox,
                        detection_confidence=float(face.confidence),
                        reason=MULTIPLE_FACES_MESSAGE
                    )
                    for face in detected_faces
                ],
                session_active=session_active,
                marked_count=0,
                timestamp=current_time,
                multiple_faces=True,
                message=MULTIPLE_FACES_MESSAGE
            )

        # Step 3: Process the single face in view. A face that appears far
        # from the previous one is treated as a different person.
        face = detected_faces[0]
        if self._last_face is not None and _bbox_iou(self._last_face.bbox, face.bbox) < self.person_change_iou:
            logger.info("Face box jumped; treating it as a new person (liveness restarted)")
            self._tracked_face = None
            self._reset_liveness()
        self._last_face = face

        face_result = self._process_single_face(
            frame=frame,
            detected_face=face,
            target_session=target_session,
            session_active=session_active,
            face_embedder=self._get_face_embedder(),
            face_recognizer=self._get_face_recognizer(),
            liveness_detector=self._get_liveness_detector(),
            current_time=current_time,
            marked_at=marked_at,
            allow_tracking=True
        )
        face_result.spoof_score = self._spoof_score()
        face_result.challenge_seconds_left = self._challenge_seconds_left(current_time)
        # A suspected spoof is reported whatever else applies to the face
        # (unknown, already marked, liveness pending), so the person sees why
        if self.require_liveness and not face_result.attendance_marked:
            spoof_reason = self._spoof_rejection_reason()
            if spoof_reason:
                face_result.reason = spoof_reason
        self._log_spoof_diagnostics(face_result, current_time)

        return RealTimeAttendanceResult(
            session_id=target_session,
            faces_detected=1,
            processed_faces=[face_result],
            session_active=session_active,
            marked_count=1 if face_result.attendance_marked else 0,
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
        marked_at: Optional[str],
        allow_tracking: bool = False
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

        # Step A: Recognise the face, or reuse the identity of the face being followed
        track = self._tracked_face if allow_tracking else None
        recognition_reused = (
            track is not None
            and _bbox_iou(track['bbox'], bbox) >= self.track_min_iou
            and current_time - track['verified_at'] < self.track_reverify_seconds
        )

        if recognition_reused:
            identity, distance = track['identity'], track['distance']
            embedding_ok = True
            is_recognized = identity != "Unknown"
            track['bbox'] = bbox  # follow the face as it moves
        else:
            identity, is_recognized, distance, embedding_ok = self._recognize_crop(
                face_crop, face_embedder, face_recognizer, frame, detected_face
            )
            # Only follow faces that produced an embedding; otherwise retry next frame
            self._tracked_face = {
                'bbox': bbox,
                'identity': identity,
                'distance': distance,
                'verified_at': current_time,
            } if (allow_tracking and embedding_ok) else None

        # A different person restarts liveness before this frame's blink and
        # head-turn data are counted, so the evidence belongs to them alone.
        if embedding_ok:
            self._note_liveness_identity(identity)

        # Spoof evidence for this person: a replayed video blinks and turns too
        if self.require_liveness:
            self._update_spoof_evidence(frame, bbox)

        # Step B: Liveness analysis (shared with the per-frame path: under its lock)
        with self._liveness_lock:
            # A new liveness test for this person: one test with a random number
            # of blinks AND (if required) a random head-turn direction, done in
            # any order within one time limit. Random, so it cannot be pre-recorded.
            if self._challenge_started_at is None:
                if (liveness_detector.enable_challenges and
                    liveness_detector.require_challenge and
                    liveness_detector.current_challenge == ChallengeType.NONE):
                    liveness_detector.start_new_challenge()  # random head-turn direction
                liveness_detector.min_blinks = random.randint(
                    config.LIVENESS_MIN_BLINKS, max(config.LIVENESS_MIN_BLINKS, config.LIVENESS_MAX_BLINKS)
                )
                self._challenge_started_at = current_time

            # Update temporal state for the face, unless update_liveness() already
            # does it on every camera frame (fast_liveness)
            # PERFORMANCE FIX: MediaPipe landmarks are now cached per-frame to avoid double inference
            if self.fast_liveness:
                ear = self._last_fast_ear
            else:
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

            # Time limit: an unfinished challenge is replaced by a new random one
            # (the next frame starts it). Spoof evidence for the person is kept.
            seconds_left = self._challenge_seconds_left(current_time)
            if liveness_state != LivenessState.LIVE and seconds_left == 0.0:
                logger.debug("Liveness challenge timed out; a new random challenge starts")
                liveness_detector.reset_temporal_state()
                self._challenge_started_at = None

        # Step C: Resolve student identity from database (cached to avoid per-frame SQLite hits).
        # The cache maps identity_string -> student dict (or None when not found).
        # It is cleared on session change so deactivation changes are not permanently hidden.
        student_db_id = None
        student_id_str = None
        student_name = None

        if is_recognized:
            try:
                if identity in self._student_record_cache:
                    student_record = self._student_record_cache[identity]
                else:
                    with time_operation("student_db_lookup"):
                        student_record = self.student_service.get_student_by_student_id(identity)
                    self._student_record_cache[identity] = student_record
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

        # Step D: In-memory duplicate cache. Checked before liveness because
        # liveness restarts after each mark: a student who is already marked
        # is told so instead of being asked to blink and turn again.
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

        # Step E: Verify Liveness Requirement: spoof checks, blinks + challenge
        liveness_reason = None
        if self.require_liveness:
            liveness_reason = self._spoof_rejection_reason()
            if liveness_reason is None and liveness_state != LivenessState.LIVE:
                liveness_reason = "Liveness verification pending / not confirmed"
            if liveness_reason is None and self._spoof_check_pending():
                liveness_reason = "Checking for photo / screen spoofing - please hold still"
        if liveness_reason is not None:
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
                reason=liveness_reason
            )

        # Step F: Verify Session Status
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

        # Step G: Check cooldown debounce to avoid rapid repeated DB calls
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

        # SAFETY: never mark attendance from a reused identity. Recognise the
        # face again on this exact frame; it must still be the same student.
        if recognition_reused:
            fresh_identity, _, fresh_distance, fresh_embedding_ok = self._recognize_crop(
                face_crop, face_embedder, face_recognizer, frame, detected_face
            )
            if fresh_identity != identity:
                self._tracked_face = None
                if fresh_embedding_ok:
                    self._note_liveness_identity(fresh_identity)  # someone else: restart
                logger.info(
                    f"Re-check before marking did not confirm '{identity}' "
                    f"(got '{fresh_identity}'); attendance not marked"
                )
                return FaceProcessingResult(
                    bbox=bbox,
                    detection_confidence=det_conf,
                    identity=fresh_identity,
                    is_recognized=fresh_identity != "Unknown",
                    recognition_distance=fresh_distance,
                    liveness_state=liveness_state,
                    liveness_confidence=liveness_confidence,
                    blink_count=blinks_detected,
                    min_blinks_required=min_blinks,
                    challenge_type=challenge_progress['challenge_type'],
                    challenge_state=challenge_progress['challenge_state'],
                    challenge_progress=challenge_progress['progress_percent'],
                    attendance_marked=False,
                    reason="Identity re-check before marking failed - verifying again"
                )
            distance = fresh_distance
            if self._tracked_face is not None:
                self._tracked_face.update(distance=fresh_distance, verified_at=current_time)

        self._cooldown_tracker[student_db_id] = current_time

        # Step H: Mark Attendance via AttendanceService
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

            # Audit photo of who was marked (for teacher review)
            self._save_snapshot(frame, bbox, target_session, student_id_str, current_time)

            # This student's liveness evidence is used up: the next person
            # must blink and complete a new head-turn challenge themselves.
            self._reset_liveness()

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
            self._reset_liveness()
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

    def __del__(self):
        """Cleanup: stop background worker thread if running."""
        if self._bg_worker_enabled:
            self.disable_background_processing()


# Alias for backward compatibility / API flexibility
RealTimeAttendanceService = RealTimeAttendanceManager
