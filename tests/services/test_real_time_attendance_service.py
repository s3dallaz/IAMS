"""
Real-Time Attendance Integration Tests

Tests real-time frame processing, face recognition integration, liveness requirement,
session lifecycle enforcement, duplicate prevention, caching, and attendance marking.
Uses isolated test databases and synthetic frames/fixtures without requiring a physical camera.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding
from src.computer_vision.face_recognizer import FaceRecognizer, RecognitionResult
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessState,
    ChallengeState,
    ChallengeType
)
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService
from src.services.real_time_attendance_service import (
    RealTimeAttendanceManager,
    RealTimeAttendanceService,
    RealTimeAttendanceResult,
    FaceProcessingResult,
    RealTimeAttendanceError
)
from src.database.schema import initialize_database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    initialize_database(db_path)
    yield db_path
    
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def student_service(temp_db):
    """Create StudentService."""
    return StudentService(temp_db)


@pytest.fixture
def session_service(temp_db):
    """Create SessionService."""
    return SessionService(temp_db)


@pytest.fixture
def attendance_service(temp_db):
    """Create AttendanceService."""
    return AttendanceService(temp_db)


@pytest.fixture
def active_session(session_service):
    """Create an active session."""
    return session_service.create_session(
        session_name="CS101 Active Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )


@pytest.fixture
def scheduled_session(session_service):
    """Create a scheduled session."""
    return session_service.create_session(
        session_name="CS101 Scheduled Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )


@pytest.fixture
def closed_session(session_service):
    """Create a closed session."""
    return session_service.create_session(
        session_name="CS101 Closed Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )


@pytest.fixture
def sample_student(student_service):
    """Create a registered student."""
    return student_service.create_student(
        student_id="2021001",
        name="Alice Smith"
    )


@pytest.fixture
def second_student(student_service):
    """Create a second registered student."""
    return student_service.create_student(
        student_id="2021002",
        name="Bob Jones"
    )


@pytest.fixture
def sample_frame():
    """Create a dummy BGR test image (100x100x3)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_cv_components():
    """Create mock CV components for predictable integration tests."""
    mock_detector = MagicMock(spec=FaceDetector)
    mock_embedder = MagicMock(spec=FaceEmbedder)
    mock_recognizer = MagicMock(spec=FaceRecognizer)
    mock_liveness = MagicMock(spec=LivenessDetector)
    
    # Configure default mock behaviors
    mock_liveness.min_blinks = 1
    mock_liveness.blink_count = 1
    mock_liveness.liveness_threshold = 0.75
    mock_liveness.enable_challenges = False
    mock_liveness.require_challenge = False
    mock_liveness.extract_ear.return_value = 0.25
    mock_liveness.update_temporal_state.return_value = True
    
    return {
        'detector': mock_detector,
        'embedder': mock_embedder,
        'recognizer': mock_recognizer,
        'liveness': mock_liveness
    }


# ============================================================================
# Test Initialization & Configuration
# ============================================================================

def test_init_with_defaults(temp_db):
    """Test initializing manager with default settings."""
    manager = RealTimeAttendanceManager(database_path=temp_db)
    assert manager is not None
    assert manager.database_path == temp_db
    assert manager.current_session_id is None
    assert len(manager.get_gallery()) == 0
    assert len(manager.get_marked_students()) == 0


def test_init_with_session_and_services(temp_db, active_session, attendance_service, session_service, student_service):
    """Test initializing manager with injected session and services."""
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        attendance_service=attendance_service,
        session_service=session_service,
        student_service=student_service,
        require_liveness=True,
        cooldown_seconds=2.0
    )
    assert manager.current_session_id == active_session
    assert manager.require_liveness is True
    assert manager.cooldown_seconds == 2.0


def test_set_session_preloads_existing_attendance(temp_db, active_session, sample_student, attendance_service):
    """Test that setting a session pre-loads already marked attendance into memory cache."""
    # Pre-record attendance for student in this session
    attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    manager = RealTimeAttendanceManager(database_path=temp_db)
    manager.set_session(active_session)
    
    # Marked students cache should contain sample_student
    assert sample_student in manager.get_marked_students()


def test_gallery_management():
    """Test setting and getting gallery."""
    manager = RealTimeAttendanceManager()
    dummy_enc = np.random.rand(128)
    gallery = [("2021001", FaceEmbedding(dummy_enc))]
    
    manager.set_gallery(gallery)
    assert len(manager.get_gallery()) == 1
    assert manager.get_gallery()[0][0] == "2021001"


# ============================================================================
# Test Frame Processing - Valid Active Session & Live Recognized Face
# ============================================================================

def test_process_frame_valid_recognized_and_live(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Test complete successful flow: face detected -> recognized -> live -> attendance created."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    # 1 face detected
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    
    # Embedding generated
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    
    # Recognized as student '2021001'
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.32, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness,
        require_liveness=True
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame, marked_at="2026-10-04 10:05:00")
    
    assert isinstance(result, RealTimeAttendanceResult)
    assert result.session_id == active_session
    assert result.faces_detected == 1
    assert result.session_active is True
    assert result.marked_count == 1
    assert len(result.processed_faces) == 1
    
    face_res = result.processed_faces[0]
    assert face_res.identity == "2021001"
    assert face_res.is_recognized is True
    assert face_res.student_id == "2021001"
    assert face_res.student_db_id == sample_student
    assert face_res.student_name == "Alice Smith"
    assert face_res.liveness_state == LivenessState.LIVE
    assert face_res.attendance_marked is True
    assert face_res.attendance_id is not None
    assert face_res.attendance_status == AttendanceStatus.PRESENT
    assert face_res.already_marked is False
    
    # Verify in DB
    att_service = AttendanceService(temp_db)
    record = att_service.get_attendance(face_res.attendance_id)
    assert record is not None
    assert record['session_id'] == active_session
    assert record['student_id'] == sample_student
    assert record['status'] == AttendanceStatus.PRESENT
    assert record['detection_confidence'] == 0.95
    assert record['liveness_score'] >= 0.75


def test_process_frame_late_attendance_status(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Test that attendance marked after late threshold receives 'late' status."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.25, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    # Session started at 10:00:00 with 15min threshold; mark at 10:20:00
    result = manager.process_frame(sample_frame, marked_at="2026-10-04 10:20:00")
    
    assert result.marked_count == 1
    assert result.processed_faces[0].attendance_status == AttendanceStatus.LATE


def test_process_frame_no_faces_detected(temp_db, active_session, sample_frame, mock_cv_components):
    """Test frame processing when no faces are detected."""
    detector = mock_cv_components['detector']
    detector.detect_faces.return_value = []
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector
    )
    
    result = manager.process_frame(sample_frame)
    assert result.faces_detected == 0
    assert result.marked_count == 0
    assert len(result.processed_faces) == 0


def test_process_frame_invalid_frame_rejected(temp_db, active_session):
    """Test that invalid or empty frames raise RealTimeAttendanceError."""
    manager = RealTimeAttendanceManager(session_id=active_session, database_path=temp_db)
    
    with pytest.raises(RealTimeAttendanceError, match="Invalid frame"):
        manager.process_frame(None)
    
    with pytest.raises(RealTimeAttendanceError, match="Invalid frame"):
        manager.process_frame(np.array([]))


# ============================================================================
# Test Session Lifecycle Enforcement
# ============================================================================

def test_scheduled_session_attendance_rejected(
    temp_db, scheduled_session, sample_student, sample_frame, mock_cv_components
):
    """Test that scheduled session rejects attendance marking."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.3, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=scheduled_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    
    assert result.session_active is False
    assert result.marked_count == 0
    assert result.processed_faces[0].attendance_marked is False
    assert "not active" in result.processed_faces[0].reason.lower()


def test_closed_session_attendance_rejected(
    temp_db, closed_session, sample_student, sample_frame, mock_cv_components
):
    """Test that closed session rejects attendance marking."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.3, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=closed_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    
    assert result.session_active is False
    assert result.marked_count == 0
    assert result.processed_faces[0].attendance_marked is False
    assert "not active" in result.processed_faces[0].reason.lower()


# ============================================================================
# Test Unknown Person & Failed Liveness
# ============================================================================

def test_unknown_face_no_attendance_marked(
    temp_db, active_session, sample_frame, mock_cv_components
):
    """Test that unrecognized face does not create attendance."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="Unknown", distance=0.85, is_match=False)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    
    result = manager.process_frame(sample_frame)
    
    assert result.marked_count == 0
    face_res = result.processed_faces[0]
    assert face_res.is_recognized is False
    assert face_res.identity == "Unknown"
    assert face_res.attendance_marked is False
    assert "Unknown person" in face_res.reason


def test_ambiguous_recognition_does_not_mark_attendance(
    temp_db, active_session, sample_frame, mock_cv_components
):
    """A face almost equally close to two students is not attributed.

    Uses the real FaceRecognizer so the whole path is exercised: the query
    embedding is nearly the same distance to two enrolled identities, the
    recognizer reports "Unknown", and no attendance may be recorded.
    """
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    liveness = mock_cv_components['liveness']

    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    embedder.generate_embedding.return_value = FaceEmbedding(np.zeros(128))

    enc_enrolled = np.zeros(128)
    enc_enrolled[0] = 0.30
    enc_other = np.zeros(128)
    enc_other[0] = 0.32

    recognizer = FaceRecognizer()  # real: threshold 0.6, margin 0.1

    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([
        ("2021001", FaceEmbedding(enc_enrolled)),
        ("2021002", FaceEmbedding(enc_other))
    ])

    result = manager.process_frame(sample_frame)

    assert result.marked_count == 0
    face_res = result.processed_faces[0]
    assert face_res.is_recognized is False
    assert face_res.identity == "Unknown"
    assert face_res.attendance_marked is False


def test_failed_liveness_no_attendance_marked(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Test that recognized face with failed / pending liveness does NOT mark attendance."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    # Configure liveness to fail (0 blinks)
    liveness.min_blinks = 1
    liveness.blink_count = 0
    liveness.update_temporal_state.return_value = False
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.25, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness,
        require_liveness=True
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    
    assert result.marked_count == 0
    face_res = result.processed_faces[0]
    assert face_res.is_recognized is True
    assert face_res.liveness_state == LivenessState.UNKNOWN
    assert face_res.attendance_marked is False
    assert "liveness" in face_res.reason.lower()
    
    # Verify no record was inserted in DB
    att_service = AttendanceService(temp_db)
    assert not att_service.attendance_exists(active_session, sample_student)


def test_active_challenge_required_and_uncompleted_no_attendance(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Test that if active challenge is enabled and not completed, attendance is not marked."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    # Blinks pass, but challenge is still ACTIVE (not COMPLETED)
    liveness.min_blinks = 1
    liveness.blink_count = 1
    liveness.enable_challenges = True
    liveness.require_challenge = True
    liveness.challenge_state = ChallengeState.ACTIVE
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.25, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness,
        require_liveness=True
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    assert result.marked_count == 0
    assert result.processed_faces[0].attendance_marked is False


# ============================================================================
# Test Duplicate Prevention & Real-Time Debounce
# ============================================================================

def test_duplicate_student_not_marked_multiple_times(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Test that processing subsequent frames with same student does NOT create duplicate records."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.25, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    # Frame 1: Attendance marked
    res1 = manager.process_frame(sample_frame)
    assert res1.marked_count == 1
    assert res1.processed_faces[0].attendance_marked is True
    assert res1.processed_faces[0].already_marked is False
    
    # Frame 2: Same student immediately processed again
    res2 = manager.process_frame(sample_frame)
    assert res2.marked_count == 0
    assert res2.processed_faces[0].attendance_marked is False
    assert res2.processed_faces[0].already_marked is True
    assert "already recorded" in res2.processed_faces[0].reason.lower()
    
    # Verify DB has exactly 1 attendance record
    att_service = AttendanceService(temp_db)
    records = att_service.get_session_attendance(active_session)
    assert len(records) == 1


def test_multiple_distinct_students_in_single_frame(
    temp_db, active_session, sample_student, second_student, sample_frame, mock_cv_components
):
    """Test that multiple distinct students in the same frame are both marked independently."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    # 2 faces detected
    detector.detect_faces.return_value = [
        DetectedFace(bbox=(10, 10, 40, 40), confidence=0.92),
        DetectedFace(bbox=(50, 50, 40, 40), confidence=0.91)
    ]
    
    enc1 = np.random.rand(128)
    enc2 = np.random.rand(128)
    embedder.generate_embedding.side_effect = [FaceEmbedding(enc1), FaceEmbedding(enc2)]
    
    recognizer.recognize.side_effect = [
        RecognitionResult(identity="2021001", distance=0.2, is_match=True),
        RecognitionResult(identity="2021002", distance=0.22, is_match=True)
    ]
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([
        ("2021001", FaceEmbedding(enc1)),
        ("2021002", FaceEmbedding(enc2))
    ])
    
    result = manager.process_frame(sample_frame)
    
    assert result.faces_detected == 2
    assert result.marked_count == 2
    assert result.processed_faces[0].student_id == "2021001"
    assert result.processed_faces[0].attendance_marked is True
    assert result.processed_faces[1].student_id == "2021002"
    assert result.processed_faces[1].attendance_marked is True
    
    # Verify in DB
    att_service = AttendanceService(temp_db)
    records = att_service.get_session_attendance(active_session)
    assert len(records) == 2


def test_inactive_student_rejected(
    temp_db, active_session, sample_student, student_service, sample_frame, mock_cv_components
):
    """Test that deactivated students are not marked present."""
    student_service.deactivate_student(sample_student)
    
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="2021001", distance=0.2, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("2021001", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    assert result.marked_count == 0
    assert result.processed_faces[0].attendance_marked is False
    assert "deactivated" in result.processed_faces[0].reason.lower()


def test_recognized_student_not_in_database(
    temp_db, active_session, sample_frame, mock_cv_components
):
    """Test gallery identity that does not exist in student DB table."""
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    dummy_enc = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(dummy_enc)
    recognizer.recognize.return_value = RecognitionResult(identity="9999999", distance=0.2, is_match=True)
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=embedder,
        face_recognizer=recognizer,
        liveness_detector=liveness
    )
    manager.set_gallery([("9999999", FaceEmbedding(dummy_enc))])
    
    result = manager.process_frame(sample_frame)
    assert result.marked_count == 0
    assert result.processed_faces[0].attendance_marked is False
    assert "not found in student database" in result.processed_faces[0].reason.lower()


def test_reset_tracking(temp_db, active_session, mock_cv_components):
    """Test reset tracking resets liveness and cooldown state."""
    liveness = mock_cv_components['liveness']
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        liveness_detector=liveness
    )
    
    manager.reset_tracking()
    liveness.reset_temporal_state.assert_called_once()


# ============================================================================
# Test Head Turn Angle Configuration
# ============================================================================

def test_head_turn_angle_passed_to_liveness_detector(temp_db, active_session):
    """Test that head_turn_angle parameter is passed to LivenessDetector."""
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        head_turn_angle=20.0
    )
    
    # Trigger lazy initialization
    liveness_detector = manager._get_liveness_detector()
    
    assert liveness_detector.head_turn_angle == 20.0


def test_head_turn_angle_default_from_config(temp_db, active_session):
    """Test that head_turn_angle defaults to config value when not specified."""
    import config
    
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db
    )
    
    liveness_detector = manager._get_liveness_detector()
    
    assert liveness_detector.head_turn_angle == config.LIVENESS_HEAD_TURN_ANGLE


def test_head_turn_angle_can_be_updated_runtime(temp_db, active_session):
    """Test that head_turn_angle can be changed at runtime."""
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        head_turn_angle=15.0
    )
    
    # Initialize the detector
    liveness_detector = manager._get_liveness_detector()
    assert liveness_detector.head_turn_angle == 15.0
    
    # Update the angle
    manager.head_turn_angle = 25.0
    liveness_detector = manager._get_liveness_detector()
    
    assert liveness_detector.head_turn_angle == 25.0


def test_head_turn_angle_validated(temp_db, active_session):
    """Test that invalid head_turn_angle values raise errors."""
    from src.computer_vision.liveness_detector import LivenessDetector
    
    # Zero angle should fail
    with pytest.raises(ValueError, match="head_turn_angle must be positive"):
        LivenessDetector(head_turn_angle=0.0)
    
    # Negative angle should fail
    with pytest.raises(ValueError, match="head_turn_angle must be positive"):
        LivenessDetector(head_turn_angle=-10.0)
