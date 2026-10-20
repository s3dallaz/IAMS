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
    RealTimeAttendanceError,
    MULTIPLE_FACES_MESSAGE
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


def _tracking_manager(temp_db, active_session, mock_cv_components):
    """Manager with a one-student gallery and liveness still pending."""
    mock_cv_components['liveness'].blink_count = 0  # nothing is marked yet
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=mock_cv_components['detector'],
        face_embedder=mock_cv_components['embedder'],
        face_recognizer=mock_cv_components['recognizer'],
        liveness_detector=mock_cv_components['liveness']
    )
    manager.set_gallery([("2021001", FaceEmbedding(np.random.rand(128)))])
    mock_cv_components['embedder'].generate_embedding.return_value = FaceEmbedding(np.random.rand(128))
    return manager


ALICE = RecognitionResult(identity="2021001", distance=0.3, is_match=True)
STRANGER = RecognitionResult(identity="Unknown", distance=0.85, is_match=False)


def test_followed_face_reuses_identity_without_new_embedding(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """The same face staying in view is recognised once, not on every frame."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    recognizer.recognize.return_value = ALICE
    manager = _tracking_manager(temp_db, active_session, mock_cv_components)

    first = manager.process_frame(sample_frame)
    detector.detect_faces.return_value = [DetectedFace(bbox=(14, 12, 50, 50), confidence=0.9)]
    second = manager.process_frame(sample_frame)  # face moved slightly

    assert recognizer.recognize.call_count == 1
    assert first.processed_faces[0].identity == "2021001"
    assert second.processed_faces[0].identity == "2021001"


def test_stranger_is_never_marked_from_followed_identity(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """Even if a stranger takes over a followed face, the re-check on the
    marking frame stops attendance being recorded for the student.
    """
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    recognizer.recognize.side_effect = [ALICE, STRANGER]
    manager = _tracking_manager(temp_db, active_session, mock_cv_components)

    manager.process_frame(sample_frame)    # Alice recognised, liveness pending
    liveness.blink_count = 1               # liveness now passes
    result = manager.process_frame(sample_frame)

    face = result.processed_faces[0]
    assert recognizer.recognize.call_count == 2   # fresh check on the marking frame
    assert result.marked_count == 0
    assert face.attendance_marked is False
    assert face.identity == "Unknown"


def test_followed_identity_is_confirmed_before_marking(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """When the fresh check confirms the student, attendance is marked."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    recognizer.recognize.return_value = ALICE
    manager = _tracking_manager(temp_db, active_session, mock_cv_components)

    manager.process_frame(sample_frame)
    liveness.blink_count = 1
    result = manager.process_frame(sample_frame)

    assert recognizer.recognize.call_count == 2
    assert result.marked_count == 1
    assert result.processed_faces[0].attendance_marked is True


@pytest.mark.parametrize("second_frame_faces", [
    [],                                                        # face left the view
    [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9),
     DetectedFace(bbox=(200, 10, 50, 50), confidence=0.9)],    # a second face appeared
    [DetectedFace(bbox=(300, 200, 50, 50), confidence=0.9)],   # box jumped elsewhere
])
def test_face_is_recognised_again_when_following_breaks(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    second_frame_faces
):
    """Losing the face, a second face, or a jump all force a new recognition."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    face = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    recognizer.recognize.return_value = ALICE
    manager = _tracking_manager(temp_db, active_session, mock_cv_components)

    detector.detect_faces.return_value = face
    manager.process_frame(sample_frame)
    detector.detect_faces.return_value = second_frame_faces
    manager.process_frame(sample_frame)

    calls_before_last_frame = recognizer.recognize.call_count
    detector.detect_faces.return_value = face
    manager.process_frame(sample_frame)

    # The face in the last frame was recognised again, not taken from the track
    assert recognizer.recognize.call_count == calls_before_last_frame + 1


def test_followed_face_is_reverified_periodically(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """After track_reverify_seconds the face is recognised again."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    recognizer.recognize.return_value = ALICE
    manager = _tracking_manager(temp_db, active_session, mock_cv_components)
    manager.track_reverify_seconds = 0.0

    manager.process_frame(sample_frame)
    manager.process_frame(sample_frame)

    assert recognizer.recognize.call_count == 2


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


def test_multiple_faces_in_frame_nobody_marked(
    temp_db, active_session, sample_student, second_student, sample_frame, mock_cv_components
):
    """Two enrolled, live-looking faces in one frame: nobody is marked.

    Liveness landmarks come from one face only, so one person's blink and
    head turn must not verify a second face (e.g. a photo held up beside
    them). The kiosk asks people to step up one at a time instead.
    """
    detector = mock_cv_components['detector']
    embedder = mock_cv_components['embedder']
    recognizer = mock_cv_components['recognizer']
    liveness = mock_cv_components['liveness']

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
    assert result.marked_count == 0
    assert result.multiple_faces is True
    assert result.message == MULTIPLE_FACES_MESSAGE
    assert len(result.processed_faces) == 2
    assert all(not f.attendance_marked and not f.is_recognized for f in result.processed_faces)
    assert all(f.reason == MULTIPLE_FACES_MESSAGE for f in result.processed_faces)
    # No embedding, recognition or liveness work while several faces are in view
    embedder.generate_embedding.assert_not_called()
    recognizer.recognize.assert_not_called()
    liveness.update_temporal_state.assert_not_called()
    # ... and any liveness progress so far is discarded
    liveness.reset_temporal_state.assert_called()

    att_service = AttendanceService(temp_db)
    assert att_service.get_session_attendance(active_session) == []


def test_single_face_result_has_no_multi_face_message(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components
):
    """The single-face path is unchanged and carries no multi-face flag."""
    detector = mock_cv_components['detector']
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)]
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    mock_cv_components['embedder'].generate_embedding.return_value = FaceEmbedding(np.random.rand(128))
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=detector,
        face_embedder=mock_cv_components['embedder'],
        face_recognizer=mock_cv_components['recognizer'],
        liveness_detector=mock_cv_components['liveness']
    )
    manager.set_gallery([("2021001", FaceEmbedding(np.random.rand(128)))])

    result = manager.process_frame(sample_frame)

    assert result.multiple_faces is False
    assert result.message == ""
    assert result.marked_count == 1
    assert result.processed_faces[0].attendance_marked is True


# ============================================================================
# Test Liveness Belongs To One Person
# ============================================================================

BOB = RecognitionResult(identity="2021002", distance=0.3, is_match=True)
FACE = DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)


@pytest.fixture
def stateful_liveness(mock_cv_components):
    """Liveness mock with blink + head-turn challenge state that really resets."""
    liveness = mock_cv_components['liveness']
    liveness.enable_challenges = True
    liveness.require_challenge = True

    def reset():
        liveness.blink_count = 0
        liveness.current_challenge = ChallengeType.NONE
        liveness.challenge_state = ChallengeState.INACTIVE

    def start_challenge():
        liveness.current_challenge = ChallengeType.TURN_LEFT
        liveness.challenge_state = ChallengeState.ACTIVE
        return ChallengeType.TURN_LEFT

    liveness.reset_temporal_state.side_effect = reset
    liveness.start_new_challenge.side_effect = start_challenge
    reset()
    return liveness


def _pass_liveness(liveness):
    """The person in front of the camera blinks (as often as the challenge asks) and turns."""
    liveness.blink_count = liveness.min_blinks
    liveness.challenge_state = ChallengeState.COMPLETED


def _liveness_manager(temp_db, active_session, mock_cv_components, liveness):
    manager = RealTimeAttendanceManager(
        session_id=active_session,
        database_path=temp_db,
        face_detector=mock_cv_components['detector'],
        face_embedder=mock_cv_components['embedder'],
        face_recognizer=mock_cv_components['recognizer'],
        liveness_detector=liveness,
        require_liveness=True
    )
    manager.set_gallery([
        ("2021001", FaceEmbedding(np.random.rand(128))),
        ("2021002", FaceEmbedding(np.random.rand(128)))
    ])
    mock_cv_components['embedder'].generate_embedding.return_value = FaceEmbedding(np.random.rand(128))
    return manager


def test_liveness_resets_after_mark_and_next_student_must_verify(
    temp_db, active_session, sample_student, second_student, sample_frame,
    mock_cv_components, stateful_liveness
):
    """After Alice is marked, Bob gets no credit for her blink and head turn."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)
    detector.detect_faces.return_value = [FACE]

    # Alice verifies and is marked
    recognizer.recognize.return_value = ALICE
    manager.process_frame(sample_frame)          # challenge starts
    _pass_liveness(liveness)
    alice = manager.process_frame(sample_frame)
    assert alice.processed_faces[0].attendance_marked is True
    assert liveness.blink_count == 0             # evidence used up
    assert liveness.challenge_state == ChallengeState.INACTIVE

    # Alice leaves, Bob steps up: he must blink and turn himself
    detector.detect_faces.return_value = []
    manager.process_frame(sample_frame)
    detector.detect_faces.return_value = [FACE]
    recognizer.recognize.return_value = BOB
    bob_pending = manager.process_frame(sample_frame)
    face = bob_pending.processed_faces[0]
    assert face.identity == "2021002"
    assert face.attendance_marked is False
    assert face.liveness_state == LivenessState.UNKNOWN
    assert "liveness" in face.reason.lower()

    _pass_liveness(liveness)
    bob = manager.process_frame(sample_frame)
    assert bob.processed_faces[0].attendance_marked is True
    assert len(AttendanceService(temp_db).get_session_attendance(active_session)) == 2


def test_marked_student_still_in_view_is_reported_already_marked(
    temp_db, active_session, sample_student, sample_frame,
    mock_cv_components, stateful_liveness
):
    """Liveness restarts after the mark, but the marked student is not asked again."""
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)
    detector.detect_faces.return_value = [FACE]
    recognizer.recognize.return_value = ALICE

    manager.process_frame(sample_frame)
    _pass_liveness(liveness)
    manager.process_frame(sample_frame)          # marked
    after = manager.process_frame(sample_frame)  # still standing there

    assert after.processed_faces[0].already_marked is True
    assert after.processed_faces[0].attendance_marked is False


def test_liveness_resets_when_face_leaves_view(
    temp_db, active_session, sample_student, sample_frame,
    mock_cv_components, stateful_liveness
):
    """Progress collected for one face is dropped once nobody is in view."""
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    liveness.blink_count = 1                     # blinked, head turn still pending
    detector.detect_faces.return_value = []
    manager.process_frame(sample_frame)

    assert liveness.blink_count == 0
    assert liveness.challenge_state == ChallengeState.INACTIVE


@pytest.mark.parametrize("first_person", [ALICE, STRANGER])
def test_liveness_resets_when_identity_changes(
    temp_db, active_session, sample_student, second_student, sample_frame,
    mock_cv_components, stateful_liveness, first_person
):
    """A different person taking over the face (no gap in detection) starts
    liveness again: neither another student's nor an unenrolled live
    person's blink + turn can be handed over to Bob.
    """
    detector = mock_cv_components['detector']
    recognizer = mock_cv_components['recognizer']
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)
    manager.track_reverify_seconds = 0.0         # recognise on every frame
    detector.detect_faces.return_value = [FACE]

    recognizer.recognize.return_value = first_person
    manager.process_frame(sample_frame)
    liveness.blink_count = 1                     # first person blinked...
    if first_person is STRANGER:
        _pass_liveness(liveness)                 # ...the stranger even passed fully

    recognizer.recognize.return_value = BOB
    result = manager.process_frame(sample_frame)

    face = result.processed_faces[0]
    assert face.identity == "2021002"
    assert face.blink_count == 0
    assert face.liveness_state == LivenessState.UNKNOWN
    assert face.attendance_marked is False
    assert result.marked_count == 0


def test_liveness_kept_while_same_student_stays(
    temp_db, active_session, sample_student, sample_frame,
    mock_cv_components, stateful_liveness
):
    """Re-recognising the same student does not wipe their progress."""
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)
    manager.track_reverify_seconds = 0.0
    detector.detect_faces.return_value = [FACE]

    manager.process_frame(sample_frame)
    liveness.blink_count = 1
    result = manager.process_frame(sample_frame)

    assert liveness.blink_count == 1
    assert result.processed_faces[0].blink_count == 1


def test_second_face_appearing_discards_liveness_progress(
    temp_db, active_session, sample_student, sample_frame,
    mock_cv_components, stateful_liveness
):
    """A second face joining mid-verification resets the first person's progress."""
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    liveness = stateful_liveness
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    _pass_liveness(liveness)
    detector.detect_faces.return_value = [FACE, DetectedFace(bbox=(60, 10, 30, 30), confidence=0.9)]
    result = manager.process_frame(sample_frame)

    assert result.multiple_faces is True
    assert result.marked_count == 0
    assert liveness.blink_count == 0
    assert liveness.challenge_state == ChallengeState.INACTIVE


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


# ============================================================================
# Spoof checks, random timed challenges and audit snapshots
# ============================================================================

import config  # noqa: E402
import cv2  # noqa: E402
from src.computer_vision.spoof_detection import AntiSpoofDetector, ScreenDetector  # noqa: E402
from src.services.real_time_attendance_service import SPOOF_RECHECK_PREFIX  # noqa: E402

FINAL_SPOOF_REASON = "Spoof suspected: face looks like a photo or screen - step out of view and try again"


def _fake_anti_spoof(score):
    fake = MagicMock(spec=AntiSpoofDetector)
    fake.available = True
    fake.real_score.return_value = score
    return fake


def _fake_screen(label=None):
    fake = MagicMock(spec=ScreenDetector)
    fake.available = True
    fake.screen_covering_face.return_value = label
    return fake


@pytest.fixture
def spoof_checks_on(monkeypatch):
    """Enable both spoof checks (conftest turns them off) and decide after 3 frames."""
    monkeypatch.setattr(config, "ANTI_SPOOF_ENABLED", True)
    monkeypatch.setattr(config, "SCREEN_DETECTION_ENABLED", True)
    monkeypatch.setattr(config, "ANTI_SPOOF_MIN_FRAMES", 3)


def _spoof_manager(temp_db, active_session, mock_cv_components, liveness, anti_spoof, screen,
                   face=FACE):
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, liveness)
    manager.anti_spoof_detector = anti_spoof
    manager.screen_detector = screen
    mock_cv_components['detector'].detect_faces.return_value = [face]
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    return manager


def test_replayed_face_is_not_marked_even_after_blink_and_turn(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, spoof_checks_on
):
    """A video replay passes blink + head turn; the anti-spoof model still blocks it."""
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             _fake_anti_spoof(0.2), _fake_screen())
    manager.process_frame(sample_frame)          # challenge starts
    _pass_liveness(stateful_liveness)
    for _ in range(3):
        face = manager.process_frame(sample_frame).processed_faces[0]

    assert face.attendance_marked is False
    assert face.reason.startswith(SPOOF_RECHECK_PREFIX)     # 0.2 keeps failing the recheck
    assert face.spoof_score == pytest.approx(0.2)


def test_real_face_is_marked_once_enough_frames_are_scored(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, spoof_checks_on
):
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             _fake_anti_spoof(0.95), _fake_screen())
    manager.process_frame(sample_frame)          # score 1, challenge starts
    _pass_liveness(stateful_liveness)

    pending = manager.process_frame(sample_frame).processed_faces[0]   # score 2 of 3
    assert pending.attendance_marked is False
    assert pending.reason.startswith("Checking for photo / screen spoofing")

    marked = manager.process_frame(sample_frame).processed_faces[0]    # score 3
    assert marked.attendance_marked is True


def test_face_on_a_phone_blocks_marking_until_the_person_leaves(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, spoof_checks_on
):
    screen = _fake_screen("cell phone")
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             _fake_anti_spoof(0.95), screen)
    detector = mock_cv_components['detector']

    manager.process_frame(sample_frame)          # phone seen covering the face
    screen.screen_covering_face.return_value = None
    _pass_liveness(stateful_liveness)
    for _ in range(3):
        face = manager.process_frame(sample_frame).processed_faces[0]
    assert face.attendance_marked is False
    assert "cell phone" in face.reason and face.reason.startswith("Spoof suspected")

    # Leaving the view clears it; a real face then verifies normally
    detector.detect_faces.return_value = []
    manager.process_frame(sample_frame)
    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    manager.process_frame(sample_frame)
    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is True


def test_spoof_evidence_restarts_when_the_person_changes(
    temp_db, active_session, sample_student, second_student, sample_frame,
    mock_cv_components, stateful_liveness, spoof_checks_on
):
    anti_spoof = _fake_anti_spoof(0.2)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    for _ in range(3):
        manager.process_frame(sample_frame)      # Alice: three low scores

    # Bob steps in at a different position, so he is recognised afresh
    mock_cv_components['detector'].detect_faces.return_value = [
        DetectedFace(bbox=(55, 55, 40, 40), confidence=0.9)
    ]
    mock_cv_components['recognizer'].recognize.return_value = BOB
    anti_spoof.real_score.return_value = 0.9
    bob = manager.process_frame(sample_frame).processed_faces[0]

    assert bob.spoof_score == pytest.approx(0.9)  # none of Alice's scores
    assert not bob.reason.startswith("Spoof suspected")


def test_spoof_checks_are_skipped_when_liveness_is_not_required(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, spoof_checks_on
):
    anti_spoof, screen = _fake_anti_spoof(0.1), _fake_screen("tv")
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, screen)
    manager.require_liveness = False

    face = manager.process_frame(sample_frame).processed_faces[0]

    assert face.attendance_marked is True
    anti_spoof.real_score.assert_not_called()
    screen.screen_covering_face.assert_not_called()


def test_each_challenge_asks_for_a_random_number_of_blinks(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch
):
    monkeypatch.setattr(config, "LIVENESS_MIN_BLINKS", 1)
    monkeypatch.setattr(config, "LIVENESS_MAX_BLINKS", 3)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness, None, None)

    with patch("src.services.real_time_attendance_service.random.randint", return_value=3) as randint:
        face = manager.process_frame(sample_frame).processed_faces[0]

    randint.assert_called_once_with(1, 3)
    assert stateful_liveness.min_blinks == 3
    assert face.min_blinks_required == 3
    stateful_liveness.blink_count = 1                # one blink is not enough now
    stateful_liveness.challenge_state = ChallengeState.COMPLETED
    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is False


def test_unfinished_challenge_is_replaced_after_the_time_limit(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch
):
    monkeypatch.setattr(config, "LIVENESS_CHALLENGE_TIMEOUT", 10.0)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness, None, None)

    first = manager.process_frame(sample_frame).processed_faces[0]
    assert first.challenge_seconds_left == pytest.approx(10.0, abs=0.5)
    assert stateful_liveness.start_new_challenge.call_count == 1

    manager._challenge_started_at -= 11.0         # the person took too long
    stateful_liveness.blink_count = stateful_liveness.min_blinks  # blinked, but never turned
    manager.process_frame(sample_frame)
    assert stateful_liveness.blink_count == 0     # progress discarded
    manager.process_frame(sample_frame)
    assert stateful_liveness.start_new_challenge.call_count == 2  # a fresh random challenge


def test_audit_snapshot_is_saved_when_a_student_is_marked(
    temp_db, active_session, sample_student, mock_cv_components, stateful_liveness,
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "ATTENDANCE_SNAPSHOTS_ENABLED", True)
    monkeypatch.setattr(config, "ATTENDANCE_SNAPSHOTS_DIR", tmp_path)
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness, None, None,
                             face=DetectedFace(bbox=(250, 150, 140, 170), confidence=0.9))
    manager.process_frame(frame)
    _pass_liveness(stateful_liveness)
    assert manager.process_frame(frame).processed_faces[0].attendance_marked is True

    snapshots = list((tmp_path / f"session_{active_session}").glob("2021001_*.jpg"))
    assert len(snapshots) == 1
    image = cv2.imread(str(snapshots[0]))
    assert image is not None and max(image.shape[:2]) <= 320


def test_failed_snapshot_never_blocks_the_attendance_mark(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "ATTENDANCE_SNAPSHOTS_ENABLED", True)
    monkeypatch.setattr(config, "ATTENDANCE_SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness, None, None)
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)

    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is True


# ============================================================================
# Per-frame liveness, head-turn identity rule and background recognition
# ============================================================================

def test_per_frame_path_measures_liveness_and_process_frame_only_reads_it(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    mock_cv_components['detector'].detect_faces.return_value = [FACE]
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager.fast_liveness = True

    for _ in range(3):
        manager.update_liveness(sample_frame)
    assert stateful_liveness.extract_ear.call_count == 3
    assert stateful_liveness.update_temporal_state.call_count == 3

    manager.process_frame(sample_frame)
    assert stateful_liveness.extract_ear.call_count == 3          # not measured again
    assert stateful_liveness.update_temporal_state.call_count == 3


def test_per_frame_path_is_skipped_when_liveness_is_not_required(
    temp_db, active_session, sample_frame, mock_cv_components, stateful_liveness
):
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    manager.require_liveness = False
    manager.update_liveness(sample_frame)
    stateful_liveness.extract_ear.assert_not_called()


def test_recognition_failing_during_a_head_turn_keeps_liveness_progress(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    """Alice turns her head for the challenge and is briefly 'Unknown': her blinks stay."""
    recognizer = mock_cv_components['recognizer']
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    manager.track_reverify_seconds = 0.0       # recognise on every frame
    mock_cv_components['detector'].detect_faces.return_value = [FACE]

    recognizer.recognize.return_value = ALICE
    manager.process_frame(sample_frame)
    stateful_liveness.blink_count = 1

    recognizer.recognize.return_value = STRANGER   # face turned: no confident match
    turned = manager.process_frame(sample_frame).processed_faces[0]
    assert stateful_liveness.blink_count == 1
    assert turned.attendance_marked is False

    recognizer.recognize.return_value = ALICE      # facing the camera again
    _pass_liveness(stateful_liveness)
    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is True


def test_face_box_jump_is_treated_as_a_new_person(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    stateful_liveness.blink_count = 1

    detector.detect_faces.return_value = [DetectedFace(bbox=(60, 60, 30, 30), confidence=0.9)]
    manager.process_frame(sample_frame)

    assert stateful_liveness.blink_count == 0


def test_live_progress_is_copied_into_the_displayed_result(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    mock_cv_components['detector'].detect_faces.return_value = [FACE]
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager.fast_liveness = True
    result = manager.process_frame(sample_frame)
    assert result.processed_faces[0].blink_count == 0

    stateful_liveness.blink_count = 2               # blinks seen by the per-frame path
    stateful_liveness.get_challenge_progress.return_value = {
        'challenge_type': 'turn_left', 'challenge_state': 'active',
        'frames_held': 1, 'frames_required': 3, 'progress_percent': 0.33,
    }
    manager.refresh_liveness_display(result)

    face = result.processed_faces[0]
    assert face.blink_count == 2
    assert face.challenge_type == 'turn_left' and face.challenge_progress == pytest.approx(0.33)


def test_background_worker_processes_the_newest_frame(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    mock_cv_components['detector'].detect_faces.return_value = [FACE]
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager.enable_background_processing()
    try:
        first = manager.process_frame_async(sample_frame)
        assert first.faces_detected == 0            # nothing finished yet: returns at once
        deadline = time.time() + 5.0
        result = first
        while result.faces_detected == 0 and time.time() < deadline:
            time.sleep(0.01)
            result = manager.process_frame_async(sample_frame)
        assert result.faces_detected == 1
        assert result.processed_faces[0].identity == "2021001"
    finally:
        manager.disable_background_processing()
    assert manager._bg_worker_thread is None


# ============================================================================
# Face-lost grace period and the "Require head turn" switch
# ============================================================================

def test_briefly_lost_face_keeps_liveness_progress(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch
):
    """The detector loses a face turned for the challenge; that is not the person leaving."""
    monkeypatch.setattr(config, "LIVENESS_FACE_LOST_GRACE", 2.0)
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    stateful_liveness.blink_count = 1

    detector.detect_faces.return_value = []          # head turned: face not detected
    manager.process_frame(sample_frame)
    manager.process_frame(sample_frame)
    assert stateful_liveness.blink_count == 1

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    assert stateful_liveness.blink_count == 1


def test_face_gone_longer_than_the_grace_period_restarts_liveness(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch
):
    monkeypatch.setattr(config, "LIVENESS_FACE_LOST_GRACE", 2.0)
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)

    detector.detect_faces.return_value = [FACE]
    manager.process_frame(sample_frame)
    stateful_liveness.blink_count = 1

    detector.detect_faces.return_value = []
    manager.process_frame(sample_frame)
    manager._face_missing_since -= 2.5               # gone for 2.5 s
    manager.process_frame(sample_frame)

    assert stateful_liveness.blink_count == 0


def test_head_turn_can_be_switched_off(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components, stateful_liveness
):
    """With the switch off, blinks (plus the spoof checks) are enough."""
    detector = mock_cv_components['detector']
    mock_cv_components['recognizer'].recognize.return_value = ALICE
    manager = _liveness_manager(temp_db, active_session, mock_cv_components, stateful_liveness)
    manager.require_head_turn = False
    detector.detect_faces.return_value = [FACE]

    manager.process_frame(sample_frame)
    stateful_liveness.start_new_challenge.assert_not_called()
    assert stateful_liveness.require_challenge is False

    stateful_liveness.blink_count = stateful_liveness.min_blinks
    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is True


# ============================================================================
# A suspected spoof is reported whoever the face is
# ============================================================================

def test_spoof_is_reported_for_an_unknown_face(
    temp_db, active_session, sample_frame, mock_cv_components, stateful_liveness, spoof_checks_on
):
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             _fake_anti_spoof(0.1), _fake_screen())
    mock_cv_components['recognizer'].recognize.return_value = STRANGER
    for _ in range(3):
        face = manager.process_frame(sample_frame).processed_faces[0]
    assert face.reason.startswith(SPOOF_RECHECK_PREFIX)
    assert face.attendance_marked is False


def test_spoof_is_reported_for_an_already_marked_student(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, spoof_checks_on
):
    anti_spoof = _fake_anti_spoof(0.95)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    manager.process_frame(sample_frame)
    assert manager.process_frame(sample_frame).processed_faces[0].attendance_marked is True

    anti_spoof.real_score.return_value = 0.05    # now her photo on a phone
    for _ in range(3):
        face = manager.process_frame(sample_frame).processed_faces[0]
    assert face.reason.startswith("Spoof suspected")


# ============================================================================
# Per-frame anti-spoof rule (calibrated: real faces >= 0.89 on every frame,
# faces on a phone flip between ~0 and ~1)
# ============================================================================

def test_phone_scores_that_average_high_are_still_blocked(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch, spoof_checks_on
):
    """1, 1, 0, 1, 1, 0 ... averages 0.67 - the old averaged rule let this pass."""
    monkeypatch.setattr(config, "ANTI_SPOOF_THRESHOLD", 0.8)
    anti_spoof = _fake_anti_spoof(1.0)
    anti_spoof.real_score.side_effect = [1.0, 1.0, 0.0] * 10
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(12)]

    assert not any(face.attendance_marked for face in faces)
    assert faces[-1].reason.startswith(SPOOF_RECHECK_PREFIX)   # the recheck keeps failing too


def test_spoof_block_lasts_until_the_person_leaves(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch, spoof_checks_on
):
    """Without a recheck left, a model flag is final until the person leaves."""
    monkeypatch.setattr(config, "ANTI_SPOOF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "ANTI_SPOOF_MAX_RECHECKS", 0)
    anti_spoof = _fake_anti_spoof(0.0)
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    manager.process_frame(sample_frame)
    manager.process_frame(sample_frame)              # two spoof frames: blocked

    anti_spoof.real_score.return_value = 1.0         # a lucky streak of "real" frames
    _pass_liveness(stateful_liveness)
    for _ in range(12):
        face = manager.process_frame(sample_frame).processed_faces[0]
    assert face.attendance_marked is False
    assert face.reason == FINAL_SPOOF_REASON

    detector = mock_cv_components['detector']
    detector.detect_faces.return_value = []          # leaves the view
    manager.process_frame(sample_frame)
    detector.detect_faces.return_value = [FACE]      # a real face comes back
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    marked = [manager.process_frame(sample_frame).processed_faces[0].attendance_marked for _ in range(3)]
    assert any(marked)


def test_real_face_scores_are_marked(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch, spoof_checks_on
):
    """Scores as measured for a real face (0.89-1.0) never block."""
    monkeypatch.setattr(config, "ANTI_SPOOF_THRESHOLD", 0.8)
    anti_spoof = _fake_anti_spoof(1.0)
    anti_spoof.real_score.side_effect = [0.89, 1.0, 0.97, 1.0, 0.93, 1.0, 1.0, 1.0]
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    marked = [manager.process_frame(sample_frame).processed_faces[0].attendance_marked for _ in range(4)]
    assert any(marked)


def test_a_recent_low_frame_delays_marking(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, monkeypatch, spoof_checks_on
):
    monkeypatch.setattr(config, "ANTI_SPOOF_THRESHOLD", 0.8)
    anti_spoof = _fake_anti_spoof(1.0)
    anti_spoof.real_score.side_effect = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             anti_spoof, _fake_screen())
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    reasons = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(5)]
    # one low frame is not a block, but marking waits for 3 real frames after it
    assert reasons[1].reason.startswith("Checking for photo / screen spoofing")
    assert reasons[-1].attendance_marked is True


# ============================================================================
# Spoof recheck: a low-quality webcam gives a real face a few low-scoring
# frames; the face is checked again instead of being blocked until it leaves
# ============================================================================

@pytest.fixture
def quick_recheck(monkeypatch, spoof_checks_on):
    """Recheck over 5 frames (at most 1 low), the last 3 real; threshold 0.8."""
    monkeypatch.setattr(config, "ANTI_SPOOF_THRESHOLD", 0.8)
    monkeypatch.setattr(config, "ANTI_SPOOF_RECHECK_FRAMES", 5)
    monkeypatch.setattr(config, "ANTI_SPOOF_RECHECK_MAX_BAD", 1)
    monkeypatch.setattr(config, "ANTI_SPOOF_RECHECK_TIMEOUT", 10.0)
    monkeypatch.setattr(config, "ANTI_SPOOF_MAX_RECHECKS", 2)


def _recheck_manager(temp_db, active_session, mock_cv_components, liveness, scores):
    """A manager whose anti-spoof model returns the given scores, one per processed frame."""
    anti_spoof = _fake_anti_spoof(1.0)
    anti_spoof.real_score.side_effect = scores
    return _spoof_manager(temp_db, active_session, mock_cv_components, liveness,
                          anti_spoof, _fake_screen())


def test_real_face_flagged_by_a_poor_camera_passes_the_recheck(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck
):
    scores = [0.3, 0.5] + [0.95, 0.6, 0.95, 0.95, 0.95] + [0.95] * 5
    manager = _recheck_manager(temp_db, active_session, mock_cv_components, stateful_liveness, scores)
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    flagged = manager.process_frame(sample_frame).processed_faces[0]    # 2 low frames
    assert flagged.reason.startswith(SPOOF_RECHECK_PREFIX)
    assert flagged.attendance_marked is False

    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(5)]
    assert not any(face.attendance_marked for face in faces[:4])        # recheck still running
    assert faces[-1].attendance_marked is True                          # 1 low of 5, last 3 real


def test_photo_keeps_failing_the_recheck_and_is_blocked_after_the_time_limit(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck
):
    manager = _recheck_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                               [0.1] * 12 + [1.0] * 10)
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    for _ in range(9):
        face = manager.process_frame(sample_frame).processed_faces[0]
    assert face.reason.startswith(SPOOF_RECHECK_PREFIX) and face.attendance_marked is False

    manager._spoof_flagged_at -= config.ANTI_SPOOF_RECHECK_TIMEOUT + 1   # the time limit passes
    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(10)]
    assert not any(face.attendance_marked for face in faces)            # real-looking frames come too late
    assert faces[-1].reason == FINAL_SPOOF_REASON


def test_phone_flipping_between_real_and_fake_never_passes_the_recheck(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck
):
    manager = _recheck_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                               [1.0, 0.0] * 20)
    manager.process_frame(sample_frame)
    _pass_liveness(stateful_liveness)
    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(30)]
    assert not any(face.attendance_marked for face in faces)
    assert faces[-1].reason.startswith(SPOOF_RECHECK_PREFIX)


def test_the_number_of_rechecks_is_limited(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck, monkeypatch
):
    monkeypatch.setattr(config, "ANTI_SPOOF_MAX_RECHECKS", 1)
    scores = [0.1, 0.1] + [0.95] * 5 + [0.1, 0.1] + [0.95] * 10
    manager = _recheck_manager(temp_db, active_session, mock_cv_components, stateful_liveness, scores)
    for _ in range(7):                                    # flagged, then the recheck passes
        manager.process_frame(sample_frame)
    assert not manager._spoof_flagged
    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(12)]
    assert not any(face.attendance_marked for face in faces)
    assert faces[-1].reason == FINAL_SPOOF_REASON         # flagged again: no recheck left


def test_rechecks_start_again_for_the_next_person(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck, monkeypatch
):
    monkeypatch.setattr(config, "ANTI_SPOOF_MAX_RECHECKS", 0)
    manager = _recheck_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                               [0.1, 0.1, 0.1])
    for _ in range(3):
        manager.process_frame(sample_frame)
    assert manager._spoof_final
    mock_cv_components['detector'].detect_faces.return_value = []    # leaves the view
    manager.process_frame(sample_frame)
    assert not manager._spoof_flagged and not manager._spoof_final
    assert manager._spoof_rechecks_passed == 0


def test_a_face_on_a_phone_is_never_rechecked(
    temp_db, active_session, sample_student, sample_frame, mock_cv_components,
    stateful_liveness, quick_recheck
):
    screen = _fake_screen("cell phone")
    manager = _spoof_manager(temp_db, active_session, mock_cv_components, stateful_liveness,
                             _fake_anti_spoof(0.95), screen)
    manager.process_frame(sample_frame)
    screen.screen_covering_face.return_value = None
    _pass_liveness(stateful_liveness)
    faces = [manager.process_frame(sample_frame).processed_faces[0] for _ in range(10)]
    assert not any(face.attendance_marked for face in faces)
    assert "cell phone" in faces[-1].reason
