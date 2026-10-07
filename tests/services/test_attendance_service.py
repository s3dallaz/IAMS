"""
Attendance Service Tests

Tests business logic, validation, and coordination for attendance management.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.services.attendance_service import (
    AttendanceService,
    AttendanceStatus,
    InvalidAttendanceDataError,
    AttendanceNotFoundError,
    DuplicateAttendanceError,
    SessionNotAcceptingAttendanceError
)
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService
from src.database.schema import initialize_database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    # Initialize schema
    initialize_database(db_path)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def attendance_service(temp_db):
    """Create an AttendanceService instance with test database."""
    return AttendanceService(temp_db)


@pytest.fixture
def session_service(temp_db):
    """Create a SessionService instance with test database."""
    return SessionService(temp_db)


@pytest.fixture
def student_service(temp_db):
    """Create a StudentService instance with test database."""
    return StudentService(temp_db)


@pytest.fixture
def active_session(session_service):
    """Create an active session for testing."""
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    return session_id


@pytest.fixture
def sample_student(student_service):
    """Create a sample student for testing."""
    student_id = student_service.create_student(
        student_id="2021001",
        name="John Doe"
    )
    return student_id


# ============================================================================
# Test Attendance Recording - Valid Cases
# ============================================================================

def test_record_attendance_valid(attendance_service, active_session, sample_student):
    """Test recording valid attendance."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        marked_at="2026-10-04 10:05:00",
        detection_confidence=0.95,
        liveness_score=0.88,
        marked_by="system",
        notes="Test attendance"
    )
    
    assert attendance_id is not None
    assert isinstance(attendance_id, int)
    assert attendance_id > 0
    
    # Verify attendance was recorded
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance is not None
    assert attendance['session_id'] == active_session
    assert attendance['student_id'] == sample_student
    assert attendance['status'] == AttendanceStatus.PRESENT
    assert attendance['detection_confidence'] == 0.95
    assert attendance['liveness_score'] == 0.88


def test_record_attendance_minimal_fields(attendance_service, active_session, sample_student):
    """Test recording attendance with minimal fields."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance is not None
    assert attendance['session_id'] == active_session
    assert attendance['student_id'] == sample_student
    assert attendance['marked_by'] == 'system'
    assert attendance['detection_confidence'] is None
    assert attendance['liveness_score'] is None


def test_record_attendance_without_marked_at_uses_current_time(attendance_service, active_session, sample_student):
    """Test that attendance without marked_at uses current timestamp."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['marked_at'] is not None
    
    # Verify marked_at is a valid timestamp string
    marked_time = datetime.fromisoformat(attendance['marked_at'].replace(' ', 'T'))
    assert isinstance(marked_time, datetime)
    
    # Verify it's a reasonable timestamp (not in distant past or future)
    # Should be within 1 day of 2026-10-04
    reference_date = datetime.now()
    assert abs((marked_time - reference_date).total_seconds()) < 86400


# ============================================================================
# Test Attendance Status Calculation - Present vs Late
# ============================================================================

def test_attendance_status_present_before_threshold(attendance_service, session_service, sample_student):
    """Test that attendance before late threshold is marked as present."""
    # Create session starting at 10:00 with 15 min late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    
    # Record attendance at 10:05 (5 minutes after start)
    attendance_id = attendance_service.record_attendance(
        session_id=session_id,
        student_id=sample_student,
        marked_at="2026-10-04 10:05:00"
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['status'] == AttendanceStatus.PRESENT


def test_attendance_status_present_just_before_threshold(attendance_service, session_service, sample_student):
    """Test attendance just before threshold is marked as present."""
    # Create session starting at 10:00 with 15 min late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    
    # Record attendance at 10:14:59 (14 minutes 59 seconds after start)
    attendance_id = attendance_service.record_attendance(
        session_id=session_id,
        student_id=sample_student,
        marked_at="2026-10-04 10:14:59"
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['status'] == AttendanceStatus.PRESENT


def test_attendance_status_late_at_threshold(attendance_service, session_service, sample_student):
    """Test attendance exactly at threshold is marked as late."""
    # Create session starting at 10:00 with 15 min late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    
    # Record attendance at 10:15:00 (exactly 15 minutes after start)
    attendance_id = attendance_service.record_attendance(
        session_id=session_id,
        student_id=sample_student,
        marked_at="2026-10-04 10:15:00"
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['status'] == AttendanceStatus.LATE


def test_attendance_status_late_after_threshold(attendance_service, session_service, sample_student):
    """Test attendance after threshold is marked as late."""
    # Create session starting at 10:00 with 15 min late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    
    # Record attendance at 10:20:00 (20 minutes after start)
    attendance_id = attendance_service.record_attendance(
        session_id=session_id,
        student_id=sample_student,
        marked_at="2026-10-04 10:20:00"
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['status'] == AttendanceStatus.LATE


def test_attendance_status_with_zero_late_threshold(attendance_service, session_service, student_service):
    """Test attendance status with zero late threshold."""
    # Create session with 0 minute late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=0,
        status=SessionStatus.ACTIVE
    )
    
    student1 = student_service.create_student("2021001", "Student 1")
    student2 = student_service.create_student("2021002", "Student 2")
    
    # Attendance before start time should be present
    attendance_id1 = attendance_service.record_attendance(
        session_id=session_id,
        student_id=student1,
        marked_at="2026-10-04 09:59:59"
    )
    
    # Attendance at or after start time with zero threshold should be late
    attendance_id2 = attendance_service.record_attendance(
        session_id=session_id,
        student_id=student2,
        marked_at="2026-10-04 10:00:00"
    )
    
    attendance1 = attendance_service.get_attendance(attendance_id1)
    attendance2 = attendance_service.get_attendance(attendance_id2)
    
    assert attendance1['status'] == AttendanceStatus.PRESENT
    assert attendance2['status'] == AttendanceStatus.LATE


def test_attendance_status_with_custom_late_threshold(attendance_service, session_service, student_service):
    """Test attendance status with custom late threshold."""
    # Create session with 30 minute late threshold
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=30,
        status=SessionStatus.ACTIVE
    )
    
    student1 = student_service.create_student("2021001", "Student 1")
    student2 = student_service.create_student("2021002", "Student 2")
    
    # Attendance at 10:25 should be present (< 30 min)
    attendance_id1 = attendance_service.record_attendance(
        session_id=session_id,
        student_id=student1,
        marked_at="2026-10-04 10:25:00"
    )
    
    # Attendance at 10:35 should be late (>= 30 min)
    attendance_id2 = attendance_service.record_attendance(
        session_id=session_id,
        student_id=student2,
        marked_at="2026-10-04 10:35:00"
    )
    
    attendance1 = attendance_service.get_attendance(attendance_id1)
    attendance2 = attendance_service.get_attendance(attendance_id2)
    
    assert attendance1['status'] == AttendanceStatus.PRESENT
    assert attendance2['status'] == AttendanceStatus.LATE


# ============================================================================
# Test Session State Validation
# ============================================================================

def test_record_attendance_scheduled_session_rejected(attendance_service, session_service, sample_student):
    """Test that scheduled session cannot accept attendance."""
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )
    
    with pytest.raises(SessionNotAcceptingAttendanceError, match="cannot accept attendance"):
        attendance_service.record_attendance(
            session_id=session_id,
            student_id=sample_student
        )


def test_record_attendance_closed_session_rejected(attendance_service, session_service, sample_student):
    """Test that closed session cannot accept attendance."""
    session_id = session_service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )
    
    with pytest.raises(SessionNotAcceptingAttendanceError, match="cannot accept attendance"):
        attendance_service.record_attendance(
            session_id=session_id,
            student_id=sample_student
        )


def test_record_attendance_nonexistent_session_rejected(attendance_service, sample_student):
    """Test that nonexistent session is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="Session with ID .* does not exist"):
        attendance_service.record_attendance(
            session_id=99999,
            student_id=sample_student
        )


# ============================================================================
# Test Student Validation
# ============================================================================

def test_record_attendance_nonexistent_student_rejected(attendance_service, active_session):
    """Test that nonexistent student is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="Student with ID .* does not exist"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=99999
        )


def test_record_attendance_invalid_student_id_rejected(attendance_service, active_session):
    """Test that invalid student ID is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="student_id must be a positive integer"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=0
        )


def test_record_attendance_invalid_session_id_rejected(attendance_service, sample_student):
    """Test that invalid session ID is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="session_id must be a positive integer"):
        attendance_service.record_attendance(
            session_id=-1,
            student_id=sample_student
        )


# ============================================================================
# Test Duplicate Attendance Prevention
# ============================================================================

def test_record_attendance_duplicate_rejected(attendance_service, active_session, sample_student):
    """Test that duplicate attendance is rejected."""
    # Record first attendance
    attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    # Attempt to record duplicate
    with pytest.raises(DuplicateAttendanceError, match="Attendance already recorded"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=sample_student
        )


def test_record_attendance_same_student_different_sessions_allowed(
    attendance_service, session_service, sample_student
):
    """Test that same student can have attendance in different sessions."""
    session1 = session_service.create_session(
        session_name="Session 1",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.ACTIVE
    )
    session2 = session_service.create_session(
        session_name="Session 2",
        start_time="2026-10-05 10:00:00",
        status=SessionStatus.ACTIVE
    )
    
    # Record attendance in both sessions
    attendance_id1 = attendance_service.record_attendance(
        session_id=session1,
        student_id=sample_student
    )
    attendance_id2 = attendance_service.record_attendance(
        session_id=session2,
        student_id=sample_student
    )
    
    assert attendance_id1 != attendance_id2


def test_record_attendance_different_students_same_session_allowed(
    attendance_service, active_session, student_service
):
    """Test that different students can have attendance in same session."""
    student1 = student_service.create_student("2021001", "Student 1")
    student2 = student_service.create_student("2021002", "Student 2")
    
    # Record attendance for both students
    attendance_id1 = attendance_service.record_attendance(
        session_id=active_session,
        student_id=student1
    )
    attendance_id2 = attendance_service.record_attendance(
        session_id=active_session,
        student_id=student2
    )
    
    assert attendance_id1 != attendance_id2


# ============================================================================
# Test Attendance Exists
# ============================================================================

def test_attendance_exists(attendance_service, active_session, sample_student):
    """Test checking if attendance exists."""
    assert attendance_service.attendance_exists(active_session, sample_student) is False
    
    # Record attendance
    attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    assert attendance_service.attendance_exists(active_session, sample_student) is True


# ============================================================================
# Test CV Metadata Passthrough
# ============================================================================

def test_record_attendance_with_detection_confidence(attendance_service, active_session, sample_student):
    """Test recording attendance with detection confidence."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        detection_confidence=0.92
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['detection_confidence'] == 0.92


def test_record_attendance_with_liveness_score(attendance_service, active_session, sample_student):
    """Test recording attendance with liveness score."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        liveness_score=0.85
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['liveness_score'] == 0.85


def test_record_attendance_with_all_metadata(attendance_service, active_session, sample_student):
    """Test recording attendance with all CV metadata."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        marked_at="2026-10-04 10:05:00",
        detection_confidence=0.95,
        liveness_score=0.88,
        marked_by="cv_system",
        notes="Automatic detection"
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['detection_confidence'] == 0.95
    assert attendance['liveness_score'] == 0.88
    assert attendance['marked_by'] == "cv_system"
    assert attendance['notes'] == "Automatic detection"


def test_record_attendance_invalid_detection_confidence_rejected(attendance_service, active_session, sample_student):
    """Test that invalid detection confidence is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="detection_confidence must be between 0.0 and 1.0"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=sample_student,
            detection_confidence=1.5
        )


def test_record_attendance_invalid_liveness_score_rejected(attendance_service, active_session, sample_student):
    """Test that invalid liveness score is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="liveness_score must be between 0.0 and 1.0"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=sample_student,
            liveness_score=-0.1
        )


# ============================================================================
# Test Attendance Retrieval
# ============================================================================

def test_get_attendance_by_id(attendance_service, active_session, sample_student):
    """Test retrieving attendance by ID."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance is not None
    assert attendance['id'] == attendance_id


def test_get_attendance_nonexistent(attendance_service):
    """Test retrieving nonexistent attendance returns None."""
    attendance = attendance_service.get_attendance(99999)
    assert attendance is None


def test_get_session_attendance(attendance_service, active_session, student_service):
    """Test retrieving all attendance for a session."""
    student1 = student_service.create_student("2021001", "Student 1")
    student2 = student_service.create_student("2021002", "Student 2")
    student3 = student_service.create_student("2021003", "Student 3")
    
    attendance_service.record_attendance(active_session, student1)
    attendance_service.record_attendance(active_session, student2)
    attendance_service.record_attendance(active_session, student3)
    
    records = attendance_service.get_session_attendance(active_session)
    assert len(records) == 3
    assert all(r['session_id'] == active_session for r in records)


def test_get_student_attendance(attendance_service, session_service, sample_student):
    """Test retrieving all attendance for a student."""
    session1 = session_service.create_session("Session 1", "2026-10-04 10:00:00", status=SessionStatus.ACTIVE)
    session2 = session_service.create_session("Session 2", "2026-10-05 10:00:00", status=SessionStatus.ACTIVE)
    session3 = session_service.create_session("Session 3", "2026-10-06 10:00:00", status=SessionStatus.ACTIVE)
    
    attendance_service.record_attendance(session1, sample_student)
    attendance_service.record_attendance(session2, sample_student)
    attendance_service.record_attendance(session3, sample_student)
    
    records = attendance_service.get_student_attendance(sample_student)
    assert len(records) == 3
    assert all(r['student_id'] == sample_student for r in records)


def test_get_attendance_by_session_and_student(attendance_service, active_session, sample_student):
    """Test retrieving specific attendance by session and student."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    attendance = attendance_service.get_attendance_by_session_and_student(
        active_session, sample_student
    )
    
    assert attendance is not None
    assert attendance['id'] == attendance_id


def test_get_attendance_by_status(attendance_service, active_session, student_service):
    """Test retrieving attendance by status."""
    student1 = student_service.create_student("2021001", "Student 1")
    student2 = student_service.create_student("2021002", "Student 2")
    
    # Record one present and one late
    attendance_service.record_attendance(
        active_session, student1, marked_at="2026-10-04 10:05:00"  # Present
    )
    attendance_service.record_attendance(
        active_session, student2, marked_at="2026-10-04 10:20:00"  # Late
    )
    
    present_records = attendance_service.get_attendance_by_status(AttendanceStatus.PRESENT)
    late_records = attendance_service.get_attendance_by_status(AttendanceStatus.LATE)
    
    assert len(present_records) == 1
    assert len(late_records) == 1
    assert present_records[0]['status'] == AttendanceStatus.PRESENT
    assert late_records[0]['status'] == AttendanceStatus.LATE


# ============================================================================
# Test Attendance Update
# ============================================================================

def test_update_attendance(attendance_service, active_session, sample_student):
    """Test updating attendance record."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        detection_confidence=0.85
    )
    
    rows_affected = attendance_service.update_attendance(
        attendance_id,
        status=AttendanceStatus.LATE,
        detection_confidence=0.92,
        notes="Updated by admin"
    )
    
    assert rows_affected == 1
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['status'] == AttendanceStatus.LATE
    assert attendance['detection_confidence'] == 0.92
    assert attendance['notes'] == "Updated by admin"


def test_update_attendance_with_verification(attendance_service, active_session, sample_student):
    """Test updating attendance with verification."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    rows_affected = attendance_service.update_attendance(
        attendance_id,
        verified_by="admin"
    )
    
    assert rows_affected == 1
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['verified_by'] == "admin"
    assert attendance['verified_at'] is not None


def test_update_attendance_nonexistent_rejected(attendance_service):
    """Test that updating nonexistent attendance raises error."""
    with pytest.raises(AttendanceNotFoundError):
        attendance_service.update_attendance(99999, status=AttendanceStatus.PRESENT)


def test_update_attendance_invalid_status_rejected(attendance_service, active_session, sample_student):
    """Test that invalid status is rejected."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student
    )
    
    with pytest.raises(InvalidAttendanceDataError, match="Invalid status"):
        attendance_service.update_attendance(attendance_id, status="invalid")


# ============================================================================
# Test Validation
# ============================================================================

def test_record_attendance_invalid_marked_at_format_rejected(attendance_service, active_session, sample_student):
    """Test that invalid marked_at format is rejected."""
    with pytest.raises(InvalidAttendanceDataError, match="valid ISO format datetime"):
        attendance_service.record_attendance(
            session_id=active_session,
            student_id=sample_student,
            marked_at="invalid-date"
        )


def test_record_attendance_whitespace_trimmed(attendance_service, active_session, sample_student):
    """Test that whitespace is trimmed from fields."""
    attendance_id = attendance_service.record_attendance(
        session_id=active_session,
        student_id=sample_student,
        marked_by="  admin  ",
        notes="  Test notes  "
    )
    
    attendance = attendance_service.get_attendance(attendance_id)
    assert attendance['marked_by'] == "admin"
    assert attendance['notes'] == "Test notes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
