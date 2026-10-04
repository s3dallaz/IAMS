"""
Session Service Tests

Tests business logic, validation, and coordination for session management.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import tempfile
from pathlib import Path

from src.services.session_service import (
    SessionService,
    SessionStatus,
    InvalidSessionDataError,
    SessionNotFoundError,
    InvalidSessionStateError
)
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
def service(temp_db):
    """Create a SessionService instance with test database."""
    return SessionService(temp_db)


# ============================================================================
# Test Session Creation
# ============================================================================

def test_create_session_valid(service):
    """Test creating a valid session."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101",
        end_time="2026-10-04 11:00:00",
        late_threshold_minutes=15,
        created_by="admin",
        notes="Test session"
    )
    
    assert session_id is not None
    assert isinstance(session_id, int)
    assert session_id > 0
    
    # Verify session was created
    session = service.get_session(session_id)
    assert session is not None
    assert session['session_name'] == "CS101 Lecture"
    assert session['course_code'] == "CS101"
    assert session['status'] == SessionStatus.SCHEDULED


def test_create_session_minimal_fields(service):
    """Test creating session with only required fields."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = service.get_session(session_id)
    assert session is not None
    assert session['session_name'] == "CS101 Lecture"
    assert session['start_time'] == "2026-10-04 10:00:00"
    assert session['course_code'] is None
    assert session['end_time'] is None
    assert session['late_threshold_minutes'] == 15  # Default
    assert session['status'] == SessionStatus.SCHEDULED


def test_create_session_with_custom_status(service):
    """Test creating session with custom initial status."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.ACTIVE
    )
    
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.ACTIVE


# ============================================================================
# Test Session Creation Validation
# ============================================================================

def test_create_session_empty_name_rejected(service):
    """Test that empty session name is rejected."""
    with pytest.raises(InvalidSessionDataError, match="session_name cannot be empty"):
        service.create_session(
            session_name="   ",
            start_time="2026-10-04 10:00:00"
        )


def test_create_session_invalid_name_type_rejected(service):
    """Test that invalid session name type is rejected."""
    with pytest.raises(InvalidSessionDataError, match="session_name must be a string"):
        service.create_session(
            session_name=123,
            start_time="2026-10-04 10:00:00"
        )


def test_create_session_empty_start_time_rejected(service):
    """Test that empty start time is rejected."""
    with pytest.raises(InvalidSessionDataError, match="start_time cannot be empty"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="   "
        )


def test_create_session_invalid_start_time_format_rejected(service):
    """Test that invalid start time format is rejected."""
    with pytest.raises(InvalidSessionDataError, match="valid ISO format datetime"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="invalid-date"
        )


def test_create_session_invalid_end_time_format_rejected(service):
    """Test that invalid end time format is rejected."""
    with pytest.raises(InvalidSessionDataError, match="valid ISO format datetime"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="2026-10-04 10:00:00",
            end_time="invalid-date"
        )


def test_create_session_end_before_start_rejected(service):
    """Test that end time before start time is rejected."""
    with pytest.raises(InvalidSessionDataError, match="end_time must be after start_time"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="2026-10-04 10:00:00",
            end_time="2026-10-04 09:00:00"
        )


def test_create_session_negative_late_threshold_rejected(service):
    """Test that negative late threshold is rejected."""
    with pytest.raises(InvalidSessionDataError, match="late_threshold_minutes cannot be negative"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="2026-10-04 10:00:00",
            late_threshold_minutes=-5
        )


def test_create_session_invalid_status_rejected(service):
    """Test that invalid status is rejected."""
    with pytest.raises(InvalidSessionDataError, match="Invalid status"):
        service.create_session(
            session_name="CS101 Lecture",
            start_time="2026-10-04 10:00:00",
            status="invalid"
        )


def test_create_session_whitespace_trimmed(service):
    """Test that whitespace is trimmed from fields."""
    session_id = service.create_session(
        session_name="  CS101 Lecture  ",
        start_time="  2026-10-04 10:00:00  ",
        course_code="  CS101  ",
        created_by="  admin  ",
        notes="  Test notes  "
    )
    
    session = service.get_session(session_id)
    assert session['session_name'] == "CS101 Lecture"
    assert session['course_code'] == "CS101"
    assert session['created_by'] == "admin"
    assert session['notes'] == "Test notes"


# ============================================================================
# Test Session Retrieval
# ============================================================================

def test_get_session_by_id(service):
    """Test retrieving session by ID."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = service.get_session(session_id)
    assert session is not None
    assert session['id'] == session_id
    assert session['session_name'] == "CS101 Lecture"


def test_get_session_nonexistent(service):
    """Test retrieving nonexistent session returns None."""
    session = service.get_session(99999)
    assert session is None


def test_get_session_invalid_id_rejected(service):
    """Test that invalid session ID is rejected."""
    with pytest.raises(InvalidSessionDataError, match="session_id must be a positive integer"):
        service.get_session(0)
    
    with pytest.raises(InvalidSessionDataError, match="session_id must be a positive integer"):
        service.get_session(-1)


def test_get_all_sessions(service):
    """Test retrieving all sessions."""
    service.create_session("Session 1", "2026-10-04 10:00:00")
    service.create_session("Session 2", "2026-10-05 10:00:00")
    service.create_session("Session 3", "2026-10-06 10:00:00")
    
    sessions = service.get_all_sessions()
    assert len(sessions) == 3


def test_get_all_sessions_filtered_by_status(service):
    """Test retrieving sessions filtered by status."""
    id1 = service.create_session("Session 1", "2026-10-04 10:00:00", status=SessionStatus.SCHEDULED)
    id2 = service.create_session("Session 2", "2026-10-05 10:00:00", status=SessionStatus.ACTIVE)
    id3 = service.create_session("Session 3", "2026-10-06 10:00:00", status=SessionStatus.CLOSED)
    
    scheduled = service.get_all_sessions(status=SessionStatus.SCHEDULED)
    assert len(scheduled) == 1
    assert scheduled[0]['session_name'] == "Session 1"
    
    active = service.get_all_sessions(status=SessionStatus.ACTIVE)
    assert len(active) == 1
    assert active[0]['session_name'] == "Session 2"


def test_get_all_sessions_invalid_status_rejected(service):
    """Test that invalid status filter is rejected."""
    with pytest.raises(InvalidSessionDataError, match="Invalid status"):
        service.get_all_sessions(status="invalid")


def test_get_sessions_by_course(service):
    """Test retrieving sessions by course code."""
    service.create_session("CS101 Lecture 1", "2026-10-04 10:00:00", course_code="CS101")
    service.create_session("CS101 Lecture 2", "2026-10-05 10:00:00", course_code="CS101")
    service.create_session("CS102 Lecture", "2026-10-06 10:00:00", course_code="CS102")
    
    cs101_sessions = service.get_sessions_by_course("CS101")
    assert len(cs101_sessions) == 2
    assert all(s['course_code'] == "CS101" for s in cs101_sessions)


def test_get_sessions_by_course_empty_rejected(service):
    """Test that empty course code is rejected."""
    with pytest.raises(InvalidSessionDataError, match="course_code cannot be empty"):
        service.get_sessions_by_course("   ")


# ============================================================================
# Test Session Lifecycle - Activate
# ============================================================================

def test_activate_session_from_scheduled(service):
    """Test activating a scheduled session."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )
    
    rows_affected = service.activate_session(session_id)
    assert rows_affected == 1
    
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.ACTIVE


def test_activate_session_already_closed_rejected(service):
    """Test that closed session cannot be activated."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )
    
    with pytest.raises(InvalidSessionStateError, match="Cannot activate session"):
        service.activate_session(session_id)


def test_activate_session_nonexistent_rejected(service):
    """Test that activating nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        service.activate_session(99999)


def test_activate_session_invalid_id_rejected(service):
    """Test that invalid session ID is rejected."""
    with pytest.raises(InvalidSessionDataError, match="session_id must be a positive integer"):
        service.activate_session(0)


# ============================================================================
# Test Session Lifecycle - Close
# ============================================================================

def test_close_session_from_active(service):
    """Test closing an active session."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.ACTIVE
    )
    
    rows_affected = service.close_session(session_id)
    assert rows_affected == 1
    
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.CLOSED
    assert session['closed_at'] is not None


def test_close_session_from_scheduled(service):
    """Test closing a scheduled session."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )
    
    rows_affected = service.close_session(session_id)
    assert rows_affected == 1
    
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.CLOSED


def test_close_session_already_closed_rejected(service):
    """Test that closing already closed session raises error."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )
    
    with pytest.raises(InvalidSessionStateError, match="already closed"):
        service.close_session(session_id)


def test_close_session_nonexistent_rejected(service):
    """Test that closing nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        service.close_session(99999)


# ============================================================================
# Test Session Lifecycle - Valid Transitions
# ============================================================================

def test_session_lifecycle_full_flow(service):
    """Test full session lifecycle: scheduled -> active -> closed."""
    # Create scheduled session
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.SCHEDULED
    
    # Activate session
    service.activate_session(session_id)
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.ACTIVE
    
    # Close session
    service.close_session(session_id)
    session = service.get_session(session_id)
    assert session['status'] == SessionStatus.CLOSED


# ============================================================================
# Test Attendance Acceptance
# ============================================================================

def test_can_accept_attendance_active_session(service):
    """Test that active session can accept attendance."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.ACTIVE
    )
    
    assert service.can_accept_attendance(session_id) is True


def test_can_accept_attendance_scheduled_session(service):
    """Test that scheduled session cannot accept attendance."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )
    
    assert service.can_accept_attendance(session_id) is False


def test_can_accept_attendance_closed_session(service):
    """Test that closed session cannot accept attendance."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )
    
    assert service.can_accept_attendance(session_id) is False


def test_can_accept_attendance_nonexistent_session(service):
    """Test that checking nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        service.can_accept_attendance(99999)


# ============================================================================
# Test Session Exists
# ============================================================================

def test_session_exists(service):
    """Test checking if session exists."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    assert service.session_exists(session_id) is True
    assert service.session_exists(99999) is False


# ============================================================================
# Test Session Update
# ============================================================================

def test_update_session(service):
    """Test updating session information."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    
    rows_affected = service.update_session(
        session_id,
        session_name="CS101 Advanced Lecture",
        course_code="CS101A",
        notes="Updated session"
    )
    
    assert rows_affected == 1
    
    session = service.get_session(session_id)
    assert session['session_name'] == "CS101 Advanced Lecture"
    assert session['course_code'] == "CS101A"
    assert session['notes'] == "Updated session"


def test_update_session_partial(service):
    """Test updating only some session fields."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    
    rows_affected = service.update_session(session_id, session_name="CS101 Lab")
    assert rows_affected == 1
    
    session = service.get_session(session_id)
    assert session['session_name'] == "CS101 Lab"
    assert session['course_code'] == "CS101"  # Unchanged


def test_update_session_nonexistent_rejected(service):
    """Test that updating nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        service.update_session(99999, session_name="New Name")


def test_update_session_invalid_name_rejected(service):
    """Test that invalid session name is rejected."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    with pytest.raises(InvalidSessionDataError, match="session_name cannot be empty"):
        service.update_session(session_id, session_name="   ")


def test_update_session_invalid_late_threshold_rejected(service):
    """Test that invalid late threshold is rejected."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    with pytest.raises(InvalidSessionDataError, match="late_threshold_minutes cannot be negative"):
        service.update_session(session_id, late_threshold_minutes=-5)


# ============================================================================
# Test Late Threshold
# ============================================================================

def test_late_threshold_default(service):
    """Test that default late threshold is 15 minutes."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = service.get_session(session_id)
    assert session['late_threshold_minutes'] == 15


def test_late_threshold_custom(service):
    """Test setting custom late threshold."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=20
    )
    
    session = service.get_session(session_id)
    assert session['late_threshold_minutes'] == 20


def test_late_threshold_zero_allowed(service):
    """Test that zero late threshold is allowed."""
    session_id = service.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=0
    )
    
    session = service.get_session(session_id)
    assert session['late_threshold_minutes'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
