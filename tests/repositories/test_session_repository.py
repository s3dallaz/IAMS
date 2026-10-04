"""
Session Repository Tests

Tests database persistence operations for the sessions table.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.data.repositories.session_repository import SessionRepository
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
def repo(temp_db):
    """Create a SessionRepository instance with test database."""
    return SessionRepository(temp_db)


def test_create_session(repo):
    """Test creating a session and verify insertion succeeds."""
    session_id = repo.create_session(
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


def test_create_session_minimal_fields(repo):
    """Test creating session with only required fields."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = repo.get_by_id(session_id)
    assert session is not None
    assert session['session_name'] == "CS101 Lecture"
    assert session['start_time'] == "2026-10-04 10:00:00"
    assert session['course_code'] is None
    assert session['end_time'] is None
    assert session['status'] == 'scheduled'


def test_get_by_id(repo):
    """Test retrieving a session by ID."""
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101",
        end_time="2026-10-04 11:00:00"
    )
    
    # Retrieve by ID
    session = repo.get_by_id(session_id)
    
    assert session is not None
    assert session['id'] == session_id
    assert session['session_name'] == "CS101 Lecture"
    assert session['course_code'] == "CS101"
    assert session['start_time'] == "2026-10-04 10:00:00"
    assert session['end_time'] == "2026-10-04 11:00:00"
    assert session['status'] == 'scheduled'
    assert session['created_at'] is not None


def test_get_by_id_nonexistent(repo):
    """Test retrieving nonexistent session returns None."""
    session = repo.get_by_id(99999)
    assert session is None


def test_get_all_sessions(repo):
    """Test retrieving all sessions."""
    # Create multiple sessions
    repo.create_session(
        session_name="CS101 Lecture 1",
        start_time="2026-10-04 10:00:00"
    )
    repo.create_session(
        session_name="CS101 Lecture 2",
        start_time="2026-10-05 10:00:00"
    )
    repo.create_session(
        session_name="CS102 Lecture",
        start_time="2026-10-06 10:00:00"
    )
    
    # Get all sessions
    sessions = repo.get_all()
    
    assert len(sessions) == 3
    # Should be ordered by start_time DESC
    assert sessions[0]['session_name'] == "CS102 Lecture"
    assert sessions[1]['session_name'] == "CS101 Lecture 2"
    assert sessions[2]['session_name'] == "CS101 Lecture 1"


def test_get_all_by_status(repo):
    """Test retrieving sessions filtered by status."""
    # Create sessions with different statuses
    id1 = repo.create_session(
        session_name="Scheduled Session",
        start_time="2026-10-04 10:00:00",
        status="scheduled"
    )
    id2 = repo.create_session(
        session_name="Active Session",
        start_time="2026-10-05 10:00:00",
        status="active"
    )
    id3 = repo.create_session(
        session_name="Closed Session",
        start_time="2026-10-06 10:00:00",
        status="closed"
    )
    
    # Get all sessions
    all_sessions = repo.get_all()
    assert len(all_sessions) == 3
    
    # Get only scheduled sessions
    scheduled = repo.get_all(status="scheduled")
    assert len(scheduled) == 1
    assert scheduled[0]['session_name'] == "Scheduled Session"
    
    # Get only active sessions
    active = repo.get_all(status="active")
    assert len(active) == 1
    assert active[0]['session_name'] == "Active Session"
    
    # Get only closed sessions
    closed = repo.get_all(status="closed")
    assert len(closed) == 1
    assert closed[0]['session_name'] == "Closed Session"


def test_get_by_course(repo):
    """Test retrieving sessions by course code."""
    # Create sessions for different courses
    repo.create_session(
        session_name="CS101 Lecture 1",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    repo.create_session(
        session_name="CS101 Lecture 2",
        start_time="2026-10-05 10:00:00",
        course_code="CS101"
    )
    repo.create_session(
        session_name="CS102 Lecture",
        start_time="2026-10-06 10:00:00",
        course_code="CS102"
    )
    
    # Get CS101 sessions
    cs101_sessions = repo.get_by_course("CS101")
    assert len(cs101_sessions) == 2
    assert all(s['course_code'] == "CS101" for s in cs101_sessions)
    
    # Get CS102 sessions
    cs102_sessions = repo.get_by_course("CS102")
    assert len(cs102_sessions) == 1
    assert cs102_sessions[0]['course_code'] == "CS102"
    
    # Get nonexistent course
    cs999_sessions = repo.get_by_course("CS999")
    assert len(cs999_sessions) == 0


def test_update_session(repo):
    """Test updating session information."""
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    
    # Update session
    rows_affected = repo.update_session(
        session_id,
        session_name="CS101 Advanced Lecture",
        course_code="CS101A",
        end_time="2026-10-04 11:30:00",
        notes="Updated session"
    )
    
    assert rows_affected == 1
    
    # Verify changes persisted
    session = repo.get_by_id(session_id)
    assert session['session_name'] == "CS101 Advanced Lecture"
    assert session['course_code'] == "CS101A"
    assert session['end_time'] == "2026-10-04 11:30:00"
    assert session['notes'] == "Updated session"
    assert session['start_time'] == "2026-10-04 10:00:00"  # Unchanged


def test_update_session_partial(repo):
    """Test updating only some session fields."""
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    
    # Update only session name
    rows_affected = repo.update_session(session_id, session_name="CS101 Lab")
    
    assert rows_affected == 1
    
    # Verify only name changed
    session = repo.get_by_id(session_id)
    assert session['session_name'] == "CS101 Lab"
    assert session['course_code'] == "CS101"  # Unchanged


def test_update_session_status(repo):
    """Test updating session status."""
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status="scheduled"
    )
    
    # Update status to active
    rows_affected = repo.update_session(session_id, status="active")
    assert rows_affected == 1
    
    session = repo.get_by_id(session_id)
    assert session['status'] == "active"
    
    # Update status to closed
    rows_affected = repo.update_session(session_id, status="closed")
    assert rows_affected == 1
    
    session = repo.get_by_id(session_id)
    assert session['status'] == "closed"


def test_update_nonexistent_session(repo):
    """Test updating a nonexistent session."""
    rows_affected = repo.update_session(99999, session_name="New Name")
    assert rows_affected == 0


def test_update_with_no_fields(repo):
    """Test update with no fields provided."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    # Update with no fields
    rows_affected = repo.update_session(session_id)
    assert rows_affected == 0


def test_close_session(repo):
    """Test closing a session."""
    # Create active session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status="active"
    )
    
    # Close session
    rows_affected = repo.close_session(session_id)
    assert rows_affected == 1
    
    # Verify session is closed
    session = repo.get_by_id(session_id)
    assert session['status'] == 'closed'
    assert session['closed_at'] is not None


def test_close_nonexistent_session(repo):
    """Test closing a nonexistent session."""
    rows_affected = repo.close_session(99999)
    assert rows_affected == 0


def test_exists(repo):
    """Test checking if session exists."""
    # Check nonexistent session
    assert repo.exists(99999) is False
    
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    # Check existing session
    assert repo.exists(session_id) is True
    
    # Check different nonexistent session
    assert repo.exists(session_id + 1000) is False


def test_persistence(repo, temp_db):
    """Test that changes persist after reopening connection."""
    # Create session
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        course_code="CS101"
    )
    
    # Create new repository instance (simulates reopening connection)
    new_repo = SessionRepository(temp_db)
    
    # Verify session exists and data is correct
    session = new_repo.get_by_id(session_id)
    assert session is not None
    assert session['session_name'] == "CS101 Lecture"
    assert session['course_code'] == "CS101"
    
    # Update through new repository
    new_repo.update_session(session_id, session_name="CS101 Lab")
    
    # Verify through original repository
    session = repo.get_by_id(session_id)
    assert session['session_name'] == "CS101 Lab"


def test_default_status_is_scheduled(repo):
    """Test that default session status is 'scheduled'."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = repo.get_by_id(session_id)
    assert session['status'] == 'scheduled'


def test_custom_status(repo):
    """Test creating session with custom status."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        status="active"
    )
    
    session = repo.get_by_id(session_id)
    assert session['status'] == 'active'


def test_late_threshold(repo):
    """Test late threshold field."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        late_threshold_minutes=20
    )
    
    session = repo.get_by_id(session_id)
    assert session['late_threshold_minutes'] == 20


def test_created_by_field(repo):
    """Test created_by field."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        created_by="admin"
    )
    
    session = repo.get_by_id(session_id)
    assert session['created_by'] == "admin"


def test_notes_field(repo):
    """Test notes field."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00",
        notes="Important lecture on data structures"
    )
    
    session = repo.get_by_id(session_id)
    assert session['notes'] == "Important lecture on data structures"


def test_created_at_timestamp(repo):
    """Test that created_at is set automatically."""
    session_id = repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    
    session = repo.get_by_id(session_id)
    assert session['created_at'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
