"""
Attendance Repository Tests

Tests database persistence operations for the attendance_records table.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.data.repositories.attendance_repository import AttendanceRepository
from src.data.repositories.student_repository import StudentRepository
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
    """Create an AttendanceRepository instance with test database."""
    return AttendanceRepository(temp_db)


@pytest.fixture
def student_repo(temp_db):
    """Create a StudentRepository instance with test database."""
    return StudentRepository(temp_db)


@pytest.fixture
def session_repo(temp_db):
    """Create a SessionRepository instance with test database."""
    return SessionRepository(temp_db)


@pytest.fixture
def sample_student(student_repo):
    """Create a sample student for testing."""
    student_id = student_repo.create_student(
        student_id="2021001",
        name="John Doe"
    )
    return student_id


@pytest.fixture
def sample_session(session_repo):
    """Create a sample session for testing."""
    session_id = session_repo.create_session(
        session_name="CS101 Lecture",
        start_time="2026-10-04 10:00:00"
    )
    return session_id


def test_create_attendance(repo, sample_session, sample_student):
    """Test creating an attendance record and verify insertion succeeds."""
    attendance_id = repo.create_attendance(
        session_id=sample_session,
        student_id=sample_student,
        status="present",
        detection_confidence=0.95,
        liveness_score=0.88,
        marked_by="system",
        notes="Test attendance"
    )
    
    assert attendance_id is not None
    assert isinstance(attendance_id, int)
    assert attendance_id > 0


def test_create_attendance_minimal_fields(repo, sample_session, sample_student):
    """Test creating attendance record with only required fields."""
    attendance_id = repo.create_attendance(
        session_id=sample_session,
        student_id=sample_student,
        status="present"
    )
    
    attendance = repo.get_by_id(attendance_id)
    assert attendance is not None
    assert attendance['session_id'] == sample_session
    assert attendance['student_id'] == sample_student
    assert attendance['status'] == "present"
    assert attendance['marked_by'] == "system"
    assert attendance['detection_confidence'] is None
    assert attendance['liveness_score'] is None


def test_get_by_id(repo, sample_session, sample_student):
    """Test retrieving an attendance record by ID."""
    # Create attendance
    attendance_id = repo.create_attendance(
        session_id=sample_session,
        student_id=sample_student,
        status="present",
        detection_confidence=0.95,
        liveness_score=0.88
    )
    
    # Retrieve by ID
    attendance = repo.get_by_id(attendance_id)
    
    assert attendance is not None
    assert attendance['id'] == attendance_id
    assert attendance['session_id'] == sample_session
    assert attendance['student_id'] == sample_student
    assert attendance['status'] == "present"
    assert attendance['detection_confidence'] == 0.95
    assert attendance['liveness_score'] == 0.88
    assert attendance['marked_at'] is not None


def test_get_by_id_nonexistent(repo):
    """Test retrieving nonexistent attendance record returns None."""
    attendance = repo.get_by_id(99999)
    assert attendance is None


def test_get_by_session(repo, sample_session, student_repo):
    """Test retrieving all attendance records for a session."""
    # Create multiple students
    student1 = student_repo.create_student(student_id="2021001", name="Student 1")
    student2 = student_repo.create_student(student_id="2021002", name="Student 2")
    student3 = student_repo.create_student(student_id="2021003", name="Student 3")
    
    # Create attendance records
    repo.create_attendance(sample_session, student1, "present")
    repo.create_attendance(sample_session, student2, "late")
    repo.create_attendance(sample_session, student3, "present")
    
    # Get all attendance for session
    records = repo.get_by_session(sample_session)
    
    assert len(records) == 3
    assert all(r['session_id'] == sample_session for r in records)


def test_get_by_student(repo, session_repo, sample_student):
    """Test retrieving all attendance records for a student."""
    # Create multiple sessions
    session1 = session_repo.create_session("Session 1", "2026-10-04 10:00:00")
    session2 = session_repo.create_session("Session 2", "2026-10-05 10:00:00")
    session3 = session_repo.create_session("Session 3", "2026-10-06 10:00:00")
    
    # Create attendance records
    repo.create_attendance(session1, sample_student, "present")
    repo.create_attendance(session2, sample_student, "late")
    repo.create_attendance(session3, sample_student, "present")
    
    # Get all attendance for student
    records = repo.get_by_student(sample_student)
    
    assert len(records) == 3
    assert all(r['student_id'] == sample_student for r in records)
    # Verify all three sessions are present
    session_ids = {r['session_id'] for r in records}
    assert session_ids == {session1, session2, session3}


def test_get_by_session_and_student(repo, sample_session, sample_student):
    """Test retrieving specific attendance record for session and student."""
    # Create attendance
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        "present",
        detection_confidence=0.95
    )
    
    # Retrieve by session and student
    attendance = repo.get_by_session_and_student(sample_session, sample_student)
    
    assert attendance is not None
    assert attendance['id'] == attendance_id
    assert attendance['session_id'] == sample_session
    assert attendance['student_id'] == sample_student


def test_get_by_session_and_student_nonexistent(repo, sample_session, sample_student):
    """Test retrieving nonexistent attendance returns None."""
    attendance = repo.get_by_session_and_student(sample_session, sample_student)
    assert attendance is None


def test_exists(repo, sample_session, sample_student, student_repo):
    """Test checking if attendance record exists."""
    # Check nonexistent attendance
    assert repo.exists(sample_session, sample_student) is False
    
    # Create attendance
    repo.create_attendance(sample_session, sample_student, "present")
    
    # Check existing attendance
    assert repo.exists(sample_session, sample_student) is True
    
    # Check different student
    other_student = student_repo.create_student(student_id="2021002", name="Other Student")
    assert repo.exists(sample_session, other_student) is False


def test_duplicate_attendance_prevented(repo, sample_session, sample_student):
    """Test that UNIQUE constraint prevents duplicate attendance records."""
    # Create first attendance record
    repo.create_attendance(sample_session, sample_student, "present")
    
    # Try to create duplicate attendance
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_attendance(sample_session, sample_student, "present")


def test_foreign_key_session_enforced(repo, sample_student):
    """Test that foreign key constraint on session_id is enforced."""
    # Try to create attendance with invalid session_id
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_attendance(
            session_id=99999,
            student_id=sample_student,
            status="present"
        )


def test_foreign_key_student_enforced(repo, sample_session):
    """Test that foreign key constraint on student_id is enforced."""
    # Try to create attendance with invalid student_id
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_attendance(
            session_id=sample_session,
            student_id=99999,
            status="present"
        )


def test_cascade_delete_session(repo, session_repo, sample_student):
    """Test that deleting a session cascades to attendance records."""
    # Create session
    session_id = session_repo.create_session("Test Session", "2026-10-04 10:00:00")
    
    # Create attendance
    attendance_id = repo.create_attendance(session_id, sample_student, "present")
    
    # Verify attendance exists
    assert repo.get_by_id(attendance_id) is not None
    
    # Delete session (using raw SQL for test)
    from src.database.connection import get_db_connection
    with get_db_connection(repo.database_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    
    # Verify attendance was deleted (CASCADE)
    assert repo.get_by_id(attendance_id) is None


def test_cascade_delete_student(repo, student_repo, sample_session):
    """Test that deleting a student cascades to attendance records."""
    # Create student
    student_id = student_repo.create_student(student_id="2021999", name="Test Student")
    
    # Create attendance
    attendance_id = repo.create_attendance(sample_session, student_id, "present")
    
    # Verify attendance exists
    assert repo.get_by_id(attendance_id) is not None
    
    # Delete student (using raw SQL for test)
    from src.database.connection import get_db_connection
    with get_db_connection(repo.database_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    
    # Verify attendance was deleted (CASCADE)
    assert repo.get_by_id(attendance_id) is None


def test_update_attendance(repo, sample_session, sample_student):
    """Test updating attendance record information."""
    # Create attendance
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        status="present",
        detection_confidence=0.85
    )
    
    # Update attendance
    rows_affected = repo.update_attendance(
        attendance_id,
        status="late",
        detection_confidence=0.92,
        liveness_score=0.88,
        notes="Arrived 10 minutes late"
    )
    
    assert rows_affected == 1
    
    # Verify changes persisted
    attendance = repo.get_by_id(attendance_id)
    assert attendance['status'] == "late"
    assert attendance['detection_confidence'] == 0.92
    assert attendance['liveness_score'] == 0.88
    assert attendance['notes'] == "Arrived 10 minutes late"


def test_update_attendance_partial(repo, sample_session, sample_student):
    """Test updating only some attendance fields."""
    # Create attendance
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        status="present",
        detection_confidence=0.85
    )
    
    # Update only status
    rows_affected = repo.update_attendance(attendance_id, status="late")
    
    assert rows_affected == 1
    
    # Verify only status changed
    attendance = repo.get_by_id(attendance_id)
    assert attendance['status'] == "late"
    assert attendance['detection_confidence'] == 0.85  # Unchanged


def test_update_with_verification(repo, sample_session, sample_student):
    """Test updating attendance with verification information."""
    # Create attendance
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        status="present"
    )
    
    # Update with verification
    rows_affected = repo.update_attendance(
        attendance_id,
        verified_by="admin"
    )
    
    assert rows_affected == 1
    
    # Verify verification fields set
    attendance = repo.get_by_id(attendance_id)
    assert attendance['verified_by'] == "admin"
    assert attendance['verified_at'] is not None


def test_update_nonexistent_attendance(repo):
    """Test updating a nonexistent attendance record."""
    rows_affected = repo.update_attendance(99999, status="present")
    assert rows_affected == 0


def test_update_with_no_fields(repo, sample_session, sample_student):
    """Test update with no fields provided."""
    attendance_id = repo.create_attendance(sample_session, sample_student, "present")
    
    # Update with no fields
    rows_affected = repo.update_attendance(attendance_id)
    assert rows_affected == 0


def test_get_all(repo, sample_session, student_repo):
    """Test retrieving all attendance records."""
    # Create multiple students and attendance records
    student1 = student_repo.create_student(student_id="2021001", name="Student 1")
    student2 = student_repo.create_student(student_id="2021002", name="Student 2")
    
    repo.create_attendance(sample_session, student1, "present")
    repo.create_attendance(sample_session, student2, "late")
    
    # Get all attendance
    records = repo.get_all()
    
    assert len(records) >= 2


def test_get_by_status(repo, sample_session, student_repo):
    """Test retrieving attendance records filtered by status."""
    # Create students
    student1 = student_repo.create_student(student_id="2021001", name="Student 1")
    student2 = student_repo.create_student(student_id="2021002", name="Student 2")
    student3 = student_repo.create_student(student_id="2021003", name="Student 3")
    
    # Create attendance with different statuses
    repo.create_attendance(sample_session, student1, "present")
    repo.create_attendance(sample_session, student2, "late")
    repo.create_attendance(sample_session, student3, "present")
    
    # Get present records
    present_records = repo.get_by_status("present")
    assert len(present_records) == 2
    assert all(r['status'] == "present" for r in present_records)
    
    # Get late records
    late_records = repo.get_by_status("late")
    assert len(late_records) == 1
    assert late_records[0]['status'] == "late"
    
    # Get absent records (none)
    absent_records = repo.get_by_status("absent")
    assert len(absent_records) == 0


def test_persistence(repo, temp_db, sample_session, sample_student):
    """Test that changes persist after reopening connection."""
    # Create attendance
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        status="present",
        detection_confidence=0.95
    )
    
    # Create new repository instance (simulates reopening connection)
    new_repo = AttendanceRepository(temp_db)
    
    # Verify attendance exists and data is correct
    attendance = new_repo.get_by_id(attendance_id)
    assert attendance is not None
    assert attendance['status'] == "present"
    assert attendance['detection_confidence'] == 0.95
    
    # Update through new repository
    new_repo.update_attendance(attendance_id, status="late")
    
    # Verify through original repository
    attendance = repo.get_by_id(attendance_id)
    assert attendance['status'] == "late"


def test_marked_at_timestamp(repo, sample_session, sample_student):
    """Test that marked_at is set automatically."""
    attendance_id = repo.create_attendance(sample_session, sample_student, "present")
    
    attendance = repo.get_by_id(attendance_id)
    assert attendance['marked_at'] is not None


def test_default_marked_by_is_system(repo, sample_session, sample_student):
    """Test that default marked_by value is 'system'."""
    attendance_id = repo.create_attendance(sample_session, sample_student, "present")
    
    attendance = repo.get_by_id(attendance_id)
    assert attendance['marked_by'] == "system"


def test_custom_marked_by(repo, sample_session, sample_student):
    """Test creating attendance with custom marked_by value."""
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        "present",
        marked_by="manual"
    )
    
    attendance = repo.get_by_id(attendance_id)
    assert attendance['marked_by'] == "manual"


def test_multiple_sessions_same_student(repo, session_repo, sample_student):
    """Test that same student can have attendance in multiple sessions."""
    # Create multiple sessions
    session1 = session_repo.create_session("Session 1", "2026-10-04 10:00:00")
    session2 = session_repo.create_session("Session 2", "2026-10-05 10:00:00")
    session3 = session_repo.create_session("Session 3", "2026-10-06 10:00:00")
    
    # Create attendance for same student in different sessions
    repo.create_attendance(session1, sample_student, "present")
    repo.create_attendance(session2, sample_student, "late")
    repo.create_attendance(session3, sample_student, "present")
    
    # Verify all records exist
    records = repo.get_by_student(sample_student)
    assert len(records) == 3


def test_multiple_students_same_session(repo, sample_session, student_repo):
    """Test that multiple students can have attendance in same session."""
    # Create multiple students
    student1 = student_repo.create_student(student_id="2021001", name="Student 1")
    student2 = student_repo.create_student(student_id="2021002", name="Student 2")
    student3 = student_repo.create_student(student_id="2021003", name="Student 3")
    
    # Create attendance for different students in same session
    repo.create_attendance(sample_session, student1, "present")
    repo.create_attendance(sample_session, student2, "late")
    repo.create_attendance(sample_session, student3, "present")
    
    # Verify all records exist
    records = repo.get_by_session(sample_session)
    assert len(records) == 3


def test_detection_confidence_and_liveness_score(repo, sample_session, sample_student):
    """Test storing detection confidence and liveness score."""
    attendance_id = repo.create_attendance(
        sample_session,
        sample_student,
        "present",
        detection_confidence=0.95,
        liveness_score=0.88
    )
    
    attendance = repo.get_by_id(attendance_id)
    assert attendance['detection_confidence'] == 0.95
    assert attendance['liveness_score'] == 0.88


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
