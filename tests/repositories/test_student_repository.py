"""
Student Repository Tests

Tests database persistence operations for the students table.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.data.repositories.student_repository import StudentRepository
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
    """Create a StudentRepository instance with test database."""
    return StudentRepository(temp_db)


def test_create_student(repo):
    """Test 1 — Create a student and verify insertion succeeds."""
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="Computer Science",
        notes="Test student"
    )
    
    assert student_db_id is not None
    assert isinstance(student_db_id, int)
    assert student_db_id > 0


def test_get_by_database_id(repo):
    """Test 2 — Create a student, retrieve by database ID, verify fields."""
    # Create student
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="Computer Science"
    )
    
    # Retrieve by database ID
    student = repo.get_by_id(student_db_id)
    
    assert student is not None
    assert student['id'] == student_db_id
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"
    assert student['email'] == "john@example.com"
    assert student['department'] == "Computer Science"
    assert student['is_active'] == 1
    assert student['enrolled_at'] is not None


def test_get_by_student_id(repo):
    """Test 3 — Create a student, retrieve using student_id, verify result."""
    # Create student
    repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    # Retrieve by student_id
    student = repo.get_by_student_id("2021001")
    
    assert student is not None
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"
    assert student['email'] == "john@example.com"


def test_get_all_students(repo):
    """Test 4 — Create multiple students and verify they are returned."""
    # Create multiple students
    repo.create_student(student_id="2021001", name="John Doe")
    repo.create_student(student_id="2021002", name="Jane Smith")
    repo.create_student(student_id="2021003", name="Bob Johnson")
    
    # Get all students
    students = repo.get_all()
    
    assert len(students) == 3
    assert students[0]['student_id'] == "2021001"
    assert students[1]['student_id'] == "2021002"
    assert students[2]['student_id'] == "2021003"


def test_get_active_only(repo):
    """Test 5 — Create active and inactive students, verify active_only filter."""
    # Create active student
    active_id = repo.create_student(student_id="2021001", name="Active Student")
    
    # Create inactive student
    inactive_id = repo.create_student(student_id="2021002", name="Inactive Student")
    repo.deactivate_student(inactive_id)
    
    # Get all students
    all_students = repo.get_all(active_only=False)
    assert len(all_students) == 2
    
    # Get active only
    active_students = repo.get_all(active_only=True)
    assert len(active_students) == 1
    assert active_students[0]['student_id'] == "2021001"
    assert active_students[0]['is_active'] == 1


def test_update_student(repo):
    """Test 6 — Update student information and verify changes persist."""
    # Create student
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="Computer Science"
    )
    
    # Update student
    rows_affected = repo.update_student(
        student_db_id,
        name="Jane Doe",
        email="jane@example.com",
        department="Mathematics",
        notes="Updated student"
    )
    
    assert rows_affected == 1
    
    # Verify changes persisted
    student = repo.get_by_id(student_db_id)
    assert student['name'] == "Jane Doe"
    assert student['email'] == "jane@example.com"
    assert student['department'] == "Mathematics"
    assert student['notes'] == "Updated student"
    assert student['student_id'] == "2021001"  # Unchanged


def test_update_student_partial(repo):
    """Test updating only some fields."""
    # Create student
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    # Update only name
    rows_affected = repo.update_student(student_db_id, name="Jane Doe")
    
    assert rows_affected == 1
    
    # Verify only name changed
    student = repo.get_by_id(student_db_id)
    assert student['name'] == "Jane Doe"
    assert student['email'] == "john@example.com"  # Unchanged


def test_deactivate_student(repo):
    """Test 7 — Deactivate a student and verify is_active = 0."""
    # Create student
    student_db_id = repo.create_student(student_id="2021001", name="John Doe")
    
    # Verify initially active
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 1
    
    # Deactivate
    rows_affected = repo.deactivate_student(student_db_id)
    assert rows_affected == 1
    
    # Verify deactivated
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 0


def test_activate_student(repo):
    """Test 8 — Activate a student and verify is_active = 1."""
    # Create and deactivate student
    student_db_id = repo.create_student(student_id="2021001", name="John Doe")
    repo.deactivate_student(student_db_id)
    
    # Verify deactivated
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 0
    
    # Activate
    rows_affected = repo.activate_student(student_db_id)
    assert rows_affected == 1
    
    # Verify activated
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 1


def test_exists_by_student_id(repo):
    """Test 9 — Verify existence check works correctly."""
    # Check nonexistent student
    assert repo.exists_by_student_id("2021999") is False
    
    # Create student
    repo.create_student(student_id="2021001", name="John Doe")
    
    # Check existing student
    assert repo.exists_by_student_id("2021001") is True
    
    # Check different nonexistent student
    assert repo.exists_by_student_id("2021002") is False


def test_duplicate_student_id(repo):
    """Test 10 — Attempt to create two students with same student_id."""
    # Create first student
    repo.create_student(student_id="2021001", name="John Doe")
    
    # Attempt to create duplicate
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_student(student_id="2021001", name="Jane Smith")


def test_nonexistent_student_get(repo):
    """Test 11 — Test behavior when querying nonexistent student."""
    # Get by database ID
    student = repo.get_by_id(99999)
    assert student is None
    
    # Get by student_id
    student = repo.get_by_student_id("2021999")
    assert student is None


def test_nonexistent_student_update(repo):
    """Test updating a nonexistent student."""
    rows_affected = repo.update_student(99999, name="New Name")
    assert rows_affected == 0


def test_nonexistent_student_deactivate(repo):
    """Test deactivating a nonexistent student."""
    rows_affected = repo.deactivate_student(99999)
    assert rows_affected == 0


def test_nonexistent_student_activate(repo):
    """Test activating a nonexistent student."""
    rows_affected = repo.activate_student(99999)
    assert rows_affected == 0


def test_persistence(repo, temp_db):
    """Test 12 — Verify changes persist after reopening connection."""
    # Create student
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    # Create new repository instance (simulates reopening connection)
    new_repo = StudentRepository(temp_db)
    
    # Verify student exists and data is correct
    student = new_repo.get_by_id(student_db_id)
    assert student is not None
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"
    assert student['email'] == "john@example.com"
    
    # Update through new repository
    new_repo.update_student(student_db_id, name="Jane Doe")
    
    # Verify through original repository
    student = repo.get_by_id(student_db_id)
    assert student['name'] == "Jane Doe"


def test_create_with_minimal_fields(repo):
    """Test creating student with only required fields."""
    student_db_id = repo.create_student(
        student_id="2021001",
        name="John Doe"
    )
    
    student = repo.get_by_id(student_db_id)
    assert student is not None
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"
    assert student['email'] is None
    assert student['department'] is None
    assert student['notes'] is None


def test_update_with_no_fields(repo):
    """Test update with no fields provided."""
    student_db_id = repo.create_student(student_id="2021001", name="John Doe")
    
    # Update with no fields
    rows_affected = repo.update_student(student_db_id)
    assert rows_affected == 0


def test_update_is_active_directly(repo):
    """Test updating is_active field directly through update_student."""
    student_db_id = repo.create_student(student_id="2021001", name="John Doe")
    
    # Deactivate using update_student
    rows_affected = repo.update_student(student_db_id, is_active=False)
    assert rows_affected == 1
    
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 0
    
    # Activate using update_student
    rows_affected = repo.update_student(student_db_id, is_active=True)
    assert rows_affected == 1
    
    student = repo.get_by_id(student_db_id)
    assert student['is_active'] == 1


def test_get_all_ordering(repo):
    """Test that get_all returns students ordered by student_id."""
    # Create students in non-alphabetical order
    repo.create_student(student_id="2021003", name="Charlie")
    repo.create_student(student_id="2021001", name="Alice")
    repo.create_student(student_id="2021002", name="Bob")
    
    students = repo.get_all()
    
    assert len(students) == 3
    assert students[0]['student_id'] == "2021001"
    assert students[1]['student_id'] == "2021002"
    assert students[2]['student_id'] == "2021003"


def test_update_student_id(repo):
    """Test updating the student_id field."""
    student_db_id = repo.create_student(student_id="2021001", name="John Doe")
    
    # Update student_id
    rows_affected = repo.update_student(student_db_id, student_id="2021999")
    assert rows_affected == 1
    
    # Verify old student_id doesn't exist
    student = repo.get_by_student_id("2021001")
    assert student is None
    
    # Verify new student_id exists
    student = repo.get_by_student_id("2021999")
    assert student is not None
    assert student['id'] == student_db_id
    assert student['name'] == "John Doe"


def test_update_to_duplicate_student_id(repo):
    """Test that updating to an existing student_id fails."""
    # Create two students
    id1 = repo.create_student(student_id="2021001", name="John Doe")
    id2 = repo.create_student(student_id="2021002", name="Jane Smith")
    
    # Try to update second student to have same student_id as first
    with pytest.raises(sqlite3.IntegrityError):
        repo.update_student(id2, student_id="2021001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
