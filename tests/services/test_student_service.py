"""
Student Service Tests

Tests business logic, validation, and coordination for student management.
Uses isolated test databases to avoid modifying the development database.
"""

import pytest
import tempfile
from pathlib import Path

from src.services.student_service import (
    StudentService,
    InvalidStudentDataError,
    DuplicateStudentError,
    StudentNotFoundError
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
    """Create a StudentService instance with test database."""
    return StudentService(temp_db)


# ============================================================================
# Test 1-2: Valid Creation
# ============================================================================

def test_valid_student_creation(service):
    """Test 1 — Valid student creation."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="Computer Science",
        notes="Test student"
    )
    
    assert student_db_id is not None
    assert isinstance(student_db_id, int)
    assert student_db_id > 0
    
    # Verify student was created
    student = service.get_student_by_id(student_db_id)
    assert student is not None
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"


def test_creation_with_optional_fields(service):
    """Test 2 — Creation with optional fields."""
    # With all optionals
    id1 = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="CS",
        notes="Note"
    )
    assert id1 > 0
    
    # With only required fields
    id2 = service.create_student(
        student_id="2021002",
        name="Jane Doe"
    )
    assert id2 > 0
    
    student = service.get_student_by_id(id2)
    assert student['email'] is None
    assert student['department'] is None
    assert student['notes'] is None


# ============================================================================
# Test 3-7: Validation - Missing/Empty/Invalid Fields
# ============================================================================

def test_missing_student_id_rejected(service):
    """Test 3 — Missing student_id rejected."""
    with pytest.raises(TypeError):
        service.create_student(name="John Doe")


def test_empty_student_id_rejected(service):
    """Test 4 — Empty student_id rejected."""
    with pytest.raises(InvalidStudentDataError, match="student_id cannot be empty"):
        service.create_student(student_id="", name="John Doe")
    
    with pytest.raises(InvalidStudentDataError, match="student_id cannot be empty"):
        service.create_student(student_id="   ", name="John Doe")


def test_missing_name_rejected(service):
    """Test 5 — Missing name rejected."""
    with pytest.raises(TypeError):
        service.create_student(student_id="2021001")


def test_empty_name_rejected(service):
    """Test 6 — Empty name rejected."""
    with pytest.raises(InvalidStudentDataError, match="name cannot be empty"):
        service.create_student(student_id="2021001", name="")
    
    with pytest.raises(InvalidStudentDataError, match="name cannot be empty"):
        service.create_student(student_id="2021001", name="   ")


def test_invalid_email_rejected(service):
    """Test 7 — Invalid email rejected."""
    invalid_emails = [
        "not-an-email",
        "missing-at-sign.com",
        "@missing-local.com",
        "missing-domain@",
        "spaces in@email.com",
        "double@@at.com",
    ]
    
    for invalid_email in invalid_emails:
        with pytest.raises(InvalidStudentDataError, match="email format is invalid"):
            service.create_student(
                student_id="2021001",
                name="John Doe",
                email=invalid_email
            )


def test_valid_email_accepted(service):
    """Test 8 — Valid email accepted."""
    valid_emails = [
        "john@example.com",
        "jane.doe@university.edu",
        "student123@school.org",
        "first+last@domain.co.uk",
    ]
    
    for idx, valid_email in enumerate(valid_emails):
        student_id = f"202100{idx+1}"
        db_id = service.create_student(
            student_id=student_id,
            name="Test Student",
            email=valid_email
        )
        
        student = service.get_student_by_id(db_id)
        assert student['email'] == valid_email


# ============================================================================
# Test 9: Whitespace Normalization
# ============================================================================

def test_whitespace_normalization(service):
    """Test 9 — Surrounding whitespace normalized."""
    student_db_id = service.create_student(
        student_id="  2021001  ",
        name="  John Doe  ",
        email="  john@example.com  ",
        department="  Computer Science  ",
        notes="  Test note  "
    )
    
    student = service.get_student_by_id(student_db_id)
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"
    assert student['email'] == "john@example.com"
    assert student['department'] == "Computer Science"
    assert student['notes'] == "Test note"


# ============================================================================
# Test 10: Duplicate Rejection
# ============================================================================

def test_duplicate_student_id_rejected(service):
    """Test 10 — Duplicate student_id rejected at service level."""
    # Create first student
    service.create_student(student_id="2021001", name="John Doe")
    
    # Attempt to create duplicate
    with pytest.raises(DuplicateStudentError, match="already exists"):
        service.create_student(student_id="2021001", name="Jane Doe")


# ============================================================================
# Test 11-14: Read Operations
# ============================================================================

def test_get_student_by_database_id(service):
    """Test 11 — Get student by database ID."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    student = service.get_student_by_id(student_db_id)
    assert student is not None
    assert student['id'] == student_db_id
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"


def test_get_student_by_student_id(service):
    """Test 12 — Get student by student_id."""
    service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    student = service.get_student_by_student_id("2021001")
    assert student is not None
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"


def test_get_all_students(service):
    """Test 13 — Get all students."""
    service.create_student(student_id="2021001", name="John Doe")
    service.create_student(student_id="2021002", name="Jane Smith")
    service.create_student(student_id="2021003", name="Bob Johnson")
    
    students = service.get_all_students()
    assert len(students) == 3


def test_get_active_students_only(service):
    """Test 14 — Get active students only."""
    # Create active student
    active_id = service.create_student(student_id="2021001", name="Active Student")
    
    # Create and deactivate another student
    inactive_id = service.create_student(student_id="2021002", name="Inactive Student")
    service.deactivate_student(inactive_id)
    
    # Get all
    all_students = service.get_all_students(active_only=False)
    assert len(all_students) == 2
    
    # Get active only
    active_students = service.get_all_students(active_only=True)
    assert len(active_students) == 1
    assert active_students[0]['student_id'] == "2021001"


# ============================================================================
# Test 15-16: Existence and Not Found
# ============================================================================

def test_student_existence_check(service):
    """Test 15 — Student existence check."""
    # Non-existent
    assert service.student_exists("2021999") is False
    
    # Create student
    service.create_student(student_id="2021001", name="John Doe")
    
    # Now exists
    assert service.student_exists("2021001") is True
    
    # Different ID still doesn't exist
    assert service.student_exists("2021002") is False


def test_nonexistent_student_retrieval(service):
    """Test 16 — Nonexistent student retrieval."""
    # Get by database ID
    student = service.get_student_by_id(99999)
    assert student is None
    
    # Get by student_id
    student = service.get_student_by_student_id("2021999")
    assert student is None


# ============================================================================
# Test 17-19: Update Operations
# ============================================================================

def test_successful_student_update(service):
    """Test 17 — Successful student update."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="Computer Science"
    )
    
    rows_affected = service.update_student(
        student_db_id,
        name="Jane Doe",
        email="jane@example.com",
        department="Mathematics"
    )
    
    assert rows_affected == 1
    
    student = service.get_student_by_id(student_db_id)
    assert student['name'] == "Jane Doe"
    assert student['email'] == "jane@example.com"
    assert student['department'] == "Mathematics"
    assert student['student_id'] == "2021001"  # Unchanged


def test_partial_student_update(service):
    """Test 18 — Partial student update."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com"
    )
    
    # Update only name
    service.update_student(student_db_id, name="Jane Doe")
    
    student = service.get_student_by_id(student_db_id)
    assert student['name'] == "Jane Doe"
    assert student['email'] == "john@example.com"  # Unchanged


def test_invalid_update_values_rejected(service):
    """Test 19 — Invalid updated values rejected."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe"
    )
    
    # Empty name
    with pytest.raises(InvalidStudentDataError, match="name cannot be empty"):
        service.update_student(student_db_id, name="")
    
    # Invalid email
    with pytest.raises(InvalidStudentDataError, match="email format is invalid"):
        service.update_student(student_db_id, email="invalid-email")


# ============================================================================
# Test 20: Prevent Duplicate Update
# ============================================================================

def test_prevent_duplicate_student_id_update(service):
    """Test 20 — Prevent changing student_id to another existing student's ID."""
    id1 = service.create_student(student_id="2021001", name="John Doe")
    id2 = service.create_student(student_id="2021002", name="Jane Smith")
    
    # Try to update second student to have same student_id as first
    with pytest.raises(DuplicateStudentError, match="already exists"):
        service.update_student(id2, student_id="2021001")


# ============================================================================
# Test 21: Update Nonexistent
# ============================================================================

def test_update_nonexistent_student(service):
    """Test 21 — Updating a nonexistent student handled correctly."""
    with pytest.raises(StudentNotFoundError, match="not found"):
        service.update_student(99999, name="New Name")


# ============================================================================
# Test 22-23: Activation/Deactivation
# ============================================================================

def test_deactivate_existing_student(service):
    """Test 22 — Deactivate existing student."""
    student_db_id = service.create_student(student_id="2021001", name="John Doe")
    
    # Verify initially active
    student = service.get_student_by_id(student_db_id)
    assert student['is_active'] == 1
    
    # Deactivate
    rows_affected = service.deactivate_student(student_db_id)
    assert rows_affected == 1
    
    # Verify deactivated
    student = service.get_student_by_id(student_db_id)
    assert student['is_active'] == 0


def test_activate_existing_student(service):
    """Test 23 — Activate existing student."""
    student_db_id = service.create_student(student_id="2021001", name="John Doe")
    service.deactivate_student(student_db_id)
    
    # Verify deactivated
    student = service.get_student_by_id(student_db_id)
    assert student['is_active'] == 0
    
    # Activate
    rows_affected = service.activate_student(student_db_id)
    assert rows_affected == 1
    
    # Verify activated
    student = service.get_student_by_id(student_db_id)
    assert student['is_active'] == 1


# ============================================================================
# Test 24: Nonexistent Activation/Deactivation
# ============================================================================

def test_nonexistent_student_activation_deactivation(service):
    """Test 24 — Nonexistent student activation/deactivation handled correctly."""
    with pytest.raises(StudentNotFoundError, match="not found"):
        service.activate_student(99999)
    
    with pytest.raises(StudentNotFoundError, match="not found"):
        service.deactivate_student(99999)


# ============================================================================
# Test 25-26: Architecture Verification
# ============================================================================

def test_service_uses_repository(service):
    """Test 25 — Verify StudentService uses StudentRepository."""
    from src.data.repositories.student_repository import StudentRepository
    assert hasattr(service, 'repository')
    assert isinstance(service.repository, StudentRepository)


def test_service_contains_no_sql(service):
    """Test 26 — Verify StudentService contains no direct SQL/database logic."""
    import inspect
    
    # Get service source code
    source = inspect.getsource(StudentService)
    
    # Should not contain SQL keywords that indicate direct database access
    sql_indicators = [
        'SELECT ',
        'INSERT INTO',
        'UPDATE ',
        'DELETE FROM',
        'CREATE TABLE',
        'sqlite3.connect',
        'cursor.execute',
    ]
    
    for indicator in sql_indicators:
        assert indicator not in source, f"Service contains SQL logic: {indicator}"


# ============================================================================
# Additional Edge Cases
# ============================================================================

def test_invalid_type_validation(service):
    """Test validation rejects invalid types."""
    # Non-string student_id
    with pytest.raises(InvalidStudentDataError, match="student_id must be a string"):
        service.create_student(student_id=123, name="John Doe")
    
    # Non-string name
    with pytest.raises(InvalidStudentDataError, match="name must be a string"):
        service.create_student(student_id="2021001", name=123)
    
    # Non-string email
    with pytest.raises(InvalidStudentDataError, match="email must be a string"):
        service.create_student(student_id="2021001", name="John Doe", email=123)


def test_empty_string_converts_to_none(service):
    """Test that empty strings for optional fields convert to None."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="",
        department="",
        notes=""
    )
    
    student = service.get_student_by_id(student_db_id)
    assert student['email'] is None
    assert student['department'] is None
    assert student['notes'] is None


def test_get_by_student_id_whitespace_handling(service):
    """Test get_by_student_id handles whitespace."""
    service.create_student(student_id="2021001", name="John Doe")
    
    # Should find with extra whitespace
    student = service.get_student_by_student_id("  2021001  ")
    assert student is not None
    assert student['student_id'] == "2021001"


def test_student_exists_whitespace_handling(service):
    """Test student_exists handles whitespace."""
    service.create_student(student_id="2021001", name="John Doe")
    
    # Should return True with extra whitespace
    assert service.student_exists("  2021001  ") is True


def test_invalid_db_id_validation(service):
    """Test validation of database ID parameters."""
    # Invalid type
    with pytest.raises(InvalidStudentDataError, match="must be a positive integer"):
        service.get_student_by_id("not-an-int")
    
    # Negative number
    with pytest.raises(InvalidStudentDataError, match="must be a positive integer"):
        service.get_student_by_id(-1)
    
    # Zero
    with pytest.raises(InvalidStudentDataError, match="must be a positive integer"):
        service.get_student_by_id(0)


def test_update_student_id_to_same_value(service):
    """Test updating student_id to the same value is allowed."""
    student_db_id = service.create_student(student_id="2021001", name="John Doe")
    
    # Update with same student_id should succeed
    rows_affected = service.update_student(
        student_db_id,
        student_id="2021001",
        name="Jane Doe"
    )
    
    assert rows_affected == 1
    student = service.get_student_by_id(student_db_id)
    assert student['student_id'] == "2021001"
    assert student['name'] == "Jane Doe"


def test_update_with_empty_string_fields(service):
    """Test updating with empty strings normalizes to None (no update)."""
    student_db_id = service.create_student(
        student_id="2021001",
        name="John Doe",
        email="john@example.com",
        department="CS"
    )
    
    # Update with empty strings - normalizes to None, which means "no update"
    # This is correct behavior: empty strings are treated as "not provided"
    service.update_student(
        student_db_id,
        email="",
        department=""
    )
    
    student = service.get_student_by_id(student_db_id)
    # Fields should remain unchanged because empty strings become None (not provided)
    assert student['email'] == "john@example.com"
    assert student['department'] == "CS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
