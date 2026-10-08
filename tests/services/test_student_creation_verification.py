"""
Tests for Student Creation and Database Verification

Verifies that creating a student:
1. Validates all inputs
2. Inserts into SQLite database
3. Commits transaction
4. Verifies student exists in database immediately
5. Handles duplicates cleanly
6. Handles error propagation clearly
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from src.services.student_service import (
    StudentService,
    InvalidStudentDataError,
    DuplicateStudentError,
    StudentNotFoundError,
    StudentServiceError
)
from src.data.repositories.student_repository import StudentRepository
from src.database.schema import initialize_database


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def student_repo(temp_db):
    return StudentRepository(temp_db)


@pytest.fixture
def student_service(temp_db):
    return StudentService(temp_db)


def test_create_student_database_insert_commit_verify(student_service, temp_db):
    """Test that create_student writes, commits, and verifies the record in SQLite."""
    student_id = "STU_2026_001"
    name = "Sarah Connor"
    email = "sarah@example.com"
    dept = "Robotics"
    notes = "VIP student"
    
    db_id = student_service.create_student(
        student_id=student_id,
        name=name,
        email=email,
        department=dept,
        notes=notes
    )
    
    assert db_id is not None
    assert isinstance(db_id, int)
    assert db_id > 0
    
    # Verify directly via independent SQLite connection (proving commit)
    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ?", (db_id,))
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row['student_id'] == student_id
    assert row['name'] == name
    assert row['email'] == email
    assert row['department'] == dept
    assert row['notes'] == notes
    assert row['is_active'] == 1


def test_create_duplicate_student_id_rejected(student_service):
    """Test that creating a duplicate student ID raises DuplicateStudentError."""
    student_service.create_student(student_id="DUP101", name="First Student")
    
    with pytest.raises(DuplicateStudentError, match="already exists"):
        student_service.create_student(student_id="DUP101", name="Duplicate Student")


def test_create_student_empty_fields_rejected(student_service):
    """Test that empty student_id or name is rejected with InvalidStudentDataError."""
    with pytest.raises(InvalidStudentDataError, match="student_id cannot be empty"):
        student_service.create_student(student_id="   ", name="Valid Name")
        
    with pytest.raises(InvalidStudentDataError, match="name cannot be empty"):
        student_service.create_student(student_id="VAL101", name="   ")


def test_student_list_immediately_contains_new_student(student_service):
    """Test that get_all_students immediately contains the created student."""
    initial_count = len(student_service.get_all_students())
    
    sid = student_service.create_student(student_id="LIST_TEST_01", name="List Student")
    
    students = student_service.get_all_students()
    assert len(students) == initial_count + 1
    
    student_ids = [s['student_id'] for s in students]
    assert "LIST_TEST_01" in student_ids
