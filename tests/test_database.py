"""
Database tests for Phase 1B

Tests database connection, schema, constraints, and initialization.
Uses in-memory SQLite databases for isolation.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.database.connection import (
    get_connection,
    get_db_connection,
    verify_foreign_keys_enabled,
    execute_query,
    execute_update,
)
from src.database.schema import (
    initialize_database,
    create_tables,
    create_indexes,
    insert_default_settings,
    get_all_tables,
    get_table_info,
    verify_schema,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def initialized_db(temp_db):
    """Create and initialize a temporary database."""
    initialize_database(temp_db)
    return temp_db


def test_database_connection(temp_db):
    """Test that database connection works."""
    conn = get_connection(temp_db)
    assert conn is not None
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_foreign_keys_enabled(temp_db):
    """Test that foreign keys are enabled."""
    conn = get_connection(temp_db)
    assert verify_foreign_keys_enabled(conn)
    conn.close()


def test_context_manager(temp_db):
    """Test database context manager."""
    with get_db_connection(temp_db) as conn:
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1


def test_create_all_tables(temp_db):
    """Test that all nine tables are created."""
    with get_db_connection(temp_db) as conn:
        create_tables(conn)
        tables = get_all_tables(conn)

    required_tables = [
        'students',
        'enrollments',
        'courses',
        'sections',
        'section_students',
        'sessions',
        'attendance_records',
        'unknown_faces',
        'settings',
    ]

    assert len(tables) == 9
    for table in required_tables:
        assert table in tables


def test_students_table_structure(initialized_db):
    """Test that students table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'students')
        column_names = [col[1] for col in columns]
    
    required_columns = [
        'id', 'student_id', 'name', 'email', 'department',
        'enrolled_at', 'is_active', 'notes'
    ]
    
    for col in required_columns:
        assert col in column_names


def test_enrollments_table_structure(initialized_db):
    """Test that enrollments table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'enrollments')
        column_names = [col[1] for col in columns]
    
    required_columns = [
        'id', 'student_id', 'encoding_path', 'enrolled_at',
        'enrollment_method', 'image_count', 'quality_score', 'is_active'
    ]
    
    for col in required_columns:
        assert col in column_names


def test_sessions_table_structure(initialized_db):
    """Test that sessions table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'sessions')
        column_names = [col[1] for col in columns]
    
    required_columns = [
        'id', 'session_name', 'course_code', 'start_time', 'end_time',
        'late_threshold_minutes', 'status', 'created_by', 'created_at',
        'closed_at', 'notes'
    ]
    
    for col in required_columns:
        assert col in column_names


def test_attendance_records_table_structure(initialized_db):
    """Test that attendance_records table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'attendance_records')
        column_names = [col[1] for col in columns]
    
    required_columns = [
        'id', 'session_id', 'student_id', 'status', 'marked_at',
        'detection_confidence', 'liveness_score', 'marked_by',
        'verified_by', 'verified_at', 'notes'
    ]
    
    for col in required_columns:
        assert col in column_names


def test_unknown_faces_table_structure(initialized_db):
    """Test that unknown_faces table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'unknown_faces')
        column_names = [col[1] for col in columns]
    
    required_columns = [
        'id', 'session_id', 'detected_at', 'image_path',
        'detection_confidence', 'liveness_score', 'closest_match_student_id',
        'closest_match_distance', 'flagged', 'resolved', 'resolution_notes'
    ]
    
    for col in required_columns:
        assert col in column_names


def test_settings_table_structure(initialized_db):
    """Test that settings table has required columns."""
    with get_db_connection(initialized_db) as conn:
        columns = get_table_info(conn, 'settings')
        column_names = [col[1] for col in columns]
    
    required_columns = ['key', 'value', 'value_type', 'description', 'updated_at']
    
    for col in required_columns:
        assert col in column_names


def test_indexes_created(initialized_db):
    """Test that required indexes exist."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
    
    # Check for some key indexes
    assert 'idx_students_student_id' in indexes
    assert 'idx_attendance_session' in indexes
    assert 'idx_sessions_status' in indexes


def test_student_id_unique_constraint(initialized_db):
    """Test that students.student_id UNIQUE constraint works."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        
        # Insert first student
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        conn.commit()
        
        # Try to insert duplicate student_id
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO students (student_id, name) VALUES (?, ?)",
                ("2021001", "Jane Doe")
            )


def test_enrollment_foreign_key_cascade(initialized_db):
    """Test ON DELETE CASCADE for enrollments."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        
        # Insert student
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        student_id = cursor.lastrowid
        
        # Insert enrollment
        cursor.execute(
            "INSERT INTO enrollments (student_id, encoding_path) VALUES (?, ?)",
            (student_id, "/path/to/encoding.npy")
        )
        conn.commit()
        
        # Verify enrollment exists
        cursor.execute("SELECT COUNT(*) FROM enrollments WHERE student_id = ?", (student_id,))
        assert cursor.fetchone()[0] == 1
        
        # Delete student
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        
        # Verify enrollment was deleted (CASCADE)
        cursor.execute("SELECT COUNT(*) FROM enrollments WHERE student_id = ?", (student_id,))
        assert cursor.fetchone()[0] == 0


def test_attendance_foreign_key_cascade(initialized_db):
    """Test ON DELETE CASCADE for attendance records."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        
        # Insert student
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        student_id = cursor.lastrowid
        
        # Insert session
        cursor.execute(
            "INSERT INTO sessions (session_name, start_time) VALUES (?, ?)",
            ("CS101 Lecture", "2026-09-25 10:00:00")
        )
        session_id = cursor.lastrowid
        
        # Insert attendance record
        cursor.execute(
            "INSERT INTO attendance_records (session_id, student_id, status) VALUES (?, ?, ?)",
            (session_id, student_id, "present")
        )
        conn.commit()
        
        # Verify attendance exists
        cursor.execute("SELECT COUNT(*) FROM attendance_records WHERE session_id = ?", (session_id,))
        assert cursor.fetchone()[0] == 1
        
        # Delete session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        
        # Verify attendance was deleted (CASCADE)
        cursor.execute("SELECT COUNT(*) FROM attendance_records WHERE session_id = ?", (session_id,))
        assert cursor.fetchone()[0] == 0


def test_unknown_faces_foreign_key_set_null(initialized_db):
    """Test ON DELETE SET NULL for unknown_faces."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        
        # Insert session
        cursor.execute(
            "INSERT INTO sessions (session_name, start_time) VALUES (?, ?)",
            ("CS101 Lecture", "2026-09-25 10:00:00")
        )
        session_id = cursor.lastrowid
        
        # Insert unknown face
        cursor.execute(
            "INSERT INTO unknown_faces (session_id, detection_confidence) VALUES (?, ?)",
            (session_id, 0.85)
        )
        unknown_id = cursor.lastrowid
        conn.commit()
        
        # Delete session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        
        # Verify unknown_face still exists but session_id is NULL
        cursor.execute("SELECT session_id FROM unknown_faces WHERE id = ?", (unknown_id,))
        result = cursor.fetchone()
        assert result[0] is None


def test_attendance_unique_constraint(initialized_db):
    """Test UNIQUE(session_id, student_id) prevents duplicate attendance."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        
        # Insert student
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        student_id = cursor.lastrowid
        
        # Insert session
        cursor.execute(
            "INSERT INTO sessions (session_name, start_time) VALUES (?, ?)",
            ("CS101 Lecture", "2026-09-25 10:00:00")
        )
        session_id = cursor.lastrowid
        
        # Insert first attendance record
        cursor.execute(
            "INSERT INTO attendance_records (session_id, student_id, status) VALUES (?, ?, ?)",
            (session_id, student_id, "present")
        )
        conn.commit()
        
        # Try to insert duplicate attendance
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO attendance_records (session_id, student_id, status) VALUES (?, ?, ?)",
                (session_id, student_id, "present")
            )


def test_default_settings_inserted(initialized_db):
    """Test that default settings are inserted."""
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM settings")
        count = cursor.fetchone()[0]
        
        assert count == 7  # 7 default settings
        
        # Check specific settings
        cursor.execute("SELECT value FROM settings WHERE key = ?", ("recognition_threshold",))
        assert cursor.fetchone()[0] == "0.6"
        
        cursor.execute("SELECT value FROM settings WHERE key = ?", ("liveness_threshold",))
        assert cursor.fetchone()[0] == "0.75"
        
        cursor.execute("SELECT value FROM settings WHERE key = ?", ("enrollment_images_required",))
        assert cursor.fetchone()[0] == "5"


def test_initialization_idempotent(temp_db):
    """Test that database initialization can run multiple times safely."""
    # Initialize first time
    result1 = initialize_database(temp_db)
    assert result1 is True
    
    # Initialize second time
    result2 = initialize_database(temp_db)
    assert result2 is True
    
    # Verify no duplicate settings
    with get_db_connection(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM settings")
        count = cursor.fetchone()[0]
        assert count == 7  # Still 7, not 14


def test_verify_schema(initialized_db):
    """Test schema verification function."""
    assert verify_schema(initialized_db) is True


def test_execute_query(initialized_db):
    """Test execute_query helper function."""
    # Insert test data
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        conn.commit()
    
    # Query using helper
    results = execute_query(
        "SELECT * FROM students WHERE student_id = ?",
        ("2021001",),
        initialized_db
    )
    
    assert len(results) == 1
    assert results[0]['student_id'] == "2021001"
    assert results[0]['name'] == "John Doe"


def test_execute_update(initialized_db):
    """Test execute_update helper function."""
    # Insert test data
    with get_db_connection(initialized_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            ("2021001", "John Doe")
        )
        conn.commit()
    
    # Update using helper
    rows_affected = execute_update(
        "UPDATE students SET name = ? WHERE student_id = ?",
        ("Jane Doe", "2021001"),
        initialized_db
    )
    
    assert rows_affected == 1
    
    # Verify update
    results = execute_query(
        "SELECT name FROM students WHERE student_id = ?",
        ("2021001",),
        initialized_db
    )
    assert results[0]['name'] == "Jane Doe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
