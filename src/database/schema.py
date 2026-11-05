"""
Database schema definition and initialization for the attendance system.

Defines all database tables, constraints, indexes, and default settings.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.database.connection import get_db_connection, verify_foreign_keys_enabled
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)


# ============================================================================
# TABLE CREATION SQL
# ============================================================================

CREATE_STUDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    department TEXT,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    notes TEXT
)
"""

CREATE_ENROLLMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    encoding_path TEXT NOT NULL,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enrollment_method TEXT DEFAULT 'camera',
    image_count INTEGER DEFAULT 1,
    quality_score REAL,
    view_angle TEXT DEFAULT 'unknown',
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
)
"""

# Accounts. role: instructor < admin < it. password_hash is scrypt (never the
# password itself); failed_attempts / locked_until implement the lockout.
# recovery_code_hash: the one-time code for a forgotten password (scrypt too).
CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('instructor', 'admin', 'it')),
    password_hash TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    must_change_password BOOLEAN DEFAULT 0,
    recovery_code_hash TEXT,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
)
"""

CREATE_COURSES_TABLE = """
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT UNIQUE NOT NULL COLLATE NOCASE,
    name TEXT NOT NULL,
    department TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
)
"""

# A course's group of students (e.g. "A", "Lab 1"), with its instructor.
# instructor_id: the instructor's account (they see and run this section);
# instructor_name: the name shown (the account's name, or typed without an account)
CREATE_SECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL,
    name TEXT NOT NULL COLLATE NOCASE,
    instructor_name TEXT,
    instructor_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    UNIQUE(course_id, name)
)
"""

# The class list of a section. added_at is UTC, like students.enrolled_at
CREATE_SECTION_STUDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS section_students (
    section_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (section_id, student_id),
    FOREIGN KEY (section_id) REFERENCES sections(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
)
"""

# section_id: the section whose class list is expected (NULL = no section).
# created_by_user_id: the account that created it (an instructor owns the
# sessions of their sections and the ones they created).
# Older databases get these columns from migrate_schema().
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT NOT NULL,
    course_code TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    late_threshold_minutes INTEGER DEFAULT 15,
    status TEXT DEFAULT 'scheduled',
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    notes TEXT,
    section_id INTEGER REFERENCES sections(id),
    created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    checkout_enabled BOOLEAN DEFAULT 0,
    checkout_minutes INTEGER DEFAULT 10
)
"""

CREATE_ATTENDANCE_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_confidence REAL,
    liveness_score REAL,
    marked_by TEXT DEFAULT 'system',
    verified_by TEXT,
    verified_at TIMESTAMP,
    notes TEXT,
    left_early BOOLEAN DEFAULT 0,
    -- Time present / late is judged by, when it is earlier than marked_at:
    -- first seen by the camera, or the start of the line the student was in.
    -- NULL = marked_at.
    arrived_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    UNIQUE(session_id, student_id)
)
"""

# Check-out and spot checks: windows of time in which students who already
# checked in show their face again. Times are local, like session times.
# status: scheduled -> open -> done, or void (cancelled / nobody checked:
# the camera was probably off, so nobody is flagged).
CREATE_ATTENDANCE_CHECKS_TABLE = """
CREATE TABLE IF NOT EXISTS attendance_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('checkout', 'spot')),
    starts_at TIMESTAMP NOT NULL,
    ends_at TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'open', 'done', 'void')),
    is_random BOOLEAN DEFAULT 0,
    created_by TEXT,
    created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_seen_at TIMESTAMP,
    void_reason TEXT,
    close_session_after BOOLEAN DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
)
"""

# Who showed their face during a check
CREATE_CHECK_MARKS_TABLE = """
CREATE TABLE IF NOT EXISTS check_marks (
    check_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    marked_at TIMESTAMP NOT NULL,
    detection_confidence REAL,
    liveness_score REAL,
    marked_by TEXT DEFAULT 'system',
    PRIMARY KEY (check_id, student_id),
    FOREIGN KEY (check_id) REFERENCES attendance_checks(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
)
"""

# Who did what, when (append-only; kept when the account or the object is deleted,
# so user_id / session_id have no foreign keys and the names are copied in)
CREATE_ACTIVITY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TIMESTAMP NOT NULL,
    user_id INTEGER,
    username TEXT,
    full_name TEXT,
    role TEXT,
    category TEXT NOT NULL,
    action TEXT NOT NULL,
    summary TEXT NOT NULL,
    target_type TEXT,
    target_id INTEGER,
    session_id INTEGER,
    details TEXT
)
"""

CREATE_UNKNOWN_FACES_TABLE = """
CREATE TABLE IF NOT EXISTS unknown_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT,
    detection_confidence REAL,
    liveness_score REAL,
    closest_match_student_id INTEGER,
    closest_match_distance REAL,
    flagged BOOLEAN DEFAULT 0,
    resolved BOOLEAN DEFAULT 0,
    resolution_notes TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    FOREIGN KEY (closest_match_student_id) REFERENCES students(id) ON DELETE SET NULL
)
"""

CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


# ============================================================================
# INDEX CREATION SQL
# ============================================================================

CREATE_INDEXES = [
    # Students indexes
    "CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id)",
    "CREATE INDEX IF NOT EXISTS idx_students_active ON students(is_active)",
    
    # Enrollments indexes
    "CREATE INDEX IF NOT EXISTS idx_enrollments_student ON enrollments(student_id)",
    "CREATE INDEX IF NOT EXISTS idx_enrollments_active ON enrollments(is_active)",
    
    # Sessions indexes
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_course ON sessions(course_code)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_section ON sessions(section_id)",

    # Courses / sections / class lists indexes
    "CREATE INDEX IF NOT EXISTS idx_sections_course ON sections(course_id)",
    "CREATE INDEX IF NOT EXISTS idx_sections_instructor ON sections(instructor_id)",
    "CREATE INDEX IF NOT EXISTS idx_section_students_student ON section_students(student_id)",

    # Attendance records indexes
    "CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance_records(student_id)",
    "CREATE INDEX IF NOT EXISTS idx_attendance_status ON attendance_records(status)",
    "CREATE INDEX IF NOT EXISTS idx_attendance_marked_at ON attendance_records(marked_at)",
    "CREATE INDEX IF NOT EXISTS idx_checks_session ON attendance_checks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_checks_status ON attendance_checks(status)",
    "CREATE INDEX IF NOT EXISTS idx_activity_at ON activity_log(at)",
    "CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_log(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_activity_target ON activity_log(target_type, target_id)",
    
    # Unknown faces indexes
    "CREATE INDEX IF NOT EXISTS idx_unknown_session ON unknown_faces(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_unknown_flagged ON unknown_faces(flagged)",
    "CREATE INDEX IF NOT EXISTS idx_unknown_resolved ON unknown_faces(resolved)",
]


# ============================================================================
# DEFAULT SETTINGS
# ============================================================================

DEFAULT_SETTINGS = [
    ('recognition_threshold', '0.6', 'float', 'Face recognition distance threshold'),
    ('liveness_threshold', '0.75', 'float', 'Liveness detection confidence threshold'),
    ('detection_confidence', '0.7', 'float', 'Face detection confidence threshold'),
    ('enrollment_images_required', '5', 'integer', 'Number of images required for enrollment'),
    ('late_threshold_default', '15', 'integer', 'Default late threshold in minutes'),
    ('require_liveness', 'true', 'boolean', 'Require liveness check for attendance'),
    ('auto_close_sessions', 'true', 'boolean', 'Automatically close sessions after end time'),
]


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create all required database tables.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    # Create tables in order (respecting foreign key dependencies)
    tables = [
        ("students", CREATE_STUDENTS_TABLE),
        ("enrollments", CREATE_ENROLLMENTS_TABLE),
        ("users", CREATE_USERS_TABLE),
        ("courses", CREATE_COURSES_TABLE),
        ("sections", CREATE_SECTIONS_TABLE),
        ("section_students", CREATE_SECTION_STUDENTS_TABLE),
        ("sessions", CREATE_SESSIONS_TABLE),
        ("attendance_records", CREATE_ATTENDANCE_RECORDS_TABLE),
        ("attendance_checks", CREATE_ATTENDANCE_CHECKS_TABLE),
        ("check_marks", CREATE_CHECK_MARKS_TABLE),
        ("activity_log", CREATE_ACTIVITY_LOG_TABLE),
        ("unknown_faces", CREATE_UNKNOWN_FACES_TABLE),
        ("settings", CREATE_SETTINGS_TABLE),
    ]
    
    for table_name, create_sql in tables:
        cursor.execute(create_sql)
        logger.debug(f"Table '{table_name}' created or verified")
    
    conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    """
    Bring a database made by an older version up to date. Only adds what
    is missing; existing data is kept. Safe to run multiple times.

    Args:
        conn: Database connection
    """
    user_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if 'recovery_code_hash' not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN recovery_code_hash TEXT")
        logger.info("Added users.recovery_code_hash (existing accounts have no recovery code yet)")
    session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    if 'section_id' not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN section_id INTEGER REFERENCES sections(id)")
        logger.info("Added sessions.section_id (existing sessions have no section)")
    if 'created_by_user_id' not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN created_by_user_id INTEGER "
                     "REFERENCES users(id) ON DELETE SET NULL")
        logger.info("Added sessions.created_by_user_id")
    if 'checkout_enabled' not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN checkout_enabled BOOLEAN DEFAULT 0")
        conn.execute("ALTER TABLE sessions ADD COLUMN checkout_minutes INTEGER DEFAULT 10")
        logger.info("Added sessions.checkout_enabled / checkout_minutes (off for existing sessions)")
    record_columns = {row[1] for row in conn.execute("PRAGMA table_info(attendance_records)")}
    if 'left_early' not in record_columns:
        conn.execute("ALTER TABLE attendance_records ADD COLUMN left_early BOOLEAN DEFAULT 0")
        logger.info("Added attendance_records.left_early")
    if 'arrived_at' not in record_columns:
        conn.execute("ALTER TABLE attendance_records ADD COLUMN arrived_at TIMESTAMP")
        logger.info("Added attendance_records.arrived_at (existing records are judged by marked_at)")
    check_columns = {row[1] for row in conn.execute("PRAGMA table_info(attendance_checks)")}
    if check_columns and 'camera_seen_at' not in check_columns:
        conn.execute("ALTER TABLE attendance_checks ADD COLUMN camera_seen_at TIMESTAMP")
        conn.execute("ALTER TABLE attendance_checks ADD COLUMN void_reason TEXT")
        conn.execute("ALTER TABLE attendance_checks ADD COLUMN close_session_after BOOLEAN DEFAULT 0")
        logger.info("Added attendance_checks.camera_seen_at / void_reason / close_session_after")
    section_columns = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}
    if 'instructor_id' not in section_columns:
        conn.execute("ALTER TABLE sections ADD COLUMN instructor_id INTEGER "
                     "REFERENCES users(id) ON DELETE SET NULL")
        logger.info("Added sections.instructor_id (existing sections keep their typed instructor name)")
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create all required database indexes.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    for index_sql in CREATE_INDEXES:
        cursor.execute(index_sql)
    
    logger.debug(f"Created {len(CREATE_INDEXES)} indexes")
    conn.commit()


def insert_default_settings(conn: sqlite3.Connection) -> None:
    """
    Insert default settings into the settings table.
    
    Safe to run multiple times - only inserts if key doesn't exist.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    for key, value, value_type, description in DEFAULT_SETTINGS:
        # Check if setting already exists
        cursor.execute("SELECT key FROM settings WHERE key = ?", (key,))
        if cursor.fetchone() is None:
            cursor.execute(
                """
                INSERT INTO settings (key, value, value_type, description)
                VALUES (?, ?, ?, ?)
                """,
                (key, value, value_type, description)
            )
            logger.debug(f"Inserted default setting: {key} = {value}")
    
    conn.commit()


def initialize_database(database_path: Optional[Path] = None) -> bool:
    """
    Initialize the database with schema, indexes, and default settings.
    
    Creates the database file if it doesn't exist.
    Safe to run multiple times.
    
    Args:
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Returns:
        True if initialization successful, False otherwise
    """
    db_path = database_path or config.DATABASE_PATH
    
    try:
        logger.info(f"Initializing database: {db_path}")
        
        # Ensure database directory exists
        config.ensure_directories()
        
        with get_db_connection(db_path) as conn:
            # Verify foreign keys are enabled
            if not verify_foreign_keys_enabled(conn):
                logger.error("Foreign key constraints are not enabled")
                return False
            
            # Create tables
            create_tables(conn)
            logger.info("Database tables created")

            # Add columns that older databases lack (before their indexes)
            migrate_schema(conn)

            # Create indexes
            create_indexes(conn)
            logger.info("Database indexes created")
            
            # Insert default settings
            insert_default_settings(conn)
            logger.info("Default settings initialized")
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def get_table_info(conn: sqlite3.Connection, table_name: str) -> list:
    """
    Get schema information for a table.
    
    Args:
        conn: Database connection
        table_name: Name of the table
    
    Returns:
        List of column information
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()


def get_all_tables(conn: sqlite3.Connection) -> list:
    """
    Get list of all tables in the database.
    
    Args:
        conn: Database connection
    
    Returns:
        List of table names
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


def verify_schema(database_path: Optional[Path] = None) -> bool:
    """
    Verify that all required tables exist.
    
    Args:
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Returns:
        True if all tables exist, False otherwise
    """
    required_tables = [
        'students',
        'enrollments',
        'users',
        'courses',
        'sections',
        'section_students',
        'sessions',
        'attendance_records',
        'attendance_checks',
        'check_marks',
        'activity_log',
        'unknown_faces',
        'settings',
    ]
    
    try:
        with get_db_connection(database_path) as conn:
            existing_tables = get_all_tables(conn)
            
            for table in required_tables:
                if table not in existing_tables:
                    logger.error(f"Missing required table: {table}")
                    return False
            
            logger.info("Schema verification passed")
            return True
            
    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False


if __name__ == "__main__":
    # Test schema initialization
    print("Testing database schema initialization...")
    
    if initialize_database():
        print("[OK] Database initialized successfully")
        
        if verify_schema():
            print("[OK] Schema verification passed")
            
            # Display table information
            with get_db_connection() as conn:
                tables = get_all_tables(conn)
                print(f"\n[OK] Found {len(tables)} tables: {', '.join(tables)}")
        else:
            print("[ERROR] Schema verification failed")
    else:
        print("[ERROR] Database initialization failed")
