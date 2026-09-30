"""
Database connection management for the attendance system.

Provides clean SQLite connection handling with proper configuration
and foreign key enforcement.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# Add project root to path for standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_connection(database_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a SQLite database connection.
    
    Args:
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Returns:
        sqlite3.Connection with foreign keys enabled
    
    Raises:
        sqlite3.Error: If connection fails
    """
    db_path = database_path or config.DATABASE_PATH
    
    try:
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection
        conn = sqlite3.connect(str(db_path))
        
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Use Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        logger.debug(f"Database connection established: {db_path}")
        return conn
        
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


@contextmanager
def get_db_connection(database_path: Optional[Path] = None):
    """
    Context manager for database connections.
    
    Ensures proper connection cleanup and transaction handling.
    
    Args:
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Yields:
        sqlite3.Connection with foreign keys enabled
    
    Example:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students")
            conn.commit()
    """
    conn = None
    try:
        conn = get_connection(database_path)
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database transaction failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")


def verify_foreign_keys_enabled(conn: sqlite3.Connection) -> bool:
    """
    Verify that foreign key constraints are enabled.
    
    Args:
        conn: Database connection
    
    Returns:
        True if foreign keys are enabled, False otherwise
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys")
    result = cursor.fetchone()
    return result[0] == 1 if result else False


def execute_query(query: str, params: tuple = (), database_path: Optional[Path] = None) -> list:
    """
    Execute a SELECT query and return results.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Returns:
        List of sqlite3.Row objects
    
    Example:
        results = execute_query("SELECT * FROM students WHERE student_id = ?", ("2021001",))
    """
    with get_db_connection(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def execute_update(query: str, params: tuple = (), database_path: Optional[Path] = None) -> int:
    """
    Execute an INSERT, UPDATE, or DELETE query.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        database_path: Path to database file. If None, uses config.DATABASE_PATH
    
    Returns:
        Number of affected rows
    
    Example:
        rows_affected = execute_update(
            "UPDATE students SET name = ? WHERE id = ?",
            ("John Doe", 1)
        )
    """
    with get_db_connection(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.rowcount


if __name__ == "__main__":
    # Test connection
    print("Testing database connection...")
    
    try:
        with get_db_connection() as conn:
            print(f"[OK] Connected to database: {config.DATABASE_PATH}")
            
            # Verify foreign keys
            if verify_foreign_keys_enabled(conn):
                print("[OK] Foreign keys are enabled")
            else:
                print("[ERROR] Foreign keys are NOT enabled")
        
        print("[OK] Connection closed successfully")
        
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
