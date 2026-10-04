"""
Session Repository for database persistence operations.

Provides clean database access for the sessions table without business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """
    Repository for session persistence operations.
    
    Handles database access for the sessions table.
    Does not contain business logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the session repository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path
    
    def create_session(
        self,
        session_name: str,
        start_time: str,
        course_code: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        status: str = 'scheduled',
        created_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new session in the database.
        
        Args:
            session_name: Name of the session (e.g., "CS101 Lecture")
            start_time: Session start time (ISO format string)
            course_code: Course identifier (optional)
            end_time: Session end time (ISO format string, optional)
            late_threshold_minutes: Minutes after start_time considered late (optional)
            status: Session status (default: 'scheduled')
            created_by: User who created the session (optional)
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created session
        
        Raises:
            sqlite3.Error: On database errors
        """
        query = """
            INSERT INTO sessions (
                session_name, course_code, start_time, end_time,
                late_threshold_minutes, status, created_by, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (session_name, course_code, start_time, end_time,
                     late_threshold_minutes, status, created_by, notes)
                )
                session_id = cursor.lastrowid
                logger.info(f"Created session: {session_name} (ID: {session_id})")
                return session_id
        except sqlite3.Error as e:
            logger.error(f"Database error creating session: {e}")
            raise
    
    def get_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session database primary key
        
        Returns:
            Session record as dictionary, or None if not found
        """
        query = "SELECT * FROM sessions WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error getting session by id {session_id}: {e}")
            raise
    
    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions, optionally filtered by status.
        
        Args:
            status: Filter by status (e.g., 'scheduled', 'active', 'closed'). If None, returns all.
        
        Returns:
            List of session records as dictionaries
        """
        if status:
            query = "SELECT * FROM sessions WHERE status = ? ORDER BY start_time DESC"
            params = (status,)
        else:
            query = "SELECT * FROM sessions ORDER BY start_time DESC"
            params = ()
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all sessions: {e}")
            raise
    
    def get_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific course.
        
        Args:
            course_code: Course identifier
        
        Returns:
            List of session records as dictionaries
        """
        query = "SELECT * FROM sessions WHERE course_code = ? ORDER BY start_time DESC"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (course_code,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting sessions for course {course_code}: {e}")
            raise
    
    def update_session(
        self,
        session_id: int,
        session_name: Optional[str] = None,
        course_code: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Update a session's information.
        
        Args:
            session_id: Session database primary key
            session_name: New session name (optional)
            course_code: New course code (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            late_threshold_minutes: New late threshold (optional)
            status: New status (optional)
            notes: New notes (optional)
        
        Returns:
            Number of rows affected (0 if session not found, 1 if updated)
        
        Raises:
            sqlite3.Error: On database errors
        """
        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []
        
        if session_name is not None:
            updates.append("session_name = ?")
            params.append(session_name)
        
        if course_code is not None:
            updates.append("course_code = ?")
            params.append(course_code)
        
        if start_time is not None:
            updates.append("start_time = ?")
            params.append(start_time)
        
        if end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)
        
        if late_threshold_minutes is not None:
            updates.append("late_threshold_minutes = ?")
            params.append(late_threshold_minutes)
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if not updates:
            # No fields to update
            logger.warning(f"No fields provided to update for session {session_id}")
            return 0
        
        query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"
        params.append(session_id)
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Updated session ID {session_id}")
                else:
                    logger.warning(f"Session ID {session_id} not found for update")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error updating session: {e}")
            raise
    
    def close_session(self, session_id: int) -> int:
        """
        Close a session (set status = 'closed', closed_at = current timestamp).
        
        Args:
            session_id: Session database primary key
        
        Returns:
            Number of rows affected (0 if session not found, 1 if closed)
        """
        query = """
            UPDATE sessions
            SET status = 'closed', closed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Closed session ID {session_id}")
                else:
                    logger.warning(f"Session ID {session_id} not found for closing")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error closing session: {e}")
            raise
    
    def exists(self, session_id: int) -> bool:
        """
        Check if a session exists by ID.
        
        Args:
            session_id: Session database primary key
        
        Returns:
            True if session exists, False otherwise
        """
        query = "SELECT 1 FROM sessions WHERE id = ? LIMIT 1"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error checking session existence: {e}")
            raise
