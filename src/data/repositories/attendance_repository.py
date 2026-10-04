"""
Attendance Repository for database persistence operations.

Provides clean database access for the attendance_records table without business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttendanceRepository:
    """
    Repository for attendance record persistence operations.
    
    Handles database access for the attendance_records table.
    Does not contain business logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the attendance repository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path
    
    def create_attendance(
        self,
        session_id: int,
        student_id: int,
        status: str,
        detection_confidence: Optional[float] = None,
        liveness_score: Optional[float] = None,
        marked_by: str = 'system',
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new attendance record in the database.
        
        Args:
            session_id: Session database ID (foreign key to sessions table)
            student_id: Student database ID (foreign key to students table)
            status: Attendance status (e.g., 'present', 'late', 'absent')
            detection_confidence: Face detection confidence score (optional)
            liveness_score: Liveness detection score (optional)
            marked_by: Who marked the attendance (default: 'system')
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created attendance record
        
        Raises:
            sqlite3.IntegrityError: If duplicate (session_id, student_id) or invalid foreign key
            sqlite3.Error: On other database errors
        """
        query = """
            INSERT INTO attendance_records (
                session_id, student_id, status, detection_confidence,
                liveness_score, marked_by, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (session_id, student_id, status, detection_confidence,
                     liveness_score, marked_by, notes)
                )
                attendance_id = cursor.lastrowid
                logger.info(
                    f"Created attendance record: Session {session_id}, "
                    f"Student {student_id}, Status {status} (ID: {attendance_id})"
                )
                return attendance_id
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create attendance record: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error creating attendance record: {e}")
            raise
    
    def get_by_id(self, attendance_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an attendance record by ID.
        
        Args:
            attendance_id: Attendance record database primary key
        
        Returns:
            Attendance record as dictionary, or None if not found
        """
        query = "SELECT * FROM attendance_records WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (attendance_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance by id {attendance_id}: {e}")
            raise
    
    def get_by_session(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a specific session.
        
        Args:
            session_id: Session database ID
        
        Returns:
            List of attendance records as dictionaries
        """
        query = "SELECT * FROM attendance_records WHERE session_id = ? ORDER BY marked_at"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance for session {session_id}: {e}")
            raise
    
    def get_by_student(self, student_id: int) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a specific student.
        
        Args:
            student_id: Student database ID
        
        Returns:
            List of attendance records as dictionaries
        """
        query = "SELECT * FROM attendance_records WHERE student_id = ? ORDER BY marked_at DESC"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance for student {student_id}: {e}")
            raise
    
    def get_by_session_and_student(
        self,
        session_id: int,
        student_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get attendance record for a specific student in a specific session.
        
        Args:
            session_id: Session database ID
            student_id: Student database ID
        
        Returns:
            Attendance record as dictionary, or None if not found
        """
        query = "SELECT * FROM attendance_records WHERE session_id = ? AND student_id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id, student_id))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(
                f"Database error getting attendance for session {session_id}, "
                f"student {student_id}: {e}"
            )
            raise
    
    def exists(self, session_id: int, student_id: int) -> bool:
        """
        Check if an attendance record exists for a student in a session.
        
        This is the key method for preventing duplicate attendance.
        
        Args:
            session_id: Session database ID
            student_id: Student database ID
        
        Returns:
            True if attendance record exists, False otherwise
        """
        query = """
            SELECT 1 FROM attendance_records
            WHERE session_id = ? AND student_id = ?
            LIMIT 1
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id, student_id))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error checking attendance existence: {e}")
            raise
    
    def update_attendance(
        self,
        attendance_id: int,
        status: Optional[str] = None,
        detection_confidence: Optional[float] = None,
        liveness_score: Optional[float] = None,
        verified_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Update an attendance record's information.
        
        Args:
            attendance_id: Attendance record database primary key
            status: New status (optional)
            detection_confidence: New detection confidence (optional)
            liveness_score: New liveness score (optional)
            verified_by: User who verified the attendance (optional)
            notes: New notes (optional)
        
        Returns:
            Number of rows affected (0 if record not found, 1 if updated)
        
        Raises:
            sqlite3.Error: On database errors
        """
        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if detection_confidence is not None:
            updates.append("detection_confidence = ?")
            params.append(detection_confidence)
        
        if liveness_score is not None:
            updates.append("liveness_score = ?")
            params.append(liveness_score)
        
        if verified_by is not None:
            updates.append("verified_by = ?")
            params.append(verified_by)
            updates.append("verified_at = CURRENT_TIMESTAMP")
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if not updates:
            # No fields to update
            logger.warning(f"No fields provided to update for attendance {attendance_id}")
            return 0
        
        query = f"UPDATE attendance_records SET {', '.join(updates)} WHERE id = ?"
        params.append(attendance_id)
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Updated attendance record ID {attendance_id}")
                else:
                    logger.warning(f"Attendance record ID {attendance_id} not found for update")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error updating attendance: {e}")
            raise
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all attendance records.
        
        Returns:
            List of attendance records as dictionaries
        """
        query = "SELECT * FROM attendance_records ORDER BY marked_at DESC"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all attendance records: {e}")
            raise
    
    def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all attendance records with a specific status.
        
        Args:
            status: Attendance status (e.g., 'present', 'late', 'absent')
        
        Returns:
            List of attendance records as dictionaries
        """
        query = "SELECT * FROM attendance_records WHERE status = ? ORDER BY marked_at DESC"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (status,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance by status {status}: {e}")
            raise
