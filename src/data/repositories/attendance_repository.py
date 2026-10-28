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

# marked_by / verified_by of attendance set or changed by hand
MANUAL_MARKER = 'instructor'


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
        notes: Optional[str] = None,
        marked_at: Optional[str] = None
    ) -> int:
        """
        Create a new attendance record in the database.
        
        Args:
            session_id: Session database ID (foreign key to sessions table)
            student_id: Student database ID (foreign key to students table)
            status: Attendance status (e.g., 'present', 'late', 'absent')
            marked_at: Timestamp when attendance was marked; uses the database
                timestamp when omitted.
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
                session_id, student_id, status, marked_at, detection_confidence,
                liveness_score, marked_by, notes
            )
            VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (session_id, student_id, status, marked_at, detection_confidence,
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
    
    def delete_attendance(self, attendance_id: int) -> int:
        """
        Delete an attendance record (the student then counts as absent).
        
        Returns:
            Number of rows affected (0 if record not found, 1 if deleted)
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.execute("DELETE FROM attendance_records WHERE id = ?", (attendance_id,))
                if cursor.rowcount:
                    logger.info(f"Deleted attendance record ID {attendance_id}")
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error deleting attendance record {attendance_id}: {e}")
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
    
    # ------------------------------------------------------------------
    # Session roster: every expected student with their status
    # ------------------------------------------------------------------
    
    # Students expected in a session: anyone with a record in it, plus every
    # active student registered before the session started (no class rosters
    # exist yet, so the register is the roster). The session's start time is
    # local; datetime(..., 'utc') converts it to UTC like the database-default
    # enrolled_at. AnalyticsRepository counts absences with the same rule.
    _ROSTER_FROM = """
        FROM students s
        LEFT JOIN attendance_records a ON a.student_id = s.id AND a.session_id = :session_id
        WHERE (a.id IS NOT NULL
               OR (s.is_active = 1
                   AND s.enrolled_at <= (SELECT datetime(start_time, 'utc') FROM sessions WHERE id = :session_id)))
    """
    
    # Present -> Late -> Absent, then by name
    _ROSTER_ORDER = """
        ORDER BY CASE COALESCE(a.status, 'absent')
                     WHEN 'present' THEN 0 WHEN 'late' THEN 1 ELSE 2 END,
                 s.name COLLATE NOCASE, s.student_id
    """
    
    def get_session_roster_counts(self, session_id: int) -> Dict[str, int]:
        """
        Number of expected students per status for a session.
        
        Args:
            session_id: Session database ID
        
        Returns:
            {status: count}; students without a record count as 'absent'
        """
        query = f"SELECT COALESCE(a.status, 'absent') AS status, COUNT(*) AS n {self._ROSTER_FROM} GROUP BY 1"
        
        try:
            with get_db_connection(self.database_path) as conn:
                rows = conn.execute(query, {'session_id': session_id}).fetchall()
                return {row['status']: row['n'] for row in rows}
        except sqlite3.Error as e:
            logger.error(f"Database error counting the roster of session {session_id}: {e}")
            raise
    
    def get_session_roster(
        self,
        session_id: int,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Expected students of a session with their status, sorted
        Present -> Late -> Absent and by name. Filtering and the row limit
        run in SQL, so a large register never reaches the page in full.
        
        Args:
            session_id: Session database ID
            status: Only this status('absent' = no record or an absent record)
            search: Case-insensitive part of the name or student ID
            limit: Maximum number of rows
        
        Returns:
            Rows with student_db_id, student_id, name, department, status,
            record_id, marked_at, detection_confidence, liveness_score,
            marked_by, verified_by, notes
        """
        query = f"""
            SELECT s.id AS student_db_id, s.student_id, s.name, s.department,
                   COALESCE(a.status, 'absent') AS status, a.id AS record_id, a.marked_at,
                   a.detection_confidence, a.liveness_score, a.marked_by, a.verified_by, a.notes
            {self._ROSTER_FROM}
        """
        params: Dict[str, Any] = {'session_id': session_id}
        if status:
            query += " AND COALESCE(a.status, 'absent') = :status"
            params['status'] = status
        if search:
            # '!' escapes LIKE wildcards, so "%" or "_" typed by the user match literally
            escaped = search.replace('!', '!!').replace('%', '!%').replace('_', '!_')
            query += " AND (s.name LIKE :search ESCAPE '!' OR s.student_id LIKE :search ESCAPE '!')"
            params['search'] = f"%{escaped}%"
        query += self._ROSTER_ORDER
        if limit:
            query += " LIMIT :limit"
            params['limit'] = int(limit)
        
        try:
            with get_db_connection(self.database_path) as conn:
                return [dict(row) for row in conn.execute(query, params).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting the roster of session {session_id}: {e}")
            raise
    
    def get_session_records_with_students(
        self,
        session_id: int,
        newest_first: bool = False,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Attendance records of a session joined with the student's ID and name
        (one query instead of one student lookup per record).
        
        Returns:
            Attendance records plus 'student_code' and 'student_name'
        """
        order = "DESC" if newest_first else "ASC"
        query = f"""
            SELECT a.*, s.student_id AS student_code, s.name AS student_name
            FROM attendance_records a
            LEFT JOIN students s ON s.id = a.student_id
            WHERE a.session_id = ?
            ORDER BY a.marked_at {order}, a.id {order}
        """
        params: List[Any] = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        
        try:
            with get_db_connection(self.database_path) as conn:
                return [dict(row) for row in conn.execute(query, params).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting records of session {session_id}: {e}")
            raise
    
    def regrade_session(self, session_id: int, late_cutoff: str) -> int:
        """
        Re-decide present / late for every present or late record of a
        session: marked before `late_cutoff` = present, otherwise late.
        Absent records, and statuses set or changed by hand (marked_by /
        verified_by = MANUAL_MARKER), are left alone.
        
        Args:
            session_id: Session database ID
            late_cutoff: Local time 'YYYY-MM-DD HH:MM:SS' (start + late threshold)
        
        Returns:
            Number of records whose status changed
        """
        query = """
            UPDATE attendance_records
            SET status = CASE WHEN datetime(marked_at) < datetime(:cutoff) THEN 'present' ELSE 'late' END
            WHERE session_id = :session_id
              AND status IN ('present', 'late')
              AND verified_by IS NULL AND COALESCE(marked_by, '') != :manual
              AND status != CASE WHEN datetime(marked_at) < datetime(:cutoff) THEN 'present' ELSE 'late' END
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.execute(query, {'session_id': session_id, 'cutoff': late_cutoff,
                                              'manual': MANUAL_MARKER})
                if cursor.rowcount:
                    logger.info(f"Re-graded {cursor.rowcount} record(s) of session {session_id}")
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error re-grading session {session_id}: {e}")
            raise
    
    def get_session_status_counts(self, session_id: int) -> Dict[str, int]:
        """Number of attendance records per status in a session."""
        query = "SELECT status, COUNT(*) AS n FROM attendance_records WHERE session_id = ? GROUP BY status"
        
        try:
            with get_db_connection(self.database_path) as conn:
                return {row['status']: row['n'] for row in conn.execute(query, (session_id,)).fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Database error counting records of session {session_id}: {e}")
            raise
