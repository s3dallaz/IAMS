"""
Analytics Repository for intelligent statistics and reporting queries.

Provides efficient SQL aggregation and filtering queries across students,
sessions, attendance records, and enrollment data without embedding business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyticsRepository:
    """
    Repository for high-level statistics and analytical data access.
    
    Executes optimized SQL aggregation queries for dashboard metrics,
    student breakdowns, session/course summaries, and filtered history.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize AnalyticsRepository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Get high-level system metrics (total students, total sessions, total records, present, late).
        
        Returns:
            Dict containing count metrics
        """
        query_counts = """
            SELECT 
                (SELECT COUNT(*) FROM students WHERE is_active = 1) AS active_students_count,
                (SELECT COUNT(*) FROM students) AS total_students_count,
                (SELECT COUNT(*) FROM sessions) AS total_sessions_count,
                (SELECT COUNT(*) FROM sessions WHERE status = 'active') AS active_sessions_count,
                (SELECT COUNT(*) FROM attendance_records) AS total_records_count,
                (SELECT COUNT(*) FROM attendance_records WHERE status = 'present') AS total_present_count,
                (SELECT COUNT(*) FROM attendance_records WHERE status = 'late') AS total_late_count
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query_counts)
                row = cursor.fetchone()
                return dict(row) if row else {}
        except sqlite3.Error as e:
            logger.error(f"Database error getting dashboard summary: {e}")
            raise

    def get_student_attendance_summary(self, student_db_id: int) -> Dict[str, Any]:
        """
        Get attendance summary for a specific student across all sessions.
        
        Args:
            student_db_id: Student database ID
            
        Returns:
            Dict with student stats (present_count, late_count, total_marked, total_sessions)
        """
        query = """
            SELECT 
                COUNT(*) AS total_marked,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM attendance_records
            WHERE student_id = ?
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (student_db_id,))
                row = cursor.fetchone()
                stats = dict(row) if row else {}
                
                # Total available closed or active sessions in system
                cursor.execute("SELECT COUNT(*) AS count FROM sessions WHERE status IN ('active', 'closed')")
                sess_row = cursor.fetchone()
                stats['total_sessions'] = sess_row['count'] if sess_row else 0
                return stats
        except sqlite3.Error as e:
            logger.error(f"Database error getting student attendance summary for {student_db_id}: {e}")
            raise

    def get_all_students_attendance_summary(self) -> List[Dict[str, Any]]:
        """
        Get attendance counts for all students.
        
        Returns:
            List of student stats dictionaries
        """
        query = """
            SELECT 
                s.id AS student_db_id,
                s.student_id,
                s.name,
                s.department,
                s.is_active,
                s.enrolled_at,
                COUNT(a.id) AS total_marked,
                SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN a.status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM students s
            LEFT JOIN attendance_records a ON s.id = a.student_id
            GROUP BY s.id
            ORDER BY s.student_id ASC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all students summary: {e}")
            raise

    def get_session_attendance_summary(self, session_id: int) -> Dict[str, Any]:
        """
        Get attendance summary for a specific session.
        
        Args:
            session_id: Session database ID
            
        Returns:
            Dict containing session metrics
        """
        query = """
            SELECT 
                sess.id AS session_id,
                sess.session_name,
                sess.course_code,
                sess.start_time,
                sess.end_time,
                sess.status,
                sess.late_threshold_minutes,
                COUNT(a.id) AS total_marked,
                SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN a.status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM sessions sess
            LEFT JOIN attendance_records a ON sess.id = a.session_id
            WHERE sess.id = ?
            GROUP BY sess.id
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                row = cursor.fetchone()
                stats = dict(row) if row else {}
                
                # Active students count in system at time
                cursor.execute("SELECT COUNT(*) AS count FROM students WHERE is_active = 1")
                stu_row = cursor.fetchone()
                stats['total_active_students'] = stu_row['count'] if stu_row else 0
                return stats
        except sqlite3.Error as e:
            logger.error(f"Database error getting session attendance summary for {session_id}: {e}")
            raise

    def get_all_sessions_attendance_summary(self) -> List[Dict[str, Any]]:
        """
        Get attendance metrics across all sessions.
        
        Returns:
            List of session summaries
        """
        query = """
            SELECT 
                sess.id AS session_id,
                sess.session_name,
                sess.course_code,
                sess.start_time,
                sess.end_time,
                sess.status,
                sess.late_threshold_minutes,
                COUNT(a.id) AS total_marked,
                SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN a.status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM sessions sess
            LEFT JOIN attendance_records a ON sess.id = a.session_id
            GROUP BY sess.id
            ORDER BY sess.start_time DESC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all sessions summary: {e}")
            raise

    def get_course_attendance_summary(self) -> List[Dict[str, Any]]:
        """
        Get aggregated attendance statistics grouped by course code.
        
        Returns:
            List of course metrics
        """
        query = """
            SELECT 
                COALESCE(sess.course_code, 'General') AS course_code,
                COUNT(DISTINCT sess.id) AS total_sessions,
                COUNT(a.id) AS total_marked,
                SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN a.status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM sessions sess
            LEFT JOIN attendance_records a ON sess.id = a.session_id
            GROUP BY COALESCE(sess.course_code, 'General')
            ORDER BY course_code ASC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting course attendance summary: {e}")
            raise

    def get_attendance_trend(self) -> List[Dict[str, Any]]:
        """Return recorded present/late counts grouped by marked date."""
        query = """
            SELECT
                DATE(marked_at) AS attendance_date,
                COUNT(*) AS total_marked,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) AS present_count,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) AS late_count
            FROM attendance_records
            GROUP BY DATE(marked_at)
            ORDER BY attendance_date ASC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting attendance trend: {e}")
            raise

    def get_filtered_attendance_history(
        self,
        student_id_str: Optional[str] = None,
        student_name: Optional[str] = None,
        course_code: Optional[str] = None,
        session_id: Optional[int] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query attendance history with flexible, multi-parameter filtering.
        
        Args:
            student_id_str: Student identification string (partial or exact)
            student_name: Student name (partial match)
            course_code: Course code (exact or partial)
            session_id: Specific session database ID
            status: Attendance status ('present', 'late', etc.)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            List of structured attendance record rows with joined student & session details
        """
        where_clauses = []
        params = []
        
        if student_id_str:
            where_clauses.append("s.student_id LIKE ?")
            params.append(f"%{student_id_str.strip()}%")
            
        if student_name:
            where_clauses.append("s.name LIKE ?")
            params.append(f"%{student_name.strip()}%")
            
        if course_code:
            where_clauses.append("sess.course_code LIKE ?")
            params.append(f"%{course_code.strip()}%")
            
        if session_id is not None:
            where_clauses.append("a.session_id = ?")
            params.append(session_id)
            
        if status and status != 'All':
            where_clauses.append("a.status = ?")
            params.append(status.lower())
            
        if start_date:
            where_clauses.append("DATE(a.marked_at) >= DATE(?)")
            params.append(start_date)
            
        if end_date:
            where_clauses.append("DATE(a.marked_at) <= DATE(?)")
            params.append(end_date)
            
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        
        query = f"""
            SELECT 
                a.id AS record_id,
                a.marked_at,
                a.status,
                a.detection_confidence,
                a.liveness_score,
                a.marked_by,
                a.verified_by,
                a.verified_at,
                a.notes,
                s.id AS student_db_id,
                s.student_id,
                s.name AS student_name,
                s.department AS student_department,
                sess.id AS session_id,
                sess.session_name,
                sess.course_code,
                sess.start_time AS session_start_time
            FROM attendance_records a
            JOIN students s ON a.student_id = s.id
            JOIN sessions sess ON a.session_id = sess.id
            {where_sql}
            ORDER BY a.marked_at DESC
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error querying filtered attendance history: {e}")
            raise

    # ------------------------------------------------------------------
    # Absences. Expected students of a session are the ones Session
    # Details lists (AttendanceRepository._ROSTER_FROM): everyone marked in
    # it plus every active student registered (enrolled_at, UTC) before it
    # started (start_time, local -> datetime(..., 'utc')). These queries
    # return the pieces; StatisticsService counts with them.
    # ------------------------------------------------------------------
    
    def get_session_expectations(self) -> List[Dict[str, Any]]:
        """
        Per session: its start time in UTC, how many students attended
        (present / late), and how many of those are active students
        registered before the start (they are already counted among the
        registered students, so they must not be counted twice).
        """
        query = """
            SELECT sess.id AS session_id, sess.session_name, sess.course_code, sess.status,
                   datetime(sess.start_time, 'utc') AS start_utc,
                   COALESCE(SUM(a.status IN ('present', 'late')), 0) AS attended,
                   COALESCE(SUM(a.status IN ('present', 'late') AND s.is_active = 1
                                AND s.enrolled_at <= datetime(sess.start_time, 'utc')), 0) AS attended_registered
            FROM sessions sess
            LEFT JOIN attendance_records a ON a.session_id = sess.id
            LEFT JOIN students s ON s.id = a.student_id
            GROUP BY sess.id
            ORDER BY sess.start_time DESC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return [dict(row) for row in conn.execute(query).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting session expectations: {e}")
            raise
    
    def get_active_registration_times(self) -> List[str]:
        """Registration times (enrolled_at, UTC) of all active students, oldest first."""
        query = """
            SELECT enrolled_at FROM students
            WHERE is_active = 1 AND enrolled_at IS NOT NULL
            ORDER BY enrolled_at
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return [row[0] for row in conn.execute(query).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting registration times: {e}")
            raise
    
    def get_closed_session_attendance_by_student(self) -> Dict[int, Tuple[int, int]]:
        """
        Per student with attendance in closed sessions:
        (closed sessions attended, of those the ones held after the student registered).
        """
        query = """
            SELECT a.student_id,
                   COUNT(*) AS attended,
                   COALESCE(SUM(s.enrolled_at <= datetime(sess.start_time, 'utc')), 0) AS attended_registered
            FROM attendance_records a
            JOIN sessions sess ON sess.id = a.session_id AND sess.status = 'closed'
            JOIN students s ON s.id = a.student_id
            WHERE a.status IN ('present', 'late')
            GROUP BY a.student_id
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return {row['student_id']: (row['attended'], row['attended_registered'])
                        for row in conn.execute(query).fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Database error getting closed-session attendance by student: {e}")
            raise
    
    def get_missed_sessions(self, student_db_id: int) -> List[Dict[str, Any]]:
        """
        Closed sessions an active student was expected at (held after they
        registered) but did not attend, newest first. Empty for inactive
        students, who are not expected anywhere.
        """
        query = """
            SELECT sess.id AS session_id, sess.session_name, sess.course_code,
                   sess.start_time AS session_start_time
            FROM sessions sess
            JOIN students s ON s.id = :student_id AND s.is_active = 1
            WHERE sess.status = 'closed'
              AND s.enrolled_at <= datetime(sess.start_time, 'utc')
              AND NOT EXISTS (
                  SELECT 1 FROM attendance_records a
                  WHERE a.session_id = sess.id AND a.student_id = s.id
                    AND a.status IN ('present', 'late'))
            ORDER BY sess.start_time DESC
        """
        try:
            with get_db_connection(self.database_path) as conn:
                return [dict(row) for row in conn.execute(query, {'student_id': student_db_id}).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting missed sessions of student {student_db_id}: {e}")
            raise
