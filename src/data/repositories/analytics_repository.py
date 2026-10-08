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
