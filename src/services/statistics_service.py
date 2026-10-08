"""
Statistics Service for analytics calculations and aggregation metrics.

Computes recorded-attendance counts, punctuality rates, temporal trends,
and summaries for students, courses, and sessions.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from src.data.repositories.analytics_repository import AnalyticsRepository
from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.session_repository import SessionRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticsService:
    """
    Service layer for statistical calculations, metric computation,
    and structured analytics.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize StatisticsService.
        
        Args:
            database_path: Optional database path
        """
        self.database_path = database_path
        self.analytics_repo = AnalyticsRepository(database_path)
        self.student_repo = StudentRepository(database_path)
        self.session_repo = SessionRepository(database_path)

    def get_overall_dashboard_stats(self) -> Dict[str, Any]:
        """
        Calculate overall high-level statistics for the dashboard.
        
        Returns:
            Dict containing:
            - total_students: int
            - active_students: int
            - total_sessions: int
            - active_sessions: int
            - total_records: int
            - total_present: int
            - total_late: int
            - total_absent: None (not supported by the current data model)
            - attendance_rate: None (not supported without a session roster)
            - punctuality_rate: float (percentage 0.0 - 100.0)
        """
        counts = self.analytics_repo.get_dashboard_summary()
        
        total_students = counts.get('total_students_count', 0)
        active_students = counts.get('active_students_count', 0)
        total_sessions = counts.get('total_sessions_count', 0)
        active_sessions = counts.get('active_sessions_count', 0)
        total_records = counts.get('total_records_count', 0)
        total_present = counts.get('total_present_count', 0)
        total_late = counts.get('total_late_count', 0)
        
        # The schema does not have a session roster or student-course enrollment
        # history.  Therefore it cannot reliably establish who was expected at a
        # historical session, and absence/attendance-rate figures must not be
        # inferred from the current active-student count.
        closed_sessions = len(self.session_repo.get_all(status='closed'))
        
        # Punctuality rate = Present / (Present + Late)
        if total_records > 0:
            punctuality_rate = round((total_present / total_records) * 100, 1)
        else:
            punctuality_rate = 0.0
            
        return {
            'total_students': total_students,
            'active_students': active_students,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'closed_sessions': closed_sessions,
            'total_records': total_records,
            'total_present': total_present,
            'total_late': total_late,
            'total_absent': None,
            'attendance_rate': None,
            'absence_supported': False,
            'punctuality_rate': punctuality_rate
        }

    def get_student_statistics(self, student_db_id: int) -> Dict[str, Any]:
        """
        Calculate detailed attendance statistics for a single student.
        
        Args:
            student_db_id: Student database ID
            
        Returns:
            Dict containing student details, recorded present/late counts, and history.
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            return {}
            
        summary = self.analytics_repo.get_student_attendance_summary(student_db_id)
        
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0) or 0
        late_count = summary.get('late_count', 0) or 0
        total_sessions = total_marked
        punctuality_percentage = self._percentage(present_count, total_marked)
            
        # Get history records for this student
        history = self.analytics_repo.get_filtered_attendance_history(student_id_str=student['student_id'])
        
        return {
            'student_id': student['student_id'],
            'name': student['name'],
            'department': student['department'] or '—',
            'is_active': bool(student['is_active']),
            'total_sessions': total_sessions,
            'total_marked': total_marked,
            'present_count': present_count,
            'late_count': late_count,
            'absent_count': None,
            'attendance_percentage': None,
            'absence_supported': False,
            'punctuality_percentage': punctuality_percentage,
            'history': history
        }

    def get_all_students_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics for all students.
        
        Returns:
            List of student stats dictionaries
        """
        raw_summaries = self.analytics_repo.get_all_students_attendance_summary()
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            punctuality_percentage = self._percentage(present_cnt, total_marked)
            
            results.append({
                'student_db_id': s['student_db_id'],
                'student_id': s['student_id'],
                'name': s['name'],
                'department': s['department'] or '—',
                'is_active': bool(s['is_active']),
                'total_marked': total_marked,
                'present_count': present_cnt,
                'late_count': late_cnt,
                'absent_count': None,
                'attendance_percentage': None,
                'absence_supported': False,
                'punctuality_percentage': punctuality_percentage
            })
            
        return results

    def get_session_statistics(self, session_id: int) -> Dict[str, Any]:
        """
        Calculate statistics for a specific session.
        
        Args:
            session_id: Session database ID
            
        Returns:
            Dict containing session metadata, recorded present/late counts, and history.
        """
        summary = self.analytics_repo.get_session_attendance_summary(session_id)
        if not summary or not summary.get('session_name'):
            return {}
            
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0) or 0
        late_count = summary.get('late_count', 0) or 0
        punctuality_percentage = self._percentage(present_count, total_marked)
            
        records = self.analytics_repo.get_filtered_attendance_history(session_id=session_id)
        
        return {
            'session_id': summary['session_id'],
            'session_name': summary['session_name'],
            'course_code': summary['course_code'] or '—',
            'start_time': summary['start_time'],
            'end_time': summary.get('end_time'),
            'status': summary['status'],
            'late_threshold_minutes': summary.get('late_threshold_minutes', 15),
            'expected_students': None,
            'total_marked': total_marked,
            'present_count': present_count,
            'late_count': late_count,
            'absent_count': None,
            'attendance_percentage': None,
            'absence_supported': False,
            'punctuality_percentage': punctuality_percentage,
            'records': records
        }

    def get_all_sessions_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all sessions.
        
        Returns:
            List of session stats
        """
        raw_summaries = self.analytics_repo.get_all_sessions_attendance_summary()
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            punctuality_percentage = self._percentage(present_cnt, total_marked)
            results.append({
                'session_id': s['session_id'],
                'session_name': s['session_name'],
                'course_code': s['course_code'] or '—',
                'start_time': s['start_time'],
                'end_time': s.get('end_time'),
                'status': s['status'],
                'total_marked': total_marked,
                'present_count': present_cnt,
                'late_count': late_cnt,
                'absent_count': None,
                'expected_students': None,
                'attendance_percentage': None,
                'absence_supported': False,
                'punctuality_percentage': punctuality_percentage
            })
            
        return results

    def get_course_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated course statistics.
        
        Returns:
            List of course statistics dictionaries
        """
        return self.analytics_repo.get_course_attendance_summary()

    def get_attendance_trend(self) -> List[Dict[str, Any]]:
        """Get daily present/late counts from recorded attendance only."""
        return self.analytics_repo.get_attendance_trend()

    def get_filtered_history(
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
        Retrieve filtered attendance records.
        """
        return self.analytics_repo.get_filtered_attendance_history(
            student_id_str=student_id_str,
            student_name=student_name,
            course_code=course_code,
            session_id=session_id,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

    @staticmethod
    def _percentage(numerator: int, denominator: int) -> float:
        """Return a safely rounded percentage for a recorded-data ratio."""
        return round((numerator / denominator) * 100, 1) if denominator else 0.0
