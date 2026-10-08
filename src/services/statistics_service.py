"""
Statistics Service for analytics calculations and aggregation metrics.

Computes rates, percentages, absence approximations, temporal trends,
and summaries for students, courses, and sessions.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from src.data.repositories.analytics_repository import AnalyticsRepository
from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.session_repository import SessionRepository
from src.data.repositories.attendance_repository import AttendanceRepository
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
        self.attendance_repo = AttendanceRepository(database_path)

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
            - total_absent: int (calculated based on closed sessions * active students - total marked)
            - attendance_rate: float (percentage 0.0 - 100.0)
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
        
        # Closed/concluded sessions count
        closed_sessions = len(self.session_repo.get_all(status='closed'))
        
        # Expected attendance instances = closed_sessions * active_students
        expected_total = closed_sessions * active_students if active_students > 0 else 0
        
        # Approximate absent count for concluded sessions
        if expected_total > 0 and expected_total >= total_records:
            total_absent = expected_total - total_records
        else:
            total_absent = 0
            
        # Attendance rate = (Present + Late) / Expected Total (or / Total Records if no expected baseline)
        if expected_total > 0:
            attendance_rate = round((total_records / expected_total) * 100, 1)
        elif total_records > 0:
            attendance_rate = 100.0
        else:
            attendance_rate = 0.0
            
        attendance_rate = min(100.0, max(0.0, attendance_rate))
        
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
            'total_absent': total_absent,
            'attendance_rate': attendance_rate,
            'punctuality_rate': punctuality_rate
        }

    def get_student_statistics(self, student_db_id: int) -> Dict[str, Any]:
        """
        Calculate detailed attendance statistics for a single student.
        
        Args:
            student_db_id: Student database ID
            
        Returns:
            Dict containing student details, present/late/absent counts, percentage, and history
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            return {}
            
        summary = self.analytics_repo.get_student_attendance_summary(student_db_id)
        
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0)
        late_count = summary.get('late_count', 0)
        total_sessions = summary.get('total_sessions', 0)
        
        absent_count = max(0, total_sessions - total_marked)
        
        if total_sessions > 0:
            attendance_percentage = round((total_marked / total_sessions) * 100, 1)
        elif total_marked > 0:
            attendance_percentage = 100.0
        else:
            attendance_percentage = 0.0
            
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
            'absent_count': absent_count,
            'attendance_percentage': attendance_percentage,
            'history': history
        }

    def get_all_students_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics for all students.
        
        Returns:
            List of student stats dictionaries
        """
        raw_summaries = self.analytics_repo.get_all_students_attendance_summary()
        closed_sessions = len(self.session_repo.get_all(status='closed'))
        active_sessions = len(self.session_repo.get_all(status='active'))
        total_relevant_sessions = closed_sessions + active_sessions
        
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            
            absent_cnt = max(0, total_relevant_sessions - total_marked) if total_relevant_sessions > 0 else 0
            
            if total_relevant_sessions > 0:
                pct = round((total_marked / total_relevant_sessions) * 100, 1)
            elif total_marked > 0:
                pct = 100.0
            else:
                pct = 0.0
                
            results.append({
                'student_db_id': s['student_db_id'],
                'student_id': s['student_id'],
                'name': s['name'],
                'department': s['department'] or '—',
                'is_active': bool(s['is_active']),
                'total_marked': total_marked,
                'present_count': present_cnt,
                'late_count': late_cnt,
                'absent_count': absent_cnt,
                'attendance_percentage': pct
            })
            
        return results

    def get_session_statistics(self, session_id: int) -> Dict[str, Any]:
        """
        Calculate statistics for a specific session.
        
        Args:
            session_id: Session database ID
            
        Returns:
            Dict containing session metadata, present/late/absent counts, and percentage
        """
        summary = self.analytics_repo.get_session_attendance_summary(session_id)
        if not summary or not summary.get('session_name'):
            return {}
            
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0) or 0
        late_count = summary.get('late_count', 0) or 0
        total_active_students = summary.get('total_active_students', 0)
        
        absent_count = max(0, total_active_students - total_marked) if total_active_students > 0 else 0
        
        if total_active_students > 0:
            attendance_percentage = round((total_marked / total_active_students) * 100, 1)
        elif total_marked > 0:
            attendance_percentage = 100.0
        else:
            attendance_percentage = 0.0
            
        records = self.analytics_repo.get_filtered_attendance_history(session_id=session_id)
        
        return {
            'session_id': summary['session_id'],
            'session_name': summary['session_name'],
            'course_code': summary['course_code'] or '—',
            'start_time': summary['start_time'],
            'end_time': summary.get('end_time'),
            'status': summary['status'],
            'late_threshold_minutes': summary.get('late_threshold_minutes', 15),
            'expected_students': total_active_students,
            'total_marked': total_marked,
            'present_count': present_count,
            'late_count': late_count,
            'absent_count': absent_count,
            'attendance_percentage': attendance_percentage,
            'records': records
        }

    def get_all_sessions_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all sessions.
        
        Returns:
            List of session stats
        """
        raw_summaries = self.analytics_repo.get_all_sessions_attendance_summary()
        active_students = len(self.student_repo.get_all(active_only=True))
        
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            absent_cnt = max(0, active_students - total_marked) if active_students > 0 else 0
            
            if active_students > 0:
                pct = round((total_marked / active_students) * 100, 1)
            elif total_marked > 0:
                pct = 100.0
            else:
                pct = 0.0
                
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
                'absent_count': absent_cnt,
                'expected_students': active_students,
                'attendance_percentage': pct
            })
            
        return results

    def get_course_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated course statistics.
        
        Returns:
            List of course statistics dictionaries
        """
        return self.analytics_repo.get_course_attendance_summary()

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
