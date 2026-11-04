"""
Statistics Service for analytics calculations and aggregation metrics.

Computes attendance counts, absences, attendance and punctuality rates,
temporal trends, and summaries for students, courses, and sessions.

Absences follow the rule of Session Details: the students expected at a
session are everyone marked in it plus its section's class list, or for a
session without a section every active student registered before it
started (see expected_students.expected_condition). Absence totals for
students, courses and the whole system count closed sessions only, where
attendance is final; a single session's figures are live while it runs.
"""

from bisect import bisect_left, bisect_right
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.data.repositories.analytics_repository import AnalyticsRepository
from src.data.repositories.attendance_repository import AttendanceRepository
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
        self.attendance_repo = AttendanceRepository(database_path)
        self.student_repo = StudentRepository(database_path)
        self.session_repo = SessionRepository(database_path)
    
    def get_overall_dashboard_stats(self) -> Dict[str, Any]:
        """
        Calculate overall high-level statistics for the dashboard.
        
        Returns:
            Dict containing:
            - total_students, active_students: int
            - total_sessions, active_sessions, closed_sessions: int
            - total_records, total_present, total_late: int (all sessions)
            - total_expected, total_absent: int (closed sessions)
            - attendance_rate: float, attended / expected in closed sessions (0.0 - 100.0)
            - punctuality_rate: float, present / (present + late) (0.0 - 100.0)
        """
        counts = self.analytics_repo.get_dashboard_summary()
        
        total_records = counts.get('total_records_count', 0)
        total_present = counts.get('total_present_count', 0)
        total_late = counts.get('total_late_count', 0)
        
        closed = [s for s in self._session_absences().values() if s['status'] == 'closed']
        total_expected = sum(s['expected'] for s in closed)
        total_absent = sum(s['absent'] for s in closed)
        
        return {
            'total_students': counts.get('total_students_count', 0),
            'active_students': counts.get('active_students_count', 0),
            'total_sessions': counts.get('total_sessions_count', 0),
            'active_sessions': counts.get('active_sessions_count', 0),
            'closed_sessions': len(closed),
            'total_records': total_records,
            'total_present': total_present,
            'total_late': total_late,
            'total_expected': total_expected,
            'total_absent': total_absent,
            'attendance_rate': self._percentage(total_expected - total_absent, total_expected),
            'absence_supported': True,
            'punctuality_rate': self._percentage(total_present, total_records)
        }
    
    def get_student_statistics(self, student_db_id: int) -> Dict[str, Any]:
        """
        Calculate detailed attendance statistics for a single student.
        
        Args:
            student_db_id: Student database ID
        
        Returns:
            Dict containing student details, present / late / absent counts,
            attendance history and the closed sessions they missed.
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            return {}
        
        summary = self.analytics_repo.get_student_attendance_summary(student_db_id)
        
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0) or 0
        late_count = summary.get('late_count', 0) or 0
        absence = self._student_absence(student, self._closed_session_starts(),
                                        self.analytics_repo.get_closed_session_attendance_by_student(),
                                        self.analytics_repo.get_closed_section_sessions_by_student())
        
        # Get history records for this student
        history = self.analytics_repo.get_filtered_attendance_history(student_id_str=student['student_id'])
        
        return {
            'student_id': student['student_id'],
            'name': student['name'],
            'department': student['department'] or '—',
            'is_active': bool(student['is_active']),
            'total_sessions': total_marked,
            'total_marked': total_marked,
            'present_count': present_count,
            'late_count': late_count,
            'expected_sessions': absence['expected'],
            'absent_count': absence['absent'],
            'attendance_percentage': absence['percentage'],
            'absence_supported': True,
            'punctuality_percentage': self._percentage(present_count, total_marked),
            'history': history,
            'missed_sessions': self.analytics_repo.get_missed_sessions(student_db_id)
        }
    
    def get_all_students_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated statistics for all students.
        
        Returns:
            List of student stats dictionaries
        """
        raw_summaries = self.analytics_repo.get_all_students_attendance_summary()
        closed_starts = self._closed_session_starts()
        closed_attendance = self.analytics_repo.get_closed_session_attendance_by_student()
        closed_listed = self.analytics_repo.get_closed_section_sessions_by_student()
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            absence = self._student_absence(
                {'id': s['student_db_id'], 'is_active': s['is_active'], 'enrolled_at': s.get('enrolled_at')},
                closed_starts, closed_attendance, closed_listed
            )
            
            results.append({
                'student_db_id': s['student_db_id'],
                'student_id': s['student_id'],
                'name': s['name'],
                'department': s['department'] or '—',
                'is_active': bool(s['is_active']),
                'total_marked': total_marked,
                'present_count': present_cnt,
                'late_count': late_cnt,
                'expected_sessions': absence['expected'],
                'absent_count': absence['absent'],
                'attendance_percentage': absence['percentage'],
                'absence_supported': True,
                'punctuality_percentage': self._percentage(present_cnt, total_marked)
            })
        
        return results
    
    def get_students_with_most_absences(self, limit: int = 20) -> List[Dict[str, Any]]:
        """The students with the most absences in closed sessions (none with zero absences)."""
        students = [s for s in self.get_all_students_statistics() if s['absent_count']]
        students.sort(key=lambda s: (-s['absent_count'], s['name'].lower()))
        return students[:limit]
    
    def get_session_statistics(self, session_id: int) -> Dict[str, Any]:
        """
        Calculate statistics for a specific session.
        
        Args:
            session_id: Session database ID
        
        Returns:
            Dict containing session metadata, present / late / absent counts
            (the numbers Session Details shows; none before the session
            starts) and the attendance records.
        """
        summary = self.analytics_repo.get_session_attendance_summary(session_id)
        if not summary or not summary.get('session_name'):
            return {}
        
        total_marked = summary.get('total_marked', 0)
        present_count = summary.get('present_count', 0) or 0
        late_count = summary.get('late_count', 0) or 0
        
        started = summary['status'] != 'scheduled'
        expected = absent = attendance_percentage = None
        if started:
            roster = self.attendance_repo.get_session_roster_counts(session_id)
            expected = sum(roster.values())
            absent = roster.get('absent', 0)
            attendance_percentage = self._percentage(present_count + late_count, expected)
        
        records = self.analytics_repo.get_filtered_attendance_history(session_id=session_id)
        session = self.session_repo.get_by_id(session_id) or {}
        
        return {
            'session_id': summary['session_id'],
            'session_name': summary['session_name'],
            'course_code': summary['course_code'] or '—',
            'section_id': session.get('section_id'),
            'section_name': session.get('section_name'),
            'left_early_count': self.attendance_repo.count_left_early(session_id),
            'instructor_name': session.get('instructor_name'),
            'start_time': summary['start_time'],
            'end_time': summary.get('end_time'),
            'status': summary['status'],
            'late_threshold_minutes': summary.get('late_threshold_minutes', 15),
            'expected_students': expected,
            'total_marked': total_marked,
            'present_count': present_count,
            'late_count': late_count,
            'absent_count': absent,
            'attendance_percentage': attendance_percentage,
            'absence_supported': started,
            'punctuality_percentage': self._percentage(present_count, total_marked),
            'records': records
        }
    
    def get_all_sessions_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all sessions.
        
        Returns:
            List of session stats (expected / absent are None before a session starts)
        """
        raw_summaries = self.analytics_repo.get_all_sessions_attendance_summary()
        absences = self._session_absences()
        results = []
        for s in raw_summaries:
            total_marked = s.get('total_marked', 0)
            present_cnt = s.get('present_count', 0) or 0
            late_cnt = s.get('late_count', 0) or 0
            absence = absences.get(s['session_id'], {})
            expected = absence.get('expected')
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
                'absent_count': absence.get('absent'),
                'expected_students': expected,
                'attendance_percentage': (self._percentage(present_cnt + late_cnt, expected)
                                          if expected is not None else None),
                'absence_supported': expected is not None,
                'punctuality_percentage': self._percentage(present_cnt, total_marked)
            })
        
        return results
    
    def get_course_statistics(self) -> List[Dict[str, Any]]:
        """
        Get aggregated course statistics, with absences and the attendance
        rate of each course's closed sessions.
        
        Returns:
            List of course statistics dictionaries
        """
        courses = self.analytics_repo.get_course_attendance_summary()
        closed_by_course: Dict[str, Dict[str, int]] = {}
        for s in self._session_absences().values():
            if s['status'] != 'closed':
                continue
            totals = closed_by_course.setdefault(s['course_code'] or 'General', {'expected': 0, 'absent': 0})
            totals['expected'] += s['expected']
            totals['absent'] += s['absent']
        for course in courses:
            totals = closed_by_course.get(course['course_code'], {'expected': 0, 'absent': 0})
            course['absent_count'] = totals['absent']
            course['attendance_rate'] = self._percentage(totals['expected'] - totals['absent'], totals['expected'])
        return courses
    
    def get_section_statistics(self) -> List[Dict[str, Any]]:
        """
        One row per section that has sessions: its sessions, present / late
        (all sessions), absences and attendance rate (closed sessions only,
        like the course summary).
        """
        sections = {s['id']: s for s in self.session_repo.get_sections_with_sessions()}
        if not sections:
            return []
        section_of = {s['id']: s['section_id'] for s in self.session_repo.get_all() if s.get('section_id')}
        totals: Dict[int, Dict[str, int]] = {}
        for s in self.get_all_sessions_statistics():
            section_id = section_of.get(s['session_id'])
            if section_id not in sections:
                continue
            row = totals.setdefault(section_id, {'sessions': 0, 'present': 0, 'late': 0,
                                                 'expected': 0, 'absent': 0})
            row['sessions'] += 1
            row['present'] += s['present_count']
            row['late'] += s['late_count']
            if s['status'] == 'closed' and s['expected_students'] is not None:
                row['expected'] += s['expected_students']
                row['absent'] += s['absent_count']
        results = []
        for section_id, row in totals.items():
            section = sections[section_id]
            results.append({
                'section_id': section_id,
                'course_code': section['course_code'],
                'section_name': section['name'],
                'instructor_name': section['instructor_name'],
                'class_list_size': section['student_count'],
                'total_sessions': row['sessions'],
                'present_count': row['present'],
                'late_count': row['late'],
                'absent_count': row['absent'],
                'attendance_rate': self._percentage(row['expected'] - row['absent'], row['expected']),
            })
        return sorted(results, key=lambda r: (r['course_code'].lower(), r['section_name'].lower()))

    def get_session_roster(self, session_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every expected student of a session with their status (present -> late -> absent), for reports."""
        return self.attendance_repo.get_session_roster(session_id, status=status)
    
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
    
    # ------------------------------------------------------------------
    # Absence helpers
    # ------------------------------------------------------------------
    
    def _session_absences(self) -> Dict[int, Dict[str, Any]]:
        """
        Expected and absent students of every session (None before it starts).
        
        Expected = the section's class list at the start (listed), or for a
        session without a section the active students registered before the
        start (counted with a binary search over the sorted registration
        times, not a scan per session) + students who attended without
        being among them.
        """
        registration_times = self.analytics_repo.get_active_registration_times()
        result = {}
        for s in self.analytics_repo.get_session_expectations():
            entry = {'status': s['status'], 'course_code': s['course_code'], 'section_id': s['section_id'],
                     'start_utc': s['start_utc'], 'expected': None, 'absent': None}
            if s['status'] != 'scheduled' and s['start_utc']:
                if s['section_id'] is not None:
                    registered = s['listed']
                else:
                    registered = bisect_right(registration_times, s['start_utc'])
                entry['absent'] = registered - s['attended_registered']
                entry['expected'] = entry['absent'] + s['attended']
            result[s['session_id']] = entry
        return result

    def _closed_session_starts(self) -> List[str]:
        """Start times (UTC) of closed sessions WITHOUT a section, earliest first."""
        return sorted(s['start_utc'] for s in self.analytics_repo.get_session_expectations()
                      if s['status'] == 'closed' and s['start_utc'] and s['section_id'] is None)
    
    def _student_absence(
        self,
        student: Dict[str, Any],
        closed_starts: List[str],
        closed_attendance: Dict[int, tuple],
        closed_listed: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        A student's expected sessions, absences and attendance percentage
        over closed sessions. Active students are expected at every closed
        session without a section held after they registered (closed_starts)
        and at the closed section sessions whose class list they were on
        (closed_listed); inactive students only where they attended.
        """
        attended, attended_registered = closed_attendance.get(student['id'], (0, 0))
        absent = 0
        if student.get('is_active'):
            held = closed_listed.get(student['id'], 0)
            if student.get('enrolled_at'):
                held += len(closed_starts) - bisect_left(closed_starts, student['enrolled_at'])
            absent = max(0, held - attended_registered)
        expected = attended + absent
        return {'expected': expected, 'absent': absent, 'percentage': self._percentage(attended, expected)}
    
    @staticmethod
    def _percentage(numerator: int, denominator: int) -> float:
        """Return a safely rounded percentage for a recorded-data ratio."""
        return round((numerator / denominator) * 100, 1) if denominator else 0.0
