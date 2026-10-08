"""
Repository layer for database persistence operations.

This package contains repository classes that provide clean database access
without business logic.
"""

from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.session_repository import SessionRepository
from src.data.repositories.attendance_repository import AttendanceRepository
from src.data.repositories.enrollment_repository import EnrollmentRepository
from src.data.repositories.analytics_repository import AnalyticsRepository

__all__ = [
    'StudentRepository',
    'SessionRepository',
    'AttendanceRepository',
    'EnrollmentRepository',
    'AnalyticsRepository'
]

