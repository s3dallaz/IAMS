"""
Service layer for the attendance system.

Contains business logic and validation for application features.
"""

from src.services.student_service import StudentService
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.real_time_attendance_service import (
    RealTimeAttendanceManager,
    RealTimeAttendanceService,
    RealTimeAttendanceResult,
    FaceProcessingResult,
    RealTimeAttendanceError
)

__all__ = [
    'StudentService',
    'SessionService',
    'SessionStatus',
    'AttendanceService',
    'AttendanceStatus',
    'RealTimeAttendanceManager',
    'RealTimeAttendanceService',
    'RealTimeAttendanceResult',
    'FaceProcessingResult',
    'RealTimeAttendanceError'
]

