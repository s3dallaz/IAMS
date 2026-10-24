"""
Attendance Service layer for business logic and validation.

Provides application-level attendance management operations with validation,
status calculation, and business rule enforcement.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.data.repositories.attendance_repository import AttendanceRepository
from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.session_repository import SessionRepository
from src.services.session_service import SessionService, SessionStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Service-level Constants
# ============================================================================

class AttendanceStatus:
    """Attendance status constants."""
    PRESENT = 'present'
    LATE = 'late'
    ABSENT = 'absent'


# Valid attendance status values
VALID_STATUSES = {AttendanceStatus.PRESENT, AttendanceStatus.LATE, AttendanceStatus.ABSENT}


# ============================================================================
# Service-level Exceptions
# ============================================================================

class AttendanceServiceError(Exception):
    """Base exception for attendance service errors."""
    pass


class InvalidAttendanceDataError(AttendanceServiceError):
    """Raised when attendance data fails validation."""
    pass


class AttendanceNotFoundError(AttendanceServiceError):
    """Raised when a requested attendance record does not exist."""
    pass


class DuplicateAttendanceError(AttendanceServiceError):
    """Raised when attempting to create duplicate attendance."""
    pass


class SessionNotAcceptingAttendanceError(AttendanceServiceError):
    """Raised when session cannot accept attendance."""
    pass


# ============================================================================
# Attendance Service
# ============================================================================

class AttendanceService:
    """
    Service layer for attendance management.
    
    Handles validation, status calculation, business rules, and coordination
    with AttendanceRepository, SessionRepository, and StudentRepository.
    Does not contain SQL or database connection logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the attendance service.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.attendance_repo = AttendanceRepository(database_path)
        self.student_repo = StudentRepository(database_path)
        self.session_repo = SessionRepository(database_path)
        self.session_service = SessionService(database_path)
        logger.debug("AttendanceService initialized")
    
    def record_attendance(
        self,
        session_id: int,
        student_id: int,
        marked_at: Optional[str] = None,
        detection_confidence: Optional[float] = None,
        liveness_score: Optional[float] = None,
        marked_by: str = 'system',
        notes: Optional[str] = None
    ) -> int:
        """
        Record attendance for a student in a session.
        
        Business logic:
        - Validates session exists and can accept attendance
        - Validates student exists
        - Prevents duplicate attendance
        - Calculates attendance status (present vs late) based on session's late_threshold_minutes
        - Records attendance with CV metadata
        
        Args:
            session_id: Session database ID
            student_id: Student database ID
            marked_at: Timestamp when attendance was marked (ISO format). 
                       If None, uses current timestamp.
            detection_confidence: Face detection confidence score (0.0-1.0, optional)
            liveness_score: Liveness detection score (0.0-1.0, optional)
            marked_by: Who marked the attendance (default: 'system')
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created attendance record
        
        Raises:
            InvalidAttendanceDataError: If validation fails
            SessionNotAcceptingAttendanceError: If session cannot accept attendance
            DuplicateAttendanceError: If attendance already exists
        """
        # Validate session_id
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        
        # Validate student_id
        if not isinstance(student_id, int) or student_id <= 0:
            raise InvalidAttendanceDataError("student_id must be a positive integer")
        
        # Validate session exists and can accept attendance
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise InvalidAttendanceDataError(f"Session with ID {session_id} does not exist")
        
        if not self.session_service.can_accept_attendance(session_id):
            raise SessionNotAcceptingAttendanceError(
                f"Session {session_id} cannot accept attendance. "
                f"Current status: '{session['status']}'. Session must be '{SessionStatus.ACTIVE}'."
            )
        
        # Validate student exists
        student = self.student_repo.get_by_id(student_id)
        if not student:
            raise InvalidAttendanceDataError(f"Student with ID {student_id} does not exist")
        
        # Check for duplicate attendance
        if self.attendance_repo.exists(session_id, student_id):
            raise DuplicateAttendanceError(
                f"Attendance already recorded for student {student_id} in session {session_id}"
            )
        
        # Validate marked_at if provided
        if marked_at is not None:
            if not isinstance(marked_at, str):
                raise InvalidAttendanceDataError("marked_at must be a string")
            marked_at = marked_at.strip()
            if not marked_at:
                marked_at = None
            else:
                # Validate format
                try:
                    datetime.fromisoformat(marked_at.replace(' ', 'T'))
                except ValueError:
                    raise InvalidAttendanceDataError("marked_at must be a valid ISO format datetime")
        
        # If marked_at not provided, use current timestamp
        if marked_at is None:
            marked_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Validate detection_confidence if provided
        if detection_confidence is not None:
            if not isinstance(detection_confidence, (int, float)):
                raise InvalidAttendanceDataError("detection_confidence must be a number")
            if not (0.0 <= detection_confidence <= 1.0):
                raise InvalidAttendanceDataError("detection_confidence must be between 0.0 and 1.0")
        
        # Validate liveness_score if provided
        if liveness_score is not None:
            if not isinstance(liveness_score, (int, float)):
                raise InvalidAttendanceDataError("liveness_score must be a number")
            if not (0.0 <= liveness_score <= 1.0):
                raise InvalidAttendanceDataError("liveness_score must be between 0.0 and 1.0")
        
        # Validate marked_by
        if not isinstance(marked_by, str):
            raise InvalidAttendanceDataError("marked_by must be a string")
        marked_by = marked_by.strip()
        if not marked_by:
            marked_by = 'system'
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidAttendanceDataError("notes must be a string")
            notes = notes.strip()
            if not notes:
                notes = None
        
        # Calculate attendance status (present vs late)
        status = self._calculate_attendance_status(session, marked_at)
        
        # Record attendance through repository
        try:
            attendance_id = self.attendance_repo.create_attendance(
                session_id=session_id,
                student_id=student_id,
                status=status,
                marked_at=marked_at,
                detection_confidence=detection_confidence,
                liveness_score=liveness_score,
                marked_by=marked_by,
                notes=notes
            )
            logger.info(
                f"Attendance recorded: Session {session_id}, Student {student_id}, "
                f"Status {status} (ID: {attendance_id})"
            )
            return attendance_id
            
        except sqlite3.IntegrityError as e:
            # Handle race condition where attendance was created between check and insert
            if "UNIQUE constraint failed" in str(e):
                raise DuplicateAttendanceError(
                    f"Attendance already recorded for student {student_id} in session {session_id}"
                )
            # Handle foreign key violations that shouldn't happen due to prior checks
            logger.error(f"Unexpected integrity error recording attendance: {e}")
            raise InvalidAttendanceDataError(f"Failed to record attendance: {e}")
        except sqlite3.Error as e:
            logger.error(f"Database error recording attendance: {e}")
            raise InvalidAttendanceDataError(f"Failed to record attendance: {e}")
    
    def _calculate_attendance_status(self, session: Dict[str, Any], marked_at: str) -> str:
        """
        Calculate attendance status based on session start time and late threshold.
        
        Business rule:
        - If attendance marked before (start_time + late_threshold_minutes): PRESENT
        - If attendance marked at or after (start_time + late_threshold_minutes): LATE
        
        Args:
            session: Session dictionary containing start_time and late_threshold_minutes
            marked_at: Timestamp when attendance was marked (ISO format string)
        
        Returns:
            Attendance status: 'present' or 'late'
        """
        # Parse timestamps
        start_time = datetime.fromisoformat(session['start_time'].replace(' ', 'T'))
        marked_time = datetime.fromisoformat(marked_at.replace(' ', 'T'))
        
        # Get late threshold (default to 15 if not set)
        late_threshold_minutes = session.get('late_threshold_minutes', 15)
        if late_threshold_minutes is None:
            late_threshold_minutes = 15
        
        # Calculate late cutoff time
        late_cutoff = start_time + timedelta(minutes=late_threshold_minutes)
        
        # Determine status
        if marked_time < late_cutoff:
            return AttendanceStatus.PRESENT
        else:
            return AttendanceStatus.LATE
    
    def get_attendance(self, attendance_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an attendance record by ID.
        
        Args:
            attendance_id: Attendance record database ID
        
        Returns:
            Attendance record as dictionary, or None if not found
        
        Raises:
            InvalidAttendanceDataError: If attendance_id is invalid
        """
        if not isinstance(attendance_id, int) or attendance_id <= 0:
            raise InvalidAttendanceDataError("attendance_id must be a positive integer")
        
        return self.attendance_repo.get_by_id(attendance_id)
    
    def get_session_attendance(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a session.
        
        Args:
            session_id: Session database ID
        
        Returns:
            List of attendance records as dictionaries
        
        Raises:
            InvalidAttendanceDataError: If session_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        
        # Validate session exists
        if not self.session_repo.exists(session_id):
            raise InvalidAttendanceDataError(f"Session with ID {session_id} does not exist")
        
        return self.attendance_repo.get_by_session(session_id)
    
    def get_student_attendance(self, student_id: int) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a student.
        
        Args:
            student_id: Student database ID
        
        Returns:
            List of attendance records as dictionaries
        
        Raises:
            InvalidAttendanceDataError: If student_id is invalid
        """
        if not isinstance(student_id, int) or student_id <= 0:
            raise InvalidAttendanceDataError("student_id must be a positive integer")
        
        # Validate student exists
        if not self.student_repo.get_by_id(student_id):
            raise InvalidAttendanceDataError(f"Student with ID {student_id} does not exist")
        
        return self.attendance_repo.get_by_student(student_id)
    
    def get_attendance_by_session_and_student(
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
        
        Raises:
            InvalidAttendanceDataError: If session_id or student_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        
        if not isinstance(student_id, int) or student_id <= 0:
            raise InvalidAttendanceDataError("student_id must be a positive integer")
        
        return self.attendance_repo.get_by_session_and_student(session_id, student_id)
    
    def attendance_exists(self, session_id: int, student_id: int) -> bool:
        """
        Check if attendance record exists for a student in a session.
        
        Args:
            session_id: Session database ID
            student_id: Student database ID
        
        Returns:
            True if attendance exists, False otherwise
        
        Raises:
            InvalidAttendanceDataError: If session_id or student_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        
        if not isinstance(student_id, int) or student_id <= 0:
            raise InvalidAttendanceDataError("student_id must be a positive integer")
        
        return self.attendance_repo.exists(session_id, student_id)
    
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
            attendance_id: Attendance record database ID
            status: New status (optional)
            detection_confidence: New detection confidence (optional)
            liveness_score: New liveness score (optional)
            verified_by: User who verified the attendance (optional)
            notes: New notes (optional)
        
        Returns:
            Number of rows affected (0 if record not found, 1 if updated)
        
        Raises:
            InvalidAttendanceDataError: If validation fails
            AttendanceNotFoundError: If attendance record does not exist
        """
        if not isinstance(attendance_id, int) or attendance_id <= 0:
            raise InvalidAttendanceDataError("attendance_id must be a positive integer")
        
        # Verify attendance exists
        attendance = self.attendance_repo.get_by_id(attendance_id)
        if not attendance:
            raise AttendanceNotFoundError(f"Attendance record with ID {attendance_id} not found")
        
        # Validate status if provided
        if status is not None:
            if not isinstance(status, str):
                raise InvalidAttendanceDataError("status must be a string")
            status = status.strip().lower()
            if status not in VALID_STATUSES:
                raise InvalidAttendanceDataError(
                    f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
                )
        
        # Validate detection_confidence if provided
        if detection_confidence is not None:
            if not isinstance(detection_confidence, (int, float)):
                raise InvalidAttendanceDataError("detection_confidence must be a number")
            if not (0.0 <= detection_confidence <= 1.0):
                raise InvalidAttendanceDataError("detection_confidence must be between 0.0 and 1.0")
        
        # Validate liveness_score if provided
        if liveness_score is not None:
            if not isinstance(liveness_score, (int, float)):
                raise InvalidAttendanceDataError("liveness_score must be a number")
            if not (0.0 <= liveness_score <= 1.0):
                raise InvalidAttendanceDataError("liveness_score must be between 0.0 and 1.0")
        
        # Validate verified_by if provided
        if verified_by is not None:
            if not isinstance(verified_by, str):
                raise InvalidAttendanceDataError("verified_by must be a string")
            verified_by = verified_by.strip()
            if not verified_by:
                verified_by = None
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidAttendanceDataError("notes must be a string")
            notes = notes.strip()
            if not notes:
                notes = None
        
        # Update through repository
        try:
            rows_affected = self.attendance_repo.update_attendance(
                attendance_id=attendance_id,
                status=status,
                detection_confidence=detection_confidence,
                liveness_score=liveness_score,
                verified_by=verified_by,
                notes=notes
            )
            
            if rows_affected > 0:
                logger.info(f"Attendance record {attendance_id} updated")
            
            return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error updating attendance: {e}")
            raise InvalidAttendanceDataError(f"Failed to update attendance: {e}")
    
    def get_all_attendance(self) -> List[Dict[str, Any]]:
        """
        Get all attendance records.
        
        Returns:
            List of attendance records as dictionaries
        """
        return self.attendance_repo.get_all()
    
    def get_attendance_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all attendance records with a specific status.
        
        Args:
            status: Attendance status (e.g., 'present', 'late', 'absent')
        
        Returns:
            List of attendance records as dictionaries
        
        Raises:
            InvalidAttendanceDataError: If status is invalid
        """
        if not isinstance(status, str):
            raise InvalidAttendanceDataError("status must be a string")
        
        status = status.strip().lower()
        if status not in VALID_STATUSES:
            raise InvalidAttendanceDataError(
                f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        
        return self.attendance_repo.get_by_status(status)

    def _roster_cutoff(self, session_id: int) -> Optional[str]:
        """
        UTC time before which active students must have been registered to
        be expected in the session: its start time. Students registered
        later (e.g. a bulk import after the class) are not listed absent.
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise InvalidAttendanceDataError(f"Session with ID {session_id} does not exist")
        try:
            start = datetime.fromisoformat(session['start_time'].replace(' ', 'T'))
        except (TypeError, ValueError):
            return None
        # Session times are local; enrolled_at is the database's UTC default
        return start.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    def get_session_roster_counts(self, session_id: int) -> Dict[str, int]:
        """
        Expected students of a session per status.
        
        Expected students are everyone with a record in the session plus
        every active student registered before the session started (there
        are no class rosters yet). Students without a record are absent.
        
        Returns:
            {'present': n, 'late': n, 'absent': n, 'expected': total}
        """
        counts = self.attendance_repo.get_session_roster_counts(session_id, self._roster_cutoff(session_id))
        result = {status: counts.get(status, 0) for status in
                  (AttendanceStatus.PRESENT, AttendanceStatus.LATE, AttendanceStatus.ABSENT)}
        result['expected'] = sum(counts.values())
        return result
    
    def get_session_roster(
        self,
        session_id: int,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Expected students of a session with their status (see
        get_session_roster_counts), sorted Present -> Late -> Absent and by
        name within each status.
        
        Args:
            session_id: Session database ID
            status: Only students with this status
            search: Part of a student's name or student ID
            limit: Maximum number of rows
        
        Raises:
            InvalidAttendanceDataError: If an argument is invalid
        """
        cutoff = self._roster_cutoff(session_id)
        if status is not None:
            status = status.strip().lower()
            if status not in VALID_STATUSES:
                raise InvalidAttendanceDataError(
                    f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
                )
        search = (search or "").strip() or None
        return self.attendance_repo.get_session_roster(
            session_id, cutoff, status=status, search=search, limit=limit
        )
    
    def get_session_records_with_students(
        self,
        session_id: int,
        newest_first: bool = False,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Attendance records of a session with each student's ID
        ('student_code') and name ('student_name'), in one query.
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        return self.attendance_repo.get_session_records_with_students(
            session_id, newest_first=newest_first, limit=limit
        )
    
    def get_session_status_counts(self, session_id: int) -> Dict[str, int]:
        """Recorded attendance of a session per status ({status: count})."""
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        return self.attendance_repo.get_session_status_counts(session_id)
