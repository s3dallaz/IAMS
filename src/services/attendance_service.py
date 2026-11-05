"""
Attendance Service layer for business logic and validation.

Provides application-level attendance management operations with validation,
status calculation, and business rule enforcement.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

import config
from src.data.repositories.attendance_repository import AttendanceRepository, MANUAL_MARKER
from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.session_repository import SessionRepository
from src.services.session_service import SessionService, SessionStatus
from src.services.activity_log import ActivityLog, Category
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


def _valid_time(value: Optional[str], name: str) -> Optional[str]:
    """A local 'YYYY-MM-DD HH:MM:SS' time, or None for an empty value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidAttendanceDataError(f"{name} must be a string")
    value = value.strip()
    if not value:
        return None
    try:
        datetime.fromisoformat(value.replace(' ', 'T'))
    except ValueError:
        raise InvalidAttendanceDataError(f"{name} must be a valid ISO format datetime")
    return value


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
        self.activity = ActivityLog(database_path)
        logger.debug("AttendanceService initialized")

    def _log_manual(self, action: str, session_id: int, student_id: int, what: str) -> None:
        """Activity log for a change made by hand: '<what> <student> in session #N "name"'."""
        student = self.student_repo.get_by_id(student_id) or {}
        session = self.session_repo.get_by_id(session_id) or {}
        label = f"{student.get('name', '?')} ({student.get('student_id', '?')})"
        self.activity.record(
            Category.ATTENDANCE, action,
            f"{what.replace('{student}', label)} in session #{session_id} \"{session.get('session_name', '?')}\"",
            target_type='student', target_id=student_id, session_id=session_id)
    
    def record_attendance(
        self,
        session_id: int,
        student_id: int,
        marked_at: Optional[str] = None,
        detection_confidence: Optional[float] = None,
        liveness_score: Optional[float] = None,
        marked_by: str = 'system',
        notes: Optional[str] = None,
        arrived_at: Optional[str] = None,
        line_since: Optional[str] = None
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
            arrived_at: When the camera first recognised the student (local
                'YYYY-MM-DD HH:MM:SS', optional). Present / late is judged by
                it when it is earlier than marked_at.
            line_since: When the line at the camera the student was in began
                (optional). If that line was already there at the late time,
                the student stays present (config.LATE_LINE_GRACE_MINUTES).

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
        
        # Calculate attendance status (present vs late) from the time the
        # student arrived, which may be earlier than the mark itself
        judged_at, line_used = self._arrival_time(session, marked_at,
                                                  _valid_time(arrived_at, 'arrived_at'),
                                                  _valid_time(line_since, 'line_since'))
        status = self._calculate_attendance_status(session, judged_at)
        if line_used:
            line_note = f"In the line at the camera since {judged_at[11:19]}"
            notes = f"{notes}; {line_note}" if notes else line_note

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
                notes=notes,
                arrived_at=judged_at if judged_at != marked_at else None
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
    
    @staticmethod
    def _late_cutoff(session: Dict[str, Any]) -> datetime:
        """Start time + late threshold (default 15 minutes)."""
        start_time = datetime.fromisoformat(session['start_time'].replace(' ', 'T'))
        late_threshold_minutes = session.get('late_threshold_minutes', 15)
        if late_threshold_minutes is None:
            late_threshold_minutes = 15
        return start_time + timedelta(minutes=late_threshold_minutes)

    def _arrival_time(self, session: Dict[str, Any], marked_at: str, arrived_at: Optional[str],
                      line_since: Optional[str]) -> tuple:
        """
        The time present / late is judged by, and whether the line rule gave it.

        - arrived_at (first seen by the camera) counts when it is earlier than
          marked_at; a later one is ignored.
        - A line that began before the late time and that the student reached
          at most config.LATE_LINE_GRACE_MINUTES after it: its start time counts.
        """
        def parse(value: str) -> datetime:
            return datetime.fromisoformat(value.replace(' ', 'T'))

        judged_at = marked_at
        if arrived_at is not None and parse(arrived_at) < parse(marked_at):
            judged_at = arrived_at

        grace = timedelta(minutes=config.LATE_LINE_GRACE_MINUTES)
        if line_since is not None and grace > timedelta(0):
            cutoff = self._late_cutoff(session)
            arrival, line_start = parse(judged_at), parse(line_since)
            if line_start < cutoff <= arrival and arrival - cutoff <= grace:
                return line_since, True
        return judged_at, False

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
        marked_time = datetime.fromisoformat(marked_at.replace(' ', 'T'))
        late_cutoff = self._late_cutoff(session)

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

    def _check_roster_session(self, session_id: int) -> None:
        """Validate the session a roster is asked for."""
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        if not self.session_repo.exists(session_id):
            raise InvalidAttendanceDataError(f"Session with ID {session_id} does not exist")
    
    def get_session_roster_counts(self, session_id: int) -> Dict[str, int]:
        """
        Expected students of a session per status.
        
        Expected students are everyone with a record in the session plus
        every active student registered before the session started (there
        are no class rosters yet), so students registered later - e.g. a bulk
        import after the class - are not listed absent. Students without a
        record are absent.
        
        Returns:
            {'present': n, 'late': n, 'absent': n, 'expected': total}
        """
        self._check_roster_session(session_id)
        counts = self.attendance_repo.get_session_roster_counts(session_id)
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
        self._check_roster_session(session_id)
        if status is not None:
            status = status.strip().lower()
            if status not in VALID_STATUSES:
                raise InvalidAttendanceDataError(
                    f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
                )
        search = (search or "").strip() or None
        return self.attendance_repo.get_session_roster(
            session_id, status=status, search=search, limit=limit
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

    def set_left_early(self, session_id: int, student_id: int, left_early: bool,
                       changed_by: str = MANUAL_MARKER) -> None:
        """
        Set or clear 'left early' by hand (e.g. the student had a good reason
        to leave, or the instructor saw them leave). Only for a student marked
        present / late in the session.

        Raises:
            InvalidAttendanceDataError: The student is not marked in this session
        """
        existing = self.attendance_repo.get_by_session_and_student(session_id, student_id)
        if not existing or existing['status'] not in (AttendanceStatus.PRESENT, AttendanceStatus.LATE):
            raise InvalidAttendanceDataError("Only a student marked present or late can be set to 'left early'.")
        if bool(existing.get('left_early')) == bool(left_early):
            return
        self.attendance_repo.set_left_early(existing['id'], left_early, verified_by=changed_by)
        logger.info(f"Session {session_id}: student {student_id} left_early={left_early} by {changed_by}")
        self._log_manual('left_early_set' if left_early else 'left_early_cleared', session_id, student_id,
                         "Marked {student} as 🚪 Left early" if left_early else "Cleared 🚪 Left early for {student}")

    def count_left_early(self, session_id: int) -> int:
        """How many present / late students of the session are flagged 'left early'."""
        return self.attendance_repo.count_left_early(session_id)

    def set_attendance_manually(
        self,
        session_id: int,
        student_id: int,
        status: str,
        changed_by: str = MANUAL_MARKER
    ) -> Optional[int]:
        """
        Set a student's attendance in a session by hand, e.g. when the camera
        did not recognise a student who was there, or to excuse a late arrival.
        
        - present / late: creates the student's record (marked_by = changed_by)
          or changes the status of the existing one (verified_by = changed_by)
        - absent: removes the student's record (no record = absent)
        
        Works for active and closed sessions (not before a session starts).
        A status set by hand is kept when the session's start time or late
        threshold is edited later (it is not re-graded).
        
        Args:
            session_id: Session database ID
            student_id: Student database ID
            status: 'present', 'late' or 'absent'
            changed_by: Who made the change (stored on the record)
        
        Returns:
            The attendance record ID, or None for 'absent'
        
        Raises:
            InvalidAttendanceDataError: If an argument is invalid
            SessionNotAcceptingAttendanceError: If the session has not started
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidAttendanceDataError("session_id must be a positive integer")
        if not isinstance(student_id, int) or student_id <= 0:
            raise InvalidAttendanceDataError("student_id must be a positive integer")
        status = (status or "").strip().lower()
        if status not in VALID_STATUSES:
            raise InvalidAttendanceDataError(
                f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise InvalidAttendanceDataError(f"Session with ID {session_id} does not exist")
        if session['status'] == SessionStatus.SCHEDULED:
            raise SessionNotAcceptingAttendanceError(
                "This session has not started yet; attendance can be set once it is active."
            )
        if not self.student_repo.get_by_id(student_id):
            raise InvalidAttendanceDataError(f"Student with ID {student_id} does not exist")
        
        existing = self.attendance_repo.get_by_session_and_student(session_id, student_id)
        try:
            if status == AttendanceStatus.ABSENT:
                if existing:
                    self.attendance_repo.delete_attendance(existing['id'])
                    logger.info(f"Session {session_id}: student {student_id} set to absent by {changed_by} "
                                f"(was {existing['status']})")
                    self._log_manual('attendance_removed', session_id, student_id,
                                     f"Set {{student}} to absent (was {existing['status']})")
                return None
            
            if existing:
                if existing['status'] != status:
                    self.attendance_repo.update_attendance(
                        existing['id'], status=status, verified_by=changed_by,
                        notes=f"Changed by hand from {existing['status']} to {status}"
                    )
                    logger.info(f"Session {session_id}: student {student_id} changed to {status} by {changed_by}")
                    self._log_manual('attendance_changed', session_id, student_id,
                                     f"Changed {{student}} from {existing['status']} to {status}")
                return existing['id']
            
            attendance_id = self.attendance_repo.create_attendance(
                session_id=session_id,
                student_id=student_id,
                status=status,
                marked_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                marked_by=changed_by,
                notes="Marked by hand"
            )
            logger.info(f"Session {session_id}: student {student_id} marked {status} by {changed_by}")
            self._log_manual('attendance_marked', session_id, student_id, f"Marked {{student}} {status} by hand")
            return attendance_id
        except sqlite3.IntegrityError:
            # The camera marked the student a moment ago: change that record instead
            return self.set_attendance_manually(session_id, student_id, status, changed_by)
        except sqlite3.Error as e:
            logger.error(f"Database error setting attendance by hand: {e}")
            raise InvalidAttendanceDataError(f"Failed to set attendance: {e}")


def marked_by_label(marked_by: Optional[str], verified_by: Optional[str] = None) -> str:
    """Who marked a record, in plain words ('Camera', 'Instructor', 'Camera, changed by Instructor')."""
    names = {'cv_realtime': 'Camera', MANUAL_MARKER: 'Instructor'}
    if not marked_by:
        return ""
    label = names.get(marked_by, marked_by)
    if verified_by:
        label += f", changed by {names.get(verified_by, verified_by)}"
    return label
