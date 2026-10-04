"""
Session Service layer for business logic and validation.

Provides application-level session management operations with validation,
lifecycle management, and business rule enforcement.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from src.data.repositories.session_repository import SessionRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Service-level Constants
# ============================================================================

class SessionStatus:
    """Session status constants."""
    SCHEDULED = 'scheduled'
    ACTIVE = 'active'
    CLOSED = 'closed'


# Valid session status values
VALID_STATUSES = {SessionStatus.SCHEDULED, SessionStatus.ACTIVE, SessionStatus.CLOSED}

# Valid session transitions
VALID_TRANSITIONS = {
    SessionStatus.SCHEDULED: {SessionStatus.ACTIVE, SessionStatus.CLOSED},
    SessionStatus.ACTIVE: {SessionStatus.CLOSED},
    SessionStatus.CLOSED: set()  # Closed sessions cannot transition
}


# ============================================================================
# Service-level Exceptions
# ============================================================================

class SessionServiceError(Exception):
    """Base exception for session service errors."""
    pass


class InvalidSessionDataError(SessionServiceError):
    """Raised when session data fails validation."""
    pass


class SessionNotFoundError(SessionServiceError):
    """Raised when a requested session does not exist."""
    pass


class InvalidSessionStateError(SessionServiceError):
    """Raised when attempting an invalid session state transition."""
    pass


# ============================================================================
# Session Service
# ============================================================================

class SessionService:
    """
    Service layer for session management.
    
    Handles validation, session lifecycle, business rules, and coordination
    with SessionRepository. Does not contain SQL or database connection logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the session service.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.repository = SessionRepository(database_path)
        logger.debug("SessionService initialized")
    
    def create_session(
        self,
        session_name: str,
        start_time: str,
        course_code: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        status: str = SessionStatus.SCHEDULED,
        created_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new session with validation.
        
        Args:
            session_name: Name of the session (e.g., "CS101 Lecture")
            start_time: Session start time (ISO format string)
            course_code: Course identifier (optional)
            end_time: Session end time (ISO format string, optional)
            late_threshold_minutes: Minutes after start_time considered late (default: 15)
            status: Initial session status (default: 'scheduled')
            created_by: User who created the session (optional)
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created session
        
        Raises:
            InvalidSessionDataError: If validation fails
        """
        # Validate session_name
        if not isinstance(session_name, str):
            raise InvalidSessionDataError("session_name must be a string")
        
        session_name = session_name.strip()
        if not session_name:
            raise InvalidSessionDataError("session_name cannot be empty")
        
        # Validate start_time
        if not isinstance(start_time, str):
            raise InvalidSessionDataError("start_time must be a string")
        
        start_time = start_time.strip()
        if not start_time:
            raise InvalidSessionDataError("start_time cannot be empty")
        
        # Validate start_time format
        try:
            datetime.fromisoformat(start_time.replace(' ', 'T'))
        except ValueError:
            raise InvalidSessionDataError("start_time must be a valid ISO format datetime")
        
        # Validate course_code if provided
        if course_code is not None:
            if not isinstance(course_code, str):
                raise InvalidSessionDataError("course_code must be a string")
            course_code = course_code.strip()
            if not course_code:
                course_code = None
        
        # Validate end_time if provided
        if end_time is not None:
            if not isinstance(end_time, str):
                raise InvalidSessionDataError("end_time must be a string")
            end_time = end_time.strip()
            if not end_time:
                end_time = None
            else:
                # Validate end_time format
                try:
                    datetime.fromisoformat(end_time.replace(' ', 'T'))
                except ValueError:
                    raise InvalidSessionDataError("end_time must be a valid ISO format datetime")
                
                # Validate end_time is after start_time
                start_dt = datetime.fromisoformat(start_time.replace(' ', 'T'))
                end_dt = datetime.fromisoformat(end_time.replace(' ', 'T'))
                if end_dt <= start_dt:
                    raise InvalidSessionDataError("end_time must be after start_time")
        
        # Validate late_threshold_minutes
        if late_threshold_minutes is not None:
            if not isinstance(late_threshold_minutes, int):
                raise InvalidSessionDataError("late_threshold_minutes must be an integer")
            if late_threshold_minutes < 0:
                raise InvalidSessionDataError("late_threshold_minutes cannot be negative")
        else:
            # Use default from database schema
            late_threshold_minutes = 15
        
        # Validate status
        if not isinstance(status, str):
            raise InvalidSessionDataError("status must be a string")
        
        status = status.strip().lower()
        if status not in VALID_STATUSES:
            raise InvalidSessionDataError(
                f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        
        # Validate created_by if provided
        if created_by is not None:
            if not isinstance(created_by, str):
                raise InvalidSessionDataError("created_by must be a string")
            created_by = created_by.strip()
            if not created_by:
                created_by = None
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidSessionDataError("notes must be a string")
            notes = notes.strip()
            if not notes:
                notes = None
        
        # Create session through repository
        try:
            session_id = self.repository.create_session(
                session_name=session_name,
                start_time=start_time,
                course_code=course_code,
                end_time=end_time,
                late_threshold_minutes=late_threshold_minutes,
                status=status,
                created_by=created_by,
                notes=notes
            )
            logger.info(f"Session created successfully: {session_name} (ID: {session_id})")
            return session_id
        except sqlite3.Error as e:
            logger.error(f"Database error creating session: {e}")
            raise InvalidSessionDataError(f"Failed to create session: {e}")
    
    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session database ID
        
        Returns:
            Session record as dictionary, or None if not found
        
        Raises:
            InvalidSessionDataError: If session_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        return self.repository.get_by_id(session_id)
    
    def get_all_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions, optionally filtered by status.
        
        Args:
            status: Filter by status (e.g., 'scheduled', 'active', 'closed')
        
        Returns:
            List of session records as dictionaries
        
        Raises:
            InvalidSessionDataError: If status is invalid
        """
        if status is not None:
            if not isinstance(status, str):
                raise InvalidSessionDataError("status must be a string")
            status = status.strip().lower()
            if status not in VALID_STATUSES:
                raise InvalidSessionDataError(
                    f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
                )
        
        return self.repository.get_all(status=status)
    
    def get_sessions_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific course.
        
        Args:
            course_code: Course identifier
        
        Returns:
            List of session records as dictionaries
        
        Raises:
            InvalidSessionDataError: If course_code is invalid
        """
        if not isinstance(course_code, str):
            raise InvalidSessionDataError("course_code must be a string")
        
        course_code = course_code.strip()
        if not course_code:
            raise InvalidSessionDataError("course_code cannot be empty")
        
        return self.repository.get_by_course(course_code)
    
    def activate_session(self, session_id: int) -> int:
        """
        Activate a session (transition to 'active' status).
        
        Args:
            session_id: Session database ID
        
        Returns:
            Number of rows affected (1 if activated, 0 if session not found)
        
        Raises:
            InvalidSessionDataError: If session_id is invalid
            SessionNotFoundError: If session does not exist
            InvalidSessionStateError: If session cannot be activated
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        # Get existing session
        session = self.repository.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session with ID {session_id} not found")
        
        current_status = session['status']
        
        # Validate transition
        if SessionStatus.ACTIVE not in VALID_TRANSITIONS.get(current_status, set()):
            raise InvalidSessionStateError(
                f"Cannot activate session with status '{current_status}'. "
                f"Session must be in '{SessionStatus.SCHEDULED}' state."
            )
        
        # Update status to active
        rows_affected = self.repository.update_session(session_id, status=SessionStatus.ACTIVE)
        
        if rows_affected > 0:
            logger.info(f"Session {session_id} activated")
        
        return rows_affected
    
    def close_session(self, session_id: int) -> int:
        """
        Close a session (transition to 'closed' status).
        
        Args:
            session_id: Session database ID
        
        Returns:
            Number of rows affected (1 if closed, 0 if session not found)
        
        Raises:
            InvalidSessionDataError: If session_id is invalid
            SessionNotFoundError: If session does not exist
            InvalidSessionStateError: If session is already closed
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        # Get existing session
        session = self.repository.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session with ID {session_id} not found")
        
        current_status = session['status']
        
        # Check if already closed
        if current_status == SessionStatus.CLOSED:
            raise InvalidSessionStateError("Session is already closed")
        
        # Close session through repository (sets status and closed_at)
        rows_affected = self.repository.close_session(session_id)
        
        if rows_affected > 0:
            logger.info(f"Session {session_id} closed")
        
        return rows_affected
    
    def can_accept_attendance(self, session_id: int) -> bool:
        """
        Check if a session can accept attendance records.
        
        Business rule: Only active sessions can accept attendance.
        
        Args:
            session_id: Session database ID
        
        Returns:
            True if session can accept attendance, False otherwise
        
        Raises:
            InvalidSessionDataError: If session_id is invalid
            SessionNotFoundError: If session does not exist
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        session = self.repository.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session with ID {session_id} not found")
        
        return session['status'] == SessionStatus.ACTIVE
    
    def session_exists(self, session_id: int) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Session database ID
        
        Returns:
            True if session exists, False otherwise
        
        Raises:
            InvalidSessionDataError: If session_id is invalid
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        return self.repository.exists(session_id)
    
    def update_session(
        self,
        session_id: int,
        session_name: Optional[str] = None,
        course_code: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Update a session's information.
        
        Note: Status cannot be updated through this method. 
        Use activate_session() or close_session() for status transitions.
        
        Args:
            session_id: Session database ID
            session_name: New session name (optional)
            course_code: New course code (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            late_threshold_minutes: New late threshold (optional)
            notes: New notes (optional)
        
        Returns:
            Number of rows affected (1 if updated, 0 if session not found)
        
        Raises:
            InvalidSessionDataError: If validation fails
            SessionNotFoundError: If session does not exist
        """
        if not isinstance(session_id, int) or session_id <= 0:
            raise InvalidSessionDataError("session_id must be a positive integer")
        
        # Verify session exists
        session = self.repository.get_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session with ID {session_id} not found")
        
        # Validate session_name if provided
        if session_name is not None:
            if not isinstance(session_name, str):
                raise InvalidSessionDataError("session_name must be a string")
            session_name = session_name.strip()
            if not session_name:
                raise InvalidSessionDataError("session_name cannot be empty")
        
        # Validate course_code if provided
        if course_code is not None:
            if not isinstance(course_code, str):
                raise InvalidSessionDataError("course_code must be a string")
            course_code = course_code.strip()
            if not course_code:
                course_code = None
        
        # Validate start_time if provided
        if start_time is not None:
            if not isinstance(start_time, str):
                raise InvalidSessionDataError("start_time must be a string")
            start_time = start_time.strip()
            if not start_time:
                raise InvalidSessionDataError("start_time cannot be empty")
            try:
                datetime.fromisoformat(start_time.replace(' ', 'T'))
            except ValueError:
                raise InvalidSessionDataError("start_time must be a valid ISO format datetime")
        
        # Validate end_time if provided
        if end_time is not None:
            if not isinstance(end_time, str):
                raise InvalidSessionDataError("end_time must be a string")
            end_time = end_time.strip()
            if not end_time:
                end_time = None
            else:
                try:
                    datetime.fromisoformat(end_time.replace(' ', 'T'))
                except ValueError:
                    raise InvalidSessionDataError("end_time must be a valid ISO format datetime")
        
        # Validate late_threshold_minutes if provided
        if late_threshold_minutes is not None:
            if not isinstance(late_threshold_minutes, int):
                raise InvalidSessionDataError("late_threshold_minutes must be an integer")
            if late_threshold_minutes < 0:
                raise InvalidSessionDataError("late_threshold_minutes cannot be negative")
        
        # Validate notes if provided
        if notes is not None:
            if not isinstance(notes, str):
                raise InvalidSessionDataError("notes must be a string")
            notes = notes.strip()
            if not notes:
                notes = None
        
        # Update through repository
        try:
            rows_affected = self.repository.update_session(
                session_id=session_id,
                session_name=session_name,
                course_code=course_code,
                start_time=start_time,
                end_time=end_time,
                late_threshold_minutes=late_threshold_minutes,
                notes=notes
            )
            
            if rows_affected > 0:
                logger.info(f"Session {session_id} updated")
            
            return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error updating session: {e}")
            raise InvalidSessionDataError(f"Failed to update session: {e}")
