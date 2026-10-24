"""
Session Repository for database persistence operations.

Provides clean database access for the sessions table without business logic.
"""

import sqlite3
from typing import Optional, List, Dict, Any, Iterable, Tuple
from pathlib import Path

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """
    Repository for session persistence operations.
    
    Handles database access for the sessions table.
    Does not contain business logic.
    """
    
    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the session repository.
        
        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path
    
    def create_session(
        self,
        session_name: str,
        start_time: str,
        course_code: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        status: str = 'scheduled',
        created_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new session in the database.
        
        Args:
            session_name: Name of the session (e.g., "CS101 Lecture")
            start_time: Session start time (ISO format string)
            course_code: Course identifier (optional)
            end_time: Session end time (ISO format string, optional)
            late_threshold_minutes: Minutes after start_time considered late (optional)
            status: Session status (default: 'scheduled')
            created_by: User who created the session (optional)
            notes: Additional notes (optional)
        
        Returns:
            Database ID of the newly created session
        
        Raises:
            sqlite3.Error: On database errors
        """
        query = """
            INSERT INTO sessions (
                session_name, course_code, start_time, end_time,
                late_threshold_minutes, status, created_by, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (session_name, course_code, start_time, end_time,
                     late_threshold_minutes, status, created_by, notes)
                )
                session_id = cursor.lastrowid
                logger.info(f"Created session: {session_name} (ID: {session_id})")
                return session_id
        except sqlite3.Error as e:
            logger.error(f"Database error creating session: {e}")
            raise
    
    def get_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session database primary key
        
        Returns:
            Session record as dictionary, or None if not found
        """
        query = "SELECT * FROM sessions WHERE id = ?"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error getting session by id {session_id}: {e}")
            raise
    
    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all sessions, optionally filtered by status.
        
        Args:
            status: Filter by status (e.g., 'scheduled', 'active', 'closed'). If None, returns all.
        
        Returns:
            List of session records as dictionaries
        """
        if status:
            query = "SELECT * FROM sessions WHERE status = ? ORDER BY start_time DESC"
            params = (status,)
        else:
            query = "SELECT * FROM sessions ORDER BY start_time DESC"
            params = ()
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting all sessions: {e}")
            raise
    
    def get_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific course.
        
        Args:
            course_code: Course identifier
        
        Returns:
            List of session records as dictionaries
        """
        query = "SELECT * FROM sessions WHERE course_code = ? ORDER BY start_time DESC"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (course_code,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Database error getting sessions for course {course_code}: {e}")
            raise
    
    def update_session(
        self,
        session_id: int,
        session_name: Optional[str] = None,
        course_code: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        late_threshold_minutes: Optional[int] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        clear_end_time: bool = False
    ) -> int:
        """
        Update a session's information.

        Args:
            session_id: Session database primary key
            session_name: New session name (optional)
            course_code: New course code (optional)
            start_time: New start time (optional)
            end_time: New end time (optional)
            late_threshold_minutes: New late threshold (optional)
            status: New status (optional)
            notes: New notes (optional)
            clear_end_time: Remove the end time (end_time=None means "unchanged")

        Returns:
            Number of rows affected (0 if session not found, 1 if updated)
        
        Raises:
            sqlite3.Error: On database errors
        """
        # Build dynamic UPDATE query based on provided fields
        updates = []
        params = []
        
        if session_name is not None:
            updates.append("session_name = ?")
            params.append(session_name)
        
        if course_code is not None:
            updates.append("course_code = ?")
            params.append(course_code)
        
        if start_time is not None:
            updates.append("start_time = ?")
            params.append(start_time)
        
        if clear_end_time:
            updates.append("end_time = NULL")
        elif end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)

        if late_threshold_minutes is not None:
            updates.append("late_threshold_minutes = ?")
            params.append(late_threshold_minutes)
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if not updates:
            # No fields to update
            logger.warning(f"No fields provided to update for session {session_id}")
            return 0
        
        query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"
        params.append(session_id)
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Updated session ID {session_id}")
                else:
                    logger.warning(f"Session ID {session_id} not found for update")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error updating session: {e}")
            raise
    
    def close_session(self, session_id: int) -> int:
        """
        Close a session (set status = 'closed', closed_at = current timestamp).
        
        Args:
            session_id: Session database primary key
        
        Returns:
            Number of rows affected (0 if session not found, 1 if closed)
        """
        query = """
            UPDATE sessions
            SET status = 'closed', closed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                rows_affected = cursor.rowcount
                
                if rows_affected > 0:
                    logger.info(f"Closed session ID {session_id}")
                else:
                    logger.warning(f"Session ID {session_id} not found for closing")
                
                return rows_affected
        except sqlite3.Error as e:
            logger.error(f"Database error closing session: {e}")
            raise

    def transition_status(
        self,
        session_id: int,
        from_statuses: Iterable[str],
        to_status: str,
        start_time: Optional[str] = None
    ) -> int:
        """
        Move a session to `to_status` only if it is still in one of `from_statuses`.

        A compare-and-set: when the scheduler and a user (or two users) change
        the same session at once, only the first change applies, so a closed
        session can never be re-opened by a late "start".

        Args:
            session_id: Session database primary key
            from_statuses: Statuses the session may currently have
            to_status: New status ('closed' also sets closed_at)
            start_time: Also set this start time, in the same update (optional)
        
        Returns:
            1 if the session moved, 0 if it was missing or in another status
        """
        from_statuses = list(from_statuses)
        placeholders = ", ".join("?" for _ in from_statuses)
        sets, params = ["status = ?"], [to_status]
        if to_status == 'closed':
            sets.append("closed_at = CURRENT_TIMESTAMP")
        if start_time is not None:
            sets.append("start_time = ?")
            params.append(start_time)
        query = f"UPDATE sessions SET {', '.join(sets)} WHERE id = ? AND status IN ({placeholders})"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (*params, session_id, *from_statuses))
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error changing status of session {session_id}: {e}")
            raise

    def apply_due_transitions(self, now: str, auto_close: Optional[bool] = None) -> Dict[str, List[int]]:
        """
        Start scheduled sessions whose start time has come and close sessions
        whose end time has passed, in one transaction.

        Times are compared with SQLite's datetime() so 'YYYY-MM-DD HH:MM:SS'
        and 'YYYY-MM-DDTHH:MM:SS' values compare correctly. A scheduled session
        whose whole time window passed (e.g. while the app was not running) is
        closed directly; it never accepted attendance.

        Args:
            now: Current local time, 'YYYY-MM-DD HH:MM:SS'
            auto_close: Close sessions after their end time. None = use the
                'auto_close_sessions' setting (default on)

        Returns:
            {'started': [ids], 'closed': [ids]}
        """
        due_to_close = """
            SELECT id FROM sessions
            WHERE status IN ('scheduled', 'active')
              AND end_time IS NOT NULL AND datetime(end_time) <= datetime(?)
        """
        due_to_start = """
            SELECT id FROM sessions
            WHERE status = 'scheduled' AND datetime(start_time) <= datetime(?)
              AND (end_time IS NULL OR datetime(end_time) > datetime(?))
        """
        changed: Dict[str, List[int]] = {'started': [], 'closed': []}

        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                if auto_close is None:
                    row = cursor.execute("SELECT value FROM settings WHERE key = 'auto_close_sessions'").fetchone()
                    auto_close = row is None or str(row[0]).strip().lower() in ('1', 'true', 'yes', 'on')
                # Cheap check first: no write lock at all when nothing is due
                check = f"SELECT EXISTS({due_to_start})" + (f" OR EXISTS({due_to_close})" if auto_close else "")
                check_params = (now, now, now) if auto_close else (now, now)
                if not cursor.execute(check, check_params).fetchone()[0]:
                    return changed

                # IMMEDIATE: select + update see one consistent state
                cursor.execute("BEGIN IMMEDIATE")
                if auto_close:
                    changed['closed'] = [row[0] for row in cursor.execute(due_to_close, (now,))]
                    if changed['closed']:
                        ids = ", ".join("?" for _ in changed['closed'])
                        cursor.execute(
                            f"UPDATE sessions SET status = 'closed', closed_at = CURRENT_TIMESTAMP "
                            f"WHERE id IN ({ids})", changed['closed']
                        )
                changed['started'] = [row[0] for row in cursor.execute(due_to_start, (now, now))]
                if changed['started']:
                    ids = ", ".join("?" for _ in changed['started'])
                    cursor.execute(f"UPDATE sessions SET status = 'active' WHERE id IN ({ids})", changed['started'])
                return changed
        except sqlite3.Error as e:
            logger.error(f"Database error applying scheduled session changes: {e}")
            raise

    def get_overview(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Sessions with their present / late counts, newest first, in one query.

        Args:
            status: Filter by status. If None, returns all.

        Returns:
            Session records with 'present_count', 'late_count' and 'marked_count'
        """
        where = "WHERE sess.status = ?" if status else ""
        query = f"""
            SELECT sess.*,
                   COUNT(a.id) AS marked_count,
                   COALESCE(SUM(a.status = 'present'), 0) AS present_count,
                   COALESCE(SUM(a.status = 'late'), 0) AS late_count
            FROM sessions sess
            LEFT JOIN attendance_records a ON a.session_id = sess.id
            {where}
            GROUP BY sess.id
            ORDER BY sess.start_time DESC, sess.id DESC
        """

        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (status,) if status else ())
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error getting session overview: {e}")
            raise

    def get_status_signature(self) -> Tuple[int, ...]:
        """
        A tiny fingerprint of the sessions table: changes whenever a session
        is added, removed or changes status. Lets an open page notice a
        scheduled start / auto close without reloading every session.
        """
        query = """
            SELECT COUNT(*), COALESCE(MAX(id), 0),
                   COALESCE(SUM(status = 'scheduled'), 0),
                   COALESCE(SUM(status = 'active'), 0),
                   COALESCE(SUM(status = 'closed'), 0)
            FROM sessions
        """

        try:
            with get_db_connection(self.database_path) as conn:
                return tuple(conn.execute(query).fetchone())
        except sqlite3.Error as e:
            logger.error(f"Database error reading session status signature: {e}")
            raise

    def find_overlapping(
        self,
        start_time: str,
        end_time: Optional[str],
        course_code: Optional[str] = None,
        exclude_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Non-closed sessions whose time window overlaps [start_time, end_time).

        A session without an end time is treated as a single instant at its
        start; sessions starting at the same time always overlap, while
        back-to-back sessions (one ends when the next starts) do not.
        Optionally limited to one course code.
        """
        query = """
            SELECT * FROM sessions
            WHERE status != 'closed'
              AND ((datetime(start_time) < datetime(?)
                    AND datetime(?) < datetime(COALESCE(end_time, start_time)))
                   OR datetime(start_time) = datetime(?))
        """
        params: List[Any] = [end_time or start_time, start_time, start_time]
        if course_code:
            query += " AND course_code = ?"
            params.append(course_code)
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)
        query += " ORDER BY start_time"

        try:
            with get_db_connection(self.database_path) as conn:
                return [dict(row) for row in conn.execute(query, params).fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database error finding overlapping sessions: {e}")
            raise

    def delete_session(self, session_id: int) -> int:
        """
        Delete a session. Its attendance records are deleted with it
        (ON DELETE CASCADE); unknown-face logs keep their rows (SET NULL).

        Returns:
            Number of rows affected (0 if session not found, 1 if deleted)
        """
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                if cursor.rowcount:
                    logger.info(f"Deleted session ID {session_id}")
                return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Database error deleting session {session_id}: {e}")
            raise

    def exists(self, session_id: int) -> bool:
        """
        Check if a session exists by ID.
        
        Args:
            session_id: Session database primary key
        
        Returns:
            True if session exists, False otherwise
        """
        query = "SELECT 1 FROM sessions WHERE id = ? LIMIT 1"
        
        try:
            with get_db_connection(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (session_id,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error checking session existence: {e}")
            raise
