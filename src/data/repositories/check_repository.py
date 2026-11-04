"""
Check Repository for database persistence operations.

Provides database access for check-out and spot checks (attendance_checks)
and who was seen during them (check_marks), without business logic.
Times are local 'YYYY-MM-DD HH:MM:SS' text, like session and attendance times.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.database.connection import get_db_connection

_CHECK_SELECT = """
    SELECT c.*, (SELECT COUNT(*) FROM check_marks m WHERE m.check_id = c.id) AS marked_count
    FROM attendance_checks c
"""


class CheckRepository:
    """Repository for the attendance_checks and check_marks tables."""

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the check repository.

        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path

    def _fetch_all(self, query: str, params: Any = ()) -> List[Dict[str, Any]]:
        with get_db_connection(self.database_path) as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _execute(self, query: str, params: Any = ()) -> int:
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, params).rowcount

    # -- checks ----------------------------------------------------------------------

    def create_check(self, session_id: int, kind: str, starts_at: str, ends_at: str, status: str,
                     is_random: bool = False, created_by: Optional[str] = None,
                     created_by_user_id: Optional[int] = None, close_session_after: bool = False) -> int:
        query = """
            INSERT INTO attendance_checks (session_id, kind, starts_at, ends_at, status, is_random,
                                           created_by, created_by_user_id, close_session_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, (session_id, kind, starts_at, ends_at, status, int(is_random),
                                        created_by, created_by_user_id, int(close_session_after))).lastrowid

    def get_check(self, check_id: int) -> Optional[Dict[str, Any]]:
        rows = self._fetch_all(f"{_CHECK_SELECT} WHERE c.id = ?", (check_id,))
        return rows[0] if rows else None

    def list_checks(self, session_id: int) -> List[Dict[str, Any]]:
        """A session's checks, earliest first, each with marked_count."""
        return self._fetch_all(f"{_CHECK_SELECT} WHERE c.session_id = ? ORDER BY c.starts_at, c.id", (session_id,))

    def get_open_check(self, session_id: int) -> Optional[Dict[str, Any]]:
        rows = self._fetch_all(f"{_CHECK_SELECT} WHERE c.session_id = ? AND c.status = 'open' "
                               f"ORDER BY c.starts_at LIMIT 1", (session_id,))
        return rows[0] if rows else None

    def set_status(self, check_id: int, status: str, ends_at: Optional[str] = None,
                   void_reason: Optional[str] = None) -> int:
        """Change a check's status (void_reason says why a void check did not count)."""
        return self._execute("UPDATE attendance_checks SET status = ?, ends_at = COALESCE(?, ends_at), "
                             "void_reason = ? WHERE id = ?", (status, ends_at, void_reason, check_id))

    def set_camera_seen(self, check_id: int, seen_at: str) -> int:
        """The camera was running while this check was open (first time only)."""
        return self._execute("UPDATE attendance_checks SET camera_seen_at = ? "
                             "WHERE id = ? AND camera_seen_at IS NULL", (seen_at, check_id))

    def set_close_session_after(self, check_id: int, close_after: bool = True) -> int:
        return self._execute("UPDATE attendance_checks SET close_session_after = ? WHERE id = ?",
                             (int(close_after), check_id))

    def set_times(self, check_id: int, starts_at: str, ends_at: str) -> int:
        return self._execute("UPDATE attendance_checks SET starts_at = ?, ends_at = ? WHERE id = ?",
                             (starts_at, ends_at, check_id))

    def checks_with_status(self, status: str) -> List[Dict[str, Any]]:
        """Checks in one status, with their session's status and end time."""
        return self._fetch_all("""
            SELECT c.*, s.status AS session_status
            FROM attendance_checks c JOIN sessions s ON s.id = c.session_id
            WHERE c.status = ? ORDER BY c.starts_at
        """, (status,))

    def sessions_with_checkout(self) -> List[Dict[str, Any]]:
        """Active sessions with automatic check-out and an end time, with their check-out check (if any)."""
        return self._fetch_all("""
            SELECT s.id AS session_id, s.end_time, s.checkout_minutes,
                   c.id AS check_id, c.status AS check_status, c.starts_at, c.ends_at
            FROM sessions s
            LEFT JOIN attendance_checks c ON c.session_id = s.id AND c.kind = 'checkout'
                                         AND c.status != 'void'
            WHERE s.status = 'active' AND s.checkout_enabled = 1 AND s.end_time IS NOT NULL
        """)

    def scheduled_checkouts_without_setting(self) -> List[int]:
        """Scheduled automatic check-outs whose session no longer wants one (switched off / no end time)."""
        rows = self._fetch_all("""
            SELECT c.id FROM attendance_checks c JOIN sessions s ON s.id = c.session_id
            WHERE c.kind = 'checkout' AND c.status = 'scheduled' AND c.created_by IS NULL
              AND (s.checkout_enabled = 0 OR s.end_time IS NULL)
        """)
        return [row['id'] for row in rows]

    # -- marks --------------------------------------------------------------------------

    def add_mark(self, check_id: int, student_id: int, marked_at: str, detection_confidence: Optional[float],
                 liveness_score: Optional[float], marked_by: str) -> bool:
        """True if added, False if the student was already seen in this check."""
        query = """
            INSERT OR IGNORE INTO check_marks (check_id, student_id, marked_at, detection_confidence,
                                               liveness_score, marked_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        return self._execute(query, (check_id, student_id, marked_at, detection_confidence,
                                     liveness_score, marked_by)) == 1

    def marked_student_ids(self, check_id: int) -> Set[int]:
        rows = self._fetch_all("SELECT student_id FROM check_marks WHERE check_id = ?", (check_id,))
        return {row['student_id'] for row in rows}

    def count_due(self, session_id: int, starts_at: str) -> int:
        """Students who must show up in a check starting at `starts_at`: present / late before it began."""
        rows = self._fetch_all("""
            SELECT COUNT(*) AS n FROM attendance_records
            WHERE session_id = ? AND status IN ('present', 'late') AND marked_at < ?
        """, (session_id, starts_at))
        return rows[0]['n'] if rows else 0

    def flag_missed(self, check_id: int) -> int:
        """
        Mark 'left early' everyone who was present / late before the check
        began but was not seen during it. Returns how many were flagged.
        """
        query = """
            UPDATE attendance_records SET left_early = 1
            WHERE session_id = (SELECT session_id FROM attendance_checks WHERE id = :check_id)
              AND status IN ('present', 'late')
              AND left_early = 0
              AND marked_at < (SELECT starts_at FROM attendance_checks WHERE id = :check_id)
              AND student_id NOT IN (SELECT student_id FROM check_marks WHERE check_id = :check_id)
        """
        return self._execute(query, {'check_id': check_id})
