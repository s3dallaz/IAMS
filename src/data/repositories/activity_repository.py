"""
Activity Repository: the activity_log table (who did what, when).

Append-only: there is no update or delete. Times are local
'YYYY-MM-DD HH:MM:SS' text, like session and attendance times.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.database.connection import get_db_connection

_COLUMNS = ('at', 'user_id', 'username', 'full_name', 'role', 'category', 'action', 'summary',
            'target_type', 'target_id', 'session_id', 'details')


class ActivityRepository:
    """Repository for the activity_log table."""

    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = database_path

    def add(self, entry: Dict[str, Any]) -> int:
        query = (f"INSERT INTO activity_log ({', '.join(_COLUMNS)}) "
                 f"VALUES ({', '.join('?' for _ in _COLUMNS)})")
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, tuple(entry.get(column) for column in _COLUMNS)).lastrowid

    @staticmethod
    def _where(date_from: Optional[str] = None, date_to: Optional[str] = None,
               username: Optional[str] = None, full_name: Optional[str] = None,
               category: Optional[str] = None,
               session_id: Optional[int] = None, target: Optional[Tuple[str, int]] = None,
               search: Optional[str] = None, actions: Optional[Sequence[str]] = None) -> Tuple[str, List[Any]]:
        clauses, params = [], []
        if date_from:
            clauses.append("at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("at <= ?")
            params.append(date_to)
        if username:
            clauses.append("username = ?")
            params.append(username)
        if full_name:                      # entries without an account (System, unknown person)
            clauses.append("username IS NULL AND full_name = ?")
            params.append(full_name)
        if category:
            clauses.append("category = ?")
            params.append(category)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if target is not None:
            clauses.append("target_type = ? AND target_id = ?")
            params.extend(target)
        if actions:
            clauses.append(f"action IN ({', '.join('?' for _ in actions)})")
            params.extend(actions)
        if search:
            clauses.append("(summary LIKE ? OR full_name LIKE ? OR username LIKE ? OR details LIKE ?)")
            params.extend([f"%{search}%"] * 4)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def search(self, limit: int = 500, **filters) -> List[Dict[str, Any]]:
        """Matching entries, newest first."""
        where, params = self._where(**filters)
        with get_db_connection(self.database_path) as conn:
            rows = conn.execute(f"SELECT * FROM activity_log{where} ORDER BY at DESC, id DESC LIMIT ?",
                                (*params, limit)).fetchall()
        return [dict(row) for row in rows]

    def count(self, **filters) -> int:
        where, params = self._where(**filters)
        with get_db_connection(self.database_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM activity_log{where}", params).fetchone()[0]

    def people(self) -> List[Dict[str, Any]]:
        """Everyone who appears in the log (latest name per username), for the filter."""
        with get_db_connection(self.database_path) as conn:
            rows = conn.execute("""
                SELECT username, full_name, role FROM activity_log a
                WHERE username IS NOT NULL
                  AND id = (SELECT MAX(id) FROM activity_log b WHERE b.username = a.username)
                ORDER BY full_name, username
            """).fetchall()
        return [dict(row) for row in rows]
