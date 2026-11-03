"""
User Repository for database persistence operations.

Provides database access for the users table without business logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.database.connection import get_db_connection

# Columns given out; password_hash only where it is needed to check a password
# (has_recovery_code tells whether a code exists, never the code's hash)
_PUBLIC_COLUMNS = """id, username, full_name, role, is_active, must_change_password, failed_attempts,
                     locked_until, last_login_at, created_at, created_by,
                     recovery_code_hash IS NOT NULL AS has_recovery_code"""


class UserRepository:
    """Repository for the users table. Usernames compare case-insensitively."""

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the user repository.

        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path

    def _fetch_one(self, query: str, params: Any = ()) -> Optional[Dict[str, Any]]:
        with get_db_connection(self.database_path) as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def _execute(self, query: str, params: Any = ()) -> int:
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, params).rowcount

    def count_users(self) -> int:
        row = self._fetch_one("SELECT COUNT(*) AS n FROM users")
        return row['n'] if row else 0

    def count_active_with_role(self, role: str) -> int:
        row = self._fetch_one("SELECT COUNT(*) AS n FROM users WHERE role = ? AND is_active = 1", (role,))
        return row['n'] if row else 0

    def create_user(self, username: str, full_name: str, role: str, password_hash: str,
                    must_change_password: bool, created_by: Optional[str]) -> int:
        """
        Raises:
            sqlite3.IntegrityError: If the username is taken
        """
        query = """
            INSERT INTO users (username, full_name, role, password_hash, must_change_password, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, (username, full_name, role, password_hash,
                                        int(must_change_password), created_by)).lastrowid

    def get_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._fetch_one(f"SELECT {_PUBLIC_COLUMNS} FROM users WHERE id = ?", (user_id,))

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one(f"SELECT {_PUBLIC_COLUMNS} FROM users WHERE username = ?", (username,))

    def get_password_hash(self, user_id: int) -> Optional[str]:
        row = self._fetch_one("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        return row['password_hash'] if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """All users, by role (IT, admin, instructor) then name."""
        query = f"""
            SELECT {_PUBLIC_COLUMNS} FROM users
            ORDER BY CASE role WHEN 'it' THEN 0 WHEN 'admin' THEN 1 ELSE 2 END, full_name COLLATE NOCASE
        """
        with get_db_connection(self.database_path) as conn:
            return [dict(row) for row in conn.execute(query).fetchall()]

    def get_recovery_code_hash(self, user_id: int) -> Optional[str]:
        row = self._fetch_one("SELECT recovery_code_hash FROM users WHERE id = ?", (user_id,))
        return row['recovery_code_hash'] if row else None

    def set_recovery_code_hash(self, user_id: int, code_hash: str) -> int:
        return self._execute("UPDATE users SET recovery_code_hash = ? WHERE id = ?", (code_hash, user_id))

    def record_failed_login(self, user_id: int, failed_attempts: int, locked_until: Optional[str]) -> None:
        self._execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                      (failed_attempts, locked_until, user_id))

    def record_login(self, user_id: int, at: str) -> None:
        self._execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                      (at, user_id))

    def unlock(self, user_id: int) -> int:
        return self._execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (user_id,))

    def set_password(self, user_id: int, password_hash: str, must_change_password: bool) -> int:
        """New password; also unlocks the account."""
        return self._execute("""
            UPDATE users SET password_hash = ?, must_change_password = ?, failed_attempts = 0, locked_until = NULL
            WHERE id = ?
        """, (password_hash, int(must_change_password), user_id))

    def update_user(self, user_id: int, full_name: str, role: str) -> int:
        return self._execute("UPDATE users SET full_name = ?, role = ? WHERE id = ?", (full_name, role, user_id))

    def set_active(self, user_id: int, active: bool) -> int:
        return self._execute("UPDATE users SET is_active = ? WHERE id = ?", (int(active), user_id))
