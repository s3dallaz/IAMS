"""
Settings Repository for database persistence operations.

Provides database access for the settings table (key -> text value)
without business logic.
"""

from pathlib import Path
from typing import Optional

from src.database.connection import get_db_connection


class SettingsRepository:
    """Repository for the settings table."""

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the settings repository.

        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path

    def get(self, key: str) -> Optional[str]:
        """The setting's value, or None if it is not set."""
        with get_db_connection(self.database_path) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row['value'] if row else None

    def set(self, key: str, value: str, description: Optional[str] = None) -> None:
        """Create or replace a setting."""
        query = """
            INSERT INTO settings (key, value, description) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """
        with get_db_connection(self.database_path) as conn:
            conn.execute(query, (key, value, description))

    def delete(self, key: str) -> None:
        """Remove a setting (nothing happens if it is not set)."""
        with get_db_connection(self.database_path) as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
