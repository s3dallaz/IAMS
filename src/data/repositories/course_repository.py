"""
Course Repository for database persistence operations.

Provides database access for courses, their sections and the sections'
class lists (section_students), without business logic.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from src.database.connection import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CourseRepository:
    """
    Repository for the courses, sections and section_students tables.
    Course codes and section names compare case-insensitively (COLLATE NOCASE).
    """

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize the course repository.

        Args:
            database_path: Path to database file. If None, uses config.DATABASE_PATH
        """
        self.database_path = database_path

    def _fetch_all(self, query: str, params: Any = ()) -> List[Dict[str, Any]]:
        with get_db_connection(self.database_path) as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def _fetch_one(self, query: str, params: Any = ()) -> Optional[Dict[str, Any]]:
        with get_db_connection(self.database_path) as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def _execute(self, query: str, params: Any = ()) -> int:
        """Run one write; returns the number of rows changed."""
        with get_db_connection(self.database_path) as conn:
            return conn.execute(query, params).rowcount

    # ------------------------------------------------------------------------
    # Courses
    # ------------------------------------------------------------------------

    def create_course(self, course_code: str, name: str, department: Optional[str] = None,
                      notes: Optional[str] = None) -> int:
        """
        Returns:
            Database ID of the new course

        Raises:
            sqlite3.IntegrityError: If the course code already exists
        """
        query = "INSERT INTO courses (course_code, name, department, notes) VALUES (?, ?, ?, ?)"
        with get_db_connection(self.database_path) as conn:
            course_id = conn.execute(query, (course_code, name, department, notes)).lastrowid
        logger.info(f"Created course {course_code} (DB ID: {course_id})")
        return course_id

    def get_course(self, course_id: int) -> Optional[Dict[str, Any]]:
        return self._fetch_one("SELECT * FROM courses WHERE id = ?", (course_id,))

    def get_course_by_code(self, course_code: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one("SELECT * FROM courses WHERE course_code = ?", (course_code,))

    def list_courses(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Courses by code, each with section_count, student_count (different
        students over all its sections) and session_count.
        """
        query = f"""
            SELECT c.*,
                   (SELECT COUNT(*) FROM sections s WHERE s.course_id = c.id) AS section_count,
                   (SELECT COUNT(DISTINCT ss.student_id) FROM section_students ss
                    JOIN sections s ON s.id = ss.section_id WHERE s.course_id = c.id) AS student_count,
                   (SELECT COUNT(*) FROM sessions sess
                    JOIN sections s ON s.id = sess.section_id WHERE s.course_id = c.id) AS session_count
            FROM courses c
            {"WHERE c.is_active = 1" if active_only else ""}
            ORDER BY c.course_code
        """
        return self._fetch_all(query)

    def update_course(self, course_id: int, course_code: str, name: str,
                      department: Optional[str], notes: Optional[str]) -> int:
        """
        Raises:
            sqlite3.IntegrityError: If the new course code belongs to another course
        """
        query = "UPDATE courses SET course_code = ?, name = ?, department = ?, notes = ? WHERE id = ?"
        return self._execute(query, (course_code, name, department, notes, course_id))

    def set_course_active(self, course_id: int, active: bool) -> int:
        return self._execute("UPDATE courses SET is_active = ? WHERE id = ?", (int(active), course_id))

    def count_course_sessions(self, course_id: int) -> int:
        row = self._fetch_one("""
            SELECT COUNT(*) AS n FROM sessions sess
            JOIN sections s ON s.id = sess.section_id WHERE s.course_id = ?
        """, (course_id,))
        return row['n'] if row else 0

    def delete_course(self, course_id: int) -> int:
        """Deletes the course with its sections and their class lists (ON DELETE CASCADE)."""
        return self._execute("DELETE FROM courses WHERE id = ?", (course_id,))

    # ------------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------------

    _SECTION_SELECT = """
        SELECT s.*, c.course_code, c.name AS course_name, c.is_active AS course_is_active,
               (SELECT COUNT(*) FROM section_students ss WHERE ss.section_id = s.id) AS student_count,
               (SELECT COUNT(*) FROM sessions sess WHERE sess.section_id = s.id) AS session_count
        FROM sections s
        JOIN courses c ON c.id = s.course_id
    """

    def create_section(self, course_id: int, name: str, instructor_name: Optional[str] = None,
                       notes: Optional[str] = None) -> int:
        """
        Returns:
            Database ID of the new section

        Raises:
            sqlite3.IntegrityError: If the course already has a section with this name
        """
        query = "INSERT INTO sections (course_id, name, instructor_name, notes) VALUES (?, ?, ?, ?)"
        with get_db_connection(self.database_path) as conn:
            section_id = conn.execute(query, (course_id, name, instructor_name, notes)).lastrowid
        logger.info(f"Created section '{name}' of course {course_id} (DB ID: {section_id})")
        return section_id

    def get_section(self, section_id: int) -> Optional[Dict[str, Any]]:
        """The section with course_code, course_name, student_count and session_count."""
        return self._fetch_one(f"{self._SECTION_SELECT} WHERE s.id = ?", (section_id,))

    def get_section_by_name(self, course_id: int, name: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one(f"{self._SECTION_SELECT} WHERE s.course_id = ? AND s.name = ?",
                               (course_id, name))

    def list_sections(self, course_id: Optional[int] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        """Sections by course code and name, with the same extra fields as get_section."""
        conditions, params = [], []
        if course_id is not None:
            conditions.append("s.course_id = ?")
            params.append(course_id)
        if active_only:
            conditions.append("s.is_active = 1 AND c.is_active = 1")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._fetch_all(f"{self._SECTION_SELECT} {where} ORDER BY c.course_code, s.name", params)

    def update_section(self, section_id: int, name: str, instructor_name: Optional[str],
                       notes: Optional[str]) -> int:
        """
        Raises:
            sqlite3.IntegrityError: If the course already has another section with this name
        """
        query = "UPDATE sections SET name = ?, instructor_name = ?, notes = ? WHERE id = ?"
        return self._execute(query, (name, instructor_name, notes, section_id))

    def set_section_active(self, section_id: int, active: bool) -> int:
        return self._execute("UPDATE sections SET is_active = ? WHERE id = ?", (int(active), section_id))

    def delete_section(self, section_id: int) -> int:
        """Deletes the section and its class list (ON DELETE CASCADE)."""
        return self._execute("DELETE FROM sections WHERE id = ?", (section_id,))

    # ------------------------------------------------------------------------
    # Class lists
    # ------------------------------------------------------------------------

    def get_class_list(self, section_id: int, search: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Students of a section by name: student_db_id, student_id, name,
        department, is_active, added_at. search matches part of the name or ID.
        """
        query = """
            SELECT st.id AS student_db_id, st.student_id, st.name, st.department, st.is_active, ss.added_at
            FROM section_students ss
            JOIN students st ON st.id = ss.student_id
            WHERE ss.section_id = :section_id
        """
        params: Dict[str, Any] = {'section_id': section_id}
        if search:
            # '!' escapes LIKE wildcards, so "%" or "_" typed by the user match literally
            escaped = search.replace('!', '!!').replace('%', '!%').replace('_', '!_')
            query += " AND (st.name LIKE :search ESCAPE '!' OR st.student_id LIKE :search ESCAPE '!')"
            params['search'] = f"%{escaped}%"
        query += " ORDER BY st.name COLLATE NOCASE, st.student_id"
        if limit:
            query += " LIMIT :limit"
            params['limit'] = int(limit)
        return self._fetch_all(query, params)

    def get_class_list_student_ids(self, section_id: int) -> Set[int]:
        """Database IDs of the students on a section's class list."""
        rows = self._fetch_all("SELECT student_id FROM section_students WHERE section_id = ?", (section_id,))
        return {row['student_id'] for row in rows}

    def add_students(self, section_id: int, student_db_ids: Iterable[int]) -> int:
        """Adds students to a class list; ones already on it are skipped. Returns how many were added."""
        query = "INSERT OR IGNORE INTO section_students (section_id, student_id) VALUES (?, ?)"
        with get_db_connection(self.database_path) as conn:
            before = conn.total_changes
            conn.executemany(query, [(section_id, student_id) for student_id in student_db_ids])
            return conn.total_changes - before

    def remove_students(self, section_id: int, student_db_ids: Iterable[int]) -> int:
        """Removes students from a class list. Returns how many were removed."""
        query = "DELETE FROM section_students WHERE section_id = ? AND student_id = ?"
        with get_db_connection(self.database_path) as conn:
            before = conn.total_changes
            conn.executemany(query, [(section_id, student_id) for student_id in student_db_ids])
            return conn.total_changes - before

    def get_student_sections(self, student_db_id: int) -> List[Dict[str, Any]]:
        """The sections a student is on, with course_code and course_name."""
        return self._fetch_all(f"""
            {self._SECTION_SELECT}
            JOIN section_students mine ON mine.section_id = s.id AND mine.student_id = ?
            ORDER BY c.course_code, s.name
        """, (student_db_id,))

    def bulk_import(self, new_courses: List[Dict[str, Any]], new_sections: List[Dict[str, Any]],
                    memberships: List[Dict[str, Any]]) -> int:
        """
        Save a class-list import in ONE transaction (all or nothing).

        Args:
            new_courses: dicts with course_code, name
            new_sections: dicts with course_code, name, instructor_name
            memberships: dicts with course_code, section, student_id (student DB ID)

        Returns:
            Number of students added to class lists (ones already on them are skipped)
        """
        insert_course = "INSERT INTO courses (course_code, name) VALUES (:course_code, :name)"
        insert_section = """
            INSERT INTO sections (course_id, name, instructor_name)
            VALUES ((SELECT id FROM courses WHERE course_code = :course_code), :name, :instructor_name)
        """
        insert_member = """
            INSERT OR IGNORE INTO section_students (section_id, student_id)
            VALUES ((SELECT s.id FROM sections s JOIN courses c ON c.id = s.course_id
                     WHERE c.course_code = :course_code AND s.name = :section), :student_id)
        """
        try:
            with get_db_connection(self.database_path) as conn:
                conn.executemany(insert_course, new_courses)
                conn.executemany(insert_section, new_sections)
                before = conn.total_changes
                conn.executemany(insert_member, memberships)
                added = conn.total_changes - before
            logger.info(f"Class list import: {len(new_courses)} course(s), {len(new_sections)} section(s) "
                        f"created, {added} student(s) added")
            return added
        except sqlite3.Error as e:
            logger.error(f"Database error in class list import (nothing was saved): {e}")
            raise
