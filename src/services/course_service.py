"""
Course Service: courses, their sections and the sections' class lists.

A course (e.g. TM103) has one or more sections (e.g. "A", "Lab 1"), each
with an instructor name and a class list of students. Sessions will be
linked to a section, so that only its class list is expected.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data.repositories.course_repository import CourseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Course codes: letters (also Arabic), digits, space, . _ - /
COURSE_CODE_PATTERN = re.compile(r"^[\w][\w .\-/]*$")
MAX_COURSE_CODE_LENGTH = 30
MAX_COURSE_NAME_LENGTH = 120
MAX_SECTION_NAME_LENGTH = 50
MAX_INSTRUCTOR_LENGTH = 100


class CourseServiceError(Exception):
    """Base exception for course service errors."""


class InvalidCourseDataError(CourseServiceError):
    """Raised when course or section data is invalid."""


class DuplicateCourseError(CourseServiceError):
    """Raised when a course code or a section name is already used."""


class CourseNotFoundError(CourseServiceError):
    """Raised when a course or section does not exist."""


class CourseInUseError(CourseServiceError):
    """Raised when a course or section cannot be deleted because sessions use it."""


def _clean(value: Optional[str], field: str, max_length: int, required: bool = False) -> Optional[str]:
    """Strip a text field; empty -> None. Raises InvalidCourseDataError."""
    if value is not None and not isinstance(value, str):
        raise InvalidCourseDataError(f"{field} must be text")
    text = re.sub(r"\s+", " ", value).strip() if value else ""
    if required and not text:
        raise InvalidCourseDataError(f"{field} cannot be empty")
    if len(text) > max_length:
        raise InvalidCourseDataError(f"{field} is longer than {max_length} characters")
    return text or None


def clean_course_code(course_code: Optional[str]) -> str:
    """Validate a course code ('TM 103' stays as typed, only trimmed)."""
    code = _clean(course_code, "Course code", MAX_COURSE_CODE_LENGTH, required=True)
    if not COURSE_CODE_PATTERN.match(code):
        raise InvalidCourseDataError(f"Course code '{code}' may only contain letters, digits, spaces, '.', '-', '_' and '/'")
    return code


def clean_section_name(name: Optional[str]) -> str:
    return _clean(name, "Section name", MAX_SECTION_NAME_LENGTH, required=True)


class CourseService:
    """
    Service layer for courses, sections and class lists: validation and
    rules. Does not contain SQL.
    """

    def __init__(self, database_path: Optional[Path] = None):
        self.repository = CourseRepository(database_path)

    # ------------------------------------------------------------------------
    # Courses
    # ------------------------------------------------------------------------

    def create_course(self, course_code: str, name: str, department: Optional[str] = None,
                      notes: Optional[str] = None) -> int:
        """
        Raises:
            InvalidCourseDataError: If a field is invalid
            DuplicateCourseError: If the course code already exists (any letter case)
        """
        code = clean_course_code(course_code)
        name = _clean(name, "Course name", MAX_COURSE_NAME_LENGTH, required=True)
        department = _clean(department, "Department", MAX_COURSE_NAME_LENGTH)
        notes = _clean(notes, "Notes", 500)
        if self.repository.get_course_by_code(code):
            raise DuplicateCourseError(f"Course {code} already exists")
        try:
            return self.repository.create_course(code, name, department, notes)
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"Course {code} already exists")

    def get_course(self, course_id: int) -> Optional[Dict[str, Any]]:
        return self.repository.get_course(course_id)

    def list_courses(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Courses with section_count, student_count and session_count."""
        return self.repository.list_courses(active_only=active_only)

    def update_course(self, course_id: int, course_code: str, name: str,
                      department: Optional[str] = None, notes: Optional[str] = None) -> None:
        """
        Raises:
            CourseNotFoundError, InvalidCourseDataError, DuplicateCourseError
        """
        self._require_course(course_id)
        code = clean_course_code(course_code)
        name = _clean(name, "Course name", MAX_COURSE_NAME_LENGTH, required=True)
        other = self.repository.get_course_by_code(code)
        if other and other['id'] != course_id:
            raise DuplicateCourseError(f"Course {code} already exists")
        try:
            self.repository.update_course(course_id, code, name,
                                          _clean(department, "Department", MAX_COURSE_NAME_LENGTH),
                                          _clean(notes, "Notes", 500))
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"Course {code} already exists")

    def set_course_active(self, course_id: int, active: bool) -> None:
        """An inactive course and its sections are hidden from new sessions; nothing is deleted."""
        self._require_course(course_id)
        self.repository.set_course_active(course_id, active)

    def delete_course(self, course_id: int) -> None:
        """
        Delete a course with its sections and class lists.

        Raises:
            CourseNotFoundError: If the course does not exist
            CourseInUseError: If sessions use one of its sections (deactivate it instead)
        """
        course = self._require_course(course_id)
        sessions = self.repository.count_course_sessions(course_id)
        if sessions:
            raise CourseInUseError(f"Course {course['course_code']} has {sessions} session(s), so it can't be "
                                   f"deleted. Deactivate it instead - its history is kept.")
        self.repository.delete_course(course_id)
        logger.info(f"Deleted course {course['course_code']}")

    def _require_course(self, course_id: int) -> Dict[str, Any]:
        course = self.repository.get_course(course_id)
        if not course:
            raise CourseNotFoundError(f"Course {course_id} not found")
        return course

    # ------------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------------

    def create_section(self, course_id: int, name: str, instructor_name: Optional[str] = None,
                       notes: Optional[str] = None) -> int:
        """
        Raises:
            CourseNotFoundError, InvalidCourseDataError,
            DuplicateCourseError: If the course already has a section with this name
        """
        course = self._require_course(course_id)
        name = clean_section_name(name)
        instructor_name = _clean(instructor_name, "Instructor name", MAX_INSTRUCTOR_LENGTH)
        if self.repository.get_section_by_name(course_id, name):
            raise DuplicateCourseError(f"{course['course_code']} already has a section '{name}'")
        try:
            return self.repository.create_section(course_id, name, instructor_name, _clean(notes, "Notes", 500))
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"{course['course_code']} already has a section '{name}'")

    def get_section(self, section_id: int) -> Optional[Dict[str, Any]]:
        """The section with course_code, course_name, student_count and session_count."""
        return self.repository.get_section(section_id)

    def list_sections(self, course_id: Optional[int] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        return self.repository.list_sections(course_id=course_id, active_only=active_only)

    def update_section(self, section_id: int, name: str, instructor_name: Optional[str] = None,
                       notes: Optional[str] = None) -> None:
        """
        Raises:
            CourseNotFoundError, InvalidCourseDataError, DuplicateCourseError
        """
        section = self._require_section(section_id)
        name = clean_section_name(name)
        other = self.repository.get_section_by_name(section['course_id'], name)
        if other and other['id'] != section_id:
            raise DuplicateCourseError(f"{section['course_code']} already has a section '{name}'")
        try:
            self.repository.update_section(section_id, name,
                                           _clean(instructor_name, "Instructor name", MAX_INSTRUCTOR_LENGTH),
                                           _clean(notes, "Notes", 500))
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"{section['course_code']} already has a section '{name}'")

    def set_section_active(self, section_id: int, active: bool) -> None:
        self._require_section(section_id)
        self.repository.set_section_active(section_id, active)

    def delete_section(self, section_id: int) -> None:
        """
        Delete a section and its class list.

        Raises:
            CourseNotFoundError: If the section does not exist
            CourseInUseError: If sessions use it (deactivate it instead)
        """
        section = self._require_section(section_id)
        if section['session_count']:
            raise CourseInUseError(f"Section {section['name']} has {section['session_count']} session(s), so it "
                                   f"can't be deleted. Deactivate it instead - its history is kept.")
        self.repository.delete_section(section_id)
        logger.info(f"Deleted section {section['course_code']} / {section['name']}")

    def _require_section(self, section_id: int) -> Dict[str, Any]:
        section = self.repository.get_section(section_id)
        if not section:
            raise CourseNotFoundError(f"Section {section_id} not found")
        return section

    # ------------------------------------------------------------------------
    # Class lists
    # ------------------------------------------------------------------------

    def get_class_list(self, section_id: int, search: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Students of a section by name; search matches part of the name or student ID."""
        return self.repository.get_class_list(section_id, search=(search or "").strip() or None, limit=limit)

    def get_class_list_student_ids(self, section_id: int) -> set:
        return self.repository.get_class_list_student_ids(section_id)

    def add_students(self, section_id: int, student_db_ids: Iterable[int]) -> int:
        """Add students to a class list (ones already on it are skipped). Returns how many were added."""
        self._require_section(section_id)
        added = self.repository.add_students(section_id, list(dict.fromkeys(student_db_ids)))
        logger.info(f"Added {added} student(s) to section {section_id}")
        return added

    def remove_students(self, section_id: int, student_db_ids: Iterable[int]) -> int:
        """Remove students from a class list. Returns how many were removed."""
        self._require_section(section_id)
        removed = self.repository.remove_students(section_id, list(student_db_ids))
        logger.info(f"Removed {removed} student(s) from section {section_id}")
        return removed

    def get_student_sections(self, student_db_id: int) -> List[Dict[str, Any]]:
        """The sections a student is on."""
        return self.repository.get_student_sections(student_db_id)
