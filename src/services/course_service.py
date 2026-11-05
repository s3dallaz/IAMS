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
from src.data.repositories.user_repository import UserRepository
from src.services.activity_log import ActivityLog, Category, changes, describe_changes
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
        self.user_repository = UserRepository(database_path)
        self.activity = ActivityLog(database_path)

    def _log(self, action: str, summary: str, target_type: str, target_id: int,
             details: Optional[Dict[str, Any]] = None) -> None:
        self.activity.record(Category.COURSES, action, summary, target_type=target_type, target_id=target_id,
                             details=details)

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
            course_id = self.repository.create_course(code, name, department, notes)
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"Course {code} already exists")
        self._log('course_added', f"Added course {code} - {name}", 'course', course_id)
        return course_id

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
        before = self._require_course(course_id)
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
        changed = changes(before, self.repository.get_course(course_id) or {},
                          ('course_code', 'name', 'department', 'notes'))
        if changed:
            self._log('course_updated', f"Changed course {before['course_code']}: "
                      f"{describe_changes(changed, {'course_code': 'code'})}", 'course', course_id, changed)

    def set_course_active(self, course_id: int, active: bool) -> None:
        """An inactive course and its sections are hidden from new sessions; nothing is deleted."""
        course = self._require_course(course_id)
        self.repository.set_course_active(course_id, active)
        self._log('course_activated' if active else 'course_deactivated',
                  f"{'Activated' if active else 'Deactivated'} course {course['course_code']} - {course['name']}",
                  'course', course_id)

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
        self._log('course_deleted', f"Deleted course {course['course_code']} - {course['name']} "
                  f"with its sections and class lists", 'course', course_id, {'course': course})

    def _require_course(self, course_id: int) -> Dict[str, Any]:
        course = self.repository.get_course(course_id)
        if not course:
            raise CourseNotFoundError(f"Course {course_id} not found")
        return course

    # ------------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------------

    def create_section(self, course_id: int, name: str, instructor_name: Optional[str] = None,
                       notes: Optional[str] = None, instructor_id: Optional[int] = None) -> int:
        """
        Args:
            instructor_id: The instructor's account (they then see and run the
                section); its name replaces instructor_name

        Raises:
            CourseNotFoundError, InvalidCourseDataError,
            DuplicateCourseError: If the course already has a section with this name
        """
        course = self._require_course(course_id)
        name = clean_section_name(name)
        instructor_name = self._instructor_name(instructor_id, instructor_name)
        if self.repository.get_section_by_name(course_id, name):
            raise DuplicateCourseError(f"{course['course_code']} already has a section '{name}'")
        try:
            section_id = self.repository.create_section(course_id, name, instructor_name,
                                                        _clean(notes, "Notes", 500), instructor_id)
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"{course['course_code']} already has a section '{name}'")
        self._log('section_added', f"Added section {course['course_code']} / {name}"
                  + (f" (instructor: {instructor_name})" if instructor_name else ""), 'section', section_id)
        return section_id

    def get_section(self, section_id: int) -> Optional[Dict[str, Any]]:
        """The section with course_code, course_name, student_count and session_count."""
        return self.repository.get_section(section_id)

    def list_sections(self, course_id: Optional[int] = None, active_only: bool = False,
                      instructor_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sections, optionally only one course's or only one instructor account's."""
        return self.repository.list_sections(course_id=course_id, active_only=active_only,
                                             instructor_id=instructor_id)

    def instructor_sections(self, user_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """The sections an account teaches (its "My Classes")."""
        return self.list_sections(active_only=active_only, instructor_id=user_id)

    def update_section(self, section_id: int, name: str, instructor_name: Optional[str] = None,
                       notes: Optional[str] = None, instructor_id: Optional[int] = None) -> None:
        """
        instructor_id: the instructor's account, or None for none (then
        instructor_name is a typed name).

        Raises:
            CourseNotFoundError, InvalidCourseDataError, DuplicateCourseError
        """
        section = self._require_section(section_id)
        name = clean_section_name(name)
        other = self.repository.get_section_by_name(section['course_id'], name)
        if other and other['id'] != section_id:
            raise DuplicateCourseError(f"{section['course_code']} already has a section '{name}'")
        try:
            self.repository.update_section(section_id, name, self._instructor_name(instructor_id, instructor_name),
                                           _clean(notes, "Notes", 500), instructor_id)
        except sqlite3.IntegrityError:
            raise DuplicateCourseError(f"{section['course_code']} already has a section '{name}'")
        changed = changes(section, self.repository.get_section(section_id) or {},
                          ('name', 'instructor_name', 'instructor_id', 'notes'))
        changed.pop('instructor_id', None) if 'instructor_name' in changed else None
        if changed:
            self._log('section_updated', f"Changed section {section['course_code']} / {section['name']}: "
                      f"{describe_changes(changed, {'instructor_name': 'instructor', 'instructor_id': 'instructor account'})}",
                      'section', section_id, changed)

    def _instructor_name(self, instructor_id: Optional[int], typed_name: Optional[str]) -> Optional[str]:
        """The account's name for a linked instructor, else the typed name (validated)."""
        if instructor_id is None:
            return _clean(typed_name, "Instructor name", MAX_INSTRUCTOR_LENGTH)
        user = self.user_repository.get_by_id(instructor_id)
        if not user or not user['is_active']:
            raise InvalidCourseDataError("The chosen instructor account does not exist or is deactivated.")
        return user['full_name']

    def set_section_active(self, section_id: int, active: bool) -> None:
        section = self._require_section(section_id)
        self.repository.set_section_active(section_id, active)
        self._log('section_activated' if active else 'section_deactivated',
                  f"{'Activated' if active else 'Deactivated'} section {section['course_code']} / {section['name']}",
                  'section', section_id)

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
        self._log('section_deleted', f"Deleted section {section['course_code']} / {section['name']} "
                  f"and its class list ({section.get('student_count', 0)} student(s))", 'section', section_id)

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
        section = self._require_section(section_id)
        ids = list(dict.fromkeys(student_db_ids))
        added = self.repository.add_students(section_id, ids)
        logger.info(f"Added {added} student(s) to section {section_id}")
        if added:
            self._log('class_list_added', f"Added {added} student(s) to the class list of "
                      f"{section['course_code']} / {section['name']}{self._names(ids)}", 'section', section_id,
                      {'student_db_ids': ids[:2000]})
        return added

    def remove_students(self, section_id: int, student_db_ids: Iterable[int]) -> int:
        """Remove students from a class list. Returns how many were removed."""
        section = self._require_section(section_id)
        ids = list(student_db_ids)
        names = self._names(ids)          # before they are removed
        removed = self.repository.remove_students(section_id, ids)
        logger.info(f"Removed {removed} student(s) from section {section_id}")
        if removed:
            self._log('class_list_removed', f"Removed {removed} student(s) from the class list of "
                      f"{section['course_code']} / {section['name']}{names}", 'section', section_id,
                      {'student_db_ids': ids[:2000]})
        return removed

    def _names(self, student_db_ids: List[int], shown: int = 5) -> str:
        """': Alice (1), Omar (2) and 3 more' for a short summary."""
        from src.data.repositories.student_repository import StudentRepository
        repository = StudentRepository(self.repository.database_path)
        names = []
        for student_db_id in student_db_ids[:shown]:
            student = repository.get_by_id(student_db_id)
            if student:
                names.append(f"{student['name']} ({student['student_id']})")
        more = len(student_db_ids) - shown
        return (": " + ", ".join(names) + (f" and {more} more" if more > 0 else "")) if names else ""

    def get_student_sections(self, student_db_id: int) -> List[Dict[str, Any]]:
        """The sections a student is on."""
        return self.repository.get_student_sections(student_db_id)
