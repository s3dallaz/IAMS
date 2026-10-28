"""
Class list import: put students on section class lists from an Excel (.xlsx)
or CSV file, one row per student and section:

    course_code, section, student_id          (required)
    course_name, instructor                   (optional, used for NEW courses / sections)

Courses and sections that don't exist yet are created. The file is checked
first (plan_class_list_import) and nothing is saved until the plan is
applied, in one transaction. Re-importing the same file is safe: students
already on a class list are skipped. Nobody is removed by an import.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.services.bulk_import_service import (
    BulkImportError, _clean_cell, _csv_bytes, normalize_column, read_table,
)
from src.services.course_service import (
    CourseService, InvalidCourseDataError, clean_course_code, clean_section_name,
)
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_LIST_COLUMNS = ("course_code", "section", "student_id", "course_name", "instructor")
REQUIRED_CLASS_LIST_COLUMNS = ("course_code", "section", "student_id")
MAX_CLASS_LIST_ROWS = 50000

# Header spellings accepted for each column (compared lower-case, with _ and - as spaces);
# anything else falls back to the student import's names (student id, id, ...)
CLASS_LIST_ALIASES = {
    "course": "course_code", "course code": "course_code", "course no": "course_code",
    "course number": "course_code", "course id": "course_code",
    "section": "section", "section name": "section", "group": "section", "class": "section",
    "course name": "course_name", "course title": "course_name", "title": "course_name",
    "instructor": "instructor", "instructor name": "instructor", "teacher": "instructor",
    "doctor": "instructor", "lecturer": "instructor",
}


@dataclass
class ClassListRow:
    """One row of the uploaded file and what the import will do with it."""
    row_number: int                  # row in the spreadsheet (the header is row 1)
    course_code: str
    section: str
    student_id: str
    course_name: Optional[str] = None
    instructor: Optional[str] = None
    student_name: str = ""
    student_db_id: Optional[int] = None
    action: str = "add"              # add | skip | error
    note: str = ""                   # why a row is skipped
    errors: List[str] = field(default_factory=list)


@dataclass
class ClassListImportPlan:
    """The checked file. Nothing is saved yet."""
    rows: List[ClassListRow]
    new_courses: List[Dict[str, Any]] = field(default_factory=list)    # course_code, name
    new_sections: List[Dict[str, Any]] = field(default_factory=list)   # course_code, name, instructor_name
    ignored_columns: List[str] = field(default_factory=list)

    def with_action(self, action: str) -> List[ClassListRow]:
        return [row for row in self.rows if row.action == action]

    @property
    def counts(self) -> Dict[str, int]:
        return {action: len(self.with_action(action)) for action in ("add", "skip", "error")}


def normalize_class_list_column(header: Any) -> str:
    key = re.sub(r"\s+", " ", str(header).strip().lower().replace("_", " ").replace("-", " "))
    return CLASS_LIST_ALIASES.get(key) or normalize_column(header)


def read_class_list_table(data: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read an uploaded class list (.csv / .xlsx).

    Raises:
        BulkImportError: unreadable file, missing required columns, too many rows
    """
    return read_table(data, filename, CLASS_LIST_COLUMNS, REQUIRED_CLASS_LIST_COLUMNS,
                      normalize_class_list_column, MAX_CLASS_LIST_ROWS,
                      "course_code, section, student_id (and optionally course_name, instructor)")


def plan_class_list_import(table: pd.DataFrame, course_service: CourseService,
                           student_service: StudentService,
                           ignored_columns: Optional[List[str]] = None) -> ClassListImportPlan:
    """Check every row and decide what importing it would do. Nothing is saved."""
    students = {s['student_id']: s for s in student_service.get_all_students(active_only=False)}
    courses = {c['course_code'].lower(): c for c in course_service.list_courses()}
    sections = {(s['course_id'], s['name'].lower()): s for s in course_service.list_sections()}
    class_lists: Dict[int, set] = {}            # section id -> student DB IDs, loaded when first needed
    new_courses: Dict[str, Dict[str, Any]] = {}       # lower code -> course
    new_sections: Dict[Tuple[str, str], Dict[str, Any]] = {}   # (lower code, lower name) -> section
    seen: Dict[Tuple[str, str, str], int] = {}  # (lower code, lower section, student_id) -> first row

    rows: List[ClassListRow] = []
    for index, record in enumerate(table.to_dict("records")):
        cells = {column: _clean_cell(record.get(column, "")) for column in CLASS_LIST_COLUMNS}
        row = ClassListRow(row_number=index + 2, course_code=cells["course_code"], section=cells["section"],
                           student_id=cells["student_id"], course_name=cells["course_name"] or None,
                           instructor=cells["instructor"] or None)
        rows.append(row)

        try:
            row.course_code = clean_course_code(row.course_code)
        except InvalidCourseDataError as e:
            row.errors.append(str(e))
        try:
            row.section = clean_section_name(row.section)
        except InvalidCourseDataError as e:
            row.errors.append(str(e))
        student = students.get(row.student_id)
        if not row.student_id:
            row.errors.append("student_id is empty")
        elif student is None:
            row.errors.append(f"student {row.student_id} is not in the Student Registry - import the student first")
        else:
            row.student_db_id, row.student_name = student['id'], student['name']
        if row.errors:
            row.action = "error"
            continue

        code_key, section_key = row.course_code.lower(), row.section.lower()
        first = seen.setdefault((code_key, section_key, row.student_id), row.row_number)
        if first != row.row_number:
            row.action, row.note = "skip", f"repeated in the file (row {first})"
            continue

        course = courses.get(code_key)
        if course is None:
            planned_course = new_courses.setdefault(code_key, {'course_code': row.course_code, 'name': None})
            planned_course['name'] = planned_course['name'] or row.course_name
            section = None
        else:
            section = sections.get((course['id'], section_key))
        if section is None:
            planned = new_sections.setdefault((code_key, section_key), {
                'course_code': course['course_code'] if course else new_courses[code_key]['course_code'],
                'name': row.section, 'instructor_name': None})
            planned['instructor_name'] = planned['instructor_name'] or row.instructor
            continue

        if section['id'] not in class_lists:
            class_lists[section['id']] = course_service.get_class_list_student_ids(section['id'])
        if row.student_db_id in class_lists[section['id']]:
            row.action, row.note = "skip", "already on the class list"

    for planned_course in new_courses.values():   # no course_name in the file: the code is the name
        planned_course['name'] = planned_course['name'] or planned_course['course_code']
    return ClassListImportPlan(rows=rows, new_courses=list(new_courses.values()),
                               new_sections=list(new_sections.values()),
                               ignored_columns=list(ignored_columns or []))


def apply_class_list_import(plan: ClassListImportPlan, course_service: CourseService) -> Dict[str, int]:
    """
    Create the new courses and sections and add the students, in ONE
    transaction: either everything is saved or nothing.
    """
    memberships = [{'course_code': row.course_code, 'section': row.section, 'student_id': row.student_db_id}
                   for row in plan.with_action("add")]
    try:
        added = course_service.repository.bulk_import(plan.new_courses, plan.new_sections, memberships)
    except Exception as e:
        raise BulkImportError(f"Nothing was saved - the database refused the import: {e}")
    return {"courses": len(plan.new_courses), "sections": len(plan.new_sections), "added": added,
            "skipped": plan.counts["skip"], "errors": plan.counts["error"]}


def class_list_template_csv() -> bytes:
    """An example file with the expected columns (UTF-8 with BOM so Excel shows Arabic names)."""
    rows = [
        {"course_code": "TM103", "section": "A", "student_id": "2021001",
         "course_name": "Computer Organization", "instructor": "Dr. Sara Ali"},
        {"course_code": "TM103", "section": "A", "student_id": "2021002", "course_name": "", "instructor": ""},
        {"course_code": "TM103", "section": "B", "student_id": "2021003", "course_name": "",
         "instructor": "Dr. Omar Nabil"},
    ]
    return _csv_bytes(list(CLASS_LIST_COLUMNS), rows)


def class_list_errors_csv(plan: ClassListImportPlan) -> bytes:
    """The rows with errors plus an 'error' column, to fix and upload again."""
    rows = [{"row": row.row_number, "course_code": row.course_code, "section": row.section,
             "student_id": row.student_id, "course_name": row.course_name, "instructor": row.instructor,
             "error": "; ".join(row.errors)}
            for row in plan.with_action("error")]
    return _csv_bytes(["row", *CLASS_LIST_COLUMNS, "error"], rows)
