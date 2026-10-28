"""
Tests for courses, sections and class lists (CourseService), the class-list
import, and the migration that adds sessions.section_id to old databases.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.database.connection import get_db_connection
from src.database.schema import initialize_database
from src.services.class_list_import_service import (
    apply_class_list_import, class_list_errors_csv, class_list_template_csv,
    plan_class_list_import, read_class_list_table,
)
from src.services.course_service import (
    CourseInUseError, CourseNotFoundError, CourseService, DuplicateCourseError, InvalidCourseDataError,
)
from src.services.student_service import StudentService


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def courses(temp_db):
    return CourseService(temp_db)


@pytest.fixture
def students(temp_db):
    service = StudentService(temp_db)
    for sid, name in [("2021001", "Alice"), ("2021002", "Omar"), ("2021003", "Sara")]:
        service.create_student(sid, name)
    return service


def _db_id(students, student_id):
    return students.get_student_by_student_id(student_id)['id']


def _add_session(db_path, section_id=None):
    with get_db_connection(db_path) as conn:
        return conn.execute("INSERT INTO sessions (session_name, start_time, section_id) "
                            "VALUES ('S', '2026-10-28 10:00:00', ?)", (section_id,)).lastrowid


# -- migration ---------------------------------------------------------------

def test_migration_adds_section_column_and_keeps_old_sessions():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    try:
        # A database from before sections existed
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_name TEXT NOT NULL,
                        course_code TEXT, start_time TIMESTAMP NOT NULL, end_time TIMESTAMP,
                        late_threshold_minutes INTEGER DEFAULT 15, status TEXT DEFAULT 'scheduled',
                        created_by TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at TIMESTAMP, notes TEXT)""")
        conn.execute("INSERT INTO sessions (session_name, course_code, start_time) "
                     "VALUES ('Old', 'TM103', '2026-10-01 10:00:00')")
        conn.commit()
        conn.close()

        assert initialize_database(db_path)
        assert initialize_database(db_path)   # safe to repeat

        with get_db_connection(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            old = dict(conn.execute("SELECT * FROM sessions").fetchone())
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert 'section_id' in columns
        assert old['session_name'] == 'Old' and old['course_code'] == 'TM103' and old['section_id'] is None
        assert {'courses', 'sections', 'section_students'} <= tables
    finally:
        db_path.unlink()


# -- courses -----------------------------------------------------------------

def test_create_and_list_course(courses):
    course_id = courses.create_course("  TM103 ", "  Computer   Organization ", "IT")
    [course] = courses.list_courses()
    assert course['id'] == course_id
    assert course['course_code'] == "TM103" and course['name'] == "Computer Organization"
    assert course['section_count'] == 0 and course['student_count'] == 0


def test_course_code_is_unique_ignoring_case(courses):
    courses.create_course("TM103", "A")
    with pytest.raises(DuplicateCourseError):
        courses.create_course("tm103", "B")


@pytest.mark.parametrize("code, name", [("", "Name"), ("TM103", ""), ("TM#103", "Name"), ("X" * 31, "Name")])
def test_invalid_course_data(courses, code, name):
    with pytest.raises(InvalidCourseDataError):
        courses.create_course(code, name)


def test_update_course_rejects_another_courses_code(courses):
    first = courses.create_course("TM103", "A")
    courses.create_course("TM104", "B")
    with pytest.raises(DuplicateCourseError):
        courses.update_course(first, "TM104", "A")
    courses.update_course(first, "tm103", "Renamed")   # its own code in another case is fine
    assert courses.get_course(first)['name'] == "Renamed"


def test_deactivate_course_hides_its_sections_from_active_lists(courses):
    course_id = courses.create_course("TM103", "A")
    courses.create_section(course_id, "A")
    courses.set_course_active(course_id, False)
    assert courses.list_courses(active_only=True) == []
    assert courses.list_sections(active_only=True) == []
    assert len(courses.list_sections()) == 1


def test_delete_course_removes_sections_and_class_lists_not_students(courses, students, temp_db):
    course_id = courses.create_course("TM103", "A")
    section_id = courses.create_section(course_id, "A")
    courses.add_students(section_id, [_db_id(students, "2021001")])
    courses.delete_course(course_id)
    assert courses.list_courses() == [] and courses.list_sections() == []
    with get_db_connection(temp_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM section_students").fetchone()[0] == 0
    assert len(students.get_all_students()) == 3


def test_course_with_sessions_cannot_be_deleted(courses, temp_db):
    course_id = courses.create_course("TM103", "A")
    section_id = courses.create_section(course_id, "A")
    _add_session(temp_db, section_id)
    with pytest.raises(CourseInUseError):
        courses.delete_course(course_id)
    with pytest.raises(CourseInUseError):
        courses.delete_section(section_id)
    assert courses.list_courses()[0]['session_count'] == 1


def test_missing_course_or_section(courses):
    with pytest.raises(CourseNotFoundError):
        courses.create_section(999, "A")
    with pytest.raises(CourseNotFoundError):
        courses.add_students(999, [1])


# -- sections ----------------------------------------------------------------

def test_section_names_are_unique_per_course(courses):
    tm103 = courses.create_course("TM103", "A")
    tm104 = courses.create_course("TM104", "B")
    courses.create_section(tm103, "A", "Dr. Sara")
    courses.create_section(tm104, "A")             # same name in another course is fine
    with pytest.raises(DuplicateCourseError):
        courses.create_section(tm103, "a")
    second = courses.create_section(tm103, "B")
    with pytest.raises(DuplicateCourseError):
        courses.update_section(second, "A")


def test_get_section_has_course_and_counts(courses, students):
    course_id = courses.create_course("TM103", "Computer Organization")
    section_id = courses.create_section(course_id, "Lab 1", "Dr. Sara")
    courses.add_students(section_id, [_db_id(students, "2021001"), _db_id(students, "2021002")])
    section = courses.get_section(section_id)
    assert section['course_code'] == "TM103" and section['course_name'] == "Computer Organization"
    assert section['instructor_name'] == "Dr. Sara"
    assert section['student_count'] == 2 and section['session_count'] == 0


# -- class lists -------------------------------------------------------------

def test_add_and_remove_students(courses, students):
    section_id = courses.create_section(courses.create_course("TM103", "A"), "A")
    alice, omar = _db_id(students, "2021001"), _db_id(students, "2021002")
    assert courses.add_students(section_id, [alice, omar, alice]) == 2
    assert courses.add_students(section_id, [alice]) == 0          # already on the list
    assert [s['name'] for s in courses.get_class_list(section_id)] == ["Alice", "Omar"]
    assert [s['name'] for s in courses.get_class_list(section_id, search="2021002")] == ["Omar"]
    assert courses.remove_students(section_id, [alice]) == 1
    assert courses.get_class_list_student_ids(section_id) == {omar}


def test_student_sections(courses, students):
    tm103 = courses.create_course("TM103", "A")
    a, b = courses.create_section(tm103, "A"), courses.create_section(tm103, "B")
    alice = _db_id(students, "2021001")
    courses.add_students(a, [alice])
    courses.add_students(b, [alice])
    assert [s['name'] for s in courses.get_student_sections(alice)] == ["A", "B"]
    assert courses.list_courses()[0]['student_count'] == 1   # one student in two sections


def test_deleting_a_student_removes_them_from_class_lists(courses, students):
    section_id = courses.create_section(courses.create_course("TM103", "A"), "A")
    alice = _db_id(students, "2021001")
    courses.add_students(section_id, [alice])
    students.delete_student(alice)
    assert courses.get_class_list(section_id) == []


# -- class list import -------------------------------------------------------

def _table(rows):
    return pd.DataFrame(rows, columns=["course_code", "section", "student_id", "course_name", "instructor"]).fillna("")


def test_import_creates_courses_sections_and_adds_students(courses, students):
    existing = courses.create_course("TM104", "Networks")
    section_b = courses.create_section(existing, "B")
    courses.add_students(section_b, [_db_id(students, "2021003")])

    plan = plan_class_list_import(_table([
        ["TM103", "A", "2021001", "Computer Organization", "Dr. Sara"],
        ["tm103", "a", "2021002", "", ""],               # same new course and section, other case
        ["TM103", "A", "2021001", "", ""],               # repeated row
        ["TM104", "B", "2021003", "", ""],               # already on the class list
        ["TM104", "B", "2021001", "", ""],               # existing section
        ["TM104", "C", "2021002", "", "Dr. Omar"],       # new section of an existing course
        ["TM105", "A", "9999999", "", ""],               # unknown student
        ["", "A", "2021001", "", ""],                    # no course
    ]), courses, students)

    assert plan.counts == {"add": 4, "skip": 2, "error": 2}
    assert plan.new_courses == [{'course_code': "TM103", 'name': "Computer Organization"}]
    assert sorted((s['course_code'], s['name'], s['instructor_name']) for s in plan.new_sections) == [
        ("TM103", "A", "Dr. Sara"), ("TM104", "C", "Dr. Omar")]
    notes = {row.row_number: row.note for row in plan.with_action("skip")}
    assert notes == {4: "repeated in the file (row 2)", 5: "already on the class list"}
    assert "not in the Student Registry" in plan.with_action("error")[0].errors[0]

    outcome = apply_class_list_import(plan, courses)
    assert outcome == {"courses": 1, "sections": 2, "added": 4, "skipped": 2, "errors": 2}

    by_name = {(s['course_code'], s['name']): s for s in courses.list_sections()}
    assert by_name[("TM103", "A")]['student_count'] == 2
    assert by_name[("TM103", "A")]['instructor_name'] == "Dr. Sara"
    assert by_name[("TM104", "B")]['student_count'] == 2
    assert by_name[("TM104", "C")]['student_count'] == 1

    # The same file again adds nothing
    again = plan_class_list_import(_table([["TM103", "A", "2021001", "", ""]]), courses, students)
    assert again.counts == {"add": 0, "skip": 1, "error": 0} and not again.new_courses


def test_new_course_without_name_uses_its_code(courses, students):
    plan = plan_class_list_import(_table([["TM200", "A", "2021001", "", ""]]), courses, students)
    assert plan.new_courses == [{'course_code': "TM200", 'name': "TM200"}]


def test_read_class_list_accepts_other_header_spellings():
    data = "Course,Group,Student ID,Teacher,Extra\nTM103,A,2021001,Dr. Sara,x\n".encode("utf-8-sig")
    table, ignored = read_class_list_table(data, "list.csv")
    assert list(table.columns) == ["course_code", "section", "student_id", "instructor"]
    assert ignored == ["Extra"]


def test_template_and_errors_csv(courses, students):
    table, _ = read_class_list_table(class_list_template_csv(), "template.csv")
    assert len(table) == 3
    plan = plan_class_list_import(_table([["TM103", "A", "nope", "", ""]]), courses, students)
    assert "not in the Student Registry" in class_list_errors_csv(plan).decode("utf-8-sig")
