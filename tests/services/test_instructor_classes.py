"""
Tests for instructors seeing only their own classes: sections linked to
instructor accounts, who owns which session, the section summary and
pages per role, and the class-list import linking instructors by name.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.database.schema import initialize_database
from src.services.auth_service import AuthService, Role
from src.services.class_list_import_service import apply_class_list_import, plan_class_list_import
from src.services.course_service import CourseService, InvalidCourseDataError
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService
from src.ui.access import owns_session, pages_for, sees_everything, visible_sessions

PASSWORD = "Secret-123"


@pytest.fixture
def world():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db = Path(f.name)
    initialize_database(db)
    auth = AuthService(db)
    it = auth.repository.get_by_id(auth.create_first_account("it.boss", "IT Boss", PASSWORD))
    sara = auth.repository.get_by_id(auth.create_user(it, "sara", "Dr. Sara Ali", Role.INSTRUCTOR, PASSWORD))
    omar = auth.repository.get_by_id(auth.create_user(it, "omar", "Dr. Omar Nabil", Role.INSTRUCTOR, PASSWORD))
    courses = CourseService(db)
    course = courses.create_course("TM103", "Computer Organization")
    a = courses.create_section(course, "A", instructor_id=sara['id'])
    b = courses.create_section(course, "B", instructor_id=omar['id'])
    c = courses.create_section(course, "C", "Dr. Guest")                 # typed name, no account
    yield {'db': db, 'auth': auth, 'it': it, 'sara': sara, 'omar': omar, 'courses': courses,
           'sessions': SessionService(db), 'a': a, 'b': b, 'c': c}
    db.unlink()


def test_linked_section_takes_the_account_name(world):
    section = world['courses'].get_section(world['a'])
    assert section['instructor_id'] == world['sara']['id']
    assert section['instructor_name'] == "Dr. Sara Ali"
    typed = world['courses'].get_section(world['c'])
    assert typed['instructor_id'] is None and typed['instructor_name'] == "Dr. Guest"


def test_deactivated_account_cannot_be_linked(world):
    world['auth'].set_active(world['it'], world['omar']['id'], False)
    with pytest.raises(InvalidCourseDataError):
        world['courses'].create_section(world['courses'].get_section(world['a'])['course_id'], "D",
                                        instructor_id=world['omar']['id'])


def test_relinking_and_unlinking_a_section(world):
    courses = world['courses']
    courses.update_section(world['c'], "C", notes=None, instructor_id=world['sara']['id'])
    assert courses.get_section(world['c'])['instructor_name'] == "Dr. Sara Ali"
    courses.update_section(world['c'], "C", "Dr. Guest", None, instructor_id=None)
    section = courses.get_section(world['c'])
    assert section['instructor_id'] is None and section['instructor_name'] == "Dr. Guest"


def test_renaming_an_account_renames_its_sections(world):
    world['auth'].update_user(world['it'], world['sara']['id'], "Prof. Sara Ali", Role.INSTRUCTOR)
    assert world['courses'].get_section(world['a'])['instructor_name'] == "Prof. Sara Ali"


def test_instructor_sections(world):
    names = [s['name'] for s in world['courses'].instructor_sections(world['sara']['id'])]
    assert names == ["A"]
    world['courses'].set_section_active(world['a'], False)
    assert world['courses'].instructor_sections(world['sara']['id']) == []


# -- who owns which session -------------------------------------------------------------

def test_sessions_carry_their_owners(world):
    sessions = world['sessions']
    on_a = sessions.start_section_session(world['a'], created_by="Dr. Sara Ali",
                                          created_by_user_id=world['sara']['id'])
    session = sessions.get_session(on_a)
    assert session['section_instructor_id'] == world['sara']['id']
    assert session['created_by_user_id'] == world['sara']['id']
    overview = next(s for s in sessions.get_sessions_overview() if s['id'] == on_a)
    assert overview['section_instructor_id'] == world['sara']['id']


def test_each_instructor_sees_only_their_sessions(world):
    sessions = world['sessions']
    on_a = sessions.create_session("A lecture", "2026-11-03 10:00:00", section_id=world['a'],
                                   created_by_user_id=world['it']['id'])     # created by IT, Sara's section
    on_b = sessions.create_session("B lecture", "2026-11-03 10:00:00", section_id=world['b'])
    own = sessions.create_session("Extra", "2026-11-03 12:00:00", created_by_user_id=world['sara']['id'])
    other = sessions.create_session("Other", "2026-11-03 12:00:00")
    all_sessions = sessions.get_sessions_overview()

    sara_ids = {s['id'] for s in visible_sessions(world['sara'], all_sessions)}
    omar_ids = {s['id'] for s in visible_sessions(world['omar'], all_sessions)}
    it_ids = {s['id'] for s in visible_sessions(world['it'], all_sessions)}
    assert sara_ids == {on_a, own}
    assert omar_ids == {on_b}
    assert it_ids == {on_a, on_b, own, other}
    assert not owns_session(world['sara'], sessions.get_session(on_b))
    assert not owns_session(None, sessions.get_session(on_a))


def test_moving_a_section_to_another_instructor_moves_its_sessions(world):
    sessions = world['sessions']
    on_a = sessions.create_session("A lecture", "2026-11-03 10:00:00", section_id=world['a'])
    world['courses'].update_section(world['a'], "A", instructor_id=world['omar']['id'])
    session = sessions.get_session(on_a)
    assert owns_session(world['omar'], session) and not owns_session(world['sara'], session)


def test_roles_and_pages(world):
    assert sees_everything(world['it']) and not sees_everything(world['sara'])
    assert "My Classes" in pages_for(world['sara'])
    assert "Courses & Sections" not in pages_for(world['sara'])


# -- class list import links instructors -------------------------------------------------

def test_import_links_instructors_by_username_or_name(world):
    students = StudentService(world['db'])
    for sid in ("1", "2", "3"):
        students.create_student(sid, f"Student {sid}")
    table = pd.DataFrame([
        ["CS220", "Lab 1", "1", "Data Structures", "sara"],            # username
        ["CS220", "Lab 2", "2", "", "dr. omar nabil"],                  # full name, other case
        ["CS220", "Lab 3", "3", "", "Eng. Visitor"],                    # no account
    ], columns=["course_code", "section", "student_id", "course_name", "instructor"])
    plan = plan_class_list_import(table, world['courses'], students)
    apply_class_list_import(plan, world['courses'])
    by_name = {s['name']: s for s in world['courses'].list_sections()}
    assert by_name["Lab 1"]['instructor_id'] == world['sara']['id']
    assert by_name["Lab 1"]['instructor_name'] == "Dr. Sara Ali"
    assert by_name["Lab 2"]['instructor_id'] == world['omar']['id']
    assert by_name["Lab 3"]['instructor_id'] is None and by_name["Lab 3"]['instructor_name'] == "Eng. Visitor"


# -- older databases ---------------------------------------------------------------------

def test_old_tables_get_the_owner_columns(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE courses (id INTEGER PRIMARY KEY AUTOINCREMENT, course_code TEXT UNIQUE NOT NULL, "
                 "name TEXT NOT NULL, department TEXT, is_active BOOLEAN DEFAULT 1, created_at TIMESTAMP, notes TEXT)")
    conn.execute("CREATE TABLE sections (id INTEGER PRIMARY KEY AUTOINCREMENT, course_id INTEGER NOT NULL, "
                 "name TEXT NOT NULL, instructor_name TEXT, is_active BOOLEAN DEFAULT 1, created_at TIMESTAMP, "
                 "notes TEXT)")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_name TEXT NOT NULL, "
                 "course_code TEXT, start_time TIMESTAMP NOT NULL, end_time TIMESTAMP, "
                 "late_threshold_minutes INTEGER DEFAULT 15, status TEXT DEFAULT 'scheduled', created_by TEXT, "
                 "created_at TIMESTAMP, closed_at TIMESTAMP, notes TEXT, section_id INTEGER)")
    conn.execute("INSERT INTO courses (course_code, name) VALUES ('TM103', 'X')")
    conn.execute("INSERT INTO sections (course_id, name, instructor_name) VALUES (1, 'A', 'Dr. Typed')")
    conn.commit()
    conn.close()
    assert initialize_database(db)
    section = CourseService(db).get_section(1)
    assert section['instructor_id'] is None and section['instructor_name'] == "Dr. Typed"
    session_id = SessionService(db).create_session("S", "2026-11-03 10:00:00", section_id=1)
    assert SessionService(db).get_session(session_id)['created_by_user_id'] is None
