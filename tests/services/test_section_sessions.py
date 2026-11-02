"""
Tests for sessions linked to a section: who is expected and counted absent
(Session Details, statistics, missed sessions), the session service's
section handling (create, edit, "start now", overlaps) and the per-section
summary. Sessions without a section keep the old rule: every active
student registered before the start is expected.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.database.connection import get_db_connection
from src.database.schema import initialize_database
from src.services.attendance_service import AttendanceService
from src.services.course_service import CourseService
from src.services.session_service import InvalidSessionDataError, SessionService, SessionStatus
from src.services.statistics_service import StatisticsService
from src.services.student_service import StudentService

REGISTERED = '2026-10-01 08:00:00'   # UTC, before every session below
LISTED = '2026-10-02 08:00:00'       # UTC, added to the class list before the sessions


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def world(temp_db):
    """
    5 students; section TM103/A lists Alice and Omar, TM103/B lists Sara.
    Huda and Kareem are on no class list.
    """
    students = StudentService(temp_db)
    ids = {name: students.create_student(sid, name) for sid, name in
           [("1", "Alice"), ("2", "Omar"), ("3", "Sara"), ("4", "Huda"), ("5", "Kareem")]}
    courses = CourseService(temp_db)
    course = courses.create_course("TM103", "Computer Organization")
    a = courses.create_section(course, "A", "Dr. Sara Ali")
    b = courses.create_section(course, "B")
    courses.add_students(a, [ids["Alice"], ids["Omar"]])
    courses.add_students(b, [ids["Sara"]])
    with get_db_connection(temp_db) as conn:
        conn.execute("UPDATE students SET enrolled_at = ?", (REGISTERED,))
        conn.execute("UPDATE section_students SET added_at = ?", (LISTED,))
    return {'db': temp_db, 'ids': ids, 'a': a, 'b': b, 'courses': courses,
            'sessions': SessionService(temp_db), 'attendance': AttendanceService(temp_db),
            'stats': StatisticsService(temp_db)}


def _session(world, section=None, status=SessionStatus.ACTIVE, start='2026-10-10 10:00:00'):
    return world['sessions'].create_session("S", start, status=status, section_id=section)


def _roster(world, session_id):
    return {r['name']: r['status'] for r in world['attendance'].get_session_roster(session_id)}


# -- who is expected ---------------------------------------------------------

def test_section_session_expects_only_its_class_list(world):
    session_id = _session(world, world['a'])
    assert _roster(world, session_id) == {"Alice": "absent", "Omar": "absent"}
    counts = world['attendance'].get_session_roster_counts(session_id)
    assert counts['expected'] == 2 and counts['absent'] == 2


def test_session_without_section_still_expects_everyone(world):
    session_id = _session(world)
    assert len(_roster(world, session_id)) == 5


def test_student_marked_by_hand_but_not_listed_is_counted(world):
    session_id = _session(world, world['a'])
    world['attendance'].set_attendance_manually(session_id, world['ids']["Huda"], "present")
    world['attendance'].set_attendance_manually(session_id, world['ids']["Alice"], "late")
    assert _roster(world, session_id) == {"Huda": "present", "Alice": "late", "Omar": "absent"}


def test_students_added_to_the_list_after_the_start_are_not_expected(world):
    session_id = _session(world, world['a'])
    world['courses'].add_students(world['a'], [world['ids']["Kareem"]])   # added now, after 2026-10-10
    assert "Kareem" not in _roster(world, session_id)


def test_inactive_listed_student_is_not_expected(world):
    StudentService(world['db']).deactivate_student(world['ids']["Omar"])
    session_id = _session(world, world['a'])
    assert _roster(world, session_id) == {"Alice": "absent"}


# -- statistics ----------------------------------------------------------------

def test_session_statistics_use_the_class_list(world):
    session_id = _session(world, world['a'])
    world['attendance'].set_attendance_manually(session_id, world['ids']["Alice"], "present")
    world['sessions'].close_session(session_id)
    stats = world['stats'].get_session_statistics(session_id)
    assert stats['expected_students'] == 2 and stats['absent_count'] == 1
    assert stats['attendance_percentage'] == 50.0
    assert stats['section_name'] == "A" and stats['instructor_name'] == "Dr. Sara Ali"

    [row] = [s for s in world['stats'].get_all_sessions_statistics() if s['session_id'] == session_id]
    assert row['expected_students'] == 2 and row['absent_count'] == 1


def test_student_absences_count_only_their_own_sections(world):
    ids = world['ids']
    a_session = _session(world, world['a'])
    b_session = _session(world, world['b'])
    open_session = _session(world)                       # no section: everyone expected
    world['attendance'].set_attendance_manually(a_session, ids["Alice"], "present")
    for session_id in (a_session, b_session, open_session):
        world['sessions'].close_session(session_id)

    by_name = {s['name']: s for s in world['stats'].get_all_students_statistics()}
    assert (by_name["Alice"]['expected_sessions'], by_name["Alice"]['absent_count']) == (2, 1)  # A + open
    assert (by_name["Omar"]['expected_sessions'], by_name["Omar"]['absent_count']) == (2, 2)    # A + open
    assert (by_name["Sara"]['expected_sessions'], by_name["Sara"]['absent_count']) == (2, 2)    # B + open
    assert (by_name["Huda"]['expected_sessions'], by_name["Huda"]['absent_count']) == (1, 1)    # open only

    single = world['stats'].get_student_statistics(ids["Omar"])
    assert single['absent_count'] == 2
    assert {m['session_id'] for m in single['missed_sessions']} == {a_session, open_session}
    assert {m['session_id'] for m in world['stats'].get_student_statistics(ids["Huda"])['missed_sessions']} \
        == {open_session}


def test_overall_and_section_summaries(world):
    ids = world['ids']
    a_session = _session(world, world['a'])
    world['attendance'].set_attendance_manually(a_session, ids["Alice"], "present")
    world['sessions'].close_session(a_session)
    _session(world, world['b'])                           # still running: not in absence totals

    overall = world['stats'].get_overall_dashboard_stats()
    assert overall['total_expected'] == 2 and overall['total_absent'] == 1

    rows = {r['section_name']: r for r in world['stats'].get_section_statistics()}
    assert rows["A"]['total_sessions'] == 1 and rows["A"]['absent_count'] == 1
    assert rows["A"]['attendance_rate'] == 50.0 and rows["A"]['class_list_size'] == 2
    assert rows["B"]['total_sessions'] == 1 and rows["B"]['absent_count'] == 0


# -- session service -------------------------------------------------------------

def test_create_with_section_takes_the_course_code(world):
    session_id = world['sessions'].create_session("Lecture", "2026-10-10 10:00:00", course_code="IGNORED",
                                                  section_id=world['a'])
    session = world['sessions'].get_session(session_id)
    assert session['course_code'] == "TM103" and session['section_id'] == world['a']
    assert session['section_name'] == "A" and session['instructor_name'] == "Dr. Sara Ali"


def test_inactive_or_missing_section_is_refused(world):
    world['courses'].set_section_active(world['b'], False)
    with pytest.raises(InvalidSessionDataError, match="inactive"):
        _session(world, world['b'])
    with pytest.raises(InvalidSessionDataError, match="not found"):
        _session(world, 999)


def test_start_section_session_now(world):
    now = datetime(2026, 10, 28, 14, 5, 30)
    session_id = world['sessions'].start_section_session(world['a'], now=now)
    session = world['sessions'].get_session(session_id)
    assert session['status'] == SessionStatus.ACTIVE
    assert session['start_time'] == "2026-10-28 14:05:30"    # seconds kept
    assert session['session_name'] == "TM103 A - 28 Oct 14:05"
    assert session['section_id'] == world['a'] and session['course_code'] == "TM103"


def test_edit_session_section(world):
    session_id = _session(world, world['a'])
    world['sessions'].update_session(session_id, section_id=world['b'])
    assert _roster(world, session_id) == {"Sara": "absent"}

    # Keeping a section that was deactivated meanwhile is fine
    world['courses'].set_section_active(world['b'], False)
    world['sessions'].update_session(session_id, session_name="Renamed", section_id=world['b'])

    world['sessions'].update_session(session_id, clear_section=True)
    assert world['sessions'].get_session(session_id)['section_id'] is None
    assert len(_roster(world, session_id)) == 5


def test_overlaps_are_per_section(world):
    sessions = world['sessions']
    first = _session(world, world['a'], status=SessionStatus.SCHEDULED, start='2026-11-01 10:00:00')
    # Section B of the same course at the same time is normal
    assert sessions.find_overlapping_sessions('2026-11-01 10:00:00', None, "TM103", section_id=world['b']) == []
    overlapping = sessions.find_overlapping_sessions('2026-11-01 10:00:00', None, "TM103", section_id=world['a'])
    assert [s['id'] for s in overlapping] == [first]


def test_sections_with_sessions_cannot_be_deleted(world):
    from src.services.course_service import CourseInUseError
    _session(world, world['a'])
    with pytest.raises(CourseInUseError):
        world['courses'].delete_section(world['a'])
