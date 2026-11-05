"""
Tests for the activity log: who is recorded (the signed-in user, System for
background work, the person signing in), what each kind of change writes,
that secrets never reach the log, the filters, and that a broken log never
breaks the change itself.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.database.schema import initialize_database
from src.services import activity_log as activity_module
from src.services.activity_log import SYSTEM_ACTOR, ActivityLog, Category, acting_as, set_actor_provider
from src.services.attendance_service import AttendanceService
from src.services.auth_service import AccountLockedError, AuthService, InvalidCredentialsError, Role
from src.services.bulk_import_service import apply_student_import, plan_student_import
from src.services.check_service import CheckService
from src.services.course_service import CourseService
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService

PASSWORD = "Secret-123"
FMT = '%Y-%m-%d %H:%M:%S'


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = Path(f.name)
    initialize_database(path)
    yield path
    set_actor_provider(None)
    path.unlink()


@pytest.fixture
def people(db):
    auth = AuthService(db)
    it = auth.repository.get_by_id(auth.create_first_account("it.boss", "IT Boss", PASSWORD))
    teacher = auth.repository.get_by_id(auth.create_user(it, "sara", "Dr. Sara Ali", Role.INSTRUCTOR, PASSWORD))
    return {'auth': auth, 'it': it, 'teacher': teacher}


def _signed_in(user):
    """Act as if `user` is signed in on the page making the change."""
    set_actor_provider(lambda: user)


def _entries(db, **filters):
    return ActivityLog(db).search(**filters)


def _actions(db, **filters):
    return [e['action'] for e in reversed(_entries(db, **filters))]


# -- who ------------------------------------------------------------------------------

def test_background_work_is_system_and_a_page_is_its_signed_in_user(db, people):
    log = ActivityLog(db)
    log.record(Category.SETTINGS, 'test', "no page")
    _signed_in(people['teacher'])
    log.record(Category.SETTINGS, 'test', "from a page")
    with acting_as(SYSTEM_ACTOR):
        log.record(Category.SETTINGS, 'test', "scheduler inside a page process")
    by_summary = {e['summary']: e for e in _entries(db, category=Category.SETTINGS)}
    assert by_summary["no page"]['full_name'] == "System (automatic)" and by_summary["no page"]['user_id'] is None
    assert by_summary["from a page"]['username'] == "sara" and by_summary["from a page"]['role'] == Role.INSTRUCTOR
    assert by_summary["scheduler inside a page process"]['full_name'] == "System (automatic)"


def test_a_broken_log_never_breaks_the_change(db, monkeypatch):
    def broken(entry):
        raise RuntimeError("disk full")
    log = ActivityLog(db)
    monkeypatch.setattr(log.repository, 'add', broken)
    assert log.record(Category.SETTINGS, 'test', "x") is None


# -- sign-in and accounts ---------------------------------------------------------------

def test_sign_ins_wrong_passwords_and_lockout_are_logged_without_secrets(db, people):
    auth = people['auth']
    auth.authenticate("sara", PASSWORD)
    with pytest.raises(InvalidCredentialsError):
        auth.authenticate("nobody", "typed-a-secret")
    for _ in range(4):
        with pytest.raises(InvalidCredentialsError):
            auth.authenticate("sara", "wrong-pass-1")
    with pytest.raises(AccountLockedError):
        auth.authenticate("sara", "wrong-pass-1")

    entries = _entries(db, category=Category.SIGN_IN)
    actions = [e['action'] for e in reversed(entries)]
    assert actions[0] == 'sign_in' and 'account_locked' in actions
    assert actions.count('sign_in_failed') == 6           # unknown username + 5 wrong passwords
    unknown = next(e for e in entries if 'unknown username' in e['summary'])
    assert unknown['full_name'] == "Unknown (not signed in)" and '"nobody"' in unknown['summary']
    everything = json.dumps(ActivityLog(db).search(limit=10000))
    assert PASSWORD not in everything and "wrong-pass-1" not in everything and "typed-a-secret" not in everything


def test_account_changes_name_who_made_them(db, people):
    auth, it, teacher = people['auth'], people['it'], people['teacher']
    auth.update_user(it, teacher['id'], "Dr. Sara Ali Hassan", Role.ADMIN)
    auth.reset_password(it, teacher['id'], "Temp-4567")
    auth.set_active(it, teacher['id'], False)
    entries = _entries(db, category=Category.ACCOUNTS, username="it.boss")
    by_action = {e['action']: e for e in entries}
    assert {'first_account', 'account_created', 'account_updated', 'password_reset',
            'account_deactivated'} <= set(by_action)
    assert 'role: "instructor" → "admin"' in by_action['account_updated']['summary']
    assert "Temp-4567" not in json.dumps(entries)
    assert by_action['account_created']['target_type'] == 'user'


# -- sessions, attendance by hand, checks ---------------------------------------------------

@pytest.fixture
def world(db, people):
    _signed_in(people['teacher'])
    students = StudentService(db)
    ids = {name: students.create_student(str(i), name) for i, name in enumerate(["Alice", "Omar"], 1)}
    sessions = SessionService(db)
    start = datetime.now().replace(microsecond=0) - timedelta(minutes=30)
    session_id = sessions.create_session("Lab 1", start.strftime(FMT),
                                         end_time=(start + timedelta(hours=2)).strftime(FMT),
                                         status=SessionStatus.ACTIVE)
    return {'db': db, 'ids': ids, 'session': session_id, 'sessions': sessions, 'start': start,
            'attendance': AttendanceService(db), 'checks': CheckService(db)}


def test_session_changes_are_logged_with_what_changed(world):
    db, sessions, session_id = world['db'], world['sessions'], world['session']
    sessions.update_session(session_id, session_name="Lab 1 (moved)", late_threshold_minutes=20)
    sessions.close_session(session_id)
    sessions.delete_session(session_id)
    entries = _entries(db, session_id=session_id, category=Category.SESSIONS)
    by_action = {e['action']: e for e in entries}
    assert ['session_created', 'session_updated', 'session_closed', 'session_deleted'] == \
        [e['action'] for e in reversed(entries)]
    assert 'name: "Lab 1" → "Lab 1 (moved)"' in by_action['session_updated']['summary']
    assert 'late after (min): "15" → "20"' in by_action['session_updated']['summary']
    assert all(e['username'] == "sara" for e in entries)


def test_scheduler_start_and_close_are_system(db, people):
    sessions = SessionService(db)
    now = datetime.now().replace(microsecond=0)
    session_id = sessions.create_session("Later", (now + timedelta(minutes=1)).strftime(FMT),
                                         end_time=(now + timedelta(minutes=5)).strftime(FMT))
    sessions.apply_scheduled_transitions(now=now + timedelta(minutes=2))
    sessions.apply_scheduled_transitions(now=now + timedelta(minutes=6))
    auto = [e for e in _entries(db, session_id=session_id) if e['action'].endswith('_auto')]
    assert {e['action'] for e in auto} == {'session_started_auto', 'session_closed_auto'}
    assert all(e['full_name'] == "System (automatic)" for e in auto)


def test_attendance_by_hand_is_logged_per_student(world):
    db, attendance, session_id, ids = world['db'], world['attendance'], world['session'], world['ids']
    attendance.set_attendance_manually(session_id, ids["Alice"], "late")
    attendance.set_attendance_manually(session_id, ids["Alice"], "present")
    attendance.set_left_early(session_id, ids["Alice"], True)
    attendance.set_left_early(session_id, ids["Alice"], True)          # no change: not logged again
    attendance.set_left_early(session_id, ids["Alice"], False)
    attendance.set_attendance_manually(session_id, ids["Alice"], "absent")
    entries = _entries(db, category=Category.ATTENDANCE, session_id=session_id)
    assert [e['action'] for e in reversed(entries)] == [
        'attendance_marked', 'attendance_changed', 'left_early_set', 'left_early_cleared', 'attendance_removed']
    assert "Changed Alice (1) from late to present" in entries[3]['summary']
    assert all(e['target_type'] == 'student' and e['target_id'] == ids["Alice"] for e in entries)


def test_camera_marks_are_not_logged(world):
    world['attendance'].record_attendance(world['session'], world['ids']["Omar"], marked_by='cv_realtime')
    assert _entries(world['db'], category=Category.ATTENDANCE) == []


def test_checks_are_logged_and_random_times_stay_hidden(world):
    db, checks, session_id = world['db'], world['checks'], world['session']
    world['attendance'].set_attendance_manually(session_id, world['ids']["Alice"], "present")
    later = datetime.now() + timedelta(seconds=5)                  # Alice checked in before the check
    check_id = checks.start_spot_check(session_id, 5, now=later)
    checks.camera_saw(check_id, now=later)
    assert checks.finish_check_now(check_id, now=later + timedelta(minutes=1)) == 1
    random_id = checks.schedule_spot_check(session_id, randomly=True, minutes=5)
    hidden = checks.repository.get_check(random_id)['starts_at']
    entries = _entries(db, category=Category.CHECKS)
    by_action = {e['action']: e for e in entries}
    assert {'spot_check_started', 'check_done', 'spot_check_scheduled'} <= set(by_action)
    assert "0 of 1 seen, 1 marked 🚪 Left early" in by_action['check_done']['summary']
    assert by_action['check_done']['username'] == "sara"            # finished by hand
    assert "random time (hidden)" in by_action['spot_check_scheduled']['summary']
    assert hidden not in json.dumps(entries) and hidden[11:16] not in by_action['spot_check_scheduled']['summary']


def test_a_check_ending_by_itself_is_system(world):
    db, checks, session_id = world['db'], world['checks'], world['session']
    check_id = checks.start_spot_check(session_id, 5)
    checks.apply_due(now=datetime.now() + timedelta(minutes=6))
    ended = next(e for e in _entries(db, category=Category.CHECKS) if e['target_id'] == check_id
                 and e['action'] == 'check_not_counted')
    assert ended['full_name'] == "System (automatic)" and "camera was off" in ended['summary']


# -- students, courses, imports ----------------------------------------------------------------

def test_students_courses_and_class_lists(db, people):
    _signed_in(people['it'])
    students, courses = StudentService(db), CourseService(db)
    alice = students.create_student("2021001", "Alice")
    students.update_student(alice, email="alice@uni.edu")
    course = courses.create_course("CS220", "Data Structures")
    section = courses.create_section(course, "Lab 1")
    courses.add_students(section, [alice])
    courses.remove_students(section, [alice])
    students.delete_student(alice)
    summaries = [e['summary'] for e in reversed(_entries(db))]
    assert "Added student Alice (2021001)" in summaries
    assert 'Changed student Alice (2021001): email: — → "alice@uni.edu"' in summaries
    assert "Added course CS220 - Data Structures" in summaries
    assert "Added 1 student(s) to the class list of CS220 / Lab 1: Alice (2021001)" in summaries
    assert "Removed 1 student(s) from the class list of CS220 / Lab 1: Alice (2021001)" in summaries
    assert any(s.startswith("Deleted student Alice (2021001)") for s in summaries)
    student_history = _entries(db, target=('student', alice))
    assert len(student_history) == 3                                 # added, changed, deleted


def test_a_bulk_import_is_one_entry(db, people):
    _signed_in(people['it'])
    table = pd.DataFrame({'student_id': [str(3000 + i) for i in range(50)],
                          'name': [f"Student {i}" for i in range(50)]})
    plan = plan_student_import(table, StudentService(db))
    apply_student_import(plan, StudentService(db))
    entries = _entries(db, category=Category.STUDENTS)
    assert len(entries) == 1 and entries[0]['action'] == 'students_imported'
    assert "50 added" in entries[0]['summary']


# -- filters ---------------------------------------------------------------------------------

def test_filters(db, people):
    log = ActivityLog(db)
    _signed_in(people['teacher'])
    log.record(Category.SETTINGS, 'camera_chosen', "Chose the camera: Brio 100")
    set_actor_provider(None)
    log.record(Category.SESSIONS, 'session_closed_auto', "Session #9 closed by itself", session_id=9)
    assert [e['summary'] for e in log.search(search="brio")] == ["Chose the camera: Brio 100"]
    assert [e['action'] for e in log.search(full_name="System (automatic)")] == ['session_closed_auto']
    assert log.count(username="sara", category=Category.SETTINGS) == 1
    assert log.count(session_id=9) == 1
    assert log.count(actions=('camera_chosen', 'nothing')) == 1
    tomorrow = (datetime.now() + timedelta(days=1)).strftime(FMT)
    assert log.count(date_from=tomorrow) == 0
    assert {p['username'] for p in log.people()} == {"it.boss", "sara"}
