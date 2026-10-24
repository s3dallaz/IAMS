"""
Tests for automatic session start / close, status-change races, edit rules,
deletion, overlap detection and the session roster (absent students).

Uses isolated temporary databases.
"""

import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import config
from src.database.schema import initialize_database
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.session_service import (
    SessionService,
    SessionStatus,
    InvalidSessionDataError,
    InvalidSessionStateError,
)
from src.services.student_service import StudentService
from src.services import session_scheduler

FMT = '%Y-%m-%d %H:%M:%S'
NOW = datetime(2026, 10, 24, 10, 0, 0)


def at(minutes: int) -> str:
    """NOW + minutes, in the stored format."""
    return (NOW + timedelta(minutes=minutes)).strftime(FMT)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sessions(temp_db):
    return SessionService(temp_db)


@pytest.fixture
def attendance(temp_db):
    return AttendanceService(temp_db)


@pytest.fixture
def students(temp_db):
    return StudentService(temp_db)


def status_of(service: SessionService, session_id: int) -> str:
    return service.get_session(session_id)['status']


# ============================================================================
# Automatic start / close
# ============================================================================

def test_due_scheduled_session_starts(sessions):
    due = sessions.create_session("Due", at(-1))
    later = sessions.create_session("Later", at(30))

    changed = sessions.apply_scheduled_transitions(now=NOW)

    assert changed == {'started': [due], 'closed': []}
    assert status_of(sessions, due) == SessionStatus.ACTIVE
    assert status_of(sessions, later) == SessionStatus.SCHEDULED


def test_session_starts_exactly_at_its_start_time(sessions):
    session_id = sessions.create_session("On time", at(0))
    sessions.apply_scheduled_transitions(now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.ACTIVE


def test_active_session_closes_at_its_end_time(sessions):
    session_id = sessions.create_session("Lecture", at(-60), end_time=at(0), status=SessionStatus.ACTIVE)

    changed = sessions.apply_scheduled_transitions(now=NOW)

    session = sessions.get_session(session_id)
    assert changed['closed'] == [session_id]
    assert session['status'] == SessionStatus.CLOSED
    assert session['closed_at'] is not None


def test_session_without_end_time_stays_active(sessions):
    session_id = sessions.create_session("Open ended", at(-600), status=SessionStatus.ACTIVE)
    sessions.apply_scheduled_transitions(now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.ACTIVE


def test_window_missed_while_app_was_down_closes_without_starting(sessions):
    """Start and end both passed (e.g. the app was not running): closed, never active."""
    session_id = sessions.create_session("Missed", at(-120), end_time=at(-60))

    changed = sessions.apply_scheduled_transitions(now=NOW)

    assert changed == {'started': [], 'closed': [session_id]}
    assert status_of(sessions, session_id) == SessionStatus.CLOSED


def test_start_time_passed_while_app_was_down_starts_on_catch_up(sessions):
    session_id = sessions.create_session("Catch up", at(-45), end_time=at(30))
    sessions.apply_scheduled_transitions(now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.ACTIVE


def test_transitions_are_idempotent(sessions):
    sessions.create_session("Due", at(-1))
    sessions.apply_scheduled_transitions(now=NOW)
    assert sessions.apply_scheduled_transitions(now=NOW) == {'started': [], 'closed': []}


def test_closed_session_is_never_restarted(sessions):
    session_id = sessions.create_session("Cancelled", at(-5), status=SessionStatus.CLOSED)
    sessions.apply_scheduled_transitions(now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.CLOSED


def test_auto_close_setting_off_keeps_sessions_open(sessions, temp_db):
    with closing(sqlite3.connect(temp_db)) as conn, conn:
        conn.execute("UPDATE settings SET value = 'false' WHERE key = 'auto_close_sessions'")
    session_id = sessions.create_session("No auto close", at(-60), end_time=at(-1), status=SessionStatus.ACTIVE)

    sessions.apply_scheduled_transitions(now=NOW)

    assert status_of(sessions, session_id) == SessionStatus.ACTIVE


def test_times_in_t_format_and_with_offsets_are_normalised(sessions):
    session_id = sessions.create_session("ISO", (NOW - timedelta(minutes=1)).isoformat())
    assert sessions.get_session(session_id)['start_time'] == at(-1)

    sessions.apply_scheduled_transitions(now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.ACTIVE

    aware = datetime(2026, 10, 24, 8, 0).astimezone()  # local time with an offset
    other = sessions.create_session("Aware", aware.isoformat())
    assert sessions.get_session(other)['start_time'] == "2026-10-24 08:00:00"


# ============================================================================
# Compare-and-set status changes
# ============================================================================

def test_manual_start_after_automatic_start_is_rejected(sessions):
    session_id = sessions.create_session("Due", at(-1))
    sessions.apply_scheduled_transitions(now=NOW)

    with pytest.raises(InvalidSessionStateError):
        sessions.activate_session(session_id)


def test_stale_start_cannot_reopen_a_closed_session(sessions, temp_db):
    """A start decided on stale data (read 'scheduled') must not reopen a session closed meanwhile."""
    session_id = sessions.create_session("Race", at(10))
    sessions.close_session(session_id)

    moved = sessions.repository.transition_status(session_id, {SessionStatus.SCHEDULED}, SessionStatus.ACTIVE)

    assert moved == 0
    assert status_of(sessions, session_id) == SessionStatus.CLOSED


def test_close_after_automatic_close_reports_already_closed(sessions):
    session_id = sessions.create_session("Ends", at(-60), end_time=at(-1), status=SessionStatus.ACTIVE)
    sessions.apply_scheduled_transitions(now=NOW)

    with pytest.raises(InvalidSessionStateError, match="already closed"):
        sessions.close_session(session_id)


def test_manual_early_start_keeps_the_scheduled_start(sessions):
    session_id = sessions.create_session("Early", at(60))
    sessions.activate_session(session_id, now=NOW)
    sessions.apply_scheduled_transitions(now=NOW)
    session = sessions.get_session(session_id)
    assert session['status'] == SessionStatus.ACTIVE
    assert session['start_time'] == at(60)


def test_manual_late_start_starts_fresh(sessions, attendance, students):
    """Started an hour late by hand: lateness counts from the real start, not the schedule."""
    with_auto_close_off(sessions)
    student = students.create_student("S1", "Ann")
    session_id = sessions.create_session("Late start", at(-60), late_threshold_minutes=15)

    sessions.activate_session(session_id, now=NOW)

    assert sessions.get_session(session_id)['start_time'] == at(0)
    attendance.record_attendance(session_id, student, marked_at=at(5))
    assert attendance.get_session_attendance(session_id)[0]['status'] == AttendanceStatus.PRESENT


def test_manual_start_after_the_end_time_is_rejected(sessions):
    with_auto_close_off(sessions)
    session_id = sessions.create_session("Too late", at(-60), end_time=at(-1))
    with pytest.raises(InvalidSessionStateError, match="end time"):
        sessions.activate_session(session_id, now=NOW)
    assert status_of(sessions, session_id) == SessionStatus.SCHEDULED


def with_auto_close_off(sessions: SessionService) -> None:
    """Keep the scheduler from closing past sessions, so a manual start can be tested on them."""
    with closing(sqlite3.connect(sessions.repository.database_path)) as conn, conn:
        conn.execute("UPDATE settings SET value = 'false' WHERE key = 'auto_close_sessions'")


# ============================================================================
# Edit rules
# ============================================================================

def test_rescheduling_a_scheduled_session_moves_its_start(sessions):
    session_id = sessions.create_session("Moved", at(-1))
    sessions.update_session(session_id, start_time=at(30))

    sessions.apply_scheduled_transitions(now=NOW)

    assert status_of(sessions, session_id) == SessionStatus.SCHEDULED


def test_end_time_must_stay_after_the_existing_start(sessions):
    session_id = sessions.create_session("S", at(60))
    with pytest.raises(InvalidSessionDataError, match="after start_time"):
        sessions.update_session(session_id, end_time=at(30))


def test_moving_the_start_past_the_existing_end_is_rejected(sessions):
    session_id = sessions.create_session("S", at(60), end_time=at(120))
    with pytest.raises(InvalidSessionDataError, match="after start_time"):
        sessions.update_session(session_id, start_time=at(180))


def test_editing_an_active_session_regrades_its_marks(sessions, attendance, students):
    ann = students.create_student("S1", "Ann")
    bob = students.create_student("S2", "Bob")
    session_id = sessions.create_session("A", at(-60), status=SessionStatus.ACTIVE, late_threshold_minutes=15)
    attendance.record_attendance(session_id, ann, marked_at=at(-55))  # present
    attendance.record_attendance(session_id, bob, marked_at=at(-30))  # late

    sessions.update_session(session_id, late_threshold_minutes=45)
    statuses = {r['student_id']: r['status'] for r in attendance.get_session_attendance(session_id)}
    assert statuses == {ann: AttendanceStatus.PRESENT, bob: AttendanceStatus.PRESENT}

    sessions.update_session(session_id, start_time=at(-20), late_threshold_minutes=0)
    statuses = {r['student_id']: r['status'] for r in attendance.get_session_attendance(session_id)}
    assert statuses == {ann: AttendanceStatus.PRESENT, bob: AttendanceStatus.PRESENT}  # both before the new start

    sessions.update_session(session_id, start_time=at(-56))
    statuses = {r['student_id']: r['status'] for r in attendance.get_session_attendance(session_id)}
    assert statuses == {ann: AttendanceStatus.LATE, bob: AttendanceStatus.LATE}


def test_active_session_end_time_can_move_or_be_cleared(sessions):
    session_id = sessions.create_session("A", at(-10), end_time=at(50), status=SessionStatus.ACTIVE)
    sessions.update_session(session_id, end_time=at(80))
    assert sessions.get_session(session_id)['end_time'] == at(80)

    sessions.update_session(session_id, clear_end_time=True)
    assert sessions.get_session(session_id)['end_time'] is None


def test_closed_session_can_be_edited_and_stays_closed(sessions):
    session_id = sessions.create_session("C", at(-60), end_time=at(-1), status=SessionStatus.CLOSED)
    sessions.update_session(session_id, session_name="Renamed", start_time=at(-50), end_time=at(10))
    sessions.apply_scheduled_transitions(now=NOW)
    session = sessions.get_session(session_id)
    assert (session['session_name'], session['start_time'], session['status']) == \
        ("Renamed", at(-50), SessionStatus.CLOSED)


# ============================================================================
# Delete and overlap
# ============================================================================

def test_delete_removes_session_and_its_records(sessions, attendance, students):
    student_db_id = students.create_student("S1", "Ann")
    session_id = sessions.create_session("To delete", at(-1), status=SessionStatus.ACTIVE)
    attendance.record_attendance(session_id, student_db_id, marked_at=at(0))
    sessions.close_session(session_id)

    sessions.delete_session(session_id)

    assert sessions.get_session(session_id) is None
    assert attendance.get_all_attendance() == []


def test_active_session_cannot_be_deleted(sessions):
    session_id = sessions.create_session("Running", at(-1), status=SessionStatus.ACTIVE)
    with pytest.raises(InvalidSessionStateError):
        sessions.delete_session(session_id)


def test_overlap_detection(sessions):
    first = sessions.create_session("CS101 A", at(0), course_code="CS101", end_time=at(60))
    sessions.create_session("Other course", at(0), course_code="MA101", end_time=at(60))

    assert [s['id'] for s in sessions.find_overlapping_sessions(at(30), at(90), "CS101")] == [first]
    # Back-to-back is not an overlap
    assert sessions.find_overlapping_sessions(at(60), at(120), "CS101") == []
    # Same start, no end times
    assert [s['id'] for s in sessions.find_overlapping_sessions(at(0), None, "CS101")] == [first]
    # Excluding the session itself (editing it)
    assert sessions.find_overlapping_sessions(at(0), at(60), "CS101", exclude_id=first) == []


def test_overview_counts(sessions, attendance, students):
    ann = students.create_student("S1", "Ann")
    bob = students.create_student("S2", "Bob")
    session_id = sessions.create_session("Lecture", at(-30), status=SessionStatus.ACTIVE, late_threshold_minutes=10)
    attendance.record_attendance(session_id, ann, marked_at=at(-25))  # present
    attendance.record_attendance(session_id, bob, marked_at=at(-5))   # late

    row = sessions.get_sessions_overview()[0]

    assert (row['present_count'], row['late_count'], row['marked_count']) == (1, 1, 2)


def test_status_signature_changes_on_transition(sessions):
    sessions.create_session("Due", at(-1))
    before = sessions.get_status_signature()
    sessions.apply_scheduled_transitions(now=NOW)
    assert sessions.get_status_signature() != before


# ============================================================================
# Session roster (absent students)
# ============================================================================

@pytest.fixture
def roster_session(sessions, attendance, students):
    """Zed present, Amy late, Bob and Cat absent; one inactive student without a record."""
    ids = {name: students.create_student(f"ID-{name}", name) for name in ("Zed", "Amy", "Bob", "Cat", "Old")}
    students.deactivate_student(ids["Old"])
    # Starts after the students were registered (a minute from now), like a real class
    start = datetime.now().replace(microsecond=0) + timedelta(minutes=1)
    session_id = sessions.create_session("Lecture", start.strftime(FMT), status=SessionStatus.ACTIVE,
                                         late_threshold_minutes=10)
    attendance.record_attendance(session_id, ids["Zed"], marked_at=(start + timedelta(minutes=2)).strftime(FMT))
    attendance.record_attendance(session_id, ids["Amy"], marked_at=(start + timedelta(minutes=25)).strftime(FMT))
    return session_id


def test_roster_lists_absent_students_in_status_then_name_order(attendance, roster_session):
    rows = attendance.get_session_roster(roster_session)

    assert [(r['name'], r['status']) for r in rows] == [
        ("Zed", AttendanceStatus.PRESENT),
        ("Amy", AttendanceStatus.LATE),
        ("Bob", AttendanceStatus.ABSENT),
        ("Cat", AttendanceStatus.ABSENT),
    ]
    assert rows[2]['record_id'] is None and rows[2]['marked_at'] is None


def test_roster_counts_match_the_list(attendance, roster_session):
    counts = attendance.get_session_roster_counts(roster_session)
    rows = attendance.get_session_roster(roster_session)

    assert counts == {'present': 1, 'late': 1, 'absent': 2, 'expected': 4}
    assert counts['expected'] == len(rows)


def test_roster_filter_search_and_limit(attendance, roster_session):
    assert [r['name'] for r in attendance.get_session_roster(roster_session, status="absent")] == ["Bob", "Cat"]
    assert [r['name'] for r in attendance.get_session_roster(roster_session, search="id-c")] == ["Cat"]
    assert len(attendance.get_session_roster(roster_session, limit=3)) == 3
    # LIKE wildcards typed by the user match literally
    assert attendance.get_session_roster(roster_session, search="%") == []


def test_deactivated_student_with_a_record_stays_listed(attendance, students, roster_session):
    zed = students.get_student_by_student_id("ID-Zed")
    students.deactivate_student(zed['id'])

    names = [r['name'] for r in attendance.get_session_roster(roster_session)]

    assert "Zed" in names and "Old" not in names


def test_students_registered_after_a_session_started_are_not_absent(sessions, attendance, students, temp_db):
    early = students.create_student("ID-Early", "Early")
    # Registered two hours ago (UTC, like the database default)
    with closing(sqlite3.connect(temp_db)) as conn, conn:
        conn.execute("UPDATE students SET enrolled_at = datetime('now', '-2 hours') WHERE id = ?", (early,))
    started = (datetime.now() - timedelta(hours=1)).strftime(FMT)  # local time, like the app
    session_id = sessions.create_session("Past", started, status=SessionStatus.ACTIVE)
    sessions.close_session(session_id)
    students.create_student("ID-New", "Newcomer")  # registered now, after the start

    names = [r['name'] for r in attendance.get_session_roster(session_id)]

    assert names == ["Early"]


# ============================================================================
# Background scheduler thread
# ============================================================================

def test_scheduler_thread_starts_due_sessions(temp_db, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_PATH", temp_db)
    service = SessionService(temp_db)
    session_id = service.create_session("Due now", (datetime.now() - timedelta(seconds=1)).strftime(FMT))
    session_scheduler.stop_session_scheduler()
    try:
        assert session_scheduler.start_session_scheduler(poll_seconds=0.05) is True
        assert session_scheduler.start_session_scheduler(poll_seconds=0.05) is False  # idempotent
        deadline = time.time() + 5
        while time.time() < deadline and status_of(service, session_id) != SessionStatus.ACTIVE:
            time.sleep(0.05)
        assert status_of(service, session_id) == SessionStatus.ACTIVE
    finally:
        session_scheduler.stop_session_scheduler()
    assert not session_scheduler.is_session_scheduler_running()
