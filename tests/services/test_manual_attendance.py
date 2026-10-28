"""
Tests for setting attendance by hand (AttendanceService.set_attendance_manually)
and the Session Details editor that saves those changes.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.database.schema import initialize_database
from src.services.attendance_service import (
    AttendanceService,
    AttendanceStatus,
    InvalidAttendanceDataError,
    SessionNotAcceptingAttendanceError,
    MANUAL_MARKER,
)
from src.services.session_service import SessionService, SessionStatus
from src.services.student_service import StudentService

FMT = '%Y-%m-%d %H:%M:%S'


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def attendance(temp_db):
    return AttendanceService(temp_db)


@pytest.fixture
def sessions(temp_db):
    return SessionService(temp_db)


@pytest.fixture
def setup(temp_db, sessions):
    """Two students registered before an active session that started 30 minutes ago."""
    students = StudentService(temp_db)
    ann = students.create_student("S1", "Ann")
    bob = students.create_student("S2", "Bob")
    start = datetime.now().replace(microsecond=0) + timedelta(seconds=1)
    session_id = sessions.create_session("Lecture", start.strftime(FMT), status=SessionStatus.ACTIVE,
                                         late_threshold_minutes=10)
    return session_id, ann, bob, start


def status_of(attendance: AttendanceService, session_id: int, student_id: int) -> str:
    record = attendance.get_attendance_by_session_and_student(session_id, student_id)
    return record['status'] if record else AttendanceStatus.ABSENT


def test_mark_a_missed_student_present(attendance, setup):
    session_id, ann, _, _ = setup

    record_id = attendance.set_attendance_manually(session_id, ann, "present")

    record = attendance.get_attendance(record_id)
    assert record['status'] == AttendanceStatus.PRESENT
    assert record['marked_by'] == MANUAL_MARKER
    assert attendance.get_session_roster_counts(session_id) == {
        'present': 1, 'late': 0, 'absent': 1, 'expected': 2}


def test_excuse_a_late_student(attendance, setup):
    session_id, ann, _, start = setup
    attendance.record_attendance(session_id, ann, marked_at=(start + timedelta(minutes=20)).strftime(FMT))
    assert status_of(attendance, session_id, ann) == AttendanceStatus.LATE

    attendance.set_attendance_manually(session_id, ann, "present")

    record = attendance.get_attendance_by_session_and_student(session_id, ann)
    assert record['status'] == AttendanceStatus.PRESENT
    assert record['marked_by'] == 'system'          # who marked it first is kept
    assert record['verified_by'] == MANUAL_MARKER   # who changed it is recorded


def test_remove_a_wrong_mark(attendance, setup):
    session_id, ann, _, start = setup
    attendance.record_attendance(session_id, ann, marked_at=(start + timedelta(minutes=1)).strftime(FMT))

    assert attendance.set_attendance_manually(session_id, ann, "absent") is None

    assert status_of(attendance, session_id, ann) == AttendanceStatus.ABSENT
    assert attendance.get_session_roster_counts(session_id)['absent'] == 2


def test_absent_student_set_absent_is_a_no_op(attendance, setup):
    session_id, ann, _, _ = setup
    assert attendance.set_attendance_manually(session_id, ann, "absent") is None
    assert status_of(attendance, session_id, ann) == AttendanceStatus.ABSENT


def test_closed_session_can_still_be_fixed(attendance, sessions, setup):
    session_id, ann, _, _ = setup
    sessions.close_session(session_id)

    attendance.set_attendance_manually(session_id, ann, "late")

    assert status_of(attendance, session_id, ann) == AttendanceStatus.LATE


def test_scheduled_session_cannot_be_changed(attendance, sessions, setup):
    _, ann, _, _ = setup
    later = (datetime.now() + timedelta(days=1)).strftime(FMT)
    scheduled = sessions.create_session("Tomorrow", later)

    with pytest.raises(SessionNotAcceptingAttendanceError):
        attendance.set_attendance_manually(scheduled, ann, "present")


@pytest.mark.parametrize("session_id, student_id, status", [
    (0, 1, "present"), (1, 0, "present"), (1, 1, "excused"), (999, 1, "present"),
])
def test_invalid_input_rejected(attendance, setup, session_id, student_id, status):
    with pytest.raises(InvalidAttendanceDataError):
        attendance.set_attendance_manually(session_id, student_id, status)


def test_hand_set_status_survives_a_regrade(attendance, sessions, setup):
    """Editing the session's times re-grades camera marks but keeps statuses set by hand."""
    session_id, ann, bob, start = setup
    late_mark = (start + timedelta(minutes=20)).strftime(FMT)
    attendance.record_attendance(session_id, ann, marked_at=late_mark)   # late, by the camera
    attendance.record_attendance(session_id, bob, marked_at=late_mark)   # late, by the camera
    attendance.set_attendance_manually(session_id, bob, "present")       # excused by hand

    sessions.update_session(session_id, late_threshold_minutes=11)        # forces a re-grade

    assert status_of(attendance, session_id, ann) == AttendanceStatus.LATE
    assert status_of(attendance, session_id, bob) == AttendanceStatus.PRESENT


def test_roster_shows_who_changed_a_record(attendance, setup):
    session_id, ann, _, start = setup
    attendance.record_attendance(session_id, ann, marked_at=(start + timedelta(minutes=20)).strftime(FMT))
    attendance.set_attendance_manually(session_id, ann, "present")

    row = attendance.get_session_roster(session_id, status="present")[0]

    assert (row['marked_by'], row['verified_by']) == ('system', MANUAL_MARKER)


# ============================================================================
# Session Details editor
# ============================================================================

def test_editor_changes_are_saved_and_the_page_redrawn(attendance, setup):
    from src.ui.pages.session_management import _save_status_changes, FLASH_KEY, ATTENDANCE_LABELS

    session_id, ann, bob, _ = setup
    rows = attendance.get_session_roster(session_id)   # both absent, Ann first
    state = {}

    with patch("streamlit.session_state", state), patch("streamlit.rerun") as mock_rerun:
        _save_status_changes(attendance, session_id, rows,
                             {0: {'Status': ATTENDANCE_LABELS['present']},
                              1: {'Status': ATTENDANCE_LABELS['absent']}})   # unchanged row
        mock_rerun.assert_called_once()

    assert status_of(attendance, session_id, ann) == AttendanceStatus.PRESENT
    assert status_of(attendance, session_id, bob) == AttendanceStatus.ABSENT
    kind, message = state[FLASH_KEY]
    assert kind == "success" and "Ann" in message and "Bob" not in message


def test_marked_by_label():
    from src.ui.pages.session_management import marked_by_label

    assert marked_by_label('cv_realtime') == "Camera"
    assert marked_by_label(MANUAL_MARKER) == "Instructor"
    assert marked_by_label('cv_realtime', MANUAL_MARKER) == "Camera, changed by Instructor"
    assert marked_by_label(None) == ""
