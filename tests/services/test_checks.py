"""
Tests for check-out and spot checks: starting, scheduling (also at a random
time), automatic check-out, closing on time, flagging "left early" only for
students who checked in before a check and missed it, the camera-off safety
net (a check the camera never saw doesn't count; one it saw counts even with
nobody seen), check-out before closing a session, the instructor's override,
and the camera recording check marks.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.computer_vision.face_detector import DetectedFace
from src.computer_vision.face_embedder import FaceEmbedding
from src.computer_vision.face_recognizer import RecognitionResult
from src.database.schema import initialize_database
from src.services.attendance_service import AttendanceService
from src.services.check_service import CheckError, CheckKind, CheckService, CheckStatus, VoidReason
from src.services.real_time_attendance_service import RealTimeAttendanceManager
from src.services.session_service import InvalidSessionDataError, SessionService, SessionStatus
from src.services.student_service import StudentService

START = datetime(2026, 11, 4, 10, 0, 0)
FMT = '%Y-%m-%d %H:%M:%S'


@pytest.fixture
def world():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db = Path(f.name)
    initialize_database(db)
    students = StudentService(db)
    ids = {name: students.create_student(str(i), name) for i, name in enumerate(["Alice", "Omar", "Sara"], 1)}
    sessions = SessionService(db)
    session_id = sessions.create_session("Lecture", START.strftime(FMT), end_time=(START + timedelta(hours=2)).strftime(FMT),
                                         status=SessionStatus.ACTIVE)
    yield {'db': db, 'ids': ids, 'session': session_id, 'sessions': sessions,
           'checks': CheckService(db), 'attendance': AttendanceService(db)}
    db.unlink()


def _check_in(world, name, minute):
    return world['attendance'].record_attendance(world['session'], world['ids'][name],
                                                 marked_at=(START + timedelta(minutes=minute)).strftime(FMT))


def _left_early(world):
    return {r['name'] for r in world['attendance'].get_session_roster(world['session']) if r['left_early']}


def test_spot_check_flags_who_missed_it(world):
    _check_in(world, "Alice", 2)
    _check_in(world, "Omar", 3)
    now = START + timedelta(minutes=30)
    check_id = world['checks'].start_spot_check(world['session'], 5, now=now)
    world['checks'].record_mark(check_id, world['ids']["Alice"], now=now + timedelta(minutes=1))
    _check_in(world, "Sara", 31)                      # arrives during the check: not due in it

    changed = world['checks'].apply_due(now=now + timedelta(minutes=6))
    assert changed['closed'] == [check_id]
    assert _left_early(world) == {"Omar"}
    check = world['checks'].list_checks(world['session'])[0]
    assert check['status'] == CheckStatus.DONE
    assert (check['marked_count'], check['due_count']) == (1, 2)


def test_camera_off_the_whole_check_means_it_does_not_count(world):
    _check_in(world, "Alice", 2)
    now = START + timedelta(minutes=30)
    check_id = world['checks'].start_spot_check(world['session'], 5, now=now)
    world['checks'].apply_due(now=now + timedelta(minutes=6))
    check = world['checks'].list_checks(world['session'])[0]
    assert (check['status'], check['void_reason']) == (CheckStatus.VOID, VoidReason.CAMERA_OFF)
    assert _left_early(world) == set()
    with pytest.raises(CheckError):
        world['checks'].record_mark(check_id, world['ids']["Alice"])


def test_camera_on_but_nobody_showed_up_flags_everyone_due(world):
    _check_in(world, "Alice", 2)
    _check_in(world, "Omar", 3)
    now = START + timedelta(minutes=30)
    check_id = world['checks'].start_spot_check(world['session'], 5, now=now)
    world['checks'].camera_saw(check_id, now=now + timedelta(seconds=2))
    assert world['checks'].finish_check_now(check_id, now=now + timedelta(minutes=1)) == 2
    assert _left_early(world) == {"Alice", "Omar"}
    assert world['checks'].list_checks(world['session'])[0]['status'] == CheckStatus.DONE


def test_finish_now_without_the_camera_says_it_did_not_count(world):
    _check_in(world, "Alice", 2)
    check_id = world['checks'].start_spot_check(world['session'], 5, now=START + timedelta(minutes=30))
    assert world['checks'].finish_check_now(check_id, now=START + timedelta(minutes=31)) is None
    assert _left_early(world) == set()


def test_only_one_check_at_a_time_and_only_while_running(world):
    now = START + timedelta(minutes=30)
    world['checks'].start_spot_check(world['session'], 5, now=now)
    with pytest.raises(CheckError, match="already running"):
        world['checks'].start_checkout_now(world['session'], now=now)
    scheduled = world['sessions'].create_session("Later", "2026-11-05 10:00:00")
    with pytest.raises(CheckError, match="running"):
        world['checks'].start_spot_check(scheduled, 5, now=now)


def test_finish_and_cancel(world):
    _check_in(world, "Alice", 2)
    _check_in(world, "Omar", 3)
    now = START + timedelta(minutes=30)
    first = world['checks'].start_spot_check(world['session'], 5, now=now)
    world['checks'].camera_saw(first, now=now)
    world['checks'].cancel_check(first)                       # cancelled: nobody flagged
    assert world['checks'].list_checks(world['session'])[0]['void_reason'] == VoidReason.CANCELLED
    second = world['checks'].start_spot_check(world['session'], 5, now=now + timedelta(minutes=1))
    world['checks'].record_mark(second, world['ids']["Alice"], now=now + timedelta(minutes=2))
    assert world['checks'].finish_check_now(second, now=now + timedelta(minutes=3)) == 1
    assert _left_early(world) == {"Omar"}


def test_scheduled_spot_check_opens_on_time(world):
    now = START + timedelta(minutes=10)
    at = START + timedelta(minutes=40)
    check_id = world['checks'].schedule_spot_check(world['session'], at=at, minutes=5, now=now)
    assert world['checks'].apply_due(now=at - timedelta(seconds=1))['opened'] == []
    assert world['checks'].apply_due(now=at)['opened'] == [check_id]
    assert world['checks'].get_open_check(world['session'])['id'] == check_id
    with pytest.raises(CheckError, match="passed"):
        world['checks'].schedule_spot_check(world['session'], at=now - timedelta(minutes=1), now=now)


def test_random_spot_check_falls_inside_the_session_and_before_check_out(world):
    world['sessions'].update_session(world['session'], checkout_enabled=True, checkout_minutes=10)
    now = START + timedelta(minutes=10)
    for _ in range(20):
        check_id = world['checks'].schedule_spot_check(world['session'], randomly=True, minutes=5, now=now)
        check = world['checks'].repository.get_check(check_id)
        starts = datetime.strptime(check['starts_at'], FMT)
        ends = datetime.strptime(check['ends_at'], FMT)
        assert now + timedelta(minutes=2) <= starts and ends <= START + timedelta(minutes=110)
        assert check['is_random']
        world['checks'].cancel_check(check_id)


def test_random_needs_an_end_time(world):
    no_end = world['sessions'].create_session("Open", START.strftime(FMT), status=SessionStatus.ACTIVE)
    with pytest.raises(CheckError, match="end time"):
        world['checks'].schedule_spot_check(no_end, randomly=True, now=START + timedelta(minutes=5))


def test_automatic_check_out_in_the_last_minutes(world):
    world['sessions'].update_session(world['session'], checkout_enabled=True, checkout_minutes=10)
    _check_in(world, "Alice", 2)
    _check_in(world, "Omar", 3)
    world['checks'].apply_due(now=START + timedelta(minutes=20))
    [checkout] = world['checks'].list_checks(world['session'])
    assert checkout['kind'] == CheckKind.CHECKOUT and checkout['status'] == CheckStatus.SCHEDULED
    assert checkout['starts_at'] == "2026-11-04 11:50:00" and checkout['ends_at'] == "2026-11-04 12:00:00"

    world['checks'].apply_due(now=START + timedelta(minutes=110))
    open_check = world['checks'].get_open_check(world['session'])
    world['checks'].record_mark(open_check['id'], world['ids']["Alice"], now=START + timedelta(minutes=112))
    world['checks'].apply_due(now=START + timedelta(minutes=120))
    assert _left_early(world) == {"Omar"}
    assert len(world['checks'].list_checks(world['session'])) == 1      # not created again


def test_automatic_check_out_follows_setting_changes(world):
    world['sessions'].update_session(world['session'], checkout_enabled=True, checkout_minutes=10)
    world['checks'].apply_due(now=START + timedelta(minutes=20))
    world['sessions'].update_session(world['session'], checkout_minutes=15)
    world['checks'].apply_due(now=START + timedelta(minutes=21))
    assert world['checks'].list_checks(world['session'])[0]['starts_at'] == "2026-11-04 11:45:00"
    world['sessions'].update_session(world['session'], checkout_enabled=False)
    world['checks'].apply_due(now=START + timedelta(minutes=22))
    assert world['checks'].list_checks(world['session'])[0]['status'] == CheckStatus.VOID


def test_check_out_minutes_are_validated(world):
    with pytest.raises(InvalidSessionDataError):
        world['sessions'].create_session("Bad", START.strftime(FMT), checkout_enabled=True, checkout_minutes=0)


def test_closing_the_session_ends_its_checks(world):
    _check_in(world, "Alice", 2)
    _check_in(world, "Omar", 3)
    now = START + timedelta(minutes=30)
    running = world['checks'].start_spot_check(world['session'], 30, now=now)
    later = world['checks'].schedule_spot_check(world['session'], at=now + timedelta(minutes=60), now=now)
    world['checks'].record_mark(running, world['ids']["Alice"], now=now)
    world['sessions'].close_session(world['session'])
    world['checks'].apply_due(now=now + timedelta(minutes=5))
    statuses = {c['id']: c['status'] for c in world['checks'].list_checks(world['session'])}
    assert statuses == {running: CheckStatus.DONE, later: CheckStatus.VOID}
    assert _left_early(world) == {"Omar"}


def test_instructor_override(world):
    _check_in(world, "Alice", 2)
    world['attendance'].set_left_early(world['session'], world['ids']["Alice"], True)
    assert _left_early(world) == {"Alice"} and world['attendance'].count_left_early(world['session']) == 1
    world['attendance'].set_left_early(world['session'], world['ids']["Alice"], False)
    assert _left_early(world) == set()
    with pytest.raises(Exception):                  # not checked in: nothing to flag
        world['attendance'].set_left_early(world['session'], world['ids']["Sara"], True)


# -- the camera ---------------------------------------------------------------------------

def test_camera_records_a_check_mark_for_a_checked_in_student(world, monkeypatch):
    from unittest.mock import MagicMock
    from src.computer_vision.face_detector import FaceDetector
    from src.computer_vision.face_embedder import FaceEmbedder
    from src.computer_vision.face_recognizer import FaceRecognizer
    from src.computer_vision.liveness_detector import LivenessDetector

    detector, embedder = MagicMock(spec=FaceDetector), MagicMock(spec=FaceEmbedder)
    recognizer, liveness = MagicMock(spec=FaceRecognizer), MagicMock(spec=LivenessDetector)
    liveness.min_blinks = liveness.blink_count = 1
    liveness.liveness_threshold = 0.75
    liveness.enable_challenges = liveness.require_challenge = False
    liveness.extract_ear.return_value = 0.25
    liveness.update_temporal_state.return_value = True
    detector.detect_faces.return_value = [DetectedFace(bbox=(10, 10, 50, 50), confidence=0.95)]
    encoding = np.random.rand(128)
    embedder.generate_embedding.return_value = FaceEmbedding(encoding)
    recognizer.recognize.return_value = RecognitionResult(identity="1", distance=0.3, is_match=True)

    manager = RealTimeAttendanceManager(session_id=world['session'], database_path=world['db'],
                                        face_detector=detector, face_embedder=embedder, face_recognizer=recognizer,
                                        liveness_detector=liveness, require_liveness=True)
    manager.set_gallery([("1", FaceEmbedding(encoding))])
    manager.cooldown_seconds = 0
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = manager.process_frame(frame, marked_at=(START + timedelta(minutes=2)).strftime(FMT)).processed_faces[0]
    assert first.attendance_marked and first.attendance_status == "present"
    again = manager.process_frame(frame).processed_faces[0]
    assert again.already_marked                                   # no check running

    check_id = world['checks'].start_spot_check(world['session'], 5, now=datetime.now())
    manager.set_open_check(world['checks'].get_open_check(world['session']), set())
    seen = manager.process_frame(frame).processed_faces[0]
    assert seen.attendance_marked and seen.attendance_status == "spot check"
    assert world['checks'].checked_student_ids(check_id) == {world['ids']["Alice"]}
    after = manager.process_frame(frame).processed_faces[0]
    assert after.already_marked                                   # already seen in this check


# -- closing a session with check-out on ---------------------------------------------------

def test_check_out_still_due(world):
    assert not world['checks'].checkout_still_due(world['session'])          # check-out off
    world['sessions'].update_session(world['session'], checkout_enabled=True)
    assert world['checks'].checkout_still_due(world['session'])
    world['checks'].start_checkout_now(world['session'], 5, now=START + timedelta(minutes=30))
    assert not world['checks'].checkout_still_due(world['session'])          # running now


def test_check_out_then_close(world):
    no_end = world['sessions'].create_session("No end", START.strftime(FMT), status=SessionStatus.ACTIVE,
                                              checkout_enabled=True, checkout_minutes=10)
    for name, minute in (("Alice", 2), ("Omar", 3)):
        world['attendance'].record_attendance(no_end, world['ids'][name],
                                              marked_at=(START + timedelta(minutes=minute)).strftime(FMT))
    now = START + timedelta(minutes=60)
    check = world['checks'].start_checkout_then_close(no_end, now=now)
    assert check['kind'] == CheckKind.CHECKOUT and check['close_session_after']
    assert check['ends_at'] == "2026-11-04 11:10:00"                      # the session's 10 minutes
    world['checks'].camera_saw(check['id'], now=now)
    world['checks'].record_mark(check['id'], world['ids']["Alice"], now=now + timedelta(minutes=1))

    world['checks'].apply_due(now=now + timedelta(minutes=5))              # still running
    assert world['sessions'].get_session(no_end)['status'] == SessionStatus.ACTIVE
    world['checks'].apply_due(now=now + timedelta(minutes=10))             # ended: flag, then close
    assert world['sessions'].get_session(no_end)['status'] == SessionStatus.CLOSED
    flagged = {r['name'] for r in world['attendance'].get_session_roster(no_end) if r['left_early']}
    assert flagged == {"Omar"}


def test_check_out_then_close_reuses_a_running_check_out(world):
    now = START + timedelta(minutes=30)
    running = world['checks'].start_checkout_now(world['session'], 5, now=now)
    assert world['checks'].start_checkout_then_close(world['session'], now=now)['id'] == running
    world['checks'].finish_check_now(running, now=now + timedelta(minutes=1))
    assert world['sessions'].get_session(world['session'])['status'] == SessionStatus.CLOSED


def test_check_out_then_close_waits_for_a_spot_check(world):
    now = START + timedelta(minutes=30)
    world['checks'].start_spot_check(world['session'], 5, now=now)
    with pytest.raises(CheckError, match="spot check"):
        world['checks'].start_checkout_then_close(world['session'], now=now)
