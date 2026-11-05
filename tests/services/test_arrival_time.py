"""
Present / late counts from when a student arrived at the camera.

- arrived_at: the time the camera first recognised the student (the blink /
  head-turn test takes a few seconds and must not make anyone late).
- line_since: a line that was already at the camera when the late time
  passed keeps the students in it Present (config.LATE_LINE_GRACE_MINUTES).

Session in these tests: starts 10:00, late after 15 minutes -> cutoff 10:15.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import config
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding
from src.computer_vision.face_recognizer import FaceRecognizer, RecognitionResult
from src.computer_vision.liveness_detector import LivenessDetector
from src.database.schema import initialize_database, migrate_schema
from src.services import real_time_attendance_service as rtas
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.real_time_attendance_service import RealTimeAttendanceManager
from src.services.session_service import SessionService, SessionStatus
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
def attendance_service(temp_db):
    return AttendanceService(temp_db)


@pytest.fixture
def session_id(temp_db):
    return SessionService(temp_db).create_session(
        session_name="CS101", start_time="2026-10-04 10:00:00",
        late_threshold_minutes=15, status=SessionStatus.ACTIVE)


@pytest.fixture
def alice(temp_db):
    return StudentService(temp_db).create_student(student_id="2021001", name="Alice Smith")


@pytest.fixture
def bob(temp_db):
    return StudentService(temp_db).create_student(student_id="2021002", name="Bob Jones")


def _record(service, session_id, student, marked_at, **kwargs):
    record_id = service.record_attendance(session_id=session_id, student_id=student,
                                          marked_at=marked_at, **kwargs)
    return service.get_attendance(record_id)


# ============================================================================
# First seen by the camera
# ============================================================================

def test_first_seen_before_the_late_time_counts_as_present(attendance_service, session_id, alice):
    """Recognised at 10:14:55, the test finished at 10:15:04: present."""
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:15:04",
                     arrived_at="2026-10-04 10:14:55")
    assert record['status'] == AttendanceStatus.PRESENT
    assert record['marked_at'] == "2026-10-04 10:15:04"      # the real mark time is kept
    assert record['arrived_at'] == "2026-10-04 10:14:55"


def test_first_seen_after_the_mark_is_ignored(attendance_service, session_id, alice):
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:16:00",
                     arrived_at="2026-10-04 10:17:00")
    assert record['status'] == AttendanceStatus.LATE
    assert record['arrived_at'] is None


def test_without_arrival_times_the_mark_time_decides(attendance_service, session_id, alice):
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:15:04")
    assert record['status'] == AttendanceStatus.LATE
    assert record['arrived_at'] is None


def test_bad_arrival_time_is_rejected(attendance_service, session_id, alice):
    with pytest.raises(Exception, match="arrived_at"):
        _record(attendance_service, session_id, alice, "2026-10-04 10:15:04", arrived_at="yesterday")


# ============================================================================
# The line at the camera
# ============================================================================

def test_line_that_began_before_the_late_time_keeps_students_present(attendance_service, session_id, alice):
    """The line began 10:14:30; Alice reached the camera at 10:17 (within 5 minutes)."""
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:17:06",
                     arrived_at="2026-10-04 10:17:00", line_since="2026-10-04 10:14:30")
    assert record['status'] == AttendanceStatus.PRESENT
    assert record['arrived_at'] == "2026-10-04 10:14:30"
    assert "In the line at the camera since 10:14:30" in record['notes']


def test_line_grace_has_a_limit(attendance_service, session_id, alice):
    """Reached the camera 6 minutes after the late time: late, even in an unbroken line."""
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:21:06",
                     arrived_at="2026-10-04 10:21:00", line_since="2026-10-04 10:14:30")
    assert record['status'] == AttendanceStatus.LATE
    assert record['arrived_at'] == "2026-10-04 10:21:00"
    assert "line" not in (record['notes'] or '')


def test_line_that_began_after_the_late_time_does_not_help(attendance_service, session_id, alice):
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:16:06",
                     arrived_at="2026-10-04 10:16:00", line_since="2026-10-04 10:15:30")
    assert record['status'] == AttendanceStatus.LATE


def test_line_rule_can_be_turned_off(attendance_service, session_id, alice, monkeypatch):
    monkeypatch.setattr(config, 'LATE_LINE_GRACE_MINUTES', 0.0)
    record = _record(attendance_service, session_id, alice, "2026-10-04 10:17:06",
                     arrived_at="2026-10-04 10:17:00", line_since="2026-10-04 10:14:30")
    assert record['status'] == AttendanceStatus.LATE


# ============================================================================
# Editing the session re-grades by the arrival time
# ============================================================================

def test_regrade_after_an_edit_uses_the_arrival_time(temp_db, attendance_service, session_id, alice, bob):
    _record(attendance_service, session_id, alice, "2026-10-04 10:10:30", arrived_at="2026-10-04 10:09:50")
    _record(attendance_service, session_id, bob, "2026-10-04 10:10:30")
    sessions = SessionService(temp_db)
    sessions.update_session(session_id, late_threshold_minutes=10)    # cutoff 10:10
    sessions.regrade_attendance(session_id)
    by_student = {r['student_id']: r['status'] for r in attendance_service.get_session_attendance(session_id)}
    assert by_student[alice] == AttendanceStatus.PRESENT     # arrived 10:09:50
    assert by_student[bob] == AttendanceStatus.LATE          # marked 10:10:30


def test_older_database_gets_the_arrived_at_column(temp_db):
    with sqlite3.connect(temp_db) as conn:
        conn.execute("ALTER TABLE attendance_records DROP COLUMN arrived_at")
        migrate_schema(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attendance_records)")}
    assert 'arrived_at' in columns


# ============================================================================
# The camera engine: first-seen time and the line
# ============================================================================

FACE = DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9)
ALICE = RecognitionResult(identity="2021001", distance=0.3, is_match=True)
BOB = RecognitionResult(identity="2021002", distance=0.3, is_match=True)
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


class Clock:
    """Stands in for time.time() in the camera engine; starts at a local wall time."""

    def __init__(self, local_time: str):
        self.now = datetime.fromisoformat(local_time).timestamp()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def engine(temp_db, session_id, monkeypatch):
    """A camera engine whose liveness test passes when `live['ok']` is True."""
    from unittest.mock import MagicMock
    detector = MagicMock(spec=FaceDetector)
    embedder = MagicMock(spec=FaceEmbedder)
    recognizer = MagicMock(spec=FaceRecognizer)
    liveness = MagicMock(spec=LivenessDetector)
    liveness.min_blinks = 1
    liveness.blink_count = 0
    liveness.liveness_threshold = 0.75
    liveness.enable_challenges = False
    liveness.require_challenge = False
    liveness.extract_ear.return_value = 0.25
    embedder.generate_embedding.return_value = FaceEmbedding(np.random.rand(128))
    detector.detect_faces.return_value = [FACE]

    clock = Clock("2026-10-04 10:14:00")
    monkeypatch.setattr(rtas.time, 'time', clock)
    manager = RealTimeAttendanceManager(
        session_id=session_id, database_path=temp_db, face_detector=detector,
        face_embedder=embedder, face_recognizer=recognizer, liveness_detector=liveness,
        require_liveness=True)
    manager.set_gallery([("2021001", FaceEmbedding(np.random.rand(128))),
                         ("2021002", FaceEmbedding(np.random.rand(128)))])

    def step_up(result, test_seconds):
        """A student steps up, takes `test_seconds` for the liveness test, and is marked."""
        recognizer.recognize.return_value = result
        liveness.blink_count = 0
        manager.process_frame(FRAME)                 # recognised: first seen now
        clock.advance(test_seconds)
        liveness.blink_count = 1                     # test passed
        face = manager.process_frame(FRAME).processed_faces[0]
        assert face.attendance_marked is True, face.reason
        return face

    def away(seconds):
        """Nobody in front of the camera for this long."""
        detector.detect_faces.return_value = []
        manager.process_frame(FRAME)
        clock.advance(seconds)
        detector.detect_faces.return_value = [FACE]

    return {'manager': manager, 'clock': clock, 'step_up': step_up, 'away': away,
            'recognizer': recognizer}


def _status(temp_db, session_id, student):
    records = AttendanceService(temp_db).get_session_attendance(session_id)
    return next(r for r in records if r['student_id'] == student)


def test_liveness_test_time_does_not_make_a_student_late(temp_db, session_id, alice, engine):
    """Recognised 10:14:50, test finished 10:15:02: present (the record keeps its real mark time)."""
    engine['clock'].advance(50)
    face = engine['step_up'](ALICE, 12)
    assert face.attendance_status == AttendanceStatus.PRESENT
    assert _status(temp_db, session_id, alice)['arrived_at'] == "2026-10-04 10:14:50"


def test_student_back_much_later_counts_from_the_return(temp_db, session_id, alice, engine):
    """Seen 10:14:00, failed and left; back 10:16 and marked: late."""
    engine['recognizer'].recognize.return_value = ALICE
    engine['manager'].process_frame(FRAME)                # seen 10:14:00, no blink
    engine['away'](120)                                   # gone two minutes
    face = engine['step_up'](ALICE, 3)                    # back 10:16:00
    assert face.attendance_status == AttendanceStatus.LATE
    assert _status(temp_db, session_id, alice)['arrived_at'] == "2026-10-04 10:16:00"


def test_line_that_reached_past_the_late_time_keeps_everyone_present(temp_db, session_id, alice, bob, engine):
    """Alice at 10:14:50, Bob steps up 10 s after her: still the same line -> present."""
    engine['clock'].advance(50)
    engine['step_up'](ALICE, 6)                           # 10:14:50 -> marked 10:14:56
    engine['away'](10)
    face = engine['step_up'](BOB, 6)                      # steps up 10:15:06
    assert face.attendance_status == AttendanceStatus.PRESENT
    record = _status(temp_db, session_id, bob)
    assert record['arrived_at'] == "2026-10-04 10:14:50"
    assert "In the line at the camera since 10:14:50" in record['notes']


def test_a_gap_breaks_the_line(temp_db, session_id, alice, bob, engine):
    """Nobody at the camera for longer than LATE_LINE_GAP_SECONDS: Bob starts a new line -> late."""
    engine['clock'].advance(50)
    engine['step_up'](ALICE, 6)
    engine['away'](config.LATE_LINE_GAP_SECONDS + 5)
    face = engine['step_up'](BOB, 6)
    assert face.attendance_status == AttendanceStatus.LATE


def test_new_session_starts_without_first_seen_or_line(engine, session_id):
    manager = engine['manager']
    manager._note_arrival(1, engine['clock']())
    manager.set_session(session_id)
    assert manager._first_seen == {}
    assert manager._line_started_at is None
