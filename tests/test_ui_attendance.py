"""
Tests for Phase 4D UI logic and Page renderers.

Verifies page functions, handlers, session lifecycle integration,
error handling, and rendering utilities without physical camera or browser.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.database.schema import initialize_database
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.student_service import StudentService
from src.ui.pages.session_management import (
    render_session_management,
    handle_create_session,
    handle_activate_session,
    handle_close_session,
    format_datetime
)
from src.ui.pages.realtime_attendance import (
    render_realtime_attendance,
    _draw_attendance_overlays,
    _build_person_info_markdown
)
from src.services.real_time_attendance_service import (
    RealTimeAttendanceResult,
    FaceProcessingResult,
    MULTIPLE_FACES_MESSAGE
)
from src.computer_vision.liveness_detector import LivenessState


@pytest.fixture
def temp_db():
    """Create a temporary database for UI test cases."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    initialize_database(db_path)
    yield db_path
    
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def session_service(temp_db):
    return SessionService(temp_db)


@pytest.fixture
def attendance_service(temp_db):
    return AttendanceService(temp_db)


@pytest.fixture
def student_service(temp_db):
    return StudentService(temp_db)


# ============================================================================
# Session Management Action Handlers & Logic Tests
# ============================================================================

def test_handle_create_session_success(session_service):
    """Test successful session creation via UI handler."""
    with patch("streamlit.success") as mock_success, \
         patch("streamlit.rerun") as mock_rerun, \
         patch("streamlit.session_state", {}):
        handle_create_session(
            session_service=session_service,
            session_name="CS101 Lab 1",
            start_time="2026-10-04 10:00:00",
            course_code="CS101",
            late_threshold_minutes=15,
            status=SessionStatus.SCHEDULED,
            created_by="Instructor A",
            notes="Introductory lab"
        )
        mock_success.assert_called_once()
        mock_rerun.assert_called_once()
        
    sessions = session_service.get_all_sessions()
    assert len(sessions) == 1
    assert sessions[0]['session_name'] == "CS101 Lab 1"


def test_handle_create_session_empty_name_error(session_service):
    """Test session creation validation with empty name."""
    with patch("streamlit.error") as mock_error:
        handle_create_session(
            session_service=session_service,
            session_name="",
            start_time="2026-10-04 10:00:00",
            course_code="CS101",
            late_threshold_minutes=15,
            status=SessionStatus.SCHEDULED,
            created_by="Instructor A",
            notes=""
        )
        mock_error.assert_called_once()
        
    sessions = session_service.get_all_sessions()
    assert len(sessions) == 0


def test_handle_activate_session_flow(session_service):
    """Test activating a scheduled session via UI handler."""
    sess_id = session_service.create_session(
        session_name="CS102 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.SCHEDULED
    )
    
    with patch("streamlit.success") as mock_success, \
         patch("streamlit.rerun") as mock_rerun:
        handle_activate_session(session_service, sess_id)
        mock_success.assert_called_once()
        mock_rerun.assert_called_once()
        
    session = session_service.get_session(sess_id)
    assert session['status'] == SessionStatus.ACTIVE


def test_handle_close_session_flow(session_service):
    """Test closing an active session via UI handler."""
    sess_id = session_service.create_session(
        session_name="CS102 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.ACTIVE
    )
    
    with patch("streamlit.success") as mock_success, \
         patch("streamlit.rerun") as mock_rerun:
        handle_close_session(session_service, sess_id)
        mock_success.assert_called_once()
        mock_rerun.assert_called_once()
        
    session = session_service.get_session(sess_id)
    assert session['status'] == SessionStatus.CLOSED
    assert session['closed_at'] is not None


def test_handle_activate_closed_session_rejected(session_service):
    """Test that activating an already closed session shows an error message."""
    sess_id = session_service.create_session(
        session_name="CS102 Lecture",
        start_time="2026-10-04 10:00:00",
        status=SessionStatus.CLOSED
    )
    
    with patch("streamlit.error") as mock_error:
        handle_activate_session(session_service, sess_id)
        mock_error.assert_called_once()


# ============================================================================
# Datetime Formatting Utility Tests
# ============================================================================

def test_format_datetime_valid():
    """Test formatting valid datetime strings."""
    assert format_datetime("2026-10-04 10:30:00") == "2026-10-04 10:30:00"
    assert format_datetime("2026-10-04T10:30:00") == "2026-10-04 10:30:00"


def test_format_datetime_none_or_empty():
    """Test formatting None or empty strings."""
    assert format_datetime(None) == "—"
    assert format_datetime("") == "—"


# ============================================================================
# Real-Time UI Overlay Drawing Tests
# ============================================================================

def test_draw_attendance_overlays():
    """Test drawing visual overlays on mirrored frame for recognized and live faces."""
    dummy_frame = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Face at x=20, w=50 in unmirrored frame (width 300)
    # Mirrored x should be 300 - (20 + 50) = 230
    face1 = FaceProcessingResult(
        bbox=(20, 20, 50, 50),
        detection_confidence=0.95,
        identity="2021001",
        is_recognized=True,
        student_name="Alice",
        liveness_state=LivenessState.LIVE,
        attendance_marked=True,
        attendance_status="present"
    )
    
    face2 = FaceProcessingResult(
        bbox=(100, 100, 40, 40),
        detection_confidence=0.88,
        identity="Unknown",
        is_recognized=False,
        liveness_state=LivenessState.UNKNOWN,
        attendance_marked=False
    )
    
    res = RealTimeAttendanceResult(
        session_id=1,
        faces_detected=2,
        processed_faces=[face1, face2],
        session_active=True,
        marked_count=1
    )
    
    output_frame = _draw_attendance_overlays(dummy_frame, res)
    assert output_frame is not None
    assert output_frame.shape == dummy_frame.shape
    assert isinstance(output_frame, np.ndarray)


def test_draw_attendance_overlays_shows_clean_mirrored_camera_view():
    """The preview is the plain mirrored camera image: no box, name or status
    is drawn on the face (those are shown in the panel next to the camera).
    """
    dummy_frame = np.zeros((200, 300, 3), dtype=np.uint8)
    dummy_frame[:, :10] = 255  # bright strip on the left edge to check the mirror
    face = FaceProcessingResult(
        bbox=(20, 20, 50, 50),
        detection_confidence=0.95,
        identity="Unknown",
        is_recognized=False,
        liveness_state=LivenessState.UNKNOWN,
        attendance_marked=False
    )
    res = RealTimeAttendanceResult(
        session_id=1,
        faces_detected=1,
        processed_faces=[face],
        session_active=True,
        marked_count=0
    )

    original = dummy_frame.copy()
    with patch("src.ui.pages.realtime_attendance.cv2.rectangle") as mock_rectangle, \
         patch("src.ui.pages.realtime_attendance.cv2.putText") as mock_put_text:
        output = _draw_attendance_overlays(dummy_frame, res)

    mock_rectangle.assert_not_called()
    mock_put_text.assert_not_called()
    # Exactly the mirrored camera image: the left strip is now on the right
    assert np.array_equal(output, original[:, ::-1])
    # The bbox stays available for the pipeline and the input frame is untouched
    assert face.bbox == (20, 20, 50, 50)
    assert np.array_equal(dummy_frame, original)


def test_person_panel_asks_for_one_person_at_a_time():
    """With several faces in view the panel says so instead of showing a person."""
    faces = [
        FaceProcessingResult(bbox=(10, 10, 40, 40), detection_confidence=0.9,
                             reason=MULTIPLE_FACES_MESSAGE),
        FaceProcessingResult(bbox=(60, 10, 40, 40), detection_confidence=0.9,
                             reason=MULTIPLE_FACES_MESSAGE),
    ]
    res = RealTimeAttendanceResult(
        session_id=1, faces_detected=2, processed_faces=faces, session_active=True,
        multiple_faces=True, message=MULTIPLE_FACES_MESSAGE
    )

    text = _build_person_info_markdown(res)

    assert MULTIPLE_FACES_MESSAGE in text
    assert "2 faces" in text
    assert "Unknown Person" not in text


def test_person_panel_already_marked_student_not_asked_to_blink():
    """A marked student still in view is not told to blink or turn again."""
    face = FaceProcessingResult(
        bbox=(10, 10, 40, 40), detection_confidence=0.9, identity="2021001",
        is_recognized=True, student_id="2021001", student_name="Alice Smith",
        blink_count=0, min_blinks_required=1, challenge_type="turn_left",
        challenge_state="active", already_marked=True,
        reason="Attendance already recorded for this session"
    )
    res = RealTimeAttendanceResult(
        session_id=1, faces_detected=1, processed_faces=[face], session_active=True
    )

    text = _build_person_info_markdown(res)

    assert "Already Marked" in text
    assert "Blink" not in text
    assert "Turn your head" not in text


def _pending_student(**overrides):
    fields = dict(
        bbox=(10, 10, 40, 40), detection_confidence=0.9, identity="2021001",
        is_recognized=True, student_id="2021001", student_name="Alice Smith",
        blink_count=1, min_blinks_required=3, challenge_type="turn_right",
        challenge_state="active", reason="Liveness verification pending / not confirmed"
    )
    fields.update(overrides)
    return FaceProcessingResult(**fields)


def _panel(face):
    return _build_person_info_markdown(RealTimeAttendanceResult(
        session_id=1, faces_detected=1, processed_faces=[face], session_active=True
    ))


def test_person_panel_states_the_random_blink_count_and_time_left():
    face = _pending_student()
    face.challenge_seconds_left = 6.2
    face.spoof_score = 0.93

    text = _panel(face)

    assert "Blink 3 times" in text and "1/3 blinks" in text
    assert "Turn your head RIGHT" in text
    assert "7 s left" in text
    assert "Real-face score:** 0.93" in text


def test_person_panel_shows_a_suspected_spoof_instead_of_instructions():
    text = _panel(_pending_student(reason="Spoof suspected: face looks like a photo or screen"))

    assert "🔴" in text and "face looks like a photo or screen" in text
    assert "Blink 3 times" not in text


def test_stream_reloads_gallery_for_existing_manager():
    """Students enrolled during an active session reach the running manager."""
    from types import SimpleNamespace
    from src.ui.pages.realtime_attendance import _run_realtime_attendance_stream

    camera = MagicMock()
    camera.is_open.return_value = True
    rt_manager = MagicMock()
    rt_manager.current_session_id = 7
    state = SimpleNamespace(
        att_camera_manager=camera,
        att_rt_manager=rt_manager,
        att_camera_active=False,  # stop before the frame loop starts
    )
    new_gallery = [("2021001", MagicMock()), ("2021002", MagicMock())]

    with patch("streamlit.session_state", state), \
         patch("src.ui.pages.realtime_attendance.EnrollmentService") as mock_enrollment:
        mock_enrollment.return_value.load_active_gallery.return_value = new_gallery
        _run_realtime_attendance_stream(
            session_id=7,
            require_liveness=True,
            cooldown_seconds=5.0,
            session_service=MagicMock(),
            attendance_service=MagicMock(),
            student_service=MagicMock(),
            head_turn_angle=15.0,
        )

    assert state.att_rt_manager is rt_manager  # same manager, not recreated
    rt_manager.set_gallery.assert_called_once_with(new_gallery)


# ============================================================================
# Streamlit Render Function Tests (Mocked UI Environment)
# ============================================================================

@patch("src.services.session_service.SessionService.get_all_sessions")
@patch("streamlit.tabs")
def test_render_session_management_page(mock_tabs, mock_get_sessions):
    """Test rendering the session management page structure."""
    mock_get_sessions.return_value = []
    mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    
    with patch("streamlit.session_state", {}):
        render_session_management()
        mock_tabs.assert_called_once()


@patch("src.services.session_service.SessionService.get_all_sessions")
def test_render_realtime_attendance_no_sessions(mock_get_sessions):
    """Test real-time attendance page rendering when no sessions exist."""
    mock_get_sessions.return_value = []
    
    with patch("streamlit.warning") as mock_warning, \
         patch("streamlit.sidebar"), \
         patch("streamlit.session_state", {}):
        render_realtime_attendance()
        mock_warning.assert_called_once()


@patch("src.ui.pages.student_registry.render_manage_face_view")
def test_render_student_registry_manage_face_active(mock_render_face):
    """Test that student registry directly renders Manage Face when view is manage_face."""
    from src.ui.pages.student_registry import render_student_registry
    
    fake_state = {'current_view': 'manage_face', 'selected_student_id': 1}
    with patch("streamlit.session_state", fake_state):
        render_student_registry()
        mock_render_face.assert_called_once_with(mock_render_face.call_args[0][0], mock_render_face.call_args[0][1], 1)


@patch("src.ui.pages.student_registry.render_edit_student_view")
def test_render_student_registry_edit_profile_active(mock_render_edit):
    """Test that student registry directly renders Edit Profile when view is edit."""
    from src.ui.pages.student_registry import render_student_registry
    
    fake_state = {'current_view': 'edit', 'selected_student_id': 1}
    with patch("streamlit.session_state", fake_state):
        render_student_registry()
        mock_render_edit.assert_called_once_with(mock_render_edit.call_args[0][0], 1)


@patch("src.ui.pages.student_registry.render_directory_view")
def test_render_student_registry_default_directory(mock_dir):
    """Test that student registry renders directory view when view is directory."""
    from src.ui.pages.student_registry import render_student_registry
    
    fake_state = {'current_view': 'directory', 'selected_student_id': None}
    with patch("streamlit.session_state", fake_state):
        render_student_registry()
        mock_dir.assert_called_once()


def test_release_registration_camera_cleans_up():
    """Test that release_registration_camera releases camera and resets state."""
    from src.ui.pages.student_registry import release_registration_camera
    
    mock_cam = MagicMock()
    fake_state = {
        'reg_cam_manager': mock_cam,
        'reg_cam_active': True,
        'captured_photo_preview': np.zeros((10, 10, 3), dtype=np.uint8)
    }
    
    with patch("streamlit.session_state", fake_state):
        release_registration_camera()
        mock_cam.release.assert_called_once()
        assert fake_state['reg_cam_manager'] is None
        assert fake_state['reg_cam_active'] is False
        assert fake_state['captured_photo_preview'] is None


@patch("streamlit.radio")
@patch("src.services.enrollment_service.EnrollmentService.get_student_samples_count")
@patch("src.services.student_service.StudentService.get_student_by_id")
@patch("src.services.student_service.StudentService.get_all_students")
def test_manage_face_camera_off_by_default(mock_get_all, mock_get_by_id, mock_get_count, mock_radio):
    """Test that opening Manage Face has camera OFF by default and shows Start Camera button."""
    from src.ui.pages.student_registry import render_manage_face_view
    
    mock_get_all.return_value = [{'id': 1, 'student_id': '2021001', 'name': 'John Doe'}]
    mock_get_by_id.return_value = {'id': 1, 'student_id': '2021001', 'name': 'John Doe', 'department': 'CS'}
    mock_get_count.return_value = 0
    mock_radio.return_value = "📹 Take Photo with Live Camera"
    
    fake_state = {
        'selected_student_id': 1,
        'reg_cam_active': False,
        'reg_cam_manager': None,
        'captured_photo_preview': None,
        'confirm_reset_face_id': None
    }
    
    mock_student_svc = MagicMock()
    mock_student_svc.get_all_students.return_value = [{'id': 1, 'student_id': '2021001', 'name': 'John Doe'}]
    mock_student_svc.get_student_by_id.return_value = {'id': 1, 'student_id': '2021001', 'name': 'John Doe', 'department': 'CS'}
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_student_samples_count.return_value = 0
    
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.button") as mock_btn:
        mock_btn.return_value = False
        render_manage_face_view(mock_student_svc, mock_enroll_svc, 1)
        assert fake_state['reg_cam_active'] is False
        assert fake_state['reg_cam_manager'] is None


@patch("streamlit.radio")
@patch("streamlit.image")
def test_manage_face_photo_preview_approval_workflow(mock_img, mock_radio):
    """Test that captured photo is frozen for approval and camera stays active."""
    from src.ui.pages.student_registry import render_manage_face_view
    
    mock_radio.return_value = "📹 Take Photo with Live Camera"
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cam = MagicMock()
    mock_cam.is_open.return_value = True
    
    fake_state = {
        'selected_student_id': 1,
        'reg_cam_active': True,
        'reg_cam_manager': mock_cam,
        'captured_photo_preview': dummy_frame,
        'confirm_reset_face_id': None
    }
    
    mock_student_svc = MagicMock()
    mock_student_svc.get_all_students.return_value = [{'id': 1, 'student_id': '2021001', 'name': 'John Doe'}]
    mock_student_svc.get_student_by_id.return_value = {'id': 1, 'student_id': '2021001', 'name': 'John Doe', 'department': 'CS'}
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_student_samples_count.return_value = 0
    
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.button") as mock_btn:
        mock_btn.return_value = False
        render_manage_face_view(mock_student_svc, mock_enroll_svc, 1)
        # Image was rendered for user review
        mock_img.assert_called_once()
        # Camera was NOT released
        mock_cam.release.assert_not_called()
        assert fake_state['reg_cam_active'] is True


def test_manage_face_reset_confirmation_warning():
    """Test that reset face dataset requires confirmation and does not delete immediately."""
    from src.ui.pages.student_registry import render_manage_face_view
    
    mock_student_svc = MagicMock()
    mock_student_svc.get_all_students.return_value = [{'id': 1, 'student_id': '2021001', 'name': 'John Doe'}]
    mock_student_svc.get_student_by_id.return_value = {'id': 1, 'student_id': '2021001', 'name': 'John Doe', 'department': 'CS'}
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_student_samples_count.return_value = 3
    
    # State with confirmation active
    fake_state = {
        'selected_student_id': 1,
        'confirm_reset_face_id': 1,
        'reg_cam_active': False,
        'reg_cam_manager': None,
        'captured_photo_preview': None
    }
    
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.warning") as mock_warn, \
         patch("streamlit.button") as mock_btn, \
         patch("streamlit.radio"):
        mock_btn.return_value = False
        render_manage_face_view(mock_student_svc, mock_enroll_svc, 1)
        # Warning shown
        mock_warn.assert_called_once()
        # Not deleted without explicit confirmation click
        mock_enroll_svc.clear_student_face_dataset.assert_not_called()
