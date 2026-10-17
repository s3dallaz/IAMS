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
    _draw_attendance_overlays
)
from src.services.real_time_attendance_service import (
    RealTimeAttendanceResult,
    FaceProcessingResult
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


def test_draw_attendance_overlays_draws_labels_without_face_rectangle():
    """The preview shows name/liveness labels but no box around the face."""
    dummy_frame = np.zeros((200, 300, 3), dtype=np.uint8)
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

    with patch("src.ui.pages.realtime_attendance.cv2.rectangle") as mock_rectangle, \
         patch("src.ui.pages.realtime_attendance.cv2.putText") as mock_put_text:
        _draw_attendance_overlays(dummy_frame, res)

    mock_rectangle.assert_not_called()
    labels = [call.args[1] for call in mock_put_text.call_args_list]
    assert labels == ["Unknown", "Live: unknown"]
    # Label is positioned from the mirrored bbox: 300 - (20 + 50) = 230
    assert mock_put_text.call_args_list[0].args[2] == (230, 20)
    # The bbox is only read, never modified, and the input frame is untouched
    assert face.bbox == (20, 20, 50, 50)
    assert not dummy_frame.any()


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
