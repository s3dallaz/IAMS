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


@patch("src.ui.pages.student_registry.render_face_dataset_tab")
def test_render_student_registry_manage_face_active(mock_render_face):
    """Test that student registry directly renders Manage Face when student ID is in session state."""
    from src.ui.pages.student_registry import render_student_registry
    
    fake_state = {'manage_face_student_id': 1}
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.button") as mock_btn:
        mock_btn.return_value = False
        render_student_registry()
        mock_render_face.assert_called_once()


@patch("src.ui.pages.student_registry.render_edit_student_tab")
def test_render_student_registry_edit_profile_active(mock_render_edit):
    """Test that student registry directly renders Edit Profile when student ID is in session state."""
    from src.ui.pages.student_registry import render_student_registry
    
    fake_state = {'edit_student_id': 1}
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.button") as mock_btn:
        mock_btn.return_value = False
        render_student_registry()
        mock_render_edit.assert_called_once()


@patch("streamlit.tabs")
def test_render_student_registry_default_tabs(mock_tabs):
    """Test that student registry renders standard tabs when no specific student action is active."""
    from src.ui.pages.student_registry import render_student_registry
    
    mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    fake_state = {'manage_face_student_id': None, 'edit_student_id': None}
    
    with patch("streamlit.session_state", fake_state), \
         patch("src.ui.pages.student_registry.render_student_list_tab"), \
         patch("src.ui.pages.student_registry.render_add_student_tab"), \
         patch("src.ui.pages.student_registry.render_face_dataset_tab"), \
         patch("src.ui.pages.student_registry.render_edit_student_tab"):
        render_student_registry()
        mock_tabs.assert_called_once()
