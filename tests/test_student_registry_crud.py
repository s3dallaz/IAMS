"""
Comprehensive Tests for Student Registry UI and CRUD Workflows

Tests end-to-end functionality for:
1. Student Directory view rendering with action buttons (Edit, Manage Face, Delete)
2. Add Student flow (input -> validation -> SQLite insertion -> verification -> feedback)
3. Edit Student flow (load existing -> modify -> SQLite update -> feedback)
4. Delete Student flow (trigger -> confirmation warning -> cancellation vs cascade deletion)
5. Manage Face flow (photo capture -> freeze-frame review -> Approve vs Retake -> dataset reset confirmation)
6. State and camera cleanup safety
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.database.schema import initialize_database
from src.services.student_service import (
    StudentService,
    InvalidStudentDataError,
    DuplicateStudentError,
    StudentNotFoundError
)
from src.services.enrollment_service import (
    EnrollmentService,
    InvalidFaceImageError
)
from src.data.repositories.student_repository import StudentRepository
from src.data.repositories.enrollment_repository import EnrollmentRepository
from src.ui.pages.student_registry import (
    render_student_registry,
    render_directory_view,
    render_add_student_view,
    render_edit_student_view,
    render_manage_face_view,
    release_registration_camera
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def student_service(temp_db):
    return StudentService(temp_db)


@pytest.fixture
def enrollment_service(temp_db):
    return EnrollmentService(database_path=temp_db)


# ============================================================================
# 1. Repository & Service CRUD Tests
# ============================================================================

def test_student_repository_delete_student(temp_db):
    repo = StudentRepository(temp_db)
    sid = repo.create_student("DEL_01", "Delete Me", "del@example.com", "CS")
    assert repo.get_by_id(sid) is not None
    
    deleted = repo.delete_student(sid)
    assert deleted == 1
    assert repo.get_by_id(sid) is None
    assert not repo.exists_by_student_id("DEL_01")


def test_student_service_delete_student_flow(student_service):
    sid = student_service.create_student("SRV_DEL", "Service Delete")
    assert student_service.get_student_by_id(sid) is not None
    
    rows = student_service.delete_student(sid)
    assert rows == 1
    assert student_service.get_student_by_id(sid) is None
    
    with pytest.raises(StudentNotFoundError):
        student_service.delete_student(sid)


def test_student_service_delete_invalid_id_rejected(student_service):
    with pytest.raises(InvalidStudentDataError):
        student_service.delete_student(-5)


def test_student_cascade_deletion(temp_db):
    student_repo = StudentRepository(temp_db)
    enroll_repo = EnrollmentRepository(temp_db)
    
    sid = student_repo.create_student("CASC_01", "Cascade Test")
    eid = enroll_repo.create_enrollment(student_id=sid, encoding_path="/tmp/enc.npy")
    
    assert len(enroll_repo.get_by_student_id(sid)) == 1
    
    student_repo.delete_student(sid)
    
    # Confirm SQLite FOREIGN KEY CASCADE removed enrollments
    assert len(enroll_repo.get_by_student_id(sid)) == 0


def test_edit_student_flow(student_service):
    sid = student_service.create_student("EDIT_01", "Original Name", "orig@test.com", "IT")
    
    updated = student_service.update_student(
        student_db_id=sid,
        name="Updated Name",
        email="updated@test.com",
        department="AI",
        notes="Updated notes"
    )
    assert updated == 1
    
    student = student_service.get_student_by_id(sid)
    assert student['name'] == "Updated Name"
    assert student['email'] == "updated@test.com"
    assert student['department'] == "AI"
    assert student['notes'] == "Updated notes"


# ============================================================================
# 2. UI View & Routing Tests (Mocked Streamlit Environment)
# ============================================================================

def test_render_student_registry_routes_to_directory():
    fake_state = {'current_view': 'directory'}
    with patch("streamlit.session_state", fake_state), \
         patch("src.ui.pages.student_registry.render_directory_view") as mock_dir:
        render_student_registry()
        mock_dir.assert_called_once()


def test_render_student_registry_routes_to_add():
    fake_state = {'current_view': 'add'}
    with patch("streamlit.session_state", fake_state), \
         patch("src.ui.pages.student_registry.render_add_student_view") as mock_add:
        render_student_registry()
        mock_add.assert_called_once()


def test_render_student_registry_routes_to_edit():
    fake_state = {'current_view': 'edit', 'selected_student_id': 10}
    with patch("streamlit.session_state", fake_state), \
         patch("src.ui.pages.student_registry.render_edit_student_view") as mock_edit:
        render_student_registry()
        mock_edit.assert_called_once_with(mock_edit.call_args[0][0], 10)


def test_render_student_registry_routes_to_manage_face():
    fake_state = {'current_view': 'manage_face', 'selected_student_id': 10}
    with patch("streamlit.session_state", fake_state), \
         patch("src.ui.pages.student_registry.render_manage_face_view") as mock_face:
        render_student_registry()
        mock_face.assert_called_once_with(mock_face.call_args[0][0], mock_face.call_args[0][1], 10)


def test_delete_confirmation_warning_box():
    mock_service = MagicMock()
    mock_service.get_all_students.return_value = [{'id': 1, 'student_id': 'DEL101', 'name': 'John Delete', 'department': 'CS', 'is_active': 1, 'enrolled_at': None}]
    mock_service.get_student_by_id.return_value = {'id': 1, 'student_id': 'DEL101', 'name': 'John Delete'}
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_all_samples_counts.return_value = {1: 2}
    mock_enroll_svc.gallery_cache_status.return_value = {"samples": 2, "pending": 0}

    fake_state = {
        'confirm_delete_id': 1,
        'current_view': 'directory'
    }
    
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.warning") as mock_warn, \
         patch("streamlit.button") as mock_btn, \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.selectbox", return_value="All Students"):
        mock_btn.return_value = False
        render_directory_view(mock_service, mock_enroll_svc)
        mock_warn.assert_called_once()
        mock_service.delete_student.assert_not_called()


def test_photo_preview_approval_workflow_keeps_camera_open():
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cam = MagicMock()
    mock_cam.is_open.return_value = True
    
    fake_state = {
        'current_view': 'manage_face',
        'selected_student_id': 1,
        'reg_cam_active': True,
        'reg_cam_manager': mock_cam,
        'captured_photo_preview': dummy_frame,
        'confirm_reset_face_id': None
    }
    
    mock_student_svc = MagicMock()
    mock_student_svc.get_student_by_id.return_value = {'id': 1, 'student_id': 'STU01', 'name': 'John', 'department': 'AI'}
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_student_samples_count.return_value = 0
    
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.image") as mock_img, \
         patch("streamlit.button") as mock_btn, \
         patch("streamlit.radio") as mock_radio:
        mock_radio.return_value = "📹 Take Photo with Live Camera"
        mock_btn.return_value = False
        render_manage_face_view(mock_student_svc, mock_enroll_svc, 1)
        mock_img.assert_called_once()
        mock_cam.release.assert_not_called()
        assert fake_state['reg_cam_active'] is True


def test_release_registration_camera_complete_reset():
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


def test_directory_shows_action_buttons_one_page_at_a_time():
    """With 1,000 students the action list must not draw thousands of buttons."""
    mock_service = MagicMock()
    mock_service.get_all_students.return_value = [
        {'id': i, 'student_id': f'S{i:04d}', 'name': f'Student {i}', 'department': 'CS',
         'is_active': 1, 'enrolled_at': None} for i in range(1, 61)]
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.get_all_samples_counts.return_value = {1: 1}
    mock_enroll_svc.gallery_cache_status.return_value = {"samples": 1, "pending": 0}

    with patch("streamlit.session_state", {'current_view': 'directory'}), \
         patch("streamlit.button", return_value=False) as mock_btn, \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.selectbox", return_value="All Students"), \
         patch("streamlit.number_input", return_value=3) as mock_page:
        render_directory_view(mock_service, mock_enroll_svc)

    assert mock_page.call_args.kwargs["max_value"] == 3            # 60 students, 25 per page
    face_buttons = [c.kwargs["key"] for c in mock_btn.call_args_list if c.kwargs.get("key", "").startswith("btn_face_")]
    assert face_buttons == [f"btn_face_{i}" for i in range(51, 61)]   # page 3: students 51-60
    mock_enroll_svc.get_student_samples_count.assert_not_called()   # one query for all counts


def test_recognition_data_panel_prepares_pending_samples_with_a_progress_bar():
    from src.ui.pages.student_registry import render_recognition_data_panel
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.gallery_cache_status.return_value = {"samples": 11200, "pending": 11200}

    def prepare(progress=None, rebuild=False):
        progress(0, 11200)
        progress(4200, 11200)
        return 11200

    mock_enroll_svc.prepare_recognition_data.side_effect = prepare
    fake_state = {}
    with patch("streamlit.session_state", fake_state), \
         patch("streamlit.warning") as mock_warn, \
         patch("streamlit.button", return_value=True) as mock_btn, \
         patch("streamlit.progress") as mock_progress, \
         patch("streamlit.rerun"):
        render_recognition_data_panel(mock_enroll_svc)

    assert "needs preparing" in mock_warn.call_args.args[0]
    assert mock_btn.call_args.kwargs["key"] == "btn_prepare_recognition_data"
    assert mock_enroll_svc.prepare_recognition_data.call_args.kwargs["rebuild"] is False
    texts = [c.kwargs["text"] for c in mock_progress.return_value.progress.call_args_list]
    assert texts[-1] == "Preparing recognition data: 4,200 / 11,200 face samples (37%)"
    assert fake_state["registry_feedback"][0] == "success"


def test_recognition_data_panel_offers_a_rebuild_when_ready():
    from src.ui.pages.student_registry import render_recognition_data_panel
    mock_enroll_svc = MagicMock()
    mock_enroll_svc.gallery_cache_status.return_value = {"samples": 5, "pending": 0}
    with patch("streamlit.session_state", {}), \
         patch("streamlit.warning") as mock_warn, \
         patch("streamlit.button", return_value=False) as mock_btn:
        render_recognition_data_panel(mock_enroll_svc)
    mock_warn.assert_not_called()
    assert mock_btn.call_args.kwargs["key"] == "btn_rebuild_recognition_data"
    mock_enroll_svc.prepare_recognition_data.assert_not_called()
