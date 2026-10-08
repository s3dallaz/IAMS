"""
Student Registry and Face Dataset Management UI Page.

Provides a clean, reliable, single-page Streamlit interface for:
1. Viewing all students in a live directory table with search, status filters, and instant action buttons.
2. Registering new students with immediate SQLite persistence, verification, and live directory updates.
3. Editing student profile information with direct feedback.
4. Deleting students safely with explicit confirmation (including cascade deletion of enrollments/attendance).
5. Managing student face datasets (continuous live preview, photo capture, approve/retake workflow, and reset dataset confirmation).
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, List

from src.services.student_service import (
    StudentService,
    InvalidStudentDataError,
    DuplicateStudentError,
    StudentNotFoundError,
    StudentServiceError
)
from src.services.enrollment_service import (
    EnrollmentService,
    EnrollmentServiceError,
    InvalidFaceImageError
)
from src.computer_vision.camera_manager import CameraManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def release_registration_camera():
    """Ensure the registration camera is released and session state reset."""
    if st.session_state.get('reg_cam_manager') is not None:
        try:
            st.session_state['reg_cam_manager'].release()
        except Exception as e:
            logger.error(f"Error releasing registration camera: {e}")
        st.session_state['reg_cam_manager'] = None
    st.session_state['reg_cam_active'] = False
    st.session_state['captured_photo_preview'] = None


def render_student_registry():
    """Main entry point for Student Registry and Face Dataset Management."""
    st.title("👥 Student Registry & Face Management")
    st.caption("Student Profiles, Face Dataset Enrollment & Appearance Management")
    st.markdown("---")
    
    student_service = StudentService()
    enrollment_service = EnrollmentService()
    
    # Initialize state keys
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'directory'  # 'directory', 'add', 'edit', 'manage_face'
    if 'selected_student_id' not in st.session_state:
        st.session_state['selected_student_id'] = None
    if 'confirm_delete_id' not in st.session_state:
        st.session_state['confirm_delete_id'] = None
    if 'confirm_reset_face_id' not in st.session_state:
        st.session_state['confirm_reset_face_id'] = None
    if 'reg_cam_active' not in st.session_state:
        st.session_state['reg_cam_active'] = False
    if 'reg_cam_manager' not in st.session_state:
        st.session_state['reg_cam_manager'] = None
    if 'captured_photo_preview' not in st.session_state:
        st.session_state['captured_photo_preview'] = None
    if 'registry_feedback' not in st.session_state:
        st.session_state['registry_feedback'] = None
        
    # Show top feedback banner if set
    if st.session_state.get('registry_feedback'):
        fb_type, fb_msg = st.session_state['registry_feedback']
        if fb_type == 'success':
            st.success(fb_msg)
        elif fb_type == 'error':
            st.error(fb_msg)
        elif fb_type == 'info':
            st.info(fb_msg)
        st.session_state['registry_feedback'] = None

    # Route based on current_view
    view = st.session_state['current_view']
    
    if view == 'manage_face' and st.session_state.get('selected_student_id') is not None:
        render_manage_face_view(student_service, enrollment_service, st.session_state['selected_student_id'])
    elif view == 'edit' and st.session_state.get('selected_student_id') is not None:
        render_edit_student_view(student_service, st.session_state['selected_student_id'])
    elif view == 'add':
        render_add_student_view(student_service)
    else:
        # Default view: Full Directory with Add expander and table actions
        render_directory_view(student_service, enrollment_service)


# ============================================================================
# View 1: Main Directory & Student List
# ============================================================================

def render_directory_view(service: StudentService, enrollment_service: EnrollmentService):
    """Render the main directory list with quick actions (Edit, Manage Face, Delete, Activate/Deactivate)."""
    
    # Top Action Bar
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.subheader("📋 Student Directory")
    with col_btn:
        if st.button("➕ Add New Student", type="primary", use_container_width=True, key="btn_open_add_student"):
            release_registration_camera()
            st.session_state['current_view'] = 'add'
            st.rerun()

    # Search and Filter
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_query = st.text_input(
            "Search by ID or Name",
            placeholder="Type student ID or name...",
            key="input_search_query"
        )
    with col_s2:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All Students", "Active Students", "Inactive Students"],
            key="select_status_filter"
        )

    # Fetch students directly from SQLite
    try:
        if status_filter == "Active Students":
            students = service.get_all_students(active_only=True)
        elif status_filter == "Inactive Students":
            all_s = service.get_all_students(active_only=False)
            students = [s for s in all_s if s['is_active'] == 0]
        else:
            students = service.get_all_students(active_only=False)

        if search_query:
            q = search_query.strip().lower()
            students = [s for s in students if q in s['student_id'].lower() or q in s['name'].lower()]

        st.caption(f"Total: {len(students)} student(s)")

        if not students:
            st.info("No students found matching current criteria. Click **➕ Add New Student** above to register.")
            return

        # Overview Table
        table_rows = []
        for s in students:
            samples_cnt = enrollment_service.get_student_samples_count(s['id'])
            face_status = f"✅ {samples_cnt} sample(s)" if samples_cnt > 0 else "⚠️ No face data"
            table_rows.append({
                'ID': s['student_id'],
                'Name': s['name'],
                'Department': s['department'] or '—',
                'Status': "Active" if s['is_active'] else "Inactive",
                'Face Dataset': face_status,
                'Enrolled At': format_datetime(s['enrolled_at'])
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("#### ⚡ Manage Students")
        
        # Confirmation box for delete if triggered
        del_target_id = st.session_state.get('confirm_delete_id')
        if del_target_id:
            del_student = service.get_student_by_id(del_target_id)
            if del_student:
                st.warning(f"⚠️ **Are you sure you want to permanently delete {del_student['name']} (`{del_student['student_id']}`)?**\nThis will also remove all face samples and attendance records.")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("❌ Cancel Deletion", key="btn_cancel_delete", use_container_width=True):
                        st.session_state['confirm_delete_id'] = None
                        st.rerun()
                with col_d2:
                    if st.button("🗑️ Yes, Delete Student", type="primary", key="btn_confirm_delete", use_container_width=True):
                        try:
                            # Clean up face samples folder first
                            enrollment_service.clear_student_face_dataset(del_target_id)
                            # Delete from database
                            service.delete_student(del_target_id)
                            st.session_state['confirm_delete_id'] = None
                            st.session_state['registry_feedback'] = ('success', f"✅ Student **{del_student['name']}** deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete student: {e}")
                st.markdown("---")

        # Action rows for each student
        for s in students:
            samples_cnt = enrollment_service.get_student_samples_count(s['id'])
            face_badge = f"📸 Manage Face ({samples_cnt})" if samples_cnt > 0 else "📸 Add Face Data"
            
            row_cols = st.columns([1.5, 2.5, 2, 1.5, 1.2, 0.8])
            row_cols[0].write(f"`{s['student_id']}`")
            row_cols[1].write(f"**{s['name']}**")
            
            with row_cols[2]:
                if st.button(face_badge, key=f"btn_face_{s['id']}", use_container_width=True):
                    release_registration_camera()
                    st.session_state['selected_student_id'] = s['id']
                    st.session_state['current_view'] = 'manage_face'
                    st.rerun()
                    
            with row_cols[3]:
                if st.button("✏️ Edit", key=f"btn_edit_{s['id']}", use_container_width=True):
                    st.session_state['selected_student_id'] = s['id']
                    st.session_state['current_view'] = 'edit'
                    st.rerun()
                    
            with row_cols[4]:
                if st.button("🗑️ Delete", key=f"btn_del_{s['id']}", use_container_width=True):
                    st.session_state['confirm_delete_id'] = s['id']
                    st.rerun()
                    
            with row_cols[5]:
                if s['is_active']:
                    if st.button("✗", key=f"btn_deact_{s['id']}", help="Deactivate student"):
                        service.deactivate_student(s['id'])
                        st.rerun()
                else:
                    if st.button("✓", key=f"btn_act_{s['id']}", help="Activate student", type="primary"):
                        service.activate_student(s['id'])
                        st.rerun()

    except Exception as e:
        logger.error(f"Error rendering student directory: {e}")
        st.error(f"❌ Error loading student directory: {e}")


# ============================================================================
# View 2: Add New Student
# ============================================================================

def render_add_student_view(service: StudentService):
    """Render Add Student form with immediate database insertion, verification, and return to directory."""
    st.subheader("➕ Register New Student")
    st.markdown("Enter student information. The student will immediately be added to SQLite and available in the directory.")
    
    if st.button("⬅️ Back to Student Directory", key="btn_back_add_top", use_container_width=False):
        st.session_state['current_view'] = 'directory'
        st.rerun()
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        new_id = st.text_input("Student ID *", placeholder="e.g., 2021001", key="input_new_student_id")
        new_name = st.text_input("Full Name *", placeholder="e.g., Jane Doe", key="input_new_name")
        new_email = st.text_input("Email Address", placeholder="e.g., jane@example.com", key="input_new_email")
    with col2:
        new_dept = st.text_input("Department / Major", placeholder="e.g., Computer Science", key="input_new_dept")
        new_notes = st.text_area("Notes", placeholder="Additional student details (optional)", key="input_new_notes")
        
    st.caption("* Required fields")
    
    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save Student Profile", type="primary", use_container_width=True, key="btn_save_new_student"):
            if not new_id or not str(new_id).strip():
                st.error("❌ Student ID is required.")
                return
            if not new_name or not str(new_name).strip():
                st.error("❌ Student Name is required.")
                return
                
            clean_id = str(new_id).strip()
            clean_name = str(new_name).strip()
            clean_email = str(new_email).strip() if new_email and str(new_email).strip() else None
            clean_dept = str(new_dept).strip() if new_dept and str(new_dept).strip() else None
            clean_notes = str(new_notes).strip() if new_notes and str(new_notes).strip() else None
            
            try:
                # 1. Create and verify in database
                db_id = service.create_student(
                    student_id=clean_id,
                    name=clean_name,
                    email=clean_email,
                    department=clean_dept,
                    notes=clean_notes
                )
                
                # 2. Confirm record existence
                verified = service.get_student_by_id(db_id)
                if not verified:
                    st.error("❌ Database insertion failed verification.")
                    return
                    
                st.session_state['registry_feedback'] = (
                    'success',
                    f"✅ Student **{clean_name}** (`{clean_id}`) registered and confirmed in database!"
                )
                # Option to immediately manage face or return to directory
                st.session_state['selected_student_id'] = db_id
                st.session_state['current_view'] = 'manage_face'
                st.rerun()
                
            except DuplicateStudentError as e:
                st.error(f"❌ {e}")
            except InvalidStudentDataError as e:
                st.error(f"❌ Validation error: {e}")
            except Exception as e:
                logger.error(f"Error creating student: {e}")
                st.error(f"❌ Failed to register student: {e}")

    with col_cancel:
        if st.button("Cancel", use_container_width=True, key="btn_cancel_new_student"):
            st.session_state['current_view'] = 'directory'
            st.rerun()


# ============================================================================
# View 3: Edit Student Profile
# ============================================================================

def render_edit_student_view(service: StudentService, student_db_id: int):
    """Render Edit Profile view for a specific student."""
    student = service.get_student_by_id(student_db_id)
    if not student:
        st.error(f"Student with DB ID {student_db_id} not found.")
        if st.button("⬅️ Back to Student Directory", key="btn_back_edit_notfound"):
            st.session_state['current_view'] = 'directory'
            st.rerun()
        return

    st.subheader(f"✏️ Edit Profile: {student['name']} (`{student['student_id']}`)")
    
    if st.button("⬅️ Back to Student Directory", key="btn_back_edit_top", use_container_width=False):
        st.session_state['current_view'] = 'directory'
        st.rerun()
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        edit_id = st.text_input("Student ID *", value=student['student_id'], key="input_edit_id")
        edit_name = st.text_input("Full Name *", value=student['name'], key="input_edit_name")
        edit_email = st.text_input("Email Address", value=student['email'] or "", key="input_edit_email")
    with col2:
        edit_dept = st.text_input("Department", value=student['department'] or "", key="input_edit_dept")
        edit_notes = st.text_area("Notes", value=student['notes'] or "", key="input_edit_notes")
        
    st.caption("* Required fields")
    
    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        if st.button("💾 Save Changes", type="primary", use_container_width=True, key="btn_save_edit_student"):
            if not edit_id or not str(edit_id).strip():
                st.error("❌ Student ID is required.")
                return
            if not edit_name or not str(edit_name).strip():
                st.error("❌ Student Name is required.")
                return
                
            try:
                service.update_student(
                    student_db_id=student_db_id,
                    student_id=str(edit_id).strip(),
                    name=str(edit_name).strip(),
                    email=str(edit_email).strip() if edit_email and str(edit_email).strip() else None,
                    department=str(edit_dept).strip() if edit_dept and str(edit_dept).strip() else None,
                    notes=str(edit_notes).strip() if edit_notes and str(edit_notes).strip() else None
                )
                st.session_state['registry_feedback'] = (
                    'success',
                    f"✅ Profile updated successfully for **{edit_name}** (`{edit_id}`)!"
                )
                st.session_state['current_view'] = 'directory'
                st.rerun()
            except Exception as e:
                logger.error(f"Error updating student: {e}")
                st.error(f"❌ Update failed: {e}")
                
    with col_cancel:
        if st.button("Cancel", use_container_width=True, key="btn_cancel_edit_student"):
            st.session_state['current_view'] = 'directory'
            st.rerun()


# ============================================================================
# View 4: Manage Face Dataset
# ============================================================================

def render_manage_face_view(
    student_service: StudentService,
    enrollment_service: EnrollmentService,
    student_db_id: int
):
    """
    Render face dataset management:
    - View current enrolled samples count
    - Upload face sample image(s)
    - Continuous live camera preview with capture, freeze-frame review, approve, and retake
    - Camera stays open during preview/review/retake until user exits
    - Safe reset confirmation
    """
    student = student_service.get_student_by_id(student_db_id)
    if not student:
        st.error(f"Student #{student_db_id} not found.")
        if st.button("⬅️ Back to Student Directory"):
            st.session_state['current_view'] = 'directory'
            st.rerun()
        return

    st.subheader(f"📸 Manage Face Dataset: {student['name']} (`{student['student_id']}`)")
    
    if st.button("⬅️ Back to Student Directory", key="btn_back_face_top", use_container_width=False):
        release_registration_camera()
        st.session_state['confirm_reset_face_id'] = None
        st.session_state['current_view'] = 'directory'
        st.rerun()
        
    st.markdown("---")
    
    sample_count = enrollment_service.get_student_samples_count(student_db_id)
    
    # Status Card & Reset Button
    col_card1, col_card2 = st.columns([3, 1])
    with col_card1:
        st.markdown(f"**Managing Face Data For:** `{student['student_id']}` — **{student['name']}** ({student['department'] or 'No Department'})")
        if sample_count > 0:
            st.success(f"✅ **{sample_count} Active Face Sample(s)** enrolled in recognition gallery.")
        else:
            st.warning("⚠️ **0 Face Samples.** This student cannot be recognized until at least one face sample is enrolled.")
            
    with col_card2:
        if sample_count > 0 and st.session_state.get('confirm_reset_face_id') != student_db_id:
            if st.button("🗑️ Reset Face Dataset", help="Remove all face samples and re-enroll from scratch", key="btn_init_reset_dataset"):
                st.session_state['confirm_reset_face_id'] = student_db_id
                st.rerun()

    # Reset Confirmation Warning Box
    if st.session_state.get('confirm_reset_face_id') == student_db_id:
        st.warning(f"⚠️ **Permanently delete all {sample_count} enrolled face samples for {student['name']}?** This action cannot be undone.")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if st.button("❌ Cancel Reset", key="btn_cancel_reset_samples", use_container_width=True):
                st.session_state['confirm_reset_face_id'] = None
                st.rerun()
        with col_res2:
            if st.button("🗑️ Yes, Delete All Samples", type="primary", key="btn_confirm_reset_samples", use_container_width=True):
                release_registration_camera()
                enrollment_service.clear_student_face_dataset(student_db_id)
                st.session_state['confirm_reset_face_id'] = None
                st.success(f"Face dataset reset successfully for **{student['name']}**!")
                st.rerun()
                
    st.markdown("---")
    
    # Enrollment Source Choice
    st.markdown("### 📥 Add / Update Face Samples")
    enroll_method = st.radio(
        "Choose Enrollment Method",
        ["📁 Upload Image(s)", "📹 Take Photo with Live Camera"],
        horizontal=True,
        key="radio_face_enroll_method"
    )
    
    if enroll_method == "📁 Upload Image(s)":
        if st.session_state.get('reg_cam_active'):
            release_registration_camera()
            
        uploaded_files = st.file_uploader(
            "Upload Clear Portrait Image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or multiple images with clearly visible face",
            key="file_uploader_samples"
        )
        
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button("🔍 Process & Enroll Uploaded Faces", type="primary", use_container_width=True, key="btn_process_uploaded"):
                with st.spinner("Analyzing and validating face samples..."):
                    success_count = 0
                    errors = []
                    
                    for f in uploaded_files:
                        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            errors.append(f"{f.name}: Invalid image format")
                            continue
                            
                        try:
                            enrollment_service.add_face_sample(
                                student_db_id=student_db_id,
                                face_image=img,
                                enrollment_method="upload"
                            )
                            success_count += 1
                        except InvalidFaceImageError as e:
                            errors.append(f"{f.name}: {e}")
                        except Exception as e:
                            errors.append(f"{f.name}: {e}")
                            
                    if success_count > 0:
                        st.success(f"✅ Successfully enrolled **{success_count}** face sample(s) for **{student['name']}**!")
                        
                    if errors:
                        for err in errors:
                            st.warning(f"⚠️ {err}")
                            
                    if success_count > 0:
                        st.rerun()
                        
    elif enroll_method == "📹 Take Photo with Live Camera":
        st.markdown("#### Live Camera Capture")
        
        # State A: Camera is OFF
        if not st.session_state.get('reg_cam_active', False):
            st.info("💡 Click **Take Photo** to start live preview, position face, and capture.")
            if st.button("📸 Take Photo", type="primary", use_container_width=True, key="btn_start_enroll_camera"):
                cam = CameraManager(camera_index=0)
                if cam.open():
                    st.session_state['reg_cam_manager'] = cam
                    st.session_state['reg_cam_active'] = True
                    st.session_state['captured_photo_preview'] = None
                    st.rerun()
                else:
                    st.error("❌ Failed to open camera hardware. Ensure webcam is connected and not in use by another app.")
                    st.session_state['reg_cam_manager'] = None
                    st.session_state['reg_cam_active'] = False
        else:
            cam = st.session_state.get('reg_cam_manager')
            if cam is None or not cam.is_open():
                st.warning("Camera is not active.")
                release_registration_camera()
                if st.button("🔄 Open Camera Again", use_container_width=True, key="btn_reopen_cam"):
                    st.rerun()
                return

            # State B: A photo is captured and awaiting user approval/retake (CAMERA STAYS OPEN)
            if st.session_state.get('captured_photo_preview') is not None:
                captured_frame = st.session_state['captured_photo_preview']
                
                st.markdown("##### 📸 Photo Preview — Check your picture")
                mirrored_captured = cv2.flip(captured_frame, 1)
                st.image(
                    cv2.cvtColor(mirrored_captured, cv2.COLOR_BGR2RGB),
                    caption="Captured Frame Preview",
                    use_container_width=True
                )
                
                col_app, col_ret = st.columns(2)
                with col_app:
                    if st.button("✓ Approve / Use Photo", type="primary", use_container_width=True, key="btn_approve_captured"):
                        with st.spinner("Validating face and saving embedding..."):
                            try:
                                # Validate and enroll the unmirrored original frame
                                enrollment_service.add_face_sample(
                                    student_db_id=student_db_id,
                                    face_image=captured_frame,
                                    enrollment_method="camera"
                                )
                                st.session_state['captured_photo_preview'] = None
                                st.success(f"✅ Face sample approved and enrolled for **{student['name']}**!")
                                st.rerun()
                            except InvalidFaceImageError as e:
                                st.error(f"❌ Validation failed: {e}")
                            except Exception as e:
                                st.error(f"❌ Error saving sample: {e}")
                                
                with col_ret:
                    if st.button("↻ Retake", use_container_width=True, key="btn_retake_captured"):
                        # Discard frame and return immediately to live streaming without reopening camera
                        st.session_state['captured_photo_preview'] = None
                        st.rerun()
                        
                st.markdown("---")
                if st.button("⏹️ Stop / Close Camera", use_container_width=True, key="btn_stop_cam_on_preview"):
                    release_registration_camera()
                    st.rerun()
                return

            # State C: Live Camera Streaming Preview
            col_cap, col_cancel = st.columns(2)
            with col_cap:
                capture_clicked = st.button("📸 Capture Photo", type="primary", use_container_width=True, key="btn_capture_frame_now")
            with col_cancel:
                cancel_clicked = st.button("⏹️ Stop / Close Camera", use_container_width=True, key="btn_stop_cam_streaming")

            if cancel_clicked:
                release_registration_camera()
                st.rerun()

            if capture_clicked:
                success, frame = cam.read_frame()
                if success and frame is not None:
                    # Freeze captured frame in state for approval without releasing the camera
                    st.session_state['captured_photo_preview'] = frame.copy()
                    st.rerun()
                else:
                    st.error("❌ Failed to grab frame from camera.")
                return

            # Live preview placeholder
            video_placeholder = st.empty()
            
            frame_count = 0
            max_frames_per_run = 150  # ~5 seconds of smooth 30fps streaming per cycle
            
            try:
                while frame_count < max_frames_per_run:
                    if not st.session_state.get('reg_cam_active', False) or st.session_state.get('captured_photo_preview') is not None:
                        break
                        
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        video_placeholder.warning("⚠️ Waiting for camera frame...")
                        time.sleep(0.05)
                        continue
                        
                    # Smooth horizontally mirrored preview for natural user experience
                    mirrored_preview = cv2.flip(frame, 1)
                    video_placeholder.image(
                        cv2.cvtColor(mirrored_preview, cv2.COLOR_BGR2RGB),
                        caption="Live Camera Feed — Center face and click 'Capture Photo' above",
                        use_container_width=True
                    )
                    
                    time.sleep(0.033)
                    frame_count += 1
                    
                # Re-trigger run to keep live stream active seamlessly
                if st.session_state.get('reg_cam_active', False) and st.session_state.get('captured_photo_preview') is None:
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Error in enrollment camera loop: {e}")
                video_placeholder.error(f"❌ Camera error: {e}")
                release_registration_camera()


# Backward compatibility helpers for tests
def render_student_list_tab(service: StudentService, enrollment_service: EnrollmentService):
    render_directory_view(service, enrollment_service)

def render_add_student_tab(student_service: StudentService, enrollment_service: EnrollmentService):
    render_add_student_view(student_service)

def render_face_dataset_tab(student_service: StudentService, enrollment_service: EnrollmentService):
    sel_id = st.session_state.get('selected_student_id')
    if sel_id is not None:
        render_manage_face_view(student_service, enrollment_service, sel_id)
    else:
        all_s = student_service.get_all_students(active_only=True)
        if all_s:
            render_manage_face_view(student_service, enrollment_service, all_s[0]['id'])
        else:
            st.info("No students registered.")

def render_edit_student_tab(service: StudentService):
    sel_id = st.session_state.get('selected_student_id')
    if sel_id is not None:
        render_edit_student_view(service, sel_id)
    else:
        all_s = service.get_all_students(active_only=False)
        if all_s:
            render_edit_student_view(service, all_s[0]['id'])
        else:
            st.info("No students registered.")

def format_datetime(dt_string: Optional[str]) -> str:
    """Format a datetime string for display."""
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dt_string
