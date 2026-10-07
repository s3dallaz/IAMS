"""
Student Registry and Face Dataset Management UI Page.

Provides a Streamlit interface for:
1. Managing student metadata (search, list, add, edit, activate/deactivate) through StudentService.
2. Managing student face datasets (capture / upload face samples, validate detectability & quality,
   generate embeddings, and update face data when appearance changes) through EnrollmentService.
Uses Service layers as application boundary - no direct database access.
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
    StudentNotFoundError
)
from src.services.enrollment_service import (
    EnrollmentService,
    EnrollmentServiceError,
    InvalidFaceImageError
)
from src.computer_vision.camera_manager import CameraManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_student_registry():
    """Main function to render the Student Registry and Face Dataset Management page."""
    st.title("👥 Student Registry & Face Dataset Management")
    st.caption("Phase 4E: Student Profiles, Multi-Sample Face Datasets & Appearance Updates")
    st.markdown("Register students, manage profiles, and enroll/update multi-sample face datasets for recognition.")
    st.markdown("---")
    
    # Initialize services
    student_service = StudentService()
    enrollment_service = EnrollmentService()
    
    # Initialize session state keys
    if 'edit_student_id' not in st.session_state:
        st.session_state['edit_student_id'] = None
    if 'manage_face_student_id' not in st.session_state:
        st.session_state['manage_face_student_id'] = None
    if 'reg_cam_active' not in st.session_state:
        st.session_state['reg_cam_active'] = False
    if 'reg_cam_manager' not in st.session_state:
        st.session_state['reg_cam_manager'] = None
        
    # If a specific workflow was requested (e.g. Manage Face or Edit Profile from Quick Actions),
    # render that workflow directly with a clean Back navigation, because st.tabs() cannot be programmatically switched.
    if st.session_state.get('manage_face_student_id') is not None:
        render_face_dataset_tab(student_service, enrollment_service)
        st.markdown("---")
        if st.button("⬅️ Back to Student Directory", key="btn_back_from_face", use_container_width=True):
            st.session_state['manage_face_student_id'] = None
            st.rerun()
        return

    if st.session_state.get('edit_student_id') is not None:
        render_edit_student_tab(student_service)
        st.markdown("---")
        if st.button("⬅️ Back to Student Directory", key="btn_back_from_edit", use_container_width=True):
            st.session_state['edit_student_id'] = None
            st.rerun()
        return

    # Top Level Navigation Tabs
    tab_list, tab_add, tab_face, tab_edit = st.tabs([
        "📋 Student Directory",
        "➕ Add New Student",
        "📸 Manage Face Dataset",
        "✏️ Edit Profile"
    ])
    
    with tab_list:
        render_student_list_tab(student_service, enrollment_service)
        
    with tab_add:
        render_add_student_tab(student_service, enrollment_service)
        
    with tab_face:
        render_face_dataset_tab(student_service, enrollment_service)
        
    with tab_edit:
        render_edit_student_tab(student_service)


# ============================================================================
# Tab 1: Student Directory
# ============================================================================

def render_student_list_tab(service: StudentService, enrollment_service: EnrollmentService):
    """Render the student list with search, status filters, and dataset actions."""
    st.subheader("📋 Student Directory")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input(
            "Search Students",
            placeholder="Search by ID or name...",
            key="dir_search_query"
        )
    with col2:
        status_filter = st.selectbox(
            "Filter Status",
            options=["All Students", "Active Students", "Inactive Students"],
            key="dir_status_filter"
        )
        
    try:
        # Retrieve students
        if status_filter == "Active Students":
            students = service.get_all_students(active_only=True)
        elif status_filter == "Inactive Students":
            all_students = service.get_all_students(active_only=False)
            students = [s for s in all_students if s['is_active'] == 0]
        else:
            students = service.get_all_students(active_only=False)
            
        if search_query:
            q = search_query.strip().lower()
            students = [s for s in students if q in s['student_id'].lower() or q in s['name'].lower()]
            
        st.caption(f"Found {len(students)} student(s)")
        
        if not students:
            st.info("No students found matching current criteria.")
            return
            
        display_data = []
        for s in students:
            samples_count = enrollment_service.get_student_samples_count(s['id'])
            face_status = f"✅ {samples_count} sample(s)" if samples_count > 0 else "⚠️ No face data"
            
            display_data.append({
                'ID': s['student_id'],
                'Name': s['name'],
                'Department': s['department'] or '—',
                'Status': "Active" if s['is_active'] else "Inactive",
                'Face Dataset': face_status,
                'Enrolled At': format_datetime(s['enrolled_at']),
                'DB ID': s['id']
            })
            
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                'DB ID': st.column_config.NumberColumn('DB ID', width='small')
            }
        )
        
        st.markdown("#### ⚡ Quick Actions")
        action_header = st.columns([1, 2, 2, 2, 1])
        action_header[0].caption("ID")
        action_header[1].caption("Student Name")
        action_header[2].caption("Face Dataset")
        action_header[3].caption("Profile")
        action_header[4].caption("Status")
        
        for s in students:
            cols = st.columns([1, 2, 2, 2, 1])
            cols[0].write(f"`{s['student_id']}`")
            cols[1].write(f"**{s['name']}**")
            
            with cols[2]:
                if st.button("📸 Manage Face", key=f"face_btn_{s['id']}", use_container_width=True):
                    st.session_state['manage_face_student_id'] = s['id']
                    st.rerun()
                    
            with cols[3]:
                if st.button("✏️ Edit Profile", key=f"edit_btn_{s['id']}", use_container_width=True):
                    st.session_state['edit_student_id'] = s['id']
                    st.rerun()
                    
            with cols[4]:
                if s['is_active']:
                    if st.button("✗", key=f"deact_{s['id']}", help="Deactivate student"):
                        service.deactivate_student(s['id'])
                        st.rerun()
                else:
                    if st.button("✓", key=f"act_{s['id']}", help="Activate student", type="primary"):
                        service.activate_student(s['id'])
                        st.rerun()
                        
    except Exception as e:
        logger.error(f"Error rendering student directory: {e}")
        st.error(f"Error loading students: {e}")


# ============================================================================
# Tab 2: Add New Student
# ============================================================================

def render_add_student_tab(student_service: StudentService, enrollment_service: EnrollmentService):
    """Render the form for adding a student profile."""
    st.subheader("➕ Register New Student Profile")
    st.markdown("Enter student information. You will be able to enroll face samples immediately after saving.")
    
    with st.form("add_student_profile_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input(
                "Student ID *",
                placeholder="e.g., 2021001",
                help="Unique student identification number"
            )
            name = st.text_input(
                "Full Name *",
                placeholder="e.g., Jane Doe",
                help="Full student name"
            )
            email = st.text_input(
                "Email Address",
                placeholder="e.g., jane@example.com",
                help="Student email address (optional)"
            )
            
        with col2:
            department = st.text_input(
                "Department / Major",
                placeholder="e.g., Computer Science",
                help="Academic department or major (optional)"
            )
            notes = st.text_area(
                "Notes",
                placeholder="Additional student details (optional)"
            )
            
        st.caption("* Required fields")
        
        submitted = st.form_submit_button("Save Student Profile", type="primary", use_container_width=True)
        
        if submitted:
            try:
                if not student_id or not student_id.strip():
                    st.error("❌ Student ID is required.")
                    return
                if not name or not name.strip():
                    st.error("❌ Student Name is required.")
                    return
                    
                db_id = student_service.create_student(
                    student_id=student_id,
                    name=name,
                    email=email if email else None,
                    department=department if department else None,
                    notes=notes if notes else None
                )
                
                st.success(f"✅ Student **{name}** (`{student_id}`) registered successfully!")
                st.session_state['manage_face_student_id'] = db_id
                st.info("👉 Switched to **Manage Face Dataset** tab to enroll face samples.")
                st.rerun()
                
            except DuplicateStudentError as e:
                st.error(f"❌ {e}")
            except InvalidStudentDataError as e:
                st.error(f"❌ Validation error: {e}")
            except Exception as e:
                st.error(f"❌ Failed to register student: {e}")


# ============================================================================
# Tab 3: Manage Face Dataset / Appearance Updates
# ============================================================================

def render_face_dataset_tab(student_service: StudentService, enrollment_service: EnrollmentService):
    """
    Render face dataset management:
    - View enrolled samples
    - Add multiple face samples via Upload or Camera
    - Update dataset when appearance changes (glasses, beard, hairstyle)
    - Validate face detectability & quality before adding
    """
    st.subheader("📸 Manage Student Face Dataset")
    st.markdown("""
    Enrolling multiple face samples ensures high recognition accuracy under varying conditions 
    (different angles, glasses, lighting, beards, hairstyles, etc.).
    """)
    
    students = student_service.get_all_students(active_only=True)
    if not students:
        st.info("No active students registered yet. Please add a student first.")
        return
        
    student_options = {s['id']: f"{s['name']} (ID: {s['student_id']})" for s in students}
    
    target_id = st.session_state.get('manage_face_student_id')
    default_idx = 0
    if target_id in student_options:
        default_idx = list(student_options.keys()).index(target_id)
        
    selected_db_id = st.selectbox(
        "Select Student to Manage Face Data",
        options=list(student_options.keys()),
        format_func=lambda x: student_options[x],
        index=default_idx,
        key="face_dataset_student_selector"
    )
    st.session_state['manage_face_student_id'] = selected_db_id
    
    student = student_service.get_student_by_id(selected_db_id)
    if not student:
        return
        
    # Student Face Status Card
    sample_count = enrollment_service.get_student_samples_count(selected_db_id)
    
    col_card1, col_card2 = st.columns([3, 1])
    with col_card1:
        st.markdown(f"**Managing Face Data For:** `{student['student_id']}` — **{student['name']}**")
        if sample_count > 0:
            st.success(f"✅ **{sample_count} Active Face Sample(s)** enrolled in gallery.")
        else:
            st.warning("⚠️ **0 Face Samples.** This student cannot be recognized until face samples are enrolled.")
            
    with col_card2:
        if sample_count > 0:
            if st.button("🗑️ Reset Face Dataset", help="Remove all face samples and re-enroll from scratch"):
                enrollment_service.clear_student_face_dataset(selected_db_id)
                st.success("Face dataset reset successfully!")
                st.rerun()
                
    st.markdown("---")
    
    # Face Enrollment / Update Methods
    st.markdown("### 📥 Add / Update Face Samples")
    enroll_method = st.radio(
        "Choose Enrollment Source",
        ["📁 Upload Face Image(s)", "📹 Capture via Camera"],
        horizontal=True,
        key="face_enroll_method"
    )
    
    if enroll_method == "📁 Upload Face Image(s)":
        uploaded_files = st.file_uploader(
            "Upload Clear Portrait Image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or multiple images with clearly visible face"
        )
        
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button("🔍 Process & Enroll Uploaded Faces", type="primary", use_container_width=True):
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
                                student_db_id=selected_db_id,
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
                        
    elif enroll_method == "📹 Capture via Camera":
        st.markdown("#### Live Face Capture")
        col_cam_ctrl1, col_cam_ctrl2 = st.columns(2)
        
        with col_cam_ctrl1:
            if not st.session_state['reg_cam_active']:
                if st.button("📹 Open Enrollment Camera", type="primary", use_container_width=True):
                    if st.session_state['reg_cam_manager'] is None:
                        st.session_state['reg_cam_manager'] = CameraManager(camera_index=0)
                    if st.session_state['reg_cam_manager'].open():
                        st.session_state['reg_cam_active'] = True
                        st.rerun()
                    else:
                        st.error("Failed to open camera.")
                        st.session_state['reg_cam_manager'] = None
                        
        with col_cam_ctrl2:
            if st.session_state['reg_cam_active']:
                if st.button("⏹️ Close Camera", use_container_width=True):
                    st.session_state['reg_cam_active'] = False
                    if st.session_state['reg_cam_manager'] is not None:
                        st.session_state['reg_cam_manager'].release()
                        st.session_state['reg_cam_manager'] = None
                    st.rerun()
                    
        if st.session_state['reg_cam_active'] and st.session_state['reg_cam_manager'] is not None:
            cam = st.session_state['reg_cam_manager']
            success, frame = cam.read_frame()
            
            if success and frame is not None:
                # Mirrored display for preview
                mirrored_preview = cv2.flip(frame, 1)
                st.image(
                    cv2.cvtColor(mirrored_preview, cv2.COLOR_BGR2RGB),
                    caption="Live Preview — Click Capture Sample",
                    width=480
                )
                
                if st.button("📸 Capture & Enroll This Sample", type="primary", use_container_width=True):
                    with st.spinner("Validating and encoding captured face..."):
                        try:
                            enrollment_service.add_face_sample(
                                student_db_id=selected_db_id,
                                face_image=frame,  # Unmirrored original frame used for embedding
                                enrollment_method="camera"
                            )
                            st.success(f"✅ Face sample captured and enrolled for **{student['name']}**!")
                            time.sleep(1)
                            st.rerun()
                        except InvalidFaceImageError as e:
                            st.error(f"❌ Capture rejected: {e}")
                        except Exception as e:
                            st.error(f"❌ Error enrolling sample: {e}")
            else:
                st.warning("Waiting for camera frame...")


# ============================================================================
# Tab 4: Edit Student Profile
# ============================================================================

def render_edit_student_tab(service: StudentService):
    """Render profile edit form for existing student."""
    st.subheader("✏️ Edit Student Profile Information")
    
    edit_id = st.session_state.get('edit_student_id')
    students = service.get_all_students(active_only=False)
    
    if not students:
        st.info("No students registered.")
        return
        
    student_map = {s['id']: f"{s['name']} (`{s['student_id']}`)" for s in students}
    default_idx = 0
    if edit_id in student_map:
        default_idx = list(student_map.keys()).index(edit_id)
        
    selected_id = st.selectbox(
        "Select Student to Edit",
        options=list(student_map.keys()),
        format_func=lambda x: student_map[x],
        index=default_idx,
        key="edit_student_select"
    )
    st.session_state['edit_student_id'] = selected_id
    
    student = service.get_student_by_id(selected_id)
    if not student:
        return
        
    with st.form("edit_student_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_student_id = st.text_input("Student ID *", value=student['student_id'])
            new_name = st.text_input("Full Name *", value=student['name'])
            new_email = st.text_input("Email", value=student['email'] or "")
        with col2:
            new_department = st.text_input("Department", value=student['department'] or "")
            new_notes = st.text_area("Notes", value=student['notes'] or "")
            
        saved = st.form_submit_button("Save Profile Changes", type="primary", use_container_width=True)
        if saved:
            try:
                service.update_student(
                    student_db_id=selected_id,
                    student_id=new_student_id,
                    name=new_name,
                    email=new_email if new_email else None,
                    department=new_department if new_department else None,
                    notes=new_notes if new_notes else None
                )
                st.success("✅ Profile updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Update failed: {e}")


def format_datetime(dt_string: Optional[str]) -> str:
    """Format a datetime string for display."""
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dt_string
