"""
Student Registry UI Page

Provides a Streamlit interface for managing students through StudentService.
Uses StudentService as the application boundary - no direct database access.
"""

import streamlit as st
from datetime import datetime
from typing import Optional

from src.services.student_service import (
    StudentService,
    InvalidStudentDataError,
    DuplicateStudentError,
    StudentNotFoundError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_student_registry():
    """
    Main function to render the Student Registry page.
    """
    st.title("👥 Student Registry")
    st.markdown("Manage student information and enrollment records.")
    st.markdown("---")
    
    # Initialize service
    service = StudentService()
    
    # Initialize session state for UI management
    if 'edit_student_id' not in st.session_state:
        st.session_state.edit_student_id = None
    if 'refresh_trigger' not in st.session_state:
        st.session_state.refresh_trigger = 0
    
    # Render main sections
    render_search_and_filters()
    render_student_list(service)
    st.markdown("---")
    render_add_student_form(service)
    st.markdown("---")
    render_edit_student_section(service)


def render_search_and_filters():
    """Render search and filter controls."""
    st.subheader("🔍 Search and Filter")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by Student ID or Name",
            placeholder="Enter student ID or name...",
            key="search_query"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All Students", "Active Students", "Inactive Students"],
            key="status_filter"
        )


def render_student_list(service: StudentService):
    """
    Render the list of students with search and filter applied.
    
    Args:
        service: StudentService instance
    """
    st.subheader("📋 Student List")
    
    try:
        # Get filter settings
        status_filter = st.session_state.get('status_filter', 'All Students')
        search_query = st.session_state.get('search_query', '').strip().lower()
        
        # Retrieve students based on filter
        if status_filter == "Active Students":
            students = service.get_all_students(active_only=True)
        elif status_filter == "Inactive Students":
            # Get all students and filter for inactive
            all_students = service.get_all_students(active_only=False)
            students = [s for s in all_students if s['is_active'] == 0]
        else:
            students = service.get_all_students(active_only=False)
        
        # Apply search filter
        if search_query:
            students = [
                s for s in students
                if search_query in s['student_id'].lower()
                or search_query in s['name'].lower()
            ]
        
        # Display count
        st.caption(f"Found {len(students)} student(s)")
        
        if not students:
            st.info("No students found matching the current filters.")
            return
        
        # Display students in a table
        display_students = []
        for student in students:
            display_students.append({
                'Student ID': student['student_id'],
                'Name': student['name'],
                'Email': student['email'] or '—',
                'Department': student['department'] or '—',
                'Status': 'Active' if student['is_active'] else 'Inactive',
                'Enrolled At': format_datetime(student['enrolled_at']),
                'DB ID': student['id']  # Hidden but available for actions
            })
        
        # Display table
        st.dataframe(
            display_students,
            use_container_width=True,
            hide_index=True,
            column_config={
                'DB ID': st.column_config.NumberColumn(
                    'DB ID',
                    help='Internal database ID',
                    width='small'
                )
            }
        )
        
        # Display action buttons for each student
        st.markdown("#### Student Actions")
        
        cols = st.columns([2, 2, 2, 2])
        
        with cols[0]:
            st.caption("Select Student to Edit")
        with cols[1]:
            st.caption("Activate")
        with cols[2]:
            st.caption("Deactivate")
        with cols[3]:
            st.caption("")
        
        for student in students:
            cols = st.columns([2, 2, 2, 2])
            
            with cols[0]:
                if st.button(
                    f"Edit {student['student_id']}",
                    key=f"edit_{student['id']}",
                    use_container_width=True
                ):
                    st.session_state.edit_student_id = student['id']
                    st.rerun()
            
            with cols[1]:
                if student['is_active'] == 0:
                    if st.button(
                        f"✓ Activate",
                        key=f"activate_{student['id']}",
                        use_container_width=True,
                        type="primary"
                    ):
                        handle_activate_student(service, student['id'])
            
            with cols[2]:
                if student['is_active'] == 1:
                    if st.button(
                        f"✗ Deactivate",
                        key=f"deactivate_{student['id']}",
                        use_container_width=True
                    ):
                        handle_deactivate_student(service, student['id'])
            
            with cols[3]:
                st.caption(f"{student['name']}")
        
    except Exception as e:
        logger.error(f"Error displaying student list: {e}")
        st.error("An error occurred while loading students. Please try again.")


def render_add_student_form(service: StudentService):
    """
    Render the form for adding a new student.
    
    Args:
        service: StudentService instance
    """
    st.subheader("➕ Add New Student")
    
    with st.form("add_student_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input(
                "Student ID *",
                placeholder="e.g., 2021001",
                help="Unique identifier for the student"
            )
            name = st.text_input(
                "Full Name *",
                placeholder="e.g., John Doe",
                help="Student's full name"
            )
            email = st.text_input(
                "Email",
                placeholder="e.g., john@example.com",
                help="Student's email address (optional)"
            )
        
        with col2:
            department = st.text_input(
                "Department",
                placeholder="e.g., Computer Science",
                help="Student's department (optional)"
            )
            notes = st.text_area(
                "Notes",
                placeholder="Additional notes about the student (optional)",
                help="Any additional information"
            )
        
        st.caption("* Required fields")
        
        submitted = st.form_submit_button(
            "Add Student",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            handle_add_student(
                service,
                student_id,
                name,
                email,
                department,
                notes
            )


def render_edit_student_section(service: StudentService):
    """
    Render the section for editing an existing student.
    
    Args:
        service: StudentService instance
    """
    st.subheader("✏️ Edit Student")
    
    edit_student_id = st.session_state.get('edit_student_id')
    
    if not edit_student_id:
        st.info("Select a student from the list above to edit their information.")
        return
    
    try:
        # Retrieve the student
        student = service.get_student_by_id(edit_student_id)
        
        if not student:
            st.error("Student not found. They may have been deleted.")
            st.session_state.edit_student_id = None
            return
        
        st.info(f"Editing: **{student['name']}** (ID: {student['student_id']})")
        
        with st.form("edit_student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_student_id = st.text_input(
                    "Student ID *",
                    value=student['student_id'],
                    help="Unique identifier for the student"
                )
                new_name = st.text_input(
                    "Full Name *",
                    value=student['name'],
                    help="Student's full name"
                )
                new_email = st.text_input(
                    "Email",
                    value=student['email'] or "",
                    help="Student's email address (optional)"
                )
            
            with col2:
                new_department = st.text_input(
                    "Department",
                    value=student['department'] or "",
                    help="Student's department (optional)"
                )
                new_notes = st.text_area(
                    "Notes",
                    value=student['notes'] or "",
                    help="Any additional information"
                )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                save_button = st.form_submit_button(
                    "Save Changes",
                    type="primary",
                    use_container_width=True
                )
            
            with col_b:
                cancel_button = st.form_submit_button(
                    "Cancel",
                    use_container_width=True
                )
            
            if cancel_button:
                st.session_state.edit_student_id = None
                st.rerun()
            
            if save_button:
                handle_update_student(
                    service,
                    edit_student_id,
                    new_student_id,
                    new_name,
                    new_email,
                    new_department,
                    new_notes
                )
    
    except Exception as e:
        logger.error(f"Error in edit student section: {e}")
        st.error("An error occurred while loading student data.")


# ============================================================================
# Action Handlers
# ============================================================================

def handle_add_student(
    service: StudentService,
    student_id: str,
    name: str,
    email: str,
    department: str,
    notes: str
):
    """
    Handle adding a new student.
    
    Args:
        service: StudentService instance
        student_id: Student ID
        name: Student name
        email: Email address
        department: Department
        notes: Additional notes
    """
    try:
        # Validate required fields
        if not student_id or not student_id.strip():
            st.error("❌ Student ID is required.")
            return
        
        if not name or not name.strip():
            st.error("❌ Name is required.")
            return
        
        # Create student through service (service handles validation)
        db_id = service.create_student(
            student_id=student_id,
            name=name,
            email=email if email else None,
            department=department if department else None,
            notes=notes if notes else None
        )
        
        st.success(f"✅ Student **{name}** (ID: {student_id}) added successfully!")
        logger.info(f"Student added via UI: {student_id} (DB ID: {db_id})")
        
        # Trigger refresh
        st.session_state.refresh_trigger += 1
        st.rerun()
        
    except DuplicateStudentError as e:
        st.error(f"❌ {str(e)}")
        logger.warning(f"Duplicate student attempt: {student_id}")
    
    except InvalidStudentDataError as e:
        st.error(f"❌ Invalid data: {str(e)}")
        logger.warning(f"Invalid student data: {e}")
    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please try again.")
        logger.error(f"Unexpected error adding student: {e}")


def handle_update_student(
    service: StudentService,
    student_db_id: int,
    student_id: str,
    name: str,
    email: str,
    department: str,
    notes: str
):
    """
    Handle updating an existing student.
    
    Args:
        service: StudentService instance
        student_db_id: Database ID of the student
        student_id: New student ID
        name: New name
        email: New email
        department: New department
        notes: New notes
    """
    try:
        # Validate required fields
        if not student_id or not student_id.strip():
            st.error("❌ Student ID is required.")
            return
        
        if not name or not name.strip():
            st.error("❌ Name is required.")
            return
        
        # Update through service
        service.update_student(
            student_db_id=student_db_id,
            student_id=student_id,
            name=name,
            email=email if email else None,
            department=department if department else None,
            notes=notes if notes else None
        )
        
        st.success(f"✅ Student **{name}** updated successfully!")
        logger.info(f"Student updated via UI: DB ID {student_db_id}")
        
        # Clear edit state and refresh
        st.session_state.edit_student_id = None
        st.session_state.refresh_trigger += 1
        st.rerun()
        
    except DuplicateStudentError as e:
        st.error(f"❌ {str(e)}")
        logger.warning(f"Duplicate student ID in update: {student_id}")
    
    except InvalidStudentDataError as e:
        st.error(f"❌ Invalid data: {str(e)}")
        logger.warning(f"Invalid student data in update: {e}")
    
    except StudentNotFoundError as e:
        st.error(f"❌ {str(e)}")
        logger.warning(f"Student not found for update: {student_db_id}")
    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please try again.")
        logger.error(f"Unexpected error updating student: {e}")


def handle_activate_student(service: StudentService, student_db_id: int):
    """
    Handle activating a student.
    
    Args:
        service: StudentService instance
        student_db_id: Database ID of the student
    """
    try:
        service.activate_student(student_db_id)
        st.success("✅ Student activated successfully!")
        logger.info(f"Student activated via UI: DB ID {student_db_id}")
        
        # Trigger refresh
        st.session_state.refresh_trigger += 1
        st.rerun()
        
    except StudentNotFoundError as e:
        st.error(f"❌ {str(e)}")
        logger.warning(f"Student not found for activation: {student_db_id}")
    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please try again.")
        logger.error(f"Unexpected error activating student: {e}")


def handle_deactivate_student(service: StudentService, student_db_id: int):
    """
    Handle deactivating a student.
    
    Args:
        service: StudentService instance
        student_db_id: Database ID of the student
    """
    try:
        service.deactivate_student(student_db_id)
        st.success("✅ Student deactivated successfully!")
        logger.info(f"Student deactivated via UI: DB ID {student_db_id}")
        
        # Trigger refresh
        st.session_state.refresh_trigger += 1
        st.rerun()
        
    except StudentNotFoundError as e:
        st.error(f"❌ {str(e)}")
        logger.warning(f"Student not found for deactivation: {student_db_id}")
    
    except Exception as e:
        st.error("❌ An unexpected error occurred. Please try again.")
        logger.error(f"Unexpected error deactivating student: {e}")


# ============================================================================
# Helper Functions
# ============================================================================

def format_datetime(dt_string: Optional[str]) -> str:
    """
    Format a datetime string for display.
    
    Args:
        dt_string: Datetime string from database
    
    Returns:
        Formatted datetime string or '—' if None
    """
    if not dt_string:
        return '—'
    
    try:
        # Parse datetime from database format
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dt_string


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # For testing the page standalone
    st.set_page_config(
        page_title="Student Registry",
        page_icon="👥",
        layout="wide"
    )
    render_student_registry()
