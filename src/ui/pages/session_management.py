"""
Session Management UI Page for the Intelligent Attendance Monitoring System.

Phase 4D: Attendance UI & Management
Provides Streamlit interface for managing sessions through SessionService and AttendanceService.
Allows viewing sessions, creating new sessions, activating scheduled sessions,
closing active sessions, and viewing session summary/attendance records.
Uses Service layer as the application boundary - no direct database/SQL queries.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.services.session_service import (
    SessionService,
    SessionStatus,
    InvalidSessionDataError,
    SessionNotFoundError,
    InvalidSessionStateError
)
from src.services.attendance_service import (
    AttendanceService,
    AttendanceStatus,
    InvalidAttendanceDataError
)
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_session_management():
    """Main function to render the Session Management page."""
    st.title("📅 Attendance Session Management")
    st.caption("Phase 4D: Session Lifecycle & Attendance Management")
    st.markdown("Create, activate, close, and monitor attendance sessions.")
    st.markdown("---")
    
    # Initialize services
    session_service = SessionService()
    attendance_service = AttendanceService()
    student_service = StudentService()
    
    # Session state for selected session view
    if 'selected_session_id' not in st.session_state:
        st.session_state['selected_session_id'] = None
    if 'session_filter_status' not in st.session_state:
        st.session_state['session_filter_status'] = "All"
    
    # Top Action / Navigation Tabs
    tab_list, tab_create, tab_details = st.tabs([
        "📋 Sessions List",
        "➕ Create New Session",
        "🔍 Session Details & Records"
    ])
    
    with tab_list:
        render_sessions_list_tab(session_service, attendance_service)
    
    with tab_create:
        render_create_session_tab(session_service)
        
    with tab_details:
        render_session_details_tab(session_service, attendance_service, student_service)


def render_sessions_list_tab(
    session_service: SessionService,
    attendance_service: AttendanceService
):
    """Render the list of sessions with filter and lifecycle action controls."""
    st.subheader("📋 Existing Sessions")
    
    col_filter, col_refresh = st.columns([3, 1])
    
    with col_filter:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", SessionStatus.ACTIVE, SessionStatus.SCHEDULED, SessionStatus.CLOSED],
            format_func=lambda x: "All Sessions" if x == "All" else f"{x.capitalize()} Sessions",
            key="session_status_filter"
        )
    
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Sessions", width="stretch"):
            st.rerun()
            
    try:
        filter_val = None if status_filter == "All" else status_filter
        sessions = session_service.get_all_sessions(status=filter_val)
        
        if not sessions:
            st.info(f"No {status_filter.lower() if status_filter != 'All' else ''} sessions found.")
            return
        
        st.caption(f"Showing {len(sessions)} session(s)")
        
        # Build display table
        table_data = []
        for s in sessions:
            # Count attendance records
            records = attendance_service.get_session_attendance(s['id'])
            present_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.PRESENT)
            late_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.LATE)
            total_marked = len(records)
            
            status_badge = {
                SessionStatus.ACTIVE: "🟢 Active",
                SessionStatus.SCHEDULED: "🟡 Scheduled",
                SessionStatus.CLOSED: "🔴 Closed"
            }.get(s['status'], s['status'])
            
            table_data.append({
                'ID': s['id'],
                'Session Name': s['session_name'],
                'Course': s['course_code'] or "—",
                'Start Time': format_datetime(s['start_time']),
                'Late Threshold': f"{s.get('late_threshold_minutes', 15)} min",
                'Status': status_badge,
                'Marked': f"{total_marked} ({present_cnt}P, {late_cnt}L)",
                'Raw Status': s['status']
            })
            
        st.dataframe(
            table_data,
            width="stretch",
            hide_index=True,
            column_config={
                'ID': st.column_config.NumberColumn('ID', width='small'),
                'Raw Status': None  # Hidden column
            }
        )
        
        # Lifecycle Quick Actions
        st.markdown("#### ⚡ Session Actions")
        action_cols_header = st.columns([1, 3, 2, 2, 2])
        action_cols_header[0].caption("ID")
        action_cols_header[1].caption("Session Name")
        action_cols_header[2].caption("Current Status")
        action_cols_header[3].caption("Action")
        action_cols_header[4].caption("Details")
        
        for s in sessions:
            cols = st.columns([1, 3, 2, 2, 2])
            cols[0].write(f"#{s['id']}")
            cols[1].write(f"**{s['session_name']}**")
            cols[2].write(f"`{s['status']}`")
            
            with cols[3]:
                if s['status'] == SessionStatus.SCHEDULED:
                    if st.button("▶️ Start", key=f"start_{s['id']}", width="stretch", type="primary"):
                        handle_activate_session(session_service, s['id'])
                elif s['status'] == SessionStatus.ACTIVE:
                    if st.button("⏹️ Close", key=f"close_{s['id']}", width="stretch"):
                        handle_close_session(session_service, s['id'])
                else:
                    st.caption("—")
                    
            with cols[4]:
                if st.button("👁️ View", key=f"view_{s['id']}", width="stretch"):
                    st.session_state['selected_session_id'] = s['id']
                    st.rerun()
                    
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        st.error(f"Failed to load sessions: {e}")


def render_create_session_tab(session_service: SessionService):
    """Render the form to create a new session."""
    st.subheader("➕ Create New Attendance Session")
    
    with st.form("create_session_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            session_name = st.text_input(
                "Session Name *",
                placeholder="e.g., CS101 Lecture 1",
                help="Name or title for the session"
            )
            course_code = st.text_input(
                "Course Code",
                placeholder="e.g., CS101",
                help="Course code or identifier (optional)"
            )
            created_by = st.text_input(
                "Created By",
                value="Admin",
                help="Instructor or administrator name"
            )
            
        with col2:
            start_date = st.date_input("Start Date *", value=datetime.now().date())
            start_time_val = st.time_input("Start Time *", value=datetime.now().time())
            
            late_threshold_minutes = st.number_input(
                "Late Threshold (Minutes)",
                min_value=0,
                max_value=120,
                value=15,
                step=1,
                help="Minutes after start time before student is considered late"
            )
            
            initial_status = st.selectbox(
                "Initial Status",
                options=[SessionStatus.SCHEDULED, SessionStatus.ACTIVE],
                format_func=lambda x: "Scheduled (activate later)" if x == SessionStatus.SCHEDULED else "Active (start now)"
            )
            
        notes = st.text_area(
            "Session Notes",
            placeholder="Additional details, topic, location, etc. (optional)"
        )
        
        st.caption("* Required fields")
        
        submitted = st.form_submit_button(
            "Create Session",
            type="primary",
            width="stretch"
        )
        
        if submitted:
            start_datetime = f"{start_date.strftime('%Y-%m-%d')} {start_time_val.strftime('%H:%M:%S')}"
            handle_create_session(
                session_service=session_service,
                session_name=session_name,
                start_time=start_datetime,
                course_code=course_code,
                late_threshold_minutes=int(late_threshold_minutes),
                status=initial_status,
                created_by=created_by,
                notes=notes
            )


def render_session_details_tab(
    session_service: SessionService,
    attendance_service: AttendanceService,
    student_service: StudentService
):
    """Render detailed view of a session with summary and full attendance table."""
    st.subheader("🔍 Session Details & Records")
    
    # Session Selector
    sessions = session_service.get_all_sessions()
    if not sessions:
        st.info("No sessions available.")
        return
        
    session_options = {s['id']: f"#{s['id']} - {s['session_name']} ({s['status']})" for s in sessions}
    
    current_selected = st.session_state.get('selected_session_id')
    default_idx = 0
    if current_selected in session_options:
        default_idx = list(session_options.keys()).index(current_selected)
        
    selected_id = st.selectbox(
        "Select Session to Inspect",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x],
        index=default_idx,
        key="session_detail_selector"
    )
    
    st.session_state.selected_session_id = selected_id
    
    session = session_service.get_session(selected_id)
    if not session:
        st.error(f"Session #{selected_id} not found.")
        return
        
    # Display Session Info Card
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown(f"**Session:** {session['session_name']}")
        st.markdown(f"**Course:** {session['course_code'] or '—'}")
        status_color = {
            SessionStatus.ACTIVE: "green",
            SessionStatus.SCHEDULED: "orange",
            SessionStatus.CLOSED: "red"
        }.get(session['status'], "blue")
        st.markdown(f"**Status:** :{status_color}[**{session['status'].upper()}**]")
        
    with col_info2:
        st.markdown(f"**Start Time:** {format_datetime(session['start_time'])}")
        st.markdown(f"**Closed At:** {format_datetime(session.get('closed_at'))}")
        st.markdown(f"**Late Cutoff:** {session.get('late_threshold_minutes', 15)} minutes")
        
    with col_info3:
        st.markdown(f"**Created By:** {session.get('created_by') or '—'}")
        st.markdown(f"**Created At:** {format_datetime(session.get('created_at'))}")
        if session.get('notes'):
            st.markdown(f"**Notes:** {session['notes']}")
            
    st.markdown("---")
    
    # Session Attendance Summary Metrics
    records = attendance_service.get_session_attendance(selected_id)
    present_count = sum(1 for r in records if r['status'] == AttendanceStatus.PRESENT)
    late_count = sum(1 for r in records if r['status'] == AttendanceStatus.LATE)
    total_marked = len(records)
    
    st.markdown("### 📊 Session Summary")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Marked", total_marked)
    with metric_cols[1]:
        st.metric("Present", present_count)
    with metric_cols[2]:
        st.metric("Late", late_count)
    with metric_cols[3]:
        can_accept = session_service.can_accept_attendance(selected_id)
        st.metric("Accepting Attendance", "Yes" if can_accept else "No")
        
    st.markdown("---")
    
    # Attendance Records Table
    st.markdown("### 👥 Attendance Records")
    
    if not records:
        st.info("No attendance records recorded for this session yet.")
        return
        
    # Filter by attendance status
    filter_col1, filter_col2 = st.columns([2, 2])
    with filter_col1:
        att_status_filter = st.selectbox(
            "Filter Attendance Records",
            options=["All", AttendanceStatus.PRESENT, AttendanceStatus.LATE],
            key="att_record_filter"
        )
        
    filtered_records = records
    if att_status_filter != "All":
        filtered_records = [r for r in records if r['status'] == att_status_filter]
        
    # Map student information
    records_display = []
    for r in filtered_records:
        student = student_service.get_student_by_id(r['student_id'])
        student_id_str = student['student_id'] if student else f"DB#{r['student_id']}"
        student_name_str = student['name'] if student else "Unknown"
        
        status_label = "✅ Present" if r['status'] == AttendanceStatus.PRESENT else "⏰ Late"
        conf_str = f"{r['detection_confidence']:.2f}" if r.get('detection_confidence') is not None else "—"
        liveness_str = f"{r['liveness_score']:.2f}" if r.get('liveness_score') is not None else "—"
        
        records_display.append({
            'Record ID': r['id'],
            'Student ID': student_id_str,
            'Student Name': student_name_str,
            'Status': status_label,
            'Marked At': format_datetime(r['marked_at']),
            'Confidence': conf_str,
            'Liveness': liveness_str,
            'Marked By': r.get('marked_by') or 'system',
            'Notes': r.get('notes') or '—'
        })
        
    st.dataframe(
        records_display,
        width="stretch",
        hide_index=True
    )


# ============================================================================
# Action Handlers
# ============================================================================

def handle_create_session(
    session_service: SessionService,
    session_name: str,
    start_time: str,
    course_code: Optional[str],
    late_threshold_minutes: int,
    status: str,
    created_by: Optional[str],
    notes: Optional[str]
):
    """Handle creating a new session through SessionService."""
    try:
        if not session_name or not session_name.strip():
            st.error("❌ Session Name is required.")
            return
            
        session_id = session_service.create_session(
            session_name=session_name,
            start_time=start_time,
            course_code=course_code if course_code else None,
            late_threshold_minutes=late_threshold_minutes,
            status=status,
            created_by=created_by if created_by else None,
            notes=notes if notes else None
        )
        
        st.success(f"✅ Session **{session_name}** created successfully (ID: #{session_id})!")
        logger.info(f"Session created via UI: ID {session_id}")
        st.session_state['selected_session_id'] = session_id
        st.rerun()
        
    except InvalidSessionDataError as e:
        st.error(f"❌ Invalid session data: {e}")
        logger.warning(f"Invalid session data in UI: {e}")
    except Exception as e:
        st.error(f"❌ Failed to create session: {e}")
        logger.error(f"Error creating session: {e}")


def handle_activate_session(session_service: SessionService, session_id: int):
    """Handle activating a session through SessionService."""
    try:
        session_service.activate_session(session_id)
        st.success(f"✅ Session #{session_id} activated successfully!")
        logger.info(f"Session #{session_id} activated via UI")
        st.rerun()
    except (InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot activate session: {e}")
    except Exception as e:
        st.error(f"❌ Error activating session: {e}")


def handle_close_session(session_service: SessionService, session_id: int):
    """Handle closing an active session through SessionService."""
    try:
        session_service.close_session(session_id)
        st.success(f"✅ Session #{session_id} closed successfully!")
        logger.info(f"Session #{session_id} closed via UI")
        st.rerun()
    except (InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot close session: {e}")
    except Exception as e:
        st.error(f"❌ Error closing session: {e}")


def format_datetime(dt_string: Optional[str]) -> str:
    """Format datetime string for clean display."""
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return dt_string
