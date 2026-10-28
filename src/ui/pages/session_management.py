"""
Session Management UI Page for the Intelligent Attendance Monitoring System.

Phase 4D: Attendance UI & Management
Provides Streamlit interface for managing sessions through SessionService and AttendanceService.
Allows viewing sessions, creating, editing and deleting sessions, starting
scheduled sessions early, closing active sessions, and viewing each session's
full attendance (every expected student, absent ones included).
Scheduled sessions start and close by themselves (src/services/session_scheduler.py);
an open page notices those changes within STATUS_CHECK_SECONDS.
Uses Service layer as the application boundary - no direct database/SQL queries.
"""

import streamlit as st
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.services.session_service import (
    SessionService,
    SessionStatus,
    InvalidSessionDataError,
    SessionNotFoundError,
    InvalidSessionStateError,
    parse_session_datetime,
    DB_DATETIME_FORMAT
)
from src.services.attendance_service import (
    AttendanceService,
    AttendanceStatus,
    marked_by_label
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TAB_LIST = "📋 Sessions List"
TAB_CREATE = "➕ Create New Session"
TAB_DETAILS = "🔍 Session Details & Records"
TABS_KEY = "session_mgmt_tab"
DETAIL_SELECTOR_KEY = "session_detail_selector"
FLASH_KEY = "session_mgmt_flash"
ROSTER_EDIT_VERSION_KEY = "roster_edit_version"
TABLE_VERSION_KEY = "sessions_table_version"
CREATE_FORM_VERSION_KEY = "create_session_form_version"

ROSTER_MAX_ROWS = 500       # students sent to the browser at most; filter / search find the rest
STATUS_CHECK_SECONDS = 10   # an open page notices a scheduled start / automatic close this fast

STATUS_BADGES = {
    SessionStatus.ACTIVE: "🟢 Active",
    SessionStatus.SCHEDULED: "🟡 Scheduled",
    SessionStatus.CLOSED: "🔴 Closed"
}
ATTENDANCE_LABELS = {
    AttendanceStatus.PRESENT: "✅ Present",
    AttendanceStatus.LATE: "⏰ Late",
    AttendanceStatus.ABSENT: "❌ Absent"
}


def render_session_management():
    """Main function to render the Session Management page."""
    st.title("📅 Attendance Session Management")
    st.caption("Phase 4D: Session Lifecycle & Attendance Management")
    st.markdown("Create, schedule, start, close, and monitor attendance sessions. "
                "Scheduled sessions start and close automatically at their times.")
    st.markdown("---")

    # Initialize services
    session_service = SessionService()
    attendance_service = AttendanceService()

    # Catch up on due starts / end times before anything is shown (the
    # background scheduler does this too, every few seconds)
    _apply_schedule(session_service)
    _show_flash()

    # Tabs track the open tab: only that tab's content runs, and the
    # "View" action can switch to the details tab
    tab_list, tab_create, tab_details = st.tabs(
        [TAB_LIST, TAB_CREATE, TAB_DETAILS], key=TABS_KEY, on_change="rerun"
    )

    if tab_list.open:
        with tab_list:
            render_sessions_list_tab(session_service)

    if tab_create.open:
        with tab_create:
            render_create_session_tab(session_service)

    if tab_details.open:
        with tab_details:
            render_session_details_tab(session_service, attendance_service)

    try:
        _status_watcher(session_service, session_service.get_status_signature())
    except Exception as e:
        logger.warning(f"Session status watcher not started: {e}")


def render_sessions_list_tab(session_service: SessionService):
    """Render the list of sessions with filter and lifecycle action controls."""
    st.subheader("📋 Existing Sessions")

    col_filter, col_search = st.columns([1, 2])

    with col_filter:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", SessionStatus.ACTIVE, SessionStatus.SCHEDULED, SessionStatus.CLOSED],
            format_func=lambda x: "All Sessions" if x == "All" else f"{x.capitalize()} Sessions",
            key="session_status_filter"
        )

    with col_search:
        search = st.text_input(
            "Search",
            placeholder="Session name or course code",
            key="session_list_search"
        ).strip().lower()

    try:
        sessions = session_service.get_sessions_overview(
            status=None if status_filter == "All" else status_filter
        )
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        st.error(f"Failed to load sessions: {e}")
        return

    if search:
        sessions = [s for s in sessions
                    if search in s['session_name'].lower() or search in (s['course_code'] or '').lower()]

    if not sessions:
        st.info("No sessions match." if search or status_filter != "All"
                else "No sessions yet. Create one in the **➕ Create New Session** tab.")
        return

    st.caption(f"Showing {len(sessions)} session(s). **Select a row** to start, close, view, edit or delete it.")

    table_data = [{
        'ID': s['id'],
        'Session Name': s['session_name'],
        'Course': s['course_code'] or "—",
        'Start Time': format_datetime(s['start_time']),
        'End Time': format_datetime(s.get('end_time')),
        'Status': STATUS_BADGES.get(s['status'], s['status']),
        'Present': s['present_count'],
        'Late': s['late_count'],
    } for s in sessions]

    # A new key per filter / after a delete: the selection never points at
    # a row that now holds another session
    table_key = (f"sessions_table_{status_filter}_{search}_"
                 f"{st.session_state.get(TABLE_VERSION_KEY, 0)}")
    event = st.dataframe(
        table_data,
        width="stretch",
        hide_index=True,
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            'ID': st.column_config.NumberColumn('ID', width='small'),
            'Present': st.column_config.NumberColumn('Present', width='small'),
            'Late': st.column_config.NumberColumn('Late', width='small'),
        }
    )

    selected_rows = event.selection.rows if event else []
    if selected_rows and selected_rows[0] < len(sessions):
        st.markdown("#### ⚡ Session Actions")
        _render_session_actions(session_service, sessions[selected_rows[0]], show_view=True)


def render_create_session_tab(session_service: SessionService):
    """Render the form to create a new session."""
    st.subheader("➕ Create New Attendance Session")
    st.caption("Times are the attendance computer's local time. A scheduled session starts by itself "
               "at its start time (no one needs to be here) and closes by itself at its end time.")

    # A new form key after each successful create gives an empty form,
    # while a rejected submission keeps what was typed
    form_version = st.session_state.get(CREATE_FORM_VERSION_KEY, 0)
    with st.form(f"create_session_form_{form_version}"):
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
            initial_status = st.selectbox(
                "Initial Status",
                options=[SessionStatus.SCHEDULED, SessionStatus.ACTIVE],
                format_func=lambda x: ("Scheduled (starts automatically at the start time)"
                                       if x == SessionStatus.SCHEDULED else "Active (start now)")
            )

        with col2:
            # "today" / "now" (not datetime.now()) keep the widgets stable
            # across reruns, so the time picked is the time submitted
            start_date = st.date_input("Start Date *", value="today")
            col_start, col_end = st.columns(2)
            with col_start:
                start_time_val = st.time_input("Start Time *", value="now", step=300)
            with col_end:
                end_time_val = st.time_input(
                    "End Time",
                    value=None,
                    step=300,
                    help="Optional. The session closes automatically at this time (same day as the start)."
                )

            late_threshold_minutes = st.number_input(
                "Late Threshold (Minutes)",
                min_value=0,
                max_value=120,
                value=15,
                step=1,
                help="Minutes after start time before student is considered late"
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
        if start_time_val is None:
            st.error("❌ Start Time is required.")
            return
        start_dt = datetime.combine(start_date, start_time_val).replace(second=0, microsecond=0)
        end_dt = datetime.combine(start_date, end_time_val).replace(second=0, microsecond=0) if end_time_val else None
        handle_create_session(
            session_service=session_service,
            session_name=session_name,
            start_dt=start_dt,
            end_dt=end_dt,
            course_code=course_code,
            late_threshold_minutes=int(late_threshold_minutes),
            status=initial_status,
            created_by=created_by,
            notes=notes
        )


def render_session_details_tab(
    session_service: SessionService,
    attendance_service: AttendanceService
):
    """Render detailed view of a session with summary and every expected student's status."""
    st.subheader("🔍 Session Details & Records")

    # Session Selector (the overview also has the counts the Delete action needs)
    try:
        sessions = session_service.get_sessions_overview()
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        st.error(f"Failed to load sessions: {e}")
        return
    if not sessions:
        st.info("No sessions available.")
        return

    session_options = {s['id']: f"#{s['id']} - {s['session_name']} ({s['status']})" for s in sessions}

    # The selector's own state is dropped while another tab is open, so the
    # choice is remembered separately (and set by the list's "View" action)
    if st.session_state.get(DETAIL_SELECTOR_KEY) not in session_options:
        remembered = st.session_state.get('selected_session_id')
        st.session_state[DETAIL_SELECTOR_KEY] = remembered if remembered in session_options else sessions[0]['id']

    selected_id = st.selectbox(
        "Select Session to Inspect",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x],
        key=DETAIL_SELECTOR_KEY
    )

    st.session_state['selected_session_id'] = selected_id

    session = next((s for s in sessions if s['id'] == selected_id), None)
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
        st.markdown(f"**End Time:** {format_datetime(session.get('end_time'))}")
        st.markdown(f"**Late Cutoff:** {session.get('late_threshold_minutes', 15)} minutes")

    with col_info3:
        st.markdown(f"**Closed At:** {format_db_timestamp(session.get('closed_at'))}")
        st.markdown(f"**Created By:** {session.get('created_by') or '—'}")
        st.markdown(f"**Created At:** {format_db_timestamp(session.get('created_at'))}")

    if session.get('notes'):
        st.markdown(f"**Notes:** {session['notes']}")

    _render_session_actions(session_service, session, show_view=False)

    st.markdown("---")

    # Session Attendance Summary Metrics: counted over the same roster the
    # table below lists, so the numbers always match the table
    try:
        counts = attendance_service.get_session_roster_counts(selected_id)
    except Exception as e:
        logger.error(f"Error loading attendance of session #{selected_id}: {e}")
        st.error(f"Failed to load attendance: {e}")
        return

    attended = counts[AttendanceStatus.PRESENT] + counts[AttendanceStatus.LATE]
    rate = f"{attended / counts['expected'] * 100:.0f}%" if counts['expected'] else "—"

    st.markdown("### 📊 Session Summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Expected", counts['expected'])
    metric_cols[1].metric("Present", counts[AttendanceStatus.PRESENT])
    metric_cols[2].metric("Late", counts[AttendanceStatus.LATE])
    metric_cols[3].metric("Absent", counts[AttendanceStatus.ABSENT])
    metric_cols[4].metric("Attendance", rate)

    roster_note = ("Expected students: every active student registered before the session started, "
                   "plus anyone marked in it.")
    if session['status'] != SessionStatus.CLOSED:
        roster_note += " Until the session closes, **Absent** means *not marked yet*."
    st.caption(roster_note)

    st.markdown("---")

    # Attendance Records Table
    st.markdown("### 👥 Students")

    if not counts['expected']:
        st.info("No students are registered yet, so nobody is expected in this session.")
        return

    _render_roster(attendance_service, selected_id, session['status'], counts)


@st.fragment
def _render_roster(
    attendance_service: AttendanceService,
    session_id: int,
    session_status: str,
    counts: Dict[str, int]
):
    """
    The session's students sorted Present -> Late -> Absent, then by name.

    A fragment: filtering and searching rerun only this part. Filtering,
    searching and the ROSTER_MAX_ROWS limit run in the database, so even a
    register of thousands of students stays quick and the page uncluttered.

    Once the session has started, the Status column can be changed by hand
    (e.g. a student the camera missed -> Present); each change is saved at
    once and the page redrawn so the counts above follow.
    """
    filter_options = ["all", AttendanceStatus.PRESENT, AttendanceStatus.LATE, AttendanceStatus.ABSENT]
    filter_labels = {"all": f"All ({counts['expected']:,})"}
    filter_labels.update({s: f"{ATTENDANCE_LABELS[s]} ({counts[s]:,})" for s in filter_options[1:]})

    col_filter, col_search = st.columns([3, 2])
    with col_filter:
        status_choice = st.segmented_control(
            "Show",
            options=filter_options,
            default="all",
            required=True,
            format_func=lambda option: filter_labels[option],
            key=f"roster_filter_{session_id}"
        )
    with col_search:
        search = st.text_input(
            "Find a student",
            placeholder="Name or student ID",
            key=f"roster_search_{session_id}"
        )

    status = None if status_choice in (None, "all") else status_choice
    try:
        rows = attendance_service.get_session_roster(
            session_id, status=status, search=search, limit=ROSTER_MAX_ROWS + 1
        )
    except Exception as e:
        logger.error(f"Error loading the students of session #{session_id}: {e}")
        st.error(f"Failed to load the students: {e}")
        return

    if not rows:
        st.info("No students match.")
        return

    truncated = len(rows) > ROSTER_MAX_ROWS
    rows = rows[:ROSTER_MAX_ROWS]

    table = [{
        'Status': ATTENDANCE_LABELS.get(r['status'], r['status']),
        'Student ID': r['student_id'],
        'Student Name': r['name'],
        'Marked At': format_datetime(r['marked_at']) if r['marked_at'] else "",
        'Confidence': r['detection_confidence'],
        'Liveness': r['liveness_score'],
        'Marked By': marked_by_label(r['marked_by'], r.get('verified_by')),
    } for r in rows]
    column_config = {
        'Confidence': st.column_config.NumberColumn('Confidence', format="%.2f"),
        'Liveness': st.column_config.NumberColumn('Liveness', format="%.2f"),
    }
    height = min(38 + 35 * len(rows), 460)

    if session_status == SessionStatus.SCHEDULED:
        st.dataframe(table, width="stretch", hide_index=True, height=height, column_config=column_config,
                     placeholder="")
        st.caption("Attendance can be changed by hand once the session has started.")
    else:
        st.caption("✏️ To fix attendance, change a student's **Status** in the table (e.g. Absent → Present). "
                   "It is saved at once and shown as *Instructor* under Marked By.")
        # A new key after each save: the editor starts clean on the re-sorted rows
        editor_key = f"roster_editor_{session_id}_{st.session_state.get(ROSTER_EDIT_VERSION_KEY, 0)}"
        column_config['Status'] = st.column_config.SelectboxColumn(
            'Status', options=list(ATTENDANCE_LABELS.values()), required=True
        )
        st.data_editor(
            table, width="stretch", hide_index=True, height=height, key=editor_key, placeholder="",
            column_config=column_config,
            disabled=[name for name in table[0] if name != 'Status'],
        )
        edited = (st.session_state.get(editor_key) or {}).get('edited_rows') or {}
        if edited:
            _save_status_changes(attendance_service, session_id, rows, edited)

    if truncated:
        total = counts['expected'] if status is None else counts.get(status, 0)
        shown = f"of {total:,}" if not search.strip() else "matches"
        st.caption(f"Showing the first {ROSTER_MAX_ROWS:,} {shown} — use the filter or search to find others.")


def _save_status_changes(
    attendance_service: AttendanceService,
    session_id: int,
    rows: list,
    edited: Dict[int, Dict[str, Any]]
):
    """Save Status changes made in the roster editor, then redraw the whole page (counts included)."""
    status_by_label = {label: status for status, label in ATTENDANCE_LABELS.items()}
    changed, errors = [], []
    for index, change in edited.items():
        new_status = status_by_label.get(change.get('Status'))
        if new_status is None or not (0 <= int(index) < len(rows)):
            continue
        row = rows[int(index)]
        if new_status == row['status']:
            continue
        try:
            attendance_service.set_attendance_manually(session_id, row['student_db_id'], new_status)
            changed.append(f"{row['name']} → {ATTENDANCE_LABELS[new_status]}")
        except Exception as e:
            logger.error(f"Could not change attendance of {row['name']}: {e}")
            errors.append(f"{row['name']}: {e}")

    st.session_state[ROSTER_EDIT_VERSION_KEY] = st.session_state.get(ROSTER_EDIT_VERSION_KEY, 0) + 1
    if errors:
        _flash("❌ Not saved — " + "; ".join(errors), kind="error")
    elif changed:
        _flash("✅ Saved: " + ", ".join(changed))
    st.rerun(scope="app")


# ============================================================================
# Session actions (start / close / view / edit / delete)
# ============================================================================

def _render_session_actions(session_service: SessionService, session: Dict[str, Any], show_view: bool):
    """Action buttons for one session, with what the scheduler will do next."""
    session_id = session['id']
    status = session['status']

    st.markdown(f"**#{session_id} — {session['session_name']}** · "
                f"{STATUS_BADGES.get(status, status)} · {_schedule_text(session)}")

    cols = st.columns(4)

    with cols[0]:
        if status == SessionStatus.SCHEDULED:
            if st.button("▶️ Start Now", key=f"start_{session_id}", width="stretch", type="primary",
                         help="Start now. Started after its scheduled time, the session starts fresh: lateness counts from now."):
                handle_activate_session(session_service, session_id)
        elif status == SessionStatus.ACTIVE:
            # Closing is final: a closed session cannot be reopened
            with st.popover("⏹️ Close", width="stretch"):
                st.write("Closing ends attendance for this session. It cannot be reopened.")
                if st.button("Close session", key=f"close_{session_id}", type="primary"):
                    handle_close_session(session_service, session_id)
        else:
            st.button("✔️ Closed", key=f"closed_{session_id}", width="stretch", disabled=True)

    with cols[1]:
        if show_view:
            st.button("👁️ View", key=f"view_{session_id}", width="stretch",
                      on_click=_open_session_details, args=(session_id,))

    with cols[2]:
        if st.button("✏️ Edit", key=f"edit_{session_id}", width="stretch"):
            _edit_session_dialog(session_service, session)

    with cols[3]:
        if status == SessionStatus.ACTIVE:
            st.button("🗑️ Delete", key=f"delete_{session_id}", width="stretch", disabled=True,
                      help="Close the session before deleting it")
        else:
            with st.popover("🗑️ Delete", width="stretch"):
                marked = session.get('marked_count', 0)
                if marked:
                    st.warning(f"Its {marked} attendance record(s) are deleted too.")
                st.write("This cannot be undone.")
                if st.button("Delete session", key=f"confirm_delete_{session_id}", type="primary"):
                    handle_delete_session(session_service, session_id)


def _open_session_details(session_id: int):
    """'View' callback: runs before the rerun, so the details tab opens on this session."""
    st.session_state['selected_session_id'] = session_id
    st.session_state[DETAIL_SELECTOR_KEY] = session_id
    st.session_state[TABS_KEY] = TAB_DETAILS


@st.dialog("✏️ Edit Session")
def _edit_session_dialog(session_service: SessionService, session: Dict[str, Any]):
    """Edit a session, whatever its status. Changing its start time or late threshold re-grades its marks."""
    status = session['status']
    start_dt = parse_session_datetime(session['start_time'], "start_time")
    end_dt = parse_session_datetime(session['end_time'], "end_time") if session.get('end_time') else None

    if status != SessionStatus.SCHEDULED:
        st.caption("Changing the start time or late threshold re-checks who was present or late. "
                   + ("The session stays closed." if status == SessionStatus.CLOSED else ""))

    with st.form(f"edit_session_form_{session['id']}"):
        session_name = st.text_input("Session Name *", value=session['session_name'])
        course_code = st.text_input("Course Code", value=session['course_code'] or "")
        start_date = st.date_input("Start Date", value=start_dt.date())
        col_start, col_end = st.columns(2)
        with col_start:
            start_time_val = st.time_input("Start Time", value=start_dt.time(), step=300)
        with col_end:
            end_time_val = st.time_input("End Time", value=end_dt.time() if end_dt else None, step=300,
                                         help="Optional; clear it for no end time")
        late_threshold = st.number_input("Late Threshold (Minutes)", min_value=0, max_value=120,
                                         value=int(session.get('late_threshold_minutes') or 15))
        notes = st.text_area("Session Notes", value=session.get('notes') or "")
        submitted = st.form_submit_button("Save Changes", type="primary", width="stretch")

    if not submitted:
        return

    changes: Dict[str, Any] = {
        'session_name': session_name,
        'course_code': course_code,
        'notes': notes,
    }
    if start_time_val is None:
        st.error("❌ Start Time is required.")
        return
    new_start = datetime.combine(start_date, start_time_val)
    changes['start_time'] = new_start.strftime(DB_DATETIME_FORMAT)
    changes['late_threshold_minutes'] = int(late_threshold)
    if end_time_val is None:
        changes['clear_end_time'] = end_dt is not None
    else:
        new_end = datetime.combine(new_start.date(), end_time_val)
        if new_end != end_dt:
            # An open session given a past end time would close at once
            if status != SessionStatus.CLOSED and new_end <= datetime.now():
                st.error("❌ The end time has already passed. To end the session now, close it instead.")
                return
            changes['end_time'] = new_end.strftime(DB_DATETIME_FORMAT)

    if not session_name.strip():
        st.error("❌ Session Name is required.")
        return

    try:
        session_service.update_session(session['id'], **changes)
    except (InvalidSessionDataError, InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot save: {e}")
        return
    except Exception as e:
        logger.error(f"Error updating session #{session['id']}: {e}")
        st.error(f"❌ Failed to save: {e}")
        return

    logger.info(f"Session #{session['id']} edited via UI")
    _apply_schedule(session_service)  # a start time moved into the past starts it now
    _flash(f"✅ Session #{session['id']} saved." + _overlap_note(
        session_service, changes.get('start_time', session['start_time']),
        changes.get('end_time', None if changes.get('clear_end_time') else session.get('end_time')),
        course_code, session['id']))
    st.rerun()


# ============================================================================
# Action Handlers
# ============================================================================

def handle_create_session(
    session_service: SessionService,
    session_name: str,
    start_dt: datetime,
    end_dt: Optional[datetime],
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
        if end_dt is not None and end_dt <= start_dt:
            st.error("❌ The end time must be after the start time.")
            return
        if end_dt is not None and end_dt <= datetime.now():
            st.error("❌ The end time has already passed.")
            return

        start_time = start_dt.strftime(DB_DATETIME_FORMAT)
        end_time = end_dt.strftime(DB_DATETIME_FORMAT) if end_dt else None
        session_id = session_service.create_session(
            session_name=session_name,
            start_time=start_time,
            course_code=course_code if course_code else None,
            end_time=end_time,
            late_threshold_minutes=late_threshold_minutes,
            status=status,
            created_by=created_by if created_by else None,
            notes=notes if notes else None
        )

        logger.info(f"Session created via UI: ID {session_id}")
        message = f"✅ Session **{session_name.strip()}** created (ID: #{session_id})."
        if status == SessionStatus.SCHEDULED:
            if start_dt <= datetime.now():
                _apply_schedule(session_service)
                message += " Its start time has passed, so it started right away."
            else:
                message += f" It starts automatically {format_datetime(start_time)} ({_relative(start_dt)})."
        message += _overlap_note(session_service, start_time, end_time, course_code, session_id)
        _flash(message)
        st.session_state['selected_session_id'] = session_id
        st.session_state[CREATE_FORM_VERSION_KEY] = st.session_state.get(CREATE_FORM_VERSION_KEY, 0) + 1
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
        logger.info(f"Session #{session_id} activated via UI")
        _flash(f"✅ Session #{session_id} started.")
        st.rerun()
    except (InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot activate session: {e}")
    except Exception as e:
        st.error(f"❌ Error activating session: {e}")


def handle_close_session(session_service: SessionService, session_id: int):
    """Handle closing an active session through SessionService."""
    try:
        session_service.close_session(session_id)
        logger.info(f"Session #{session_id} closed via UI")
        _flash(f"✅ Session #{session_id} closed.")
        st.rerun()
    except (InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot close session: {e}")
    except Exception as e:
        st.error(f"❌ Error closing session: {e}")


def handle_delete_session(session_service: SessionService, session_id: int):
    """Handle deleting a (not active) session through SessionService."""
    try:
        session_service.delete_session(session_id)
        logger.info(f"Session #{session_id} deleted via UI")
        _flash(f"🗑️ Session #{session_id} deleted.")
        st.session_state[TABLE_VERSION_KEY] = st.session_state.get(TABLE_VERSION_KEY, 0) + 1
        st.rerun()
    except (InvalidSessionStateError, SessionNotFoundError) as e:
        st.error(f"❌ Cannot delete session: {e}")
    except Exception as e:
        st.error(f"❌ Error deleting session: {e}")


# ============================================================================
# Helpers
# ============================================================================

def _apply_schedule(session_service: SessionService):
    """Apply due scheduled starts / automatic closes now (cheap when nothing is due)."""
    try:
        session_service.apply_scheduled_transitions()
    except Exception as e:
        logger.error(f"Could not apply scheduled session changes: {e}")


@st.fragment(run_every=STATUS_CHECK_SECONDS)
def _status_watcher(session_service: SessionService, rendered_signature: tuple):
    """Reload the page when a session started / closed by itself (or elsewhere) since it was drawn."""
    try:
        if session_service.get_status_signature() != rendered_signature:
            st.rerun(scope="app")
    except Exception as e:
        logger.warning(f"Session status check failed: {e}")


def _flash(message: str, kind: str = "success"):
    """Show a message on the next run (st.rerun() would discard one shown now)."""
    st.session_state[FLASH_KEY] = (kind, message)


def _show_flash():
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)


def _overlap_note(
    session_service: SessionService,
    start_time: str,
    end_time: Optional[str],
    course_code: Optional[str],
    session_id: int
) -> str:
    """A warning sentence when open sessions of the same course overlap this one ('' otherwise)."""
    if not (course_code or "").strip():
        return ""
    try:
        overlapping = session_service.find_overlapping_sessions(start_time, end_time, course_code, session_id)
    except Exception as e:
        logger.warning(f"Overlap check failed: {e}")
        return ""
    if not overlapping:
        return ""
    others = ", ".join(f"#{s['id']} {s['session_name']}" for s in overlapping[:3])
    return f" ⚠️ Overlaps another open {course_code.strip()} session: {others}."


def _schedule_text(session: Dict[str, Any]) -> str:
    """What happens next to a session, e.g. 'starts automatically 2026-10-24 10:00 (in 25 min)'."""
    status = session['status']
    try:
        start_dt = parse_session_datetime(session['start_time'], "start_time")
        end_dt = parse_session_datetime(session['end_time'], "end_time") if session.get('end_time') else None
    except InvalidSessionDataError:
        return ""
    if status == SessionStatus.SCHEDULED:
        return f"starts automatically {start_dt:%Y-%m-%d %H:%M} ({_relative(start_dt)})"
    if status == SessionStatus.ACTIVE:
        if end_dt:
            return f"closes automatically {end_dt:%Y-%m-%d %H:%M} ({_relative(end_dt)})"
        return "no end time — close it when the class ends"
    return f"closed {format_db_timestamp(session.get('closed_at'))}"


def _relative(moment: datetime) -> str:
    """'in 25 min', 'in 2 h 5 min', 'in 3 days' (or 'now')."""
    seconds = (moment - datetime.now()).total_seconds()
    if seconds < 60:
        return "now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"in {minutes} min"
    if minutes < 24 * 60:
        return f"in {minutes // 60} h {minutes % 60} min"
    return f"in {minutes // (24 * 60)} days"


def format_datetime(dt_string: Optional[str]) -> str:
    """Format a local-time datetime string (session times, marked_at) for clean display."""
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return dt_string


def format_db_timestamp(dt_string: Optional[str]) -> str:
    """
    Format a database-default timestamp (created_at, closed_at): SQLite's
    CURRENT_TIMESTAMP is UTC, so convert it to local time for display.
    """
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return dt_string
