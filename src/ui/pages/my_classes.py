"""
My Classes: the sections linked to the signed-in account (an instructor's
classes). Per section: start a session now (or open the camera of the one
running), its recent sessions, and its class list (read only - Admin / IT
change class lists in Courses & Sections).
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.services.course_service import CourseService
from src.services.session_service import SessionService
from src.ui.pages.courses_sections import render_start_now
from src.ui.pages.login import current_user
from src.ui.pages.session_management import (
    DETAIL_SELECTOR_KEY, STATUS_BADGES, TAB_DETAILS, TABS_KEY, format_datetime,
)

NAV_KEY = "navigation"
PAGE_SESSIONS = "Session Management"
SECTION_KEY = "my_classes_section"
RECENT_SESSIONS = 10      # sessions of the section listed
CLASS_LIST_ROWS = 500     # class list rows shown at once (search finds the rest)


def render_my_classes():
    """Main function to render the My Classes page."""
    user = current_user()
    st.title("📚 My Classes")
    st.markdown("Your sections: start a session, see its recent sessions and its students.")
    st.markdown("---")

    course_service = CourseService()
    sections = course_service.instructor_sections(user['id'])
    if not sections:
        st.info("No sections are linked to your account yet. Ask an administrator to choose you as the "
                "instructor of your sections (Courses & Sections → section settings → Instructor account).")
        return

    labels = {s['id']: f"{s['course_code']} - {s['course_name']} · Section {s['name']} · "
                       f"{s['student_count']:,} student(s)" for s in sections}
    if st.session_state.get(SECTION_KEY) not in labels:
        st.session_state[SECTION_KEY] = sections[0]['id']
    section_id = st.selectbox("Class", list(labels), format_func=labels.get, key=SECTION_KEY)
    section = next(s for s in sections if s['id'] == section_id)

    render_start_now(section)
    st.markdown("---")
    _render_recent_sessions(section)
    st.markdown("---")
    _render_class_list(course_service, section)


def _render_recent_sessions(section: Dict[str, Any]):
    st.markdown("#### 🗓️ Recent sessions")
    sessions = [s for s in SessionService().get_sessions_overview() if s.get('section_id') == section['id']]
    if not sessions:
        st.caption("No sessions yet for this class.")
        return
    sessions = sessions[:RECENT_SESSIONS]
    for session in sessions:
        col_text, col_button = st.columns([5, 1], vertical_alignment="center")
        col_text.markdown(f"{STATUS_BADGES.get(session['status'], session['status'])} · "
                          f"**#{session['id']} {session['session_name']}** · {format_datetime(session['start_time'])}"
                          f" · ✅ {session['present_count']} · ⏰ {session['late_count']}")
        col_button.button("🔍 Details", key=f"my_classes_details_{session['id']}", width="stretch",
                          on_click=_open_details, args=(session['id'],))


def _open_details(session_id: int):
    """Details button: open Session Management on this session (runs before the rerun)."""
    st.session_state[NAV_KEY] = PAGE_SESSIONS
    st.session_state['selected_session_id'] = session_id
    st.session_state[DETAIL_SELECTOR_KEY] = session_id
    st.session_state[TABS_KEY] = TAB_DETAILS


def _render_class_list(course_service: CourseService, section: Dict[str, Any]):
    st.markdown(f"#### 📋 Class list ({section['student_count']:,} student(s))")
    if not section['student_count']:
        st.info("This class list is empty. Ask an administrator to add the students.")
        return
    search = st.text_input("Search the class list", placeholder="Name or student ID",
                           key=f"my_class_search_{section['id']}").strip()
    students = course_service.get_class_list(section['id'], search=search, limit=CLASS_LIST_ROWS)
    if not students:
        st.info("No student on this class list matches.")
        return
    st.dataframe(pd.DataFrame([{
        'Student ID': s['student_id'], 'Name': s['name'], 'Department': s['department'] or '—',
    } for s in students]), hide_index=True, width="stretch")
    if len(students) == CLASS_LIST_ROWS:
        st.caption(f"Showing the first {CLASS_LIST_ROWS:,}; search to find others.")
    st.caption("Class lists are changed by an administrator in Courses & Sections.")
