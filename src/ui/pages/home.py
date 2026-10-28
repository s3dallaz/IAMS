"""
Home page: the first page the app opens on.

Sections, top to bottom:
1. Getting started - the setup steps, ticked off from the data (folded away
   once all are done)
2. Today - live counts of active sessions, today's sessions, the next one,
   and quick buttons (refreshes itself every few seconds)
3. Overview - students, missing face data, attendance rate, a 14-day chart
4. System check - models, scheduler, recognition data, camera, AI Assistant

Read-only: it never opens the camera; it only reports whether it is on.
"""

import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import config
from src.services.enrollment_service import EnrollmentService
from src.services.session_scheduler import is_session_scheduler_running
from src.services.session_service import SessionService, SessionStatus
from src.services.statistics_service import StatisticsService
from src.services.student_service import StudentService
from src.ui.pages.session_management import (
    DETAIL_SELECTOR_KEY, TAB_CREATE, TAB_DETAILS, TABS_KEY,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sidebar page names (app.py)
NAV_KEY = "navigation"
NAV_PENDING_KEY = "home_nav_pending"
PAGE_ATTENDANCE = "Real-Time Attendance"
PAGE_SESSIONS = "Session Management"
PAGE_STUDENTS = "Student Registry"
PAGE_REPORTS = "Statistics & Reports"
PAGE_ASSISTANT = "AI Assistant"

CAMERA_SELECTOR_KEY = "realtime_session_selector"  # session picker of the camera page
REALTIME_PAGE_MODULE = "src.ui.pages.realtime_attendance"

TODAY_REFRESH_SECONDS = 10
TREND_DAYS = 14
DB_FORMAT = '%Y-%m-%d %H:%M:%S'

STATUS_LABELS = {
    SessionStatus.ACTIVE: "🟢 Active",
    SessionStatus.SCHEDULED: "🕒 Scheduled",
    SessionStatus.CLOSED: "✅ Closed",
}

# (file, what it is for)
MODEL_FILES = [
    ("face_detection_yunet_2023mar.onnx", "finds faces"),
    ("face_recognition_sface_2021dec.onnx", "recognises faces"),
    ("face_landmarker.task", "blink check"),
    ("MiniFASNetV2.onnx", "photo / screen check"),
    ("MiniFASNetV1SE.onnx", "photo / screen check"),
]


# -- navigation (button callbacks: they run before the rerun) ------------------

def _go(page: str, **state: Any) -> None:
    st.session_state[NAV_KEY] = page
    for key, value in state.items():
        st.session_state[key] = value
    # A click inside the "Today" fragment reruns only the fragment; this
    # flag makes it rerun the whole app so the new page opens
    st.session_state[NAV_PENDING_KEY] = True


def _go_create_session() -> None:
    _go(PAGE_SESSIONS, **{TABS_KEY: TAB_CREATE})


def _go_session_details(session_id: int) -> None:
    _go(PAGE_SESSIONS, **{TABS_KEY: TAB_DETAILS, 'selected_session_id': session_id,
                          DETAIL_SELECTOR_KEY: session_id})


def _go_camera(session_id: int) -> None:
    _go(PAGE_ATTENDANCE, **{CAMERA_SELECTOR_KEY: session_id})


# -- data helpers (plain functions, tested on their own) -----------------------

def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:19], DB_FORMAT)
    except ValueError:
        return None


def todays_sessions(sessions: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
    """Sessions that start today, plus active ones started earlier; by start time."""
    today = now.date()
    picked = []
    for s in sessions:
        start = _parse(s.get('start_time'))
        if (start and start.date() == today) or s.get('status') == SessionStatus.ACTIVE:
            picked.append(s)
    return sorted(picked, key=lambda s: str(s.get('start_time') or ''))


def next_session(sessions: List[Dict[str, Any]], now: datetime) -> Optional[Dict[str, Any]]:
    """The scheduled session that starts next (after now), or None."""
    upcoming = [s for s in sessions
                if s.get('status') == SessionStatus.SCHEDULED
                and (_parse(s.get('start_time')) or now) > now]
    return min(upcoming, key=lambda s: str(s['start_time']), default=None)


def sessions_in_last_days(sessions: List[Dict[str, Any]], now: datetime, days: int = 7) -> int:
    """Sessions that started (or start) within the last `days` days up to now."""
    since = now - timedelta(days=days)
    return sum(1 for s in sessions
               if (start := _parse(s.get('start_time'))) and since <= start <= now)


def trend_last_days(trend: List[Dict[str, Any]], today: datetime, days: int = TREND_DAYS) -> pd.DataFrame:
    """Present / late per day for the last `days` days; days without records show 0."""
    by_day = {str(row.get('attendance_date')): row for row in trend}
    rows = []
    for offset in range(days - 1, -1, -1):
        day = (today - timedelta(days=offset)).date()
        row = by_day.get(day.isoformat(), {})
        rows.append({'Day': day.strftime('%d %b'),
                     'Present': int(row.get('present_count') or 0),
                     'Late': int(row.get('late_count') or 0)})
    return pd.DataFrame(rows)


def guide_steps(students: int, students_with_faces: int, sessions: int, records: int) -> List[Dict[str, Any]]:
    """The getting-started steps, each with whether it is done and where to do it."""
    return [
        {'title': "Add your students", 'done': students > 0, 'page': PAGE_STUDENTS,
         'how': "Student Registry → add one by one, or bulk import an Excel / CSV file."},
        {'title': "Add their face photos", 'done': students_with_faces > 0, 'page': PAGE_STUDENTS,
         'how': "Student Registry → take photos with the camera, or import a ZIP of photos named by student ID."},
        {'title': "Create a session", 'done': sessions > 0, 'page': PAGE_SESSIONS,
         'how': "Session Management → Create New Session. Give it a start time; it starts and closes on time."},
        {'title': "Take attendance", 'done': records > 0, 'page': PAGE_ATTENDANCE,
         'how': "Real-Time Attendance → pick the active session and start the camera."},
    ]


def _camera_session_id() -> Optional[int]:
    """Session the camera runs for, or None when it is off. Never opens the camera."""
    module = sys.modules.get(REALTIME_PAGE_MODULE)
    if module is None:  # camera page not opened yet in this server -> camera can't be on
        return None
    try:
        engine = module._get_engine()
        return engine.session_id if engine.is_running() else None
    except Exception as e:
        logger.debug(f"Camera status unavailable: {e}")
        return None


def system_checks(students_without_faces: int, pending_samples: int,
                  scheduler_running: bool, camera_session: Optional[int],
                  ai_ready: bool, missing_models: List[str]) -> List[Dict[str, str]]:
    """Rows of the system check: icon, item, result."""
    def row(ok: Optional[bool], item: str, result: str) -> Dict[str, str]:
        icon = "✅" if ok else ("ℹ️" if ok is None else "⚠️")
        return {'': icon, 'Item': item, 'Result': result}

    return [
        row(True, "Database", "Connected"),
        row(not missing_models, "Face models",
            "All found" if not missing_models else "Missing: " + ", ".join(missing_models)),
        row(scheduler_running, "Session timer",
            "Running - sessions start and close on time" if scheduler_running
            else "Not running - sessions start / close when a page is opened"),
        row(pending_samples == 0, "Recognition data",
            "Ready" if pending_samples == 0
            else f"{pending_samples} face sample(s) not prepared yet - done automatically when attendance starts"),
        row(students_without_faces == 0, "Face data",
            "Every active student has face data" if students_without_faces == 0
            else f"{students_without_faces} active student(s) have no face data and can't be recognised"),
        row(None, "Camera",
            f"On - session #{camera_session}" if camera_session
            else "Off - it turns on only during an active session"),
        row(ai_ready, "AI Assistant",
            "Ready" if ai_ready else "No API key - add OMNIROUTE_API_KEY to .env to use it"),
    ]


# -- page ----------------------------------------------------------------------

def render_home():
    """Render the Home page."""
    st.session_state.pop(NAV_PENDING_KEY, None)
    now = datetime.now()
    st.title("🏠 Home")
    st.caption(f"{now.strftime('%A, %d %B %Y')} · Intelligent Attendance Monitoring System")

    session_service = SessionService()
    stats_service = StatisticsService()
    student_service = StudentService()
    enrollment_service = EnrollmentService()

    try:
        stats = stats_service.get_overall_dashboard_stats()
        active_students = student_service.get_all_students(active_only=True)
        sample_counts = enrollment_service.get_all_samples_counts()
    except Exception as e:
        logger.error(f"Home page data failed to load: {e}")
        st.error(f"Could not load the dashboard data: {e}")
        return

    with_faces = sum(1 for s in active_students if sample_counts.get(s['id'], 0) > 0)
    without_faces = len(active_students) - with_faces

    _render_getting_started(stats, with_faces)
    _render_today()
    st.markdown("---")
    _render_overview(stats_service, session_service, stats, len(active_students), without_faces, now)
    st.markdown("---")
    _render_system_check(enrollment_service, without_faces)


def _render_getting_started(stats: Dict[str, Any], with_faces: int) -> None:
    steps = guide_steps(stats['total_students'], with_faces, stats['total_sessions'], stats['total_records'])
    done = sum(step['done'] for step in steps)
    all_done = done == len(steps)
    title = ("✅ Getting started - all steps done" if all_done
             else f"🚀 Getting started - {done} of {len(steps)} steps done")

    with st.expander(title, expanded=not all_done):
        st.progress(done / len(steps))
        next_step = next((i for i, step in enumerate(steps) if not step['done']), None)
        for i, step in enumerate(steps):
            col_text, col_button = st.columns([5, 1])
            mark = "✅" if step['done'] else ("👉" if i == next_step else "⬜")
            col_text.markdown(f"{mark} **{i + 1}. {step['title']}**  \n{step['how']}")
            if i == next_step:
                col_button.button("Go", key=f"home_guide_{i}", type="primary",
                                  on_click=_go, args=(step['page'],), width="stretch")
        st.caption("Then see the results in **Statistics & Reports**, or ask the **AI Assistant**.")


@st.fragment(run_every=TODAY_REFRESH_SECONDS)
def _render_today() -> None:
    """Today's sessions; refreshes by itself so the live counts stay current."""
    if st.session_state.pop(NAV_PENDING_KEY, False):
        st.rerun(scope="app")
    now = datetime.now()
    session_service = SessionService()
    stats_service = StatisticsService()
    try:
        sessions = session_service.get_sessions_overview()
    except Exception as e:
        st.error(f"Could not load today's sessions: {e}")
        return

    st.subheader("📅 Today")

    # Horizontal rows wrap on narrow screens instead of cutting text off
    quick = st.container(horizontal=True)
    quick.button("➕ New session", key="home_new_session", on_click=_go_create_session)
    quick.button("👤 Add students", key="home_students", on_click=_go, args=(PAGE_STUDENTS,))
    quick.button("📊 Reports", key="home_reports", on_click=_go, args=(PAGE_REPORTS,))
    quick.button("🤖 AI Assistant", key="home_ai", on_click=_go, args=(PAGE_ASSISTANT,))

    active = [s for s in sessions if s['status'] == SessionStatus.ACTIVE]
    for session in active:
        _render_active_session(stats_service, session)

    today = todays_sessions(sessions, now)
    if today:
        rows = [{
            'Time': _time_range(s),
            'Session': f"#{s['id']} {s['session_name']}",
            'Course': s.get('course_code') or '—',
            'Status': STATUS_LABELS.get(s['status'], s['status']),
            'Present': s.get('present_count', 0),
            'Late': s.get('late_count', 0),
        } for s in today]
        st.markdown("##### Today's sessions")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif not active:
        st.info("No sessions today.")

    upcoming = next_session(sessions, now)
    if upcoming:
        start = _parse(upcoming['start_time'])
        when = start.strftime('%H:%M today') if start.date() == now.date() else start.strftime('%a %d %b, %H:%M')
        col_text, col_button = st.columns([5, 1])
        col_text.markdown(f"⏭️ **Next session:** #{upcoming['id']} {upcoming['session_name']} "
                          f"({upcoming.get('course_code') or '—'}) - starts {when}, by itself.")
        col_button.button("View", key="home_next_view", on_click=_go_session_details,
                          args=(upcoming['id'],), width="stretch")
    st.caption(f"Updated {now.strftime('%H:%M:%S')} · refreshes every {TODAY_REFRESH_SECONDS} s")


def _render_active_session(stats_service: StatisticsService, session: Dict[str, Any]) -> None:
    stats = stats_service.get_session_statistics(session['id']) or {}
    with st.container(border=True):
        st.markdown(f"🟢 **Running now: #{session['id']} {session['session_name']}** · "
                    f"{session.get('course_code') or '—'} · {_time_range(session)}")
        numbers = st.container(horizontal=True, gap="medium")
        numbers.metric("Expected", _number(stats.get('expected_students')), width="content")
        numbers.metric("Present", _number(stats.get('present_count')), width="content")
        numbers.metric("Late", _number(stats.get('late_count')), width="content")
        numbers.metric("Absent so far", _number(stats.get('absent_count')), width="content")
        rate = stats.get('attendance_percentage')
        numbers.metric("Attendance", f"{rate}%" if rate is not None else "—", width="content")
        buttons = st.container(horizontal=True)
        buttons.button("📷 Open camera", key=f"home_camera_{session['id']}", type="primary",
                       on_click=_go_camera, args=(session['id'],))
        buttons.button("🔍 Details", key=f"home_details_{session['id']}",
                       on_click=_go_session_details, args=(session['id'],))


def _render_overview(stats_service: StatisticsService, session_service: SessionService,
                     stats: Dict[str, Any], active_students: int, without_faces: int,
                     now: datetime) -> None:
    st.subheader("📊 Overview")
    sessions = session_service.get_all_sessions()
    cards = st.container(horizontal=True)
    cards.metric("Active students", f"{active_students:,}", border=True)
    cards.metric("No face data", f"{without_faces:,}", border=True,
                 help="Active students who can't be recognised yet. Add photos in Student Registry.")
    cards.metric("Attendance rate", f"{stats['attendance_rate']}%", border=True,
                 help="Attended / expected, over closed sessions")
    cards.metric("Sessions (last 7 days)", sessions_in_last_days(sessions, now), border=True)

    st.markdown(f"##### Attendance - last {TREND_DAYS} days")
    trend = trend_last_days(stats_service.get_attendance_trend(), now)
    if trend[['Present', 'Late']].to_numpy().sum() > 0:
        st.bar_chart(trend, x='Day', y=['Present', 'Late'], color=["#22c55e", "#f59e0b"])
    else:
        st.caption(f"No attendance in the last {TREND_DAYS} days yet.")


def _render_system_check(enrollment_service: EnrollmentService, without_faces: int) -> None:
    st.subheader("🩺 System check")
    try:
        pending = enrollment_service.gallery_cache_status()['pending']
    except Exception as e:
        logger.debug(f"Gallery cache status unavailable: {e}")
        pending = 0
    missing = [name for name, _ in MODEL_FILES if not (config.PROJECT_ROOT / name).exists()]
    rows = system_checks(without_faces, pending, is_session_scheduler_running(),
                         _camera_session_id(), config.is_ai_assistant_configured(), missing)
    problems = sum(1 for r in rows if r[''] == "⚠️")
    if problems:
        st.warning(f"{problems} item(s) need attention.")
    else:
        st.success("Everything looks good.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _time_range(session: Dict[str, Any]) -> str:
    """'14:10 - 15:30'; with the date when it did not start today ('26 Oct 14:10')."""
    start = _parse(session.get('start_time'))
    end = _parse(session.get('end_time'))
    if not start:
        text = '—'
    elif start.date() == datetime.now().date():
        text = start.strftime('%H:%M')
    else:
        text = start.strftime('%d %b %H:%M')
    return f"{text} - {end.strftime('%H:%M')}" if end else text


def _number(value: Optional[int]) -> str:
    return "—" if value is None else f"{value:,}"
