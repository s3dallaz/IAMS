"""
Intelligent Attendance Monitoring System
Main Streamlit application entry point

Everyone signs in first (src/ui/pages/login.py); the sidebar then shows the
pages their role may open (src/ui/access.py), starting with the Home dashboard.
"""

import streamlit as st
from dotenv import load_dotenv

import config
from src.utils.logger import get_logger
from src.database.schema import initialize_database
from src.services.activity_log import set_actor_provider
from src.services.auth_service import AuthService
from src.services.course_service import CourseService
from src.services.session_scheduler import start_session_scheduler
from src.ui.access import pages_for
from src.ui.pages.login import (
    acting_user, has_pending_recovery_code, load_signed_in_user, render_account_box, render_first_setup,
    render_forced_password_change, render_login, render_recovery_code_screen,
)

# Load environment variables
load_dotenv()

# The activity log records changes under the signed-in user of the page making them
set_actor_provider(acting_user)

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Preparing the database...")
def prepare_database() -> bool:
    """initialize_database() once per server process (it is safe to repeat)."""
    ok = initialize_database()
    if ok:
        logger.info("Database ready")
    return ok


def main():
    """Main application entry point."""
    
    # Validate configuration on startup
    if not config.validate_config():
        st.error("⚠️ Configuration validation failed. Please check your setup.")
        logger.error("Configuration validation failed")
        return
    
    # Create the database, or add what a newer version needs (new tables /
    # columns) to an existing one - once per server start
    if not prepare_database():
        prepare_database.clear()  # try again on the next page load
        st.error("⚠️ Database initialization failed.")
        logger.error("Database initialization failed")
        return
    
    # Start / close scheduled sessions on time (no-op when already running,
    # e.g. started by run_app.py before the server)
    start_session_scheduler()

    # Sign in first: nothing below is shown without an account
    auth = AuthService()
    if auth.needs_first_account():
        render_first_setup(auth)
        return
    user = load_signed_in_user(auth)
    if user is None:
        render_login(auth)
        return
    if has_pending_recovery_code():   # a new recovery code is shown once, before anything else
        render_recovery_code_screen(user)
        return
    if user['must_change_password']:
        render_forced_password_change(auth, user)
        return

    # Sidebar navigation: only the pages this user's role may open. A page
    # chosen elsewhere (e.g. a Home button) that they may not open falls back to the first.
    teaches = bool(CourseService().instructor_sections(user['id']))   # Admin / IT who also teach get My Classes
    pages = pages_for(user, teaches=teaches)
    if st.session_state.get("navigation") not in pages:
        st.session_state["navigation"] = pages[0]
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", pages, key="navigation")
    render_account_box(auth, user)

    # Render selected page
    if page == "Home":
        render_home_page()
    elif page == "My Classes":
        render_my_classes_page()
    elif page == "Real-Time Attendance":
        render_realtime_attendance_page()
    elif page == "Session Management":
        render_session_management_page()
    elif page == "Student Registry":
        render_student_registry_page()
    elif page == "Courses & Sections":
        render_courses_sections_page()
    elif page == "Statistics & Reports":
        render_statistics_reports_page()
    elif page == "AI Assistant":
        render_ai_assistant_page()
    elif page == "Users":
        render_users_page()
    elif page == "Activity Log":
        render_activity_log_page()


def render_activity_log_page():
    """Render the Activity Log page."""
    from src.ui.pages.activity_log import render_activity_log
    render_activity_log()


def render_home_page():
    """Render the Home page."""
    from src.ui.pages.home import render_home
    render_home()


def render_my_classes_page():
    """Render the My Classes page."""
    from src.ui.pages.my_classes import render_my_classes
    render_my_classes()


def render_student_registry_page():
    """Render the Student Registry page."""
    from src.ui.pages.student_registry import render_student_registry
    render_student_registry()


def render_courses_sections_page():
    """Render the Courses & Sections page."""
    from src.ui.pages.courses_sections import render_courses_sections
    render_courses_sections()


def render_session_management_page():
    """Render the Session Management page."""
    from src.ui.pages.session_management import render_session_management
    render_session_management()


def render_realtime_attendance_page():
    """Render the Real-Time Attendance page."""
    from src.ui.pages.realtime_attendance import render_realtime_attendance
    render_realtime_attendance()


def render_statistics_reports_page():
    """Render the Statistics & Reports page."""
    from src.ui.pages.statistics_reports import render_statistics_reports_page
    render_statistics_reports_page()


def render_ai_assistant_page():
    """Render the AI Assistant page."""
    from src.ui.pages.ai_assistant import render_ai_assistant
    render_ai_assistant()


def render_users_page():
    """Render the Users page."""
    from src.ui.pages.users import render_users
    render_users()


if __name__ == "__main__":
    main()
