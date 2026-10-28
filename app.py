"""
Intelligent Attendance Monitoring System
Main Streamlit application entry point

The app opens on the Home dashboard (src/ui/pages/home.py).
"""

import streamlit as st
from dotenv import load_dotenv

import config
from src.utils.logger import get_logger
from src.database.schema import initialize_database
from src.services.session_scheduler import start_session_scheduler

# Load environment variables
load_dotenv()

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

    logger.info("Application started successfully")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Real-Time Attendance",
            "Session Management",
            "Student Registry",
            "Courses & Sections",
            "Statistics & Reports",
            "AI Assistant"
        ],
        key="navigation"
    )
    
    # Render selected page
    if page == "Home":
        render_home_page()
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


def render_home_page():
    """Render the Home page."""
    from src.ui.pages.home import render_home
    render_home()


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


if __name__ == "__main__":
    main()
