"""
Intelligent Attendance Monitoring System
Main Streamlit application entry point

The old Home page was removed; a new one is planned for a later phase.
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


def main():
    """Main application entry point."""
    
    # Validate configuration on startup
    if not config.validate_config():
        st.error("⚠️ Configuration validation failed. Please check your setup.")
        logger.error("Configuration validation failed")
        return
    
    # Initialize database on first run
    if not config.DATABASE_PATH.exists():
        with st.spinner("Initializing database..."):
            if initialize_database():
                logger.info("Database initialized successfully")
            else:
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
            "Real-Time Attendance",
            "Session Management",
            "Student Registry",
            "Statistics & Reports",
            "AI Assistant"
        ],
        key="navigation"
    )
    
    # Render selected page
    if page == "Real-Time Attendance":
        render_realtime_attendance_page()
    elif page == "Session Management":
        render_session_management_page()
    elif page == "Student Registry":
        render_student_registry_page()
    elif page == "Statistics & Reports":
        render_statistics_reports_page()
    elif page == "AI Assistant":
        render_ai_assistant_page()


def render_student_registry_page():
    """Render the Student Registry page."""
    from src.ui.pages.student_registry import render_student_registry
    render_student_registry()


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
