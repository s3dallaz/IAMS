"""
Intelligent Attendance Monitoring System
Main Streamlit application entry point

Phase 1E: Student Registry UI integrated
"""

import streamlit as st
from dotenv import load_dotenv

import config
from src.utils.logger import get_logger
from src.database.schema import initialize_database

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
    
    logger.info("Application started successfully")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Student Registry", "Liveness Test"],
        key="navigation"
    )
    
    # Render selected page
    if page == "Home":
        render_home_page()
    elif page == "Student Registry":
        render_student_registry_page()
    elif page == "Liveness Test":
        render_liveness_test_page()


def render_home_page():
    """Render the home/welcome page."""
    # Header
    st.title("🎓 " + config.APP_NAME)
    st.caption(f"Version {config.APP_VERSION}")
    
    # Welcome message
    st.markdown("---")
    st.success("✅ **Phase 1 Foundation Complete**")
    
    st.markdown("""
    ### Welcome to the Intelligent Attendance Monitoring System
    
    This is a computer vision-based attendance system with face recognition,
    liveness detection, and AI-powered assistant capabilities.
    
    #### System Status
    """)
    
    # Display system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Configuration", "✓ Valid", delta="Ready")
    
    with col2:
        st.metric("Database", "✓ Initialized", delta="Ready")
    
    with col3:
        st.metric("Application", "✓ Running", delta="Phase 1E")
    
    st.markdown("---")
    
    # Information about current phase
    st.markdown("""
    #### 📋 Phase 1 Foundation Complete
    
    The project foundation has been successfully established:
    
    - ✅ **Phase 1A**: Project structure and configuration
    - ✅ **Phase 1B**: Database schema and connection layer
    - ✅ **Phase 1C**: Student repository (data access)
    - ✅ **Phase 1D**: Student service (business logic)
    - ✅ **Phase 1E**: Student Registry UI
    
    #### 🚀 Available Features
    
    **Student Registry** (available in sidebar):
    - View all students
    - Search by ID or name
    - Filter by active/inactive status
    - Add new students
    - Edit student information
    - Activate/deactivate students
    
    #### 🔮 Future Phases
    
    **Phase 2** will add:
    - Face detection and recognition
    - Face enrollment for students
    
    **Phase 3** will add:
    - Liveness detection (anti-spoofing)
    
    **Phase 4** will add:
    - Attendance logic and session management
    
    **Phase 5** will add:
    - Statistics and reports
    
    **Phase 6** will add:
    - AI assistant integration
    """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("Quick Start Guide")
    
    st.markdown("""
    1. **Navigate to Student Registry** using the sidebar
    2. **Add students** using the form at the bottom
    3. **Search and filter** students using the controls at the top
    4. **Edit or manage** student status using the action buttons
    """)
    
    st.markdown("---")
    
    # Debug information (if enabled)
    if config.APP_DEBUG:
        with st.expander("🔧 Debug Information"):
            st.code(f"""
Project Root: {config.PROJECT_ROOT}
Database Path: {config.DATABASE_PATH}
Data Directory: {config.DATA_DIR}
Debug Mode: {config.APP_DEBUG}
Log Level: {config.LOG_LEVEL}
            """)
    
    # Footer
    st.markdown("---")
    st.caption("Intelligent Attendance Monitoring System - Graduation Project")


def render_student_registry_page():
    """Render the Student Registry page."""
    from src.ui.pages.student_registry import render_student_registry
    render_student_registry()


def render_liveness_test_page():
    """Render the Liveness Test page."""
    from src.ui.pages.liveness_test import render_liveness_test
    render_liveness_test()


if __name__ == "__main__":
    main()
