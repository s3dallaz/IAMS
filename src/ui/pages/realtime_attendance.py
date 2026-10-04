"""
Real-Time Attendance UI Page for the Intelligent Attendance Monitoring System.

Phase 4D: Attendance UI & Management
Provides Streamlit interface for taking attendance in real-time via camera feed.
Integrates CameraManager, RealTimeAttendanceManager, FaceDetector, FaceEmbedder,
FaceRecognizer, LivenessDetector, SessionService, and AttendanceService.
Runs continuous frame processing loop with st.empty() without full script rerun per frame.
"""

import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding
from src.computer_vision.face_recognizer import FaceRecognizer
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessState,
    ChallengeState,
    ChallengeType
)
from src.services.real_time_attendance_service import (
    RealTimeAttendanceManager,
    RealTimeAttendanceResult,
    FaceProcessingResult
)
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_realtime_attendance():
    """Main function to render the Real-Time Attendance page."""
    st.title("📹 Real-Time Attendance Monitoring")
    st.caption("Phase 4D: Live Camera Attendance with Face Recognition & Anti-Spoofing")
    st.markdown("Select an active session, start the camera feed, and monitor real-time attendance marking.")
    st.markdown("---")
    
    # Initialize services
    session_service = SessionService()
    attendance_service = AttendanceService()
    student_service = StudentService()
    
    # Sidebar Configuration
    with st.sidebar:
        st.subheader("⚙️ Attendance Settings")
        
        require_liveness = st.checkbox(
            "Require Liveness Check",
            value=True,
            help="Enforce anti-spoofing; only live verified faces will be marked"
        )
        
        cooldown_seconds = st.slider(
            "Debounce Cooldown (Seconds)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Minimum seconds between re-evaluating the same student"
        )
        
        st.markdown("---")
        st.subheader("ℹ️ Live Instructions")
        st.markdown("""
        1. **Select an Active Session** (or activate a scheduled one).
        2. Click **Start Attendance Stream**.
        3. Position students in front of the camera.
        4. Recognized + Live students will be **marked present/late automatically**.
        5. The live table updates immediately.
        6. Click **Stop Attendance Stream** when finished.
        """)
        
    # Get active and scheduled sessions
    all_sessions = session_service.get_all_sessions()
    active_sessions = [s for s in all_sessions if s['status'] == SessionStatus.ACTIVE]
    scheduled_sessions = [s for s in all_sessions if s['status'] == SessionStatus.SCHEDULED]
    
    if not all_sessions:
        st.warning("⚠️ No attendance sessions exist. Please create a session first in **Session Management**.")
        return
        
    # Session Selector Section
    st.subheader("🎯 Target Session")
    
    col_select, col_activate = st.columns([3, 1])
    
    with col_select:
        if active_sessions:
            session_options = {s['id']: f"🟢 #{s['id']} - {s['session_name']} (Active)" for s in active_sessions}
            # Also allow choosing scheduled sessions to activate
            for s in scheduled_sessions:
                session_options[s['id']] = f"🟡 #{s['id']} - {s['session_name']} (Scheduled)"
        else:
            session_options = {s['id']: f"🟡 #{s['id']} - {s['session_name']} (Scheduled)" for s in scheduled_sessions}
            for s in all_sessions:
                if s['id'] not in session_options:
                    session_options[s['id']] = f"🔴 #{s['id']} - {s['session_name']} (Closed)"
                    
        selected_session_id = st.selectbox(
            "Select Session for Attendance",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            key="realtime_session_selector"
        )
        
    selected_session = session_service.get_session(selected_session_id)
    is_active = selected_session and selected_session['status'] == SessionStatus.ACTIVE
    
    with col_activate:
        st.write("")
        st.write("")
        if not is_active and selected_session and selected_session['status'] == SessionStatus.SCHEDULED:
            if st.button("▶️ Activate Session", type="primary", use_container_width=True):
                try:
                    session_service.activate_session(selected_session_id)
                    st.success(f"Session #{selected_session_id} activated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to activate session: {e}")
        elif is_active:
            if st.button("⏹️ Close Session", use_container_width=True):
                try:
                    session_service.close_session(selected_session_id)
                    st.success(f"Session #{selected_session_id} closed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to close session: {e}")
                    
    # Display selected session banner
    if selected_session:
        status_color = "green" if is_active else ("orange" if selected_session['status'] == SessionStatus.SCHEDULED else "red")
        st.info(
            f"**Selected Session:** #{selected_session['id']} — **{selected_session['session_name']}** | "
            f"Course: `{selected_session['course_code'] or '—'}` | "
            f"Status: :{status_color}[**{selected_session['status'].upper()}**] | "
            f"Late Cutoff: `{selected_session.get('late_threshold_minutes', 15)} min`"
        )
        
    st.markdown("---")
    
    # Initialize Streamlit Session State for Camera & Detectors
    if 'att_camera_active' not in st.session_state:
        st.session_state.att_camera_active = False
    if 'att_camera_manager' not in st.session_state:
        st.session_state.att_camera_manager = None
    if 'att_rt_manager' not in st.session_state:
        st.session_state.att_rt_manager = None
        
    # Camera Control Buttons
    col_cam1, col_cam2 = st.columns(2)
    
    with col_cam1:
        if not st.session_state.att_camera_active:
            start_disabled = not is_active
            start_help = "Session must be ACTIVE before starting attendance" if start_disabled else None
            if st.button(
                "📹 Start Attendance Stream",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
                help=start_help
            ):
                if st.session_state.att_camera_manager is None:
                    st.session_state.att_camera_manager = CameraManager(camera_index=0)
                    
                if st.session_state.att_camera_manager.open():
                    st.session_state.att_camera_active = True
                    logger.info(f"Attendance stream started for session #{selected_session_id}")
                    st.rerun()
                else:
                    st.error("❌ Failed to open camera device. Please check hardware connection.")
                    st.session_state.att_camera_manager = None
                    
    with col_cam2:
        if st.session_state.att_camera_active:
            if st.button("⏹️ Stop Attendance Stream", use_container_width=True):
                st.session_state.att_camera_active = False
                if st.session_state.att_camera_manager is not None:
                    st.session_state.att_camera_manager.release()
                    st.session_state.att_camera_manager = None
                logger.info("Attendance stream stopped")
                st.rerun()
                
    st.markdown("---")
    
    # Render Live Stream & Records
    if st.session_state.att_camera_active and is_active:
        _run_realtime_attendance_stream(
            session_id=selected_session_id,
            require_liveness=require_liveness,
            cooldown_seconds=cooldown_seconds,
            session_service=session_service,
            attendance_service=attendance_service,
            student_service=student_service
        )
    else:
        # Show Current Session Attendance Records Table when stream is idle
        render_session_records_panel(selected_session_id, attendance_service, student_service)


def _run_realtime_attendance_stream(
    session_id: int,
    require_liveness: bool,
    cooldown_seconds: float,
    session_service: SessionService,
    attendance_service: AttendanceService,
    student_service: StudentService
):
    """
    Run the live camera feed and real-time attendance processing loop.
    Uses st.empty() updates for smooth, responsive real-time streaming.
    """
    camera = st.session_state.att_camera_manager
    if camera is None or not camera.is_open():
        st.error("Camera is not open.")
        return
        
    # Initialize RealTimeAttendanceManager in session state if needed
    if st.session_state.att_rt_manager is None or st.session_state.att_rt_manager.current_session_id != session_id:
        rt_manager = RealTimeAttendanceManager(
            session_id=session_id,
            attendance_service=attendance_service,
            session_service=session_service,
            student_service=student_service,
            require_liveness=require_liveness,
            cooldown_seconds=cooldown_seconds,
            marked_by="cv_realtime"
        )
        st.session_state.att_rt_manager = rt_manager
    else:
        rt_manager = st.session_state.att_rt_manager
        rt_manager.require_liveness = require_liveness
        rt_manager.cooldown_seconds = cooldown_seconds
        
    # Layout placeholders
    col_video, col_live_status = st.columns([3, 2])
    
    with col_video:
        video_placeholder = st.empty()
        live_alert_placeholder = st.empty()
        
    with col_live_status:
        st.markdown("### 📊 Live Session Stats")
        metrics_placeholder = st.empty()
        st.markdown("### 👤 Last Detected Person")
        person_info_placeholder = st.empty()
        
    st.markdown("---")
    st.markdown("### 📋 Marked Students in this Session")
    records_table_placeholder = st.empty()
    
    frame_count = 0
    max_frames_per_run = 300  # Stream batch frames per execution cycle (~10 seconds at 30 fps)
    
    try:
        while frame_count < max_frames_per_run:
            if not st.session_state.att_camera_active:
                break
                
            success, frame = camera.read_frame()
            if not success or frame is None:
                video_placeholder.warning("⚠️ Frame read timeout — waiting for camera...")
                time.sleep(0.05)
                continue
                
            # Process frame through RealTimeAttendanceManager
            result: RealTimeAttendanceResult = rt_manager.process_frame(frame, session_id=session_id)
            
            # Draw overlays on display frame
            display_frame = _draw_attendance_overlays(frame, result)
            
            # Update video placeholder
            video_placeholder.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )
            
            # Update Metrics Placeholder
            records = attendance_service.get_session_attendance(session_id)
            present_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.PRESENT)
            late_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.LATE)
            total_cnt = len(records)
            
            with metrics_placeholder.container():
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Total Marked", total_cnt)
                m_col2.metric("Present", present_cnt)
                m_col3.metric("Late", late_cnt)
                
            # Update Last Detected Person Info
            if result.processed_faces:
                face_info = result.processed_faces[0]
                with person_info_placeholder.container():
                    if face_info.is_recognized:
                        st.markdown(f"**Name:** {face_info.student_name or '—'}")
                        st.markdown(f"**ID:** `{face_info.student_id}`")
                        liveness_badge = "🟢 Live" if face_info.liveness_state == LivenessState.LIVE else "🟡 Pending"
                        st.markdown(f"**Liveness:** {liveness_badge}")
                        if face_info.attendance_marked:
                            st.success(f"✅ Attendance Marked: **{face_info.attendance_status.upper()}**")
                        elif face_info.already_marked:
                            st.info("ℹ️ Already Marked for this session")
                        else:
                            st.warning(f"⚠️ {face_info.reason}")
                    else:
                        st.markdown("**Identity:** :red[**Unknown Person**]")
                        st.markdown(f"**Confidence:** {face_info.detection_confidence:.2f}")
                        st.info("No attendance record created for unknown face.")
            else:
                with person_info_placeholder.container():
                    st.caption("No face currently detected in frame.")
                    
            # Update Records Table (periodically every 10 frames to avoid overhead)
            if frame_count % 10 == 0:
                _update_live_records_table(records_table_placeholder, records, student_service)
                
            time.sleep(0.033)
            frame_count += 1
            
        # Re-trigger run to continue streaming smoothly
        if st.session_state.att_camera_active:
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error in real-time attendance loop: {e}", exc_info=True)
        video_placeholder.error(f"❌ Real-time error: {e}")
        if st.session_state.att_camera_active:
            time.sleep(0.2)
            st.rerun()


def _draw_attendance_overlays(
    frame: np.ndarray,
    result: RealTimeAttendanceResult
) -> np.ndarray:
    """
    Draw bounding boxes and attendance status text overlays on video frame.
    
    Applies horizontal mirror (cv2.flip(frame, 1)) to the display preview
    so natural user movements visually match a front-facing camera/mirror,
    while correctly translating bounding box coordinates: mirrored_x = width - (x + w).
    """
    # Flip the displayed frame horizontally (left <-> right)
    display_frame = cv2.flip(frame, 1)
    frame_w = display_frame.shape[1]
    
    for face in result.processed_faces:
        x, y, w, h = face.bbox
        
        # Calculate mirrored bounding box x coordinate
        mirrored_x = frame_w - (x + w)
        
        # Color coding
        if face.attendance_marked or face.already_marked:
            color = (0, 255, 0)  # Green for marked/present
        elif face.is_recognized and face.liveness_state != LivenessState.LIVE:
            color = (0, 255, 255)  # Yellow for recognized but pending liveness
        else:
            color = (0, 0, 255)  # Red for unknown
            
        # Draw bounding box on mirrored frame
        cv2.rectangle(display_frame, (mirrored_x, y), (mirrored_x + w, y + h), color, 2)
        
        # Header text
        if face.is_recognized:
            display_name = face.student_name or face.student_id or "Recognized"
            label = f"{display_name}"
            if face.attendance_status:
                label += f" ({face.attendance_status})"
            elif face.already_marked:
                label += " (Marked)"
        else:
            label = "Unknown"
            
        cv2.putText(
            display_frame,
            label,
            (mirrored_x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Footer text: Liveness status
        liveness_text = f"Live: {face.liveness_state.value}"
        cv2.putText(
            display_frame,
            liveness_text,
            (mirrored_x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
    return display_frame


def _update_live_records_table(
    placeholder,
    records: List[Dict[str, Any]],
    student_service: StudentService
):
    """Update live attendance records table."""
    if not records:
        placeholder.info("No attendance recorded yet.")
        return
        
    table_data = []
    for r in reversed(records):  # Show newest first
        student = student_service.get_student_by_id(r['student_id'])
        student_id_str = student['student_id'] if student else f"DB#{r['student_id']}"
        student_name_str = student['name'] if student else "Unknown"
        
        table_data.append({
            'Student ID': student_id_str,
            'Name': student_name_str,
            'Status': "✅ Present" if r['status'] == AttendanceStatus.PRESENT else "⏰ Late",
            'Marked At': r['marked_at'],
            'Confidence': f"{r['detection_confidence']:.2f}" if r.get('detection_confidence') is not None else "—",
            'Liveness': f"{r['liveness_score']:.2f}" if r.get('liveness_score') is not None else "—",
            'Marked By': r.get('marked_by') or 'system'
        })
        
    placeholder.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


def render_session_records_panel(
    session_id: Optional[int],
    attendance_service: AttendanceService,
    student_service: StudentService
):
    """Render attendance records panel when camera is idle."""
    if not session_id:
        return
        
    records = attendance_service.get_session_attendance(session_id)
    
    st.markdown("### 📋 Session Attendance Records")
    col_m1, col_m2, col_m3, col_ref = st.columns([1, 1, 1, 1])
    
    present_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.PRESENT)
    late_cnt = sum(1 for r in records if r['status'] == AttendanceStatus.LATE)
    
    with col_m1:
        st.metric("Total Marked", len(records))
    with col_m2:
        st.metric("Present", present_cnt)
    with col_m3:
        st.metric("Late", late_cnt)
    with col_ref:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Table", key="refresh_idle_records", use_container_width=True):
            st.rerun()
            
    if not records:
        st.info("No attendance records for this session yet.")
        return
        
    table_data = []
    for r in records:
        student = student_service.get_student_by_id(r['student_id'])
        student_id_str = student['student_id'] if student else f"DB#{r['student_id']}"
        student_name_str = student['name'] if student else "Unknown"
        
        table_data.append({
            'Record ID': r['id'],
            'Student ID': student_id_str,
            'Name': student_name_str,
            'Status': "✅ Present" if r['status'] == AttendanceStatus.PRESENT else "⏰ Late",
            'Marked At': r['marked_at'],
            'Confidence': f"{r['detection_confidence']:.2f}" if r.get('detection_confidence') is not None else "—",
            'Liveness': f"{r['liveness_score']:.2f}" if r.get('liveness_score') is not None else "—",
            'Marked By': r.get('marked_by') or 'system'
        })
        
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )
