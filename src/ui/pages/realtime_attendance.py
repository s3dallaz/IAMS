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
from src.computer_vision.detection_diagnostics import get_diagnostics
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
from src.services.enrollment_service import EnrollmentService
from src.utils.logger import get_logger
from src.utils.performance_timer import time_operation, log_performance_summary

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

        # Configurable head-turn angle for liveness challenges
        head_turn_angle = st.select_slider(
            "Liveness Head-Turn Angle (°)",
            options=[5, 10, 15, 20, 25, 30, 35],
            value=15,
            help="Required head turn angle for liveness challenges. Lower angles are easier, higher angles more secure."
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
            if st.button("▶️ Activate Session", type="primary", width="stretch"):
                try:
                    session_service.activate_session(selected_session_id)
                    st.success(f"Session #{selected_session_id} activated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to activate session: {e}")
        elif is_active:
            if st.button("⏹️ Close Session", width="stretch"):
                try:
                    session_service.close_session(selected_session_id)
                    if st.session_state.att_camera_manager is not None:
                        st.session_state.att_camera_manager.release()
                        st.session_state.att_camera_manager = None
                    st.session_state.att_camera_active = False
                    st.session_state.att_rt_manager = None
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
                width="stretch",
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
            if st.button("⏹️ Stop Attendance Stream", width="stretch"):
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
            student_service=student_service,
            head_turn_angle=head_turn_angle
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
    student_service: StudentService,
    head_turn_angle: float
):
    """
    Run the live camera feed and real-time attendance processing loop.
    Uses st.empty() updates for smooth, responsive real-time streaming.

    Optimizations to reduce per-frame overhead:
    - Database queries throttled (every 30 frames, not every frame)
    - Widgets pre-created and content updated via markdown/image only
    - Minimal Streamlit API calls inside the frame loop
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
            marked_by="cv_realtime",
            head_turn_angle=head_turn_angle
        )
        enrollment_service = EnrollmentService()
        gallery = enrollment_service.load_active_gallery()
        rt_manager.set_gallery(gallery)
        st.session_state.att_rt_manager = rt_manager
    else:
        rt_manager = st.session_state.att_rt_manager
        rt_manager.require_liveness = require_liveness
        rt_manager.cooldown_seconds = cooldown_seconds
        rt_manager.head_turn_angle = head_turn_angle
        # Reload the gallery on every page run (about every 30 s while
        # streaming) so students enrolled or deactivated during an active
        # session are picked up without restarting it. Loading takes only
        # a few milliseconds.
        rt_manager.set_gallery(EnrollmentService().load_active_gallery())

    # Layout placeholders
    col_video, col_live_status = st.columns([3, 2])

    with col_video:
        video_placeholder = st.empty()

    with col_live_status:
        st.markdown("### 📊 Live Session Stats")
        metrics_placeholder = st.empty()
        st.markdown("### 👤 Last Detected Person")
        person_info_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### 📋 Marked Students in this Session")
    records_table_placeholder = st.empty()

    frame_count = 0
    max_frames_per_run = 900  # Stream batch frames per execution cycle (~30 seconds at 30 fps)
    # BUG FIX: Increased from 300 to 900 frames to reduce page refresh frequency
    # from every 10 seconds to every 30 seconds, making refreshes less disruptive

    # Cache for database queries (refreshed periodically, not every frame)
    cached_records = []
    cached_present_cnt = 0
    cached_late_cnt = 0
    cached_total_cnt = 0

    # Cache for CV processing results (to avoid processing every frame)
    # The key bottleneck: face detection/recognition/liveness is expensive
    # We process every 5th frame but display all frames for smooth preview
    last_cv_result: Optional[RealTimeAttendanceResult] = None
    cv_process_interval = 5  # Process CV every 5 frames = 6fps processing at 30fps preview

    # Performance measurement counters
    frame_times = []
    cv_process_times = []
    performance_log_interval = 150  # Log performance every 150 frames (~5 seconds)

    try:
        while frame_count < max_frames_per_run:
            if not st.session_state.att_camera_active:
                break

            frame_start_time = time.perf_counter()

            with time_operation("camera_read_frame"):
                success, frame = camera.read_frame()
            if not success or frame is None:
                video_placeholder.warning("⚠️ Frame read timeout — waiting for camera...")
                time.sleep(0.05)
                continue

            # KEY FIX: Only run CV processing every Nth frame to reduce lag
            # The expensive operations (face detection, embedding, recognition, liveness)
            # run at ~6 fps while the camera preview stays smooth at 30 fps
            if frame_count % cv_process_interval == 0:
                # Process frame through RealTimeAttendanceManager
                cv_start_time = time.perf_counter()
                with time_operation("cv_frame_processing"):
                    result: RealTimeAttendanceResult = rt_manager.process_frame(frame, session_id=session_id)
                cv_process_time = (time.perf_counter() - cv_start_time) * 1000.0
                cv_process_times.append(cv_process_time)
                last_cv_result = result
            else:
                # Reuse previous CV result for smooth preview with stale detection boxes
                # This maintains 30fps display while CV runs at lower rate
                result = last_cv_result if last_cv_result is not None else RealTimeAttendanceResult(
                    session_id=session_id,
                    faces_detected=0,
                    processed_faces=[],
                    session_active=True,
                    marked_count=0
                )

            # Draw overlays on display frame (uses cached result between CV updates)
            with time_operation("frame_annotation"):
                display_frame = _draw_attendance_overlays(frame, result)

                # Diagnostics annotations on top of the mirrored preview.
                # Returns the input unchanged unless FACE_DIAGNOSTICS_ENABLED=true.
                display_frame = get_diagnostics().overlay(display_frame, _result_faces(result))

            # Update video placeholder (only the image, no widget creation)
            with time_operation("streamlit_image_update"):
                video_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch"
                )

            # Refresh database cache periodically (every 30 frames = once per second at 30fps)
            # This reduces database queries from 300/run to 10/run
            if frame_count % 30 == 0:
                cached_records = attendance_service.get_session_attendance(session_id)
                cached_present_cnt = sum(1 for r in cached_records if r['status'] == AttendanceStatus.PRESENT)
                cached_late_cnt = sum(1 for r in cached_records if r['status'] == AttendanceStatus.LATE)
                cached_total_cnt = len(cached_records)

                # Update metrics using pre-formatted markdown (no st.columns creation per frame)
                metrics_placeholder.markdown(
                    f"**Total:** {cached_total_cnt} | **Present:** {cached_present_cnt} | **Late:** {cached_late_cnt}"
                )

            # Update Last Detected Person Info (build markdown string, single update)
            person_info_markdown = _build_person_info_markdown(result)
            person_info_placeholder.markdown(person_info_markdown, unsafe_allow_html=False)

            # Update Records Table (every 30 frames to match database refresh rate)
            if frame_count % 30 == 0:
                _update_live_records_table(records_table_placeholder, cached_records, student_service)

            # Record frame processing time
            frame_end_time = time.perf_counter()
            frame_time = (frame_end_time - frame_start_time) * 1000.0
            frame_times.append(frame_time)

            # Log performance statistics periodically
            if frame_count > 0 and frame_count % performance_log_interval == 0:
                if frame_times:
                    avg_frame_time = sum(frame_times[-performance_log_interval:]) / min(len(frame_times), performance_log_interval)
                    effective_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0

                if cv_process_times:
                    avg_cv_time = sum(cv_process_times[-30:]) / min(len(cv_process_times), 30)  # Last 30 CV frames

                logger.info(f"PERFORMANCE: Frame {frame_count}, "
                           f"Avg frame time: {avg_frame_time:.1f}ms, "
                           f"Effective FPS: {effective_fps:.1f}, "
                           f"Avg CV time: {avg_cv_time:.1f}ms")

                # Log detailed performance summary every 5 seconds
                log_performance_summary()

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


def _build_person_info_markdown(result: RealTimeAttendanceResult) -> str:
    """
    Build a single markdown string for person info display.

    Reduces per-frame Streamlit API calls from 6-8 widget creations
    to a single markdown update.
    """
    if not result.processed_faces:
        return "_No face currently detected in frame._"

    face_info = result.processed_faces[0]

    if face_info.is_recognized:
        name = face_info.student_name or '—'
        student_id = face_info.student_id or '—'

        # Build liveness status with instructions
        if face_info.liveness_state == LivenessState.LIVE:
            liveness_display = "🟢 **Live** — Verified"
        else:
            # Show actionable instructions based on actual detector state
            blink_status = f"{face_info.blink_count}/{face_info.min_blinks_required} blinks"

            challenge_instruction = ""
            if face_info.challenge_type != "none":
                if face_info.challenge_state == "completed":
                    challenge_instruction = "✅ Challenge completed"
                elif face_info.challenge_state == "active":
                    if face_info.challenge_type == "turn_left":
                        challenge_instruction = f"👈 **Turn your head LEFT** ({int(face_info.challenge_progress * 100)}%)"
                    elif face_info.challenge_type == "turn_right":
                        challenge_instruction = f"👉 **Turn your head RIGHT** ({int(face_info.challenge_progress * 100)}%)"
                    else:
                        challenge_instruction = f"Challenge: {face_info.challenge_type}"
                elif face_info.challenge_state == "inactive":
                    challenge_instruction = "⏳ Initializing challenge..."
                else:
                    challenge_instruction = f"Challenge: {face_info.challenge_state}"

            # Determine what the user needs to do
            needs_blinks = face_info.blink_count < face_info.min_blinks_required
            needs_challenge = (face_info.challenge_type != "none" and
                             face_info.challenge_state != "completed")

            if needs_blinks and needs_challenge:
                instruction = f"🔵 **Blink naturally** ({blink_status}) and {challenge_instruction}"
            elif needs_blinks:
                instruction = f"🔵 **Blink naturally** to verify liveness ({blink_status})"
            elif needs_challenge:
                instruction = f"🔵 {challenge_instruction}"
            else:
                instruction = "🟡 **Verifying...** Almost there!"

            liveness_display = instruction

        lines = [
            f"**Name:** {name}",
            f"**ID:** `{student_id}`",
            f"**Liveness:** {liveness_display}",
        ]

        if face_info.attendance_marked:
            lines.append(f"✅ **Attendance Marked: {face_info.attendance_status.upper()}**")
        elif face_info.already_marked:
            lines.append("ℹ️ _Already Marked for this session_")
        else:
            lines.append(f"⚠️ _{face_info.reason}_")

        return "\n\n".join(lines)
    else:
        conf = face_info.detection_confidence
        # Check if we can provide enrollment guidance
        enrollment_guidance = ""
        if conf >= 0.7:  # Good detection quality suggests enrollable face
            enrollment_guidance = "\n\n💡 _This appears to be a clear face detection. Consider enrolling this person in the Student Registry if they should have access._"
        
        return (
            f"**Identity:** :red[**Unknown Person**]\n\n"
            f"**Confidence:** {conf:.2f}\n\n"
            f"_No attendance record created for unknown face._"
            f"{enrollment_guidance}"
        )


def _draw_attendance_overlays(
    frame: np.ndarray,
    result: RealTimeAttendanceResult
) -> np.ndarray:
    """
    Draw attendance status text overlays on video frame.

    No rectangle is drawn around the face; the bounding box is only used to
    position the name and liveness labels next to it.

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


def _result_faces(result) -> List[DetectedFace]:
    """
    Convert a RealTimeAttendanceResult into DetectedFace objects.

    Lets the diagnostics overlay reuse the existing bbox/confidence values.
    Returns an empty list when nothing was detected.
    """
    faces: List[DetectedFace] = []
    for face in getattr(result, "processed_faces", []) or []:
        faces.append(DetectedFace(tuple(face.bbox), float(face.detection_confidence)))
    return faces


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
        width="stretch",
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
        if st.button("🔄 Refresh Table", key="refresh_idle_records", width="stretch"):
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
        width="stretch",
        hide_index=True
    )
