"""
Real-Time Attendance UI Page for the Intelligent Attendance Monitoring System.

Phase 4D: Attendance UI & Management
Provides Streamlit interface for taking attendance in real-time via camera feed.

The camera and the whole attendance pipeline run in a separate vision engine
process (src/realtime/engine.py). The page shows the engine's MJPEG video
stream in an <img> tag and refreshes the status panel a few times per second,
so Streamlit never handles individual video frames.
"""

import math

import pandas as pd
import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import config
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
    SPOOF_RECHECK_PREFIX,
    RealTimeAttendanceManager,
    RealTimeAttendanceResult,
    FaceProcessingResult
)
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.enrollment_service import EnrollmentService
from src.realtime.engine_client import EngineController
from src.realtime.protocol import result_from_dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATUS_REFRESH_SECONDS = 0.5   # side panel refresh while streaming
RECORDS_REFRESH_SECONDS = 2.0  # marked-students table refresh while streaming
LIVE_TABLE_ROWS = 25           # latest marks shown while streaming (the full list is in Session Management)
STATUS_CHECK_SECONDS = 10      # idle page notices a scheduled start / automatic close this fast
ATTENDANCE_LABELS = {
    AttendanceStatus.PRESENT: "✅ Present",
    AttendanceStatus.LATE: "⏰ Late",
    AttendanceStatus.ABSENT: "❌ Absent",
}
PERSON_PANEL_HEIGHT = 300      # px; fixed so the changing text never moves the page


@st.cache_resource
def _get_engine() -> EngineController:
    """One vision engine per app process (the camera can only be opened once)."""
    return EngineController()


def _show_recognition_data_hint() -> None:
    """Before starting: say so when face samples still need preparing (the engine reads them first)."""
    try:
        status = EnrollmentService().gallery_cache_status()
    except Exception as e:
        logger.warning(f"Could not read the recognition data status: {e}")
        return
    if status["pending"]:
        st.info(f"ℹ️ {status['pending']:,} of {status['samples']:,} face samples are not prepared yet: "
                f"recognition begins once they are read (the video shows the progress). To skip the wait, "
                f"use **Student Registry → Prepare recognition data** before starting.")


def render_realtime_attendance():
    """Main function to render the Real-Time Attendance page."""
    st.title("📹 Real-Time Attendance Monitoring")
    st.caption("Phase 4D: Live Camera Attendance with Face Recognition & Anti-Spoofing")
    st.markdown("Select an active session, start the camera feed, and monitor real-time attendance marking.")
    st.markdown("---")

    # Initialize services
    session_service = SessionService()
    attendance_service = AttendanceService()

    # Sidebar Configuration
    with st.sidebar:
        st.subheader("⚙️ Attendance Settings")

        require_liveness = st.checkbox(
            "Require Liveness Check",
            value=True,
            help="Enforce anti-spoofing; only live verified faces will be marked"
        )

        require_head_turn = st.checkbox(
            "Require Head Turn",
            value=config.LIVENESS_REQUIRE_CHALLENGE,
            disabled=not require_liveness,
            help="Liveness test = random blinks + a random head turn (left/right) within the time "
                 "limit. Switch off to need blinks only; the anti-spoof checks stay on."
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

    # Apply due scheduled starts / automatic closes before showing statuses
    # (the background scheduler does this too, every few seconds)
    try:
        session_service.apply_scheduled_transitions()
    except Exception as e:
        logger.error(f"Could not apply scheduled session changes: {e}")

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
                session_options[s['id']] = _scheduled_label(s)
        else:
            session_options = {s['id']: _scheduled_label(s) for s in scheduled_sessions}
            for s in all_sessions:
                if s['id'] not in session_options:
                    session_options[s['id']] = f"🔴 #{s['id']} - {s['session_name']} (Closed)"
        # Keep the chosen session listed after it closes (e.g. automatically
        # at its end time), so the page says so instead of switching sessions
        current = st.session_state.get("realtime_session_selector")
        for s in all_sessions:
            if s['id'] == current and current not in session_options:
                session_options[current] = f"🔴 #{s['id']} - {s['session_name']} (Closed)"

        selected_session_id = st.selectbox(
            "Select Session for Attendance",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            key="realtime_session_selector"
        )

    selected_session = session_service.get_session(selected_session_id)
    is_active = bool(selected_session and selected_session['status'] == SessionStatus.ACTIVE)
    is_scheduled = bool(selected_session and selected_session['status'] == SessionStatus.SCHEDULED)

    with col_activate:
        st.write("")
        st.write("")
        if not is_active and selected_session and selected_session['status'] == SessionStatus.SCHEDULED:
            if st.button("▶️ Activate Session", type="primary", width="stretch",
                         help="Start now. Started after its scheduled time, the session starts fresh: lateness counts from now."):
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
                    _get_engine().stop()
                    st.session_state.att_camera_active = False
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
        if is_scheduled:
            st.caption(f"⏰ Starts automatically at {selected_session['start_time']}. "
                       f"The camera can be started once the session is active.")

    st.markdown("---")

    # Initialize Streamlit Session State for the camera stream
    if 'att_camera_active' not in st.session_state:
        st.session_state.att_camera_active = False

    engine = _get_engine()
    engine_settings = {
        "require_liveness": require_liveness,
        "require_head_turn": require_head_turn,
        "cooldown_seconds": cooldown_seconds,
        "head_turn_angle": float(head_turn_angle),
    }

    # The session was closed (by hand elsewhere, or automatically at its end
    # time) or deleted while the camera ran: stop the camera
    if st.session_state.att_camera_active and not is_active:
        if engine.session_id in (None, selected_session_id):
            engine.stop()
            st.session_state.att_camera_active = False
            st.info("ℹ️ This session is closed, so the camera stream was stopped.")

    if not st.session_state.att_camera_active:
        _show_recognition_data_hint()

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
                with st.spinner("Starting the camera and loading the face models..."):
                    started, error = engine.start(selected_session_id, engine_settings)
                if started:
                    st.session_state.att_camera_active = True
                    st.session_state.att_engine_settings = engine_settings
                    logger.info(f"Attendance stream started for session #{selected_session_id}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to start the camera stream: {error}")

    with col_cam2:
        if st.session_state.att_camera_active:
            if st.button("⏹️ Stop Attendance Stream", width="stretch"):
                st.session_state.att_camera_active = False
                engine.stop()
                logger.info("Attendance stream stopped")
                st.rerun()

    st.markdown("---")

    # Render Live Stream & Records
    if st.session_state.att_camera_active and is_active:
        _render_live_stream(
            engine=engine,
            session_id=selected_session_id,
            session_status=selected_session['status'],
            settings=engine_settings,
            session_service=session_service,
            attendance_service=attendance_service
        )
    else:
        # Show Current Session Attendance Records Table when stream is idle
        render_session_records_panel(selected_session_id, attendance_service)
        if is_scheduled or is_active:
            _session_status_watcher(session_service, selected_session_id, selected_session['status'])


def _scheduled_label(session: Dict[str, Any]) -> str:
    return f"🟡 #{session['id']} - {session['session_name']} (Scheduled, starts {session['start_time'][:16]})"


def _render_live_stream(
    engine: EngineController,
    session_id: int,
    session_status: str,
    settings: Dict[str, Any],
    session_service: SessionService,
    attendance_service: AttendanceService
):
    """
    Show the engine's live video and the status panel.

    The video is the engine's MJPEG stream in an <img> tag: the browser plays
    it at the camera's frame rate without Streamlit handling any frames. Only
    the status panel is refreshed, a few times per second, by a fragment.
    """
    if engine.is_running() and engine.session_id != session_id:
        # Another session was selected while the camera runs: say so, keep
        # it running (the Stop button stays) rather than silently stopping it
        st.warning(f"📹 The camera is running for session #{engine.session_id}. Select that session "
                   f"to watch it, or stop the stream.")
        return
    if not engine.is_running():
        st.session_state.att_camera_active = False
        st.error("❌ The camera stream stopped. Click **Start Attendance Stream** to start it again.")
        return

    # Settings changed in the sidebar reach the running engine immediately
    if st.session_state.get('att_engine_settings') != settings:
        engine.update_settings(settings)
        st.session_state.att_engine_settings = settings

    col_video, col_live_status = st.columns([3, 2])
    with col_video:
        st.markdown(
            f'<img src="{engine.stream_url}" alt="Live camera" '
            f'style="width:100%; border-radius:0.5rem; background:#111;">',
            unsafe_allow_html=True
        )
    with col_live_status:
        _live_status_panel(engine, session_id, session_status, session_service, attendance_service)

    st.markdown("---")
    st.markdown("### 📋 Latest Marked Students")
    _live_records_table(session_id, attendance_service)


@st.fragment(run_every=STATUS_REFRESH_SECONDS)
def _live_status_panel(
    engine: EngineController,
    session_id: int,
    session_status: str,
    session_service: SessionService,
    attendance_service: AttendanceService
):
    """Live stats and the person in front of the camera (refreshed twice per second)."""
    # The engine stopped (e.g. its session closed at its end time) or the
    # session started / closed: redraw the whole page to show it
    session = session_service.get_session(session_id)
    if not engine.is_running() or session is None or session['status'] != session_status:
        st.rerun(scope="app")

    status = engine.status() or {}
    counts = attendance_service.get_session_status_counts(session_id)
    present_cnt = counts.get(AttendanceStatus.PRESENT, 0)
    late_cnt = counts.get(AttendanceStatus.LATE, 0)
    st.markdown("### 📊 Live Session Stats")
    fps = status.get("preview_fps")
    fps_text = f" | **Preview:** {fps:.0f} fps" if fps is not None else ""
    st.markdown(f"**Total:** {sum(counts.values())} | **Present:** {present_cnt} | **Late:** {late_cnt}{fps_text}")

    st.markdown("### 👤 Last Detected Person")
    # Fixed height: the text changes every refresh (instructions, warnings),
    # and a growing / shrinking panel would push the page up and down
    with st.container(height=PERSON_PANEL_HEIGHT, border=True):
        if status.get("state") == "error":
            st.error(f"❌ {status.get('error')}")
        elif status.get("error"):
            st.warning(f"⚠️ {status.get('error')}")
        if status.get("result"):
            result = result_from_dict(status["result"])
        else:
            result = RealTimeAttendanceResult(session_id=session_id, faces_detected=0,
                                              processed_faces=[], session_active=True)
        st.markdown(_build_person_info_markdown(result))
        last_mark = status.get("last_mark")
        if last_mark:
            marked_at = datetime.fromtimestamp(last_mark["time"]).strftime("%H:%M:%S")
            st.caption(f"✅ Last marked: **{last_mark['name']}** ({last_mark['status'].lower()}) at {marked_at}")


@st.fragment(run_every=RECORDS_REFRESH_SECONDS)
def _live_records_table(
    session_id: int,
    attendance_service: AttendanceService
):
    """
    The latest LIVE_TABLE_ROWS students marked in this session, newest first.
    A static table (st.table): unlike the interactive st.dataframe grid it
    does not flicker or reset its scroll position when it is refreshed with
    the same rows. Kept short so a refresh every two seconds stays light;
    Session Management lists every student, absent ones included.
    """
    records = attendance_service.get_session_records_with_students(
        session_id, newest_first=True, limit=LIVE_TABLE_ROWS
    )
    rows = _live_records_rows(records)
    if not rows:
        st.info("No attendance recorded yet.")
        return
    st.table(pd.DataFrame(rows).style.hide(axis="index"))
    total = sum(attendance_service.get_session_status_counts(session_id).values())
    if total > len(rows):
        st.caption(f"Latest {len(rows)} of {total} marked students — the full list (with absent students) "
                   f"is in **Session Management → Session Details**.")


@st.fragment(run_every=STATUS_CHECK_SECONDS)
def _session_status_watcher(session_service: SessionService, session_id: int, rendered_status: str):
    """While the camera is off: redraw the page when the session starts or closes by itself."""
    try:
        session = session_service.get_session(session_id)
    except Exception as e:
        logger.warning(f"Session status check failed: {e}")
        return
    if session is None or session['status'] != rendered_status:
        st.rerun(scope="app")


def _build_person_info_markdown(result: RealTimeAttendanceResult) -> str:
    """
    Build a single markdown string for person info display.

    Reduces per-frame Streamlit API calls from 6-8 widget creations
    to a single markdown update.
    """
    if result.multiple_faces:
        # One person at a time: nobody is recognised or marked meanwhile
        return (
            f"👥 **{result.faces_detected} faces in view**\n\n"
            f"⚠️ **{result.message}**\n\n"
            f"_Nobody is marked while more than one face is in front of the camera._"
        )

    if not result.processed_faces:
        return "_No face currently detected in frame._"

    face_info = result.processed_faces[0]

    # A suspected photo / screen replay is shown first, whoever the face is
    if face_info.reason.startswith("Spoof suspected"):
        who = face_info.student_name or ("Unknown person" if not face_info.is_recognized else face_info.identity)
        if face_info.reason.startswith(SPOOF_RECHECK_PREFIX):
            return (
                f"🟠 **{face_info.reason}**\n\n"
                f"**Face:** {who}\n\n"
                f"_Some frames looked like a photo or screen (often a low-quality "
                f"camera or poor light). If the face looks real again, checking continues._"
            )
        return (
            f"🔴 **{face_info.reason}**\n\n"
            f"**Face:** {who}\n\n"
            f"_Nobody is marked from a photo or screen._"
        )

    if face_info.is_recognized:
        name = face_info.student_name or '—'
        student_id = face_info.student_id or '—'

        # Build liveness status with instructions
        if face_info.already_marked:
            # Liveness restarts after each mark; no need to ask again
            liveness_display = "— not needed, already marked"
        elif face_info.liveness_state == LivenessState.LIVE:
            liveness_display = "🟢 **Live** — Verified"
        else:
            # Show actionable instructions based on actual detector state
            blink_status = f"{face_info.blink_count}/{face_info.min_blinks_required} blinks"
            # The number of blinks is random per challenge, so say it explicitly
            times = "time" if face_info.min_blinks_required == 1 else "times"
            blink_request = f"**Blink {face_info.min_blinks_required} {times}**"

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
                instruction = f"🔵 {blink_request} ({blink_status}) and {challenge_instruction}"
            elif needs_blinks:
                instruction = f"🔵 {blink_request} to verify liveness ({blink_status})"
            elif needs_challenge:
                instruction = f"🔵 {challenge_instruction}"
            else:
                instruction = "🟡 **Verifying...** Almost there!"

            # The challenge has a time limit; a new random one replaces it
            seconds_left = getattr(face_info, "challenge_seconds_left", None)
            if seconds_left is not None and (needs_blinks or needs_challenge):
                instruction += f" — ⏱ {math.ceil(seconds_left)} s left"

            liveness_display = instruction

        # A suspected photo / screen replay overrides the instructions
        if face_info.reason.startswith(SPOOF_RECHECK_PREFIX):
            liveness_display = f"🟠 **{face_info.reason}**"
        elif face_info.reason.startswith("Spoof suspected"):
            liveness_display = f"🔴 **{face_info.reason}**"

        lines = [
            f"**Name:** {name}",
            f"**ID:** `{student_id}`",
            f"**Liveness:** {liveness_display}",
        ]
        spoof_score = getattr(face_info, "spoof_score", None)
        if spoof_score is not None and not face_info.already_marked:
            lines.append(f"**Real-face score:** {spoof_score:.2f}")

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
    Prepare the camera frame for the preview.

    The preview is a clean, mirrored camera view: no box, name, or status
    is drawn on the face. Name, ID, liveness and attendance status are shown
    in the "Last Detected Person" panel to the right of the camera instead.
    `result` is accepted so the call site stays the same.

    Applies horizontal mirror (cv2.flip(frame, 1)) so natural user movements
    visually match a front-facing camera/mirror.
    """
    return cv2.flip(frame, 1)


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


def _live_records_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows of the live marked-students table (records already joined with the student)."""
    return [{
        'Student ID': r.get('student_code') or f"DB#{r['student_id']}",
        'Name': r.get('student_name') or "Unknown",
        'Status': ATTENDANCE_LABELS.get(r['status'], r['status']),
        'Marked At': r['marked_at'],
        'Confidence': f"{r['detection_confidence']:.2f}" if r.get('detection_confidence') is not None else "—",
        'Liveness': f"{r['liveness_score']:.2f}" if r.get('liveness_score') is not None else "—",
        'Marked By': r.get('marked_by') or 'system'
    } for r in records]


def render_session_records_panel(
    session_id: Optional[int],
    attendance_service: AttendanceService
):
    """Render attendance records panel when camera is idle."""
    if not session_id:
        return

    records = attendance_service.get_session_records_with_students(session_id)

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

    st.caption("Marked students only. **Session Management → Session Details** lists every student, "
               "absent ones included.")

    if not records:
        st.info("No attendance records for this session yet.")
        return

    st.dataframe(
        _live_records_rows(records),
        width="stretch",
        hide_index=True,
        height=min(38 + 35 * len(records), 460)
    )
