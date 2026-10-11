"""
Liveness Detection Test Page for the Intelligent Attendance Monitoring System.

Phase 3C: Active Liveness & Anti-Spoofing
Provides real-time camera interface for testing liveness detection with active challenges.
"""

import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional

from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.liveness_detector import (
    LivenessDetector,
    LivenessState,
    ChallengeType,
    ChallengeState
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_liveness_test():
    """Render the Liveness Detection Test page."""
    st.title("🔍 Liveness Detection Test")
    st.caption("Phase 3C: Active Challenge-Response Anti-Spoofing")
    
    st.markdown("""
    This page demonstrates liveness detection with active challenges.
    The system will:
    1. Start camera once (stays open for multiple tests)
    2. Detect your face
    3. Monitor for natural blinks (Phase 3B)
    4. Present a randomized head-turn challenge (Phase 3C)
    5. Verify you complete BOTH requirements
    6. **Take as long as you need - NO TIME LIMIT**
    7. Allow multiple people to test without restarting camera
    """)
    
    st.markdown("---")
    
    # Configuration sidebar
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        enable_challenges = st.checkbox(
            "Enable Active Challenges",
            value=True,
            help="Require head-turn challenges for liveness verification"
        )
        
        require_challenge = st.checkbox(
            "Require Challenge Completion",
            value=True,
            help="Must complete challenge to pass liveness check",
            disabled=not enable_challenges
        )
        
        head_turn_angle = st.slider(
            "Head Turn Angle (degrees)",
            min_value=15.0,
            max_value=40.0,
            value=25.0,
            step=5.0,
            help="Required head rotation angle"
        )
        
        min_blinks = st.number_input(
            "Minimum Blinks Required",
            min_value=0,
            max_value=5,
            value=1,
            help="Minimum blinks needed for liveness"
        )
        
        st.markdown("---")
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. Click **Start Camera** (only needed once)
        2. Click **Start/Reset Test** to begin
        3. Position your face in frame
        4. **Blink naturally**
        5. **Follow the challenge instruction**
        6. Hold your head in position
        7. **Take as long as you need** - no time limit
        8. Test completes when BOTH requirements met
        9. Click **Start/Reset Test** for next person
        10. Click **Stop Camera** when completely done
        """)
    
    # Initialize session state
    # Camera session (long-lived, persistent)
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = None
    
    # Test session (short-lived, resettable)
    if 'test_active' not in st.session_state:
        st.session_state.test_active = False
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'test_result_message' not in st.session_state:
        st.session_state.test_result_message = ""
    
    # Detectors (initialized lazily to avoid blocking UI)
    if 'liveness_detector' not in st.session_state:
        st.session_state.liveness_detector = None
    if 'face_detector' not in st.session_state:
        st.session_state.face_detector = None
    if 'detectors_initialized' not in st.session_state:
        st.session_state.detectors_initialized = False
    if 'detectors_initializing' not in st.session_state:
        st.session_state.detectors_initializing = False
    
    # Camera controls
    col1, col2 = st.columns(2)
    
    # Initialize detectors IMMEDIATELY when page loads if not done
    # This prevents startup delay - models load once at page start
    if not st.session_state.detectors_initialized and not st.session_state.detectors_initializing:
        st.session_state.detectors_initializing = True
        with st.spinner("Initializing detection models..."):
            start_init = time.perf_counter()
            
            if st.session_state.face_detector is None:
                st.session_state.face_detector = FaceDetector()
            
            if st.session_state.liveness_detector is None:
                st.session_state.liveness_detector = LivenessDetector(
                    enable_challenges=enable_challenges,
                    require_challenge=require_challenge,
                    head_turn_angle=head_turn_angle,
                    min_blinks=min_blinks
                )
            
            init_time = time.perf_counter() - start_init
            logger.info(f"Detectors initialized in {init_time:.2f}s")
            
            st.session_state.detectors_initialized = True
            st.session_state.detectors_initializing = False
    
    with col1:
        if not st.session_state.camera_active:
            if st.button("📹 Start Camera", width="stretch"):
                # Detectors should already be initialized at page load
                if not st.session_state.detectors_initialized:
                    st.warning("⏳ Detectors still initializing, please wait...")
                    st.rerun()
                    return
                
                # Initialize camera once
                if st.session_state.camera_manager is None:
                    st.session_state.camera_manager = CameraManager(camera_index=0)
                
                # Try to open camera - should be fast
                start_camera = time.perf_counter()
                if st.session_state.camera_manager.open():
                    camera_time = time.perf_counter() - start_camera
                    logger.info(f"Camera opened in {camera_time:.3f}s")
                    st.session_state.camera_active = True
                    st.rerun()
                else:
                    st.error("❌ Failed to open camera")
                    st.session_state.camera_manager = None
        else:
            # Camera is running - show Start/Reset Test button
            button_label = "🔄 Start/Reset Test" if st.session_state.test_completed else "▶️ Start Test"
            if st.button(button_label, disabled=st.session_state.test_active, width="stretch"):
                # Ensure detectors are ready
                if not st.session_state.detectors_initialized:
                    st.warning("Detectors not ready yet, please wait...")
                    return
                
                st.session_state.test_active = True
                st.session_state.test_completed = False
                st.session_state.test_result_message = ""
                # Reset liveness state for new test
                st.session_state.liveness_detector.reset_temporal_state()
                if enable_challenges:
                    st.session_state.liveness_detector.start_new_challenge()
                st.rerun()
    
    with col2:
        if st.session_state.camera_active:
            if st.button("⏹️ Stop Camera", width="stretch"):
                # Stop camera session completely
                st.session_state.camera_active = False
                st.session_state.test_active = False
                st.session_state.test_completed = False
                st.session_state.test_result_message = ""
                if st.session_state.camera_manager is not None:
                    st.session_state.camera_manager.release()
                    st.session_state.camera_manager = None
                st.rerun()
    
    st.markdown("---")
    
    # Show test result if completed
    if st.session_state.test_completed and st.session_state.test_result_message:
        st.success(st.session_state.test_result_message)
        st.markdown("---")
    
    # Camera feed and detection
    if st.session_state.camera_active:
        _run_liveness_detection(
            enable_challenges=enable_challenges,
            require_challenge=require_challenge,
            head_turn_angle=head_turn_angle,
            min_blinks=min_blinks
        )
    else:
        st.info("👆 Click 'Start Camera' to begin")
        
        # Display warning about limitations
        with st.expander("⚠️ Important Security Notice"):
            st.warning("""
            **Phase 3C Limitations:**
            
            This active liveness implementation provides protection against:
            - ✅ Static printed photos
            - ✅ Photos displayed on phones/monitors
            - ✅ Masks or cutouts
            
            However, it does NOT fully protect against:
            - ⚠️ Pre-recorded videos with head movements
            - ⚠️ Deep-fake or synthetic video attacks
            - ⚠️ Sophisticated presentation attacks
            
            **This is a behavioral challenge method suitable for basic anti-spoofing.**
            
            For production systems requiring higher security:
            - Add texture/frequency analysis
            - Implement 3D depth sensing
            - Use multiple challenge types
            - Combine with additional biometric factors
            """)


def _run_liveness_detection(
    enable_challenges: bool,
    require_challenge: bool,
    head_turn_angle: float,
    min_blinks: int
):
    """
    Run real-time liveness detection with camera feed.
    
    Uses continuous frame processing loop with st.empty() updates.
    NO st.rerun() per frame - smooth camera performance.
    
    Args:
        enable_challenges: Enable active challenges
        require_challenge: Require challenge completion
        head_turn_angle: Required head turn angle
        min_blinks: Minimum blinks required
    """
    # Use pre-initialized detectors and camera from session state
    detector = st.session_state.liveness_detector
    face_detector = st.session_state.face_detector
    camera = st.session_state.camera_manager
    
    # UI placeholders for camera and status displays
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    challenge_placeholder = st.empty()
    requirements_placeholder = st.empty()
    
    try:
        # Camera should already be open from Start Camera button
        if not camera.is_open():
            video_placeholder.error("❌ Camera is not open")
            return
        
        # CONTINUOUS FRAME PROCESSING LOOP
        # This runs until test completes or camera is stopped
        # Updates st.empty() placeholders without full script rerun
        frame_count = 0
        max_frames_per_run = 300  # Process up to 300 frames per script execution (~10 seconds at 30fps)
        
        while frame_count < max_frames_per_run:
            # Check if camera was stopped or test state changed via button click
            # Note: Button clicks will cause a rerun, exiting this loop naturally
            if not st.session_state.camera_active:
                break
            
            # Capture frame
            success, frame = camera.read_frame()
            
            if not success or frame is None:
                # Frame read failed - display warning but continue
                video_placeholder.warning("⚠️ Failed to read frame - camera remains active")
                time.sleep(0.05)
                continue
            
            # Process the frame based on test state
            if st.session_state.test_active and not st.session_state.test_completed:
                # Test is active - perform liveness detection
                faces = face_detector.detect_faces(frame)
                
                if len(faces) == 0:
                    # No face detected - show camera with overlay warning
                    display_frame = frame.copy()
                    
                    # Add semi-transparent overlay for better text visibility
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 120), (0, 0, 0), -1)
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                    
                    # Draw warning text on camera feed
                    cv2.putText(
                        display_frame,
                        "No face detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 165, 255),  # Orange
                        2
                    )
                    cv2.putText(
                        display_frame,
                        "Position yourself in frame",
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),  # Orange
                        2
                    )
                    
                    # Display camera feed with warning overlay
                    video_placeholder.image(
                        cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        width=640
                    )
                    
                    # Small delay then continue
                    time.sleep(0.033)
                    frame_count += 1
                    continue
                
                # Use first detected face
                face = faces[0]
                
                # Extract EAR and update liveness detector
                ear = detector.extract_ear(frame, face)
                blink_detected = detector.update_temporal_state(ear, frame, face)
                
                # Draw face bbox and landmarks on frame
                display_frame = _draw_detection_overlay(
                    frame,
                    face,
                    detector,
                    enable_challenges
                )
                
                # Display frame
                video_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width=640
                )
                
                # Update status
                _update_status_display(
                    status_placeholder,
                    detector,
                    enable_challenges,
                    require_challenge,
                    min_blinks
                )
                
                # Update metrics
                _update_metrics_display(
                    metrics_placeholder,
                    detector,
                    ear,
                    blink_detected
                )
                
                # Update challenge progress
                if enable_challenges:
                    _update_challenge_display(
                        challenge_placeholder,
                        detector
                    )
                
                # Update requirements status
                _update_requirements_display(
                    requirements_placeholder,
                    detector,
                    enable_challenges,
                    require_challenge,
                    min_blinks
                )
                
                # Check for OVERALL completion (both blinks AND challenge)
                blink_requirement_met = detector.blink_count >= min_blinks
                challenge_requirement_met = (not require_challenge or not enable_challenges or 
                                            detector.challenge_state == ChallengeState.COMPLETED)
                
                if blink_requirement_met and challenge_requirement_met:
                    # BOTH requirements satisfied - TEST COMPLETE
                    st.session_state.test_completed = True
                    st.session_state.test_active = False
                    st.session_state.test_result_message = (
                        f"✅ **LIVE / Test Completed!**\n\n"
                        f"✓ Blinks: {detector.blink_count} (required: {min_blinks})\n\n"
                        f"✓ Challenge: Completed"
                    )
                    # DO NOT release camera - trigger rerun to show completed state
                    st.rerun()
                    return
            else:
                # Test not active or completed - just show camera feed
                display_frame = frame.copy()
                if st.session_state.test_completed:
                    # Add semi-transparent overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                    
                    cv2.putText(
                        display_frame,
                        "Test Completed - Click Start/Reset Test",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
                    # Add semi-transparent overlay
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                    
                    cv2.putText(
                        display_frame,
                        "Camera Ready - Click Start/Reset Test",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                video_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width=640
                )
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 fps
            frame_count += 1
        
        # After processing max_frames_per_run frames, trigger rerun to continue
        # This prevents infinite loop while maintaining smooth camera feed
        if st.session_state.camera_active:
            st.rerun()
        
    except Exception as e:
        video_placeholder.error(f"❌ Error during liveness detection: {e}")
        logger.error(f"Liveness detection error: {e}", exc_info=True)
        # Even on error, trigger rerun to keep camera alive if still active
        if st.session_state.camera_active:
            time.sleep(0.1)
            st.rerun()


def _draw_detection_overlay(
    frame: np.ndarray,
    face: DetectedFace,
    detector: LivenessDetector,
    enable_challenges: bool
) -> np.ndarray:
    """
    Draw detection overlay on frame.
    
    Args:
        frame: Input frame
        face: Detected face
        detector: Liveness detector
        enable_challenges: Whether challenges are enabled
    
    Returns:
        Frame with overlay drawn
    """
    display_frame = frame.copy()
    x, y, w, h = face.bbox
    
    # Draw face bounding box
    color = (0, 255, 0)  # Green
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw eye state
    eye_state_text = f"Eyes: {detector.eye_state.value}"
    cv2.putText(
        display_frame,
        eye_state_text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
    # Draw blink count
    blink_text = f"Blinks: {detector.blink_count}"
    cv2.putText(
        display_frame,
        blink_text,
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
    # Draw challenge instruction if active
    if enable_challenges and detector.challenge_state == ChallengeState.ACTIVE:
        challenge_type = detector.current_challenge
        if challenge_type == ChallengeType.TURN_LEFT:
            instruction = "Turn head LEFT"
            arrow = "<<<"
        elif challenge_type == ChallengeType.TURN_RIGHT:
            instruction = "Turn head RIGHT"
            arrow = ">>>"
        else:
            instruction = ""
            arrow = ""
        
        if instruction:
            # Draw instruction at top center
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(
                display_frame,
                instruction,
                (text_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),  # Yellow
                3
            )
            
            # Draw arrow
            cv2.putText(
                display_frame,
                arrow,
                (text_x, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 255),
                3
            )
    
    return display_frame


def _update_status_display(
    placeholder,
    detector: LivenessDetector,
    enable_challenges: bool,
    require_challenge: bool,
    min_blinks: int
):
    """Update status display with clear distinction between challenge completion and overall test completion."""
    blink_requirement_met = detector.blink_count >= min_blinks
    challenge_requirement_met = (not require_challenge or not enable_challenges or 
                                detector.challenge_state == ChallengeState.COMPLETED)
    
    if not enable_challenges:
        if blink_requirement_met:
            placeholder.success(f"✅ Blink requirement met: {detector.blink_count}")
        else:
            placeholder.info(f"👁️ Monitoring blinks... Detected: {detector.blink_count}")
    else:
        # Both requirements needed for overall LIVE status
        if blink_requirement_met and challenge_requirement_met:
            placeholder.success("✅ BOTH requirements met - Test will complete!")
        elif detector.challenge_state == ChallengeState.COMPLETED and not blink_requirement_met:
            # Critical: Challenge done but blinks missing - NOT LIVE
            placeholder.warning(f"⚠️ Head challenge completed — waiting for {min_blinks - detector.blink_count} more blink(s)")
        elif detector.challenge_state == ChallengeState.ACTIVE:
            if blink_requirement_met:
                placeholder.info("🎯 Blink requirement completed — complete the head challenge")
            else:
                placeholder.info("🎯 Performing active challenge and monitoring blinks...")
        elif detector.challenge_state == ChallengeState.INACTIVE:
            if blink_requirement_met:
                placeholder.info("⏳ Blink requirement completed — waiting for challenge to start")
            else:
                placeholder.info("⏳ Monitoring for blinks and challenge...")


def _update_metrics_display(
    placeholder,
    detector: LivenessDetector,
    ear: Optional[float],
    blink_detected: bool
):
    """Update metrics display."""
    col1, col2, col3 = placeholder.columns(3)
    
    with col1:
        st.metric("Blinks Detected", detector.blink_count)
    
    with col2:
        ear_value = f"{ear:.3f}" if ear is not None else "N/A"
        st.metric("Eye Aspect Ratio", ear_value)
    
    with col3:
        eye_state = detector.eye_state.value.upper()
        st.metric("Eye State", eye_state)


def _update_challenge_display(
    placeholder,
    detector: LivenessDetector
):
    """Update challenge progress display."""
    progress = detector.get_challenge_progress()
    
    with placeholder.container():
        st.markdown("### 🎯 Challenge Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Challenge:** {progress['challenge_type'].replace('_', ' ').title()}")
            st.markdown(f"**State:** {progress['challenge_state'].title()}")
        
        with col2:
            st.markdown(f"**Frames Held:** {progress['frames_held']} / {progress['frames_required']}")
        
        # Progress bar
        st.progress(progress['progress_percent'])


def _update_requirements_display(
    placeholder,
    detector: LivenessDetector,
    enable_challenges: bool,
    require_challenge: bool,
    min_blinks: int
):
    """Update overall requirements status."""
    blink_requirement_met = detector.blink_count >= min_blinks
    challenge_requirement_met = (not require_challenge or not enable_challenges or 
                                detector.challenge_state == ChallengeState.COMPLETED)
    
    with placeholder.container():
        st.markdown("### ✓ Requirements Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if blink_requirement_met:
                st.success(f"✅ Blinks: {detector.blink_count} / {min_blinks}")
            else:
                st.warning(f"⏳ Blinks: {detector.blink_count} / {min_blinks}")
        
        with col2:
            if not enable_challenges or not require_challenge:
                st.info("Challenge: Disabled")
            elif challenge_requirement_met:
                st.success("✅ Challenge: Completed")
            else:
                st.warning("⏳ Challenge: Pending")
