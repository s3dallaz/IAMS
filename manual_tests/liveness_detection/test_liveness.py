"""
Manual Liveness Detection Testing Script

Tests the LivenessDetector with a real webcam in continuous real-time mode.
Processes frames incrementally to detect natural blinks and determine liveness state
without stuttering or frame drops.

Controls:
  Q or ESC : Exit the test
  R        : Reset sequence and blink count
  V        : Toggle verbose console output
"""

import sys
from pathlib import Path
from collections import deque
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import config
from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.liveness_detector import LivenessDetector, LivenessState, EyeState


class LivenessTestSession:
    """Manages continuous real-time liveness testing session state."""

    def __init__(self, min_frames: int = 10, min_blinks: int = 1, liveness_threshold: float = 0.75):
        """
        Initialize the test session.

        Args:
            min_frames: Minimum face frames required before evaluating liveness
            min_blinks: Minimum blinks required for LIVE state
            liveness_threshold: Base confidence score for LIVE state
        """
        self.min_frames = min_frames
        self.min_blinks = min_blinks
        self.liveness_threshold = liveness_threshold

        # Frame statistics
        self.total_frames = 0
        self.frames_with_faces = 0
        self.consecutive_no_face = 0

        # Blink and Liveness State
        self.blinks_detected = 0
        self.current_state = LivenessState.UNKNOWN
        self.confidence = 0.0
        self.current_ear = None
        self.current_eye_state = EyeState.OPEN
        self.status_reason = "Initializing..."

        # Recent EAR history for display/analysis
        self.ear_history = deque(maxlen=60)

        # FPS tracking
        self.fps_samples = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.start_time = time.time()

    def update_frame_timing(self) -> float:
        """Update FPS calculation and return average FPS."""
        current_time = time.time()
        self.total_frames += 1

        delta = current_time - self.last_frame_time
        if delta > 0:
            fps = 1.0 / delta
            self.fps_samples.append(fps)
        self.last_frame_time = current_time

        return sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0

    def process_frame(self, ear: float, blink_occurred: bool, eye_state: EyeState) -> bool:
        """
        Process single-frame EAR observation.

        Args:
            ear: Calculated EAR float
            blink_occurred: True if a blink completed on this frame
            eye_state: Current EyeState

        Returns:
            True if liveness state changed, False otherwise
        """
        self.frames_with_faces += 1
        self.consecutive_no_face = 0
        self.current_ear = ear
        self.current_eye_state = eye_state
        self.ear_history.append(ear)

        old_state = self.current_state

        if blink_occurred:
            self.blinks_detected += 1

        # Evaluate liveness state
        if self.frames_with_faces < self.min_frames:
            self.current_state = LivenessState.UNKNOWN
            self.confidence = 0.0
            self.status_reason = f"Collecting frames: {self.frames_with_faces}/{self.min_frames}"
        elif self.blinks_detected >= self.min_blinks:
            self.current_state = LivenessState.LIVE
            self.confidence = min(
                self.liveness_threshold + (self.blinks_detected - self.min_blinks) * 0.05,
                1.0
            )
            self.status_reason = f"Blinks detected: {self.blinks_detected} (threshold: {self.min_blinks})"
        else:
            self.current_state = LivenessState.UNKNOWN
            self.confidence = 0.0
            self.status_reason = f"Insufficient blinks: {self.blinks_detected} < {self.min_blinks} required"

        return self.current_state != old_state

    def process_no_face(self) -> None:
        """Handle a frame where no face was detected."""
        self.consecutive_no_face += 1
        self.current_ear = None
        if self.consecutive_no_face > 30:
            self.status_reason = "No face detected"

    def reset(self) -> str:
        """Reset session state and blink counts."""
        self.total_frames = 0
        self.frames_with_faces = 0
        self.consecutive_no_face = 0
        self.blinks_detected = 0
        self.current_state = LivenessState.UNKNOWN
        self.confidence = 0.0
        self.current_ear = None
        self.current_eye_state = EyeState.OPEN
        self.status_reason = "Sequence reset"
        self.ear_history.clear()
        self.start_time = time.time()
        return "[INFO] Session reset successfully"

    def get_stats(self) -> dict:
        """Return a snapshot of current session statistics."""
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
        return {
            'total_frames': self.total_frames,
            'frames_with_faces': self.frames_with_faces,
            'state': self.current_state,
            'confidence': self.confidence,
            'blinks': self.blinks_detected,
            'current_ear': self.current_ear,
            'eye_state': self.current_eye_state,
            'reason': self.status_reason,
            'fps': avg_fps
        }


def draw_overlay(frame: np.ndarray, session: LivenessTestSession, face_bbox: tuple = None) -> np.ndarray:
    """
    Draw a clean, informative HUD overlay on the preview frame.

    Args:
        frame: OpenCV BGR frame
        session: Active LivenessTestSession
        face_bbox: Bounding box (x, y, w, h) or None

    Returns:
        Frame with rendered HUD
    """
    result = frame.copy()
    height, width = frame.shape[:2]
    stats = session.get_stats()

    # Draw face bounding box
    if face_bbox is not None:
        x, y, w, h = face_bbox
        # Green box if LIVE, Orange if UNKNOWN
        box_color = (0, 255, 0) if stats['state'] == LivenessState.LIVE else (0, 165, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)

    # Semi-transparent HUD overlay banner
    overlay_height = 190
    overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
    overlay[:] = (30, 30, 30)

    # State colors and text
    if stats['state'] == LivenessState.LIVE:
        state_color = (0, 255, 0)  # Green
        state_text = "LIVE"
    elif stats['state'] == LivenessState.SPOOF:
        state_color = (0, 0, 255)  # Red
        state_text = "SPOOF"
    else:
        state_color = (0, 165, 255)  # Orange
        state_text = "UNKNOWN"

    y_pos = 28
    line_spacing = 28

    # Title & FPS
    cv2.putText(overlay, "LIVENESS DETECTION - PHASE 3B", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    fps_color = (0, 255, 0) if stats['fps'] >= 24 else (0, 255, 255) if stats['fps'] >= 15 else (0, 100, 255)
    cv2.putText(overlay, f"{stats['fps']:.1f} FPS", (width - 120, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)
    y_pos += line_spacing + 4

    # State (large)
    cv2.putText(overlay, f"STATE: {state_text}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

    # Confidence
    cv2.putText(overlay, f"Confidence: {stats['confidence']:.3f}", (260, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_pos += line_spacing

    # EAR and Eye State
    if stats['current_ear'] is not None:
        eye_label = "CLOSED" if stats['eye_state'] == EyeState.CLOSED else "OPEN"
        eye_color = (0, 100, 255) if stats['eye_state'] == EyeState.CLOSED else (0, 255, 0)
        ear_text = f"EAR: {stats['current_ear']:.3f} [{eye_label}]"
    else:
        ear_text = "EAR: N/A"
        eye_color = (150, 150, 150)

    cv2.putText(overlay, ear_text, (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)

    # Blinks detected
    blink_color = (0, 255, 0) if stats['blinks'] >= session.min_blinks else (255, 255, 255)
    cv2.putText(overlay, f"Blinks: {stats['blinks']} (Req: {session.min_blinks})", (260, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
    y_pos += line_spacing

    # Frames count & Reason
    frames_text = f"Face Frames: {stats['frames_with_faces']}/{session.min_frames} | {stats['reason']}"
    cv2.putText(overlay, frames_text, (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y_pos += line_spacing - 4

    # Controls hint
    cv2.putText(overlay, "Q/ESC: Exit   |   R: Reset Session   |   V: Toggle Verbose", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

    # Blend overlay with original frame
    result[0:overlay_height] = cv2.addWeighted(
        result[0:overlay_height], 0.25, overlay, 0.75, 0
    )

    return result


def main():
    """Main execution loop for manual webcam liveness validation."""
    print("\n" + "=" * 70)
    print("LIVENESS DETECTION TEST - Phase 3B (Optimized Real-Time)")
    print("=" * 70)
    print("\nStarting video stream. Passive blink detection running in real-time.")
    print("\nControls:")
    print("  Q or ESC : Exit")
    print("  R        : Reset sequence and blinks")
    print("  V        : Toggle verbose logging")
    print("\nInitializing hardware & models...\n")

    camera = None
    detector = None
    liveness_detector = None
    verbose = False

    try:
        # 1. Camera Initialization
        print("[INFO] Opening webcam...")
        camera = CameraManager(camera_index=0)
        if not camera.open():
            print("[FAILED] Could not open camera.")
            print("         Please check that your webcam is connected and not used by another app.")
            return

        dimensions = camera.get_frame_dimensions()
        if dimensions:
            print(f"[OK] Camera ready at resolution: {dimensions[0]}x{dimensions[1]}")

        # 2. Face Detector Initialization
        print("[INFO] Initializing FaceDetector...")
        detector = FaceDetector()
        print("[OK] FaceDetector initialized.")

        # 3. Liveness Detector Initialization
        print("[INFO] Initializing LivenessDetector...")
        liveness_detector = LivenessDetector()
        print("[OK] LivenessDetector initialized.")

        print("\n" + "=" * 70)
        print("WEBCAM ACTIVE - Position your face in front of the camera")
        print("=" * 70)
        print("\nBlink naturally to confirm liveness.\n")

        session = LivenessTestSession(
            min_frames=liveness_detector.min_frames,
            min_blinks=liveness_detector.min_blinks,
            liveness_threshold=liveness_detector.liveness_threshold
        )

        last_state = LivenessState.UNKNOWN
        last_logged_blinks = 0

        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            session.update_frame_timing()

            # Detect face
            detected_faces = detector.detect_faces(frame)
            face = detected_faces[0] if detected_faces else None
            face_bbox = face.bbox if face else None

            if face is not None:
                # Incremental single-frame EAR extraction (fast & responsive)
                ear = liveness_detector.extract_ear(frame, face)
                blink_occurred = liveness_detector.update_temporal_state(ear)
                state_changed = session.process_frame(ear, blink_occurred, liveness_detector.eye_state)

                # Only log state changes to avoid stuttering from excessive console I/O
                if state_changed:
                    elapsed = time.time() - session.start_time
                    print(f"[{elapsed:6.1f}s] STATE: {session.current_state.value.upper()} | "
                          f"Confidence: {session.confidence:.2f} | Blinks: {session.blinks_detected}")

                # Verbose mode: periodic detailed logging (every 60 frames to minimize I/O)
                if verbose and session.total_frames % 60 == 0:
                    elapsed = time.time() - session.start_time
                    print(f"[{elapsed:6.1f}s] EAR: {ear:.3f} | Eye: {liveness_detector.eye_state.value} | "
                          f"Frames: {session.frames_with_faces} | Reason: {session.status_reason}")
            else:
                liveness_detector.update_temporal_state(None)
                session.process_no_face()

            # Render HUD overlay and show frame
            display_frame = draw_overlay(frame, session, face_bbox)
            cv2.imshow('Liveness Detection Test - Phase 3B', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("\n[INFO] Exiting...")
                break
            elif key in (ord('r'), ord('R')):
                liveness_detector.reset_temporal_state()
                msg = session.reset()
                print(msg)
            elif key in (ord('v'), ord('V')):
                verbose = not verbose
                print(f"[INFO] Verbose mode: {'ON' if verbose else 'OFF'}")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlace face_landmarker.task in project root directory.")

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Print comprehensive final summary
        if 'session' in locals():
            stats = session.get_stats()
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            print(f"Total Frames Processed:    {stats['total_frames']}")
            print(f"Frames with Detected Face: {stats['frames_with_faces']}")
            print(f"Average FPS:               {stats['fps']:.1f}")
            print(f"Final Liveness State:      {stats['state'].value.upper()}")
            print(f"Final Confidence:          {stats['confidence']:.3f}")
            print(f"Total Blinks Detected:     {stats['blinks']}")
            print(f"Final Reason:              {stats['reason']}")
            print("=" * 70)

        if camera is not None:
            camera.release()
        if detector is not None:
            detector.close()
        if liveness_detector is not None:
            liveness_detector.close()
        cv2.destroyAllWindows()
        print("[OK] Resources released successfully\n")


if __name__ == "__main__":
    main()
