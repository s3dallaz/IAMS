"""
Diagnostic Liveness Detection Testing Script

Measures actual performance bottlenecks and EAR values to identify root causes
of webcam stuttering and missed blink detection.

Controls:
  Q or ESC : Exit the test
  R        : Reset sequence and blink count
  D        : Toggle detailed diagnostics display
  S        : Save EAR data to CSV file
"""

import sys
from pathlib import Path
from collections import deque
import time
import csv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import config
from src.computer_vision.camera_manager import CameraManager
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.liveness_detector import LivenessDetector, LivenessState, EyeState


class PerformanceDiagnostics:
    """Track detailed performance metrics for each stage of processing."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.camera_capture_times = deque(maxlen=window_size)
        self.face_detection_times = deque(maxlen=window_size)
        self.landmark_extraction_times = deque(maxlen=window_size)
        self.ear_calculation_times = deque(maxlen=window_size)
        self.state_update_times = deque(maxlen=window_size)
        self.display_times = deque(maxlen=window_size)
        self.total_frame_times = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)

    def record_camera_capture(self, elapsed_ms):
        self.camera_capture_times.append(elapsed_ms)

    def record_face_detection(self, elapsed_ms):
        self.face_detection_times.append(elapsed_ms)

    def record_landmark_extraction(self, elapsed_ms):
        self.landmark_extraction_times.append(elapsed_ms)

    def record_ear_calculation(self, elapsed_ms):
        self.ear_calculation_times.append(elapsed_ms)

    def record_state_update(self, elapsed_ms):
        self.state_update_times.append(elapsed_ms)

    def record_display(self, elapsed_ms):
        self.display_times.append(elapsed_ms)

    def record_total_frame(self, elapsed_ms):
        self.total_frame_times.append(elapsed_ms)
        self.frame_timestamps.append(time.time())

    def get_stats(self):
        """Calculate statistics for all metrics."""
        def calc_stats(times):
            if not times:
                return {'avg': 0, 'min': 0, 'max': 0, 'p95': 0}
            times_list = list(times)
            times_sorted = sorted(times_list)
            p95_idx = int(len(times_sorted) * 0.95)
            return {
                'avg': np.mean(times_list),
                'min': np.min(times_list),
                'max': np.max(times_list),
                'p95': times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1]
            }

        # Calculate FPS from timestamps
        fps = 0
        if len(self.frame_timestamps) >= 2:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                fps = (len(self.frame_timestamps) - 1) / time_span

        return {
            'fps': fps,
            'camera_capture': calc_stats(self.camera_capture_times),
            'face_detection': calc_stats(self.face_detection_times),
            'landmark_extraction': calc_stats(self.landmark_extraction_times),
            'ear_calculation': calc_stats(self.ear_calculation_times),
            'state_update': calc_stats(self.state_update_times),
            'display': calc_stats(self.display_times),
            'total_frame': calc_stats(self.total_frame_times)
        }


class EARDataLogger:
    """Log EAR values and state transitions for analysis."""

    def __init__(self, max_entries=5000):
        self.entries = []
        self.max_entries = max_entries

    def log(self, timestamp, ear, eye_state, blink_count, face_detected):
        """Log a single frame's data."""
        self.entries.append({
            'timestamp': timestamp,
            'ear': ear if ear is not None else -1,
            'eye_state': eye_state.value if eye_state else 'none',
            'blink_count': blink_count,
            'face_detected': face_detected
        })

        # Keep only recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def save_to_csv(self, filename='ear_diagnostic_data.csv'):
        """Save logged data to CSV file."""
        if not self.entries:
            return False

        filepath = project_root / 'manual_tests' / 'liveness_detection' / filename
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'ear', 'eye_state', 'blink_count', 'face_detected'])
                writer.writeheader()
                writer.writerows(self.entries)
            return filepath
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return None

    def get_recent_ear_values(self, count=60):
        """Get recent EAR values for display."""
        recent = self.entries[-count:] if len(self.entries) >= count else self.entries
        return [e['ear'] for e in recent if e['ear'] >= 0]


def draw_diagnostic_overlay(frame, perf_stats, ear, eye_state, blink_count, ear_logger,
                            close_threshold, open_threshold, show_detailed=False):
    """Draw comprehensive diagnostic overlay on frame."""
    result = frame.copy()
    height, width = frame.shape[:2]

    # Semi-transparent background
    overlay_height = 280 if show_detailed else 200
    overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
    overlay[:] = (20, 20, 20)

    y_pos = 25
    line_spacing = 25

    # Title
    cv2.putText(overlay, "LIVENESS DIAGNOSTIC MODE", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # FPS with color coding
    fps = perf_stats['fps']
    fps_color = (0, 255, 0) if fps >= 25 else (0, 255, 255) if fps >= 20 else (0, 100, 255)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (width - 150, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    y_pos += line_spacing + 5

    # EAR and thresholds
    if ear is not None:
        ear_color = (0, 100, 255) if ear < close_threshold else (0, 255, 255) if ear < open_threshold else (0, 255, 0)
        cv2.putText(overlay, f"EAR: {ear:.4f}", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)
    else:
        cv2.putText(overlay, "EAR: N/A", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)

    cv2.putText(overlay, f"Close<{close_threshold:.2f} Open>={open_threshold:.2f}", (200, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    y_pos += line_spacing

    # Eye state and blink count
    state_text = eye_state.value.upper() if eye_state else "UNKNOWN"
    state_color = (0, 100, 255) if state_text == "CLOSED" else (0, 255, 0)
    cv2.putText(overlay, f"Eye State: {state_text}", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    cv2.putText(overlay, f"Blinks: {blink_count}", (250, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += line_spacing

    # Performance breakdown
    total_stats = perf_stats['total_frame']
    cv2.putText(overlay, f"Frame Time: {total_stats['avg']:.1f}ms (p95: {total_stats['p95']:.1f}ms)",
                (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)
    y_pos += line_spacing

    if show_detailed:
        # Detailed timing breakdown
        cv2.putText(overlay, "TIMING BREAKDOWN (avg ms):", (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += line_spacing - 5

        timing_items = [
            ('Capture', perf_stats['camera_capture']['avg']),
            ('Detection', perf_stats['face_detection']['avg']),
            ('Landmarks', perf_stats['landmark_extraction']['avg']),
            ('EAR Calc', perf_stats['ear_calculation']['avg']),
            ('State', perf_stats['state_update']['avg']),
            ('Display', perf_stats['display']['avg'])
        ]

        for name, value in timing_items:
            color = (0, 255, 0) if value < 10 else (0, 255, 255) if value < 20 else (0, 150, 255)
            cv2.putText(overlay, f"  {name}: {value:.1f}ms", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_pos += 20

    # EAR history visualization (mini graph)
    if ear is not None:
        recent_ears = ear_logger.get_recent_ear_values(60)
        if len(recent_ears) > 1:
            graph_height = 50
            graph_width = 200
            graph_x = width - graph_width - 15
            graph_y = overlay_height - graph_height - 15

            # Graph background
            cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (40, 40, 40), -1)

            # Threshold lines
            close_y = int(graph_y + graph_height * (1 - (close_threshold / 0.5)))
            open_y = int(graph_y + graph_height * (1 - (open_threshold / 0.5)))
            cv2.line(overlay, (graph_x, close_y), (graph_x + graph_width, close_y), (0, 100, 255), 1)
            cv2.line(overlay, (graph_x, open_y), (graph_x + graph_width, open_y), (0, 255, 0), 1)

            # EAR values
            x_step = graph_width / max(len(recent_ears) - 1, 1)
            for i in range(len(recent_ears) - 1):
                ear_val1 = np.clip(recent_ears[i], 0, 0.5)
                ear_val2 = np.clip(recent_ears[i + 1], 0, 0.5)

                y1 = int(graph_y + graph_height * (1 - (ear_val1 / 0.5)))
                y2 = int(graph_y + graph_height * (1 - (ear_val2 / 0.5)))
                x1 = int(graph_x + i * x_step)
                x2 = int(graph_x + (i + 1) * x_step)

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Controls hint
    y_pos = overlay_height - 10
    cv2.putText(overlay, "Q:Exit | R:Reset | D:Detail | S:Save CSV", (15, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

    # Blend overlay with frame
    result[0:overlay_height] = cv2.addWeighted(
        result[0:overlay_height], 0.2, overlay, 0.8, 0
    )

    return result


def main():
    """Main diagnostic loop."""
    print("\n" + "=" * 80)
    print("LIVENESS DETECTION DIAGNOSTIC MODE")
    print("=" * 80)
    print("\nThis diagnostic tool measures actual performance and EAR values.")
    print("Use it to identify:")
    print("  - Frame processing bottlenecks (camera, inference, display)")
    print("  - EAR value ranges during natural blinks")
    print("  - State transition behavior")
    print("\nControls:")
    print("  Q or ESC : Exit and show summary")
    print("  R        : Reset blink count and clear data")
    print("  D        : Toggle detailed timing breakdown")
    print("  S        : Save EAR data to CSV file")
    print("\n" + "=" * 80)

    camera = None
    detector = None
    liveness_detector = None

    try:
        # Initialize components
        print("\n[INFO] Initializing components...")
        camera = CameraManager(camera_index=0)
        if not camera.open():
            print("[FAILED] Could not open camera.")
            return

        detector = FaceDetector()
        liveness_detector = LivenessDetector()

        perf_diagnostics = PerformanceDiagnostics()
        ear_logger = EARDataLogger()

        show_detailed = False
        session_start = time.time()
        total_frames = 0
        frames_with_face = 0

        print("[OK] Diagnostic mode active. Press D for detailed breakdown.\n")

        while True:
            frame_start = time.time()

            # 1. Camera capture
            capture_start = time.time()
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue
            capture_time = (time.time() - capture_start) * 1000
            perf_diagnostics.record_camera_capture(capture_time)

            total_frames += 1

            # 2. Face detection
            detection_start = time.time()
            detected_faces = detector.detect_faces(frame)
            detection_time = (time.time() - detection_start) * 1000
            perf_diagnostics.record_face_detection(detection_time)

            face = detected_faces[0] if detected_faces else None

            ear = None
            eye_state = None
            blink_count = 0

            if face is not None:
                frames_with_face += 1

                # 3. Landmark extraction
                landmark_start = time.time()
                landmarks = liveness_detector._extract_eye_landmarks(frame, face)
                landmark_time = (time.time() - landmark_start) * 1000
                perf_diagnostics.record_landmark_extraction(landmark_time)

                if landmarks:
                    # 4. EAR calculation
                    ear_start = time.time()
                    left_eye, right_eye = landmarks
                    left_ear = liveness_detector._calculate_ear(left_eye)
                    right_ear = liveness_detector._calculate_ear(right_eye)
                    if left_ear is not None and right_ear is not None:
                        ear = (left_ear + right_ear) / 2.0
                    ear_time = (time.time() - ear_start) * 1000
                    perf_diagnostics.record_ear_calculation(ear_time)

                    # 5. State update
                    state_start = time.time()
                    liveness_detector.update_temporal_state(ear)
                    state_time = (time.time() - state_start) * 1000
                    perf_diagnostics.record_state_update(state_time)

                    eye_state = liveness_detector.eye_state
                    blink_count = liveness_detector.blink_count
                else:
                    # No landmarks - update with None
                    liveness_detector.update_temporal_state(None)
                    perf_diagnostics.record_landmark_extraction(0)
                    perf_diagnostics.record_ear_calculation(0)
                    perf_diagnostics.record_state_update(0)
            else:
                # No face detected
                liveness_detector.update_temporal_state(None)
                perf_diagnostics.record_landmark_extraction(0)
                perf_diagnostics.record_ear_calculation(0)
                perf_diagnostics.record_state_update(0)

            # Log EAR data
            ear_logger.log(time.time() - session_start, ear, eye_state, blink_count, face is not None)

            # 6. Display
            display_start = time.time()
            perf_stats = perf_diagnostics.get_stats()
            display_frame = draw_diagnostic_overlay(
                frame, perf_stats, ear, eye_state, blink_count, ear_logger,
                liveness_detector.ear_close_threshold,
                liveness_detector.ear_open_threshold,
                show_detailed
            )
            cv2.imshow('Liveness Diagnostic', display_frame)
            display_time = (time.time() - display_start) * 1000
            perf_diagnostics.record_display(display_time)

            # Total frame time
            frame_time = (time.time() - frame_start) * 1000
            perf_diagnostics.record_total_frame(frame_time)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Exiting diagnostic mode...")
                break
            elif key in (ord('r'), ord('R')):
                liveness_detector.reset_temporal_state()
                ear_logger.entries.clear()
                print("\n[INFO] Reset: Blink count and data cleared")
            elif key in (ord('d'), ord('D')):
                show_detailed = not show_detailed
                print(f"\n[INFO] Detailed timing: {'ON' if show_detailed else 'OFF'}")
            elif key in (ord('s'), ord('S')):
                filepath = ear_logger.save_to_csv()
                if filepath:
                    print(f"\n[INFO] EAR data saved to: {filepath}")
                else:
                    print("\n[ERROR] Failed to save EAR data")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Stopped by user")

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Print diagnostic summary
        if 'perf_diagnostics' in locals():
            print("\n" + "=" * 80)
            print("DIAGNOSTIC SUMMARY")
            print("=" * 80)

            session_duration = time.time() - session_start
            stats = perf_diagnostics.get_stats()

            print(f"\nSession Duration: {session_duration:.1f}s")
            print(f"Total Frames: {total_frames}")
            print(f"Frames with Face: {frames_with_face}")
            print(f"Average FPS: {stats['fps']:.2f}")

            print("\n--- PERFORMANCE BREAKDOWN (milliseconds) ---")
            print(f"{'Component':<20} {'Avg':>8} {'Min':>8} {'Max':>8} {'P95':>8}")
            print("-" * 60)

            components = [
                ('Camera Capture', stats['camera_capture']),
                ('Face Detection', stats['face_detection']),
                ('Landmark Extract', stats['landmark_extraction']),
                ('EAR Calculation', stats['ear_calculation']),
                ('State Update', stats['state_update']),
                ('Display Render', stats['display']),
                ('TOTAL Frame Time', stats['total_frame'])
            ]

            for name, component_stats in components:
                print(f"{name:<20} {component_stats['avg']:>8.2f} {component_stats['min']:>8.2f} "
                      f"{component_stats['max']:>8.2f} {component_stats['p95']:>8.2f}")

            # Identify bottleneck
            print("\n--- BOTTLENECK ANALYSIS ---")
            max_avg_time = 0
            bottleneck = None
            for name, component_stats in components[:-1]:  # Exclude total
                if component_stats['avg'] > max_avg_time:
                    max_avg_time = component_stats['avg']
                    bottleneck = name

            if bottleneck:
                print(f"Primary bottleneck: {bottleneck} ({max_avg_time:.2f}ms avg)")

            # Frame time analysis
            total_avg = stats['total_frame']['avg']
            target_frame_time = 33.3  # 30 FPS
            if total_avg > target_frame_time:
                print(f"⚠ Average frame time ({total_avg:.1f}ms) exceeds 30 FPS target ({target_frame_time:.1f}ms)")
            else:
                print(f"✓ Average frame time ({total_avg:.1f}ms) meets 30 FPS target")

            print("\n" + "=" * 80)

        # Cleanup
        if camera is not None:
            camera.release()
        if detector is not None:
            detector.close()
        if liveness_detector is not None:
            liveness_detector.close()
        cv2.destroyAllWindows()
        print("[OK] Resources released\n")


if __name__ == "__main__":
    main()
