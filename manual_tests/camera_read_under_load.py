"""
Camera read-time diagnostic under app-like load (manual test - uses the webcam).

In the Real-Time Attendance page camera_read_frame averages ~70 ms, while
manual_tests/camera_open_timing.py measures ~8 ms for the same camera. This
script reproduces the page's frame loop more closely to find what makes the
reads slow, running each variant in a fresh Python process (the MSMF flag only
takes effect before cv2 is imported):

  1. control      - HW transforms off, main thread, no CV work
  2. +cv          - adds the page's CV work: MediaPipe face detection and
                    face landmarks every 5th frame, a dlib face embedding
                    every 20th frame, mirror + RGB + JPEG encode every frame
  3. +cv, thread  - same, but opened and read in a worker thread while the
                    main thread idles (how Streamlit runs a page script)
  4. HW on        - variant 3 with OpenCV's default MSMF hardware transforms
                    (opens in ~15 s)
  5. reader       - variant 3 with a background thread that keeps reading
                    the camera; the loop just takes the newest frame

Each variant reports the read time, the loop frame rate, how many frames per
second the camera actually delivers and the average image brightness (dim
light makes webcams lengthen exposure and drop their frame rate).

Only project config and model files are read; nothing is written and no
project logger is used. Run it with the Streamlit app CLOSED:
    .venv\\Scripts\\python.exe manual_tests\\camera_read_under_load.py
    .venv\\Scripts\\python.exe manual_tests\\camera_read_under_load.py --frames 300
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

MSMF_FLAG = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (title, HW transforms flag, CV load, worker thread, reader thread)
VARIANTS = {
    "control": ("HW off, main thread, no CV", "0", False, False, False),
    "cv": ("HW off, main thread, + CV", "0", True, False, False),
    "cv_thread": ("HW off, worker thread, + CV (like the app)", "0", True, True, False),
    "hw_on": ("HW on, worker thread, + CV", "1", True, True, False),
    "reader": ("HW off, worker thread, + CV, background reader", "0", True, True, True),
}

CV_INTERVAL = 5          # realtime_attendance.py: cv_process_interval
EMBED_INTERVAL = 20      # dlib embedding; the app does it about every 60 frames
LOOP_SLEEP = 0.033       # realtime_attendance.py: time.sleep(0.033)


class LatestFrameReader:
    """Reads the camera continuously in a daemon thread; keeps the newest frame."""

    def __init__(self, capture):
        self._capture = capture
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._last_seq = 0
        self._running = True
        self.grab_times = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self._capture.read()
            now = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self.grab_times.append(now)
                self._cond.notify_all()

    def read(self, timeout=1.0):
        """Return a frame newer than the last one returned (waits if needed)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._last_seq, timeout)
            if self._frame is None:
                return False, None
            self._last_seq = self._seq
            return True, self._frame

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


def load_cv():
    """Load the same models the Real-Time page uses."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import face_recognition

    detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=os.path.join(PROJECT_ROOT, "blaze_face_short_range.tflite")),
        min_detection_confidence=0.5,
    ))
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=os.path.join(PROJECT_ROOT, "face_landmarker.task")),
        num_faces=1,
    ))
    return mp, detector, landmarker, face_recognition


def cv_work(cv, frame, index, timings):
    """Detection + landmarks every 5th frame, dlib embedding every 20th."""
    import cv2
    mp, detector, landmarker, face_recognition = cv
    if index % CV_INTERVAL:
        return
    start = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    faces = detector.detect(image).detections
    landmarker.detect(image)
    if index % EMBED_INTERVAL == 0:
        h, w = rgb.shape[:2]
        if faces:
            box = faces[0].bounding_box
            top, left = max(box.origin_y, 0), max(box.origin_x, 0)
            location = (top, min(left + box.width, w), min(top + box.height, h), left)
        else:  # no face in view: still pay for one embedding, centre box
            location = (h // 4, 3 * w // 4, 3 * h // 4, w // 4)
        face_recognition.face_encodings(rgb, known_face_locations=[location])
    timings.append((time.perf_counter() - start) * 1000.0)


def measure(variant: str, camera_index: int, frames: int) -> dict:
    """Run one variant. Executes in the child process."""
    _, _, with_cv, worker, use_reader = VARIANTS[variant]
    sys.path.insert(0, PROJECT_ROOT)
    import config  # noqa: F401 - same import order as the app (flag before cv2)
    import cv2

    cv = load_cv() if with_cv else None
    result = {}

    def body():
        # Same calls and order as CameraManager.open() with default arguments
        start = time.perf_counter()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            result["error"] = "camera did not open"
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.read()
        open_s = time.perf_counter() - start

        reader = LatestFrameReader(cap) if use_reader else None
        read = reader.read if reader else cap.read

        # Let auto-exposure settle before measuring
        settle_end = time.perf_counter() + 2.0
        while time.perf_counter() < settle_end:
            read()
        if reader:
            reader.grab_times.clear()

        reads, cv_times, brightness = [], [], []
        delivered = 0
        loop_start = time.perf_counter()
        for index in range(frames):
            t0 = time.perf_counter()
            ok, frame = read()
            reads.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                continue
            delivered += 1
            if with_cv:
                cv_work(cv, frame, index, cv_times)
                # Preview: mirror, RGB, encode (stand-in for st.image)
                display = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                cv2.imencode(".jpg", display)
            if index % 30 == 0:
                brightness.append(float(frame.mean()))
            time.sleep(LOOP_SLEEP)
        loop_s = time.perf_counter() - loop_start

        if reader:
            grabs = reader.grab_times
            camera_fps = (len(grabs) - 1) / (grabs[-1] - grabs[0]) if len(grabs) > 1 else 0.0
            reader.stop()
        else:
            camera_fps = None

        reads_sorted = sorted(reads)
        result.update({
            "backend": cap.getBackendName(),
            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "reported_fps": cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "open_s": round(open_s, 2),
            "read_avg": round(statistics.mean(reads), 1),
            "read_median": round(statistics.median(reads), 1),
            "read_p90": round(reads_sorted[int(len(reads_sorted) * 0.9)], 1),
            "read_max": round(reads_sorted[-1], 1),
            "loop_fps": round(delivered / loop_s, 1),
            "camera_fps": round(camera_fps, 1) if camera_fps is not None else None,
            "cv_avg": round(statistics.mean(cv_times), 1) if cv_times else None,
            "brightness": round(statistics.mean(brightness), 1) if brightness else None,
        })
        cap.release()

    if worker:
        thread = threading.Thread(target=body, name="ScriptRunner-like")
        thread.start()
        thread.join()
    else:
        body()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--frames", type=int, default=240,
                        help="loop iterations per variant (default 240)")
    parser.add_argument("--only", choices=list(VARIANTS), nargs="+",
                        help="run only these variants")
    parser.add_argument("--child", choices=list(VARIANTS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(measure(args.child, args.camera, args.frames)))
        return

    rows = []
    for name in args.only or list(VARIANTS):
        title, flag = VARIANTS[name][:2]
        print(f"\n=== {title} ===", flush=True)
        env = dict(os.environ, **{MSMF_FLAG: flag})
        command = [sys.executable, __file__, "--child", name,
                   "--camera", str(args.camera), "--frames", str(args.frames)]
        try:
            proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print("  timed out after 300 s")
            continue
        except (json.JSONDecodeError, IndexError):
            print("  failed:", (proc.stderr or proc.stdout).strip()[-800:])
            continue
        if "error" in result:
            print("  ERROR:", result["error"])
            continue
        print(f"  backend={result['backend']} {result['resolution']}  opened in {result['open_s']} s  "
              f"reported fps={result['reported_fps']}  auto_exposure={result['auto_exposure']}  "
              f"exposure={result['exposure']}  brightness={result['brightness']}")
        print(f"  read avg {result['read_avg']} ms  median {result['read_median']}  "
              f"p90 {result['read_p90']}  max {result['read_max']}")
        print(f"  loop {result['loop_fps']} fps  camera delivers "
              f"{result['camera_fps'] if result['camera_fps'] is not None else '-'} fps  "
              f"CV avg {result['cv_avg'] if result['cv_avg'] is not None else '-'} ms")
        rows.append((title, result))

    if rows:
        print("\n=== Summary ===")
        print(f"  {'variant':<48} {'read avg':>9} {'median':>7} {'loop fps':>9} {'bright':>7}")
        for title, r in rows:
            print(f"  {title:<48} {r['read_avg']:>7.1f}ms {r['read_median']:>7.1f} "
                  f"{r['loop_fps']:>9.1f} {r['brightness'] or 0:>7.1f}")


if __name__ == "__main__":
    main()
