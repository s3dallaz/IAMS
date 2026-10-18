"""
Camera read-time diagnostic inside Streamlit (manual test - uses the webcam).

manual_tests/camera_read_under_load.py measures ~7 ms per camera read in a
plain Python process (with or without the app's CV work, in a worker thread,
HW transforms on or off), while the Real-Time Attendance page logs ~70 ms.
This minimal Streamlit page runs the same frame loop as that page (read,
optional CV work, mirror, st.image, sleep 33 ms) inside a Streamlit server,
in two modes:

  direct  - camera.read() in the loop, as CameraManager does today
  reader  - a background thread keeps reading; the loop takes the newest frame

It reports the read time, loop frame rate, how many frames per second the
camera delivers, brightness and the exposure values the driver reports.
Results are shown on the page and printed to the terminal. Nothing is written.

Run it with the IAMS app CLOSED (the camera must be free):
    .venv\\Scripts\\python.exe -m streamlit run manual_tests\\camera_read_streamlit.py
To compare with OpenCV's default hardware transforms (camera opens in ~15 s):
    $env:OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS="1"; .venv\\Scripts\\python.exe -m streamlit run manual_tests\\camera_read_streamlit.py
"""

import os
import statistics
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402,F401 - sets the MSMF flag before cv2, as in the app
import cv2  # noqa: E402
import streamlit as st  # noqa: E402

from camera_read_under_load import LatestFrameReader, cv_work, load_cv  # noqa: E402

FRAMES = 200
LOOP_SLEEP = 0.033


@st.cache_resource
def get_cv():
    return load_cv()


def open_camera():
    """Same calls and order as CameraManager.open() with default arguments."""
    start = time.perf_counter()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, 0.0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.read()
    return cap, time.perf_counter() - start


def run_mode(mode, cv, preview, status):
    cap, open_s = open_camera()
    if cap is None:
        return {"mode": mode, "error": "camera did not open"}
    reader = LatestFrameReader(cap) if mode == "reader" else None
    read = reader.read if reader else cap.read

    settle_end = time.perf_counter() + 2.0  # let auto-exposure settle
    while time.perf_counter() < settle_end:
        read()
    if reader:
        reader.grab_times.clear()

    reads, cv_times, brightness = [], [], []
    loop_start = time.perf_counter()
    for index in range(FRAMES):
        t0 = time.perf_counter()
        ok, frame = read()
        reads.append((time.perf_counter() - t0) * 1000.0)
        if not ok:
            continue
        if cv is not None:
            cv_work(cv, frame, index, cv_times)
        preview.image(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB),
                      channels="RGB", width="stretch")
        if index % 30 == 0:
            brightness.append(float(frame.mean()))
            status.caption(f"{mode}: frame {index}/{FRAMES}")
        time.sleep(LOOP_SLEEP)
    loop_s = time.perf_counter() - loop_start

    camera_fps = None
    if reader:
        grabs = reader.grab_times
        if len(grabs) > 1:
            camera_fps = round((len(grabs) - 1) / (grabs[-1] - grabs[0]), 1)
        reader.stop()

    result = {
        "mode": mode,
        "HW transforms": os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"),
        "open s": round(open_s, 2),
        "read avg ms": round(statistics.mean(reads), 1),
        "read median ms": round(statistics.median(reads), 1),
        "read min ms": round(min(reads), 1),
        "read max ms": round(max(reads), 1),
        "loop fps": round(FRAMES / loop_s, 1),
        "camera fps": camera_fps,
        "CV avg ms": round(statistics.mean(cv_times), 1) if cv_times else None,
        "brightness": round(statistics.mean(brightness), 1) if brightness else None,
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "thread": threading.current_thread().name,
    }
    cap.release()
    return result


st.title("Camera read timing inside Streamlit")
st.caption(f"OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS="
           f"{os.environ.get('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS')}  |  OpenCV {cv2.__version__}")
with_cv = st.checkbox("Run the app's CV work (MediaPipe + dlib)", value=True)

if st.button("Run test (about 30 s)"):
    cv = get_cv() if with_cv else None
    status = st.empty()
    preview = st.empty()
    results = []
    for mode in ("direct", "reader"):
        result = run_mode(mode, cv, preview, status)
        print("CAMERA-TEST", result, flush=True)
        results.append(result)
    status.empty()
    preview.empty()
    st.dataframe(results)
    st.code("\n".join(str(r) for r in results))
