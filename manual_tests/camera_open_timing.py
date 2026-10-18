"""
Camera open-time diagnostic (manual test - uses the webcam).

Measures where the time goes when OpenCV opens the camera on Windows:
the VideoCapture constructor, each property set() that CameraManager.open()
performs, and the first frame reads. Four variants are compared:

  1. MSMF, HW transforms off - OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
  2. DirectShow              - cv2.CAP_DSHOW, also built into OpenCV
  3. App                     - flag not preset; importing config.py sets it,
                               exactly as when the application starts
  4. MSMF, HW transforms on  - OpenCV's out-of-the-box behaviour (~15 s
                               here), run last because it is the slowest

Each variant runs in a fresh Python process, because the MSMF environment
flag only takes effect if it is set before cv2 is imported. Nothing in the
project is modified and no frames are saved.

Run it with the Streamlit app CLOSED so the camera is free:
    .venv\\Scripts\\python.exe manual_tests\\camera_open_timing.py
    .venv\\Scripts\\python.exe manual_tests\\camera_open_timing.py --camera 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

MSMF_FLAG = "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (title, backend, extra environment, import project config before cv2)
VARIANTS = [
    ("MSMF, HW transforms off", "CAP_MSMF", {MSMF_FLAG: "0"}, False),
    ("DirectShow", "CAP_DSHOW", {}, False),
    # Exactly how the app starts now: flag not preset, config.py sets it
    ("App (flag set by config.py)", "CAP_ANY", {}, True),
    # Slowest (~15 s), so it runs last
    ("MSMF, HW transforms on (OpenCV default)", "CAP_ANY", {MSMF_FLAG: "1"}, False),
]

# Same calls, same order as CameraManager.open() with its default arguments
APP_SETTINGS = [
    ("CAP_PROP_BUFFERSIZE", 1),
    ("CAP_PROP_FPS", 30),
    ("CAP_PROP_AUTO_EXPOSURE", 0.25),
]


def measure(backend: str, camera_index: int, import_config: bool) -> dict:
    """Open the camera once and time every step. Runs in the child process."""
    if import_config:
        sys.path.insert(0, PROJECT_ROOT)
        import config  # noqa: F401 - sets the MSMF flag, as in the app
    import cv2

    steps = []

    def timed(label, func):
        start = time.perf_counter()
        value = func()
        steps.append((label, (time.perf_counter() - start) * 1000.0))
        return value

    cap = timed("open (VideoCapture constructor)",
                lambda: cv2.VideoCapture(camera_index, getattr(cv2, backend)))
    if not cap.isOpened():
        return {"error": "camera did not open", "steps": steps}

    backend_name = cap.getBackendName()
    for prop, value in APP_SETTINGS:
        ok = timed(f"set {prop}={value}", lambda p=prop, v=value: cap.set(getattr(cv2, p), v))
        if not ok:
            steps[-1] = (steps[-1][0] + " (ignored)", steps[-1][1])

    ok, frame = timed("first frame read", cap.read)
    later_reads = []
    for _ in range(10):
        start = time.perf_counter()
        cap.read()
        later_reads.append((time.perf_counter() - start) * 1000.0)
    timed("release", cap.release)

    return {
        "backend": backend_name,
        "opencv": cv2.__version__,
        "frame_ok": bool(ok),
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if ok else "-",
        "brightness": round(float(frame.mean()), 1) if ok else None,
        "avg_read_ms": round(sum(later_reads) / len(later_reads), 1),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--child", metavar="BACKEND", help=argparse.SUPPRESS)
    parser.add_argument("--import-config", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(measure(args.child, args.camera, args.import_config)))
        return

    totals = []
    for title, backend, extra_env, import_config in VARIANTS:
        print(f"\n=== {title} ===")
        env = {k: v for k, v in os.environ.items() if k != MSMF_FLAG}
        env.update(extra_env)
        command = [sys.executable, __file__, "--child", backend, "--camera", str(args.camera)]
        if import_config:
            command.append("--import-config")
        try:
            proc = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=90,
            )
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print("  timed out after 90 s")
            continue
        except (json.JSONDecodeError, IndexError):
            print("  failed:", (proc.stderr or proc.stdout).strip()[-500:])
            continue

        for label, ms in result["steps"]:
            print(f"  {label:<42} {ms:>9.1f} ms")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        open_to_frame = sum(ms for label, ms in result["steps"] if label != "release")
        totals.append((title, open_to_frame))
        print(f"  {'TOTAL open -> first frame':<42} {open_to_frame:>9.1f} ms")
        print(f"  backend={result['backend']}  resolution={result['resolution']}  "
              f"brightness={result['brightness']}  avg later read={result['avg_read_ms']} ms  "
              f"(OpenCV {result['opencv']})")

    if totals:
        print("\n=== Summary: time until the first frame ===")
        for title, ms in sorted(totals, key=lambda item: item[1]):
            print(f"  {title:<40} {ms / 1000.0:>6.2f} s")


if __name__ == "__main__":
    main()
