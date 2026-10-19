"""
Live spoof-check calibration (manual test - uses the webcam).

Shows the webcam with what the attendance spoof checks see, and measures the
anti-spoof "real face" score separately for each kind of presentation, using
the face boxes of BOTH face detectors (YuNet, used by the app now, and
MediaPipe BlazeFace, used before), because the anti-spoof model judges a crop
around the face box and a different box changes what it sees.

Keys (press the number when you START showing that presentation):
    1 = your real face     2 = a PHOTO of a face on a phone
    3 = a VIDEO of a face on a phone      4 = printed photo (optional)
    q / Esc = quit and print the results table

On exit a table shows, per presentation and detector: frames, mean / min /
max real-face score, and how often a phone/screen was seen covering the face.
A good ANTI_SPOOF_THRESHOLD is above the phone/photo scores and below the
real-face scores.

Run it with the Streamlit app CLOSED:
    .venv\\Scripts\\python.exe manual_tests\\spoof_check_live.py
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402 - sets the MSMF flag before cv2
import cv2  # noqa: E402

from src.computer_vision.face_detector import FaceDetector  # noqa: E402
from src.computer_vision.spoof_detection import AntiSpoofDetector, ScreenDetector  # noqa: E402

GREEN, RED, ORANGE, CYAN, WHITE = (0, 200, 0), (0, 0, 230), (0, 140, 255), (255, 200, 0), (255, 255, 255)
PHASES = {ord("1"): "real face", ord("2"): "phone photo", ord("3"): "phone video", ord("4"): "printed photo"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    detectors = {"YuNet": FaceDetector(backend="yunet"), "BlazeFace": FaceDetector(backend="mediapipe")}
    colors = {"YuNet": GREEN, "BlazeFace": CYAN}
    anti_spoof = AntiSpoofDetector()
    screens = ScreenDetector()
    if not anti_spoof.available:
        print("Anti-spoof models missing (MiniFASNetV2.onnx / MiniFASNetV1SE.onnx in the project root)")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Camera did not open - is the Streamlit app still running?")
        return

    threshold = config.ANTI_SPOOF_THRESHOLD
    phase = "real face"
    scores = defaultdict(list)          # (phase, detector) -> scores
    screen_hits = defaultdict(int)      # phase -> frames with a screen covering the face
    screen_frames = defaultdict(int)    # phase -> frames checked
    print(f"Phase: {phase}. Keys: 1 real face, 2 phone photo, 3 phone video, 4 printed photo, q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = cv2.flip(frame, 1)
        width = frame.shape[1]
        y_text = 25

        yunet_box = None
        for name, detector in detectors.items():
            faces = detector.detect_faces(frame)
            if len(faces) != 1:
                continue
            bbox = faces[0].bbox
            if name == "YuNet":
                yunet_box = bbox
            score = anti_spoof.real_score(frame, bbox)
            if score is None:
                continue
            scores[(phase, name)].append(score)
            x, y, w, h = bbox
            mx = width - x - w  # box position in the mirrored view
            cv2.rectangle(view, (mx, y), (mx + w, y + h), colors[name], 2)
            verdict = "real" if score >= threshold else "SPOOF"
            cv2.putText(view, f"{name}: {score:.2f} {verdict}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
            y_text += 24

        if yunet_box is not None and screens.available:
            covered = screens.screen_covering_face(frame, yunet_box)
            screen_frames[phase] += 1
            if covered:
                screen_hits[phase] += 1
                cv2.putText(view, f"FACE ON A {covered.upper()}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            for label, score, (sx, sy, sw, sh) in screens.detect_screens(frame):
                mx = width - sx - sw
                cv2.rectangle(view, (mx, sy), (mx + sw, sy + sh), ORANGE, 2)
                cv2.putText(view, f"{label} {score:.2f}", (mx, max(15, sy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)

        cv2.putText(view, f"showing: {phase.upper()}   (1 real  2 photo  3 video  4 print  q quit)",
                    (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.imshow("IAMS spoof calibration", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in PHASES:
            phase = PHASES[key]
            print(f"Phase: {phase}")

    cap.release()
    cv2.destroyAllWindows()
    print_table(scores, screen_hits, screen_frames, threshold)


def print_table(scores, screen_hits, screen_frames, threshold) -> None:
    print(f"\nAnti-spoof real-face scores (threshold {threshold}):")
    print(f"  {'presentation':<14} {'detector':<10} {'frames':>6} {'mean':>6} {'min':>6} {'p10':>6} "
          f"{'max':>6} {'below thr':>9}")
    for phase in PHASES.values():
        for name in ("YuNet", "BlazeFace"):
            values = sorted(scores.get((phase, name), []))
            if not values:
                continue
            print(f"  {phase:<14} {name:<10} {len(values):>6} {statistics.mean(values):>6.2f} "
                  f"{values[0]:>6.2f} {values[len(values) // 10]:>6.2f} {values[-1]:>6.2f} "
                  f"{sum(v < threshold for v in values) / len(values):>8.0%}")
    print("\nPhone/screen covering the face (screen detector):")
    for phase in PHASES.values():
        if screen_frames.get(phase):
            print(f"  {phase:<14} {screen_hits[phase]}/{screen_frames[phase]} frames "
                  f"({screen_hits[phase] / screen_frames[phase]:.0%})")


if __name__ == "__main__":
    main()
