"""
Live spoof-check viewer (manual test - uses the webcam).

Shows the webcam with what the attendance spoof checks see:
  - the face box and the anti-spoof "real face" score for this frame and the
    rolling mean the app decides on (config.ANTI_SPOOF_WINDOW frames),
    green = passes config.ANTI_SPOOF_THRESHOLD, red = suspected spoof
  - phone / laptop / TV boxes from the screen detector, and whether one
    covers the face (that blocks marking in the app)

Use it to calibrate: look at the camera yourself, then hold up a phone or
tablet playing a video (or showing a photo) of your face. On exit it prints
the score statistics so you can pick ANTI_SPOOF_THRESHOLD (set it in .env).

Run it with the Streamlit app CLOSED:
    .venv\\Scripts\\python.exe manual_tests\\spoof_check_live.py
Keys: q / Esc = quit, r = reset the statistics (e.g. before switching to the phone)
"""

import argparse
import os
import statistics
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402 - sets the MSMF flag before cv2
import cv2  # noqa: E402

from src.computer_vision.face_detector import FaceDetector  # noqa: E402
from src.computer_vision.spoof_detection import AntiSpoofDetector, ScreenDetector  # noqa: E402

GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 230), (0, 140, 255), (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    face_detector = FaceDetector()
    anti_spoof = AntiSpoofDetector()
    screens = ScreenDetector()
    if not anti_spoof.available:
        print("Anti-spoof models missing (MiniFASNetV2.onnx / MiniFASNetV1SE.onnx in the project root)")
    if not screens.available:
        print("Screen detector model missing (efficientdet_lite0.tflite in the project root)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Camera did not open - is the Streamlit app still running?")
        return

    threshold = config.ANTI_SPOOF_THRESHOLD
    window = deque(maxlen=max(1, config.ANTI_SPOOF_WINDOW))
    all_scores = []
    print(f"Threshold {threshold}, window {window.maxlen} frames. Press r to reset stats, q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()

        for label, score, (x, y, w, h) in screens.detect_screens(frame):
            cv2.rectangle(view, (x, y), (x + w, y + h), ORANGE, 2)
            cv2.putText(view, f"{label} {score:.2f}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

        faces = face_detector.detect_faces(frame)
        if len(faces) == 1:
            bbox = faces[0].bbox
            x, y, w, h = bbox
            score = anti_spoof.real_score(frame, bbox)
            covered = screens.screen_covering_face(frame, bbox) if screens.available else None
            if score is not None:
                window.append(score)
                all_scores.append(score)
            mean = sum(window) / len(window) if window else None
            passed = mean is not None and mean >= threshold and not covered
            color = GREEN if passed else RED
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            text = f"real {score:.2f}  mean {mean:.2f}" if score is not None else "no score"
            cv2.putText(view, text, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if covered:
                cv2.putText(view, f"FACE ON A {covered.upper()}", (x, y + h + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        else:
            window.clear()
            msg = "no face" if not faces else f"{len(faces)} faces - one at a time"
            cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        cv2.putText(view, f"threshold {threshold}  |  r = reset stats  q = quit",
                    (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.imshow("IAMS spoof check", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            if all_scores:
                print_stats(all_scores, threshold)
            all_scores.clear()
            window.clear()
            print("--- statistics reset ---")

    cap.release()
    cv2.destroyAllWindows()
    if all_scores:
        print_stats(all_scores, threshold)


def print_stats(scores, threshold) -> None:
    ordered = sorted(scores)
    print(f"frames {len(scores)}  mean {statistics.mean(scores):.3f}  min {ordered[0]:.3f}  "
          f"p10 {ordered[len(ordered) // 10]:.3f}  median {statistics.median(scores):.3f}  "
          f"max {ordered[-1]:.3f}  below threshold {sum(s < threshold for s in scores)}")


if __name__ == "__main__":
    main()
