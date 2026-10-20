# Real-Time Attendance Rebuild: Performance and Anti-Spoofing

This document records the October 2026 rebuild of the Real-Time Attendance
pipeline: what was measured, what was changed and why. It is written to be
reused in the project report.

## 1. Problems measured before the rebuild

All numbers come from the application's own performance log
(`data/logs/app.log`, `PERFORMANCE SUMMARY`) during live sessions on the
development laptop (640x480 webcam).

| Measurement | Value | Effect |
|---|---|---|
| Camera frame interval inside the Streamlit process | 68 ms (~15 fps); the same camera gives 30 fps in a plain Python process | choppy preview |
| `st.image()` per frame inside Streamlit | 42-57 ms (1.7 ms outside Streamlit) | preview capped at ~13 fps |
| dlib face embedding | 233-239 ms per face, holding Python's GIL | preview froze during recognition |
| Eye (blink) sampling rate | ~1.5 times per second (every 5th of ~8 frames/s) | blinks of 100-250 ms were mostly missed |
| Head-turn challenge | restarted 8 times in 8 s | the detector lost the turned face and the app treated it as "person left" |

Root cause: everything (camera, recognition, liveness, video display) ran in
one Python thread inside the Streamlit server process. Python lets only one
thread run Python code at a time (the GIL), dlib and Streamlit both compete
for it, and Streamlit is not designed to transport video frame by frame.

## 2. New architecture

```
 Streamlit app (UI only)                    Vision engine process (src/realtime/engine.py)
 ┌──────────────────────────┐   HTTP     ┌──────────────────────────────────────────────┐
 │ Real-Time Attendance page│──────────▶ │ camera reader thread ─▶ newest frame          │
 │  <img src=stream.mjpg> ◀─┼── MJPEG ── │ main loop, every frame:                       │
 │  status panel (2x/s) ◀───┼── JSON ─── │   blink + head-turn check (MediaPipe, ~10 ms) │
 │  Start / Stop / settings ┼── POST ──▶ │   instruction bar drawn on the video          │
 └──────────────────────────┘ localhost  │ recognition worker thread:                    │
                                         │   YuNet detection, SFace recognition,         │
            SQLite database ◀────────────│   anti-spoof checks, attendance marking       │
                                         └──────────────────────────────────────────────┘
```

- **Separate process.** The camera and all computer-vision work run in their
  own Python process, with its own GIL, so the Streamlit server can no longer
  slow them down. Started and stopped by `src/realtime/engine_client.py`;
  it stops automatically if the app exits.
- **MJPEG video.** The engine serves the annotated video as an MJPEG stream on
  `127.0.0.1`. The page shows it with an `<img>` tag; the browser plays it
  natively at the camera frame rate. Streamlit no longer handles frames, and
  the page stays responsive while streaming.
- **Instructions on the video.** "Alice: Blink 2 times (1/2) + Turn head LEFT (7s)",
  "MARKED: Alice (PRESENT)", "SPOOF SUSPECTED", "Multiple faces..." are drawn
  on the video itself, so feedback is immediate. The side panel refreshes
  twice per second (Streamlit fragment).
- **Local HTTP API.** `/stream.mjpg`, `/status.json`, `POST /settings`,
  `POST /stop` (token-protected). A future desktop or mobile front end can use
  the same engine without changing the vision code.
- **Threads inside the engine.** A camera reader thread always holds the newest
  frame; per-frame liveness runs on every frame; recognition and marking run in
  a background worker on the newest frame whenever it is free.
- **Load control.** Recognition runs at most 5 times per second
  (`RECOGNITION_INTERVAL`), the phone/screen check every 0.5 s, and the face
  landmarker runs in MediaPipe's VIDEO mode (tracks the face between frames).
  A busy worker otherwise starved the camera thread: in the first engine test
  the camera still delivered only ~17 fps while the worker ran non-stop.
- **Stable page layout.** The person panel has a fixed height and the
  marked-students table is a static table refreshed every 2 s, so the
  twice-per-second status refresh no longer makes the page jump.

## 3. Technology choices

| Task | Before | After | Why |
|---|---|---|---|
| Face detection | MediaPipe BlazeFace short-range | **OpenCV YuNet** (`face_detection_yunet_2023mar.onnx`, 0.2 MB) | ~7 ms, detects turned faces better, gives 5 landmarks for alignment |
| Face recognition | dlib ResNet via `face_recognition` (~240 ms) | **OpenCV SFace** (`face_recognition_sface_2021dec.onnx`, 37 MB) | ~5 ms; OpenCV releases the GIL, so other threads keep running; similar LFW accuracy (99.4 %) |
| Liveness landmarks | MediaPipe Face Landmarker | unchanged, but on **every** frame | blinks are no longer missed |
| Anti-spoofing | none | **MiniFASNetV2 + MiniFASNetV1SE** (Silent-Face-Anti-Spoofing, ONNX, ~2 ms) | rejects photos and screen replays |
| Phone/screen detection | none | **MediaPipe EfficientDet-Lite0** (COCO "cell phone", "laptop", "tv") | blocks a face shown on a device |
| Video to browser | `st.image()` per frame | **MJPEG over HTTP** from the engine | no per-frame Streamlit messages |
| Runtime | one Streamlit thread | **engine process + threads** | no GIL contention with Streamlit |

All models are open source (Apache-2.0 / MIT) and run on the CPU through
libraries the project already used (OpenCV, MediaPipe); no new Python
packages were added.

Recognition accuracy on the project's own enrollment photos (5 students,
19 photos, distances between every pair of photos):

| Model (threshold) | Same student, largest distance | Different students, smallest distance | Gap |
|---|---|---|---|
| SFace (1.128) | 1.156 | 1.290 | +0.134 |
| dlib (0.6) | 0.510 | 0.590 | +0.080 |

Under dlib two different students came within 0.590, below the 0.6 match
threshold; only the ambiguity margin prevented a wrong match. SFace keeps
different students further apart.

## 4. Liveness and anti-spoofing design

One liveness test per person, all of which must pass before attendance is marked:

1. **Random blinks**: 1-3 blinks (random per test), counted on every camera frame.
2. **Random head turn**: left or right (random), unless "Require Head Turn" is
   switched off on the page.
3. **Time limit**: blinks and turn in any order within 10 s; otherwise a new
   random test starts. A pre-recorded video cannot know the test in advance.
4. **Anti-spoof model, judged per frame**: a processed frame scoring below 0.8
   is a spoof frame; 2 spoof frames among the last 10 block the person;
   marking needs the last 5 frames all above 0.8.
   **Recheck**: a low-quality webcam (blur, noise, low light) gives a real
   face a few low frames, so a block is checked again for up to 10 s: it is
   lifted when at most 1 of the next 15 frames is low and the last 5 are all
   real. A photo scores low on nearly every frame and a phone on about half,
   so they keep failing. Up to 2 rechecks per person in view; after that, or
   a timed-out recheck, the block lasts until the person leaves the view.
5. **No screen**: a phone, laptop or TV covering the face blocks marking until
   the person leaves the view.
6. **One person at a time**: with two faces in view nobody is marked.

A suspected spoof is shown first on the video and in the side panel, whether
the face is unknown, already marked or still doing the liveness test.

### Anti-spoof calibration

Measured with `manual_tests/spoof_check_live.py` on the development webcam,
scoring every frame with the face boxes of both detectors:

| Presentation | Detector | Frames | Mean | Min | Frames below 0.6 |
|---|---|---|---|---|---|
| Real face | YuNet | 198 | 1.00 | 0.89 | 0 % |
| Real face | BlazeFace | 201 | 1.00 | 0.97 | 0 % |
| Phone photo | YuNet | 270 | 0.60 | 0.00 | 43 % |
| Phone photo | BlazeFace | 317 | 0.57 | 0.00 | 46 % |
| Phone video | YuNet | 244 | 0.46 | 0.00 | 64 % |
| Phone video | BlazeFace | 308 | 0.40 | 0.00 | 66 % |

Findings:
- The detector's face box does not matter (YuNet and BlazeFace score alike).
- A real face scores close to 1.0 on every frame; a face on a phone flips
  between about 0 and 1 from frame to frame. The first rule (average of the
  last 10 frames >= 0.6) therefore let phones through about half the time.
  The per-frame rule above follows from this: no real-face frame fell below
  0.89, while about half of the phone frames fall near 0, so two low frames
  in ten identify a phone within about a second (98 % per 10-frame window for
  the photo, at 5 checks per second).
- The phone/screen object detector rarely recognises a phone held up to the
  camera (0 % of photo frames, 3 % of video frames); it remains as an extra
  check for visible devices, while the anti-spoof model does the main work.

Per-person state: the test restarts when another student is recognised, when an
unrecognised face turns into a student, when the face jumps to another place,
or when the face is gone for more than 2 s. Recognition briefly failing while
the student turns their head ("Unknown" between two recognitions of the same
student) does **not** restart the test.

Every mark saves an **audit snapshot** (`data/attendance_snapshots/session_<id>/`)
for teacher review.

## 5. Face data migration (dlib to SFace)

dlib and SFace embeddings are not comparable. Each enrollment sample's photo
(`data/enrollment_images/<student>/face_<id>.jpg`) is re-encoded with SFace into
`enc_sface_<id>.npy`, next to the existing dlib `enc_<id>.npy`. This happens
automatically the first time the gallery loads; nothing is overwritten or
deleted. New enrollments are stored as SFace directly.

## 6. Configuration (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `FACE_DETECTOR_BACKEND` | `yunet` | `mediapipe` = previous detector |
| `FACE_RECOGNITION_BACKEND` | `sface` | `dlib` = previous recogniser (uses the old encodings) |
| `SFACE_RECOGNITION_THRESHOLD` / `_MARGIN` | 1.128 / 0.1 | match distance / ambiguity margin |
| `LIVENESS_REQUIRE_CHALLENGE` | `true` | default of the "Require Head Turn" switch |
| `LIVENESS_MIN_BLINKS`, `LIVENESS_MAX_BLINKS` | 1, 3 | random blink range |
| `LIVENESS_CHALLENGE_TIMEOUT` | 10 | seconds per liveness test |
| `LIVENESS_FACE_LOST_GRACE` | 2 | seconds a face may be lost before the test restarts |
| `ANTI_SPOOF_THRESHOLD`, `_BAD_FRAMES`, `_MIN_FRAMES`, `_WINDOW` | 0.8, 2, 5, 10 | per-frame anti-spoof decision |
| `ANTI_SPOOF_RECHECK_FRAMES`, `_RECHECK_MAX_BAD`, `_RECHECK_TIMEOUT`, `ANTI_SPOOF_MAX_RECHECKS` | 15, 1, 10 s, 2 | recheck of a model block (0 rechecks = none) |
| `SPOOF_DIAGNOSTICS` | `true` | log a SPOOF CHECK line per second (scores, screens) |
| `SCREEN_DETECTION_CONFIDENCE`, `SCREEN_FACE_OVERLAP` | 0.4, 0.5 | phone/screen check |
| `ATTENDANCE_SNAPSHOTS_ENABLED` | `true` | audit photos |
| `RECOGNITION_INTERVAL` | 0.2 | seconds between recognition runs in the engine |
| `SCREEN_CHECK_INTERVAL` | 0.5 | seconds between phone/screen checks |

## 7. Results after the rebuild

Bench measurements (development laptop, CPU only):

| Step | Before | After |
|---|---|---|
| Face detection | BlazeFace ~5 ms | YuNet 6.7 ms (also turned faces) |
| Face embedding | dlib 233-239 ms | SFace 4.8 ms |
| Recognition worker per frame | 64 ms average | 6.9 ms (no face) |
| Eye sampling | ~1.5 per second | every camera frame |

Live session, first engine version (before load control): preview 16-17 fps,
camera frame interval 59 ms, face embedding 6.7 ms (was 239 ms), recognition
worker 111 ms per cycle running non-stop, per-frame liveness 15 ms.

Live session measurements after load control are taken from `ENGINE
PERFORMANCE` lines in `data/logs/app.log` and should be added here.
