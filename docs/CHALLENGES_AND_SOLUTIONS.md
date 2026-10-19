# IAMS: Challenges and Solutions

A study guide to every major problem met while building the Intelligent
Attendance Monitoring System, from the first computer-vision phase to the
real-time rebuild (May to October 2026). For each one: what we saw, the real
cause, how we found it, what we changed, and the result.

Sources: the git history, the fix reports in the project root
(`*_FIX_REPORT.md`, `PHASE_*` reports) and the measurements in
`data/logs/app.log`. Numbers are from the development laptop.

---

## Summary

| # | Challenge | Solution | Result |
|---|---|---|---|
| 1 | `face_recognition` would not import on Python 3.14 | Use Python 3.10 in a virtual environment | dlib recognition works |
| 2 | Real-time page lagged (full CV on every frame) | Run CV every 5th frame, show every frame | Smooth enough at the time |
| 3 | Page refreshed every 10 s; widgets and DB queries per frame | Longer batches, throttled queries, fewer widgets | Fewer refreshes |
| 4 | Head-turn challenge never started | Start the challenge automatically; show instructions | Liveness test works |
| 5 | Turned faces not recognised | Multi-angle enrollment (frontal, left, right) | Better recognition at angles |
| 6 | Two similar students could be confused | Ambiguity margin between best and runner-up | Unsure matches are "Unknown" |
| 7 | Student face data committed to git | Stop tracking it; fix `.gitignore` | Biometric data stays local |
| 8 | Camera took ~15 s to open | Disable MSMF hardware transforms | Opens in ~1 s |
| 9 | Recognition (~230 ms) blocked the preview | Recognise once, then follow the face | CV step 246 ms → 57 ms |
| 10 | Several people in frame: one person's blink verified everyone; liveness never reset | One person at a time; liveness per person | No shared or reused liveness |
| 11 | A video of a student passes blink and head turn | Anti-spoof model, phone detector, random timed test, audit photo | Replays blocked |
| 12 | Camera reads ~70 ms inside the app | Measured and isolated: the Streamlit process | Led to the rebuild (#15) |
| 13 | Blinks not detected | Liveness on every camera frame | No missed blinks |
| 14 | Head-turn test kept restarting | Identity rule + 2 s grace period | Turning keeps progress |
| 15 | Laggy preview inside Streamlit | Separate vision engine process + MJPEG video | Smooth preview |
| 16 | dlib too slow, blocked other threads | OpenCV YuNet + SFace | Embedding 239 ms → 4.8 ms |
| 17 | Engine load still slowed the camera | Throttle recognition, video-mode landmarks | Smooth camera |
| 18 | Marked-students table flickered and jumped | Fixed-height panel, static table | Stable page |
| 19 | Anti-spoof seemed switched off after the rebuild | Show spoof first; calibrate a per-frame rule | Phones blocked within ~1 s |

---

## Phase 1 to 6: building the system (May to October 2026)

### 1. `face_recognition` would not install / import
- **Symptom:** `ModuleNotFoundError: No module named 'pkg_resources'` when
  importing `face_recognition`.
- **Cause:** `face_recognition_models` (last updated 2018) uses
  `pkg_resources`, which Python 3.14 no longer provides.
- **How we found it:** the import traceback; reinstalling setuptools did not help
  (documented in `PHASE_2A_INSTALLATION_NOTES.txt`).
- **Solution:** run the project on Python 3.10 in `.venv` (pinned in
  `requirements.txt`, Windows `setup.bat` / `run.bat`).
- **Lesson:** old ML libraries often lag behind the newest Python; pin the
  Python version for the whole project.

### 2. Real-time attendance lagged (first version)
- **Symptom:** the Real-Time Attendance preview stuttered, while the Student
  Registry camera was smooth.
- **Cause:** detection, embedding, recognition, liveness and database lookups
  ran on **every** frame (30 times a second), and the preview waited for them.
- **How we found it:** comparing the two pages: the registry only processes a
  frame when you click Capture (`REALTIME_ATTENDANCE_LAG_FIX.md`).
- **Solution:** process every 5th frame, reuse the last result for the
  frames in between.
- **Lesson:** separate "show the video" from "analyse the video"; analysis
  does not need every frame (but see #13: liveness does).

### 3. Random page refreshes and per-frame overhead
- **Symptom:** the page reloaded every ~10 s; the camera "loaded like a
  terminal".
- **Cause:** the loop ran 300 frames then called `st.rerun()`; it also queried
  the database and recreated widgets on every frame
  (`CAMERA_SMOOTHNESS_FIX_REPORT.md`, `REALTIME_ATTENDANCE_BUGS_FIX_REPORT.md`).
- **Solution:** 900-frame batches, database queries once per second, update
  existing placeholders instead of creating widgets.
- **Lesson:** Streamlit re-runs scripts; anything done inside a frame loop is
  multiplied by the frame rate.

### 4. Liveness challenge never started
- **Symptom:** students could never pass liveness in real-time attendance.
- **Cause:** `start_new_challenge()` was only called on the Liveness Test page,
  never by the real-time service, and the UI showed no instructions
  (`LIVENESS_VERIFICATION_FIX_REPORT.md`).
- **How we found it:** searching the code for where challenges start.
- **Solution:** start the challenge automatically, require blinks **and** the
  challenge, show progress and instructions.

### 5. Turned faces not recognised
- **Symptom:** a student turning their head became "Unknown".
- **Cause:** each student was enrolled from one frontal photo.
- **Solution:** multi-angle enrollment (frontal, left, right) with coverage
  shown in the registry; configurable head-turn angle
  (`FACE_RECOGNITION_ROBUSTNESS_REPORT.md`).

### 6. Two similar students could be confused
- **Symptom risk:** a face almost equally close to two students was given to
  whichever was slightly closer.
- **Solution:** ambiguity margin: if the best match and the closest *other*
  student differ by less than `FACE_RECOGNITION_MARGIN` (0.1), the result is
  "Unknown". Later measured: under dlib two students were only 0.590 apart,
  under the 0.6 threshold, so the margin really was protecting the system.

### 7. Biometric data in git
- **Problem:** student face encodings were committed to the repository.
- **Solution:** stop tracking them and fix `.gitignore` for per-student
  folders, database journals and snapshots.
- **Lesson:** biometric data is personal data; it must never be pushed.

---

## Phase 7: real-time performance and security (October 2026)

### 8. Camera took ~15 s to open
- **Symptom:** clicking Start waited ~15 s before the first frame.
- **Cause:** Windows' camera backend (MSMF) spent ~7 s setting up hardware
  colour conversion when the camera opened, and ~7 s again when the FPS was set.
- **How we found it:** `manual_tests/camera_open_timing.py` timed every step
  (constructor, each `set()`, first read) for several backends.
- **Solution:** `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`, set in `config.py`
  before OpenCV is imported. DirectShow was tested too (1.7 s but slower frames).
- **Result:** 15.15 s → 0.96 s.
- **Lesson:** time each step separately; the slow part is rarely where you guess.

### 9. Recognition blocked the preview
- **Symptom:** the preview froze briefly several times per second.
- **Cause:** the dlib embedding took ~230 ms and ran on every processed frame.
- **Solution:** recognise a face once, then *follow* it while its box keeps
  overlapping (IoU); re-recognise when it is lost, jumps, a second face
  appears, or every 2 s. Safety: before marking, the face is recognised again
  on that exact frame.
- **Result:** average CV step 246 ms → 57 ms.

### 10. Several people in the frame
- **Question asked:** what happens with two people in front of the camera?
- **Found by reading the code:**
  - one liveness detector was shared by all faces, and the landmark model only
    tracked one face, so **one person's blink and head turn counted for
    everyone** in the frame (e.g. a photo held next to a live person);
  - liveness was **never reset** during a session: after the first student
    passed, everyone after them was marked without blinking.
- **Solution:** one person at a time (with two faces nobody is marked, and the
  screen says so); liveness restarts after each mark, when the face leaves, and
  when a different person is recognised.
- **Lesson:** state that belongs to a person must be reset when the person changes.

### 11. A video of a student passes the liveness test
- **Problem:** a recorded video of a student blinking and turning passes blink
  and head-turn checks; a friend on a video call could follow instructions.
- **Key insight:** a video already contains real 3D motion, so "software 3D"
  (depth from motion) cannot tell it from a real face. With an ordinary webcam
  the replay is betrayed by the **screen itself** (texture, moiré, glare, bezel).
- **Solution (all software, no new hardware):**
  - **anti-spoof model**: MiniFASNet (Silent-Face-Anti-Spoofing, ONNX, ~2 ms)
    classifies the face crop as real or photo/screen;
  - **phone/screen detector**: MediaPipe EfficientDet-Lite0 finds a phone,
    laptop or TV covering the face;
  - **random timed test**: 1-3 blinks and a left/right turn, chosen at random,
    within 10 s, so a pre-recorded video cannot know what to do;
  - **audit snapshot**: a small photo with every mark for teacher review.
- **Stretch idea not built:** a colour-flash challenge (the screen flashes
  random colours that a real face reflects), the only software defence against
  a live video-call relay.

### 12. Camera reads took ~70 ms inside the app
- **Symptom:** after #8 the preview ran at ~11 fps; `camera_read_frame`
  averaged ~70 ms (it was ~8 ms two days earlier).
- **How we investigated (measure, don't guess):**
  1. the log showed the slowdown started with the first session after #8;
  2. a standalone script reproduced the app's loop with every variable
     (hardware transforms on/off, worker thread, MediaPipe + dlib load,
     background reader); every variant read in ~7 ms;
  3. OpenCV's source showed MSMF reads are asynchronous (a queue of 3 frames),
     so a 70 ms read means frames *arrive* slowly, not that reading is slow;
  4. a camera thread inside the app still measured 68 ms between frames, and
     `st.image()` took 57 ms in the app vs 1.7 ms outside.
- **Conclusion:** the Streamlit process itself was the bottleneck (see #15).
- **Lesson:** reproduce the problem outside the app to split "the camera" from
  "the environment".

### 13. Blinks were not detected
- **Symptom:** students blinked and nothing happened.
- **Cause (simple arithmetic):** ~130 ms per loop and CV every 5th loop =
  eyes checked ~1.5 times per second; a blink lasts 100-250 ms, so most blinks
  fell between two checks.
- **Solution:** the eye/head check (MediaPipe Face Landmarker, ~10 ms) runs on
  **every** camera frame; slow recognition runs separately in the background.
- **Lesson:** the sampling rate must be faster than the event you want to see.

### 14. The head-turn test kept restarting
- **Symptom:** the turn instruction kept disappearing; the log showed 8 new
  challenges in 8 s.
- **Causes:** (1) while turning, recognition briefly said "Unknown" and the
  app treated that as a new person; (2) the face detector briefly lost the
  turned face and the app treated that as "person left".
- **Solution:** "Student A → Unknown → A" keeps progress (but "Unknown →
  student" and "A → B" still restart, for security); a face must be gone for
  2 s before the test restarts. "Require Head Turn" switch added.
- **Lesson:** a security rule that is too strict breaks the normal user flow;
  find the rule that blocks attacks but tolerates real behaviour.

### 15. Laggy preview inside Streamlit → separate vision engine
- **Cause:** Python runs only one thread's Python code at a time (the **GIL**).
  dlib held it for ~240 ms per face and Streamlit competed for it; Streamlit
  is also not built to send video frame by frame.
- **Options considered:** desktop app (Qt) - near-native but every page rewritten;
  streamlit-webrtc - new heavy dependencies, still in-process; **separate process**
  - chosen.
- **Solution:** `src/realtime/engine.py` runs camera, liveness, recognition,
  spoof checks and marking in its own process (own GIL). The video reaches the
  page as an **MJPEG** stream shown in an `<img>` tag; status and control use a
  small localhost HTTP API; instructions are drawn on the video. The engine can
  later serve a desktop or mobile front end too.
- **Engineering details met on the way:** the engine stops when the app exits
  (it watches its stdin; a raw `os.read` avoids a Python shutdown crash);
  `os.kill(pid, 0)` was avoided because on Windows it *terminates* the process.

### 16. dlib was too slow → YuNet + SFace
- **Cause:** dlib embedding ~240 ms per face, holding the GIL.
- **Solution:** OpenCV's **YuNet** detector (~7 ms, finds turned faces, gives 5
  landmarks) and **SFace** recogniser (~5 ms, OpenCV releases the GIL).
- **Data migration:** dlib and SFace vectors are not compatible. Each saved
  enrollment photo is re-encoded with SFace into `enc_sface_<id>.npy` next to
  the old file; nothing is overwritten, and `FACE_RECOGNITION_BACKEND=dlib`
  switches back.
- **Validated on the enrolled photos:** SFace separation gap +0.134 vs dlib
  +0.080 (dlib had two students within its match threshold).

### 17. The engine's own load slowed the camera
- **Symptom:** in the first engine test the camera still delivered ~17 fps.
- **Cause:** the recognition worker ran non-stop (~111 ms per cycle), the phone
  detector took 37 ms each time, and drawing the instruction bar blended the
  whole frame.
- **Solution:** recognition at most 5 times per second, landmarks in MediaPipe
  VIDEO mode (tracking), blend only the bar strip (10 ms → 0.4 ms), plus a
  timing probe in the log.
- **Result:** the user reported the camera as perfectly smooth.

### 18. The marked-students table flickered and jumped
- **Cause:** the status panel refreshed twice per second; the person text
  changed height and pushed the table up and down, and the interactive
  `st.dataframe` grid was rebuilt each time.
- **Solution:** fixed-height person panel, table moved below the video as a
  static `st.table`, refreshed every 2 s.

### 19. Anti-spoof seemed switched off after the rebuild
- **Symptoms:** a phone photo of a stranger showed "Unknown", a phone photo of
  a marked student showed "already marked", no spoof warning.
- **Causes:** (1) the display showed identity *before* the spoof result;
  (2) the decision rule was wrong for the real data.
- **How we found it:** a calibration tool scored every frame for real face,
  phone photo and phone video, with both detectors' face boxes:
  real faces scored ≥ 0.89 on every frame; phone frames flipped between ~0 and
  ~1 (43-66 % near 0). Averaging 10 frames (≥ 0.6) let phones pass about half
  the time. The detector's box made no difference.
- **Solution:** judge each frame: below 0.8 is a spoof frame; 2 spoof frames
  in the last 10 block the person until they leave; marking needs the last 5
  frames real. Spoof warnings are shown first.
- **Also learned:** the phone object detector rarely sees a phone held up to
  the camera (0-3 % of frames); the anti-spoof model does the main work.
- **Lesson:** calibrate thresholds on real measurements; averages hide
  bimodal (0-or-1) behaviour.

---

## Key concepts to study

| Concept | Meaning in this project |
|---|---|
| **Face embedding** | A vector of 128 numbers describing a face; same person = vectors close together |
| **Distance threshold / margin** | Max distance to accept a match / min gap to the next student |
| **EAR (Eye Aspect Ratio)** | Eye height ÷ width from landmarks; drops sharply during a blink |
| **Yaw** | Left/right head rotation, estimated from landmarks for the head-turn test |
| **IoU** | Overlap of two boxes (0-1); used to follow a face between frames |
| **Presentation attack** | Showing a photo, video, mask or screen instead of a real face |
| **APCER / BPCER** | Share of attacks accepted / real users rejected - the standard anti-spoofing metrics |
| **GIL** | Python's lock allowing one thread to run Python code at a time |
| **Process vs thread** | Separate processes each have their own GIL; threads share one |
| **MJPEG** | Video sent as a continuous series of JPEG images over HTTP |
| **MSMF / DirectShow** | Windows camera backends used by OpenCV |
| **Sampling rate** | How often something is measured; must beat the event duration (blinks) |

## General lessons

1. **Measure before changing.** Every real fix here started with a number
   (timings in the log, a standalone script, a calibration table).
2. **Reproduce in isolation** to separate the component from its environment.
3. **Read the library source** when behaviour is surprising (MSMF's frame
   queue, Streamlit's image encoding).
4. **Security and usability pull in opposite directions**; design rules that
   block attacks without punishing normal behaviour (identity rule, grace period).
5. **Keep a way back**: new models were added next to the old ones, behind a
   setting.
6. **Automated tests catch mistakes during big changes** (847 tests; one of
   them caught a duplicated method during the rebuild).
