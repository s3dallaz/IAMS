# Liveness Detection Manual Testing

This directory contains manual testing scripts for Phase 3B Liveness Detection.

## Test Script

### `test_liveness.py`

Tests the LivenessDetector with a real webcam in continuous real-time mode.

**Features:**
- Real-time webcam processing at steady 30 FPS without stuttering or video freezing
- Optimized console output - only logs state changes (no per-frame or per-blink printing)
- Incremental single-frame landmark & EAR extraction (no redundant batch reprocessing)
- Simplified two-state blink detection (OPEN ↔ CLOSED) with hysteresis
- Hysteresis thresholds: `ear_close_threshold=0.21`, `ear_open_threshold=0.25`
- Live HUD overlay showing current state, real-time EAR, blinks detected, confidence, and FPS
- Monotonic session blink accumulation and responsive instant reset on keypress

**Requirements:**
- Webcam connected and accessible
- MediaPipe Face Landmarker model (`face_landmarker.task`) in project root

## Running the Test

From the project root directory:

```bash
python manual_tests/liveness_detection/test_liveness.py
```

## Controls

- **Q** or **ESC**: Exit the test and print session summary
- **R**: Reset the sequence, EAR history, and blink count
- **V**: Toggle verbose console output on/off (periodic detailed logging every 60 frames)

## Display Information

The live video feed shows a HUD overlay with:
- **State**: Current liveness state (LIVE, SPOOF, or UNKNOWN)
  - Green: LIVE (sufficient blinks detected)
  - Orange: UNKNOWN (insufficient frames or blinks)
  - Red: SPOOF (placeholder for Phase 3C)
- **EAR**: Real-time Eye Aspect Ratio value and status (`[OPEN]` or `[CLOSED]`)
- **Blinks**: Number of complete blinks detected (and required threshold)
- **Confidence**: Liveness confidence score (0.0 - 1.0)
- **Face Frames**: Number of face frames processed in active session
- **FPS**: Live frame rate measurement
- **Face Bounding Box**: Color-coded box around detected face

## Testing Scenarios

### 1. Live Person with Natural Blinking
**Procedure:**
1. Position your face in frame
2. Blink naturally 1-2 times
3. Observe the state change to LIVE

**Expected Result:**
- State: LIVE (green)
- Blinks Detected: 1 or more
- Confidence: 0.75 or higher

### 2. Live Person Keeping Eyes Open
**Procedure:**
1. Position your face in frame
2. Keep eyes open without blinking
3. Observe the state remains UNKNOWN

**Expected Result:**
- State: UNKNOWN (orange)
- Blinks Detected: 0
- Confidence: 0.0
- Reason: Insufficient blinks

### 3. Face Loss and Recovery
**Procedure:**
1. Start with face in frame
2. Move out of frame for 2-3 seconds
3. Return to frame

**Expected Result:**
- Frame gracefully reports no face detected
- Resumes detection smoothly upon face return
- Retains session blink count

### 4. Reset Sequence
**Procedure:**
1. Accumulate blinks to reach LIVE state
2. Press 'R' to reset

**Expected Result:**
- State resets to UNKNOWN
- Blinks Detected resets to 0
- Instant visual feedback

## Understanding the Results

### LIVE State
- Appears when **1 or more complete blinks** are detected AND **at least 10 face frames** have been observed.
- Complete blink = direct transition with hysteresis: OPEN (EAR ≥ 0.25) → CLOSED (EAR < 0.21) → OPEN (EAR ≥ 0.25).
- With `ear_consecutive_frames=1`, single-frame closures are detected (optimized for fast natural blinks).
- Confidence scales up with additional blinks.

### UNKNOWN State
- Appears when:
  - Fewer than 10 face frames observed
  - No blinks detected yet
  - Insufficient facial landmarks

## Known Behavior & Limitations

### What Works
✓ Detects fast, natural human blinks in real-time (100-250ms duration)
✓ Smooth webcam preview without stuttering or freezing
✓ Minimal console output to avoid I/O blocking the camera loop
✓ Prevents oscillation at threshold boundary using hysteresis
✓ Simplified two-state machine (OPEN ↔ CLOSED) for reliable fast blink detection
✓ Handles prolonged eye closures (only counted as 1 blink upon reopening)
✓ Handles temporary face loss gracefully

### What Doesn't Work (Phase 3B Limitations)
✗ **Cannot detect video replay attacks** (e.g. video of a blinking person)
✗ Cannot detect 3D masks or printed photos moving across the frame
✗ May classify a very still person keeping eyes open as UNKNOWN (not SPOOF)
- Additional anti-spoofing measures (texture analysis, motion analysis, challenge-response) will be added in Phase 3C.
