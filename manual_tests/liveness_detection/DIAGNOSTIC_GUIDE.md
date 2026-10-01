# Liveness Detection Diagnostic Guide

## Purpose

This diagnostic tool measures **actual performance bottlenecks** and **real EAR values** to identify the root causes of:
1. Webcam stuttering during liveness detection
2. Missed natural blink detection

**Do not rely on assumptions.** Use this tool to gather concrete measurements before implementing fixes.

---

## Running the Diagnostic

### Command
```bash
python manual_tests/liveness_detection/diagnostic_test.py
```

### Controls
- **Q** or **ESC**: Exit and display performance summary
- **R**: Reset blink count and clear EAR data
- **D**: Toggle detailed timing breakdown (shows per-component performance)
- **S**: Save EAR values to CSV file for offline analysis

---

## What It Measures

### 1. Performance Breakdown (Milliseconds per Frame)
- **Camera Capture**: Time to read frame from webcam
- **Face Detection**: MediaPipe face detection time
- **Landmark Extraction**: Time to extract eye landmarks
- **EAR Calculation**: Time to compute Eye Aspect Ratio
- **State Update**: Time to update blink state machine
- **Display Render**: Time to draw overlay and show frame
- **TOTAL Frame Time**: End-to-end time per frame

### 2. Frame Rate
- **FPS**: Actual frames per second achieved
- **Target**: 30 FPS (33.3ms per frame)

### 3. EAR Values
- **Current EAR**: Real-time Eye Aspect Ratio
- **Thresholds**: Close threshold (0.21) and open threshold (0.25)
- **Eye State**: Current state (OPEN or CLOSED)
- **Blink Count**: Total blinks detected in session
- **EAR History Graph**: Visual plot of last 60 EAR values

### 4. Statistics
- Average, Min, Max, and P95 (95th percentile) for all timing metrics
- Total frames processed
- Frames with detected face

---

## Interpreting Results

### Identifying Webcam Stuttering

**Symptom**: Preview freezes briefly during operation

**Diagnostic Steps**:
1. Run diagnostic with **D** (detailed mode) enabled
2. Observe the timing breakdown in real-time
3. Check the performance summary after exiting

**Look for**:
- Which component has the **highest average time**?
- Are there **spikes in P95 or Max times** for any component?
- Is **Display Render** time unusually high (>15ms)?
- Is **Face Detection** or **Landmark Extraction** slow (>30ms)?

**Expected Values for Smooth 30 FPS**:
- Camera Capture: <5ms
- Face Detection: <20ms
- Landmark Extraction: <15ms
- EAR Calculation: <1ms
- State Update: <1ms
- Display Render: <10ms
- **TOTAL: <33ms**

**Root Causes**:
- If **Camera Capture** is slow: USB bandwidth issue or driver problem
- If **Face Detection** is slow: MediaPipe model inference bottleneck
- If **Landmark Extraction** is slow: MediaPipe Face Landmarker bottleneck
- If **Display Render** is slow: OpenCV rendering or overlay drawing bottleneck
- If **P95 times spike**: Intermittent GC pauses or system interruptions

---

### Analyzing Missed Blinks

**Symptom**: Natural blinks not detected; need to close eyes forcefully

**Diagnostic Steps**:
1. Run diagnostic script
2. Blink naturally while watching the HUD overlay
3. Observe EAR values during your blinks
4. Press **S** to save CSV data for analysis
5. Check the EAR history graph (bottom-right mini graph)

**Look for**:
- What is the **typical EAR range when eyes are open**? (e.g., 0.25-0.35)
- What is the **EAR range when eyes are closed during a natural blink**? (e.g., 0.15-0.20)
- Does EAR drop **below 0.21** (close threshold) during natural blinks?
- Does EAR rise **above 0.25** (open threshold) when eyes reopen?
- Is there **sufficient separation** between open and closed EAR values?

**Expected Behavior**:
- **Open eyes**: EAR typically 0.25-0.40
- **Closed eyes (natural blink)**: EAR typically 0.10-0.20
- **Close threshold (0.21)**: Should trigger reliably during natural closure
- **Open threshold (0.25)**: Should trigger reliably when eyes reopen

**Root Causes**:
- **EAR never drops below 0.21**: Threshold too low; natural blinks don't reach it
- **EAR fluctuates around 0.21**: Noisy landmarks; needs smoothing or hysteresis adjustment
- **EAR drops but state doesn't change**: State machine logic issue
- **Blink detected but counter doesn't increment**: Blink counting logic issue

**CSV Analysis**:
- Open the saved CSV file in Excel or Python
- Plot `ear` vs `timestamp` to see temporal behavior
- Identify gaps or patterns during missed blinks

---

## Common Findings

### Finding 1: MediaPipe Inference is the Bottleneck
**Evidence**: Face Detection + Landmark Extraction > 25ms average

**Implication**: Webcam stuttering is caused by synchronous MediaPipe inference blocking the camera loop.

**Potential Fixes**:
- Reduce MediaPipe processing frequency (skip frames)
- Use async/threaded inference (requires thread-safe queue)
- Lower MediaPipe model precision or resolution

---

### Finding 2: EAR Values Don't Cross Thresholds
**Evidence**: Natural blinks produce EAR values like 0.22-0.24, which don't drop below 0.21

**Implication**: Close threshold (0.21) is too low for this user/camera/lighting combination.

**Potential Fixes**:
- Adjust `LIVENESS_EAR_CLOSE_THRESHOLD` based on observed values
- Implement adaptive thresholding based on baseline calibration
- Use relative EAR change detection instead of absolute thresholds

---

### Finding 3: EAR Values Are Noisy
**Evidence**: EAR fluctuates rapidly (e.g., 0.26 → 0.20 → 0.27 in 3 frames)

**Implication**: Landmark detection noise causes false state transitions.

**Potential Fixes**:
- Enable `LIVENESS_EAR_SMOOTHING_WINDOW` (e.g., 3-5 frames)
- Increase `LIVENESS_EAR_CONSECUTIVE_FRAMES` to require sustained closure
- Widen hysteresis band (increase difference between close and open thresholds)

---

### Finding 4: Display Rendering is Slow
**Evidence**: Display Render > 15ms average

**Implication**: OpenCV overlay drawing is too complex or inefficient.

**Potential Fixes**:
- Simplify overlay graphics
- Reduce text rendering calls
- Use smaller overlay region
- Pre-render static elements

---

## Next Steps After Diagnosis

1. **Record the measurements** from the performance summary
2. **Save EAR data** if investigating missed blinks (press **S**)
3. **Identify the actual bottleneck** using the guide above
4. **Report findings** before implementing any fixes
5. **Only fix the measured root cause**, not speculative issues

---

## Example Diagnostic Session

### Session Output
```
Session Duration: 30.5s
Total Frames: 892
Frames with Face: 845
Average FPS: 29.3

--- PERFORMANCE BREAKDOWN (milliseconds) ---
Component              Avg      Min      Max      P95
------------------------------------------------------------
Camera Capture        2.34     1.89     8.12     3.21
Face Detection       18.45    15.23    42.67    22.15
Landmark Extract     12.34    10.11    35.89    16.78
EAR Calculation       0.23     0.18     0.89     0.34
State Update          0.12     0.08     0.45     0.18
Display Render       11.23     8.45    28.34    14.56
TOTAL Frame Time     44.71    38.23    98.45    52.34

--- BOTTLENECK ANALYSIS ---
Primary bottleneck: Face Detection (18.45ms avg)
⚠ Average frame time (44.71ms) exceeds 30 FPS target (33.3ms)
```

### Interpretation
- **Total frame time (44.71ms) > 33.3ms**: Cannot maintain 30 FPS
- **Face Detection (18.45ms)** + **Landmark Extract (12.34ms)** = 30.79ms
- **Bottleneck**: MediaPipe inference (face detection + landmarks)
- **Max spike (98.45ms)**: Occasional severe stutter
- **Display is also high (11.23ms)**: Secondary optimization target

### Recommended Action
1. Reduce MediaPipe processing frequency (process every 2nd frame)
2. Simplify display overlay to reduce render time
3. Investigate P95/Max spikes (possible GC or system interruption)

---

## Questions to Answer

Use the diagnostic to answer:
1. **What is the actual FPS achieved?**
2. **Which component takes the most time?**
3. **Do frame times spike? When?**
4. **What are typical open-eye EAR values?**
5. **What are typical closed-eye EAR values during natural blinks?**
6. **Do natural blinks cross the close threshold (0.21)?**
7. **Does the eye state change correctly during blinks?**
8. **Is there landmark detection noise or dropout?**

Do not implement fixes without answering these questions first.
