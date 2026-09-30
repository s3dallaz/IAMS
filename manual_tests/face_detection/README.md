# Manual Face Detection Testing

This directory contains a standalone testing setup for the face detection system. It tests the MediaPipe FaceDetector on real images and displays the results with bounding boxes.

## Directory Structure

```
manual_tests/face_detection/
├── README.md                    # This file
├── test_detection.py            # Testing script
├── single_face/                 # Images with one face
│   ├── person1.jpg
│   ├── photo2.jpg
│   └── ...
├── multiple_faces/              # Images with multiple faces
│   ├── group1.jpg
│   ├── classroom.jpg
│   └── ...
└── no_face/                     # Images without faces
    ├── landscape.jpg
    ├── object.jpg
    └── ...
```

## How to Use

### 1. Prepare Test Images

Place your test images in the appropriate folders:

- **single_face/**: Images containing exactly one face
- **multiple_faces/**: Images containing multiple faces (groups, classroom photos, etc.)
- **no_face/**: Images without any faces (landscapes, objects, etc.)

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

### 2. Run the Test

From the project root directory:

```bash
.venv\Scripts\python.exe manual_tests\face_detection\test_detection.py
```

Or on Unix/macOS:
```bash
.venv/bin/python manual_tests/face_detection/test_detection.py
```

### 3. Navigate Through Results

The script will:
1. Process all images in `single_face/` first
2. Then process images in `multiple_faces/`
3. Finally process images in `no_face/`

For each image:
- A window displays the image with green bounding boxes around detected faces
- Each face is labeled with its number and detection confidence
- Console shows detection details (filename, number of faces, coordinates, confidence)

**Controls:**
- **Press any key**: Move to the next image
- **Press ESC**: Exit the test early

## Example Output

```
======================================================================
FACE DETECTION TEST
======================================================================

This script tests the FaceDetector on images in test folders.
Press any key to navigate through images, ESC to exit.

[INFO] Initializing FaceDetector...
[OK] FaceDetector initialized

======================================================================
Testing: single_face/
======================================================================
[INFO] Found 3 image(s)

──────────────────────────────────────────────────────────────────────
Image: photo1.jpg
──────────────────────────────────────────────────────────────────────
[OK] Detected 1 face(s)
     Face 1: bbox=(120, 85, 180, 220), confidence=0.956

──────────────────────────────────────────────────────────────────────
Image: portrait.jpg
──────────────────────────────────────────────────────────────────────
[OK] Detected 1 face(s)
     Face 1: bbox=(200, 100, 250, 310), confidence=0.923

======================================================================
Testing: multiple_faces/
======================================================================
[INFO] Found 2 image(s)

──────────────────────────────────────────────────────────────────────
Image: group_photo.jpg
──────────────────────────────────────────────────────────────────────
[OK] Detected 4 face(s)
     Face 1: bbox=(50, 80, 120, 150), confidence=0.945
     Face 2: bbox=(180, 75, 130, 160), confidence=0.912
     Face 3: bbox=(320, 90, 125, 145), confidence=0.889
     Face 4: bbox=(450, 85, 115, 140), confidence=0.867

======================================================================
Testing: no_face/
======================================================================
[INFO] Found 1 image(s)

──────────────────────────────────────────────────────────────────────
Image: landscape.jpg
──────────────────────────────────────────────────────────────────────
[OK] Detected 0 face(s)

======================================================================
TESTING COMPLETE
======================================================================

[OK] Resources released
```

## What the Script Does

**For each image:**
1. Loads the image file
2. Runs face detection using MediaPipe FaceDetector
3. Draws green bounding boxes around detected faces
4. Labels each face with its number and confidence score
5. Displays the result in a window
6. Prints detection details to console

**Bounding box format:** `(x, y, width, height)` in pixels
- `x, y`: Top-left corner coordinates
- `width, height`: Box dimensions

**Confidence score:** 0.0 to 1.0 (higher = more confident)

## Error Handling

The script handles errors gracefully:

- **Invalid images**: Skips and reports
- **Corrupted files**: Skips and continues
- **Detection errors**: Reports and moves to next image
- **Missing folders**: Reports and continues
- **Empty folders**: Reports and continues

Errors for individual images do not stop the entire test.

## Configuration

The FaceDetector uses the detection confidence threshold from `config.py`:

```python
FACE_DETECTION_CONFIDENCE = 0.7
```

Faces with confidence below this threshold are filtered out automatically.

To change the threshold, modify `config.py` (affects the entire application) or adjust the FaceDetector initialization in the test script:

```python
detector = FaceDetector(min_detection_confidence=0.5)  # Lower threshold
```

## Tips for Testing

**Good test images:**
- Clear, well-lit photos
- Faces at various angles and distances
- Different lighting conditions
- Various backgrounds

**Testing multiple faces:**
- Group photos
- Classroom scenes
- Family photos
- Crowds

**Testing no faces:**
- Landscapes
- Objects
- Abstract images
- Blurred or very distant faces

## Notes

- This test is completely separate from the main application
- No data is written to the database
- Uses the same FaceDetector as the production system
- MediaPipe BlazeFace model is used for detection
- Detection speed depends on image resolution

## Troubleshooting

**"Failed to initialize FaceDetector"**
- Check that `blaze_face_short_range.tflite` exists in the project root
- Verify MediaPipe is installed correctly

**"No faces detected" on images with faces**
- Face may be too small or at extreme angle
- Try images with larger, front-facing faces
- Check lighting conditions
- Lower the detection confidence threshold

**"Could not load image"**
- Check file format is supported
- Verify image is not corrupted
- Try re-saving the image

**Image window doesn't appear**
- Check that OpenCV can create windows on your system
- Verify display is available (required for GUI)
- Try running in a graphical environment
